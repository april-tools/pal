from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import math
from typing import Callable
import torch
import time
from gasp.torch.wmipa.drop_in_polynomial import (
    calc_single_polynomial,
    create_tensors_from_polynomial,
)
# import gasp.torch.wmipa.wmipa_monkeypatch  # noqa
from wmipa.integration.integrator import Integrator

import gasp.torch.numerics.grundmann_moeller as gm
from wmipa.integration.polytope import Polynomial
import gasp.torch.wmipa.triangulate_inequalities as triangulate
from pysmt.shortcuts import LE, LT

# # hack for the "RuntimeError: CUDA driver initialization failed." error
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# different configurations:


@dataclass
class WeightedFormulaMode:
    pass


@dataclass
class FunctionMode:
    f: Callable[[torch.Tensor], torch.Tensor]


IntegratorModes = WeightedFormulaMode | FunctionMode


class NumericalSymbIntegratorPA(Integrator):
    def __init__(
        self,
        mode: IntegratorModes,  # whether to integrate over the weights or the function
        total_degree: int,  # total degree of the polynomial, so max over sum of exponents for each monomial
        variable_map: dict[str, int],  # name => index
        sum_seperately=False,
        with_sorting=False,
        batch_size=None,
        monomials_lower_precision=True,
        n_workers=7,
    ):
        self.mode = mode
        self.total_degree = total_degree
        self.variable_map = variable_map
        self.dtype = torch.float64
        self.gm_points = {}
        if total_degree < 20:
            self.device = torch.device("cpu")
            _ = self.get_gm_points(total_degree)
        # self.prepare_grundmann_moeller()
        self.device = torch.device("cpu")
        self.sum_seperately = sum_seperately
        self.with_sorting = with_sorting
        self.batch_size = batch_size
        self.monomials_lower_precision = monomials_lower_precision
        self.sequential_integration_time = 0.0
        self.n_workers = n_workers
        super().__init__()

    def prepare_grundmann_moeller(self, degree):
        """
        Prepares the Grundmann-Moeller quadrature rule for integration over a simplex.
        """
        total_degree = degree
        s = math.ceil((float(total_degree) - 1) / 2)
        self.s = s
        assert 2 * s + 1 >= total_degree
        coefficients, points = gm.prepare_grundmann_moeller(s, len(self.variable_map))
        coefficients = torch.tensor(coefficients, dtype=self.dtype)
        points = torch.tensor(points, dtype=self.dtype)

        return (coefficients, points)

    def set_device(self, device):
        self.device = device
        # move the GM points to the device
        for key in self.gm_points:
            (c, p) = self.gm_points[key]
            self.gm_points[key] = (c.to(device), p.to(device))
        
        if isinstance(self.mode, FunctionMode):
            f = self.mode.f
            # if f is a torch Module, move it to the device
            if isinstance(f, torch.nn.Module):
                f.to(device)

    def set_dtype(self, dtype):
        self.dtype = dtype
        # move the GM points to the dtype
        for key in self.gm_points:
            (c, p) = self.gm_points[key]
            self.gm_points[key] = (c.to(dtype), p.to(dtype))

        if isinstance(self.mode, FunctionMode):
            f = self.mode.f
            # if f is a torch Module, change the dtype
            if isinstance(f, torch.nn.Module):
                f.to(dtype)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def integrate(self, atom_assignments, weight, aliases, *args, **kwargs):  # type: ignore
        res_list, _ = self.integrate_batch(
            [atom_assignments, weight, aliases], *args, **kwargs
        )
        return res_list[0]

    def _make_problem(self, weight, bounds, aliases):
        """Makes the problem to be solved by gm."""

        A, b = triangulate.pysmt_to_matrices(bounds, self.variable_map, aliases)
        v_rep = triangulate.h_rep_to_v_rep(A, b)
        if v_rep is None:
            n_vars = len(self.variable_map)
            zero_simplices = torch.zeros([0, n_vars + 1, n_vars])
            zero_coeffs = torch.zeros([0, 0], dtype=self.dtype)
            zero_exponents = torch.zeros(
                [0, 0, n_vars], dtype=torch.int64
            )
            return zero_simplices, zero_coeffs, zero_exponents
        else:
            if self.monomials_lower_precision:
                dtype = torch.float32
            else:
                dtype = self.dtype
            
            simplices = triangulate.triangulate_v_rep(v_rep)
            simplices = torch.tensor(simplices, dtype=self.dtype)

            match self.mode:
                case FunctionMode(_):
                    # we don't need the weights
                    return simplices, None, None
                case WeightedFormulaMode():
                    # we need the weights
                    integrand = Polynomial(weight, aliases)
                    coeff, exponents = create_tensors_from_polynomial(
                        integrand, self.variable_map, dtype
                    )
                    return simplices, coeff, exponents
        
    def get_gm_points(self, degree) -> tuple[torch.Tensor, torch.Tensor]:
        # self.gm_points is max_degree -> (coefficients, points)
        # find the smallest key that is larger than degree
        # keys = np.array(list(self.gm_points.keys()))
        # valid_keys = keys[keys >= degree]
        # key = valid_keys.min() if len(valid_keys) > 0 else None
        # if key is None or key > 2*degree:
        #     print(f"Gen Key: {key}, Degree: {degree}")
        #     # if there is no such key, prepare the GM points for the given degree
        #     (c, p) = self.prepare_grundmann_moeller(degree)
        #     self.gm_points[degree] = (c.to(self.device), p.to(self.device))
        #     key = degree
        # else:
        #     print(f"Found Key: {key}, Degree: {degree}")

        if degree not in self.gm_points:
            (c, p) = self.prepare_grundmann_moeller(degree)
            self.gm_points[degree] = (c.to(self.device), p.to(self.device))
        key = degree

        (coefficients, points) = self.gm_points[key]
        return (coefficients, points)

    def integrate_simplex(self, simplex, coeffs, exponents):
        match self.mode:
            case FunctionMode(_):
                total_deg = self.total_degree
            case WeightedFormulaMode():
                total_deg = exponents.sum(-1).max().item()
        gm_coefficients, gm_points = self.get_gm_points(total_deg)

        match self.mode:
            case FunctionMode(f_raw):
                if self.monomials_lower_precision:
                    f = lambda x: f_raw(x.to(torch.float32)).to(gm_coefficients.dtype)
                else:
                    f = f_raw
            case WeightedFormulaMode():
                if self.monomials_lower_precision:
                    f = lambda x: calc_single_polynomial(x.to(torch.float32), coeffs, exponents).to(gm_coefficients.dtype)
                else:
                    f = lambda x: calc_single_polynomial(x, coeffs, exponents)

        return gm.integrate(
            f,
            gm_coefficients,
            gm_points,
            simplex,
            sum_seperately=self.sum_seperately,
            with_sorting=self.with_sorting,
            batched=self.batch_size,
        )

    def _convert_to_problem(self, atom_assignments, weight, aliases):
        bounds = []
        for atom, value in atom_assignments.items():
            assert isinstance(value, bool), "Assignment value should be Boolean"

            # Skip atoms without variables
            if len(atom.get_free_variables()) == 0:
                continue

            if value is False:
                # If the negative literal is an inequality, change its
                # direction
                if atom.is_le():
                    left, right = atom.args()
                    atom = LT(right, left)
                elif atom.is_lt():
                    left, right = atom.args()
                    atom = LE(right, left)

            # Add a bound if the atom is an inequality
            if atom.is_le() or atom.is_lt():
                bounds.append(atom)

        return self._make_problem(weight, bounds, aliases)
    
    def integrate_batch(self, problems, *args, **kwargs):  # type: ignore
        start_time = time.time()
        results = []
        import tqdm

        def convert_to_problem_wrapper(problem):
            atom_assignments, weight, aliases, cond_assignments = problem
            return self._convert_to_problem(atom_assignments, weight, aliases)

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_problem = {executor.submit(convert_to_problem_wrapper, problem): problem for problem in problems}
            for future in tqdm.tqdm(as_completed(future_to_problem), total=len(problems), disable=False):
                try:
                    simplices, coeffs, exponents = future.result()
                    if simplices.shape[0] == 0:
                        # this won't work if results is empty but this will hopefully never happen :)
                        results.append(torch.zeros_like(results[-1]))
                        continue
                    simplices = simplices.to(self.device)
                    coeffs = coeffs.to(self.device)
                    exponents = exponents.to(self.device)
                    # batch over simplices
                    results_per_batch = []
                    s_size = int(self.batch_size / 10)
                    for i in range(0, simplices.shape[0], s_size):
                        simplices_batch = simplices[i:i+s_size]
                        integral_simplices = torch.vmap(
                            lambda s: self.integrate_simplex(s, coeffs, exponents)
                        )(simplices_batch)
                        integral_simplices_batch = (
                            torch.sum(integral_simplices, dim=0).to("cpu").unsqueeze(-1)
                        )
                        results_per_batch.append(integral_simplices_batch)

                    integral_polytope = torch.cat(results_per_batch, dim=-1).sum(dim=-1)
                    # integral_simplices = torch.vmap(
                    #     lambda s: self.integrate_simplex(s, coeffs, exponents)
                    # )(simplices)
                    # integral_polytope = (
                    #     torch.sum(integral_simplices, dim=0).to("cpu").unsqueeze(-1)
                    # )
                    results.append(integral_polytope.item())
                except Exception as exc:
                    # print(f'Problem {problem} generated an exception: {exc}')
                    raise exc
            
        self.sequential_integration_time = time.time() - start_time
        # results = torch.concatenate(results, dim=-1)
        return results, 0

    def integrate_batch_sequential(self, problems, *args, **kwargs):  # type: ignore
        start_time = time.time()
        results = []
        # print(f"N problems: {len(problems)}")
        import tqdm
        for index, (atom_assignments, weight, aliases, cond_assignments) in tqdm.tqdm(enumerate(
            problems
        ), total=len(problems)):
            simplices, coeffs, exponents = self._convert_to_problem(
                atom_assignments, weight, aliases
            )
            if simplices.shape[0] == 0:
                # this won't work if results is empty but this will hopefully never happen :)
                results.append(torch.zeros_like(results[-1]))
                continue
            simplices = simplices.to(self.device)
            coeffs = coeffs.to(self.device)
            exponents = exponents.to(self.device)
            integral_simplices = torch.vmap(
                lambda s: self.integrate_simplex(s, coeffs, exponents)
            )(simplices)
            integral_polytope = (
                torch.sum(integral_simplices, dim=0).to("cpu").unsqueeze(-1)
            )
            results.append(integral_polytope.item())
        
        self.sequential_integration_time = time.time() - start_time
        # results = torch.concatenate(results, dim=-1)
        return results, 0

    def to_short_str(self):
        return "torch"

    def to_json(self):
        return {
            "name": "torch",
            "total_degree": self.total_degree,
            "variable_map": self.variable_map,
            "monomials_lower_precision": self.monomials_lower_precision,
            "sum_seperately": self.sum_seperately,
            "with_sorting": self.with_sorting,
        }

    def get_parallel_integration_time(self):
        return self.sequential_integration_time

    def get_sequential_integration_time(self):
        return self.sequential_integration_time
