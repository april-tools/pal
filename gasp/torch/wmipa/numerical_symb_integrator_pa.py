from concurrent.futures import ThreadPoolExecutor, as_completed
import math
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

# hack for the "RuntimeError: CUDA driver initialization failed." error
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class NumericalSymbIntegratorPA(Integrator):
    def __init__(
        self,
        total_degree: int,  # total degree of the polynomial, so max over sum of exponents for each monomial
        variable_map: dict[str, int],  # name => index
        add_to_s=0,
        sum_seperately=False,
        with_sorting=False,
        batch_size=None,
        monomials_lower_precision=True,
        n_workers=7,
    ):
        self.total_degree = total_degree
        self.variable_map = variable_map
        self.add_to_s = add_to_s
        self.dtype = torch.float64
        self.prepare_grundmann_moeller()
        self.device = torch.device("cpu")
        self.sum_seperately = sum_seperately
        self.with_sorting = with_sorting
        self.batch_size = batch_size
        self.monomials_lower_precision = monomials_lower_precision
        self.sequential_integration_time = 0.0
        self.n_workers = n_workers
        super().__init__()

    def prepare_grundmann_moeller(self):
        """
        Prepares the Grundmann-Moeller quadrature rule for integration over a simplex.
        """
        total_degree = self.total_degree
        s = math.ceil((float(total_degree) - 1) / 2)
        s += self.add_to_s
        self.s = s
        assert 2 * s + 1 >= total_degree
        coefficients, points = gm.prepare_grundmann_moeller(s, len(self.variable_map))
        self.coefficients = torch.tensor(coefficients, dtype=self.dtype)
        self.points = torch.tensor(points, dtype=self.dtype)

    def set_device(self, device):
        self.device = device
        self.coefficients = self.coefficients.to(device)
        self.points = self.points.to(device)

    def set_dtype(self, dtype):
        self.dtype = dtype
        self.coefficients = self.coefficients.to(dtype)
        self.points = self.points.to(dtype)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def integrate(self, atom_assignments, weight, aliases, *args, **kwargs):  # type: ignore
        res_list, _ = self.integrate_batch(
            [atom_assignments, weight, aliases], *args, **kwargs
        )
        return res_list[0]

    def _make_problem(self, weight, bounds, aliases):
        """Makes the problem to be solved by gm."""
        integrand = Polynomial(weight, aliases)

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
            coeff, exponents = create_tensors_from_polynomial(
                integrand, self.variable_map, dtype
            )
            simplices = triangulate.triangulate_v_rep(v_rep)
            simplices = torch.tensor(simplices, dtype=self.dtype)

            return simplices, coeff, exponents

    def integrate_simplex(self, simplex, coeffs, exponents):
        if self.monomials_lower_precision:
            f = lambda x: calc_single_polynomial(x.to(torch.float32), coeffs, exponents).to(self.coefficients.dtype)
        else:
            f = lambda x: calc_single_polynomial(x, coeffs, exponents)
        return gm.integrate(
            f,
            self.coefficients,
            self.points,
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
            for future in tqdm.tqdm(as_completed(future_to_problem), total=len(problems), disable=True):
                problem = future_to_problem[future]
                try:
                    simplices, coeffs, exponents = future.result()
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
                except Exception as exc:
                    print(f'Problem {problem} generated an exception: {exc}')
            
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
