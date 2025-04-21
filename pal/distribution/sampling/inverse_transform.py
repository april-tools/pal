from typing import Callable
import torch
import pysmt as smt

from pal.distribution.spline_distribution import SplineSQ2D
from pal.distribution.torch_polynomial import SquaredTorchPolynomial
import pal.logic.lra as lra
from pal.logic.lra_pysmt import translate_to_pysmt
from pysmt.shortcuts import get_env

from pal.wmi.compute_integral import compute_integral


def bisection_search(
    f: Callable[[float], float],
    lower: float,
    upper: float,
    target: float,
    tol: float = 1e-2,
):
    count = 0
    while upper - lower > tol:
        mid = (upper + lower) / 2
        if f(mid) < target:
            lower = mid
        else:
            upper = mid
        count += 1

    print(f"bisection_search took {count} iterations")
    return (upper + lower) / 2


def conditioned_function(
    f: Callable[[torch.Tensor], torch.Tensor], constant: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Wraps a given function `f` to condition it on a constant tensor.

    This function creates a new function that appends the `constant` tensor
    to the input tensor `x` along the last dimension before passing it to `f`.

    It essentially computes `f(x') = f([constant, x'])`.

    Args:
        f (Callable[[torch.Tensor], torch.Tensor]):
            The function to be conditioned. It takes a tensor as input and
            returns a tensor as output.
        constant (torch.Tensor):
            A tensor that will be appended to the input tensor `x` before
            calling `f`. It is unsqueezed along the first dimension before
            concatenation.
    """

    def wrapped_function(x: torch.Tensor) -> torch.Tensor:
        x_full = torch.cat([x, constant.unsqueeze(0)], dim=-1)
        return f(x_full)

    return wrapped_function


def integrate_constrainted_cdf(
    constraints: lra.LRA,
    f: Callable[[torch.Tensor], torch.Tensor],
    var: str,
    upper_lim: float,
    integration_kwargs: dict,
) -> float:
    # smt.environment.reset_env()  # type: ignore

    limited_constraints = constraints & lra.LinearInequality(
        lhs={var: 1.0},
        rhs=upper_lim,
        symbol="<=",
    )

    # need this for Plus(*[])
    get_env().enable_infix_notation = True

    pysmt_constraints, symb_cache = translate_to_pysmt(limited_constraints)

    integral = compute_integral(
        constraints=pysmt_constraints,
        f=f,
        **integration_kwargs,
    )

    return integral.item()


def get_sub_total_degree(p: SquaredTorchPolynomial, start_idx):
    sub_powers = p.powers[:, start_idx:]
    return 2*sub_powers.sum(dim=1).max().item()


def inverse_transform_sampling_gasp(
    constraints: lra.LRAProblem,
    poly: SquaredTorchPolynomial,
    device: torch.device,
    precision: torch.dtype = torch.float64,
    gasp_kwargs: dict | None = None,
    wmi_pa_mode: str = "SAE4WMI",
):
    limits = constraints.get_global_limits()
    variables = sorted(
        list(poly.get_variable_map_dict().keys()),
        key=lambda x: poly.get_variable_map_dict()[x],
    )
    assert poly.get_variable_map_dict()[variables[0]] == 0
    sampled_dimensions: dict[str, float] = {}

    variable_map = poly.get_variable_map_dict()

    for i, var in enumerate(variables):

        def condition_linear_ineq(linear_ineq: lra.LinearInequality):
            linear_ineq_res = linear_ineq
            for var, const in sampled_dimensions.items():
                linear_ineq_res = linear_ineq_res.replace_variable_with_constant(
                    var, const
                )
            return linear_ineq_res

        the_constraints = constraints.map_constraints(
            f=condition_linear_ineq, drop_vars=list(sampled_dimensions.keys())
        )

        degree = get_sub_total_degree(poly, i)

        poly_wrapped = conditioned_function(
            f=poly,
            constant=torch.tensor(
                [sampled_dimensions[var] for var in variables[:i]]
            ).to(device),
        )

        lower, upper = limits[var]

        constrained_var_map_dict = {
            var: (i - idx) for var, idx in variable_map.items() if i >= idx
        }

        integration_kwargs = {
            "total_degree": degree,
            "output_shape": (1,),
            "variable_map": constrained_var_map_dict,
            "device": device,
            "precision": precision,
            "gasp_kwargs": gasp_kwargs,
            "wmi_pa_mode": wmi_pa_mode,
        }

        full_integral = integrate_constrainted_cdf(
            the_constraints, poly_wrapped, var, upper, integration_kwargs
        )

        def f(upper_limit: float):
            return (
                integrate_constrainted_cdf(
                    the_constraints, poly_wrapped, var, upper_limit, integration_kwargs
                )
                / full_integral
            )

        target = torch.rand(1).item()
        sampled_dimensions[var] = bisection_search(f, lower, upper, target)

    # construct the sampled point
    sampled_point = torch.tensor(
        [sampled_dimensions[var] for var in variables], device=device
    ).to(precision)
    return sampled_point


def sample_spline_distribution(
    dist: SplineSQ2D,
    device: torch.device,
    precision: torch.dtype = torch.float64,
    gasp_kwargs: dict | None = None,
    wmi_pa_mode: str = "SAE4WMI",
) -> torch.Tensor:
    """
    Sample a single point from the SplineSQ2D distribution.
    """
    assert not dist.is_batched()
    mixture_weights = dist.mixture_weights.unsqueeze(0)
    # sample the component index from the mixture weights
    component_index = torch.multinomial(mixture_weights, 1, replacement=True).item()

    pieces = dist.enumerate_pieces(selected_mixture=component_index)

    pieces_weights = [w.item() for (_, _, w) in pieces]
    pieces_sum = sum(pieces_weights)
    pieces_weights = [w / pieces_sum for w in pieces_weights]

    # sample the piece index from the pieces weights
    piece_index = torch.multinomial(
        torch.tensor(pieces_weights), 1, replacement=True
    ).item()
    (box, f, w) = pieces[piece_index]
    # sample the point from the piece

    problem = dist.constraints

    boxed_problem = problem & box

    assert isinstance(f, SquaredTorchPolynomial)

    f.to(device)
    f = f.to(precision)

    return inverse_transform_sampling_gasp(
        boxed_problem,
        f,
        device=device,
        precision=precision,
        gasp_kwargs=gasp_kwargs,
        wmi_pa_mode=wmi_pa_mode,
    )
