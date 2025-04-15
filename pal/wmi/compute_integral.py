from typing import Callable, TypeVar
from pysmt.shortcuts import Bool, Real
from pysmt.shortcuts import get_env
from pysmt.fnode import FNode
import torch
from functools import reduce

from pal.logic.lra_pysmt import translate_to_pysmt
from pal.polynomial.constrained_distribution import ConditionalConstraintedDistribution, ConstrainedDistributionBuilder, box_to_lra
from pal.polynomial.spline_distribution import ConditionalSplineSQ2D, SplineSQ2DBuilder
from pal.wmi.gasp.gasp.torch.wmipa.numerical_symb_integrator_pa import FunctionMode, NumericalSymbIntegratorPA
from wmipa.wmi import WMI as WMI_PA
from wmipa import WMI


def compute_integral(
    constraints: FNode,
    f: Callable[[torch.Tensor], torch.Tensor],
    output_shape: tuple[int, ...],
    total_degree: int,
    variable_map: dict[str, int],
    device: torch.device,
    precision: torch.dtype = torch.float64,
    gasp_kwargs: dict | None = None,
    wmi_pa_mode: str = "SAE4WMI",
) -> torch.Tensor:
    """
    Computes the integral of a function over the region defined by the constraints.

    Args:
        constraints (FNode): The constraints defining the region.
        f (Callable[[torch.Tensor], torch.Tensor]): The function to integrate.
        output_shape (tuple[int, ...]): The shape of the output tensor for function f.
        total_degree (int): The total degree of the polynomial.
        variable_map (dict[str, int]): A mapping from variable names to indices.
        device (torch.device): The device to use for computation.
        precision (torch.dtype): The precision of the computation.
            Defaults to torch.float64.
        gasp_kwargs (dict | None): Additional arguments for GASP integration.
            Defaults to None.
        wmi_pa_mode (str): The mode to use for WMI integration.
            Defaults to "SAE4WMI".

    Returns:
        torch.Tensor: The result of the integration.
    """
    # need this for Plus(*[])
    get_env().enable_infix_notation = True

    mode = FunctionMode(f, output_shape)

    if gasp_kwargs is None:
        gasp_kwargs = {}

    integrator = NumericalSymbIntegratorPA(
        mode=mode,
        total_degree=total_degree,
        variable_map=variable_map,
        **gasp_kwargs
    )
    integrator.set_device(device)
    integrator.set_dtype(precision)

    wmi = WMI(chi=constraints, weight=Real(1), integrator=integrator)

    phi = Bool(True)

    assert wmi_pa_mode in WMI_PA.MODES
    with torch.no_grad():
        result_pa, _ = wmi.computeWMI(phi, mode=wmi_pa_mode)
    result_pa: torch.Tensor = result_pa.to(device)

    return result_pa


A = TypeVar("A", bound=ConditionalConstraintedDistribution)


def integrate_distribution(
    d: ConstrainedDistributionBuilder[A],
    device: torch.device,
    precision: torch.dtype = torch.float64,
    gasp_kwargs: dict | None = None,
    wmi_pa_mode: str = "SAE4WMI",
) -> A:
    """
    Integrate the distribution over the constraints.
    Args:
        d (ConstrainedDistributionBuilder): The distribution to integrate.
        device (torch.device): The device to use for computation.
        precision (torch.dtype): The precision of the computation.
            Defaults to torch.float64.
        gasp_kwargs (dict | None): Additional arguments for GASP integration.
            Defaults to None.
        wmi_pa_mode (str): The mode to use for WMI integration.
            Defaults to "SAE4WMI".
    Returns:
        ConditionalSplineSQ2D: The integrated spline.
    """
    constraints_all_lra = d.constraints
    variable_map = d.var_positions
    total_degree = d.total_degree

    integrals = {}
    for box, f, shape in d.enumerate_pieces():
        constraints_boxed = constraints_all_lra.expression & box_to_lra(box)
        constraints_smt, _ = translate_to_pysmt(constraints_boxed)

        box_integral = compute_integral(
            constraints=constraints_smt,
            f=f,
            output_shape=shape,
            total_degree=total_degree,
            variable_map=variable_map,
            device=device,
            precision=precision,
            gasp_kwargs=gasp_kwargs,
            wmi_pa_mode=wmi_pa_mode,
        )
        integrals[box.id] = box_integral

    return d.get_distribution(integrated=integrals)
