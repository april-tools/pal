old_path = "/disk/scratch/lkurscheidt/constrained-prob-ml"
# add to sys.path
import sys
import os
sys.path.append(old_path)
import torch

from constrained_prob_ml.symbolic.constraints import LinearConstraint, LinearIneqLogic as LT
import constrained_prob_ml.polynomial.hermite_spline as hs
from pysmt.shortcuts import Bool
from constrained_prob_ml.symbolic.constraints import to_pytorch_logic_module
import constrained_prob_ml.polynomial.torch_polynomial as tp
from constrained_prob_ml.pywmi.wmipa.pa_engine import create_wmipa

def recreate_mc_example():
    bound_x = LT.global_limit("x", 0, 1)
    bound_y = LT.global_limit("y", 0, 2)
    
    box_constraints = bound_x & bound_y

    problem_constraints = LT(None) & box_constraints

    # constraint1 = LI({"x": 1.0, "y": 1.0}, "<=", 2.0)  # x + y <= 2
    # constraint2 = LI({"x": 1.0}, ">=", 0.0)  # x >= 0
    # constraint3 = LI({"y": 1.0}, ">=", 0.0)  # y >= 0
    # constraint4 = LI({"x": 1.0}, "<=", 1.0)  # x <= 1

    c1 = LinearConstraint(
        lhs_coeffs=[1.0, 1.0],
        lhs=["x", "y"],
        symbol="<=",
        rhs=2.0,
    )
    c2 = LinearConstraint(
        lhs_coeffs=[1.0],
        lhs=["x"],
        symbol=">=",
        rhs=0.0,
    )
    c3 = LinearConstraint(
        lhs_coeffs=[1.0],
        lhs=["y"],
        symbol=">=",
        rhs=0.0,
    )
    c4 = LinearConstraint(
        lhs_coeffs=[1.0],
        lhs=["x"],
        symbol="<=",
        rhs=1.0,
    )

    constraints = problem_constraints & c1 & c2 & c3 & c4

    var_dict = {
        "x": 0,
        "y": 1,
    }

    y_pos_dict = {i: name for name, i in var_dict.items()}

    resolution = 500

    device = torch.device("cpu")

    limits = constraints.get_global_limits()
    linspaces = [
        torch.linspace(
            limits[y_pos_dict[i]][0],
            limits[y_pos_dict[i]][1],
            resolution,
            device=device,
        )
        for i in range(len(var_dict))
    ]

    # img_extent = [
    #     limits[y_pos_dict[0]][0],
    #     limits[y_pos_dict[0]][1],
    #     limits[y_pos_dict[1]][0],
    #     limits[y_pos_dict[1]][1],
    # ]

    # mesh = torch.meshgrid(*linspaces)
    # mesh = torch.stack(mesh, dim=-1).reshape(-1, len(var_dict))
    # mesh = mesh.to(device)

    # # assert constraints_generic.expression is not None
    # valid = pytorch_constraints(mesh).reshape(resolution, resolution).cpu()

    num_knots = 3

    knots = [
        torch.linspace(
            limits[y_pos_dict[i]][0],
            limits[y_pos_dict[i]][1],
            num_knots,
            device=device,
        )
        for i in range(len(var_dict))
    ]
    knots = torch.stack(knots, dim=-1)

    hs2d_train = hs.HermiteSpline2DUnivariate(
        knots=knots,
        var_map=var_dict,
    )

    pytorch_constraints = to_pytorch_logic_module(constraints.expression, var_dict)  # type: ignore
    pytorch_constraints = pytorch_constraints.to(device)

    max_order = 3

    # create polynomial
    poly_unsquared_unordered = tp.TorchPolynomial.construct(
        max_order=len(var_dict) * max_order,
        max_terms=max_order,
        var_map_dict=var_dict,
    )

    reordering = hs.compute_reordering_of_parameter_positions2d(max_order, poly_unsquared_unordered.param_map_dict)

    poly_unsquared = poly_unsquared_unordered.reorder_parameter_positions(reordering)

    poly_squared = poly_unsquared.square()

    pa_integrated_polynomials = {}
    constraint_boxes = {}

    from tqdm import tqdm

    for idx, box in tqdm(hs2d_train.enumerate_boxes().items()):
        constraint_boxes[idx] = box
        constraints_and_box = constraints & box
        kwargs = {}
        kwargs["monomials_lower_precision"] = False
        kwargs["directly_add_accumulator"] = False
        box_start = hs2d_train.get_lower_left_knots(*idx)
        poly_squared.set_shift((-1) * box_start)
        vectorized_polynomial, integrator, wmi = create_wmipa(
            constraints_and_box,
            poly_squared,
            add_to_s=0,
            sum_seperately=True,
            with_sorting=True,
            **kwargs,
        )
        integrator.set_dtype(torch.float64)
        integrator.set_device(device=device)
        integrator.set_batch_size(256)

        phi = Bool(True)

        with torch.no_grad():
            result_pa, _ = wmi.computeWMI(phi, mode="SAE4WMI")
        result_pa = result_pa.to(device)

        pa_integrated_poly = vectorized_polynomial.to_integrated_polynomial(result_pa)
        pa_integrated_polynomials[idx] = pa_integrated_poly

    coeffs_2dgrid = torch.zeros((knots.shape[0]-1, knots.shape[0]-1, pa_integrated_poly.coeffs.shape[0]), device=device)

    for idx, integrated in pa_integrated_polynomials.items():
        coeffs_2dgrid[idx[0], idx[1]] = integrated.coeffs

    assert not (coeffs_2dgrid == 0.0).all()

    integrated_poly_grid = hs.GridSquared2DParamsWithCoefficientsTorchPolynomial(
        coeffs=coeffs_2dgrid,
        indices_params=pa_integrated_poly.indices_params,
        variable_map_dict=var_dict,
        param_map_dict=pa_integrated_poly.param_map_dict,
    )

    return integrated_poly_grid