from typing import Any, Dict, Self
from frozendict import frozendict
import torch
from constrained_prob_ml.polynomial.polynomial import AbstractPolynomial
from constrained_prob_ml.symbolic.constraints import LinearIneqLogic as LT
from constrained_prob_ml.symbolic.linear import SymbolicLinear
import constrained_prob_ml.polynomial.torch_polynomial as tp
import torch.nn as nn
import numpy as np


class CubicPiecewisePolynomial1D(torch.nn.Module):
    # a + b * x + c * x^2 + d * x^3
    def __init__(
        self,
        knots: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
    ):
        super().__init__()
        self.knots = knots
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def get_coefficients(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        knot_idx = torch.searchsorted(self.knots, x)
        knot_idx = torch.clamp(knot_idx - 1, 0, len(self.knots) - 2)

        a = self.a[knot_idx]
        b = self.b[knot_idx]
        c = self.c[knot_idx]
        d = self.d[knot_idx]

        return a, b, c, d

    def enumerate_coefficients(self) -> torch.Tensor:
        """
        Enumerate the coefficients of the piecewise polynomial.

        Returns:
            torch.Tensor: A tensor of shape (num_pieces, 4) representing the coefficients of the piecewise polynomial.
        """
        coefficients = torch.stack([self.a, self.b, self.c, self.d], dim=1)
        return coefficients

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # find the knot that x is in
        a, b, c, d = self.get_coefficients(x)
        return a + b * x + c * x**2 + d * x**3


class CubicPiecewisePolynomial2DUnivariate(torch.nn.Module):
    # (a0 + b0 * x0 + c0 * x0^2 + d0 * x0^3) * (a1 + b1 * x1 + c1 * x1^2 + d1 * x1^3)
    def __init__(
        self,
        knots: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
        shift_polynomials: bool = True,  # shift plynoial or the input
    ):
        super().__init__()
        self.knots = knots
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.shift_polynomials = shift_polynomials

    def get_coefficients(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        def get_coeff_1d(knots, x, a, b, c, d):
            knot_idx = torch.searchsorted(knots, x)
            knot_idx = torch.clamp(knot_idx - 1, 0, len(knots) - 2)

            a = a[knot_idx]
            b = b[knot_idx]
            c = c[knot_idx]
            d = d[knot_idx]

            return a, b, c, d

        a, b, c, d = torch.vmap(get_coeff_1d, in_dims=-1, out_dims=-1)(
            self.knots, x, self.a, self.b, self.c, self.d
        )

        return a, b, c, d

    def get_starting_coords_bin(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the starting coordinates of the bin that x is in.
        """
        x0 = torch.searchsorted(self.knots[:, 0], x[:, 0])
        x0 = torch.clamp(x0 - 1, 0, len(self.knots) - 2)

        x1 = torch.searchsorted(self.knots[:, 1], x[:, 1])
        x1 = torch.clamp(x1 - 1, 0, len(self.knots) - 2)

        return torch.stack([self.knots[x0, 0], self.knots[x1, 1]], dim=1)

    def enumerate_coefficients(self) -> torch.Tensor:
        """
        Enumerate the coefficients of the piecewise polynomial.

        Returns:
            torch.Tensor: A tensor of shape (num_pieces, 4, 2) representing the coefficients
            of the piecewise polynomial.
        """
        coefficients = torch.stack([self.a, self.b, self.c, self.d], dim=1)
        return coefficients

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b, c, d = self.get_coefficients(x)
        # naive implementation makes it easier to understand and test
        if self.shift_polynomials:
            eval_point = x
        else:
            eval_point = x - self.get_starting_coords_bin(x)
        return (
            a[..., 0]
            + b[..., 0] * eval_point[..., 0]
            + c[..., 0] * eval_point[..., 0] ** 2
            + d[..., 0] * eval_point[..., 0] ** 3
        ) * (
            a[..., 1]
            + b[..., 1] * eval_point[..., 1]
            + c[..., 1] * eval_point[..., 1] ** 2
            + d[..., 1] * eval_point[..., 1] ** 3
        )


def construct_polynomials(
    knots: torch.Tensor,
    differences: torch.Tensor,
    y: torch.Tensor,
    dy: torch.Tensor,
    shift: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct a piecewise cubic polynomial from the given values and derivatives at the knots.
    """
    assert y.shape[0] == knots.shape[0] and dy.shape[0] == knots.shape[0]
    # compute the coefficients of the polynomials
    # a + b * x + c * x^2 + d * x^3
    dy0 = dy[:-1] * differences
    dy1 = dy[1:] * differences
    y0 = y[:-1]
    y1 = y[1:]

    # polynomial on the interval [0, 1]
    a = y0
    b = dy0
    c = 3 * (y1 - y0) - 2 * dy0 - dy1
    d = 2 * (y0 - y1) + dy0 + dy1

    if shift:
        # transform the coefficients to the interval [knots[i], knots[i + 1]]
        x0 = knots[:-1]
        transformed_a = (
            a
            - b * (1 / differences) * x0
            + c * (1 / differences**2) * x0**2
            - d * (1 / differences**3) * x0**3
        )
        transformed_b = (
            b * (1 / differences)
            - 2 * c * (1 / differences**2) * x0
            + 3 * d * (1 / differences**3) * x0**2
        )
        transformed_c = c * (1 / differences**2) - 3 * d * (1 / differences**3) * x0
        transformed_d = d * (1 / differences**3)

        return transformed_a, transformed_b, transformed_c, transformed_d
    else:
        # transform the coefficients to the interval [0, (knots[i + 1] - knots[i])]
        transformed_a = a
        transformed_b = b * 1 / differences
        transformed_c = c * 1 / differences**2
        transformed_d = d * 1 / differences**3

        return transformed_a, transformed_b, transformed_c, transformed_d


class HermiteSpline1D(torch.nn.Module):
    knots: torch.Tensor

    def __init__(self, knots: torch.Tensor, varname: str):
        super().__init__()
        self.knots = knots

        # assert that the knots are sorted
        assert torch.all(knots[:-1] <= knots[1:])

        self.differences = self.knots[1:] - self.knots[:-1]
        self.varname = varname

    def construct_polynomials(
        self, y: torch.Tensor, dy: torch.Tensor
    ) -> CubicPiecewisePolynomial1D:
        """
        Construct a piecewise cubic polynomial from the given values and derivatives at the knots.

        Example:
            >>> knots = torch.tensor([0.0, 1.0, 2.0])
            >>> y = torch.tensor([0.0, 1.0, 0.0])
            >>> dy = torch.tensor([1.0, -1.0, 1.0])
            >>> spline = HermiteSpline1D(knots, "y")
            >>> polynomial = spline.construct_polynomials(y, dy)
            >>> assert torch.allclose(polynomial(knots), y)
        """
        transformed_a, transformed_b, transformed_c, transformed_d = (
            construct_polynomials(self.knots, self.differences, y, dy)
        )

        return CubicPiecewisePolynomial1D(
            self.knots, transformed_a, transformed_b, transformed_c, transformed_d
        )

    def enumerate_boxes(self) -> list[LT]:
        """
        Enumerate the boxes that the spline is defined on.

        Returns:
            list[LT]: A list of linear inequalities representing the boxes that the spline is defined on.
        """
        boxes = []
        for i in range(len(self.knots) - 1):
            lower = self.knots[i].item()
            upper = self.knots[i + 1].item()
            box = LT.limit(self.varname, lower, upper)
            boxes.append(box)
        return boxes


class HermiteSpline2DUnivariate(torch.nn.Module):
    """
    A 2d Spline that is defined as a product of two 1d splines, so dimension wise.
    """
    knots: torch.Tensor   # (2, num_knots) because of memory layout
    differences: torch.Tensor

    def __init__(self, knots: torch.Tensor   # (num_knots, 2)
                 , var_map: dict[str, int]):
        super().__init__()
        self.register_buffer("knots", knots.permute(1, 0).contiguous())

        # assert that the knots are sorted
        assert torch.all(knots[:-1] <= knots[1:])

        assert len(var_map) == 2

        self.differences = knots[1:] - knots[:-1]
        self.var_map = var_map

    def construct_polynomials(
        self, y: torch.Tensor, dy: torch.Tensor,
        shift: bool = True  # True shifts the polynomial, False the input
    ) -> CubicPiecewisePolynomial2DUnivariate:
        """
        Construct a piecewise cubic polynomial from the given values and derivatives at the knots.

        Example:
            >>> knots = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
            >>> y = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
            >>> dy = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [1.0, 1.0]])
            >>> spline = HermiteSpline2DUnivariate(knots, {"x": 0, "y": 1})
            >>> polynomial = spline.construct_polynomials(y, dy)
            >>> assert torch.allclose(polynomial(knots), y)

        Args:
            y (torch.Tensor): The values of the function at the knots.
            dy (torch.Tensor): The derivatives of the function at the knots.
            shift (bool): Whether to shift the polynomial (True) or the point the
                polynomial is evaluated at (False).
        """
        do_construct = lambda knots, differences, y, dy: construct_polynomials(
            knots, differences, y, dy, shift=shift
        )
        transformed_a, transformed_b, transformed_c, transformed_d = torch.vmap(
            do_construct, in_dims=-1, out_dims=-1
        )(self.knots.permute(1, 0), self.differences, y, dy)

        return CubicPiecewisePolynomial2DUnivariate(
            self.knots.permute(1, 0), transformed_a, transformed_b, transformed_c, transformed_d,
            shift_polynomials=shift
        )

    def enumerate_boxes(self) -> dict[tuple[int, int], LT]:
        """
        Enumerate the boxes that the spline is defined on, maps indices to boxes.
        """
        var_map_inv = {v: k for k, v in self.var_map.items()}

        knots = self.knots.permute(1, 0)

        boxes = {}
        for i in range(knots.shape[0] - 1):
            lower_x0 = knots[i, 0].item()
            upper_x0 = knots[i + 1, 0].item()
            varname0 = var_map_inv[0]
            box0 = LT.limit(varname0, lower_x0, upper_x0)
            for j in range(knots.shape[0] - 1):
                lower_x1 = knots[j, 1].item()
                upper_x1 = knots[j + 1, 1].item()
                varname1 = var_map_inv[1]
                box1 = LT.limit(varname1, lower_x1, upper_x1)
                boxes[(i, j)] = box0 & box1

        return boxes

    def get_lower_left_knots(self, i: int, j: int) -> torch.Tensor:
        """
        Get the lower left knot of the (i, j)-bin.
        """
        knots = self.knots.permute(1, 0)
        value1 = knots[i, 0]
        value2 = knots[j, 1]
        return torch.stack([value1, value2])


# evaluation of mixture of splines


class GridSquared2DParamsWithCoefficientsTorchPolynomial(AbstractPolynomial, torch.nn.Module):  # type: ignore
    """
    A class representing a squared polynomial, for which the variables are integrated out,
    so it is only a function of the parameters, but on a grid.
    """

    coeffs: torch.Tensor
    indices_params: torch.Tensor

    def __init__(
        self,
        coeffs: torch.Tensor,
        indices_params: torch.Tensor,
        param_map_dict: Dict[str, int] | frozendict[str, int],
        variable_map_dict: Dict[str, int] | frozendict[str, int],
    ) -> None:
        super().__init__()
        assert len(coeffs.shape) == 3
        self.register_buffer("coeffs", coeffs)
        self.register_buffer("indices_params", indices_params)
        self.param_map_dict = param_map_dict
        self.variable_map_dict = variable_map_dict

    def eval_tensor(
        self,
        y_tensor: torch.Tensor | None,
        param_tensor: torch.Tensor,
        sq_epsilon: float = -1,
    ) -> torch.Tensor:
        assert y_tensor is None

        def eval_param_single_instance(
            param: torch.Tensor, coeffs: torch.Tensor
        ) -> torch.Tensor:
            combs = coeffs * torch.combinations(param, 2, with_replacement=True).prod(
                dim=-1
            )
            return combs.sum(dim=-1)

        def eval_param_instance_on_grid(param_gridded: torch.Tensor) -> torch.Tensor:
            grid_1d_func = torch.vmap(eval_param_single_instance)
            grid_2d_func = torch.vmap(grid_1d_func)

            result = grid_2d_func(param_gridded, self.coeffs)
            return result.sum(dim=-1).sum(dim=-1)

        params_combinations = torch.vmap(eval_param_instance_on_grid)(param_tensor)
        if sq_epsilon != -1:
            with torch.no_grad():
                params_combinations[params_combinations < sq_epsilon] = sq_epsilon
        return params_combinations

    def prepare_tensor(self) -> None:
        pass

    def integrate(self, var, lower: SymbolicLinear, upper: SymbolicLinear) -> Self:
        raise NotImplementedError

    def square(self) -> Self:
        raise NotImplementedError

    def to_state(self) -> Any:
        state = self.state_dict()
        state["param_map_dict"] = self.param_map_dict
        state["variable_map_dict"] = self.variable_map_dict
        return state

    @classmethod
    def from_state(cls, state_dict: Any) -> Self:
        return cls(
            coeffs=state_dict["coeffs"],
            indices_params=state_dict["indices_params"],
            param_map_dict=state_dict["param_map_dict"],
            variable_map_dict=state_dict["variable_map_dict"],
        )

    def numerically_stable_comp(
        self,
        y_tensor: torch.Tensor | None,
        param_tensor: torch.Tensor,
        sq_epsilon: float = -1,
    ) -> torch.Tensor:
        assert y_tensor is None
        raise NotImplementedError("Not implemented yet")
        np_param_tensor: np.ndarray = param_tensor.detach().cpu().numpy()
        np_coeffs: np.ndarray = self.coeffs.detach().cpu().numpy()

        np_param_tensor = np_param_tensor.astype(np.float128)
        np_coeffs = np_coeffs.astype(np.float128)

        results = []
        for i in range(np_param_tensor.shape[0]):
            params = np_param_tensor[i]
            params_idxs = torch.arange(0, len(params))
            comdinations_idxs = torch.combinations(
                params_idxs, 2, with_replacement=True
            ).numpy()
            combinations = [
                params[comdinations_idxs[:, i]]
                for i in range(comdinations_idxs.shape[1])
            ]
            combinations_prod = np.stack(combinations, axis=-1).prod(axis=-1)
            results.append(tp.stable_np_sum(np_coeffs * combinations_prod, axis=-1))
        results_tensor = torch.tensor(
            results, device=param_tensor.device, dtype=param_tensor.dtype
        )
        if sq_epsilon != -1:
            with torch.no_grad():
                results_tensor[results_tensor < sq_epsilon] = sq_epsilon
        return results_tensor

    def get_param_map_dict(self) -> Dict[str, int]:
        return self.param_map_dict  # type: ignore

    def get_variable_map_dict(self) -> Dict[str, int]:
        return self.variable_map_dict  # type: ignore

    def __hash__(self) -> int:
        return (
            tp.hash_torch_tensor(self.coeffs)
            + hash(self.param_map_dict)
            + hash(self.variable_map_dict)
        )

    def subs(self, subsitutions: Dict[str, Any]) -> Self:
        raise NotImplementedError

    def __add__(self, other: Self) -> Self:
        raise NotImplementedError

    def __mul__(self, other: Self | float) -> Self:
        raise NotImplementedError

    def to(self, device: torch.device) -> None:  # type: ignore
        torch.nn.Module.to(self, device)


def compute_reordering_of_parameter_positions2d(
    deg: int, param_map_dict: Dict[str, int]
):
    """
    Compute the reordering of the parameter positions for the polynomial so that
    it conforms to the combination.
    """
    exponents_y0 = torch.arange(deg + 1)
    exponents_y1 = torch.arange(deg + 1)
    outer_product_exponents = torch.cartesian_prod(exponents_y0, exponents_y1)
    param_index_map: Dict[int, int] = {}
    for resulting_param_index in range(outer_product_exponents.shape[0]):
        (exponent_y0, exponent_y1) = outer_product_exponents[resulting_param_index]
        param_name = (
            "param" + "".join(["0"] * exponent_y0) + "".join(["1"] * exponent_y1)
        )
        current_param_index = param_map_dict[param_name]
        param_index_map[current_param_index] = resulting_param_index
    return param_index_map


def eval_spline_on_grid(
    ys: torch.Tensor,
    values_dens: torch.Tensor,
    value_dens_derivatives: torch.Tensor,
    hs2d: HermiteSpline2DUnivariate,
    pa_integrated_polynomials: GridSquared2DParamsWithCoefficientsTorchPolynomial,
    poly_unsquared: AbstractPolynomial,
    return_unsquared=False,
):
    def compute_per_item(
        y: torch.Tensor, the_pval: torch.Tensor, the_p_der: torch.Tensor
    ):
        y = y.unsqueeze(0)
        learned_poly_shifted = hs2d.construct_polynomials(
            y=the_pval, dy=the_p_der, shift=False
        )

        bin_coords_start = learned_poly_shifted.get_starting_coords_bin(y)
        ys_pred_coeffs = learned_poly_shifted.get_coefficients(y)
        ys_pred_coeffs = torch.stack(ys_pred_coeffs, dim=-2).squeeze(0)
        ys_pred_coeffs_0 = ys_pred_coeffs[..., 0]
        ys_pred_coeffs_1 = ys_pred_coeffs[..., 1]

        ys_pred_coeffs_poly = torch.cartesian_prod(
            ys_pred_coeffs_0, ys_pred_coeffs_1
        ).prod(-1)

        all_coeffs_poly = learned_poly_shifted.enumerate_coefficients()  # (num_pieces, 4, 2)
        all_coeffs_poly_0 = all_coeffs_poly[..., 0]
        all_coeffs_poly_1 = all_coeffs_poly[..., 1]

        # compute all combinations of the coefficients via outer product
        # resulting in a tensor of shape (num_pieces, num_pieces, 4, 2)
        # via meshgrid
        mesh_coeffs_0, mesh_coeffs_1 = torch.vmap(lambda a, b: torch.meshgrid(a, b), in_dims=1, out_dims=2)(
            all_coeffs_poly_0, all_coeffs_poly_1
        )

        meshed_coeffs = torch.stack([mesh_coeffs_0, mesh_coeffs_1], dim=-1)

        def compute_integral_helper(coeff):
            coeff_0 = coeff[..., 0]
            coeff_1 = coeff[..., 1]
            coeffs_integral_poly = torch.cartesian_prod(coeff_0, coeff_1).prod(-1)
            return coeffs_integral_poly

        coeffs_integral = torch.vmap(torch.vmap(compute_integral_helper))(
            meshed_coeffs
        )

        integral = pa_integrated_polynomials.eval_tensor(
            y_tensor=None, param_tensor=coeffs_integral.unsqueeze(0)
        )

        return bin_coords_start, ys_pred_coeffs_poly, integral

    vectorized_compute_per_item = torch.vmap(compute_per_item)

    bin_coords_start, ys_pred_coeffs_poly, integrals = vectorized_compute_per_item(
        ys, values_dens, value_dens_derivatives
    )

    save_abs = poly_unsquared.absolute 
    poly_unsquared.absolute = not return_unsquared
    ys_pred_poly = poly_unsquared.eval_tensor(
        y_tensor=ys - bin_coords_start.squeeze(1),
        param_tensor=ys_pred_coeffs_poly.squeeze(1),
    )
    poly_unsquared.absolute = save_abs

    if not return_unsquared:
        return ys_pred_poly**2, integrals
    else:
        return ys_pred_poly, integrals


def eval_mixture_model(
    ys: torch.Tensor,
    mixture_weights: torch.Tensor,
    values_dens: torch.Tensor,
    value_dens_derivatives: torch.Tensor,
    hs2d: HermiteSpline2DUnivariate,
    pa_integrated_polynomials: GridSquared2DParamsWithCoefficientsTorchPolynomial,
    poly_unsquared: AbstractPolynomial,
    return_per_mixture: bool = False,
    return_details: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:

    compute_per_mixture = (
        lambda single_value_density, single_value_density_derivative: eval_spline_on_grid(
            ys=ys,
            values_dens=single_value_density,
            value_dens_derivatives=single_value_density_derivative,
            hs2d=hs2d,
            pa_integrated_polynomials=pa_integrated_polynomials,
            poly_unsquared=poly_unsquared,
            return_unsquared=return_details,
        )
    )

    y_preds_point_eval_all_com, integrals_all_comp = torch.vmap(
        compute_per_mixture, in_dims=1, out_dims=1
    )(values_dens, value_dens_derivatives)

    # y_preds_squared_all_com = torch.stack(y_preds_squared_all_com, dim=0)
    # integrals_all_comp = torch.stack(integrals_all_comp, dim=0)

    if not return_details:
        # y_preds_point_eval_all_com is squared
        ys_pred_poly = (mixture_weights * y_preds_point_eval_all_com)
        if not return_per_mixture:
            ys_pred_poly = ys_pred_poly.sum(-1)
        integrals = (mixture_weights * integrals_all_comp.squeeze(-1))
        if not return_per_mixture:
            integrals = integrals.sum(-1)
        return ys_pred_poly, integrals
    else:
        # y_preds_point_eval_all_com is not squared
        # components separately
        return y_preds_point_eval_all_com, integrals_all_comp
