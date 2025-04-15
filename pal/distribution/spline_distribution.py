from typing import Any, Callable, Dict, Self
from frozendict import frozendict
import torch
import torch.nn as nn
import numpy as np

from pal.logic.lra_torch import PLRA, lra_to_torch
from pal.distribution.constrained_distribution import (
    Box,
    ConstrainedDistribution,
    ConstrainedDistributionBuilder,
    ConditionalConstraintedDistribution,
)
import pal.logic.lra as lra
from pal.distribution.torch_polynomial import (
    CubicPiecewisePolynomial2DUnivariate,
    SquaredParamsWithCoefficientsTorchPolynomial,
    TorchPolynomial,
    calculate_coefficients_from_hermite_spline,
)


def compute_reordering_of_parameter_positions2d(
    deg: int, powers: torch.Tensor  # shape [n_mon, 2]
) -> Dict[int, int]:
    """
    Compute the reordering of the parameter positions for the polynomial so that
    it conforms to the combination.
    """
    exponents_y0 = torch.arange(deg + 1)
    exponents_y1 = torch.arange(deg + 1)
    outer_product_exponents = torch.cartesian_prod(exponents_y0, exponents_y1)

    param_start_positions: Dict[tuple[int, int], int] = {}
    assert powers.shape[1] == 2
    assert powers.dtype == torch.int64
    for i in range(powers.shape[0]):
        exponent_y0 = powers[i, 0].item()
        exponent_y1 = powers[i, 1].item()
        param_name = (exponent_y0, exponent_y1)
        param_start_positions[param_name] = i

    param_index_map: Dict[int, int] = {}
    for resulting_param_index in range(outer_product_exponents.shape[0]):
        (exponent_y0, exponent_y1) = outer_product_exponents[resulting_param_index]
        current_param_index = param_start_positions[
            (exponent_y0.item(), exponent_y1.item())
        ]
        param_index_map[current_param_index] = resulting_param_index
    return param_index_map


class SplineSQ2DBuilder(
    ConstrainedDistributionBuilder["ConditionalSplineSQ2D"], torch.nn.Module
):
    """
    A class that represents a constrained distribution that has not been integrated yet.

    This class is a builder for a 2D squared, hermite spline distribution with mixtures.
    The splines are axis-aligned and the knots are equally spaced.
    """

    knots: (
        torch.Tensor
    )  # (2, num_knots) because of memory layout, these are the positions

    def __init__(
        self,
        constraints: lra.LRAProblem,
        var_positions: Dict[str, int],
        num_knots: int,
        num_mixtures: int,
    ):
        """
        Initializes the builder with the constraints and the number of knots and mixtures.
        """
        # Call the constructor of ConstrainedDistributionBuilder
        ConstrainedDistributionBuilder.__init__(self, var_positions, constraints)
        assert len(var_positions) == 2

        # Call the constructor of torch.nn.Module
        torch.nn.Module.__init__(self)

        self.num_knots = num_knots
        self.num_mixtures = num_mixtures

        y_pos_dict = {i: name for name, i in var_positions.items()}
        self.y_pos_dict = y_pos_dict

        limits = constraints.get_global_limits()
        knots = [
            torch.linspace(
                limits[y_pos_dict[i]][0],
                limits[y_pos_dict[i]][1],
                num_knots,
            )
            for i in range(len(var_positions))
        ]
        knots = torch.stack(knots, dim=-1)
        self.register_buffer("knots", knots)

        max_order = 3

        # create polynomial
        poly_unsquared_unordered = TorchPolynomial.construct(
            max_order=len(var_positions) * max_order,
            max_terms=max_order,
            var_map_dict=var_positions,
        )

        reordering = compute_reordering_of_parameter_positions2d(
            max_order, poly_unsquared_unordered.powers
        )

        self.poly_unsquared = poly_unsquared_unordered.reorder_parameter_positions(
            reordering
        )

        self.squared_poly = self.poly_unsquared.square()

    @property
    def total_degree(self) -> int:
        return self.squared_poly.get_max_total_degree()

    def enumerate_pieces(
        self,
    ) -> list[tuple[Box, Callable[[torch.Tensor], torch.Tensor], tuple[int, ...]]]:
        results = []

        shape: tuple[int] = (self.squared_poly.combinations_coefficient.shape[0],)

        for i in range(self.knots.shape[1] - 1):
            for j in range(self.knots.shape[1] - 1):
                lower_x0 = self.knots[i, 0]
                upper_x0 = self.knots[i + 1, 0]
                lower_x1 = self.knots[j, 1]
                upper_x1 = self.knots[j + 1, 1]

                varname0 = self.y_pos_dict[0]
                varname1 = self.y_pos_dict[1]

                lower_left = torch.stack([lower_x0, lower_x1], dim=0)

                box = Box(
                    id=(i, j),
                    constraints={
                        varname0: (lower_x0.item(), upper_x0.item()),
                        varname1: (lower_x1.item(), upper_x1.item()),
                    },
                )

                def eval_with_shift(y: torch.Tensor) -> torch.Tensor:
                    y_shifted = y - lower_left
                    return self.squared_poly.eval_tensor_vectorized(y_shifted)

                results.append((box, eval_with_shift, shape))
        return results

    def get_distribution(self, integrated) -> "ConditionalSplineSQ2D":
        fst = integrated[(0, 0)]
        coeffs_2dgrid = torch.zeros(
            (self.knots.shape[0] - 1, self.knots.shape[0] - 1, fst.shape[0]),
            device=fst.device,
        )
        for idx, result in integrated.items():
            i, j = idx
            coeffs_2dgrid[i, j] = result

        assert not (coeffs_2dgrid == 0.0).all()

        return ConditionalSplineSQ2D(
            constraints=self.constraints,
            var_positions=self.var_positions,
            num_knots=self.num_knots,
            num_mixtures=self.num_mixtures,
            poly_unsquared=self.poly_unsquared,
            integral_coeffs=coeffs_2dgrid,
            knots=self.knots,
        )


class ConditionalSplineSQ2D(
    ConditionalConstraintedDistribution[ConstrainedDistribution], torch.nn.Module
):
    """
    A class that represents a constrained distribution P(Y|psi) on some unknown parameters psi.
    """

    integral_coeffs: torch.Tensor
    knots: (
        torch.Tensor
    )  # (num_knots, 2) because of memory layout, these are the positions
    differences: torch.Tensor  # (num_knots-1, 2) because of memory layout

    def __init__(
        self,
        constraints: lra.LRAProblem,
        var_positions: Dict[str, int],
        num_knots: int,
        num_mixtures: int,
        poly_unsquared: TorchPolynomial,
        integral_coeffs: torch.Tensor,
        knots: torch.Tensor,
    ):
        """
        Initializes the builder with the constraints and the number of knots and mixtures.
        """
        # Call the constructor of ConstrainedDistributionBuilder
        ConditionalConstraintedDistribution.__init__(self, constraints)
        assert len(var_positions) == 2

        # Call the constructor of torch.nn.Module
        torch.nn.Module.__init__(self)
        self.register_buffer("integral_coeffs", integral_coeffs)
        differences = knots[1:] - knots[:-1]
        self.register_buffer("differences", differences)
        self.register_buffer("knots", knots)
        self.torch_constraints = lra_to_torch(constraints, var_positions)
        self.var_positions = var_positions
        self.num_knots = num_knots
        self.num_mixtures = num_mixtures
        self.poly_unsquared = poly_unsquared

    def _calculate_partition_function_component(
        self,
        param_tensor: torch.Tensor,  # (2, num_knots-1, 4)
    ) -> torch.Tensor:  # (num_knots-1, num_knots-1)
        """
        Returns the partition function of the distribution.
        """
        # compute grid

        all_coeffs_poly_0 = param_tensor[0]  # (num_knots-1, 4)
        all_coeffs_poly_1 = param_tensor[1]  # (num_knots-1, 4)

        # compute all combinations of the coefficients via outer product
        # resulting in a tensor of shape (num_pieces, num_pieces, 4, 2)
        # via meshgrid
        mesh_coeffs_0, mesh_coeffs_1 = torch.vmap(lambda a, b: torch.meshgrid(a, b), in_dims=1, out_dims=2)(
            all_coeffs_poly_0, all_coeffs_poly_1
        )

        meshed_coeffs = torch.stack([mesh_coeffs_0, mesh_coeffs_1], dim=-1)  # (num_knots, num_knots, 4, 2)

        def compute_integral_helper(coeff):
            coeff_0 = coeff[..., 0]
            coeff_1 = coeff[..., 1]
            coeffs_integral_poly = torch.cartesian_prod(coeff_0, coeff_1).prod(-1)
            return coeffs_integral_poly

        coeffs_grid = torch.vmap(torch.vmap(compute_integral_helper))(
            meshed_coeffs
        )  # (num_knots, num_knots, 16)

        # compute the integral of the polynomial
        # per grid:

        def eval_param_single_instance(
            param: torch.Tensor, coeffs: torch.Tensor
        ) -> torch.Tensor:
            combs = coeffs * torch.combinations(param, 2, with_replacement=True).prod(
                dim=-1
            )
            return combs.sum(dim=-1)

        def eval_param_instance_on_grid(
                param_gridded: torch.Tensor
                ) -> torch.Tensor:
            grid_1d_func = torch.vmap(eval_param_single_instance)
            grid_2d_func = torch.vmap(grid_1d_func)

            result = grid_2d_func(param_gridded, self.integral_coeffs)
            return result

        # params_combinations: torch.Tensor = torch.vmap(eval_param_instance_on_grid)(
        #     coeffs_grid
        # )
        return eval_param_instance_on_grid(coeffs_grid)
    
    def calculate_partition_function(
        self,
        poly_params: torch.Tensor,  # (num_mixtures, 2, num_knots-1, 4)
    ) -> torch.Tensor:  # (num_mixtures, num_knots-1, num_knots-1)
        """
        Returns the partition function of the distribution on the grid.
        """
        return torch.vmap(self._calculate_partition_function_component)(
            poly_params
        )

    def forward(
        self,
        params: torch.Tensor,  # [batch_size, num_mixtures, 2, self.num_knots, 2] or [num_mixtures, 2, self.num_knots, 2]
    ) -> "SplineSQ2D":
        if len(params.shape) == 3:
            params = params.unsqueeze(0)

        def do_calc_for_1dcomponent(
            params_for_1dcomponent: torch.Tensor,  # (num_knots, value + derivative)
            knots1d: torch.Tensor,  # (num_knots)
            differences1d: torch.Tensor,  # (num_knots-1)
        ):
            """
            Calculate the polynomial parameters for a 1d component.
            """
            params_1d_value = params_for_1dcomponent[:, 0]
            params_1d_derivative = params_for_1dcomponent[:, 1]
            poly_params = calculate_coefficients_from_hermite_spline(
                knots=knots1d,
                differences=differences1d,
                y=params_1d_value,
                dy=params_1d_derivative,
            )
            return torch.stack(poly_params, dim=-1)  # (num_knots-1, 4)

        def do_calc_for_2dcomponent(
            params_for_component: torch.Tensor,  # (2, num_knots, value + derivative)
        ):
            """
            Calculate the polynomial parameters for a 2d component.
            """
            return torch.vmap(do_calc_for_1dcomponent, in_dims=(0,1,1))(
                params_for_component, self.knots, self.differences
            )  # (2, num_knots-1, 4)

        def do_calc_for_mixture(
            params_for_mixture: torch.Tensor,  # (num_mixtures, 2, num_knots, value + derivative)
        ):
            """
            Calculate the polynomial parameters for a mixture.
            """
            return torch.vmap(do_calc_for_2dcomponent)(
                params_for_mixture
            )  # (num_mixtures, 2, num_knots-1, 4)

        poly_params = torch.vmap(do_calc_for_mixture)(params)  # output (b, num_mixtures, 2, num_knots-1, 4)

        # integrate the polynomial
        integrals_on_grid = torch.vmap(
            self.calculate_partition_function
        )(poly_params)  # (b, num_mixtures, num_knots-1, num_knots-1)

        return SplineSQ2D(
            constraints=self.constraints,
            var_positions=self.var_positions,
            num_knots=self.num_knots,
            num_mixtures=self.num_mixtures,
            poly_unsquared=self.poly_unsquared,
            integrals_2dgrid=integrals_on_grid,
            knots=self.knots,
            differences=self.differences,
            poly_params=poly_params,
        )

    def parameter_shape(self):
        return (
            self.num_mixtures,
            2,
            self.num_knots,
            2,
        )  # (num_mixtures, 2, num_knots, value + derivative)


class SplineSQ2D(ConstrainedDistribution, torch.nn.Module):
    """
    A class that represents a constrained distribution P(Y) for some squared, univariate mixture of splines.
    """

    knots: torch.Tensor  # (num_knots, 2) because of memory layout, these are the positions
    differences: torch.Tensor  # (num_knots-1, 2) because of memory layout
    integrals_2dgrid: torch.Tensor  # (b, num_mixtures, num_knots, num_knots), b=1 gets broadcasted
    poly_params: torch.Tensor  # (b, num_mixtures, 2, num_knots, 4), b=1 gets broadcasted

    def __init__(
        self,
        constraints: lra.LRAProblem,
        torch_constraints: PLRA,
        var_positions: Dict[str, int],
        num_knots: int,
        num_mixtures: int,
        poly_unsquared: TorchPolynomial,
        knots: torch.Tensor,
        differences: torch.Tensor,
        integrals_2dgrid: torch.Tensor,
        poly_params: torch.Tensor,
    ):
        """
        Initializes the builder with the constraints and the number of knots and mixtures.
        """
        # Call the constructor of ConstrainedDistributionBuilder
        ConstrainedDistribution.__init__(self, constraints)
        assert len(var_positions) == 2

        # Call the constructor of torch.nn.Module
        torch.nn.Module.__init__(self)
        self.poly_unsquared = poly_unsquared
        self.num_knots = num_knots
        self.num_mixtures = num_mixtures
        self.var_positions = var_positions
        self.constraints = constraints
        self.torch_constraints = torch_constraints
        self.register_buffer("knots", knots)
        self.register_buffer("differences", differences)
        self.register_buffer("integrals_2dgrid", integrals_2dgrid)
        self.register_buffer("poly_params", poly_params)
