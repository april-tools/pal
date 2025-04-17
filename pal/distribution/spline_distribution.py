from typing import Callable, Dict
import torch

from pal.logic.lra_torch import PLRA, lra_to_torch
from pal.distribution.constrained_distribution import (
    Box,
    ConstrainedDistribution,
    ConstrainedDistributionBuilder,
    ConditionalConstraintedDistribution,
)
import pal.logic.lra as lra
from pal.distribution.torch_polynomial import (
    TorchPolynomial,
    calc_log_mixture,
    calculate_coefficients_from_hermite_spline,
)


def compute_reordering_of_parameter_positions2d(
    deg: int, powers: torch.Tensor  # shape [n_mon, 2]
) -> Dict[int, int]:
    """
    Compute the reordering of the parameter positions for the polynomial so that
    it conforms to the combination of univariate polynomials.

    Parameters:
    - deg (int): The degree of the polynomial.
    - powers (torch.Tensor): A tensor of shape [n_mon, 2] containing the powers of the polynomial terms.
      Each row represents a monomial with two variables.

    Returns:
    - Dict[int, int]: A mapping from the current parameter index to the reordered parameter index.
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
        assert num_knots > 1
        self.num_mixtures = num_mixtures
        assert num_mixtures > 0

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

        for i in range(self.knots.shape[0] - 1):
            for j in range(self.knots.shape[0] - 1):
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

                def eval_with_shift(y: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
                    y_shifted = y - shift
                    return self.squared_poly.eval_tensor_vectorized(y_shifted)

                # due to reuse of the scope we need to bing the shift
                def scoped_eval(shift: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
                    return lambda y: eval_with_shift(y, shift)

                results.append((box, scoped_eval(lower_left), shape))
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


Args_forward = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class ConditionalSplineSQ2D(
    ConditionalConstraintedDistribution[ConstrainedDistribution, Args_forward]
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
        self.register_buffer("knots", knots.contiguous())
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
        mesh_coeffs_0, mesh_coeffs_1 = torch.vmap(
            lambda a, b: torch.meshgrid(a, b), in_dims=1, out_dims=2
        )(all_coeffs_poly_0, all_coeffs_poly_1)

        meshed_coeffs = torch.stack(
            [mesh_coeffs_0, mesh_coeffs_1], dim=-1
        )  # (num_knots, num_knots, 4, 2)

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

        def eval_param_instance_on_grid(param_gridded: torch.Tensor) -> torch.Tensor:
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
        return torch.vmap(self._calculate_partition_function_component)(poly_params)

    def forward(
        self,
        params_value: torch.Tensor,  # [batch_size, num_mixtures, 2, self.num_knots]
                                     # or [num_mixtures, 2, self.num_knots]
        params_derivative: torch.Tensor,  # [batch_size, num_mixtures, 2, self.num_knots]
                                          # or [num_mixtures, 2, self.num_knots]
        params_mixture_weights: torch.Tensor,  # [batch_size, num_mixtures] or [num_mixtures]
    ) -> "SplineSQ2D":
        if len(params_value.shape) == 3:
            assert len(params_derivative.shape) == 3
            assert len(params_mixture_weights.shape) == 1

            params_value = params_value.unsqueeze(0)
            params_derivative = params_derivative.unsqueeze(0)
            params_mixture_weights = params_mixture_weights.unsqueeze(0)

        def do_calc_for_1dcomponent(
            params_1d_value: torch.Tensor,  # (num_knots)
            params_1d_derivative,  # (num_knots)
            knots1d: torch.Tensor,  # (num_knots)
            differences1d: torch.Tensor,  # (num_knots-1)
        ):
            """
            Calculate the polynomial parameters for a 1d component.
            """
            poly_params = calculate_coefficients_from_hermite_spline(
                knots=knots1d,
                differences=differences1d,
                y=params_1d_value,
                dy=params_1d_derivative,
            )
            return torch.stack(poly_params, dim=-1)  # (num_knots-1, 4)

        def do_calc_for_2dcomponent(
            params_2d_value: torch.Tensor,  # (2, num_knots)
            params_2d_derivative,  # (2, num_knots)
        ):
            """
            Calculate the polynomial parameters for a 2d component.
            """
            return torch.vmap(do_calc_for_1dcomponent, in_dims=(0, 0, 1, 1))(
                params_2d_value, params_2d_derivative, self.knots, self.differences
            )  # (2, num_knots-1, 4)

        def do_calc_for_mixture(
            params_for_mixture_value: torch.Tensor,  # (num_mixtures, 2, num_knots)
            params_for_mixture_derivative: torch.Tensor,  # (num_mixtures, 2, num_knots)
        ):
            """
            Calculate the polynomial parameters for a mixture.
            """
            return torch.vmap(do_calc_for_2dcomponent)(
                params_for_mixture_value, params_for_mixture_derivative
            )  # (num_mixtures, 2, num_knots-1, 4)

        poly_params = torch.vmap(do_calc_for_mixture)(
            params_value, params_derivative
        )  # output (b, num_mixtures, 2, num_knots-1, 4)

        # integrate the polynomial
        integrals_on_grid = torch.vmap(self.calculate_partition_function)(
            poly_params
        )  # (b, num_mixtures, num_knots-1, num_knots-1)

        return SplineSQ2D(
            constraints=self.constraints,
            torch_constraints=self.torch_constraints,
            var_positions=self.var_positions,
            num_knots=self.num_knots,
            num_mixtures=self.num_mixtures,
            poly_unsquared=self.poly_unsquared,
            integrals_2dgrid=integrals_on_grid,
            knots=self.knots,
            differences=self.differences,
            poly_params=poly_params,
            mixture_weights=params_mixture_weights,
        )

    def __call__(
        self,
        params_value: torch.Tensor,
        params_derivative: torch.Tensor,
        params_mixture_weights: torch.Tensor,
    ) -> "SplineSQ2D":
        return super().__call__(params_value, params_derivative, params_mixture_weights)

    def __repr__(self) -> str:
        m = self.num_mixtures
        n = self.num_knots
        c = self.constraints.name
        return (
            f"ConditionalSplineSQ2D(num_mixtures={m}, num_knots={n}, constraints={c})"
        )

    def parameter_shape(self):
        knots = (self.num_mixtures, 2, self.num_knots)
        mixture_shape = (self.num_mixtures,)
        return [
            knots,  # value
            knots,  # derivative
            mixture_shape,  # mixture weights
        ]


class SplineSQ2D(ConstrainedDistribution, torch.nn.Module):
    """
    A class that represents a constrained distribution P(Y) for some squared, univariate mixture of splines.
    """

    knots: (
        torch.Tensor
    )  # (num_knots, 2) because of memory layout, these are the positions
    differences: torch.Tensor  # (num_knots-1, 2) because of memory layout
    integrals_2dgrid: (
        torch.Tensor
    )  # (b, num_mixtures, num_knots, num_knots), b=1 gets broadcasted
    poly_params: (
        torch.Tensor
    )  # (b, num_mixtures, 2, num_knots, 4), b=1 gets broadcasted
    mixture_weights: torch.Tensor  # (b, num_mixtures), b=1 gets broadcasted

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
        mixture_weights: torch.Tensor,
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
        self.torch_constraints = torch_constraints
        self.register_buffer("knots", knots)
        self.register_buffer("differences", differences)
        self.register_buffer("integrals_2dgrid", integrals_2dgrid)
        self.register_buffer("poly_params", poly_params)
        self.register_buffer("mixture_weights", mixture_weights)

    def get_starting_coords_bin(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the starting coordinates of the bin that x is in.
        """
        x0 = torch.searchsorted(self.knots[:, 0], x[:, 0])
        x0 = torch.clamp(x0 - 1, 0, len(self.knots) - 2)

        x1 = torch.searchsorted(self.knots[:, 1], x[:, 1])
        x1 = torch.clamp(x1 - 1, 0, len(self.knots) - 2)

        return torch.stack([self.knots[x0, 0], self.knots[x1, 1]], dim=1)

    def get_bins(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the indices and the starting coordinates of the bin that x is in.
        """

        def get_info_1d(
            knots,  # (num_knots)
            xs,  # (batch_size)
        ):
            knot_idx = torch.searchsorted(knots, xs)
            knot_idx = torch.clamp(knot_idx - 1, 0, len(knots) - 2)

            start_coords = knots[knot_idx]

            return start_coords, knot_idx

        get_info_2d = torch.vmap(get_info_1d, in_dims=(1, 1), out_dims=1)

        return get_info_2d(self.knots, x)

    def log_dens(self, x, eps=-1, with_indicator=False) -> torch.Tensor:
        """
        Essentially computes log p(x) for the distribution, where p(x) is
        composed of:
        p_i(x) = poly_i^b(x - start_coords^b)^2 / integral_coeffs_i^b"""
        start_coords, knot_idxs = self.get_bins(x)

        eval_point = x - start_coords

        if self.poly_params.shape[0] == 1:
            # single parameter set!
            log_dens = torch.vmap(
                lambda x_elem, knot_elem: calc_log_mixture(
                    x_elem,
                    knot_elem,
                    self.poly_params[0],
                    self.mixture_weights[0],
                    self.integrals_2dgrid[0],
                    eps,
                )
            )(eval_point, knot_idxs)
        else:
            # batch of parameters
            log_dens = torch.vmap(
                lambda *args: calc_log_mixture(*args, eps=eps)
            )(
                eval_point,
                knot_idxs,
                self.poly_params,
                self.mixture_weights,
                self.integrals_2dgrid,
            )
        if with_indicator:
            is_valid = self.torch_constraints(x)
            log_dens[~is_valid] = float("-inf")
        
        return log_dens
    
    def enumerate_pieces(self):
        raise NotImplementedError

    def __repr__(self):
        m = self.num_mixtures
        n = self.num_knots
        c = self.constraints.name
        if self.poly_params.shape[0] == 1:
            b = ""
        else:
            b = f"batched={self.poly_params.shape[0]}, "
        return f"SplineSQ2D({b}num_mixtures={m}, num_knots={n}, constraints={c})"
