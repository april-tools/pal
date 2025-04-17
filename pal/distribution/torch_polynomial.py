from typing import Dict, List, Self, Any
from frozendict import frozendict
import torch

import numpy as np

from hashlib import sha1


def generate_monomial(
    n: int, max_total_degree: int, max_individual_degree: int
) -> List[List[int]]:
    max_exponent = min(max_individual_degree, max_total_degree)
    if n == 1:
        return [[i] for i in range(max_exponent + 1)]
    else:
        return [
            [i] + monomial
            for i in range(max_exponent + 1)
            for monomial in generate_monomial(
                n - 1, max_total_degree - i, max_individual_degree
            )
        ]


def hash_torch_tensor(tensor: torch.Tensor) -> int:
    np_tensor = tensor.detach().cpu().numpy()
    np_tensor = np.ascontiguousarray(np_tensor)
    dig = sha1(np_tensor).digest()
    return int.from_bytes(dig, byteorder="big")


def stable_np_sum(x: np.ndarray, axis: int) -> np.ndarray:
    x_plus = np.sort(x.clip(min=0), axis=axis)
    x_minus = np.sort(x.clip(max=0), axis=axis)[..., ::-1]
    return np.sum(x_plus, axis=axis) + np.sum(x_minus, axis=axis)


class TorchPolynomial(torch.nn.Module):
    """
    A class representing a parametarized polynomial in PyTorch.

    This class inherits from torch.nn.Module, and it is designed to handle polynomial operations
    using PyTorch tensors.

    Args:
        coeffs (torch.Tensor): Tensor containing the coefficients of the polynomial. Shape: (num_terms,).
        powers (torch.Tensor): Tensor containing the powers of the polynomial. Shape: (num_terms, num_vars).
        variable_map_dict (Dict[str, int] | frozendict[str, int]): Dictionary mapping variable names to their indices.
        absolute (bool): Whether to take the absolute value of the polynomial. Defaults to True.
    """

    coeffs: torch.Tensor
    powers: torch.Tensor

    def __init__(
        self,
        coeffs: torch.Tensor,
        powers: torch.Tensor,
        variable_map_dict: Dict[str, int] | frozendict[str, int],
        absolute: bool = True,
    ) -> None:
        super().__init__()
        self.register_buffer("coeffs", coeffs)
        self.register_buffer("powers", powers)
        self.variable_map_dict = variable_map_dict
        assert coeffs.shape[0] == powers.shape[0]
        self.absolute = absolute

    @classmethod
    def construct(
        cls, max_order: int, max_terms: int, var_map_dict: Dict[str, int]
    ) -> Self:
        n = len(var_map_dict)
        monomials = generate_monomial(n, max_order, max_terms)
        coeffs = torch.ones(len(monomials))
        powers = torch.tensor(monomials, dtype=torch.int64)

        return cls(
            coeffs=coeffs,
            powers=powers,
            variable_map_dict=var_map_dict,
        )

    def reorder_parameter_positions(
        self, param_index_map: Dict[int, int]
    ) -> "TorchPolynomial":
        """
        Reorders the parameters in the polynomial according to the given mapping.
        """
        # assert the new indices are a permutation of the old indices
        assert set(param_index_map.keys()) == set(param_index_map.values())
        new_coeffs = torch.zeros_like(self.coeffs)
        new_powers = torch.zeros_like(self.powers)
        for i in range(self.powers.shape[0]):
            new_position = param_index_map[i]
            new_coeffs[new_position] = self.coeffs[i]
            new_powers[new_position] = self.powers[i]
        return TorchPolynomial(
            coeffs=new_coeffs,
            powers=new_powers,
            variable_map_dict=self.variable_map_dict,
            absolute=self.absolute,
        )

    @torch.compile
    def eval_tensor(
        self,
        y_tensor: torch.Tensor | None,
        param_tensor: torch.Tensor,
        sq_epsilon: float = -1,
    ) -> torch.Tensor:
        assert y_tensor is not None

        @torch.compile
        def eval_single_monomial(
            y: torch.Tensor, monomial: torch.Tensor
        ) -> torch.Tensor:
            # shape of monomial: (num_vars,)
            # shape of y: (num_vars)
            return torch.pow(y, monomial).prod(dim=-1)

        @torch.compile
        def eval_polynomial_single(
            y: torch.Tensor, param: torch.Tensor
        ) -> torch.Tensor:
            monomials = torch.vmap(lambda mon: eval_single_monomial(y, mon))(
                self.powers
            )
            weighted = (self.coeffs * param) * monomials
            return weighted.sum(dim=-1)

        polys = torch.vmap(eval_polynomial_single)(y_tensor, param_tensor)
        if self.absolute:
            polys = polys.abs()
        if sq_epsilon != -1:
            with torch.no_grad():
                polys[polys < sq_epsilon] = sq_epsilon
        return polys

    @torch.compile
    def eval_monomials(self, y_tensor: torch.Tensor) -> torch.Tensor:
        @torch.compile
        def eval_single_monomial(
            y: torch.Tensor, monomial: torch.Tensor
        ) -> torch.Tensor:
            return torch.pow(y, monomial).prod(dim=-1)

        monomials = torch.vmap(
            lambda mon: eval_single_monomial(y_tensor, mon), out_dims=-1
        )(self.powers)
        return monomials

    def square(self) -> "SquaredTorchPolynomial":
        return SquaredTorchPolynomial(
            coeffs=self.coeffs,
            powers=self.powers,
            variable_map_dict=self.variable_map_dict,
        )

    def to_state(self) -> Any:
        state = self.state_dict()
        state["variable_map_dict"] = self.variable_map_dict
        return state

    @classmethod
    def from_state(cls, state_dict: Any) -> Self:
        return cls(
            coeffs=state_dict["coeffs"],
            powers=state_dict["powers"],
            variable_map_dict=state_dict["variable_map_dict"],
        )

    def get_variable_map_dict(self) -> Dict[str, int]:
        return self.variable_map_dict  # type: ignore

    def __hash__(self) -> int:
        return (
            hash_torch_tensor(self.coeffs)
            + hash_torch_tensor(self.powers)
            + hash(self.variable_map_dict)
        )


class SquaredParamsWithCoefficientsTorchPolynomial(torch.nn.Module):
    """
    A class representing a squared polynomial, for which the variables are integrated out,
    so it is only a function of the parameters.
    """

    coeffs: torch.Tensor
    indices_params: torch.Tensor

    def __init__(
        self,
        coeffs: torch.Tensor,
        indices_params: torch.Tensor,
        variable_map_dict: Dict[str, int] | frozendict[str, int]
    ) -> None:
        super().__init__()
        self.register_buffer("coeffs", coeffs)
        self.register_buffer("indices_params", indices_params)
        self.variable_map_dict = variable_map_dict

    @torch.compile
    def eval_tensor(
        self,
        param_tensor: torch.Tensor,
        sq_epsilon: float = -1,
    ) -> torch.Tensor:

        @torch.compile
        def eval_param_instance(param: torch.Tensor) -> torch.Tensor:
            combs = self.coeffs * torch.combinations(
                param, 2, with_replacement=True
            ).prod(dim=-1)
            return combs.sum(dim=-1)

        params_combinations = torch.vmap(eval_param_instance)(param_tensor)

        if sq_epsilon != -1:
            with torch.no_grad():
                params_combinations[params_combinations < sq_epsilon] = sq_epsilon
        return params_combinations

    def to_state(self) -> Any:
        state = self.state_dict()
        state["variable_map_dict"] = self.variable_map_dict
        return state

    @classmethod
    def from_state(cls, state_dict: Any) -> Self:
        return cls(
            coeffs=state_dict["coeffs"],
            indices_params=state_dict["indices_params"],
            variable_map_dict=state_dict["variable_map_dict"],
        )

    def get_variable_map_dict(self) -> Dict[str, int]:
        return self.variable_map_dict  # type: ignore

    def __hash__(self) -> int:
        return (
            hash_torch_tensor(self.coeffs)
            + hash(self.variable_map_dict)
        )


class SquaredTorchPolynomial(torch.nn.Module):
    """
    A class representing a squared polynomial using PyTorch tensors.

    This class inherits from torch.nn.Module, and it is designed to
    handle polynomial operations with PyTorch tensors. The polynomial is represented by
    its coefficients and powers, and it supports both vectorized and non-vectorized evaluations.

    The vectorized version represents a symbolic polynomial as a vectorized, non-symbolic polynomial.
    So, for example, the polynomial x^2*param1^2 + x*z*param1*param2 (symbolic in the parameters)
    would be represented as [x^2, x*z], where the first element represents monomial associated with param1^2
    and the second element represents the monomial associated with param1*param2.
    This allows us to evaluate a symbolic polynomial using PyTorch.
    """

    coeffs: torch.Tensor
    powers: torch.Tensor
    combinations_coefficient: torch.Tensor
    indices_params: torch.Tensor
    shift: torch.Tensor

    def __init__(
        self,
        coeffs: torch.Tensor,
        powers: torch.Tensor,
        variable_map_dict: Dict[str, int] | frozendict[str, int],
        shift: torch.Tensor | None = None,
        indices_params: torch.Tensor | None = None,
        combinations_coefficient: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("coeffs", coeffs)
        self.register_buffer("powers", powers)
        if shift is None:
            shift = torch.zeros(
                len(variable_map_dict), device=powers.device, dtype=coeffs.dtype
            )
        self.register_buffer("shift", shift)
        self._variable_map_dict = variable_map_dict

        if indices_params is None:
            assert combinations_coefficient is None
            indexes = torch.arange(0, powers.shape[0], device=powers.device)
            assert indexes.shape[0] == powers.shape[0]
            combinations = torch.combinations(indexes, 2, with_replacement=True)
            self.register_buffer("indices_params", combinations, persistent=False)
            combinations_coefficient = torch.ones(
                combinations.shape[0], dtype=torch.float32
            )
            combinations_coefficient[combinations[:, 0] != combinations[:, 1]] = 2
            self.register_buffer(
                "combinations_coefficient", combinations_coefficient, persistent=False
            )
        else:
            assert combinations_coefficient is not None
            assert indices_params.shape[0] == combinations_coefficient.shape[0]
            self.register_buffer("indices_params", indices_params, persistent=False)
            self.register_buffer(
                "combinations_coefficient", combinations_coefficient, persistent=False
            )

    def set_shift(self, shift: torch.Tensor) -> None:
        """
        Shifts the point the polynomial is evaluated at.
        """
        self.shift = shift

    def with_shift(self, shift: torch.Tensor) -> "SquaredTorchPolynomial":
        """
        Returns a new polynomial with the given shift.
        """
        return SquaredTorchPolynomial(
            coeffs=self.coeffs,
            powers=self.powers,
            variable_map_dict=self._variable_map_dict,
            shift=shift,
            indices_params=self.indices_params,
            combinations_coefficient=self.combinations_coefficient,
        )

    def get_max_total_degree(self) -> int:
        """
        Returns the maximum total degree of the polynomial.

        Returns:
            int: The maximum total degree.
        """
        return 2 * int(self.powers.sum(dim=-1).max().item())

    def get_num_vars(self) -> int:
        """
        Returns the number of variables in the polynomial.

        Returns:
            int: The number of variables.
        """
        return self.powers.shape[-1]

    def set_dtype(self, dtype: torch.dtype) -> None:
        """
        Sets the data type of the polynomial.

        Args:
            dtype (torch.dtype): The data type to be set.
        """
        self.coeffs = self.coeffs.to(dtype)
        self.combinations_coefficient = self.combinations_coefficient.to(dtype)

    def to_integrated_polynomial(
        self, coeff: torch.Tensor
    ) -> SquaredParamsWithCoefficientsTorchPolynomial:
        """
        Converts an *evaluated* vectorized polynomial (= symbolic polynomial)
        to a SquaredParamsWithCoefficientsTorchPolynomial.

        Args:
            coeff: The coefficients of the polynomial. Results from calling forward (and merging).

        Returns:
            A SquaredParamsWithCoefficientsTorchPolynomial object.
        """
        return SquaredParamsWithCoefficientsTorchPolynomial(
            coeffs=coeff,
            indices_params=self.indices_params,
            variable_map_dict=self._variable_map_dict,
        )

    @torch.compile
    def eval_tensor(
        self,
        y_tensor: torch.Tensor,
        param_tensor: torch.Tensor | None,
        sq_epsilon: float = -1,
        vectorized: bool = False,
    ) -> torch.Tensor:

        y_tensor += self.shift.unsqueeze(0)

        @torch.compile
        def eval_single_monomial(
            y: torch.Tensor, monomial: torch.Tensor
        ) -> torch.Tensor:
            # shape of monomial: (num_vars,)
            # shape of y: (num_vars)
            return torch.pow(y, monomial).prod(dim=-1)

        if not vectorized:

            @torch.compile
            def eval_polynomial_single(
                y: torch.Tensor, param: torch.Tensor
            ) -> torch.Tensor:
                monomials = torch.vmap(lambda mon: eval_single_monomial(y, mon))(
                    self.powers
                )
                weighted = (self.coeffs * param) * monomials
                return weighted.sum(dim=-1) ** 2

            polys = torch.vmap(eval_polynomial_single)(y_tensor, param_tensor)

            if sq_epsilon != -1:
                with torch.no_grad():
                    polys[polys < sq_epsilon] = sq_epsilon
            return polys
        else:
            assert param_tensor is None
            assert sq_epsilon == -1

            @torch.compile
            def eval_vectorized_polynomial(y: torch.Tensor) -> torch.Tensor:
                monomials = torch.vmap(lambda mon: eval_single_monomial(y, mon))(
                    self.powers
                )
                monomials = self.coeffs * monomials
                monomials_combinations = torch.combinations(
                    monomials, 2, with_replacement=True
                ).prod(dim=-1)
                return monomials_combinations * self.combinations_coefficient

            vectorized_mons = torch.vmap(
                eval_vectorized_polynomial, in_dims=0, out_dims=1
            )(y_tensor)
            return vectorized_mons
    
    @torch.compile
    def eval_tensor_vectorized(
        self,
        y_tensor: torch.Tensor
    ) -> torch.Tensor:
        """"
        Evaluates the polynomial at the given tensor y_tensor per parameter.
        This is the vectorized version of the polynomial.
        """
        return self.eval_tensor(y_tensor, None, vectorized=True)

    @torch.compile
    def forward(self, x):
        return self.eval_tensor_vectorized(x)

    def to_state(self) -> Any:
        state = self.state_dict()
        state["variable_map_dict"] = self._variable_map_dict
        return state

    @classmethod
    def from_state(cls, state_dict: Any) -> Self:
        return cls(
            coeffs=state_dict["coeffs"],
            powers=state_dict["powers"],
            variable_map_dict=state_dict["variable_map_dict"],
        )

    def get_variable_map_dict(self) -> Dict[str, int]:
        return self._variable_map_dict  # type: ignore

    def __hash__(self) -> int:
        return (
            hash_torch_tensor(self.coeffs)
            + hash_torch_tensor(self.powers)
            + hash_torch_tensor(self.combinations_coefficient)
            + hash(self._variable_map_dict)
        )
    

@torch.compile
def calculate_coefficients_from_hermite_spline(
    knots: torch.Tensor,
    differences: torch.Tensor,
    y: torch.Tensor,
    dy: torch.Tensor,
    shift: bool = False,
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


@torch.compile
def calc_2d_spline_component(
    x: torch.Tensor,  # (2)
    knot_idx: torch.Tensor,  # (2)
    params: torch.Tensor,  # (2, num_pieces, 4)
) -> torch.Tensor:
    """
    Calculate the 2D spline component for the given x and knot_idx.
    The spline is the product of two cubic polynomials, one for each dimension.
    Args:
        x: The input tensor.
        knot_idx: The indices of the knots.
        params: The parameters of the spline.
    Returns:
        The calculated spline component.
    """
    def per_param(
        p: torch.Tensor,  # (2, num_pieces)
    ) -> torch.Tensor:
        nonlocal knot_idx
        # Ensure knot_idx is of shape (2, 1) for broadcasting
        knot_idx = knot_idx.unsqueeze(-1)
        # Gather the parameters corresponding to knot_idx
        selected_params = torch.gather(p, dim=1, index=knot_idx)
        return selected_params.squeeze(-1)  # (2)

    poly_params = torch.vmap(per_param, in_dims=-1, out_dims=-1)(params)

    poly_per_component = (
        poly_params[..., 0]
        + poly_params[..., 1] * x
        + poly_params[..., 2] * x**2
        + poly_params[..., 3] * x**3
    )

    return poly_per_component.prod(dim=-1)


@torch.compile
def calc_log_mixture(
    x: torch.Tensor,  # (2)
    knot_idx: torch.Tensor,  # (2)
    m_params: torch.Tensor,  # (num_mixtures, 2, num_pieces, 4)
    m_weights: torch.Tensor,  # (num_mixtures)
    m_normalization: torch.Tensor,  # (num_mixtures, num_knots, num_knots)
    eps: float = -1,  # (num_mixtures)
):
    """
    Calculate the mixture of 2D splines.
    Args:
        x: The input tensor.
        knot_idx: The indices of the knots.
        m_params: The parameters of the mixture.
        m_weights: The weights of the mixture.
        m_normalization: The normalization coefficients of the mixture.
        eps: A small value to avoid numerical instability.
    Returns:
        The log of the mixture of 2D splines.
    """
    mixture_poly: torch.Tensor = torch.vmap(
        lambda p: calc_2d_spline_component(x, knot_idx, p)
    )(
        m_params
    )  # .abs_()  # (num_mixtures)
    m_normalization_coeff = m_normalization.sum(dim=-1).sum(
        dim=-1
    )  # (num_mixtures)

    densities_components = mixture_poly ** 2 / m_normalization_coeff
    poly_log = torch.logsumexp(
        torch.log(densities_components) + torch.log(m_weights), dim=0
    )

    return poly_log
