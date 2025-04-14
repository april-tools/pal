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
        coeffs (torch.Tensor): Tensor containing the coefficients of the polynomial.
        powers (torch.Tensor): Tensor containing the powers of the polynomial.
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
        assert set(param_index_map.keys()) == set(self.param_map_dict.values())
        new_coeffs = torch.zeros_like(self.coeffs)
        new_powers = torch.zeros_like(self.powers)
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

    def numerically_stable_comp(
        self,
        y_tensor: torch.Tensor | None,
        param_tensor: torch.Tensor,
        sq_epsilon: float = -1,
    ) -> torch.Tensor:
        assert y_tensor is not None

        np_y_tensor: np.ndarray = y_tensor.detach().cpu().numpy()
        np_param_tensor: np.ndarray = param_tensor.detach().cpu().numpy()
        np_coeffs: np.ndarray = self.coeffs.detach().cpu().numpy()
        np_powers: np.ndarray = self.powers.detach().cpu().numpy()

        np_y_tensor = np_y_tensor.astype(np.float128)
        np_param_tensor = np_param_tensor.astype(np.float128)
        np_coeffs = np_coeffs.astype(np.float128)

        monomials = np.power(
            np_y_tensor[..., np.newaxis, :], np_powers[np.newaxis, :]
        ).prod(axis=-1)
        weighted = (np_coeffs[np.newaxis, :] * np_param_tensor) * monomials
        polys = stable_np_sum(weighted, axis=-1)
        polys = polys.astype(np.float32)

        poly_torch = torch.tensor(polys, device=y_tensor.device, dtype=y_tensor.dtype)

        if self.absolute:
            poly_torch = poly_torch.abs()

        if sq_epsilon != -1:
            with torch.no_grad():
                poly_torch[poly_torch < sq_epsilon] = sq_epsilon
        return poly_torch

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

    def numerically_stable_comp(
        self,
        y_tensor: torch.Tensor | None,
        param_tensor: torch.Tensor,
        sq_epsilon: float = -1,
    ) -> torch.Tensor:
        assert y_tensor is None
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
            results.append(stable_np_sum(np_coeffs * combinations_prod, axis=-1))
        results_tensor = torch.tensor(
            results, device=param_tensor.device, dtype=param_tensor.dtype
        )
        if sq_epsilon != -1:
            with torch.no_grad():
                results_tensor[results_tensor < sq_epsilon] = sq_epsilon
        return results_tensor

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

        y_tensor = y_tensor + self.shift.unsqueeze(0)

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

    def numerically_stable_comp(
        self,
        y_tensor: torch.Tensor | None,
        param_tensor: torch.Tensor,
        sq_epsilon: float = -1,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_variable_map_dict(self) -> Dict[str, int]:
        return self._variable_map_dict  # type: ignore

    def __hash__(self) -> int:
        return (
            hash_torch_tensor(self.coeffs)
            + hash_torch_tensor(self.powers)
            + hash_torch_tensor(self.combinations_coefficient)
            + hash(self._variable_map_dict)
        )


def create_torch_polynomial(vars: List[str], config: Dict[str, Any]) -> TorchPolynomial:
    return TorchPolynomial.construct(
        max_order=config["max_order"],
        max_terms=config["max_terms"],
        var_map_dict=config["var_map_dict"],
    )
