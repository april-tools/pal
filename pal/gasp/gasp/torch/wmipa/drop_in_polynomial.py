import torch
from wmipa.integration.polynomial import Polynomial, Monomial


def create_tensors_from_polynomial(
    poly: Polynomial, variable_map: dict[str, int], dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    # I think we should really prepare it beforehand instead of building the same thing over and over again!!!
    monomials: list[Monomial] = poly.monomials
    monomial_size = len(monomials)

    coeffs = torch.zeros(monomial_size, dtype=dtype)
    exponents = torch.zeros(monomial_size, len(variable_map), dtype=torch.int64)

    for i, monomial in enumerate(monomials):
        coeffs[i] = float(monomial.coefficient)
        for var, exp in monomial.exponents.items():
            exponents[i, variable_map[str(var)]] += int(exp)

    return coeffs, exponents


class SinglePolynomial(torch.nn.Module):

    coeffs: torch.Tensor  # shape [n_monomials]
    exponents: torch.Tensor  # shape [n_monomials, n_vars]

    def __init__(
        self,
        coeffs: torch.Tensor,
        exponents: torch.Tensor,
        variable_map: dict[str, int],  # name => index
    ):
        super().__init__()
        # register coeffs as module buffer
        self.register_buffer("coeffs", coeffs)
        # register exponents as module buffer
        self.register_buffer("exponents", exponents)
        assert exponents.dtype == torch.int64

        self.variable_map = variable_map

    @classmethod
    def from_pa_polynomal(cls, poly: Polynomial, variable_map: dict[str, int]):
        # I think we should really prepare it beforehand instead of building the same thing over and over again!!!
        coeffs, exponents = create_tensors_from_polynomial(poly, variable_map)

        return cls(coeffs, exponents, variable_map)

    def forward(self, xs: torch.Tensor):

        def calc_monomial(coeff, exponents, x):
            return coeff * torch.prod(x**exponents)

        def calc_polynomial(coeffs, exponents_poly, x):
            monomials = torch.vmap(lambda c, expo: calc_monomial(c, expo, x))(
                coeffs, exponents_poly
            )
            return monomials.sum()

        return torch.vmap(lambda x: calc_polynomial(self.coeffs, self.exponents, x))(xs)

    def get_max_total_degree(self):
        return self.exponents.sum(dim=1).max().item()

    def get_num_vars(self):
        return self.exponents.shape[1]

    def get_variable_map_dict(self):
        return self.variable_map


def calc_batched_polynomial(
    xs: torch.Tensor,  # shape [batch_size, n_vars]
    coeffs_batched: torch.Tensor,  # shape [batch_size, n_monomials]
    exponents_batched: torch.Tensor,  # shape [batch_size, n_monomials, n_vars]
) -> torch.Tensor:  # shape [batch_size]
    """
    Instead of calculating a single polynomial, calculate a batch of polynomials.
    Each polynomial has it's own input, so we have as many inputs as we have polynomials.
    """

    def calc_monomial(coeff, exponents, x):
        return coeff * torch.prod(x**exponents)

    def calc_polynomial(coeffs, exponents_poly, x):
        monomials = torch.vmap(lambda c, expo: calc_monomial(c, expo, x))(
            coeffs, exponents_poly
        )
        return monomials.sum(dim=0)

    return torch.vmap(calc_polynomial)(coeffs_batched, exponents_batched, xs)


def calc_single_polynomial(
    xs: torch.Tensor,  # shape [batch_size, n_vars]
    coeffs: torch.Tensor,  # shape [n_monomials]
    exponents: torch.Tensor,  # shape [n_monomials, n_vars]
):
    """
    Calculate a single polynomial for a batch of inputs.
    """
    def calc_monomial(coeff, exponents, x):
        return coeff * torch.prod(x**exponents)

    def calc_polynomial(x):
        monomials = torch.vmap(lambda c, expo: calc_monomial(c, expo, x))(
            coeffs, exponents
        )
        return monomials.sum(dim=0)

    return torch.vmap(calc_polynomial)(xs)
