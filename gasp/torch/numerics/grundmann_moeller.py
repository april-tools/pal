from typing import Callable
import torch
import math
import numpy as np

# hack for the "RuntimeError: CUDA driver initialization failed." error
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# based on:
# A. Grundmann, H. M. Möller,
# "Invariant integration formulas for the n-simplex by combinatorial methods",
# SIAM J. Numer. Anal. 15, 282-290 (1978), DOI 10.1137/0715019.


def get_summing_up_to(max_degree, sum, length, non_increasing=True, current=0):
    """
    Generates tuples of integers that sum up to a given value.

    This function recursively generates all possible tuples of a specified length
    where the elements sum up to a given value. The elements of the tuples can be
    constrained to be non-increasing.

    Parameters:
    max_degree (int): The maximum value any element in the tuple can take.
    sum (int): The target sum for the elements of the tuple.
    length (int): The number of elements in the tuple.
    non_increasing (bool, optional): If True, the elements in the tuple will be in non-increasing order.
                                     Defaults to True.
    current (int, optional): The current sum of elements in the tuple during recursion.
                             Defaults to 0.

    Returns:
    list of tuples: A list of tuples where each tuple contains integers that sum up to the specified value.
    """
    assert sum >= 0
    assert length >= 1
    if length == 1:
        if non_increasing:
            if sum <= max_degree:
                return [(sum,)]
            else:
                return []
        else:
            residual = sum - current
            if residual <= max_degree:
                return [(residual,)]
            else:
                return []
    if non_increasing:
        max_element = min(max_degree, sum)
        return [
            (i,) + t
            for i in range(max_element + 1)
            for t in get_summing_up_to(
                i, sum - i, length - 1,
                non_increasing=True, current=current + i
            )
        ]
    else:
        max_element = min(max_degree, sum - current)
        return [
            (i,) + t
            for i in range(max_element + 1)
            for t in get_summing_up_to(
                max_degree, sum, length - 1,
                non_increasing=False, current=current + i
            )
        ]


def prepare_grundmann_moeller(s, n, non_increasing=False):
    """
    Prepares the Grundmann-Moeller quadrature rule for integration over a simplex.
    Exact up to degree 2s+1.

    Parameters:
    s (int): The parameter s in the Grundmann-Moeller quadrature rule.
    n (int): The dimension of the simplex.

    Returns:
    tuple: A tuple containing:
        - coefficients (numpy.ndarray): The weights for the quadrature points.
        - points (numpy.ndarray): The quadrature points in the simplex.
    """
    d = 2 * s + 1

    coefficients = []
    cs = []
    points = []
    for i in range(s + 1):
        c_nom = ((-1) ** i) * (2 ** (-2 * s)) * (d + n - 2 * i) ** d
        c_denom = math.factorial(i) * math.factorial(d + n - i)
        c = c_nom / c_denom
        betas = get_summing_up_to(s-i, s - i, n + 1, non_increasing=non_increasing)
        ps = []
        # beta is a vector as a tuple
        for beta in betas:
            if non_increasing:
                assert all(beta[i] >= beta[i + 1] for i in range(n))
            assert len(beta) == n + 1
            assert sum(beta) == s - i

            def to_point(beta):
                nom = 2 * beta + 1
                denom = d + n - 2 * i
                return nom / denom

            position = [to_point(b) for b in beta]
            ps.append(np.array(position))

        ps = np.stack(ps, axis=0)
        cs.append(c)
        coefficients.append(np.repeat(c, ps.shape[0]))
        points.append(ps)

    points = np.concatenate(points, axis=0)
    coefficients = np.concatenate(coefficients, axis=0)
    coefficients /= np.sum(coefficients)
    assert coefficients.shape[0] == points.shape[0]
    return coefficients, points


@torch.jit.script
def sum_weighted(
    values: torch.Tensor, sum_seperately: bool = True, with_sorting: bool = True
) -> torch.Tensor:
    if sum_seperately:
        w_positive = values.clone().clamp(min=0)
        w_negative = values.clone().clamp(max=0)
        # c_positive = coeffs >= 0
        # w_positive = weighted[:, c_positive]
        # w_negative = weighted[:, ~c_positive]
        if with_sorting:
            # is this improving numerical stability?
            w_positive = w_positive.sort(dim=-1, descending=False).values
            w_negative = w_negative.sort(dim=-1, descending=True).values
        w_sum = w_positive.sum(dim=-1) + w_negative.sum(dim=-1)
        final = w_sum
    else:
        final = values.sum(dim=-1)
    return final


def integrate(
    f: Callable[[torch.Tensor], torch.Tensor],
    coeffs: torch.Tensor,
    points: torch.Tensor,
    simplex_vs: torch.Tensor,
    sum_seperately=True,
    with_sorting=False,
    batched: int | None = None
) -> torch.Tensor:
    """
    Integrates a function over a simplex using the Grundmann-Moeller quadrature method.

    Args:
        f (Callable[[torch.Tensor], torch.Tensor]): The function to integrate.
        coeffs (torch.Tensor): The coefficients for the quadrature points.
        points (torch.Tensor): The quadrature points in the reference simplex.
        simplex_vs (torch.Tensor): The vertices of the simplex over which to integrate.
        sum_seperately (bool, optional): If True, positive and negative contributions to the integral are
            summed separately to improve numerical stability. Defaults to True.
        with_sorting (bool, optional): If True, sorts the positive and negative contributions
            before summing to further improve numerical stability. Defaults to False.
        batched (int, optional): If not None, the function will be evaluated in batches of this size (over points).

    Returns:
        torch.Tensor: The computed integral of the function over the simplex.
    """
    # assert simplex_vs.shape[0] == points.shape[-1] + 1

    xs = points @ simplex_vs
    vol = simplex_volume(simplex_vs)

    if batched is None:
        out = f(xs)
        if len(out.shape) == 1:
            out = out.unsqueeze(0)
        values = coeffs.unsqueeze(0) * out
    else:
        values = []
        for i in range(0, points.shape[0], batched):
            cs = coeffs[i:i + batched]
            out = f(xs[i:i + batched])
            if len(out.shape) == 1:
                out = out.unsqueeze(0)
            weighted = cs.unsqueeze(0) * out
            batch_sum_weighted = sum_weighted(weighted, sum_seperately=sum_seperately, with_sorting=with_sorting)
            values.append(batch_sum_weighted.unsqueeze(-1))
        values = torch.cat(values, dim=-1)
    if sum_seperately:
        w_positive = values.clone().clamp(min=0)
        w_negative = values.clone().clamp(max=0)
        # c_positive = coeffs >= 0
        # w_positive = weighted[:, c_positive]
        # w_negative = weighted[:, ~c_positive]
        if with_sorting:
            # is this improving numerical stability?
            w_positive = w_positive.sort(dim=-1, descending=False).values
            w_negative = w_negative.sort(dim=-1, descending=True).values
        w_sum = w_positive.sum(dim=-1) + w_negative.sum(dim=-1)
        final = w_sum * vol
    else:
        final = values.sum(dim=-1) * vol
    if len(values.shape) == 1:
        return final.squeeze(0)
    else:
        return final


def simplex_volume(vs):
    """
    Calculate the volume of a simplex given its vertices.

    Parameters:
    vs (torch.Tensor): A tensor of shape (n+1, n) representing the vertices of the simplex,
                       where n is the dimension of the simplex.

    Returns:
    float: The volume of the simplex.
    """
    n = vs.shape[-1]
    return torch.abs(torch.det(vs[1:] - vs[0])) / math.factorial(n)


def stroud(monomial_exponents, n):
    """
    Calculates the exactl integral for the monomial over the unit simplex of dimension n.

    Parameters:
    monomial_exponents (list): The exponents of the monomial.
    n (int): The dimension of the simplex.

    Returns:
    float: The exact integral of the monomial over the simplex.
    """
    # A. H. Stroud, Approximate Calculation of Multiple Integrals,
    # Prentice-Hall, Englewood Cliffs, NJ, 1971
    nom = [math.factorial(k) for k in monomial_exponents]
    denom = math.factorial(n + sum(monomial_exponents))
    return math.prod(nom) / denom


# check if main
# used for debugging
if __name__ == "__main__":
    # simplex vertices
    v1 = [1.0, 1.0, 1.0]
    v2 = [2.0, 2.0, 3.0]
    v3 = [3.0, 4.0, 5.0]
    v4 = [3.0, 2.0, 1.0]
    # simplex
    S = [v1, v2, v3, v4]
    S = torch.tensor(S)
    # polynomial to integrate

    # x**4 + y + 2 * x * y**2 - 3 * z
    def p(batch):
        x = batch[:, 0]
        y = batch[:, 1]
        z = batch[:, 2]

        return x**4 + y + 2 * x * (y**2) - 3 * z
    # integral
    coefficients, points = prepare_grundmann_moeller(4, 3)
    coefficients = torch.tensor(coefficients).to(torch.float32)
    points = torch.tensor(points).to(torch.float32)
    print(integrate(p, coefficients, points, S))

    unit_simplex = torch.zeros((4, 3))
    for k in range(3):
        unit_simplex[k + 1, k] = 1

    def mono(x):
        return x[:, 0] ** 1 * x[:, 1] ** 2 * x[:, 2] ** 3

    val_gm = integrate(mono, coefficients, points, unit_simplex).item()

    stroud_val = stroud([1, 2, 3], 3)

    assert np.allclose(val_gm, stroud_val)
