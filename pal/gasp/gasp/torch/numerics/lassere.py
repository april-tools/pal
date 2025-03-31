# from https://www.r-bloggers.com/2022/12/exact-integral-of-a-polynomial-on-a-simplex/  # noqa
# only used as a reference in the tests
# TODO: license?

from math import factorial
from sympy import Poly
import numpy as np
from sympy.abc import x, y, z
import sympy as sp


def _term(Q, monom):
    coef = Q.coeff_monomial(monom)
    powers = list(monom)
    j = sum(powers)
    if j == 0:
        return coef
    coef = coef * np.prod(list(map(factorial, powers)))
    n = len(monom)
    return coef / np.prod(list(range(n + 1, n + j + 1)))


def integratePolynomialOnSimplex(P, S, domain="RR", with_rational=False):
    gens = P.gens
    n = len(gens)
    S = np.asarray(S)
    v = S[n, :]
    columns = []
    for i in range(n):
        columns.append(S[i, :] - v)
    B = np.column_stack(tuple(columns))
    dico = {}
    for i in range(n):
        newvar = v[i]
        if with_rational:
            newvar = sp.Rational(newvar)
        for j in range(n):
            coeff = B[i, j]
            if with_rational:
                coeff = sp.Rational(coeff)
            newvar = newvar + coeff * Poly(gens[j], gens, domain=domain)
        dico[gens[i]] = newvar.as_expr()
    Q = P.subs(dico, simultaneous=True).as_expr().as_poly(gens)
    # print(Q)
    monoms = Q.monoms()
    s = 0.0
    if with_rational:
        s = sp.Rational(0.0)
    for monom in monoms:
        s = s + _term(Q, monom)
    return np.abs(np.linalg.det(B)) / factorial(n) * s


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
    # polynomial to integrate

    P = Poly(x**4 + y + 2 * x * (y**2) - 3 * z, x, y, z, domain="RR")
    # integral
    print(integratePolynomialOnSimplex(P, S))
