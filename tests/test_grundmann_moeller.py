import math
import torch
import numpy as np
from gasp.torch.numerics.grundmann_moeller import integrate, prepare_grundmann_moeller, stroud
from gasp.torch.numerics.lassere import integratePolynomialOnSimplex
from sympy import symbols, Poly


def test_integrate_polynomial():
    # Define the vertices of the simplex
    v1 = [1.0, 1.0, 1.0]
    v2 = [2.0, 2.0, 3.0]
    v3 = [3.0, 4.0, 5.0]
    v4 = [3.0, 2.0, 1.0]
    S = torch.tensor([v1, v2, v3, v4], dtype=torch.float32)

    # Define the polynomial function to integrate
    def p(batch):
        x = batch[:, 0]
        y = batch[:, 1]
        z = batch[:, 2]
        return x**4 + y + 2 * x * (y**2) - 3 * z

    # Prepare the Grundmann-Moeller quadrature rule
    coefficients, points = prepare_grundmann_moeller(4, 3)
    coefficients = torch.tensor(coefficients, dtype=torch.float32)
    points = torch.tensor(points, dtype=torch.float32)

    # Perform the integration
    result = integrate(p, coefficients, points, S)

    # Assert the result is a tensor
    assert isinstance(result, torch.Tensor)

    # Calculate the exact integral using Lassere's method
    # sympy poly for polynomial p
    p_sympy = Poly("x**4 + y + 2 * x * (y**2) - 3 * z", symbols("x y z"), domain="RR")
    lassere_integral = integratePolynomialOnSimplex(p_sympy, S.numpy())

    # Assert the result is close to the exact value
    assert np.allclose(result.item(), float(lassere_integral))


def test_integrate_monomial():
    # Define the unit simplex
    unit_simplex = torch.zeros((4, 3), dtype=torch.float32)
    for k in range(3):
        unit_simplex[k + 1, k] = 1

    # Define the monomial function to integrate
    def mono(x):
        return x[:, 0] ** 1 * x[:, 1] ** 2 * x[:, 2] ** 3

    # Prepare the Grundmann-Moeller quadrature rule
    coefficients, points = prepare_grundmann_moeller(4, 3)
    coefficients = torch.tensor(coefficients, dtype=torch.float32)
    points = torch.tensor(points, dtype=torch.float32)

    # Perform the integration
    val_gm = integrate(mono, coefficients, points, unit_simplex).item()

    # Calculate the exact integral using Stroud's method
    stroud_val = stroud([1, 2, 3], 3)

    # Assert the result is close to the exact value
    assert np.allclose(val_gm, stroud_val)


def test_integrate_with_sorting():
    # Define the unit simplex
    unit_simplex = torch.zeros((4, 3), dtype=torch.float32)
    for k in range(3):
        unit_simplex[k + 1, k] = 1

    # Define the monomial function to integrate
    def mono(x):
        return x[:, 0] ** 1 * x[:, 1] ** 2 * x[:, 2] ** 3

    # Prepare the Grundmann-Moeller quadrature rule
    coefficients, points = prepare_grundmann_moeller(4, 3)
    coefficients = torch.tensor(coefficients, dtype=torch.float32)
    points = torch.tensor(points, dtype=torch.float32)

    # Perform the integration with sorting
    val_gm_sorted = integrate(mono, coefficients, points, unit_simplex, with_sorting=True).item()

    # Perform the integration without sorting
    val_gm_unsorted = integrate(mono, coefficients, points, unit_simplex, with_sorting=False).item()

    # Calculate the exact integral using Stroud's method
    stroud_val = stroud([1, 2, 3], 3)

    # Assert the results are close to the exact value
    assert np.allclose(val_gm_sorted, stroud_val)
    assert np.allclose(val_gm_unsorted, stroud_val)


def test_integrate_high_exponent_unit_simplex():
    # Define the unit simplex
    unit_simplex = torch.zeros((4, 3), dtype=torch.float32)
    for k in range(3):
        unit_simplex[k + 1, k] = 1
    
    deg = 12

    # Define the monomial function to integrate
    def mono(x):
        return x[:, 0] ** 0 * x[:, 1] ** deg * x[:, 2] ** 0
    
    s = math.ceil((deg - 1) / 2)
    # Prepare the Grundmann-Moeller quadrature rule
    coefficients, points = prepare_grundmann_moeller(s, 3)

    coefficients = torch.tensor(coefficients, dtype=torch.float32)
    points = torch.tensor(points, dtype=torch.float32)

    # Perform the integration with sorting
    val_gm_sorted = integrate(mono, coefficients, points, unit_simplex, with_sorting=True).item()

    # Calculate the exact integral using Stroud's method
    stroud_val = stroud([0, 12, 0], 3)

    # Assert the results are close to the exact value
    assert np.allclose(val_gm_sorted, stroud_val)


def test_integrate_high_exponent_triangle_nstar():
    triangle = np.array([
        [8.66025404e+00, -5.00000000e+00],
        [4.00617226e-16,  1.00000000e+01],
        [-8.66025404e+00, -5.00000000e+00]])
    
    triangle_torch = torch.tensor(triangle, dtype=torch.float64)
    
    deg = 12

    # Define the monomial function to integrate
    def mono(x):
        return x[:, 0] ** 0 * x[:, 1] ** deg
    
    s = math.ceil((deg - 1) // 2)
    s = s + 1
    # Prepare the Grundmann-Moeller quadrature rule
    coefficients, points = prepare_grundmann_moeller(s, 2)

    coefficients = torch.tensor(coefficients, dtype=torch.float64)
    points = torch.tensor(points, dtype=torch.float64)

    # Perform the integration with sorting
    val_gm_sorted = integrate(mono, coefficients, points, triangle_torch, with_sorting=False).item()

    # Calculate the integral using lassere
    p_sympy = Poly("y**12", symbols("x y"), domain="QQ")
    lassere_integral = integratePolynomialOnSimplex(p_sympy, triangle, domain="QQ", with_rational=True)

    # Assert the results are close to the exact value
    assert np.allclose(val_gm_sorted, float(lassere_integral))


def test_integrate_high_exponent_triangle_nstar_batched():
    triangle = np.array([
        [8.66025404e+00, -5.00000000e+00],
        [4.00617226e-16,  1.00000000e+01],
        [-8.66025404e+00, -5.00000000e+00]])
    
    triangle_torch = torch.tensor(triangle, dtype=torch.float64)
    
    deg = 12

    # Define the monomial function to integrate
    def mono(x):
        return x[:, 0] ** 0 * x[:, 1] ** deg
    
    s = math.ceil((deg - 1) // 2)
    s = s + 1
    # Prepare the Grundmann-Moeller quadrature rule
    coefficients, points = prepare_grundmann_moeller(s, 2)

    coefficients = torch.tensor(coefficients, dtype=torch.float64)
    points = torch.tensor(points, dtype=torch.float64)

    batch_size = 10
    assert batch_size < points.shape[0]

    # Perform the integration with sorting
    val_gm_sorted = integrate(mono, coefficients, points, triangle_torch, with_sorting=False, batched=batch_size).item()

    # Calculate the integral using lassere
    p_sympy = Poly("y**12", symbols("x y"), domain="QQ")
    lassere_integral = integratePolynomialOnSimplex(p_sympy, triangle, domain="QQ", with_rational=True)

    # Assert the results are close to the exact value
    assert np.allclose(val_gm_sorted, float(lassere_integral))