from pal.wmi.compute_integral import integrate_distribution
import torch


def test_spline_integration_analytic_case2():
    # Define a more complex LRA problem with known constraints
    from pal.logic.lra import LinearInequality as LI, And, LRAProblem
    from pal.distribution.spline_distribution import SplineSQ2DBuilder

    # Create a more complex LRA problem: x + y <= 2, x >= 0, y >= 0, x <= 1
    constraint1 = LI({"x": 1.0, "y": 1.0}, "<=", 2.0)  # x + y <= 2
    constraint2 = LI({"x": 1.0}, ">=", 0.0)  # x >= 0
    constraint3 = LI({"y": 1.0}, ">=", 0.0)  # y >= 0
    constraint4 = LI({"x": 1.0}, "<=", 1.0)  # x <= 1

    expression = And(constraint1, constraint2, constraint3, constraint4)

    # Define variable bounds
    variables = {"x": (0.0, 1.0), "y": (0.0, 2.0)}

    # Create the LRAProblem
    constraints = LRAProblem(expression=expression, variables=variables, name="Trapezoid")

    # Define variable positions
    var_positions = {"x": 0, "y": 1}

    # Create a spline distribution builder
    spline_builder = SplineSQ2DBuilder(
        constraints=constraints,
        var_positions=var_positions,
        num_knots=4,  # More complex spline with 4 knots
        num_mixtures=3,  # Three mixture components
    )

    # Integrate the distribution
    integrated_distribution = integrate_distribution(
        d=spline_builder,
        device=torch.device("cpu"),
        precision=torch.float64,
    )

    # Define the analytic integral for the region
    # The region is a trapezoid with vertices (0, 0), (1, 0), (1, 1), (0, 2)
    # The area can be calculated as: 0.5 * (base1 + base2) * height = 0.5 * (1 + 2) * 1 = 1.5
    analytic_integral = 1.5

    # Compute the integral using the spline distribution
    # the first coefficient corresponds to the integral over the constant (^0) term
    computed_integral = integrated_distribution.integral_coeffs[:,:,0].sum()

    # Assert that the computed integral matches the analytic integral
    assert torch.isclose(computed_integral, torch.tensor(analytic_integral), atol=1e-3), \
        f"Computed integral {computed_integral} does not match analytic integral {analytic_integral}"
