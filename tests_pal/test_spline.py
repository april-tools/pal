from pal.logic.lra import LinearInequality as LI, And, LRAProblem
from pal.distribution.spline_distribution import SplineSQ2DBuilder
import torch

def test_spline_knots_values():
    """
    Test that the value at the knots matches the parameterized value.
    """

    # Define simple constraints
    constraint1 = LI({"x": 1.0}, ">=", 0.0)  # x >= 0
    constraint2 = LI({"x": 1.0}, "<=", 1.0)  # x <= 1
    constraint3 = LI({"y": 1.0}, ">=", 0.0)  # y >= 0
    constraint4 = LI({"y": 1.0}, "<=", 1.0)  # y <= 1

    expression = And(constraint1, constraint2, constraint3, constraint4)

    # Define variable bounds
    variables = {"x": (0, 1), "y": (0, 1)}

    # Create the LRAProblem
    constraints = LRAProblem(expression=expression, variables=variables, name="Square")

    # Define variable positions
    var_positions = {"x": 0, "y": 1}

    # Create a spline distribution builder with 1 mixture and 3 knots
    spline_builder = SplineSQ2DBuilder(
        constraints=constraints,
        var_positions=var_positions,
        num_knots=3,
        num_mixtures=1,
    )

    # Create a ConditionalSplineSQ2D distribution directly
    coeffs_2dgrid = torch.ones(
        (spline_builder.knots.shape[0] - 1, spline_builder.knots.shape[0] - 1, 1),
        device=torch.device("cpu"),
    ) / ((spline_builder.num_knots - 1)**2)
    # coeffs_2dgrid = coeffs_2dgrid / coeffs_2dgrid.sum(dim=-1).sum(dim=-1)
    spline_dist = spline_builder.get_distribution({(i, j): coeffs_2dgrid[i, j] for i in range(coeffs_2dgrid.shape[0]) for j in range(coeffs_2dgrid.shape[1])})

    # Define value_params to match the number of knots
    value_params = torch.rand(2, spline_builder.num_knots)
    params_value = value_params.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, num_knots)

    params_mixture_weights = torch.ones((1, 1))

    # Obtain a SplineSQ2D object by calling the forward method
    params_derivative = torch.zeros_like(params_value)
    spline_dist = spline_dist(params_value, params_derivative, params_mixture_weights)

    # set normalizing integral to 1
    integral_sum = spline_dist.integrals_2dgrid.sum(dim=-1).sum(dim=-1)
    spline_dist.integrals_2dgrid = spline_dist.integrals_2dgrid / integral_sum

    # Check that the value at the knots matches the parameterized value
    for i in range(spline_builder.num_knots):
        for j in range(spline_builder.num_knots):
            x = spline_builder.knots[i, 0]
            y = spline_builder.knots[j, 1]
            eval_point = torch.tensor([[x, y]])

            expected = (value_params[0, i] * value_params[1, j])**2

            log_density = spline_dist.log_dens(eval_point)
            density = torch.exp(log_density)
            assert torch.isclose(density, expected, atol=1e-5), \
                f"Density at knot ({x}, {y}) does not match parameterized value."
    print("Test passed: Values at knots match parameterized values.")