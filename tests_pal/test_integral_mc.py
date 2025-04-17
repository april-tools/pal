from pal.logic.lra import LinearInequality as LI, And, LRAProblem
from pal.distribution.spline_distribution import SplineSQ2DBuilder
from pal.wmi.compute_integral import integrate_distribution
from pal.logic.lra_torch import lra_to_torch
import torch

def test_spline_integration_mc_pipeline():
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Define a more complex LRA problem with known constraints

    # Create a more complex LRA problem: x + y <= 2, x >= 0, y >= 0, x <= 1
    constraint1 = LI({"x": 1.0, "y": 1.0}, "<=", 2.0)  # x + y <= 2
    constraint2 = LI({"x": 1.0}, ">=", 0.0)  # x >= 0
    constraint3 = LI({"y": 1.0}, ">=", 0.0)  # y >= 0
    constraint4 = LI({"x": 1.0}, "<=", 1.0)  # x <= 1

    expression = And(constraint1, constraint2, constraint3, constraint4)

    # Define variable bounds
    variables = {"x": (0, 1), "y": (0, 2)}

    # Create the LRAProblem
    constraints = LRAProblem(expression=expression, variables=variables, name="Trapezoid")

    # Define variable positions
    var_positions = {"x": 0, "y": 1}

    # Create a spline distribution builder
    spline_builder = SplineSQ2DBuilder(
        constraints=constraints,
        var_positions=var_positions,
        num_knots=3,  # More complex spline with 4 knots
        num_mixtures=1,  # Three mixture components
    )

    # Integrate the distribution
    integrated_distribution = integrate_distribution(
        d=spline_builder,
        device=torch.device("cpu"),
        precision=torch.float64,
    )

    # Compile the constraints to test support
    torch_constraints = lra_to_torch(constraints, var_positions)

    # Monte Carlo integration via rejection sampling
    num_samples = 5_000_000
    samples = []
    while sum(s.shape[0] for s in samples) < num_samples:
        # Sample uniformly from the bounding box [0, 1] x [0, 2]
        candidate_samples = torch.rand(100_000, 2)
        candidate_samples[:, 1] *= 2  # Scale y to [0, 2]

        # Apply rejection criterion using compiled constraints
        valid_samples = candidate_samples[torch_constraints(candidate_samples)]
        samples.append(valid_samples)

    samples = torch.cat(samples, dim=0)

    # Randomly initialize parameters for the spline distribution
    random_parameter = [torch.rand(1, *s) for s in integrated_distribution.parameter_shape()]
    random_parameter[0] = 10 * random_parameter[0]  # Scale the values to be larger than the derivatives
    # Softmax over mixture weights
    random_parameter[-1] = torch.nn.functional.softmax(random_parameter[-1], dim=-1)

    # Create the spline distribution
    spline_dist = integrated_distribution(*random_parameter)

    # Evaluate the spline distribution's density at the sampled points
    log_densities = spline_dist.log_dens(samples)
    densities = torch.exp(log_densities)

    area = integrated_distribution.integral_coeffs[:, :, 0].sum()

    assert torch.isclose(area, torch.tensor(1.5), atol=1e-2), \
        f"Area {area} is not close to the expected value 1.5"

    # Compute the Monte Carlo estimate of the integral
    mc_integral = densities.mean() * area  # Multiply by the area of the bounding box

    # Assert that the Monte Carlo estimate converges to 1
    assert torch.isclose(mc_integral, torch.tensor(1.0), atol=1e-2), \
        f"Monte Carlo integral {mc_integral} does not converge to 1"