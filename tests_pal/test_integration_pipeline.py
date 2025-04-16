import pal.problem.sdd as csdd
import pal.distribution.spline_distribution as psd
from pal.wmi.compute_integral import integrate_distribution
import torch


def test_integration_pipeline():
    sdd = csdd.SDDSingleImageTrajectory(
        img_id=12,
        path="./data/sdd",
    )

    # load the constraints
    lra_problem = sdd.create_constraints()

    spline_distribution_builder = psd.SplineSQ2DBuilder(
        constraints=lra_problem,
        var_positions=sdd.get_y_vars(),
        num_knots=3,
        num_mixtures=5,
    )

    # create the distribution
    conditional_spline_dist = integrate_distribution(
        d=spline_distribution_builder,
        device=torch.device("cpu"),
        precision=torch.float64,
    )

    random_parameter = [torch.rand(1, *s) for s in conditional_spline_dist.parameter_shape()]

    spline_dist = conditional_spline_dist(*random_parameter)
    assert spline_dist is not None

    random_points = torch.rand(10, 2)
    spline_log_dens = spline_dist.log_dens(random_points, 1e-8)