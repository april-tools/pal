import pal.problem.sdd as csdd
import pal.polynomial.spline_distribution as psd
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