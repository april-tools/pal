import pal.problem.sdd as csdd
import pal.polynomial.spline_distribution as psd

def test_integration_pipeline():
    sdd = csdd.SDDSingleImageTrajectory(
        img_id=12,
        sdd_data_path="./data/sdd",
    )

    # load the constraints
    lra_problem = sdd.create_constraints()