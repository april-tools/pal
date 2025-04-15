import pytest
from pal.problem.sdd import SDDSingleImageTrajectory
from pal.problem.constrained_problem import DatasetResult


@pytest.fixture
def sdd_instance():
    return SDDSingleImageTrajectory(
        img_id=1,
        path="./data/sdd",
        window_size=5,
        sampling_rate=70,
        predict_horizon_samples=10,
    )


def test_initialization(sdd_instance: SDDSingleImageTrajectory):
    assert sdd_instance.dataset is not None
    assert sdd_instance.path == "./data/sdd"
    assert sdd_instance.window_size == 5
    assert sdd_instance.sampling_rate == 70
    assert sdd_instance.predict_horizon_samples == 10


def test_load_dataset(sdd_instance: SDDSingleImageTrajectory):
    dataset_result = sdd_instance.load_dataset()
    assert isinstance(dataset_result, DatasetResult)
    assert dataset_result.train is not None
    assert dataset_result.val is not None
    assert dataset_result.test is not None


def test_create_constraints(sdd_instance: SDDSingleImageTrajectory):
    lra_problem = sdd_instance.create_constraints()
    assert lra_problem is not None
    assert hasattr(lra_problem, "expression")
    assert hasattr(lra_problem, "_variables")


def test_get_y_vars(sdd_instance: SDDSingleImageTrajectory):
    y_vars = sdd_instance.get_y_vars()
    assert y_vars == {"yw": 0, "yh": 1}


def test_get_x_shape(sdd_instance: SDDSingleImageTrajectory):
    x_shape = sdd_instance.get_x_shape()
    assert x_shape == [10]  # window_size * 2