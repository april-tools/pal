import sdd.constrained_sdd as csdd

from pal.problem.constrained_problem import ConstrainedProblem, DatasetResult


class SDDSingleImageTrajectory(ConstrainedProblem):
    """"
    A class that representes the trajectory-prediction task for a single image for the SDD dataset.
    """

    def __init__(self, img_id: int, path: str = "./data/sdd"):
        self.dataset = csdd.ConstrainedStanfordDroneDataset(
            img_id=img_id,
            sdd_data_path=path,
        )
        self.path = path

    def load_dataset(self):
        train, val, test = self.dataset.get_trajectory_prediction_dataset()
        return DatasetResult(train, val, test)

    def create_constraints(self):
        constraints_obstacles = self.dataset.get_ineqs()

        

    def get_y_vars(self):
        raise NotImplementedError

    def get_x_shape(self):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError

    def get_config(self):
        raise NotImplementedError
