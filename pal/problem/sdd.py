import sdd.constrained_sdd as csdd

from pal.problem.constrained_problem import ConstrainedProblem, DatasetResult

import pal.logic.lra as lra
import os


def logic_translate_compatiblity(
    idx_to_label: dict[int, str],
    dnf: csdd.DNF,
) -> lra.LRA:
    # we need to negate, as these constraints are fore the obstacles
    # not the valid space
    outside_polygon = []
    for polytope_h in dnf.polytopes:
        not_in_obstacle = []
        A = polytope_h.A
        b = polytope_h.b
        for i in range(A.shape[0]):
            # not (Ax <= b)
            # <=> exists i: x^T A[i] > b[i]
            constraint = lra.LI(
                lhs={idx_to_label[j]: A[i, j].item() for j in range(A.shape[1])},
                symbol=">=",
                rhs=b[i].item(),
            )
            not_in_obstacle.append(constraint)
        not_in_obstacle = lra.Or(*not_in_obstacle)
        outside_polygon.append(not_in_obstacle)
    return lra.And(*outside_polygon)


class SDDSingleImageTrajectory(ConstrainedProblem):
    """ "
    A class that representes the trajectory-prediction task for a single image for the SDD dataset.
    """

    def __init__(
        self,
        img_id: int,  # 12 is the default image id
        path: str = "./data/sdd",
        window_size: int = 5,  # how many points are used for the moving-window we condition on
        sampling_rate: int = 70,  # the sampling rate for these points
        predict_horizon_samples: int = 10,  # only used for validation/test as we sample during training
    ):
        if not os.path.exists(path):
            # Ensure the parent directory exists
            if os.path.exists(os.path.dirname(path)):
                os.makedirs(path, exist_ok=True)
            else:
                raise FileNotFoundError(
                    f"Parent directory does not exist: {os.path.dirname(path)}"
                )
        self.dataset = csdd.ConstrainedStanfordDroneDataset(
            img_id=img_id,
            sdd_data_path=path,
        )
        self.path = path
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.predict_horizon_samples = predict_horizon_samples

    def load_dataset(self):
        train, val, test = self.dataset.get_trajectory_prediction_dataset(
            window_size=self.window_size,
            sampling_rate=self.sampling_rate,
            predict_horizon_samples=self.predict_horizon_samples,
        )
        return DatasetResult(train, val, test)

    def create_constraints(self):
        constraints = self.dataset.get_ineqs(do_rescale=True)
        image = self.dataset.get_image()

        bounds = {
            "yw": (0, image.shape[1] * self.dataset.scale),
            "yh": (0, image.shape[0] * self.dataset.scale),
        }
        idx_to_label = {i: name for name, i in self.get_y_vars().items()}
        lra_constraints = logic_translate_compatiblity(idx_to_label, constraints)

        name = f"SDD({self.dataset.img_id})"
        lra_problem = lra.LRAProblem(expression=lra_constraints, variables=bounds, name=name)

        return lra_problem

    def get_y_vars(self):
        return {
            "yw": 0,
            "yh": 1,
        }

    def get_x_shape(self):
        return [self.window_size * 2]

    def get_name(self):
        return f"Constrained_SDD_{self.dataset.img_id}"

    def get_config(self):
        return {
            "img_id": self.dataset.img_id,
            "window_size": self.window_size,
            "sampling_rate": self.sampling_rate,
            "predict_horizon_samples": self.predict_horizon_samples,
        }
