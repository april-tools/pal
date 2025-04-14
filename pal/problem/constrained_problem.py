import abc
from typing import Any
from torch.utils.data import Dataset
# from pyxadd.diagram import Diagram
from pal.logic.lra import LRAProblem
from dataclasses import dataclass


@dataclass
class DatasetResult:
    train: Dataset
    val: Dataset
    test: Dataset


class ConstrainedProblem(abc.ABC):
    @abc.abstractmethod
    def load_dataset(self) -> DatasetResult:
        """
        Loads the dataset and returns the training, validation, and test datasets.

        Returns:
            DatasetResult: The training, validation, and test datasets.
        """
        ...

    @abc.abstractmethod
    def create_constraints(self) -> LRAProblem:
        ...

    @abc.abstractmethod
    def get_y_vars(self) -> dict[str, int]:
        """
        Returns the y variable names and their indices.
        """
        ...

    @abc.abstractmethod
    def get_x_shape(self) -> list[int]:
        ...

    @abc.abstractmethod
    def get_name(self) -> str:
        ...

    @abc.abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the problem.
        Used as a unique key for the problem and will be pickled.
        """
        ...
