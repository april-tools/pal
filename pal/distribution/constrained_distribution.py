from abc import abstractmethod, ABC
from typing import Callable, Generic, overload
import torch
from typing import TypeVar
import pal.logic.lra as lra


class Box:
    def __init__(self, id: tuple[int, ...], constraints: dict[str, tuple[float, float]]):
        self.id = id
        self.constraints = constraints


def neutral_box(vars: list[str]) -> Box:
    """
    Creates a box with no constraints.
    """
    return Box((0,), {var: (-float("inf"), float("inf")) for var in vars})


def box_to_lra(box: Box) -> lra.LRA:
    """
    Converts a box to a LRA.
    """
    constraints = []
    for var, (lb, ub) in box.constraints.items():
        axis_constraints = []
        if lb != -float("inf"):
            axis_constraints.append(lra.LinearInequality({var: 1.0}, ">=", lb))
        if ub != float("inf"):
            axis_constraints.append(lra.LinearInequality({var: 1.0}, "<=", ub))
        if axis_constraints:
            constraints.append(lra.And(*axis_constraints))
    return lra.And(*constraints)


A = TypeVar("A", bound="ConditionalConstraintedDistribution")


class ConstrainedDistributionBuilder(Generic[A], ABC):
    """ "
    A class that represents a constrained distribution that has not been integrated yet."
    """
    _var_positions: dict[str, int]
    _constraints: lra.LRAProblem

    def __init__(self, var_positions: dict[str, int], constraints: lra.LRAProblem):
        """
        Initializes the distribution with the variable positions and constraints.
        """
        self._var_positions = var_positions
        self._constraints = constraints
        # validate the var_positions
        for var in constraints._variables.keys():
            if var not in var_positions:
                raise ValueError(f"Variable {var} not in var_positions")
            
    @property
    def var_positions(self) -> dict[str, int]:
        """
        Returns the variable positions in the tensor.
        """
        return self._var_positions
    
    @property
    def constraints(self) -> lra.LRAProblem:
        """
        Returns the constraints of the distribution.
        """
        return self._constraints

    @property
    @abstractmethod
    def total_degree(self) -> int:
        """
        Returns the total degree of the distribution.
        Maximum in case it's piecewise.
        """

    @abstractmethod
    def enumerate_pieces(
        self,
    ) -> list[tuple[Box, Callable[[torch.Tensor], torch.Tensor], tuple[int, ...]]]:
        """
        Enumerates the pieces of the distribution.
        Each piece is a tuple of a box, a function that maps the input to the output per
        parameter (vectorized) and a shape of the output of the function.

        Let's say the unnormalized density is f(x) = psi_1 * x*2 + psi_2 * x + psi_3 for some unknown parameters psi.
        Then the function will be: f'(x) = [x**2, x, 1]. It is used for integration over the box.
        """

    @abstractmethod
    def get_distribution(self, integrated: dict[tuple[int,...], torch.Tensor]) -> A:
        """
        Returns the distribution by integrating the pieces returned by enumerate_pieces.
        The results are indexed by the box id.
        """


B = TypeVar("B", bound="ConditionalConstraintedDistribution")
C = TypeVar("Args", bound=tuple)


class ConditionalConstraintedDistribution(Generic[B, C], ABC, torch.nn.Module):
    """
    A class that represents a constrained distribution P(Y|psi) on some unknown parameters psi.
    """
    _constraints: lra.LRAProblem

    def __init__(self, constraints: lra.LRAProblem):
        """
        Initializes the distribution with the constraints.
        """
        self._constraints = constraints

    @property
    def constraints(self) -> lra.LRAProblem:
        """
        Returns the constraints of the distribution.
        """
        return self._constraints

    @abstractmethod
    def forward(self, *psi: C) -> B:
        """
        Returns the distribution by giving the parameters psi.
        """

    def __call__(self, *psi: C) -> B:
        return super().__call__(*psi)

    @abstractmethod
    def parameter_shape(self) -> list[tuple[int, ...]]:
        """
        Returns the shape of the parameters psi.
        """


class ConstrainedDistribution(ABC, torch.nn.Module):
    """
    A class that represents a constrained distribution P(Y) over some constraints phi.
    """
    _constraints: lra.LRAProblem

    def __init__(self, constraints: lra.LRAProblem):
        """
        Initializes the distribution with the constraints.
        """
        self._constraints = constraints

    @property
    def constraints(self) -> lra.LRAProblem:
        """
        Returns the constraints of the distribution.
        """
        return self._constraints

    @abstractmethod
    def log_dens(self, x: torch.Tensor, eps: float = -1, with_indicator=False) -> torch.Tensor:
        """
        Returns the log probability of the distribution.
        """

    @abstractmethod
    def enumerate_pieces(
        self,
    ) -> list[tuple[Box, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]]:
        """
        Enumerates the pieces of the distribution.
        Each piece is a tuple of a box, the integral over the box and the log_dens function.
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the distribution.
        """
        return self.log_dens(x)
