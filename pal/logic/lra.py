

from logging import warning
from typing import Callable


class LinearInequality:
    inequality_pattern = r"(.*)(<=?|>=?)(.*)"

    def __init__(
            self,
            lhs: dict[str, float],
            symbol: str,
            rhs: float,
            string_repr: str | None = None,
    ):
        self.lhs = lhs
        assert symbol in ["<=", ">="]
        self.symbol = symbol
        self.rhs = rhs
        self.string_repr = string_repr

    @classmethod
    def empty(cls) -> "LinearInequality":
        """
        Creates an empty LinearInequality instance.
        It's always true that {} <= 0.

        Returns:
            LinearInequality: An empty LinearInequality instance.
        """
        return cls({}, "<=", 0.0)

    def is_empty(self) -> bool:
        """
        Checks if the LinearInequality instance is empty.
        An empty LinearInequality is defined as having no left-hand side variables.
        Returns:
            bool: True if the LinearInequality is empty, False otherwise.
        """
        return len(self.lhs) == 0

    def __str__(self) -> str:
        """
        Returns a string representation of the constraint.

        If a custom string representation is set, it will be returned.
        Otherwise, it constructs the string representation based on the
        left-hand side (lhs), symbol, and right-hand side (rhs) of the
        inequality.

        Returns:
            str: The string representation of the constraint.
        """
        if self.string_repr is not None:
            return self.string_repr
        lhs = ''
        for var, coeff in self.lhs.items():
            pos = coeff >= 0
            if lhs == "" and pos:
                lhs += f"{coeff}*{var}"
            elif pos:
                lhs += f" + {coeff}*{var}"
            else:
                lhs += f" - {-coeff}*{var}"
        return f"{lhs} {self.symbol} {self.rhs}"

    def __and__(self, other: "LRA"):
        """
        Combines two LinearInequality objects using a logical AND operation.

        Args:
            other (LinearInequality): The other LinearInequality object to combine with.

        Returns:
            AndTree: An instance of the AndTree class representing the combined constraint.
        """
        if self.is_empty():
            return other
        if other.is_empty():
            return self
        return And(self, other)

    def __or__(self, other: "LRA"):
        """
        Combines two LinearInequality objects using a logical OR operation.

        Args:
            other (LinearInequality): The other LinearInequality object to combine with.

        Returns:
            OrTree: An instance of the OrTree class representing the combined constraint.

        Raises:
            ValueError: If either of the LinearInequality objects is empty.
        """
        if self.is_empty() or other.is_empty():
            raise ValueError("Cannot combine empty LinearInequality objects using OR.")
        return Or(self, other)


LI = LinearInequality


class And:
    def __init__(self, *children: "LRA"):
        assert len(children) > 0
        self.children = children

    def __and__(self, other: "LRA"):
        if isinstance(other, And):
            return And(*(self.children + other.children))
        else:
            return And(*(self.children + [other]))

    def __str__(self) -> str:
        return "(" + " & ".join([str(child) for child in self.children]) + ")"


class Or:
    def __init__(self, *children: "LRA"):
        assert len(children) > 0
        self.children = children

    def __str__(self) -> str:
        return "(" + " | ".join([str(child) for child in self.children]) + ")"


LRA = LinearInequality | And | Or


def gather_variables(expr: LRA) -> set[str]:
    if isinstance(expr, LinearInequality):
        return set(expr.lhs.keys())
    elif isinstance(expr, And):
        gathered_vars = [gather_variables(child) for child in expr.children]
        return set().union(*gathered_vars)
    elif isinstance(expr, Or):
        gathered_vars = [gather_variables(child) for child in expr.children]
        return set().union(*gathered_vars)
    else:
        raise ValueError(f"Unknown type {type(expr)}")
    

class LRAProblem:
    """
    A class representing a Linear Rational Arithmetic (LRA) problem.
    Holds additional metadata about the problem, such as the variable limits.
    """
    _expression: LRA
    _variables: dict[str, tuple[float, float]] | list[str]  # var_name -> (lower, upper)

    def __init__(
            self,
            expression: LRA,
            variables: dict[str, tuple[float, float]] | list[str] | None
    ):
        self._expression = expression
        if variables is None:
            self._variables = gather_variables(expression)
        else:
            self._variables = variables
            mentioned_vars = gather_variables(expression)
            for var in mentioned_vars:
                if var not in self._variables:
                    raise ValueError(
                        f"Variable {var} is mentioned in the expression but not in the variables list."
                    )
                
    @property
    def expression(self) -> LRA:
        # check if self._variables is empty dict
        if not self._variables:
            warning("No bounds for the variables defined in the constraint")
            # all empty
            return self._expression
        elif isinstance(self._variables, dict):
            if self._expression is not None:
                variables = gather_variables(self._expression)
                assert variables.issubset(set(self._variables.keys()))

            def to_logic(var: str, lb: float, ub: float):
                lower = LinearInequality({var: 1.0}, ">=", lb)
                upper = LinearInequality({var: 1.0}, "<=", ub)
                return And(lower, upper)

            bounds = [to_logic(var, lb, ub) for var, (lb, ub) in self._variables.items()]

            bound = And(*bounds)

            if self._expression is None:
                return bound
            else:
                return And(self._expression, bound)
        else:
            # it's a list of variables
            return self._expression
                
    def map_constraints(self, f: Callable[[LinearInequality], LinearInequality]) -> "LRAProblem":
        """
        Applies a function `f` to each `LinearConstraint` in the expression tree while keeping
        the general structure.

        Args:
            f (Callable[[LinearConstraint], LinearConstraint]): The function to apply to each
            `LinearConstraint`.

        Returns:
            LinearIneqLogicTree: A new expression tree with the modified `LinearConstraint` objects.

        """

        def recurse_expression(expr: LRA) -> LRA:
            if isinstance(expr, LinearInequality):
                return f(expr)
            elif isinstance(expr, And):
                mapped_children = [
                    recurse_expression(child) for child in expr.children
                ]
                return And(
                    *mapped_children
                )
            elif isinstance(expr, Or):
                mapped_children = [
                    recurse_expression(child) for child in expr.children
                ]
                return Or(
                    *mapped_children
                )

        if self.expression is None:
            return LRAProblem(None, self._variables)
        else:
            expr = recurse_expression(self._expression)
            variables = gather_variables(expr)
            if isinstance(self._variables, dict):
                sub_vars = {var: self._variables[var] for var in variables}
            else:
                sub_vars = [var for var in self._variables if var in variables]
            return LRAProblem(expr, sub_vars)
        
    def get_global_limits(self) -> dict[str, tuple[float, float]]:
        """
        Returns the global limits.

        Returns:
            dict[str, tuple[float, float]]: The global limits.
        """
        if isinstance(self._variables, dict):
            return self._variables
        else:
            raise ValueError(
                "Global limits are not available when variables are provided as a list."
            )
        
    def __repr__(self) -> str:
        return f"LRAProblem({str(self.expression)}, {self._variables})"