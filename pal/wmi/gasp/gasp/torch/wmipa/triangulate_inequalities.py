from typing import Dict
from wmipa.integration.polytope import Polytope
from scipy.optimize import linprog
import numpy as np
from scipy.spatial import HalfspaceIntersection
from scipy.spatial import Delaunay
from pysmt.fnode import FNode


def polytope_to_matrices(p: Polytope, var_map: Dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a Polytope object to matrices A and b such that A * x + b <= 0.

    Args:
        p: The Polytope object.
        var_map: A dictionary mapping variable names to indices.

    Returns:
        tuple[np.ndarray, np.ndarray]: The matrices A and b.
    """
    A = np.zeros((len(p.bounds), len(var_map)))
    b = np.zeros(len(p.bounds))
    for i, bound in enumerate(p.bounds):
        for var, coeff in bound.coefficients.items():
            A[i, var_map[var]] = coeff
        b[i] = bound.constant
    return A, (-1)*b


def pysmt_to_matrices(
        expressions: list[FNode],
        var_map: Dict[str, int],
        aliases: Dict[FNode, FNode] | None = None
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a list of pysmt formulas to matrices A and b such that A * x + b <= 0.
    """
    A = np.zeros((len(expressions), len(var_map)))
    b = np.zeros(len(expressions))
    for i, expression in enumerate(expressions):
        if not (expression.is_le() or expression.is_lt()):
            raise ValueError("All expressions should be inequalities.")
        left, right = expression.args()

        def insert(symbol: FNode, coeff: float):
            if aliases is not None and symbol in aliases:
                symbol = aliases[symbol]
            j = var_map[symbol.symbol_name()]
            A[i, j] += coeff

        def insert_terms(node: FNode, coeff=1):
            if node.is_times():
                t1, t2 = node.args()
                if t1.is_constant():
                    assert t2.is_symbol()
                    insert(t2, coeff * float(t1.constant_value()))
                elif t2.is_constant():
                    assert t1.is_symbol()
                    insert(t1, coeff * float(t2.constant_value()))
            elif node.is_plus():
                for term in node.args():
                    insert_terms(term, coeff)
            elif node.is_symbol():
                insert(node, coeff)
            elif node.is_constant():
                b[i] += coeff*float(node.constant_value())
            else:
                raise ValueError("Invalid expression.")
        
        insert_terms(left)
        insert_terms(right, -1)
    return A, b


def find_interiour_point(A: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """
    Finds an interiour point of the polytope defined by A * x + b <= 0.
    Solved via Chebyshev center of the polyhedron.

    Args:
        A: The matrix A.
        b: The vector b.

    Returns:
        np.ndarray: An interiour point of the polytope.
    """
    # find Chebyshev center of the polyhedron
    # max_{x,r} r
    # s.t. A * x + r * ||A_i|| <= -b
    
    # ||A_i||
    norms = np.reshape(np.linalg.norm(A, axis=1),(-1, 1))

    # [A, ||A_i||]
    weights = np.hstack((A, norms))

    bound = (-1)*b

    costs = np.zeros((A.shape[1]+1, 1))
    costs[-1] = -1

    res = linprog(costs, A_ub=weights, b_ub=bound, bounds=A.shape[1]*[(None, None)] + [(0,None)])
    if not res.success or (res.x[-1] <= 1e-10):
        if res.x[-1] > 0:
            print(f"Warning: Chebyshev center is positive but super tiny! {res.x[-1]}")
        assert res.x[-1] >= -1e-6  # just for safety
        # res.x[-1] == 0.0 means that the interior of polytope is empty, but not contradictory
        # i sometimes get negative values, but they are very close to zero. This should not be possible.
        return None
    else:
        # check it: TODO remove
        assert np.all(np.dot(A, res.x[:-1]) <= -b)
        return res.x[:-1]
    

def find_min_max_inequality(A: np.ndarray, b: np.ndarray) -> tuple:
    """
    Finds the minimum and maximum values of x that satisfy the inequality A * x + b <= 0 for all elements in A and b.

    Args:
        A (np.ndarray): The coefficient vector.
        b (np.ndarray): The constant term vector.

    Returns:
        tuple: A tuple containing the minimum and maximum values of x.
    """
    if A.shape != b.shape:
        raise ValueError("A and b must have the same shape")

    x_bounds = -b / A

    min_x = -np.inf
    max_x = np.inf

    for i in range(len(A)):
        if A[i] > 0:
            max_x = min(max_x, x_bounds[i])
        elif A[i] < 0:
            min_x = max(min_x, x_bounds[i])
        else:
            if b[i] > 0:
                raise ValueError("No solution exists for A[i] = 0 and b[i] > 0")

    return min_x, max_x


def h_rep_to_v_rep(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Converts a polytope defined by H-representation to V-representation.
    H representation: A * x + b <= 0.
    V representation: The convex hull of a set of points.

    Args:
        A: The matrix A.
        b: The vector b.

    Returns:
        np.ndarray: The V-representation of the polytope.
    """
    # find interiour point
    x0 = find_interiour_point(A, b)

    if x0 is None:
        return None
    else:
        halfspaces = np.concatenate((A, b[:, np.newaxis]), axis=1)
        assert (b == halfspaces[:,-1]).all()

        if x0.shape[0] == 1:
            l, u = find_min_max_inequality(A.squeeze(-1), b)
            return np.array([[l], [u]])
        else:
            # find V-representation
            hs = HalfspaceIntersection(halfspaces, x0)
            V = hs.intersections
            return V


def triangulate_v_rep(V: np.ndarray) -> np.ndarray:
    """
    Triangulates the V-representation of a polytope.

    Args:
        V: The V-representation of the polytope.

    Returns:
        np.ndarray: The simplices of the triangulation, shape (nsimplex, ndim+1, ndim).
    """
    if V.shape[1] == 1:
        # in 1d we only have one simplex
        return V[np.newaxis, :, :]
    else:
        tri = Delaunay(V)
        simpl_indices = tri.simplices
        # we return the points!
        simplices = V[simpl_indices.reshape(-1), :].reshape(simpl_indices.shape[0],V.shape[1]+1, V.shape[1])
        return simplices
