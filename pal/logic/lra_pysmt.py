import lra
from pysmt.shortcuts import get_env, Symbol, Real, Plus, And, Or
import pysmt.typing as stypes
from pysmt.fnode import FNode


def translate_to_pysmt(
    constraint: lra.LRAProblem | lra.LRA,
) -> tuple[FNode, dict[str, FNode]]:

    # need this for Plus(*[])
    get_env().enable_infix_notation = True

    # TODO check cache with the cachedXADDWalker!
    symb_cache: dict[str, FNode] = {}

    def get_symb(sym: str):
        if sym not in symb_cache:
            symb_cache[sym] = Symbol(sym, typename=stypes.REAL)
        return symb_cache[sym]

    # This should be equivalent to ToXaddWalker using bool mode
    # important for future bool additions!
    def recursive_translate(
        node: lra.LRA,
    ) -> FNode:
        if isinstance(node, lra.LinearInequality):
            # canonical is ax + by + cz + ... < d
            left_sum = Plus(
                *[
                    Real(lhs_coeff) * get_symb(lhs_var)
                    for lhs_coeff, lhs_var in node.lhs.items()
                ]
            )
            if node.symbol == "<=":
                return left_sum <= Real(node.rhs)
            elif node.symbol == ">=":
                return left_sum >= Real(node.rhs)
            else:
                raise NotImplementedError()
        elif isinstance(node, lra.And):
            result_left = recursive_translate(node.left)
            result_right = recursive_translate(node.right)
            return And(result_left, result_right)
        elif isinstance(node, lra.Or):
            result_left = recursive_translate(node.left)
            result_right = recursive_translate(node.right)
            return Or(result_left, result_right)

    if isinstance(constraint, lra.LRAProblem):
        expr = constraint.expression
        assert expr is not None
        return recursive_translate(expr), symb_cache
    else:
        return recursive_translate(constraint), symb_cache
