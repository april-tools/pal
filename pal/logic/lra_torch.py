import torch
from lra import LRAProblem, LinearInequality, And, Or, LRA


class PLinearInequality(torch.nn.Module):
    def __init__(self, linear_constraint: LinearInequality, var_dict: dict[str, int],
                 coeff_tensor: torch.Tensor | None = None, indices_tensor: torch.Tensor | None = None):
        super().__init__()
        self.linear_constraint = linear_constraint
        self.var_dict = var_dict
        coeff_tensor = torch.tensor(
            [c for _, c in linear_constraint.lhs], dtype=torch.float32
        )
        self.register_buffer("coeff_tensor", coeff_tensor)
        indices_tensor = torch.tensor(
            [var_dict[var] for var in linear_constraint.lhs], dtype=torch.int64
        )
        self.register_buffer("indices_tensor", indices_tensor)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        res = super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self.var_dict = state_dict[prefix + "var_dict"]
        unexpected_keys.remove(prefix + "var_dict")
        return res

    def state_dict(self, *args, **kwargs):
        st = super().state_dict(*args, **kwargs)
        prefix = kwargs.get("prefix", "")
        st[f"{prefix}var_dict"] = self.var_dict
        return st

    def to(self, arg):
        self.coeff_tensor = self.coeff_tensor.to(arg)
        self.indices_tensor = self.indices_tensor.to(arg)

    def forward(self, x):
        lhs = torch.vmap(lambda b: torch.gather(b, 0, self.indices_tensor))(x)
        lhs = torch.vmap(lambda b: b * self.coeff_tensor)(lhs).sum(dim=1)
        if self.linear_constraint.symbol == "<=":
            return lhs <= self.linear_constraint.rhs
        elif self.linear_constraint.symbol == ">=":
            return lhs >= self.linear_constraint.rhs
        
    def var_map_dict(self) -> dict[str, int]:
        return self.var_dict


class PAnd(torch.nn.Module):
    def __init__(
        self,
        orig: And,
        children: list["PLRA"],
    ):
        super().__init__()
        self.orig = orig
        self.children = torch.nn.ModuleList(children)

    def forward(self, x):
        results = [child(x) for child in self.children]
        results = torch.stack(results, dim=0)
        # via and operation on the first dimension
        return torch.all(results, dim=0)
    
    def var_map_dict(self) -> dict[str, int]:
        return self.children[0].var_map_dict()


class POr(torch.nn.Module):
    def __init__(
        self,
        orig: Or,
        left: "PLRA",
        right: "PLRA",
    ):
        super().__init__()
        self.orig = orig
        self.left = left
        self.right = right

    def forward(self, x):
        return self.left(x) | self.right(x)
    
    def var_map_dict(self) -> dict[str, int]:
        return self.left.var_map_dict()


PLRA = PLinearInequality | PAnd | POr


def lra_to_torch(
    tree: LRA | LRAProblem, var_dict: dict[str, int]
) -> PLRA:
    if isinstance(tree, LinearInequality):
        return PLinearInequality(tree, var_dict)
    elif isinstance(tree, And):
        return PAnd(
            tree,
            lra_to_torch(tree.left, var_dict),
            lra_to_torch(tree.right, var_dict),
        )
    elif isinstance(tree, Or):
        return POr(
            tree,
            lra_to_torch(tree.left, var_dict),
            lra_to_torch(tree.right, var_dict),
        )
    elif isinstance(tree, LRAProblem):
        return lra_to_torch(tree.expression, var_dict)
    else:
        raise ValueError(f"Unknown type {type(tree)}")


def lra_state_dict_to_torch(tree: LRA, state_dict: dict, prefix="") -> PLRA:
    if isinstance(tree, LinearInequality):
        # remove leading dot if present
        if prefix.startswith("."):
            prefix = prefix[1:]
        var_dict = state_dict[f"{prefix}.var_dict"]
        coeff_tensor = state_dict[f"{prefix}.coeff_tensor"]
        indices_tensor = state_dict[f"{prefix}.indices_tensor"]
        return PLinearInequality(tree, var_dict, coeff_tensor=coeff_tensor, indices_tensor=indices_tensor)
    elif isinstance(tree, And):
        return PAnd(
            tree,
            lra_state_dict_to_torch(tree.left, state_dict, prefix=f"{prefix}.left"),
            lra_state_dict_to_torch(tree.right, state_dict, prefix=f"{prefix}.right"),
        )
    elif isinstance(tree, Or):
        return POr(
            tree,
            lra_state_dict_to_torch(tree.left, state_dict, prefix=f"{prefix}.left"),
            lra_state_dict_to_torch(tree.right, state_dict, prefix=f"{prefix}.right"),
        )
    elif isinstance(tree, LinearInequality):
        return lra_state_dict_to_torch(tree.expression, state_dict, prefix)
    else:
        raise ValueError(f"Unknown type {type(tree)}")


def torch_to_lra(tree: PLRA) -> LRA:
    if isinstance(tree, PLinearInequality):
        return tree.linear_constraint
    elif isinstance(tree, PAnd):
        return tree.orig
    elif isinstance(tree, POr):
        return tree.orig
    else:
        raise ValueError(f"Unknown type {type(tree)}")
