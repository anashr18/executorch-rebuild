from typing import Optional, Set, Union
import sympy
import torch
import torch.fx.experimental
import torch.fx.experimental.symbolic_shapes
def eval_expr(symint: Union[int, torch.SymInt]) -> Optional[int]:
    if isinstance(symint, int):
        return symint
    try:
        return int(symint.node.shape_env.size_hint(symint.node.expr))
    except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
        return None
    
def eval_shape(shape):
    return [eval_expr(s) for s in shape]

def collect_free_symbols(shape) -> Set[sympy.Symbol]:
    symset = set()
    for sz in shape:
        if isinstance(sz, torch.SymInt):
            symset.update(sz.node.expr.free_symbols)
    return symset