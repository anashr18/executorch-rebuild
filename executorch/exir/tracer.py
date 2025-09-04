# executorch/exir/tracer.py

from __future__ import annotations
from contextlib import contextmanager
from typing import Any, cast

import torch
import torch.fx as fx
import torch.fx.node as fx_node
# from torch.fx import node 

# --- Global tracer singleton used by PythonTensor.__torch_dispatch__ ---
TRACER: "DispatchTracer | None" = None


class PythonTensor(torch.Tensor):
    """Tiny Tensor subclass that carries an FX Proxy in `.proxy` and
    retains a direct handle to the underlying base tensor in `.elem` to
    safely bypass subclass dispatch when running eager ops.
    """
    __slots__ = ["proxy", "elem"]
    # Provide a typed wrapper for disabling __torch_function__ so static
    # checkers see the expected classmethod signature.
    @classmethod
    def __torch_function__(
        cls,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        if kwargs is None:
            kwargs = {}
        return torch._C._disabled_torch_function_impl(func, types, args, kwargs)

    @staticmethod
    def __new__(cls, elem: torch.Tensor, proxy: fx.Proxy) -> "PythonTensor":
        # Some torch versions only accept positional 'requires_grad'
        r = torch.Tensor._make_subclass(cls, elem, elem.requires_grad)
        # Keep a reference to the original base tensor to avoid calling
        # Tensor.as_subclass during dispatch (which can recurse).
        r.proxy = proxy
        r.elem = elem
        return r

    def __repr__(self, *, tensor_contents=None):
        return f"PythonTensor({super().__repr__(tensor_contents=tensor_contents)})"

    @classmethod
    def __torch_dispatch__(cls, func_overload, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # --- 1) Run the real op eagerly WITHOUT our subclass participating ---
        # Strip our subclass to base torch.Tensor to avoid re-entry.
        def _strip(x):
            # Avoid as_subclass() here; it can trigger __torch_dispatch__ again
            # for some PyTorch versions. Use the stored base tensor instead.
            return x.elem if isinstance(x, PythonTensor) else x

        eager_args = fx_node.map_aggregate(args, _strip)
        eager_kwargs = fx_node.map_aggregate(kwargs, _strip)
        out = func_overload(*eager_args, **eager_kwargs)

        # --- 2) If tracing, create a Proxy node and rewrap tensor outputs ---
        if TRACER is not None:
            tracer = cast("DispatchTracer", TRACER)
            def to_proxy(x):
                if isinstance(x, PythonTensor):
                    return x.proxy
                elif isinstance(x, torch.Tensor):
                    return tracer.create_arg(x)
                else:
                    return x

            proxy_args = fx_node.map_aggregate(args, to_proxy)
            proxy_kwargs = fx_node.map_aggregate(kwargs, to_proxy)

            node = tracer.create_proxy(
                kind="call_function",
                target=func_overload,
                args=proxy_args,
                kwargs=proxy_kwargs,
            )

            def rewrap(t):
                return PythonTensor(t, node) if isinstance(t, torch.Tensor) else t

            return fx_node.map_aggregate(out, rewrap)

        # --- 3) No tracing: just return eager result ---
        return out


class DispatchTracer(fx.Tracer):
    """Tracer that unwraps PythonTensor -> Proxy before FX's default checks."""

    def __init__(self):
        super().__init__()
        # Ensure a graph exists across torch versions
        if not hasattr(self, "graph") or self.graph is None:
            self.graph = fx.Graph()

    def create_arg(self, a: Any):
        # 1) Prefer explicit narrowing to our subclass for type safety
        if isinstance(a, PythonTensor):
            return a.proxy

        # 1b) Duck-typed fallback: other Tensor subclasses that carry `.proxy`
        if isinstance(a, torch.Tensor) and getattr(a, "proxy", None) is not None:
            return cast(PythonTensor, a).proxy

        # 2) Handle aggregates so elements are unwrapped via THIS method
        if isinstance(a, (list, tuple, set)):
            return type(a)(self.create_arg(x) for x in a)
        if isinstance(a, dict):
            return {k: self.create_arg(v) for k, v in a.items()}

        # 3) Default behavior
        return super().create_arg(a)

    def placeholder_tensor(self, name: str, example: torch.Tensor) -> PythonTensor:
        # Create a placeholder node directly on our graph
        n = self.graph.placeholder(name)
        p = fx.Proxy(n, self)
        return PythonTensor(example, p)

def dispatch_trace(
    module: torch.nn.Module,
    example_args: tuple | list = (),
    example_kwargs: dict | None = None,
    input_names: list[str] | None = None,
) -> fx.GraphModule:
    """Trace a module by executing it with PythonTensor placeholders.

    - module: the PyTorch module to trace
    - example_args: positional example tensors matching forward signature
    - example_kwargs: keyword example tensors matching forward signature
    - input_names: optional names for positional inputs; defaults to arg0, arg1, ...

    Returns an `fx.GraphModule` with the captured graph.
    """
    if example_kwargs is None:
        example_kwargs = {}

    if not isinstance(example_args, (list, tuple)):
        raise TypeError("example_args must be a tuple or list of tensors")

    tracer = DispatchTracer()

    with tracing(tracer):
        # Create placeholders for positional args
        pt_args = []
        for i, ex in enumerate(example_args):
            name = input_names[i] if input_names and i < len(input_names) else f"arg{i}"
            if not isinstance(ex, torch.Tensor):
                raise TypeError(f"example_args[{i}] must be a torch.Tensor, got {type(ex)!r}")
            pt_args.append(tracer.placeholder_tensor(name, ex))

        # Create placeholders for keyword args
        pt_kwargs = {}
        for k, ex in example_kwargs.items():
            if not isinstance(ex, torch.Tensor):
                raise TypeError(f"example_kwargs['{k}'] must be a torch.Tensor, got {type(ex)!r}")
            pt_kwargs[k] = tracer.placeholder_tensor(str(k), ex)

        # Execute the module to record operations
        _ = module(*pt_args, **pt_kwargs)

    return fx.GraphModule(module, tracer.graph)

@contextmanager
def tracing(tracer: DispatchTracer):
    global TRACER
    prev = TRACER
    TRACER = tracer
    try:
        yield
    finally:
        TRACER = prev


# --- Smoke test ---
if __name__ == "__main__":
    class M(torch.nn.Module):
        def forward(self, x, y):
            z = x + y
            return (z * 2).relu()

    mod = M()
    ex = torch.randn(2, 3)
    ey = torch.randn(2, 3)

    tracer = DispatchTracer()

    with tracing(tracer):
        x_pt = tracer.placeholder_tensor("x", ex)
        y_pt = tracer.placeholder_tensor("y", ey)
        _ = mod(x_pt, y_pt)  # executes eagerly, records proxies

    gm = fx.GraphModule(mod, tracer.graph)
    print(gm.graph)  # expect a clean call_function graph
