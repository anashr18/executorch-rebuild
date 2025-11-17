from __future__ import annotations

import operator
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.fx as fx
import torch.nn.functional as F


class GraphExportError(RuntimeError):
    """Raised when the exporter cannot translate a module into ExecuTorch graph form."""


_FUNCTION_OP_MAP = {
    operator.add: "aten::add",
    torch.add: "aten::add",
    F.relu: "aten::relu",
    torch.relu: "aten::relu",
    F.relu6: "aten::hardtanh",
    F.hardswish: "aten::hardswish",
    torch.flatten: "aten::flatten",
    F.adaptive_avg_pool2d: "aten::adaptive_avg_pool2d",
}

_METHOD_OP_MAP = {
    "relu": "aten::relu",
    "relu_": "aten::relu",
    "flatten": "aten::flatten",
}

_SUPPORTED_MODULES = (
    torch.nn.Linear,
    torch.nn.Conv2d,
    torch.nn.BatchNorm2d,
    torch.nn.ReLU,
    torch.nn.ReLU6,
    torch.nn.Hardswish,
    torch.nn.Flatten,
    torch.nn.AdaptiveAvgPool2d,
    torch.nn.Dropout,
)

_LITERAL_ARG_TYPES = (int, float, bool, str, torch.dtype)


def build_execution_graph(module: torch.nn.Module) -> Dict[str, Any]:
    traced = fx.symbolic_trace(module)
    instructions: List[Dict[str, Any]] = []
    input_names: List[str] = []
    output_names: List[str] = []

    for node in traced.graph.nodes:
        if node.op == "placeholder":
            input_names.append(node.name)
        elif node.op == "call_function":
            op_name = _map_function(node.target)
            inputs = _flatten_args(node.args)
            attributes = _build_function_attributes(op_name, node.args, node.kwargs)
            instructions.append(
                {
                    "name": node.name,
                    "op": op_name,
                    "inputs": inputs,
                    "outputs": [node.name],
                    "attributes": attributes,
                }
            )
        elif node.op == "call_module":
            submodule = traced.get_submodule(node.target)
            op_name = _map_module(submodule)
            inputs = _flatten_args(node.args)
            attributes = _build_module_attributes(node.target, submodule)
            instructions.append(
                {
                    "name": node.name,
                    "op": op_name,
                    "inputs": inputs,
                    "outputs": [node.name],
                    "attributes": attributes,
                }
            )
        elif node.op == "call_method":
            op_name = _map_method(node.target)
            inputs = _flatten_args(node.args)
            attributes = _build_method_attributes(op_name, node.args, node.kwargs)
            instructions.append(
                {
                    "name": node.name,
                    "op": op_name,
                    "inputs": inputs,
                    "outputs": [node.name],
                    "attributes": attributes,
                }
            )
        elif node.op == "output":
            output_names = _flatten_outputs(node.args[0])
        else:
            raise GraphExportError(f"Unsupported FX node type: {node.op}")

    if not output_names:
        raise GraphExportError("No outputs discovered while tracing module")

    return {
        "inputs": input_names,
        "outputs": output_names,
        "instructions": instructions,
    }


def _map_function(target: Any) -> str:
    try:
        return _FUNCTION_OP_MAP[target]
    except KeyError as exc:
        raise GraphExportError(f"Unsupported call_function target: {target}") from exc


def _map_method(name: str) -> str:
    try:
        return _METHOD_OP_MAP[name]
    except KeyError as exc:
        raise GraphExportError(f"Unsupported call_method target: {name}") from exc


def _map_module(module: torch.nn.Module) -> str:
    if isinstance(module, _SUPPORTED_MODULES):
        if isinstance(module, torch.nn.ReLU6):
            return "aten::hardtanh"
        if isinstance(module, torch.nn.Hardswish):
            return "aten::hardswish"
        return {
            torch.nn.Linear: "aten::linear",
            torch.nn.Conv2d: "aten::conv2d",
            torch.nn.BatchNorm2d: "aten::batch_norm",
            torch.nn.ReLU: "aten::relu",
            torch.nn.Flatten: "aten::flatten",
            torch.nn.AdaptiveAvgPool2d: "aten::adaptive_avg_pool2d",
            torch.nn.Dropout: "aten::dropout",
        }.get(type(module), "aten::relu")
    raise GraphExportError(f"Unsupported call_module target: {module.__class__.__name__}")


def _flatten_args(args: Any) -> List[str]:
    flat: List[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, fx.Node):
            flat.append(value.name)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif value is None or isinstance(value, _LITERAL_ARG_TYPES):
            return
        else:
            raise GraphExportError(f"Unsupported argument type: {type(value)}")

    visit(args)
    return flat


def _flatten_outputs(value: Any) -> List[str]:
    if isinstance(value, fx.Node):
        return [value.name]
    if isinstance(value, (tuple, list)):
        names: List[str] = []
        for item in value:
            names.extend(_flatten_outputs(item))
        return names
    raise GraphExportError(f"Unsupported output value: {type(value)}")


def _attrs(numeric: Dict[str, float] | None = None, string: Dict[str, str] | None = None) -> Dict[str, Dict[str, Any]]:
    return {
        "numeric": dict(numeric or {}),
        "string": dict(string or {}),
    }


def _build_function_attributes(op_name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    numeric: Dict[str, float] = {}
    string: Dict[str, str] = {}

    if op_name == "aten::add":
        alpha = kwargs.get("alpha", 1.0)
        if isinstance(alpha, fx.Node):
            raise GraphExportError("aten::add alpha argument must be a literal")
        numeric["alpha"] = float(alpha)

    elif op_name == "aten::hardtanh":
        numeric["min_val"] = float(kwargs.get("min", kwargs.get("min_val", 0.0)))
        numeric["max_val"] = float(kwargs.get("max", kwargs.get("max_val", 6.0)))

    elif op_name == "aten::flatten":
        start_dim = kwargs.get("start_dim", args[1] if len(args) > 1 else 0)
        end_dim = kwargs.get("end_dim", args[2] if len(args) > 2 else -1)
        if isinstance(start_dim, fx.Node) or isinstance(end_dim, fx.Node):
            raise GraphExportError("flatten dims must be literal")
        numeric["start_dim"] = float(start_dim)
        numeric["end_dim"] = float(end_dim)

    elif op_name == "aten::adaptive_avg_pool2d":
        output_size = kwargs.get("output_size", args[1] if len(args) > 1 else None)
        size = _normalize_output_size(output_size)
        numeric["output_h"] = float(size[0])
        numeric["output_w"] = float(size[1])

    return _attrs(numeric, string)


def _build_method_attributes(op_name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    numeric: Dict[str, float] = {}
    if op_name == "aten::flatten":
        start_dim = kwargs.get("start_dim", args[1] if len(args) > 1 else 0)
        end_dim = kwargs.get("end_dim", args[2] if len(args) > 2 else -1)
        numeric["start_dim"] = float(start_dim)
        numeric["end_dim"] = float(end_dim)
    return _attrs(numeric, {})


def _build_module_attributes(target: str, module: torch.nn.Module) -> Dict[str, Dict[str, Any]]:
    numeric: Dict[str, float] = {}
    string: Dict[str, str] = {}

    if isinstance(module, torch.nn.Linear):
        string["weight"] = f"{target}.weight"
        if module.bias is not None:
            string["bias"] = f"{target}.bias"

    elif isinstance(module, torch.nn.Conv2d):
        string["weight"] = f"{target}.weight"
        if module.bias is not None:
            string["bias"] = f"{target}.bias"
        stride_h, stride_w = module.stride
        pad_h, pad_w = module.padding
        dil_h, dil_w = module.dilation
        numeric.update(
            stride_h=float(stride_h),
            stride_w=float(stride_w),
            padding_h=float(pad_h),
            padding_w=float(pad_w),
            dilation_h=float(dil_h),
            dilation_w=float(dil_w),
            groups=float(module.groups),
        )

    elif isinstance(module, torch.nn.BatchNorm2d):
        string["weight"] = f"{target}.weight"
        string["bias"] = f"{target}.bias"
        string["running_mean"] = f"{target}.running_mean"
        string["running_var"] = f"{target}.running_var"
        numeric["eps"] = float(module.eps)

    elif isinstance(module, torch.nn.Flatten):
        numeric["start_dim"] = float(module.start_dim)
        numeric["end_dim"] = float(module.end_dim)

    elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
        size = _normalize_output_size(module.output_size)
        numeric["output_h"] = float(size[0])
        numeric["output_w"] = float(size[1])

    elif isinstance(module, torch.nn.ReLU6):
        numeric["min_val"] = 0.0
        numeric["max_val"] = 6.0

    elif isinstance(module, torch.nn.Dropout):
        numeric["p"] = float(module.p)
        numeric["training"] = 1.0 if module.training else 0.0
        numeric["inplace"] = 1.0 if module.inplace else 0.0

    return _attrs(numeric, string)


def _normalize_output_size(value: Any) -> Tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, Sequence) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise GraphExportError(f"Unsupported adaptive output size: {value}")
