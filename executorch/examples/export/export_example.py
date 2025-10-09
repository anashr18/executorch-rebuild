from __future__ import annotations

import argparse
from pathlib import Path

from executorch.examples.models import get_model_spec, list_available_models
from executorch.exir import capture


def parse_args() -> argparse.Namespace:
    available = list_available_models()
    parser = argparse.ArgumentParser(
        description="Capture a PyTorch nn.Module into an ExecuTorch flatbuffer."
    )
    parser.add_argument(
        "--model_name",
        required=True,
        choices=available,
        help="Model identifier from examples.models.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run capture on (default: cpu).",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Optional output path for the generated .ff file.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    spec = get_model_spec(args.model_name, device=args.device)
    captured = capture(
        spec.module,
        spec.example_inputs,
        spec.example_kwargs,
        device=args.device,
    )
    program = captured.to_edge().to_executorch()

    output = Path(args.output_path or f"{args.model_name}.ff")
    program.save(output)
    print(f"Wrote ExecuTorch program to {output.resolve()}")


if __name__ == "__main__":
    main()
