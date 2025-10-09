# Examples

This directory hosts runnable samples that exercise the rebuilt ExecuTorch export stack.

## Available models
- `add`: simple tensor addition
- `linear`: single-layer linear network
- `mv2`: torchvision MobileNetV2 (random weights)

All models expose a `build_model_spec(device=...)` helper returning a ready-to-run module plus example inputs.

## Exporting a model

```bash
python -m examples.export.export_example --model_name add



**Verify**
1. From the repo root, install editable package: `python -m pip install -e .`
2. Generate an artifact: `python -m examples.export.export_example --model_name add`
3. Inspect the output: `python - <<'PY'\nfrom pathlib import Path\nprint(Path('add.ff').read_text()[:200])\nPY`

Once this works, we can advance to Milestone 3 and start reconstructing the C++ runtime scaffolding.
