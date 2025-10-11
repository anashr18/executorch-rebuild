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

#cmake
##To make clang use the cmake compiler flags 
 1. create a clangd file 
    CompileFlags:
        CompilationDatabase: cmake-out
2. create the db 
cmake -S . -B cmake-out -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
# optional: expose it at root for tools that auto-detect
ln -sf cmake-out/compile_commands.json compile_commands.json
now the IDE follows the cmake compiler flags
3. cmake build commands
cmake -S . -B cmake-out
cmake --build cmake-out -j
./cmake-out/executor_runner --model_path add.ff

