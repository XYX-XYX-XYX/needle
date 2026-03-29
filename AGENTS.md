# Repository Guidelines

## Project Structure & Module Organization
`python/needle/` contains the core framework: autograd, ops, neural network layers, data loaders, and optimizers. Native backends live in `src/` with CPU C++ and CUDA implementations compiled into the Python package. Use `apps/` for runnable training examples and helper scripts. Keep generated artifacts in `build/`, and treat `data/` plus `tests/hw4/data/` as input fixtures rather than source code. Tests are split between homework coverage in `tests/hw4/` and project-specific checks and benchmarks in `tests/project/`.

## Build, Test, and Development Commands
Run `make lib` to configure CMake and build the native backend into `build/`. Use `make format` to apply `black` to Python and `clang-format` to `src/*.cc` and `src/*.cu`. `make clean` removes compiled outputs.

For local runs, set the package path explicitly when needed: `PYTHONPATH=./python pytest tests/hw4/test_nd_backend.py`. Example app entry points include `PYTHONPATH=./python python apps/simple_ml.py`. Prefer targeted test files over running the entire suite when iterating on one module.

## Coding Style & Naming Conventions
Follow the existing codebase style: 4-space indentation in Python and 2-space indentation in C++/CUDA. Use `snake_case` for Python modules, functions, and variables; use `CapWords` for classes such as `Tensor` and `Adam`. Keep backend file names descriptive, for example `ndarray_backend_cpu.cc` and `ops_flashattention.py`. Run `make format` before opening a PR; there is no separate linter configured in this repository.

## Testing Guidelines
Use `pytest` for all tests. Name new test files `test_<feature>.py` and keep parametrized cases close to the implementation area they validate. Run focused checks with commands such as `PYTHONPATH=./python pytest tests/project/test_flashattention.py` or `PYTHONPATH=./python pytest tests/hw4/test_transformer.py -k attention`. GPU tests should follow the current pattern of skipping when CUDA is unavailable. No formal coverage threshold is configured, so contributors should add or update tests for every behavior change.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commits, often with prefixes like `feat:` and `fix:`. Keep subjects concise and scoped to one change, for example `feat: add transformer layer support`. Pull requests should explain the affected subsystem, list validation commands run, and call out CPU/CUDA impact. Link related issues when available, and include benchmark output or screenshots only when behavior or performance changes are user-visible.
