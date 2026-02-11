# AGENTS.md

## Environment

- Use the project virtual environment at `./venv`.
- Before running Python or build commands, activate it:
  - `source ./venv/bin/activate`

## Build / Compile

- Compile the Python package and C++ extension via editable install:
  - `pip install -e .`
- Do not use ad-hoc CMake builds as the default path; prefer `pip install -e .`.
