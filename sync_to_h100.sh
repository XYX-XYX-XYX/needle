#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-H100:/home/xyx/needle}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

paths=(
  "src"
  # "tests"
  # "python"
  # "CMakeLists.txt"
  # "project.ipynb"
)

cd "${ROOT_DIR}"

rsync -av --relative "${paths[@]}" "${DEST}/"
