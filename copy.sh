#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST="${1:-A100}"
REMOTE_BASE_DIR="${2:-/root/needle}"

dirs=()

while IFS= read -r dir; do
  name="${dir#./}"

  if [[ "$name" == "3rdparty" || "$name" == .* ]]; then
    continue
  fi

  dirs+=("$name")
done < <(find . -mindepth 1 -maxdepth 1 -type d | sort)

if [[ ${#dirs[@]} -eq 0 ]]; then
  echo "No directories to copy."
  exit 0
fi

echo "Copying directories to ${REMOTE_HOST}:${REMOTE_BASE_DIR}/"

for dir in "${dirs[@]}"; do
  echo "  - $dir"
  scp -r "$dir" "${REMOTE_HOST}:${REMOTE_BASE_DIR}/"
done

echo "Done."
