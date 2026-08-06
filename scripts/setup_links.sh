#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${1:-}"
DATASET_PATH="${2:-}"
CACHE_PATH="${3:-}"

link_checked() {
  local source_path="$1"
  local target_path="$2"
  if [[ ! -e "$source_path" ]]; then
    echo "error: link source does not exist: $source_path" >&2
    exit 2
  fi
  if [[ -e "$target_path" || -L "$target_path" ]]; then
    if [[ "$(readlink -f "$target_path")" == "$(readlink -f "$source_path")" ]]; then
      return
    fi
    echo "error: target already exists and points elsewhere: $target_path" >&2
    exit 2
  fi
  ln -s "$source_path" "$target_path"
}

mkdir -p "$ROOT_DIR/external"
[[ -z "$MODEL_PATH" ]] || link_checked "$MODEL_PATH" "$ROOT_DIR/external/models"
[[ -z "$DATASET_PATH" ]] || link_checked "$DATASET_PATH" "$ROOT_DIR/external/datasets"
[[ -z "$CACHE_PATH" ]] || link_checked "$CACHE_PATH" "/autodl-fs/data/cclanro/billm-v2-output/cache"
