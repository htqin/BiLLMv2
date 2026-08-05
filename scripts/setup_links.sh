#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BILLM_PATH="${1:-"$ROOT_DIR/../BiLLM"}"
MODEL_PATH="${2:-}"
DATASET_PATH="${3:-}"
CACHE_PATH="${4:-}"

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
link_checked "$BILLM_PATH" "$ROOT_DIR/external/BiLLM"
[[ -z "$MODEL_PATH" ]] || link_checked "$MODEL_PATH" "$ROOT_DIR/external/models"
[[ -z "$DATASET_PATH" ]] || link_checked "$DATASET_PATH" "$ROOT_DIR/external/datasets"
[[ -z "$CACHE_PATH" ]] || link_checked "$CACHE_PATH" "$ROOT_DIR/cache"
