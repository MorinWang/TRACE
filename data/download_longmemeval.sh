#!/usr/bin/env bash
# Fetch LongMemEval-S cleaned dataset for the TRACE release.
# Usage: bash data/download_longmemeval.sh   (run from repo root)
#
# Downloads from the official HuggingFace dataset release:
#   https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
#
# Falls back to manual instructions if the download fails.
set -euo pipefail

URL="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
DEST_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${DEST_DIR}/longmemeval_s_cleaned.json"

if [[ -f "${TARGET}" ]]; then
  echo "Already present: ${TARGET} ($(wc -c < "${TARGET}") bytes)"
  exit 0
fi

echo "Downloading longmemeval_s_cleaned.json from HuggingFace ..."
echo "  URL: ${URL}"
echo "  Dest: ${TARGET}"

if command -v curl >/dev/null 2>&1; then
  curl -L -f -o "${TARGET}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${TARGET}" "${URL}"
else
  echo "ERROR: neither curl nor wget is available." >&2
  echo "Install one, or download manually from:" >&2
  echo "  ${URL}" >&2
  echo "and place the file at ${TARGET}" >&2
  exit 1
fi

echo "Done. Wrote ${TARGET} ($(wc -c < "${TARGET}") bytes)."
