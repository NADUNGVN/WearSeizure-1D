#!/usr/bin/env bash
# Pull result summaries (report.json / *.metrics.json) back from the training
# server for local analysis in notebooks/ -- deliberately excludes
# checkpoints (*.pt) and exported tensors (*.npz), which stay on the server.
#
# Usage: scripts/pull_results.sh <server-host> [remote-artifacts-dir] [local-dir]
# Example: scripts/pull_results.sh SERVER-02 '~/Manh/WearSeizure-1D-artifacts' ./artifacts/from_server
set -euo pipefail

SERVER_HOST="${1:?Usage: pull_results.sh <server-host> [remote-artifacts-dir] [local-dir]}"
REMOTE_DIR="${2:-~/Manh/WearSeizure-1D-artifacts}"
LOCAL_DIR="${3:-./artifacts/from_server}"

mkdir -p "$LOCAL_DIR"

rsync -avz --progress \
  --include='*/' \
  --include='*.json' \
  --exclude='*.pt' \
  --exclude='*.npz' \
  --exclude='*' \
  "${SERVER_HOST}:${REMOTE_DIR}/" "${LOCAL_DIR}/"

echo "Pulled report/metrics JSON files to ${LOCAL_DIR}"
