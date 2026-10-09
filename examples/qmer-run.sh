#!/usr/bin/env bash
set -euo pipefail
# Run one shard on one allocated GPU. SHARD_ID is 0..7.
: "${SHARD_ID:?Set SHARD_ID to 0..7}"
: "${QMER_CONFIG:?Set QMER_CONFIG to your calculation YAML}"
: "${QMER_MANIFESTS:?Set QMER_MANIFESTS to the prepared manifest directory}"
: "${QMER_OUTPUT:?Set QMER_OUTPUT to the shared result root}"
case "$SHARD_ID" in [0-7]) ;; *) echo 'SHARD_ID must be 0..7' >&2; exit 2 ;; esac
exec python -m gpu4pyscf_gau.qmer run \
  --config "$QMER_CONFIG" --manifests "$QMER_MANIFESTS" \
  --shard "$SHARD_ID" --output "$QMER_OUTPUT" "$@"
