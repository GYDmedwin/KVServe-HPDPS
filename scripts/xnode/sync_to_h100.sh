#!/usr/bin/env bash
# Copy the KVServe cross-node test files from this machine into the H100
# container (gyd_KV_4). /data is NOT shared between the two hosts.
#
# Requires: sshpass, and SSHPASS exported with the H100 password.
set -euo pipefail

H100_USER="${H100_USER:-guyida}"
H100_HOST="${H100_HOST:-10.18.91.4}"
H100_PORT="${H100_PORT:-2133}"
H100_DOCKER="${H100_DOCKER:-gyd_KV_4}"
REPO="${REPO:-/data/KVServe-HPDPS-cur}"
: "${SSHPASS:?export SSHPASS with the H100 password}"

SSH="sshpass -e ssh -p ${H100_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=10 ${H100_USER}@${H100_HOST}"

FILES=(
  tests/test_pd_cross_node.py
  scripts/xnode/xnode_role.sh
  kvserve_v1/transport/nccl_transport.py
  kvserve_v1/connector/compressed_kv_connector.py
  kvserve_v1/compression/compression_manager.py
  kvserve_v1/compression/manager.py
  kvserve_v1/compression/tilelang_manager.py
  kvserve_v1/compression/codec/lc_codec.py
  kvserve_v1/tilelang_ops/kv_hadamard_quant.py
  kvserve_v1/tilelang_ops/kv_hadamard_quant_v2.py
  tests/bench_compress_v3.py
)

echo "[sync] packing ${#FILES[@]} files -> ${H100_DOCKER}:${REPO}"
tar -C "$REPO" -czf - "${FILES[@]}" | base64 -w0 | \
  $SSH "base64 -d | docker exec -i ${H100_DOCKER} bash -lc 'cd ${REPO} && tar -xzf - && echo [sync] done && ls -l ${FILES[*]}'"
