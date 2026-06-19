#!/usr/bin/env bash
# Run ONE PD role (prefill or decode) for a cross-node KVServe test.
# This file is identical on both machines; per-machine NIC names are passed in
# via env (ETH_IFACE / IB_HCA) because the two hosts use different NIC names.
#
# Required env:
#   ROLE        prefill | decode
#   KV_IP       decode-node IP the producer connects to (e.g. 10.18.91.4)
#   NET         ib | eth          (which fabric NCCL data plane uses)
#   ETH_IFACE   this host's ethernet iface carrying KV_IP's subnet
#   IB_HCA      this host's IB HCA (only used when NET=ib)
# Optional env (with defaults):
#   KV_PORT=25010  MODEL=/data/models/Qwen2.5-7B-Instruct  GPU=0
#   GPU_MEM_UTIL=0.6  MODE=none  NUM_REQUESTS=10  MAX_TOKENS=30
#   STATS_PATH=<unset>  REPO=/data/KVServe-HPDPS-cur  EXTRA_ARGS=<unset>
set -euo pipefail

: "${ROLE:?set ROLE=prefill|decode}"
: "${KV_IP:?set KV_IP}"
: "${NET:?set NET=ib|eth}"
: "${ETH_IFACE:?set ETH_IFACE}"
KV_PORT="${KV_PORT:-25010}"
MODEL="${MODEL:-/data/models/Qwen2.5-7B-Instruct}"
GPU="${GPU:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
MODE="${MODE:-none}"
NUM_REQUESTS="${NUM_REQUESTS:-10}"
MAX_TOKENS="${MAX_TOKENS:-30}"
REPO="${REPO:-/data/KVServe-HPDPS-cur}"

# ── NCCL transport selection ────────────────────────────────────────────────
# NOTE: NCCL_IB_DISABLE alone does NOT force ethernet when an external IB net
# plugin (e.g. IBext_v10) is present — it keeps using IB. NCCL_NET pins the
# transport explicitly: Socket = TCP over NCCL_SOCKET_IFNAME (ethernet),
# IB = RDMA verbs. Confirm in logs: "via NET/Socket" vs "via NET/IB*".
export NCCL_SOCKET_IFNAME="$ETH_IFACE"   # control/bootstrap (and Socket data)
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
case "$NET" in
  eth)
    export NCCL_IB_DISABLE=1
    export NCCL_NET=Socket
    ;;
  ib)
    : "${IB_HCA:?set IB_HCA for NET=ib}"
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA="$IB_HCA"
    export NCCL_NET=IB
    ;;
  *) echo "NET must be ib|eth (got $NET)" >&2; exit 2;;
esac

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export KVSERVE_STREAM_GROUP="${STREAM_GROUP:-1}"
export KVSERVE_COMPRESS_OFFLOAD="${COMPRESS_OFFLOAD:-1}"
export KVSERVE_ASYNC_SEND="${ASYNC:-1}"
export KVSERVE_MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

args=(
  --role "$ROLE"
  --kv-ip "$KV_IP"
  --kv-port "$KV_PORT"
  --model "$MODEL"
  --gpu "$GPU"
  --gpu-mem-util "$GPU_MEM_UTIL"
  --num-requests "$NUM_REQUESTS"
  --max-tokens "$MAX_TOKENS"
  --mode "$MODE"
  --dataset "${DATASET:-builtin}"
  --dataset-config "${DATASET_CONFIG:-hotpotqa}"
  --dataset-split "${DATASET_SPLIT:-test}"
  --max-prompt-tokens "${MAX_PROMPT_TOKENS:-3840}"
)
if [[ "${STATS_PATH:-}" != "" ]]; then
  args+=(--compression-stats-path "$STATS_PATH")
fi
if [[ "${TRANSFER_STATS_PATH:-}" != "" ]]; then
  args+=(--transfer-stats-path "$TRANSFER_STATS_PATH")
fi
if [[ "${DECODE_STATS_PATH:-}" != "" ]]; then
  args+=(--decode-stats-path "$DECODE_STATS_PATH")
fi
if [[ "${TIMELINE_PATH:-}" != "" ]]; then
  args+=(--timeline-path "$TIMELINE_PATH")
fi
if [[ "${OVERLAP:-off}" != "off" ]]; then
  args+=(--overlap "$OVERLAP")
fi
if [[ "${VARIANT:-}" != "" ]]; then
  args+=(--variant "$VARIANT")
fi
if [[ "${THROTTLE_GX:-0}" != "0" ]]; then
  args+=(--throttle-gx "$THROTTLE_GX")
fi
if [[ "${SIDE_PRIORITY:-0}" != "0" ]]; then
  args+=(--side-priority "$SIDE_PRIORITY")
fi
if [[ "${CODEC:-nvcomp}" != "nvcomp" ]]; then
  args+=(--codec "$CODEC")
fi
if [[ "${EXTRA_ARGS:-}" != "" ]]; then
  # shellcheck disable=SC2206
  args+=($EXTRA_ARGS)
fi

echo "[xnode_role] ROLE=$ROLE NET=$NET MODE=$MODE GPU=$GPU KV=$KV_IP:$KV_PORT" \
     "ETH_IFACE=$ETH_IFACE IB_HCA=${IB_HCA:-n/a}"
exec python3 tests/test_pd_cross_node.py "${args[@]}"
