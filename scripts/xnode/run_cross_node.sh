#!/usr/bin/env bash
# Orchestrate a cross-node KVServe PD test from the LOCAL (RTX Pro 6000) host.
#
#   Topology:  PREFILL (producer) = local gyd_KV_5
#              DECODE  (consumer)  = H100 gyd_KV_4  (binds KV_IP)
#
#   For each (NET in eth/ib) x (COMP in none/custom):
#     1. launch decode inside the H100 container in the background
#     2. wait until it prints DECODE_READY
#     3. run prefill locally (foreground) -> sends KV over NET
#     4. collect the decode result + compression ratio
#
# Usage:
#   export SSHPASS='...'                       # H100 password
#   scripts/xnode/run_cross_node.sh            # full matrix: eth/ib x none/custom
#   NETS="eth" COMPS="none" scripts/xnode/run_cross_node.sh   # single combo
#
# Env knobs: NETS, COMPS, NUM_REQUESTS, MAX_TOKENS, KV_PORT, MODEL
set -euo pipefail

# ── Topology / per-machine config ───────────────────────────────────────────
H100_USER="${H100_USER:-guyida}"
H100_HOST="${H100_HOST:-10.18.91.4}"
H100_PORT="${H100_PORT:-2133}"
H100_DOCKER="${H100_DOCKER:-gyd_KV_4}"
REPO="${REPO:-/data/KVServe-HPDPS-cur}"

KV_IP="${KV_IP:-10.18.91.4}"          # decode (H100) ethernet IP
KV_PORT="${KV_PORT:-25010}"
MODEL="${MODEL:-/data/models/Qwen2.5-7B-Instruct}"
NUM_REQUESTS="${NUM_REQUESTS:-10}"
MAX_TOKENS="${MAX_TOKENS:-30}"
DATASET="${DATASET:-longbench}"            # builtin | longbench
DATASET_CONFIG="${DATASET_CONFIG:-hotpotqa}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-3840}"

# NIC names differ between the two hosts:
LOCAL_ETH="${LOCAL_ETH:-enp65s0f0}"   # local prefill node
LOCAL_HCA="${LOCAL_HCA:-mlx5_0}"
H100_ETH="${H100_ETH:-ens65f0np0}"    # H100 decode node
H100_HCA="${H100_HCA:-mlx5_2}"

LOCAL_GPU="${LOCAL_GPU:-0}"
H100_GPU="${H100_GPU:-0}"

NETS="${NETS:-eth ib}"
# Standard going forward: compare no-compression vs full_v3 (fused TileLang
# compress_v3 = transform+quantize + codec) — the default compression mode that
# wins on IB and on slow links. Override with COMPS="none full full_v3" etc.
COMPS="${COMPS:-none full_v3}"
REPEATS="${REPEATS:-1}"   # repeat each combo R times; final table reports medians
TIMELINE="${TIMELINE:-0}" # 1 = emit [TL] critical-path epoch markers in logs

: "${SSHPASS:?export SSHPASS with the H100 password}"
SSH="sshpass -e ssh -p ${H100_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=10 ${H100_USER}@${H100_HOST}"

OUT_LOCAL="${REPO}/sim_outputs/xnode"
mkdir -p "$OUT_LOCAL"
REMOTE_OUT="${REPO}/sim_outputs/xnode"

dexec() { $SSH "docker exec ${H100_DOCKER} bash -lc \"$1\""; }

run_combo() {
  local net="$1" comp="$2"
  # comp is colon-separated: mode[:overlap][:variant][:gxN]
  #   e.g. full_v3 | full_v3:tq | full_v3:tq:vec:gx47
  local mode="${comp%%:*}" overlap="off" variant="" gx="0" sprio="0" codec="nvcomp" grp="1" ofl="1" async="1"
  local _p; IFS=':' read -ra _parts <<< "$comp"
  for _p in "${_parts[@]:1}"; do
    case "$_p" in
      tq|full|stream|off) overlap="$_p";;
      gx*) gx="${_p#gx}";;
      p-*|p[0-9]*) sprio="${_p#p}";;
      vec*|shfl*) variant="vec_shfl";;
      lc) codec="lc";;
      grp) grp="1";;
      ofl) ofl="1";;
      perreq) grp="0"; ofl="0";;
      nogrp) grp="0"; ofl="0";;
      noofl) ofl="0";;
      syncsend) async="0";;
    esac
  done
  local tag="${net}_${comp//:/_}"
  local rlog="${REMOTE_OUT}/decode_${tag}.log"
  local plog="${OUT_LOCAL}/prefill_${tag}.log"
  local stats="${OUT_LOCAL}/stats_${tag}.jsonl"
  local tstats="${OUT_LOCAL}/transfer_${tag}.jsonl"

  echo
  echo "############################################################"
  echo "# COMBO  net=${net}  comp=${comp}"
  echo "############################################################"

  dexec "mkdir -p ${REMOTE_OUT}; rm -f ${rlog}"
  rm -f "$plog" "$stats" "$tstats"

  # Robust pre-combo cleanup: a prior combo's decode can linger (hang after
  # DECODE_DONE or time out), holding GPU memory + ZMQ port ${KV_PORT} and making
  # every later combo fail. Kill any leftover engine/test procs on BOTH hosts and
  # wait for the GPU memory to drain before launching this combo's decode.
  echo "[orch] pre-combo cleanup (kill leftover decode/engine procs) ..."
  dexec "pkill -9 -f test_pd_cross_node.py; pkill -9 -f EngineCore; pkill -9 -f spawn_main; pkill -9 -f resource_tracker; sleep 5; true"
  pkill -9 -f test_pd_cross_node.py 2>/dev/null || true

  # 1) launch decode on H100 (background, detached)
  echo "[orch] starting decode on ${H100_DOCKER} ..."
  local rdstats="${REMOTE_OUT}/decstats_${tag}.jsonl"
  local rtl="" ltl=""
  if [[ "$TIMELINE" == "1" ]]; then
    rtl="${REMOTE_OUT}/tl_${tag}.txt"; ltl="${OUT_LOCAL}/tl_${tag}.txt"
  fi
  dexec "cd ${REPO} && ROLE=decode KV_IP=${KV_IP} KV_PORT=${KV_PORT} NET=${net} \
        ETH_IFACE=${H100_ETH} IB_HCA=${H100_HCA} MODEL=${MODEL} GPU=${H100_GPU} \
        GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6} MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096} \
        MODE=${mode} OVERLAP=${overlap} VARIANT=${variant} THROTTLE_GX=${gx} SIDE_PRIORITY=${sprio} CODEC=${codec} STREAM_GROUP=${grp} COMPRESS_OFFLOAD=${ofl} ASYNC=${async} ASYNC_COMP=${ASYNC_COMP:-1} \
        NUM_REQUESTS=${NUM_REQUESTS} MAX_TOKENS=${MAX_TOKENS} REPO=${REPO} \
        DATASET=${DATASET} DATASET_CONFIG=${DATASET_CONFIG} DATASET_SPLIT=${DATASET_SPLIT} \
        MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS} DECODE_STATS_PATH=${rdstats} TIMELINE_PATH=${rtl} \
        setsid bash scripts/xnode/xnode_role.sh >${rlog} 2>&1 </dev/null & echo decode-launched"

  # 2) wait for DECODE_READY
  echo -n "[orch] waiting for DECODE_READY "
  local ready="" i
  for i in $(seq 1 120); do   # up to ~10 min (model load)
    if dexec "grep -q DECODE_READY ${rlog} 2>/dev/null"; then ready=1; break; fi
    if dexec "grep -qE 'Traceback|Error|Exception' ${rlog} 2>/dev/null"; then
      echo " FAILED (decode errored during init)"; dexec "tail -n 40 ${rlog}"; return 1
    fi
    echo -n "."; sleep 5
  done
  echo
  if [[ -z "$ready" ]]; then
    echo "[orch] decode never became ready; tail:"; dexec "tail -n 40 ${rlog}"
    dexec "pkill -f test_pd_cross_node.py; pkill -9 -f EngineCore; true"; return 1
  fi
  echo "[orch] decode READY."

  # 3) run prefill locally (foreground)
  echo "[orch] running prefill locally ..."
  set +e
  ROLE=prefill KV_IP=${KV_IP} KV_PORT=${KV_PORT} NET=${net} \
    ETH_IFACE=${LOCAL_ETH} IB_HCA=${LOCAL_HCA} MODEL=${MODEL} GPU=${LOCAL_GPU} \
    GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6} MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096} \
    MODE=${mode} OVERLAP=${overlap} VARIANT=${variant} THROTTLE_GX=${gx} SIDE_PRIORITY=${sprio} CODEC=${codec} STREAM_GROUP=${grp} COMPRESS_OFFLOAD=${ofl} ASYNC=${async} ASYNC_COMP=${ASYNC_COMP:-1} \
    NUM_REQUESTS=${NUM_REQUESTS} MAX_TOKENS=${MAX_TOKENS} REPO=${REPO} \
    DATASET=${DATASET} DATASET_CONFIG=${DATASET_CONFIG} DATASET_SPLIT=${DATASET_SPLIT} \
    MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS} STATS_PATH=${stats} TRANSFER_STATS_PATH=${tstats} \
    TIMELINE_PATH=${ltl} \
    bash "${REPO}/scripts/xnode/xnode_role.sh" >"$plog" 2>&1
  local prc=$?
  set -e
  echo "[orch] prefill exited rc=${prc} (log: ${plog})"

  # 4) wait for decode result
  echo -n "[orch] waiting for DECODE_DONE "
  for i in $(seq 1 60); do  # up to 5 min
    if dexec "grep -q DECODE_DONE ${rlog} 2>/dev/null"; then break; fi
    echo -n "."; sleep 5
  done
  echo

  echo "---- decode result (${tag}) ----"
  local ptiming dtiming netpath
  ptiming="$(grep 'XNODE\] PREFILL_TIMING' "$plog" 2>/dev/null | tail -1)"
  dtiming="$(dexec "grep 'XNODE\] DECODE_TIMING' ${rlog} 2>/dev/null | tail -1" || true)"
  netpath="$(dexec "grep -oE 'via NET/[A-Za-z]+' ${rlog} 2>/dev/null | tail -1" || true)"
  echo "  prefill: ${ptiming:-<none>}"
  echo "  decode : ${dtiming:-<none>}"
  echo "  nccl   : ${netpath:-<unknown>}"

  # Parse + accumulate one metrics row for the final table.
  python3 - "$tag" "$net" "$comp" "$NUM_REQUESTS" "$stats" "$tstats" \
                   "$SUMMARY" "$ptiming" "$dtiming" "$netpath" <<'PY'
import json, os, sys, re
tag, net, comp, n, stats, tstats, summary, ptiming, dtiming, netpath = sys.argv[1:11]
n = int(n)
def kv(s):
    return {k: float(v) for k, v in re.findall(r'(\w+)=([-\d.]+)', s or '')}
p, d = kv(ptiming), kv(dtiming)
def ratio(path):
    rs=[]
    if path and os.path.exists(path):
        for ln in open(path):
            try:
                r=json.loads(ln); o=float(r["original_bytes"]); c=float(r["compressed_bytes"])
                if o>0 and c>0: rs.append(o/c)
            except Exception: pass
    return sum(rs)/len(rs) if rs else None
makespan = None
if p.get("start_epoch") and d.get("end_epoch"):
    makespan = (d["end_epoch"] - p["start_epoch"]) * 1000.0
row = {
    "tag": tag, "net": net, "comp": comp, "n": n,
    "nccl": (netpath or "").replace("via NET/", "") or "?",
    "prefill_ms": p.get("gen_ms"),
    "decode_ms": d.get("gen_ms"),
    "comm_ms": p.get("comm_ms"),
    "comm_count": int(p.get("comm_count", 0)),
    "comm_mb": p.get("comm_bytes", 0) / 1e6,
    "compress_ms": p.get("compress_ms"),
    "decompress_ms": d.get("decompress_ms"),
    "quant_ms": p.get("quant_ms"),
    "codec_ms": p.get("codec_ms"),
    "prefill_span_ms": p.get("prefill_span_ms"),
    "codec_decode_ms": d.get("codec_decode_ms"),
    "dequant_ms": d.get("dequant_ms"),
    "makespan_ms": makespan,
    "avg_req_ms": (makespan / n) if makespan and n else None,
    "ratio": ratio(stats),
}
with open(summary, "a") as f:
    f.write(json.dumps(row) + "\n")
PY

  if [[ "$TIMELINE" == "1" && -n "$rtl" ]]; then
    dexec "cat ${rtl} 2>/dev/null" > "${OUT_LOCAL}/tl_decode_${tag}.txt" 2>/dev/null || true
    echo "[orch] timeline saved: ${ltl} (prefill) , ${OUT_LOCAL}/tl_decode_${tag}.txt (decode)"
  fi

  dexec "pkill -f test_pd_cross_node.py; pkill -9 -f EngineCore; true"
  sleep 2
}

SUMMARY="${OUT_LOCAL}/summary.jsonl"
rm -f "$SUMMARY"

for net in $NETS; do
  for comp in $COMPS; do
    for r in $(seq 1 "$REPEATS"); do
      echo "[orch] === ${net}/${comp} repeat ${r}/${REPEATS} ==="
      run_combo "$net" "$comp" || echo "[orch] combo ${net}/${comp} r${r} FAILED, continuing"
    done
  done
done

echo
echo "[orch] all combos done. Local logs in ${OUT_LOCAL}"
echo
# ── Final comparison table ──────────────────────────────────────────────────
python3 - "$SUMMARY" <<'PY'
import json, sys, os
from statistics import median
path = sys.argv[1]
rows = []
if os.path.exists(path):
    for ln in open(path):
        try: rows.append(json.loads(ln))
        except Exception: pass

# Group repeats by (net, comp), preserving first-seen order.
groups = {}
order = []
for r in rows:
    k = (r.get("net"), r.get("comp"))
    if k not in groups:
        groups[k] = []; order.append(k)
    groups[k].append(r)

def med(rs, key):
    vals = [r.get(key) for r in rs if isinstance(r.get(key), (int, float))]
    return median(vals) if vals else None
def spread(rs, key):
    vals = [r.get(key) for r in rs if isinstance(r.get(key), (int, float))]
    if len(vals) < 2: return ""
    return "  [%.0f-%.0f]" % (min(vals), max(vals))
def f(x, suf=""):
    return ("%.1f%s" % (x, suf)) if isinstance(x, (int, float)) else "-"
def fx(x):
    return ("%.2fx" % x) if isinstance(x, (int, float)) else "-"

# Clean transfer-critical-path metric, built from single-clock instrument
# measurements (NOT the noise-dominated cross-machine makespan):
#   xfer_path     = compress + comm + decompress   (serial, what runs today)
#   xfer_overlap  = compress + comm                (decompress hidden in decode)
def xfer_path(rs):
    c = med(rs, "compress_ms") or 0.0
    m = med(rs, "comm_ms") or 0.0
    d = med(rs, "decompress_ms") or 0.0
    return c + m + d
def xfer_overlap(rs):
    c = med(rs, "compress_ms") or 0.0
    m = med(rs, "comm_ms") or 0.0
    return c + m

hdr = ["net", "comp", "nccl", "reps", "comm_ms", "comm_MB", "compr_ms",
       "decmpr_ms", "xfer_path", "xfer_if_ovlp", "makespan(med)", "ratio"]
w = [5, 8, 7, 4, 9, 8, 9, 9, 10, 12, 14, 7]
def line(cells):
    print(" | ".join(str(c).ljust(w[i]) for i, c in enumerate(cells)))
print("\n" + "=" * 130)
print("CROSS-NODE PD COMPARISON  (medians over repeats)")
print("  xfer_path = compress+comm+decompress (serial today, CLEAN single-clock); "
      "xfer_if_ovlp = compress+comm (decompress hidden in decode); "
      "makespan is cross-machine and noise-dominated — prefer xfer_path.")
print("=" * 130)
line(hdr)
line(["-" * x for x in w])
for k in order:
    rs = groups[k]
    nccl = next((r.get("nccl") for r in rs if r.get("nccl")), "?")
    line([k[0], k[1], nccl, len(rs),
          f(med(rs, "comm_ms")), f(med(rs, "comm_mb")), f(med(rs, "compress_ms")),
          f(med(rs, "decompress_ms")), f(xfer_path(rs)), f(xfer_overlap(rs)),
          f(med(rs, "makespan_ms")) + spread(rs, "makespan_ms"),
          fx(med(rs, "ratio"))])
print("=" * 130)

# ── Overlap-analysis table (compression breakdown, medians) ─────────────────
ov = [k for k in order if k[1] not in (None, "none")]
if ov:
    print("\n" + "=" * 124)
    print("OVERLAP ANALYSIS  (medians; quant=per-layer transform+quantize; "
          "codec=monolithic; decmp split = codec-decode + per-layer dequant)")
    print("=" * 124)
    h2 = ["net", "comp", "quant_ms", "codec_ms", "compress_ms",
          "decmp:codec", "decmp:dequant", "decompress_ms"]
    w2 = [5, 8, 9, 9, 11, 11, 13, 13]
    def l2(c): print(" | ".join(str(x).ljust(w2[i]) for i, x in enumerate(c)))
    l2(h2); l2(["-" * x for x in w2])
    for k in ov:
        rs = groups[k]
        l2([k[0], k[1], f(med(rs, "quant_ms")), f(med(rs, "codec_ms")),
            f(med(rs, "compress_ms")), f(med(rs, "codec_decode_ms")),
            f(med(rs, "dequant_ms")), f(med(rs, "decompress_ms"))])
    print("=" * 124)
PY
