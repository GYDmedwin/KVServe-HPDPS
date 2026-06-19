"""Cross-node PD test: run prefill OR decode as a standalone single-role process.

Unlike tests/test_kvserve.py (which forks both engines in one process on one
host), this script runs exactly ONE role so prefill and decode can live on two
different machines and transfer KV over the real network (Ethernet or IB).

  Decode node (consumer, binds kv_ip:kv_port):
    python tests/test_pd_cross_node.py --role decode  --kv-ip <DECODE_IP> ...

  Prefill node (producer, connects to kv_ip:kv_port):
    python tests/test_pd_cross_node.py --role prefill --kv-ip <DECODE_IP> ...

Both sides MUST use the same --model, --num-requests, --max-tokens, --mode and
the same --kv-ip/--kv-port. Prompts are deterministic and built in-process, so
both sides produce the same prompt list and matching transfer_ids (sim-<i>).

Start the DECODE side first (it binds the port and prints "[XNODE] DECODE_READY"),
then start the PREFILL side. The producer blocks in NCCL init until the consumer
completes the handshake.

NCCL transport selection (set in the environment, not here):
  Ethernet:  NCCL_IB_DISABLE=1  NCCL_SOCKET_IFNAME=<eth-iface>
  InfiniBand: NCCL_IB_DISABLE=0 NCCL_IB_HCA=<hca>  NCCL_SOCKET_IFNAME=<eth-iface>
"""

import argparse
import json
import os
import time

# ---------------------------------------------------------------------------
# Configuration constants (kept in sync with tests/test_kvserve.py)
# ---------------------------------------------------------------------------

MODEL_PATH = "/data/models/Qwen2.5-7B-Instruct"
GPU_MEMORY_UTILIZATION = 0.6
MAX_MODEL_LEN = int(os.environ.get("KVSERVE_MAX_MODEL_LEN", "4096"))
DEFAULT_NUM_REQUESTS = 10
DEFAULT_KV_PORT = 25010
# Keep the prompt below the model window with headroom for generated tokens.
DEFAULT_MAX_PROMPT_TOKENS = MAX_MODEL_LEN - 256
# HuggingFace cache shared (by path, separate copies) on both nodes.
HF_CACHE_HOME = "/data/datasets/huggingface"
# LongBench (Xnhyacinth/long_bench) is cached on both machines with an
# identical hash, so loading it offline yields the same rows in the same
# order on prefill and decode -> transfer_ids line up.
LONGBENCH_NAME = "Xnhyacinth/long_bench"

CUSTOM_COMPRESSION_CFG = {
    "enabled": True,
    "pipeline": ["quantizer", "codec"],
    "quantizer_config": {
        "model_name": "Qwen2.5-7B-Instruct",
        "hybrid_ratio": 0.5,
        "high_key_max_value": 12,
        "high_value_max_value": 8,
        "low_key_max_value": 6,
        "low_value_max_value": 4,
        "axis_key": "channel",
        "axis_value": "token",
        "split_type": "head",
    },
    "codec_config": {
        "codec_type": "nvcomp",
        "nvcomp_algorithm": "ANS",
        "data_type": "|u1",
    },
    "min_compress_size": 0,
}

# Full 3-stage pipeline: transform (Hadamard) + quantizer + codec.
FULL_COMPRESSION_CFG = {
    "enabled": True,
    "pipeline": ["transformer", "quantizer", "codec"],
    "transformer_config": {"transform_type": "hadamard", "seed": 0x3333},
    "quantizer_config": dict(CUSTOM_COMPRESSION_CFG["quantizer_config"]),
    "codec_config": dict(CUSTOM_COMPRESSION_CFG["codec_config"]),
    "min_compress_size": 0,
}

# Same as FULL but the transform+quantize stage runs the fused TileLang
# compress_v3 kernel instead of the per-layer PyTorch loop (codec unchanged).
FULL_V3_COMPRESSION_CFG = {
    "enabled": True,
    "pipeline": ["transformer", "quantizer", "codec"],
    "backend": "tilelang",
    "transformer_config": {"transform_type": "hadamard", "seed": 0x3333},
    "quantizer_config": dict(CUSTOM_COMPRESSION_CFG["quantizer_config"]),
    "codec_config": dict(CUSTOM_COMPRESSION_CFG["codec_config"]),
    "min_compress_size": 0,
}

# Deterministic built-in prompts (identical on both nodes -> matching token ids).
_BUILTIN_PROMPTS = [
    (
        "Climate change represents one of the most pressing challenges facing humanity "
        "in the 21st century, requiring comprehensive and coordinated efforts across "
        "scientific research, technological innovation, policy development, and global "
        "cooperation. The accumulation of greenhouse gases in the Earth's atmosphere, "
        "primarily carbon dioxide from fossil fuel combustion and deforestation, has "
        "led to unprecedented warming trends. Renewable energy technologies such as "
        "solar panels, wind turbines, and geothermal installations offer promising "
        "pathways toward decarbonization. In summary, the key actions needed to "
        "address climate change are"
    ),
    (
        "The field of artificial intelligence has undergone remarkable transformations "
        "over the past several decades, evolving from simple rule-based systems to "
        "sophisticated neural networks capable of understanding and generating human-like "
        "text. Machine learning algorithms, particularly deep learning models, have "
        "demonstrated exceptional capabilities in tasks ranging from image recognition "
        "and natural language processing to autonomous decision-making and creative "
        "content generation. The development of transformer architectures has "
        "revolutionized how we approach sequence-to-sequence problems. In conclusion, "
        "the future of artificial intelligence will likely involve"
    ),
]


def _set_hf_offline_env() -> None:
    """Force HuggingFace to use only the local cache (deterministic, no network)."""
    os.environ.setdefault("HF_HOME", HF_CACHE_HOME)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_CACHE_HOME, "datasets"))
    for var in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[var] = "1"


def _middle_truncate(model: str, text: str, max_tokens: int) -> str:
    """Truncate a prompt to max_tokens, keeping its head and tail.

    LongBench-style middle truncation preserves the long-context prefix AND the
    trailing question/answer cue. Deterministic given the same model tokenizer,
    so prefill and decode produce identical prompts.
    """
    if max_tokens <= 0:
        return text
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    half = max_tokens // 2
    kept = ids[:half] + ids[-(max_tokens - half):]
    return tok.decode(kept, skip_special_tokens=True)


def _load_longbench_prompts(model: str, config: str, split: str,
                            num_requests: int, max_prompt_tokens: int) -> list:
    _set_hf_offline_env()
    from datasets import load_dataset
    ds = load_dataset(LONGBENCH_NAME, config, split=split)
    n = min(num_requests, ds.num_rows)
    prompts = []
    for i in range(n):
        row = ds[i]
        ctx = row.get("context", "") or ""
        q = row.get("question", "") or ""
        prefix = row.get("answer_prefix", "") or ""
        text = f"{ctx}\n\n{q}\n{prefix}".strip()
        prompts.append(_middle_truncate(model, text, max_prompt_tokens))
    return prompts


def build_prompts(args) -> list:
    if args.dataset == "longbench":
        prompts = _load_longbench_prompts(
            args.model, args.dataset_config, args.dataset_split,
            args.num_requests, args.max_prompt_tokens)
        print(f"[Prompts] Loaded {len(prompts)} LongBench/{args.dataset_config} "
              f"prompts (max_prompt_tokens={args.max_prompt_tokens})", flush=True)
    else:
        prompts = [_BUILTIN_PROMPTS[i % len(_BUILTIN_PROMPTS)]
                   for i in range(args.num_requests)]
        print(f"[Prompts] Using {len(prompts)} deterministic built-in prompts",
              flush=True)
    lengths = [len(p) for p in prompts]
    if lengths:
        print(f"[Prompts] chars min/avg/max = "
              f"{min(lengths)}/{sum(lengths)//len(lengths)}/{max(lengths)}", flush=True)
    return prompts


def make_compression_spec(args):
    if args.mode == "default":
        return "default"
    if args.mode == "controller":
        if not args.library_path:
            raise ValueError("--library-path is required for controller mode")
        return {
            "mode": "controller",
            "library_path": args.library_path,
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "service_config": {
                "bandwidth_mbps": args.bandwidth_mbps,
                "slo_ms": args.slo_ms,
                "accuracy_requirement": args.accuracy_req,
                "t_model_ms": args.t_model_ms,
            },
        }
    if args.mode == "custom":
        return CUSTOM_COMPRESSION_CFG
    if args.mode == "full":
        # NOTE: the original PyTorch pipeline needs a per-model head-score CSV and
        # has NO uniform fallback. It works on Qwen (CSV present, 4 heads) but on
        # Llama it has no CSV → head_scores=None → it can't compress (falls back to
        # raw). compress_v3 (full_v3) added a uniform fallback that fixes this.
        return FULL_COMPRESSION_CFG
    if args.mode == "full_v3":
        spec = dict(FULL_V3_COMPRESSION_CFG)
        # Match the quantizer's model_name to the running model so the correct
        # head-score CSV loads (or falls back to uniform) — instead of the
        # hard-coded Qwen name, which mismatches on e.g. Llama (8 KV heads).
        import os as _os
        qc = dict(spec["quantizer_config"])
        qc["model_name"] = _os.path.basename(args.model.rstrip("/"))
        spec["quantizer_config"] = qc
        if getattr(args, "overlap", "off") != "off":
            spec["overlap"] = args.overlap  # off | tq | full
        if getattr(args, "variant", None):
            spec["variant"] = args.variant          # e.g. vec_shfl
        if getattr(args, "throttle_gx", 0):
            spec["throttle_gx"] = args.throttle_gx  # persistent-grid throttle
        if getattr(args, "side_priority", 0):
            spec["side_stream_priority"] = args.side_priority
        if getattr(args, "codec", "nvcomp") == "lc":
            cc = dict(spec["codec_config"]); cc["codec_type"] = "lc"
            spec["codec_config"] = cc
        return spec
    return None  # none


def _build_llm(model, kv_role, kv_rank, kv_ip, kv_port, gpu_mem_util,
               compression_spec):
    from vllm import LLM
    from vllm.config import KVTransferConfig

    kv_cfg = KVTransferConfig(
        kv_connector="CompressedKVConnector",
        kv_connector_module_path="kvserve_v1.connector.compressed_kv_connector",
        kv_role=kv_role,
        kv_rank=kv_rank,
        kv_parallel_size=2,
        kv_ip=kv_ip,
        kv_port=kv_port,
        kv_connector_extra_config={"compression": compression_spec},
    )
    return LLM(
        model=model,
        kv_transfer_config=kv_cfg,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=1,
        enable_prefix_caching=False,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )


def _sum_transfer_stats(path):
    """Return (total_ms, count, total_bytes) from the transport stats JSONL."""
    if not path or not os.path.exists(path):
        return 0.0, 0, 0
    total_ms, count, total_bytes = 0.0, 0, 0
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_ms += float(row.get("transfer_ms", 0.0))
            count += 1
            total_bytes += int(row.get("bytes", 0))
    return total_ms, count, total_bytes


def _sum_field(path, field):
    """Sum one numeric field across a JSONL stats file (0.0 if absent)."""
    if not path or not os.path.exists(path):
        return 0.0
    total = 0.0
    with open(path) as f:
        for line in f:
            try:
                total += float(json.loads(line).get(field, 0.0))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return total


def run_prefill(args, prompts, compression_spec):
    from vllm import SamplingParams

    # Env must be set before LLM() so the worker process inherits it.
    if args.timeline_path:
        os.environ["KVSERVE_TIMELINE_PATH"] = args.timeline_path
        if os.path.exists(args.timeline_path):
            os.remove(args.timeline_path)
    if args.compression_stats_path:
        os.environ["KVSERVE_COMPRESSION_STATS_PATH"] = args.compression_stats_path
    if args.transfer_stats_path:
        os.environ["KVSERVE_TRANSFER_STATS_PATH"] = args.transfer_stats_path
        if os.path.exists(args.transfer_stats_path):
            os.remove(args.transfer_stats_path)

    llm = _build_llm(args.model, "kv_producer", 0, args.kv_ip, args.kv_port,
                     args.gpu_mem_util, compression_spec)
    params = [
        SamplingParams(max_tokens=1, temperature=0,
                       extra_args={"kv_transfer_params": {"transfer_id": f"sim-{i}"}})
        for i in range(len(prompts))
    ]
    # Timing starts HERE — model load above is excluded.
    start_epoch = time.time()
    t0 = time.monotonic()
    print(f"[XNODE] PREFILL_START start_epoch={start_epoch:.3f}", flush=True)
    llm.generate(prompts, sampling_params=params)
    gen_ms = (time.monotonic() - t0) * 1e3
    end_epoch = time.time()

    comm_ms, comm_n, comm_bytes = _sum_transfer_stats(args.transfer_stats_path)
    compress_ms = _sum_field(args.compression_stats_path, "compress_ms")
    quant_ms = _sum_field(args.compression_stats_path, "quant_ms")
    codec_ms = _sum_field(args.compression_stats_path, "codec_ms")
    prefill_span_ms = _sum_field(args.compression_stats_path, "prefill_span_ms")
    print(f"[Prefill] Done - KV produced/sent. generate_wall={gen_ms:.0f}ms",
          flush=True)
    print(f"[XNODE] PREFILL_TIMING gen_ms={gen_ms:.1f} "
          f"start_epoch={start_epoch:.3f} end_epoch={end_epoch:.3f} "
          f"comm_ms={comm_ms:.1f} comm_count={comm_n} comm_bytes={comm_bytes} "
          f"compress_ms={compress_ms:.1f} quant_ms={quant_ms:.1f} "
          f"codec_ms={codec_ms:.1f} prefill_span_ms={prefill_span_ms:.1f}",
          flush=True)
    print("[XNODE] PREFILL_DONE", flush=True)


def run_decode(args, prompts, compression_spec):
    from vllm import SamplingParams

    # Env must be set before LLM() so the worker process inherits it.
    if args.timeline_path:
        os.environ["KVSERVE_TIMELINE_PATH"] = args.timeline_path
        if os.path.exists(args.timeline_path):
            os.remove(args.timeline_path)
    if args.decode_stats_path:
        os.environ["KVSERVE_DECODE_STATS_PATH"] = args.decode_stats_path
        if os.path.exists(args.decode_stats_path):
            os.remove(args.decode_stats_path)

    llm = _build_llm(args.model, "kv_consumer", 1, args.kv_ip, args.kv_port,
                     args.gpu_mem_util, compression_spec)
    # Printed AFTER the consumer transport has bound its ROUTER, so the
    # orchestrator can safely launch the prefill side now.
    print("[XNODE] DECODE_READY", flush=True)

    params = [
        SamplingParams(max_tokens=args.max_tokens, temperature=0,
                       extra_args={"kv_transfer_params": {"transfer_id": f"sim-{i}"}})
        for i in range(len(prompts))
    ]
    # Timing starts HERE — model load above is excluded. decode generate wall
    # includes the per-request wait for remote KV (start_load_kv) + compute.
    start_epoch = time.time()
    t0 = time.monotonic()
    outputs = llm.generate(prompts, sampling_params=params)
    gen_ms = (time.monotonic() - t0) * 1e3
    end_epoch = time.time()

    n = len(outputs)
    avg_tok = sum(len(o.outputs[0].token_ids) for o in outputs) / max(n, 1)
    if args.print_outputs:
        for i, o in enumerate(outputs):
            print(f"[Decode][{i}] {o.prompt[:50]!r}... -> "
                  f"{o.outputs[0].text!r}", flush=True)

    # Output fingerprint: deterministic (temp=0), so a correct KV transfer must
    # produce the SAME hash regardless of compression/overlap mode. Lets us
    # catch silent corruption (e.g. decode running on incomplete streamed KV).
    import hashlib
    _toks = [tuple(int(t) for t in o.outputs[0].token_ids) for o in outputs]
    out_hash = hashlib.md5(repr(_toks).encode()).hexdigest()[:12]

    decompress_ms = _sum_field(args.decode_stats_path, "decompress_ms")
    codec_decode_ms = _sum_field(args.decode_stats_path, "codec_decode_ms")
    dequant_ms = _sum_field(args.decode_stats_path, "dequant_ms")
    print(f"\n{'='*60}", flush=True)
    print(f"[XNODE] DECODE_RESULT mode={args.mode} n={n}/{len(prompts)} "
          f"avg_out_tokens={avg_tok:.1f} decode_wall_ms={gen_ms:.0f} "
          f"out_hash={out_hash}", flush=True)
    print(f"[XNODE] DECODE_TIMING gen_ms={gen_ms:.1f} "
          f"start_epoch={start_epoch:.3f} end_epoch={end_epoch:.3f} "
          f"n={n} max_tokens={args.max_tokens} decompress_ms={decompress_ms:.1f} "
          f"codec_decode_ms={codec_decode_ms:.1f} dequant_ms={dequant_ms:.1f}",
          flush=True)
    print(f"{'='*60}", flush=True)
    print("[XNODE] DECODE_DONE", flush=True)


def main():
    p = argparse.ArgumentParser(
        description="Cross-node PD test (single role per process)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--role", required=True, choices=["prefill", "decode"])
    p.add_argument("--kv-ip", required=True,
                   help="Decode (consumer) node IP that the producer connects to")
    p.add_argument("--kv-port", type=int, default=DEFAULT_KV_PORT)
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--gpu", type=int, default=0,
                   help="Local GPU index for this role (sets CUDA_VISIBLE_DEVICES)")
    p.add_argument("--gpu-mem-util", type=float, default=GPU_MEMORY_UTILIZATION)
    p.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS)
    p.add_argument("--max-tokens", type=int, default=30)
    p.add_argument("--dataset", choices=["builtin", "longbench"], default="builtin",
                   help="Prompt source. 'longbench' uses the cached "
                        "Xnhyacinth/long_bench dataset (long contexts).")
    p.add_argument("--dataset-config", default="hotpotqa",
                   help="[longbench] subset/config name")
    p.add_argument("--dataset-split", default="test",
                   help="[longbench] split name")
    p.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS,
                   help="Middle-truncate prompts to this many tokens")
    p.add_argument("--mode",
                   choices=["none", "default", "custom", "full", "full_v3", "controller"],
                   default="none",
                   help="custom=quantizer+codec, full=transformer+quantizer+codec, "
                        "full_v3=full with fused TileLang compress_v3")
    p.add_argument("--print-outputs", action="store_true", default=False)
    p.add_argument("--compression-stats-path", default=None,
                   help="[prefill] JSONL path to log original/compressed bytes")
    p.add_argument("--transfer-stats-path", default=None,
                   help="[prefill] JSONL path to log per-transfer NCCL wire time")
    p.add_argument("--decode-stats-path", default=None,
                   help="[decode] JSONL path to log per-request decompress time")
    p.add_argument("--timeline-path", default=None,
                   help="path for [TL] critical-path epoch markers")
    p.add_argument("--overlap", choices=["off", "tq", "full", "stream"], default="off",
                   help="[full_v3] multi-stream overlap: off | tq (transform+quant "
                        "on side stream) | full (per-layer +codec) | stream "
                        "(per-layer compress+send immediately; decode streams)")
    p.add_argument("--variant", default=None,
                   help="[full_v3] compress_v3 kernel variant, e.g. vec_shfl")
    p.add_argument("--throttle-gx", type=int, default=0,
                   help="[full_v3] persistent-grid throttle (active SM cap); 0=off")
    p.add_argument("--side-priority", type=int, default=0,
                   help="[full_v3 overlap] side-stream CUDA priority (-1=high, 0=default)")
    p.add_argument("--codec", choices=["nvcomp", "lc"], default="nvcomp",
                   help="[full_v3] codec backend: nvcomp (closed) or lc (open-source)")

    # controller-only
    p.add_argument("--library-path", default=None)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--bandwidth-mbps", type=float, default=1000.0)
    p.add_argument("--slo-ms", type=float, default=200.0)
    p.add_argument("--accuracy-req", type=float, default=0.92)
    p.add_argument("--t-model-ms", type=float, default=0.0)

    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    compression_spec = make_compression_spec(args)
    prompts = build_prompts(args)

    print(f"\n{'='*60}", flush=True)
    print(f"CROSS-NODE PD TEST  role={args.role}", flush=True)
    print(f"  model        : {args.model}", flush=True)
    print(f"  kv_ip:port   : {args.kv_ip}:{args.kv_port}", flush=True)
    print(f"  gpu          : {args.gpu}", flush=True)
    print(f"  mode         : {args.mode}", flush=True)
    print(f"  requests     : {len(prompts)}  max_tokens={args.max_tokens}", flush=True)
    print(f"  NCCL_IB_DISABLE={os.environ.get('NCCL_IB_DISABLE','<unset>')} "
          f"NCCL_IB_HCA={os.environ.get('NCCL_IB_HCA','<unset>')} "
          f"NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME','<unset>')}",
          flush=True)
    print(f"{'='*60}\n", flush=True)

    if args.role == "prefill":
        run_prefill(args, prompts, compression_spec)
    else:
        run_decode(args, prompts, compression_spec)


if __name__ == "__main__":
    main()
