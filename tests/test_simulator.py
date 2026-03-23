"""PD Separation end-to-end test with optional KV compression and lm-eval prompts.

Mirrors /root/workspaces/KVServe/test/test_simulator.py in scope but uses the
real vLLM v1 PD separation (CompressedKVConnector + NCCL transport) instead of
the file-based SimulatorBackend from the original project.

USAGE
=====
  python tests/test_simulator.py                         # no compression, built-in prompts
  python tests/test_simulator.py --mode custom           # custom compression config
  python tests/test_simulator.py --mode default          # built-in default config
  python tests/test_simulator.py --mode controller       # online adaptive (needs --library-path)
      --library-path /path/to/profiles.json
      --bandwidth-mbps 1000 --slo-ms 200 --accuracy-req 0.92
  python tests/test_simulator.py --lmeval-task wikitext --num-requests 20

CONFIGURATION
=============
Edit the constants block below to change model, GPU memory, ports, etc.
"""

import argparse
import csv
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

MODEL_PATH = "/root/data/models/Qwen2.5-7B-Instruct"
GPU_MEMORY_UTILIZATION = 0.6
MAX_MODEL_LEN = 4096
DEFAULT_NUM_REQUESTS = 10
DEFAULT_KV_PORT = 25010
OUTPUT_DIR = "./sim_outputs"

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

# Built-in fallback prompts used when --lmeval-task is not given.
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

# ---------------------------------------------------------------------------
# lm-eval prompt loading  (lm-eval-harness >= 0.4)
# ---------------------------------------------------------------------------

def load_lmeval_prompts(task_name: str, num_requests: int,
                        offline: bool = True) -> list:
    """Load prompts from an lm-eval 0.4 task.

    Tries test -> validation -> train splits in order; uses doc_to_text to
    convert each document to a plain string prompt.

    Args:
        offline: If True (default), force HuggingFace datasets to use only
                 local cache and skip Hub version checks.  Set to False if you
                 want to allow downloading the dataset on first run.
    """
    try:
        from lm_eval.tasks import TaskManager, get_task_dict
    except ImportError:
        raise RuntimeError("lm-eval not installed. Run: pip install lm-eval")

    if offline:
        # Force all HuggingFace libraries to use local cache only.
        # HF_HUB_OFFLINE  — huggingface_hub (used by lm-eval task loading)
        # HF_DATASETS_OFFLINE — datasets library
        # TRANSFORMERS_OFFLINE — transformers
        # Must be set before TaskManager() / get_task_dict() are called.
        for _var in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ[_var] = "1"

    task_manager = TaskManager()
    task_dict = get_task_dict([task_name], task_manager)

    if task_name not in task_dict:
        sample = sorted(task_manager.all_tasks)[:20]
        raise ValueError(
            f"Task '{task_name}' not found in lm-eval registry.\n"
            f"Sample available tasks: {sample}\n"
            f"Full list: python -m lm_eval --tasks list"
        )

    task = task_dict[task_name]

    docs = None
    for split_getter in ("test_docs", "validation_docs", "train_docs"):
        if hasattr(task, split_getter):
            try:
                docs = list(getattr(task, split_getter)())
                if docs:
                    break
            except Exception:
                continue

    if not docs:
        raise RuntimeError(f"Task '{task_name}' returned no documents from any split.")

    prompts = []
    for doc in docs:
        if len(prompts) >= num_requests:
            break
        try:
            prompts.append(task.doc_to_text(doc))
        except Exception:
            if isinstance(doc, str):
                prompts.append(doc)

    if not prompts:
        raise RuntimeError(f"Task '{task_name}': could not convert any document to text.")
    return prompts


def build_prompts(lmeval_task: Optional[str], num_requests: int,
                  offline: bool = True) -> list:
    if lmeval_task:
        prompts = load_lmeval_prompts(lmeval_task, num_requests, offline=offline)
        print(f"[Prompts] Loaded {len(prompts)} docs from lm-eval '{lmeval_task}'")
    else:
        prompts = [_BUILTIN_PROMPTS[i % len(_BUILTIN_PROMPTS)] for i in range(num_requests)]
        print(f"[Prompts] Using {len(prompts)} built-in prompts")
    return prompts


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    request_id: int
    prompt_chars: int
    output_text: str
    output_tokens: int
    decode_latency_ms: float
    compression_mode: str


# ---------------------------------------------------------------------------
# Compression spec builder
# ---------------------------------------------------------------------------

def make_compression_spec(args) -> object:
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
    return None  # none


# ---------------------------------------------------------------------------
# Worker processes
# ---------------------------------------------------------------------------

def run_prefill(model, prefill_gpu, kv_port, prefill_done_event,
                gpu_mem_util, compression_spec, prompts):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(prefill_gpu)

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    kv_cfg = KVTransferConfig(
        kv_connector="CompressedKVConnector",
        kv_connector_module_path="kvserve_v1.connector.compressed_kv_connector",
        kv_role="kv_producer",
        kv_rank=0,
        kv_parallel_size=2,
        kv_ip="127.0.0.1",
        kv_port=kv_port,
        kv_connector_extra_config={"compression": compression_spec},
    )
    llm = LLM(
        model=model,
        kv_transfer_config=kv_cfg,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )
    llm.generate(prompts, sampling_params=SamplingParams(max_tokens=1, temperature=0))
    print("[Prefill] Done - KV sent.", flush=True)
    prefill_done_event.set()


def run_decode(model, decode_gpu, kv_port, prefill_done_event, result_queue,
               gpu_mem_util, compression_spec, prompts, max_tokens, mode_label):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(decode_gpu)

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    kv_cfg = KVTransferConfig(
        kv_connector="CompressedKVConnector",
        kv_connector_module_path="kvserve_v1.connector.compressed_kv_connector",
        kv_role="kv_consumer",
        kv_rank=1,
        kv_parallel_size=2,
        kv_ip="127.0.0.1",
        kv_port=kv_port,
        kv_connector_extra_config={"compression": compression_spec},
    )
    llm = LLM(
        model=model,
        kv_transfer_config=kv_cfg,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )

    print("[Decode] Engine ready, waiting for prefill...", flush=True)
    prefill_done_event.wait(timeout=300)

    t_start = time.monotonic()
    outputs = llm.generate(
        prompts, sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0))
    total_ms = (time.monotonic() - t_start) * 1e3
    per_req_ms = total_ms / max(len(prompts), 1)

    results = []
    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        r = RequestResult(
            request_id=i,
            prompt_chars=len(out.prompt),
            output_text=text,
            output_tokens=len(out.outputs[0].token_ids),
            decode_latency_ms=per_req_ms,
            compression_mode=mode_label,
        )
        results.append(r)
        print(f"[Decode] [{i}] {out.prompt[:60]!r}... -> {text!r}", flush=True)

    result_queue.put(results)
    print(f"[Decode] Done. total={total_ms:.0f}ms  avg/req={per_req_ms:.0f}ms", flush=True)


# ---------------------------------------------------------------------------
# CSV export + summary
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "request_id", "compression_mode", "prompt_chars",
    "output_tokens", "decode_latency_ms", "output_text",
]


def save_csv(results: list, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["output_text"] = row["output_text"][:200]
            writer.writerow({k: row[k] for k in _CSV_FIELDS})
    print(f"[Results] Saved {len(results)} rows -> {path}")


def print_summary(results: list) -> None:
    n = len(results)
    if n == 0:
        return
    avg_lat = sum(r.decode_latency_ms for r in results) / n
    avg_tok = sum(r.output_tokens for r in results) / n
    empty = sum(1 for r in results if not r.output_text.strip())
    print(f"\n{'='*60}")
    print(f"SUMMARY  (n={n}, mode={results[0].compression_mode})")
    print(f"{'='*60}")
    print(f"  Avg decode latency/req : {avg_lat:.1f} ms")
    print(f"  Avg output tokens/req  : {avg_tok:.1f}")
    print(f"  Empty outputs          : {empty}/{n}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PD separation test (real NCCL transport)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Prompt source
    parser.add_argument("--lmeval-task", default=None,
                        help="lm-eval task name (e.g. longbench_qasper, gsm8k). "
                             "Omit to use built-in long prompts.")
    parser.add_argument("--online", action="store_true", default=False,
                        help="Allow HuggingFace Hub access when loading lm-eval datasets. "
                             "By default datasets are loaded from local cache only.")
    parser.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS)
    parser.add_argument("--max-tokens", type=int, default=30,
                        help="Max new tokens per decode request")

    # Hardware
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--prefill-gpu", type=int, default=0)
    parser.add_argument("--decode-gpu", type=int, default=1)
    parser.add_argument("--kv-port", type=int, default=DEFAULT_KV_PORT)
    parser.add_argument("--gpu-mem-util", type=float, default=GPU_MEMORY_UTILIZATION)

    # Compression mode
    parser.add_argument("--mode",
                        choices=["none", "default", "custom", "controller"],
                        default="none",
                        help="Compression mode")

    # Controller-only options
    parser.add_argument("--library-path", default=None,
                        help="[controller] Profile library JSON path")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="[controller] epsilon-greedy exploration rate")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="[controller] EWMA learning rate")
    parser.add_argument("--bandwidth-mbps", type=float, default=1000.0,
                        help="[controller] Estimated bandwidth in MB/s")
    parser.add_argument("--slo-ms", type=float, default=200.0,
                        help="[controller] SLO budget in ms")
    parser.add_argument("--accuracy-req", type=float, default=0.92,
                        help="[controller] Required accuracy 0-1")
    parser.add_argument("--t-model-ms", type=float, default=0.0,
                        help="[controller] Estimated model compute latency in ms")

    # Output
    parser.add_argument("--output-dir", default=OUTPUT_DIR)

    args = parser.parse_args()

    compression_spec = make_compression_spec(args)
    prompts = build_prompts(args.lmeval_task, args.num_requests,
                            offline=not args.online)

    print(f"\n{'='*60}")
    print("PD SEPARATION TEST")
    print(f"{'='*60}")
    print(f"  Model        : {args.model}")
    src = ("lm-eval:" + args.lmeval_task) if args.lmeval_task else "built-in"
    print(f"  Prompts      : {len(prompts)} ({src})")
    print(f"  Compression  : {args.mode}")
    print(f"  Prefill GPU  : {args.prefill_gpu}")
    print(f"  Decode GPU   : {args.decode_gpu}")
    print(f"  KV port      : {args.kv_port}")
    print(f"{'='*60}\n")

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    prefill_done = manager.Event()
    result_queue = manager.Queue()

    p_prefill = mp.Process(
        target=run_prefill,
        args=(args.model, args.prefill_gpu, args.kv_port,
              prefill_done, args.gpu_mem_util, compression_spec, prompts),
    )
    p_decode = mp.Process(
        target=run_decode,
        args=(args.model, args.decode_gpu, args.kv_port,
              prefill_done, result_queue, args.gpu_mem_util,
              compression_spec, prompts, args.max_tokens, args.mode),
    )

    p_prefill.start()
    p_decode.start()

    results = None
    deadline = time.time() + 600
    while time.time() < deadline:
        if not result_queue.empty():
            results = result_queue.get()
            break
        if not p_decode.is_alive() and result_queue.empty():
            print("[Main] Decode process exited unexpectedly.", flush=True)
            break
        time.sleep(1)

    p_prefill.terminate()
    p_decode.terminate()
    p_prefill.join(timeout=10)
    p_decode.join(timeout=10)

    if not results:
        print("FAIL: no results received")
        os._exit(1)

    print_summary(results)

    csv_name = f"results_{args.mode}_{args.lmeval_task or 'builtin'}.csv"
    save_csv(results, os.path.join(args.output_dir, csv_name))

    n, expected = len(results), len(prompts)
    if n == expected:
        print(f"PASS: {n}/{expected} requests completed")
        os._exit(0)
    else:
        print(f"FAIL: {n}/{expected} completed")
        os._exit(1)


if __name__ == "__main__":
    main()
