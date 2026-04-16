"""End-to-end PD separation test using vLLM V1 + CompressedKVConnector.

Two processes:
  - Prefill (producer): runs on prefill_gpu, prefills prompts, sends KV via NCCL.
  - Decode  (consumer): runs on decode_gpu, waits for KV, runs decode.

Both use vllm.LLM (synchronous) — consistent with vLLM's official examples.
Chunked prefill is disabled to avoid multi-step KV accumulation complexity.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

PROMPTS = [
    "The capital of France is",
    "Machine learning is a branch of",
    "The first human to walk on the moon was",
]


def _shutdown_vllm(llm) -> None:
    """Best-effort shutdown so EngineCore subprocesses exit with the worker."""
    if llm is None:
        return
    try:
        engine_core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        pass


def run_prefill(model: str, prefill_gpu: int, kv_port: int,
                prefill_done_event, gpu_mem_util: float):
    # Must set CUDA_VISIBLE_DEVICES before any CUDA/torch/vllm import
    os.environ["CUDA_VISIBLE_DEVICES"] = str(prefill_gpu)

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    kv_cfg = KVTransferConfig(
        kv_connector="CompressedKVConnector",
        kv_connector_module_path=(
            "kvserve_v1.connector.compressed_kv_connector"),
        kv_role="kv_producer",
        kv_rank=0,
        kv_parallel_size=2,
        kv_ip="127.0.0.1",
        kv_port=kv_port,
    )

    llm = LLM(
        model=model,
        kv_transfer_config=kv_cfg,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=512,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )
    try:
        sampling_params = SamplingParams(max_tokens=1, temperature=0)
        llm.generate(PROMPTS, sampling_params=sampling_params)
        print("[Prefill] Done generating — KV sent via NCCL.", flush=True)
        prefill_done_event.set()
    finally:
        _shutdown_vllm(llm)


def run_decode(model: str, decode_gpu: int, kv_port: int,
               prefill_done_event, result_queue, gpu_mem_util: float):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(decode_gpu)

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    kv_cfg = KVTransferConfig(
        kv_connector="CompressedKVConnector",
        kv_connector_module_path=(
            "kvserve_v1.connector.compressed_kv_connector"),
        kv_role="kv_consumer",
        kv_rank=1,
        kv_parallel_size=2,
        kv_ip="127.0.0.1",
        kv_port=kv_port,
    )

    llm = LLM(
        model=model,
        kv_transfer_config=kv_cfg,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=512,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )
    try:
        print("[Decode] Engine ready, waiting for prefill to finish…", flush=True)
        prefill_done_event.wait(timeout=300)

        sampling_params = SamplingParams(max_tokens=20, temperature=0)
        outputs = llm.generate(PROMPTS, sampling_params=sampling_params)

        results = []
        for out in outputs:
            text = out.outputs[0].text
            results.append(text)
            print(f"[Decode] prompt: {out.prompt!r}  →  {text!r}", flush=True)

        result_queue.put(results)
        print("[Decode] Done.", flush=True)
    finally:
        _shutdown_vllm(llm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default="/root/data/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--prefill-gpu", type=int, default=0)
    parser.add_argument("--decode-gpu", type=int, default=1)
    parser.add_argument("--kv-port", type=int, default=25001)
    parser.add_argument("--gpu-mem-util", type=float, default=0.6)
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    manager = None
    p_prefill = p_decode = None
    exit_code = 1

    try:
        manager = mp.Manager()
        prefill_done = manager.Event()
        result_queue = manager.Queue()

        p_prefill = mp.Process(
            target=run_prefill,
            args=(args.model, args.prefill_gpu, args.kv_port,
                  prefill_done, args.gpu_mem_util),
        )
        p_decode = mp.Process(
            target=run_decode,
            args=(args.model, args.decode_gpu, args.kv_port,
                  prefill_done, result_queue, args.gpu_mem_util),
        )

        p_prefill.start()
        p_decode.start()

        deadline = time.time() + 600
        results = None
        while time.time() < deadline:
            if not result_queue.empty():
                results = result_queue.get()
                break
            if not p_decode.is_alive() and result_queue.empty():
                print("[Main] Decode process exited unexpectedly.", flush=True)
                break
            time.sleep(1)

        if results is None:
            print("✗ FAIL: no results received", flush=True)
        else:
            n = len(results)
            expected = len(PROMPTS)
            if n == expected:
                print(f"✓ PASS: {n}/{expected} decode requests completed",
                      flush=True)
                exit_code = 0
            else:
                print(f"✗ FAIL: only {n}/{expected} decode requests completed",
                      flush=True)
    finally:
        for p in (p_prefill, p_decode):
            if p is not None and p.is_alive():
                p.terminate()
        for p in (p_prefill, p_decode):
            if p is not None:
                p.join(timeout=30)
        for p in (p_prefill, p_decode):
            if p is not None and p.is_alive():
                p.kill()
                p.join(timeout=10)
        if manager is not None:
            try:
                manager.shutdown()
            except Exception:
                pass

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
