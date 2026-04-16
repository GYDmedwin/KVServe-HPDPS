"""End-to-end PD separation test with KV compression enabled.

Three compression modes are supported via --mode:
  custom      (default) explicit pipeline config dict
  default     built-in DEFAULT_COMPRESSION_CONFIG from kvserve
  controller  online adaptive selection via OnlineController + ProfileLibrary
              (requires --library-path pointing to a profiles JSON file)

Usage examples:
  python test_pd_with_compression.py --mode custom
  python test_pd_with_compression.py --mode default
  python test_pd_with_compression.py --mode controller \\
      --library-path /path/to/profiles.json \\
      --bandwidth-mbps 1000 --slo-ms 200 --accuracy-req 0.92
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
import importlib.util

PROMPTS = [
    (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to "
        "the natural intelligence displayed by animals including humans. AI research has been "
        "defined as the field of study of intelligent agents, which refers to any system that "
        "perceives its environment and takes actions that maximize its chance of achieving its "
        "goals. The term 'artificial intelligence' had previously been used to describe machines "
        "that mimic and display human cognitive skills associated with the human mind, such as "
        "learning and problem-solving. This definition has since been rejected by major AI "
        "researchers who now describe AI in terms of rationality and acting rationally, which "
        "does not limit how intelligence can be articulated. AI applications include advanced "
        "web search engines, recommendation systems, understanding human speech, self-driving "
        "cars, generative or creative tools, and competing at the highest level in strategic "
        "games. As machines become increasingly capable, tasks considered to require intelligence "
        "are often removed from the definition of AI, a phenomenon known as the AI effect. "
        "In summary, artificial intelligence refers to"
    ),
    (
        "The Python programming language was created by Guido van Rossum and first released in "
        "1991. Python is designed to be easy to read and simple to implement. It is open source, "
        "which means it is free to use, even for commercial applications. Python is a general "
        "purpose programming language that is becoming increasingly popular for data science. "
        "Python supports multiple programming paradigms, including structured, object-oriented "
        "and functional programming. Python is often described as a 'batteries included' language "
        "due to its comprehensive standard library. The language provides constructs that enable "
        "clear programming on both small and large scales. Python features a dynamic type system "
        "and automatic memory management. It supports multiple programming paradigms, including "
        "object-oriented, imperative, functional and procedural, and has a large and comprehensive "
        "standard library. Python interpreters are available for many operating systems. A global "
        "community of programmers develops and maintains CPython, an open source reference "
        "implementation. In conclusion, Python is"
    ),
    (
        "The theory of evolution by natural selection was proposed by Charles Darwin and Alfred "
        "Russel Wallace in the 19th century. Darwin published his theory in his 1859 book 'On "
        "the Origin of Species'. Evolution explains the diversity of life on Earth. Natural "
        "selection is the process by which organisms with favorable traits are more likely to "
        "survive and reproduce. Over time, this leads to populations becoming better adapted to "
        "their environments. Darwin observed variations in animals across the Galapagos Islands "
        "that provided key evidence for his theory. The fossil record also supports evolution, "
        "showing gradual changes in organisms over millions of years. Genetic science has further "
        "confirmed Darwin's insights, revealing how traits are inherited and how mutations drive "
        "variation. Evolution is now the unifying theory of biology, explaining everything from "
        "antibiotic resistance to the development of complex organs. In essence, evolution by "
        "natural selection demonstrates that"
    ),
]

# ── Compression spec builders ───────────────────────────────────────────────

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


def _has_nvcomp_stack() -> bool:
    """Check whether nvcomp runtime dependencies are available."""
    return (
        importlib.util.find_spec("cupy") is not None
        and importlib.util.find_spec("nvidia.nvcomp") is not None
    )


def make_compression_spec(args) -> object:
    """Return the compression spec for kv_connector_extra_config["compression"]."""
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

    # custom (default): gracefully degrade when nvcomp deps are unavailable.
    if _has_nvcomp_stack():
        return CUSTOM_COMPRESSION_CFG

    degraded = dict(CUSTOM_COMPRESSION_CFG)
    degraded["pipeline"] = ["quantizer"]
    degraded.pop("codec_config", None)
    print(
        "[Main] cupy/nvcomp not found; falling back to custom quantizer-only mode.",
        flush=True,
    )
    return degraded


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


# ── Worker functions ────────────────────────────────────────────────────────

def run_prefill(model, prefill_gpu, kv_port, prefill_done_event, gpu_mem_util,
                compression_spec):
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
        max_model_len=1024,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )
    try:
        sampling_params = SamplingParams(max_tokens=1, temperature=0)
        llm.generate(PROMPTS, sampling_params=sampling_params)
        print("[Prefill] Done — compressed KV sent.", flush=True)
        prefill_done_event.set()
    finally:
        _shutdown_vllm(llm)


def run_decode(model, decode_gpu, kv_port, prefill_done_event, result_queue,
               gpu_mem_util, compression_spec):
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
        max_model_len=1024,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )
    try:
        print("[Decode] Engine ready, waiting for prefill…", flush=True)
        prefill_done_event.wait(timeout=300)

        sampling_params = SamplingParams(max_tokens=30, temperature=0)
        outputs = llm.generate(PROMPTS, sampling_params=sampling_params)

        results = []
        for out in outputs:
            text = out.outputs[0].text
            results.append(text)
            print(f"[Decode] {out.prompt!r}  ->  {text!r}", flush=True)

        result_queue.put(results)
        print("[Decode] Done.", flush=True)
    finally:
        _shutdown_vllm(llm)


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/gyd/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--prefill-gpu", type=int, default=0)
    parser.add_argument("--decode-gpu", type=int, default=1)
    parser.add_argument("--kv-port", type=int, default=25002)
    parser.add_argument("--gpu-mem-util", type=float, default=0.7)

    # Compression mode
    parser.add_argument("--mode", choices=["custom", "default", "controller"],
                        default="custom",
                        help="Compression mode: custom | default | controller")

    # Controller-mode options
    parser.add_argument("--library-path", default=None,
                        help="[controller] Path to profile library JSON")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="[controller] ε-greedy exploration rate")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="[controller] EWMA learning rate")
    parser.add_argument("--bandwidth-mbps", type=float, default=1000.0,
                        help="[controller] Estimated network bandwidth in MB/s")
    parser.add_argument("--slo-ms", type=float, default=200.0,
                        help="[controller] SLO latency budget in ms")
    parser.add_argument("--accuracy-req", type=float, default=0.92,
                        help="[controller] Required accuracy (0–1)")
    parser.add_argument("--t-model-ms", type=float, default=0.0,
                        help="[controller] Estimated model compute latency in ms")

    args = parser.parse_args()

    compression_spec = make_compression_spec(args)
    print(f"[Main] Compression mode: {args.mode}  spec={compression_spec!r}", flush=True)

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
                  prefill_done, args.gpu_mem_util, compression_spec),
        )
        p_decode = mp.Process(
            target=run_decode,
            args=(args.model, args.decode_gpu, args.kv_port,
                  prefill_done, result_queue, args.gpu_mem_util, compression_spec),
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
            print("FAIL: no results received", flush=True)
        else:
            n = len(results)
            expected = len(PROMPTS)
            if n == expected:
                print(f"PASS: {n}/{expected} compressed decode requests completed",
                      flush=True)
                exit_code = 0
            else:
                print(f"FAIL: only {n}/{expected} completed", flush=True)
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
