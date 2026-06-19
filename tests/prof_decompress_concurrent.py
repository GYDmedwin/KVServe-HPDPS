"""Does concurrent multi-stream decompress speed up the consumer?

The consumer's ~467ms exposure is serial GPU decompress (32 layer-groups, one
after another on the default stream). The groups are independent (different
layers). If a single group's decompress underutilizes the GPU, running several
on parallel CUDA streams should shrink the total wall time.

This builds N layer-groups (each = M requests' quantized KV, one grouped codec),
then times decompress_layer_group: serial (default stream) vs K concurrent
streams. Local GPU; no cross-node.

Usage: PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python tests/prof_decompress_concurrent.py
"""
import argparse
import time

import torch

from kvserve_v1.compression.tilelang_manager import TileLangCompressionManager


def _evt():
    return torch.cuda.Event(enable_timing=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=32)   # layers
    ap.add_argument("--members", type=int, default=16)  # requests per layer-group
    ap.add_argument("--blocks", type=int, default=240)  # ~3840 tokens / bs=16
    ap.add_argument("--heads", type=int, default=8)     # Llama-8B KV heads
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    dev = "cuda"

    mgr = TileLangCompressionManager(
        num_kv_heads=args.heads, head_size=args.dim, base_seed=0x3333,
        quantizer_config={"model_name": None}, codec_config={"codec_type": "lc"},
        fused_variant="vec_shfl", throttle_gx=0)
    mgr.warmup(args.bs)
    mgr._decompress_no_sync = True  # let decompress be async for concurrency

    # Build N grouped CompressedKVData (each layer = M members' quantized KV).
    print(f"[build] {args.groups} groups x {args.members} members "
          f"(blocks={args.blocks}, heads={args.heads}) ...", flush=True)
    groups = []
    for g in range(args.groups):
        q_perms, cmetas, tids, nblocks = [], [], [], []
        for m in range(args.members):
            kv = torch.randn(2, args.blocks, args.bs, args.heads, args.dim,
                             dtype=torch.bfloat16, device=dev)
            q, cm = mgr.compress_quant_layer(kv, g)
            q_perms.append(q); cmetas.append(cm)
            tids.append(f"r{m}"); nblocks.append(int(q.shape[2]))
        cd = mgr.finalize_layer_group(q_perms, cmetas, tids, nblocks, g)
        groups.append(cd)
    torch.cuda.synchronize()
    total_mb = sum(cd.compressed_tensor.numel() for cd in groups) / 1e6
    print(f"[build] done, compressed total {total_mb:.0f}MB", flush=True)

    def serial():
        for cd in groups:
            mgr.decompress_layer_group(cd)

    def concurrent(nstreams):
        streams = [torch.cuda.Stream() for _ in range(nstreams)]
        for i, cd in enumerate(groups):
            with torch.cuda.stream(streams[i % nstreams]):
                mgr.decompress_layer_group(cd)
        for s in streams:
            s.synchronize()

    def timed(fn, iters=10, warmup=3):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t = time.monotonic()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.monotonic() - t) / iters * 1e3  # ms

    s_ms = timed(serial)
    print(f"\n  serial (1 stream)      : {s_ms:7.2f} ms")
    for ns in (2, 4, 8):
        try:
            c_ms = timed(lambda: concurrent(ns))
            print(f"  concurrent ({ns} streams) : {c_ms:7.2f} ms   "
                  f"speedup {s_ms/c_ms:.2f}x")
        except Exception as e:  # noqa: BLE001
            print(f"  concurrent ({ns} streams) : FAILED {e}")


if __name__ == "__main__":
    main()
