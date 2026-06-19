"""Microbenchmark: how much time does the LC codec actually cost?

Profiles LC encode/decode kernel time in isolation on a realistic quantized-KV
sized uint8 buffer, to decide whether fusing the codec into the quant kernel is
worth it. Reports:
  - batched (1 call over all layers)  vs  per-layer (N small calls)
  - raw kernel time (CUDA events)     vs  full method (incl. .item()+clone tax)
  - throughput GB/s

The codec input is the quantized buffer: [layers, 2, heads, blocks, bs, dim] uint8
(heads-outer layout, matching tilelang_manager.compress_all_layers).

Usage: python tests/bench_lc_codec.py --num-tokens 3840 --layers 28
"""

import argparse
import ctypes
import time

import torch

from kvserve_v1.compression.codec.lc_codec import LCCodec


def _evt():
    return torch.cuda.Event(enable_timing=True)


def time_kernel(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = _evt(), _evt()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tokens", type=int, default=3840)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    dev = "cuda"
    blocks = args.num_tokens // args.block_size
    # [layers, 2, heads, blocks, bs, dim] uint8 — the quantized buffer the codec sees.
    per_layer_shape = (2, args.heads, blocks, args.block_size, args.head_dim)
    per_layer_bytes = 1
    for s in per_layer_shape:
        per_layer_bytes *= s
    total_bytes = per_layer_bytes * args.layers
    print(f"[bench] tokens={args.num_tokens} layers={args.layers} "
          f"per-layer={per_layer_bytes/1e6:.2f}MB total={total_bytes/1e6:.2f}MB")

    # Realistic-ish low-entropy quantized data (mostly small values + zeros).
    torch.manual_seed(0)
    full = (torch.rand(args.layers, *per_layer_shape, device=dev) ** 4 * 16
            ).to(torch.uint8)  # skew toward 0 -> compressible like real quant KV

    codec = LCCodec()

    # ---- encode: full method (incl .item()+clone) ----
    def enc_batched_full():
        codec.encode(0, full)
    def enc_perlayer_full():
        for l in range(args.layers):
            codec.encode(l, full[l])

    # ---- encode: raw kernel only (no .item, no clone) ----
    d_in = full.reshape(-1).contiguous()
    insize = d_in.numel()
    maxsize = codec._enc.lc_maxsize(insize)
    chunks = codec._enc.lc_chunks(insize)
    bcount = codec._blocks_for(torch.device(dev))
    d_out = torch.empty(maxsize, dtype=torch.uint8, device=dev)
    d_outsize = torch.zeros(1, dtype=torch.int64, device=dev)
    d_fullcarry = torch.zeros(chunks, dtype=torch.int64, device=dev)
    def enc_batched_raw():
        stream = torch.cuda.current_stream(dev).cuda_stream
        codec._enc.lc_encode_stream(
            ctypes.c_void_p(stream), ctypes.c_void_p(d_in.data_ptr()),
            ctypes.c_longlong(insize), ctypes.c_void_p(d_out.data_ptr()),
            ctypes.c_void_p(d_outsize.data_ptr()),
            ctypes.c_void_p(d_fullcarry.data_ptr()), ctypes.c_int(bcount))

    # compressed blobs for decode bench
    comp_batched = codec.encode(0, full)
    comp_layers = [codec.encode(l, full[l]) for l in range(args.layers)]
    bshape = list(full.shape)
    lshape = list(per_layer_shape)
    def dec_batched_full():
        codec.decode(0, comp_batched, "uint8", bshape, dev)
    def dec_perlayer_full():
        for l in range(args.layers):
            codec.decode(l, comp_layers[l], "uint8", lshape, dev)

    print(f"\n[bench] ratio: batched {total_bytes/comp_batched.numel():.2f}x  "
          f"compressed={comp_batched.numel()/1e6:.2f}MB")

    gb = total_bytes / 1e9
    print("\n=== ENCODE ===")
    for name, fn in [("batched full (item+clone)", enc_batched_full),
                     ("batched raw kernel", enc_batched_raw),
                     ("per-layer full x%d" % args.layers, enc_perlayer_full)]:
        ms = time_kernel(fn, args.iters)
        print(f"  {name:30s}: {ms:7.3f} ms   {gb/(ms/1e3):6.1f} GB/s")

    print("\n=== DECODE ===")
    for name, fn in [("batched full", dec_batched_full),
                     ("per-layer full x%d" % args.layers, dec_perlayer_full)]:
        ms = time_kernel(fn, args.iters)
        print(f"  {name:30s}: {ms:7.3f} ms   {gb/(ms/1e3):6.1f} GB/s")


if __name__ == "__main__":
    main()
