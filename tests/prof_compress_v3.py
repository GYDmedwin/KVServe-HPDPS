"""Profile compress_v3 to decide the 'bandwidth-friendly kernel' direction.

The exposed "compute-over-window" bucket (~417ms @Llama-8B/batch64) is NOT SM
contention (throttle ruled it out). Prime suspect: HBM bandwidth — compress_v3
moves a lot of bytes (an intermediate bf16 `had_out` between FWHT and quantize,
plus the heads-outer `permute.contiguous()`), so it competes with the forward for
memory bandwidth and can't fully hide.

This measures, on the producer GPU, for one (layer, request) of Llama-8B dims:
  - reference d2d copy bandwidth (practical peak)
  - op.compress_v3 time + achieved effective HBM BW (vs peak)
  - the permute.contiguous() time + BW
  - the theoretical byte reduction from fusing had_out + dropping permute

Usage: PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python tests/prof_compress_v3.py
"""
import argparse
import time

import torch

from kvserve_v1.tilelang_ops.kv_hadamard_quant_v2 import KVHadamardQuantOp


def _evt():
    return torch.cuda.Event(enable_timing=True)


def timed(fn, iters=50, warmup=10):
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
    ap.add_argument("--rows", type=int, default=3840)   # one request's tokens
    ap.add_argument("--heads", type=int, default=8)     # Llama-3-8B KV heads
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=16)
    args = ap.parse_args()

    dev = "cuda"
    rows = args.rows
    blocks = rows // args.block_size
    H, D, BS = args.heads, args.dim, args.block_size
    kv = torch.randn(2, blocks, BS, H, D, dtype=torch.bfloat16, device=dev)
    print(f"[prof] KV layer [2,{blocks},{BS},{H},{D}] bf16 = "
          f"{kv.numel()*2/1e6:.1f}MB  rows={rows}")

    # practical peak BW via a big d2d copy
    big = torch.empty(256 * 1024 * 1024 // 2, dtype=torch.bfloat16, device=dev)
    src = torch.randn_like(big)
    cp_ms = timed(lambda: big.copy_(src), iters=30)
    peak = (2 * big.numel() * 2 / 1e9) / (cp_ms / 1e3)   # read+write
    print(f"[prof] practical peak BW (d2d copy) = {peak:.0f} GB/s")

    op = KVHadamardQuantOp(
        num_heads=H, head_dim=D, base_seed=0x3333, dtype=torch.bfloat16,
        quant_type="minmax", model_name=None, hybrid_ratio=0.5,
        high_key_max_value=12, high_value_max_value=8, low_key_max_value=6,
        low_value_max_value=4, axis_key="channel", axis_value="token",
        split_type="head", fused_variant="vec_shfl", throttle_gx=0)

    # warmup (JIT compile)
    for _ in range(3):
        q, meta = op.compress_v3(kv, layer_id=0)
        q.permute(0, 3, 1, 2, 4).contiguous()
    torch.cuda.synchronize()

    op_ms = timed(lambda: op.compress_v3(kv, layer_id=0))
    q0, _ = op.compress_v3(kv, layer_id=0)
    perm_ms = timed(lambda: q0.permute(0, 3, 1, 2, 4).contiguous())

    base = rows * H * D  # element count per (K or V) tensor, ×2 for both
    # current HBM bytes (estimate): per K/V path
    #   read in bf16(2) + write had_out bf16(2) + read had_out bf16(2) + write q uint8(1)
    cur_compress = 2 * base * (2 + 2 + 2 + 1)          # K and V
    perm_bytes = 2 * base * (1 + 1)                    # read q + write q_perm (uint8)
    fused_bytes = 2 * base * (2 + 1)                   # read in bf16 + write q_perm uint8
    op_bw = (cur_compress / 1e9) / (op_ms / 1e3)
    perm_bw = (perm_bytes / 1e9) / (perm_ms / 1e3)

    print(f"\n=== compress_v3 op (4 kernels: FWHT+reduce, quantize ×2) ===")
    print(f"  time        = {op_ms:.3f} ms")
    print(f"  est HBM bytes= {cur_compress/1e6:.1f} MB  → eff BW {op_bw:.0f} GB/s "
          f"({100*op_bw/peak:.0f}% of peak)")
    print(f"\n=== permute.contiguous() (heads-outer) ===")
    print(f"  time        = {perm_ms:.3f} ms  → eff BW {perm_bw:.0f} GB/s "
          f"({100*perm_bw/peak:.0f}% of peak)")
    print(f"  permute / (op+permute) = {100*perm_ms/(op_ms+perm_ms):.0f}% of compress time")

    print(f"\n=== fusion potential (drop had_out + drop permute) ===")
    print(f"  current bytes (op+permute) = {(cur_compress+perm_bytes)/1e6:.1f} MB")
    print(f"  fused bytes               = {fused_bytes/1e6:.1f} MB")
    print(f"  reduction                 = "
          f"{100*(1-fused_bytes/(cur_compress+perm_bytes)):.0f}%")
    print(f"  total compress time now   = {op_ms+perm_ms:.3f} ms/layer")
    print(f"  (×32 layers ×... → compare to the 417ms over-window bucket)")


if __name__ == "__main__":
    main()
