"""Benchmark the fused TileLang compress_v3 (Hadamard transform + quantize)
against the current per-layer transform+quantize stage.

compress_v3 fuses what the current pipeline does as a Python per-layer loop of
KVServeTransformer.transform + KVServeQuantizer.quantize. It does NOT include
the codec (entropy coding), so the fair comparison is:

    compress_v3 time   vs   current t_quant_ms   (the ~804ms quant stage)

Hadamard parameters are aligned to the FULL pipeline: base_seed=0x3333,
scale=1/sqrt(head_dim), and the same Rademacher sign scheme. Quantizer config
(hybrid ratio / bit-widths / axes) is aligned to CUSTOM/FULL.

Usage:
    python tests/bench_compress_v3.py --num-tokens 3840 --layers 28
"""

import argparse
import time

import torch

MODEL_NAME = "Qwen2.5-7B-Instruct"
SEED = 0x3333
NUM_KV_HEADS = 4
HEAD_DIM = 128
BLOCK_SIZE = 16

# Aligned with FULL_COMPRESSION_CFG in tests/test_pd_cross_node.py
QUANT_CFG = {
    "model_name": MODEL_NAME,
    "hybrid_ratio": 0.5,
    "high_key_max_value": 12,
    "high_value_max_value": 8,
    "low_key_max_value": 6,
    "low_value_max_value": 4,
    "axis_key": "channel",
    "axis_value": "token",
    "split_type": "head",
}


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_kv(layers, blocks, device):
    # [layers, 2, blocks, block_size, heads, head_dim] bf16, realistic-ish.
    shape = (layers, 2, blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    return (torch.randn(shape, dtype=torch.bfloat16, device=device) * 0.5)


def bench_current(stacked):
    """Current pipeline: returns (quant_ms, codec_ms, decompress breakdown, ratio)."""
    from kvserve_v1.compression.manager import KVCompressionAdapter
    spec = {
        "enabled": True,
        "pipeline": ["transformer", "quantizer", "codec"],
        "transformer_config": {"transform_type": "hadamard", "seed": SEED},
        "quantizer_config": dict(QUANT_CFG),
        "codec_config": {"codec_type": "nvcomp", "nvcomp_algorithm": "ANS",
                         "data_type": "|u1"},
        "min_compress_size": 0,
    }
    adapter = KVCompressionAdapter(spec, num_kv_heads=NUM_KV_HEADS,
                                   head_size=HEAD_DIM, model_name=MODEL_NAME)
    # warmup
    _ = adapter.compress(stacked.clone(), "warmup")
    _sync()
    t0 = time.monotonic()
    compressed = adapter.compress(stacked.clone(), "bench")
    _sync()
    total_ms = (time.monotonic() - t0) * 1e3
    meta = compressed.metadata
    # closed-loop accuracy on the same data
    restored = adapter.decompress(compressed)
    cos = torch.cosine_similarity(stacked.float().flatten(),
                                  restored.float().flatten(), dim=0).item()
    return {
        "total_ms": total_ms,
        "quant_ms": float(meta.get("t_quant_ms", 0.0)),
        "codec_ms": float(meta.get("t_codec_ms", 0.0)),
        "orig_mb": stacked.numel() * stacked.element_size() / 1e6,
        "comp_mb": compressed.compressed_size / 1e6,
        "ratio": (stacked.numel() * stacked.element_size()) / max(compressed.compressed_size, 1),
        "cosine": cos,
    }


def bench_fused(stacked):
    """Fused compress_v3 per layer: returns timing + closed-loop accuracy."""
    from kvserve_v1.tilelang_ops.kv_hadamard_quant import KVHadamardQuantOp
    op = KVHadamardQuantOp(
        num_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, base_seed=SEED,
        dtype=torch.bfloat16, quant_type="minmax", **QUANT_CFG)
    layers = stacked.shape[0]

    # warmup (triggers lazy kernel compilation)
    q0, m0 = op.compress_v3(stacked[0], layer_id=0)
    _ = op.decompress_v3(q0, m0, layer_id=0)
    _sync()

    # compress
    t0 = time.monotonic()
    qs, ms = [], []
    out_bytes = 0
    for l in range(layers):
        q, m = op.compress_v3(stacked[l], layer_id=l)
        qs.append(q); ms.append(m); out_bytes += q.numel() * q.element_size()
    _sync()
    compress_ms = (time.monotonic() - t0) * 1e3

    # decompress
    t1 = time.monotonic()
    restored = [op.decompress_v3(qs[l], ms[l], layer_id=l) for l in range(layers)]
    _sync()
    decompress_ms = (time.monotonic() - t1) * 1e3

    # closed-loop accuracy
    orig = stacked.float().flatten()
    rec = torch.stack(restored).float().flatten()
    cos = torch.cosine_similarity(orig, rec, dim=0).item()
    mse = torch.nn.functional.mse_loss(orig, rec).item()
    return {
        "compress_ms": compress_ms,
        "decompress_ms": decompress_ms,
        "quant_out_mb": out_bytes / 1e6,  # pre-codec uint8 size
        "cosine": cos,
        "mse": mse,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-tokens", type=int, default=3840)
    p.add_argument("--layers", type=int, default=28)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    blocks = (args.num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"KV shape: [{args.layers}, 2, {blocks}, {BLOCK_SIZE}, {NUM_KV_HEADS}, {HEAD_DIM}]"
          f"  (~{args.num_tokens} tokens)  seed=0x{SEED:x}")

    stacked = build_kv(args.layers, blocks, device)
    orig_mb = stacked.numel() * stacked.element_size() / 1e6
    print(f"KV size: {orig_mb:.1f} MB bf16\n")

    print("=== Current pipeline (transformer+quantizer+codec) ===")
    cur = bench_current(stacked)
    print(f"  transform+quantize (t_quant_ms): {cur['quant_ms']:.1f} ms")
    print(f"  codec (t_codec_ms)             : {cur['codec_ms']:.1f} ms")
    print(f"  compress total                 : {cur['total_ms']:.1f} ms")
    print(f"  ratio (after codec)            : {cur['ratio']:.2f}x  "
          f"({cur['orig_mb']:.1f} -> {cur['comp_mb']:.1f} MB)")
    print(f"  closed-loop cosine sim         : {cur['cosine']:.6f}\n")

    print("=== Fused compress_v3 (Hadamard+quantize, NO codec) ===")
    fused = bench_fused(stacked)
    print(f"  compress_v3 total              : {fused['compress_ms']:.1f} ms")
    print(f"  decompress_v3 total            : {fused['decompress_ms']:.1f} ms")
    print(f"  quant output (pre-codec)       : {fused['quant_out_mb']:.1f} MB")
    print(f"  closed-loop cosine sim         : {fused['cosine']:.6f}")
    print(f"  closed-loop MSE                : {fused['mse']:.6e}\n")

    print("=== Comparison (quant stage) ===")
    speedup = cur['quant_ms'] / max(fused['compress_ms'], 1e-6)
    print(f"  current transform+quantize : {cur['quant_ms']:.1f} ms")
    print(f"  fused compress_v3          : {fused['compress_ms']:.1f} ms")
    print(f"  speedup                    : {speedup:.2f}x")


if __name__ == "__main__":
    main()
