"""Diagnose why the codec compresses compress_v3 output worse than the current
quantizer output: compare the pre-codec quantized byte entropy + unique values.
ANS ratio ~ 8 / entropy_bits_per_byte, so higher entropy -> worse codec ratio.
"""
import torch
from kvserve_v1.compression.manager import KVCompressionAdapter

MODEL = "Qwen2.5-7B-Instruct"
HEADS, DIM, BS = 4, 128, 16
QUANT = {"model_name": MODEL, "hybrid_ratio": 0.5, "high_key_max_value": 12,
         "high_value_max_value": 8, "low_key_max_value": 6, "low_value_max_value": 4,
         "axis_key": "channel", "axis_value": "token", "split_type": "head"}


def entropy_stats(u8: torch.Tensor):
    u8 = u8.flatten().to(torch.int64)
    counts = torch.bincount(u8, minlength=256).float()
    p = counts / counts.sum()
    nz = p[p > 0]
    ent = float(-(nz * nz.log2()).sum())
    nuniq = int((counts > 0).sum())
    return ent, nuniq


def split_keys_values(u8_kv):
    # u8_kv: [..., 2, ...] with dim-? actually we pass full quantized buffers.
    return u8_kv


def main():
    torch.manual_seed(0)
    layers, blocks = 28, (3840 + BS - 1) // BS
    kv = (torch.randn(layers, 2, blocks, BS, HEADS, DIM, dtype=torch.bfloat16, device="cuda") * 0.5)

    # --- current quantizer output (no codec) ---
    spec_cur = {"enabled": True, "pipeline": ["transformer", "quantizer"],
                "transformer_config": {"transform_type": "hadamard", "seed": 0x3333},
                "quantizer_config": dict(QUANT), "min_compress_size": 0}
    ad = KVCompressionAdapter(spec_cur, num_kv_heads=HEADS, head_size=DIM, model_name=MODEL)
    comp = ad.compress(kv.clone(), "r")
    cur_u8 = comp.compressed_tensor.flatten().view(torch.uint8)
    e_cur, n_cur = entropy_stats(cur_u8)

    # --- compress_v3 output (no codec) ---
    from kvserve_v1.tilelang_ops.kv_hadamard_quant import KVHadamardQuantOp
    op = KVHadamardQuantOp(num_heads=HEADS, head_dim=DIM, base_seed=0x3333,
                           dtype=torch.bfloat16, quant_type="minmax", **QUANT)
    qk_all, qv_all = [], []
    for l in range(layers):
        q, _ = op.compress_v3(kv[l].contiguous(), layer_id=l)  # [2,blocks,bs,heads,dim] u8
        qk_all.append(q[0]); qv_all.append(q[1])
    v3_keys = torch.stack(qk_all).flatten()
    v3_vals = torch.stack(qv_all).flatten()
    v3_all = torch.cat([v3_keys, v3_vals])
    e_v3, n_v3 = entropy_stats(v3_all)
    e_vk, n_vk = entropy_stats(v3_keys)
    e_vv, n_vv = entropy_stats(v3_vals)

    def line(name, e, n):
        print(f"  {name:18s} entropy={e:.3f} bits/byte  unique={n:3d}  "
              f"ideal_codec_ratio={8.0/max(e,1e-6):.2f}x")

    print("Pre-codec quantized byte distribution:")
    line("current(K+V)", e_cur, n_cur)
    line("compress_v3(K+V)", e_v3, n_v3)
    line("  v3 keys", e_vk, n_vk)
    line("  v3 values", e_vv, n_vv)


if __name__ == "__main__":
    main()
