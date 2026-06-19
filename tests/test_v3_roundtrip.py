"""Single-node round-trip test for the full_v3 (TileLang compress_v3) path,
including the GPU-resident wire serialization (the risky metadata part).

Validates: compress -> build_wire -> restore_from_wire -> decompress reproduces
the KV (closed-loop cosine), and reports compression ratio. Compares full_v3
against full on the same synthetic KV.
"""
import time
import torch

from kvserve_v1.compression.manager import KVCompressionAdapter
from kvserve_v1.compression.wire import build_wire, restore_from_wire

MODEL = "Qwen2.5-7B-Instruct"
HEADS, DIM, BS = 4, 128, 16
QUANT = {"model_name": MODEL, "hybrid_ratio": 0.5, "high_key_max_value": 12,
         "high_value_max_value": 8, "low_key_max_value": 6, "low_value_max_value": 4,
         "axis_key": "channel", "axis_value": "token", "split_type": "head"}
CODEC = {"codec_type": "nvcomp", "nvcomp_algorithm": "ANS", "data_type": "|u1"}

FULL = {"enabled": True, "pipeline": ["transformer", "quantizer", "codec"],
        "transformer_config": {"transform_type": "hadamard", "seed": 0x3333},
        "quantizer_config": dict(QUANT), "codec_config": dict(CODEC), "min_compress_size": 0}
FULL_V3 = {**FULL, "backend": "tilelang"}


def run(spec, name, stacked):
    adapter = KVCompressionAdapter(spec, num_kv_heads=HEADS, head_size=DIM, model_name=MODEL)
    _ = adapter.compress(stacked.clone(), "warmup")  # warmup/compile
    torch.cuda.synchronize()
    t0 = time.monotonic()
    compressed = adapter.compress(stacked.clone(), "r0")
    torch.cuda.synchronize()
    comp_ms = (time.monotonic() - t0) * 1e3

    wire = build_wire(compressed, 512 * 1024 * 1024)
    restored_wire = restore_from_wire(wire)

    t1 = time.monotonic()
    out = adapter.decompress(restored_wire)
    torch.cuda.synchronize()
    dec_ms = (time.monotonic() - t1) * 1e3

    cos = torch.cosine_similarity(stacked.float().flatten(), out.float().flatten(), dim=0).item()
    ratio = (stacked.numel() * stacked.element_size()) / max(wire.nbytes, 1)
    body_bytes = sum(t.numel() * t.element_size() for t in wire.body_chunks)
    aux_bytes = sum(t.numel() * t.element_size() for t in wire.aux_tensors)
    print(f"[{name}] compress={comp_ms:.1f}ms decompress={dec_ms:.1f}ms "
          f"wire={wire.nbytes/1e6:.1f}MB (codec_body={body_bytes/1e6:.1f}MB "
          f"meta_aux={aux_bytes/1e6:.1f}MB) ratio={ratio:.2f}x cosine={cos:.6f} "
          f"aux_tensors={len(wire.aux_tensors)} shape_ok={list(out.shape)==list(stacked.shape)}")
    return cos, ratio


if __name__ == "__main__":
    torch.manual_seed(0)
    layers, blocks = 28, (3840 + BS - 1) // BS
    stacked = (torch.randn(layers, 2, blocks, BS, HEADS, DIM, dtype=torch.bfloat16, device="cuda") * 0.5)
    print(f"KV {list(stacked.shape)} = {stacked.numel()*stacked.element_size()/1e6:.1f} MB\n")
    run(FULL, "full   ", stacked)
    run(FULL_V3, "full_v3", stacked)
    print("\nOK" )
