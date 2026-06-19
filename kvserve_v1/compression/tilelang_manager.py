"""TileLang-fused compression path: compress_v3 (Hadamard+quantize) + codec.

Drop-in alternative to CompressionManager for the FULL pipeline that swaps the
per-layer transform+quantize loop for the fused TileLang compress_v3 kernel,
then runs the same codec (nvCOMP) for entropy coding.

Exposes the same surface the connector/adapter use:
  compress_all_layers(all_layers_data, request_id, config, metadata) -> CompressedKVData
  decompress_all_layers(compressed_data, config) -> torch.Tensor
  self._last_decompress_timing = {"t_codec_decode_ms", "t_dequant_ms"}

Metadata is canonicalized to be wire-safe: all tensors live under
`quantization_params` (which compression.wire packs into NCCL aux tensors) and
there are no float dict keys. Only key_meta/value_meta travel — decompress_v3
does not use the max_vals.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import torch
from vllm.logger import init_logger

from kvserve_v1.compression.compression_manager import CompressedKVData

logger = init_logger(__name__)


def _f16(t: Any) -> Any:
    """Shrink float32 quant metadata to fp16 for the wire. decompress_v3
    re-casts scales to bf16 anyway, so this is effectively lossless."""
    if isinstance(t, torch.Tensor) and t.dtype == torch.float32:
        return t.to(torch.float16)
    return t


def _canon_meta(meta: dict) -> dict:
    """compress_v3 meta -> wire-safe form (tensors only, no float keys)."""
    km = dict(meta["key_meta"])
    for k in ("min_val", "quant_scale"):
        if k in km:
            km[k] = _f16(km[k])
    out: dict[str, Any] = {"key_meta": km}
    vm = meta["value_meta"]
    if "per_group_meta" in vm:
        groups = [
            {"head_idx": g["head_idx"], "min_val": _f16(g["min_val"]),
             "quant_scale": _f16(g["quant_scale"])}
            for g in vm["per_group_meta"].values()
        ]
        out["value_meta"] = {"per_group_list": groups}
    else:
        out["value_meta"] = dict(vm)
    return out


def _decanon_meta(c: dict) -> dict:
    """wire-safe form -> meta dict accepted by decompress_v3."""
    out: dict[str, Any] = {"key_meta": c["key_meta"]}
    vm = c["value_meta"]
    if "per_group_list" in vm:
        out["value_meta"] = {
            "per_group_meta": {i: g for i, g in enumerate(vm["per_group_list"])}
        }
    else:
        out["value_meta"] = vm
    return out


class TileLangCompressionManager:
    def __init__(self, num_kv_heads: int, head_size: int,
                 quantizer_config: Optional[dict], codec_config: Optional[dict],
                 base_seed: int = 0x3333, fused_variant: Optional[str] = None,
                 throttle_gx: int = 0):
        from kvserve_v1.compression.codec import KVServeCodec
        from kvserve_v1.compression.codec.lc_codec import LCCodec
        # The v2 op adds the vec_shfl kernel variant + persistent-grid throttle.
        # Use it whenever a variant or throttle is requested; otherwise the
        # original op (keeps the validated default full_v3 path byte-identical).
        if fused_variant or throttle_gx:
            from kvserve_v1.tilelang_ops.kv_hadamard_quant_v2 import KVHadamardQuantOp
        else:
            from kvserve_v1.tilelang_ops.kv_hadamard_quant import KVHadamardQuantOp

        qc = dict(quantizer_config or {})
        op_kwargs = dict(
            num_heads=num_kv_heads,
            head_dim=head_size,
            base_seed=base_seed,
            dtype=torch.bfloat16,
            quant_type="minmax",
            model_name=qc.get("model_name"),
            hybrid_ratio=qc.get("hybrid_ratio", 0.5),
            high_key_max_value=qc.get("high_key_max_value", 12),
            high_value_max_value=qc.get("high_value_max_value", 8),
            low_key_max_value=qc.get("low_key_max_value", 6),
            low_value_max_value=qc.get("low_value_max_value", 4),
            axis_key=qc.get("axis_key", "channel"),
            axis_value=qc.get("axis_value", "token"),
            split_type=qc.get("split_type", "head"),
        )
        if fused_variant or throttle_gx:
            op_kwargs["fused_variant"] = fused_variant or "vec_shfl"
            op_kwargs["throttle_gx"] = throttle_gx
        self.op = KVHadamardQuantOp(**op_kwargs)
        self.codec_config = dict(codec_config or {})
        # Open-source LC codec (codec_type="lc") vs closed nvCOMP. LC is faster
        # and stream-async-friendly (no internal syncs blocking overlap).
        if self.codec_config.get("codec_type") == "lc":
            self.codec = LCCodec(**self.codec_config)
            logger.info("[TileLangCompressionManager] codec=LC (open-source)")
        else:
            self.codec = KVServeCodec(**self.codec_config)
        self._last_decompress_timing: Optional[dict] = None
        self._num_kv_heads = num_kv_heads
        self._head_size = head_size
        logger.info("[TileLangCompressionManager] ready: heads=%d head_dim=%d "
                    "seed=0x%x", num_kv_heads, head_size, base_seed)

    def warmup(self, block_size: int = 16) -> None:
        """Trigger TileLang JIT compilation of the compress_v3/decompress_v3
        kernels off the request critical path (the first real compress otherwise
        pays ~0.6s of one-time compilation)."""
        if not torch.cuda.is_available():
            return
        try:
            t0 = time.monotonic()
            dummy = torch.randn(2, 1, block_size, self._num_kv_heads,
                                self._head_size, dtype=torch.bfloat16, device="cuda")
            q, meta = self.op.compress_v3(dummy, layer_id=0)
            self.op.decompress_v3(q, meta, layer_id=0)
            torch.cuda.synchronize()
            logger.info("[TileLangCompressionManager] kernel warmup done in %.0fms",
                        (time.monotonic() - t0) * 1e3)
        except Exception as e:  # noqa: BLE001
            logger.warning("[TileLangCompressionManager] warmup failed: %s", e)

        # Warm up the codec too: the LC library's first encode/decode pays a
        # one-time cudaMalloc + first-launch cost (~7ms) that otherwise lands on
        # the first real request. The codec kernel itself is sub-ms (see
        # tests/bench_lc_codec.py), so this is the only codec cost worth hiding.
        try:
            t1 = time.monotonic()
            buf = torch.zeros(2, self._num_kv_heads, 1, block_size,
                              self._head_size, dtype=torch.uint8, device="cuda")
            comp = self.codec.encode(0, buf, **self.codec_config)
            self.codec.decode(0, comp, "uint8", list(buf.shape), "cuda",
                              **self.codec_config)
            torch.cuda.synchronize()
            logger.info("[TileLangCompressionManager] codec warmup done in %.0fms",
                        (time.monotonic() - t1) * 1e3)
        except Exception as e:  # noqa: BLE001
            logger.warning("[TileLangCompressionManager] codec warmup failed: %s", e)

    # ── compress ────────────────────────────────────────────────────────────
    def compress_all_layers(self, all_layers_data: torch.Tensor,
                            request_id: str, config: Any,
                            metadata: dict) -> Optional[CompressedKVData]:
        if not isinstance(all_layers_data, torch.Tensor) or all_layers_data.dim() != 6:
            logger.error("[TileLangCompressionManager] expected 6D tensor, got %s",
                         getattr(all_layers_data, "shape", type(all_layers_data)))
            return None

        num_layers = all_layers_data.shape[0]
        original_size = all_layers_data.numel() * all_layers_data.element_size()

        if all_layers_data.is_cuda:
            torch.cuda.synchronize()
        t0 = time.monotonic()
        q_layers = []
        qparams = []
        for l in range(num_layers):
            q_l, meta_l = self.op.compress_v3(all_layers_data[l].contiguous(), layer_id=l)
            # Reorder to heads-outer [2, heads, blocks, bs, dim] so each nvCOMP
            # chunk stays within one head (shared quant scale) -> homogeneous ->
            # better ANS ratio. Matches the current pipeline's codec layout.
            q_layers.append(q_l.permute(0, 3, 1, 2, 4).contiguous())
            qparams.append(_canon_meta(meta_l))
        buffer = torch.stack(q_layers, dim=0).contiguous()  # [L,2,heads,blocks,bs,dim] uint8
        if buffer.is_cuda:
            torch.cuda.synchronize()
        t_quant_ms = (time.monotonic() - t0) * 1e3

        t1 = time.monotonic()
        compressed_tensor = self.codec.encode(num_layers - 1, buffer, **self.codec_config)
        if isinstance(compressed_tensor, torch.Tensor) and compressed_tensor.is_cuda:
            torch.cuda.synchronize()
        t_codec_ms = (time.monotonic() - t1) * 1e3

        comp_meta = {
            "request_id": request_id,
            "num_layers": num_layers,
            "codec_applied": True,
            "codec_shape": list(buffer.shape),
            "codec_dtype": "uint8",
            "device": str(buffer.device),
            "original_dtype": str(all_layers_data.dtype).replace("torch.", ""),
            "original_shape": list(all_layers_data.shape),
            "quantization_params": qparams,
            "backend": "tilelang_v3",
            "t_quant_ms": t_quant_ms,
            "t_codec_ms": t_codec_ms,
            **metadata,
        }
        compressed_size = compressed_tensor.numel() * compressed_tensor.element_size()
        return CompressedKVData(
            request_id=request_id,
            layer_id=num_layers - 1,
            compressed_tensor=compressed_tensor,
            metadata=comp_meta,
            original_size=original_size,
            compressed_size=compressed_size,
        )

    # ── per-layer entry points (for multi-stream overlap) ───────────────────
    def compress_quant_layer(self, kv_layer: torch.Tensor, layer_id: int):
        """Transform+quantize ONE layer (fused compress_v3), heads-outer layout.
        Returns (q_perm[2,heads,blocks,bs,dim] uint8, canon_meta). No codec.
        Cheap enough to launch on a side stream during the prefill forward."""
        q, meta = self.op.compress_v3(kv_layer.contiguous(), layer_id=layer_id)
        q_perm = q.permute(0, 3, 1, 2, 4).contiguous()
        return q_perm, _canon_meta(meta)

    def finalize_codec(self, q_perm_layers, canon_metas, request_id,
                       original_dtype, original_shape):
        """Stack pre-quantized heads-outer layers and run the codec once.
        Mirrors compress_all_layers' output, for the overlap path where
        transform+quant already ran per-layer on a side stream."""
        buffer = torch.stack(q_perm_layers, dim=0).contiguous()
        num_layers = buffer.shape[0]
        compressed_tensor = self.codec.encode(num_layers - 1, buffer,
                                              **self.codec_config)
        comp_meta = {
            "request_id": request_id,
            "num_layers": num_layers,
            "codec_applied": True,
            "codec_shape": list(buffer.shape),
            "codec_dtype": "uint8",
            "device": str(buffer.device),
            "original_dtype": original_dtype,
            "original_shape": list(original_shape),
            "quantization_params": canon_metas,
            "backend": "tilelang_v3",
        }
        return CompressedKVData(
            request_id=request_id,
            layer_id=num_layers - 1,
            compressed_tensor=compressed_tensor,
            metadata=comp_meta,
            original_size=int(torch.tensor(original_shape).prod().item()) * 2,
            compressed_size=compressed_tensor.numel() * compressed_tensor.element_size(),
        )

    def finalize_layer_group(self, q_perms, cmetas, member_tids, member_blocks,
                             layer_id):
        """LOSSLESS per-layer group codec: each request was transform+quantized
        INDEPENDENTLY (per-request scale → no accuracy change); here we just
        concatenate the uint8 quantized buffers along the block axis and run ONE
        codec over the whole layer (codec is lossless, so merging doesn't touch
        accuracy). Returns a CompressedKVData carrying the single blob + per-member
        (transfer_id, canon meta, block count) so the consumer can split + dequant
        each request with its own scale. Cuts codec/.item()/send to once per layer."""
        big_q = torch.cat(q_perms, dim=2)  # [2, heads, total_blocks, bs, dim]
        blob = self.codec.encode(layer_id, big_q, **self.codec_config)
        meta = {
            "grouped": True,
            "num_members": len(q_perms),
            "member_tids": list(member_tids),
            "member_blocks": [int(b) for b in member_blocks],
            "codec_shape": list(big_q.shape),
            "codec_dtype": "uint8",
            "device": str(big_q.device),
            "layer_id": int(layer_id),
            "quantization_params": list(cmetas),  # one canon meta per member
            "backend": "tilelang_v3",
        }
        return CompressedKVData(
            request_id="grp",
            layer_id=int(layer_id),
            compressed_tensor=blob,
            metadata=meta,
            original_size=sum(int(b) for b in member_blocks),
            compressed_size=blob.numel() * blob.element_size(),
        )

    def decompress_layer_group(self, compressed_data):
        """Inverse of finalize_layer_group: codec-decode the layer blob ONCE, then
        split per member and dequant each with its own meta. Returns a list of
        (transfer_id, kv_layer) where kv_layer is [2, blocks, bs, heads, dim]."""
        meta = compressed_data.metadata
        shape = [int(x) for x in meta["codec_shape"]]
        device = meta["device"]
        layer_id = int(meta["layer_id"])
        big_q = self.codec.decode(layer_id, compressed_data.compressed_tensor,
                                  "uint8", shape, device, **self.codec_config)
        if isinstance(big_q, torch.Tensor):
            big_q = big_q.view(torch.uint8).reshape(shape)
        out = []
        off = 0
        for i, (tid, nb) in enumerate(zip(meta["member_tids"], meta["member_blocks"])):
            sl = big_q[:, :, off:off + nb, :, :]      # [2, heads, nb, bs, dim]
            off += nb
            meta_l = _decanon_meta(meta["quantization_params"][i])
            layer_q = sl.permute(0, 2, 3, 1, 4).contiguous()  # [2, nb, bs, heads, dim]
            kv = self.op.decompress_v3(layer_q, meta_l, layer_id=layer_id)
            out.append((tid, kv))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return out

    def compress_layer_full(self, kv_layer: torch.Tensor, layer_id: int):
        """ONE layer all the way through: transform+quantize (compress_v3) AND
        codec. Returns (codec_bytes, chunk_meta) so the codec ALSO overlaps the
        prefill forward on the side stream (overlap=full). The per-layer chunks
        are assembled into a chunked CompressedKVData by finalize_chunked."""
        q_perm, cmeta = self.compress_quant_layer(kv_layer, layer_id)
        comp = self.codec.encode(layer_id, q_perm, **self.codec_config)
        chunk_meta = {
            "codec_applied": True,
            "codec_shape": list(q_perm.shape),
            "codec_dtype": "uint8",
            "quantization_params": [cmeta],
            "layer_id": int(layer_id),  # physical layer (for streamed/out-of-order chunks)
        }
        return comp, chunk_meta

    def finalize_chunked(self, chunks, chunk_metas, request_id,
                         original_dtype, original_shape):
        """Assemble per-layer codec chunks into a chunked CompressedKVData."""
        comp_size = sum(c.numel() * c.element_size() for c in chunks)
        return CompressedKVData(
            request_id=request_id,
            layer_id=len(chunks) - 1,
            compressed_tensor=None,
            metadata={
                "request_id": request_id,
                "num_layers": len(chunks),
                "device": str(chunks[0].device),
                "original_dtype": original_dtype,
                "original_shape": list(original_shape),
                "backend": "tilelang_v3",
            },
            original_size=int(torch.tensor(original_shape).prod().item()) * 2,
            compressed_size=comp_size,
            is_chunked=True,
            chunks=chunks,
            chunk_metadata=chunk_metas,
        )

    # ── decompress ──────────────────────────────────────────────────────────
    def decompress_all_layers(self, compressed_data: CompressedKVData,
                              config: Any) -> Optional[torch.Tensor]:
        meta = compressed_data.metadata
        if compressed_data.is_chunked:
            return self._decompress_chunked(compressed_data)
        num_layers = int(meta["num_layers"])
        shape = [int(x) for x in meta["codec_shape"]]
        device = meta["device"]

        t0 = time.monotonic()
        buffer = self.codec.decode(num_layers - 1, compressed_data.compressed_tensor,
                                   "uint8", shape, device, **self.codec_config)
        if isinstance(buffer, torch.Tensor):
            buffer = buffer.view(torch.uint8).reshape(shape)
            if buffer.is_cuda:
                torch.cuda.synchronize()
        t_codec_decode_ms = (time.monotonic() - t0) * 1e3

        t1 = time.monotonic()
        outs = []
        for l in range(num_layers):
            meta_l = _decanon_meta(meta["quantization_params"][l])
            # Undo the heads-outer reorder: [2,heads,blocks,bs,dim] -> [2,blocks,bs,heads,dim]
            layer_q = buffer[l].permute(0, 2, 3, 1, 4).contiguous()
            outs.append(self.op.decompress_v3(layer_q, meta_l, layer_id=l))
        result = torch.stack(outs, dim=0)
        if result.is_cuda:
            torch.cuda.synchronize()
        self._last_decompress_timing = {
            "t_codec_decode_ms": t_codec_decode_ms,
            "t_dequant_ms": (time.monotonic() - t1) * 1e3,
        }
        return result

    def _decompress_chunked(self, compressed_data: CompressedKVData):
        """Decompress per-layer codec chunks (overlap=full output)."""
        meta = compressed_data.metadata
        device = meta["device"]
        chunks = compressed_data.chunks
        chunk_metas = compressed_data.chunk_metadata
        t0 = time.monotonic()
        t_codec = 0.0
        outs = []
        for l, (chunk, cm) in enumerate(zip(chunks, chunk_metas)):
            shape = [int(x) for x in cm["codec_shape"]]
            tc = time.monotonic()
            q_perm = self.codec.decode(l, chunk, "uint8", shape, device,
                                       **self.codec_config)
            if isinstance(q_perm, torch.Tensor):
                q_perm = q_perm.view(torch.uint8).reshape(shape)
            t_codec += time.monotonic() - tc
            meta_l = _decanon_meta(cm["quantization_params"][0])
            layer_q = q_perm.permute(0, 2, 3, 1, 4).contiguous()
            outs.append(self.op.decompress_v3(layer_q, meta_l,
                                              layer_id=int(cm.get("layer_id", l))))
        result = torch.stack(outs, dim=0)
        if result.is_cuda:
            torch.cuda.synchronize()
        total = time.monotonic() - t0
        self._last_decompress_timing = {
            "t_codec_decode_ms": t_codec * 1e3,
            "t_dequant_ms": (total - t_codec) * 1e3,
        }
        return result
