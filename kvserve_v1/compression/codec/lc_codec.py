"""Open-source LC-framework codec (BIT_2 RZE_1), a drop-in for KVServeCodec.

Why: nvCOMP is closed-source with internal stream syncs that block async overlap
and prevent fusion. LC is open, ~3x faster (517/567 GB/s vs ~179), and its
encode/decode run on the CURRENT CUDA stream — so in the connector's side-stream
overlap path they actually overlap the forward, unlike nvCOMP.

Pipeline BIT_2 RZE_1 chosen by LC stage search on real quantized KV (best 2-stage,
3.10x; 3-stage only +0.7%). Encode/decode are stream-aware ctypes calls into
liblcencode_fat.so / liblcdecode_fat.so (fat sm_90+sm_120 binaries), matching the
KVServeCodec.encode/decode signatures so it slots into CompressionManager.
"""

import ctypes
import os
from typing import Any

import torch

_ENC_SO = os.environ.get("LC_ENCODE_SO", "/data/LC-framework/liblcencode_fat.so")
_DEC_SO = os.environ.get("LC_DECODE_SO", "/data/LC-framework/liblcdecode_fat.so")


def _load(so, decode=False):
    lib = ctypes.CDLL(so)
    lib.lc_cs.restype = ctypes.c_int
    lib.lc_chunks.restype = ctypes.c_longlong
    lib.lc_chunks.argtypes = [ctypes.c_longlong]
    lib.lc_default_blocks.restype = ctypes.c_int
    lib.lc_default_blocks.argtypes = [ctypes.c_int, ctypes.c_int]
    if decode:
        lib.lc_decode_stream.restype = None
        lib.lc_decode_stream.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int]
    else:
        lib.lc_maxsize.restype = ctypes.c_longlong
        lib.lc_maxsize.argtypes = [ctypes.c_longlong]
        lib.lc_encode_stream.restype = None
        lib.lc_encode_stream.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    return lib


class LCCodec:
    """LC BIT_2 RZE_1 codec with the KVServeCodec encode/decode surface."""

    def __init__(self, **kwargs):
        self._enc = _load(_ENC_SO, decode=False)
        self._dec = _load(_DEC_SO, decode=True)
        self.CS = self._enc.lc_cs()
        # throttle: fewer grid-stride blocks. 0/None -> hardware default.
        self._blocks = int(kwargs.get("lc_blocks", 0))
        # reusable scratch keyed by device (fullcarry sized per insize on demand)
        self._dev_blocks: dict = {}

    def update_params(self, **kwargs) -> None:
        if "lc_blocks" in kwargs:
            self._blocks = int(kwargs["lc_blocks"]) or 0

    def _blocks_for(self, device) -> int:
        if self._blocks > 0:
            return self._blocks
        if device not in self._dev_blocks:
            p = torch.cuda.get_device_properties(device)
            self._dev_blocks[device] = self._enc.lc_default_blocks(
                p.multi_processor_count, p.max_threads_per_multi_processor)
        return self._dev_blocks[device]

    # ── encode ──────────────────────────────────────────────────────────────
    def encode(self, layer_id: int, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        d_in = tensor.contiguous().view(torch.uint8).reshape(-1)
        insize = d_in.numel()
        dev = d_in.device
        blocks = self._blocks_for(dev)
        chunks = self._enc.lc_chunks(insize)
        maxsize = self._enc.lc_maxsize(insize)
        d_out = torch.empty(maxsize, dtype=torch.uint8, device=dev)
        d_outsize = torch.zeros(1, dtype=torch.int64, device=dev)
        d_fullcarry = torch.zeros(chunks, dtype=torch.int64, device=dev)
        stream = torch.cuda.current_stream(dev).cuda_stream
        self._enc.lc_encode_stream(
            ctypes.c_void_p(stream), ctypes.c_void_p(d_in.data_ptr()),
            ctypes.c_longlong(insize), ctypes.c_void_p(d_out.data_ptr()),
            ctypes.c_void_p(d_outsize.data_ptr()),
            ctypes.c_void_p(d_fullcarry.data_ptr()), ctypes.c_int(blocks))
        enc_size = int(d_outsize.item())  # syncs the side stream for this layer
        # Compact clone: the side-stream memcpy is off the critical path (it
        # overlaps the forward) and a view would pin the whole maxsize buffer
        # in flight — measured makespan-neutral, so keep the smaller footprint.
        return d_out[:enc_size].clone()   # compressed stream (header + data)

    # ── decode ──────────────────────────────────────────────────────────────
    def decode(self, layer_id: int, compressed_data: torch.Tensor,
               original_dtype: str, original_shape: list, device: str,
               **kwargs) -> torch.Tensor:
        comp = compressed_data.contiguous().view(torch.uint8)
        outsize = 1
        for s in original_shape:
            outsize *= int(s)  # uint8 buffer -> bytes == elements
        d_out = torch.empty(outsize, dtype=torch.uint8, device=device)
        d_outsize = torch.zeros(1, dtype=torch.int64, device=device)
        blocks = self._blocks_for(torch.device(device))
        stream = torch.cuda.current_stream(torch.device(device)).cuda_stream
        self._dec.lc_decode_stream(
            ctypes.c_void_p(stream), ctypes.c_void_p(comp.data_ptr()),
            ctypes.c_void_p(d_out.data_ptr()),
            ctypes.c_void_p(d_outsize.data_ptr()), ctypes.c_int(blocks))
        return d_out.reshape([int(s) for s in original_shape])
