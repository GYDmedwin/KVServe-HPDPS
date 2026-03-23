"""
Codec compression components for KV cache compression
"""

# Placeholder for future codec compression implementations
# Components should inherit from kvserve_v1.compression.components.Codec
from kvserve_v1.compression.codec.nvcomp_func import nvCOMPCodec
from kvserve_v1.compression.codec.kvserve_codec import KVServeCodec

__all__ = [
    "nvCOMPCodec",
    "KVServeCodec",
]


