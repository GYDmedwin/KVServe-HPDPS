import io
import pickle
import torchac_cuda
import numpy as np
import torch

from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class CacheGenGPUBytestream:
    bytestream: torch.Tensor
    bytestream_lengths: torch.Tensor  # [nlayers, nchannels, bytestream_length]
    ntokens: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

@dataclass 
class CacheGenGPUEncoderOutput:
    data_chunks: List[CacheGenGPUBytestream]
    cdf: torch.Tensor
    meta_data: List

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

    def to_bytes(self) -> bytes:
        """ Save the output to a file """
        with io.BytesIO() as f:
            pickle.dump(self, f)
            return f.getvalue()

    @staticmethod
    def from_bytes(bs: bytes) -> "CacheGenGPUEncoderOutput":
        with io.BytesIO(bs) as f:
            return pickle.load(f)


def reshape_tensor(tensor: torch.Tensor) -> torch.Tensor:

    # [bsz(1), head_num, seq_len, head_size] -> [each_layer(1), seq_len, num_heads * head_size]

    # [bsz(1), head_num, seq_len, head_size] -> [bsz(1), seq_len, head_num, head_size]
    tensor = tensor.transpose(1, 2)
    # [bsz(1), seq_len, head_num, head_size] -> [seq_len, num_heads, head_size]
    tensor = tensor.squeeze(0)
    S, H, D = tensor.shape

    tensor = tensor.reshape(-1, S, H * D)

    return tensor

def collect_bytes(output_buffer, output_lengths) -> torch.Tensor:
    """
    Collect a byte tensor from the output_buffer + output_lengths
    """
    output_buffer_size = output_buffer.shape[-1]
    flattened_lengths = output_lengths.flatten()
    flattened_buffer = output_buffer.flatten()
    summed_length = (output_buffer_size - flattened_lengths).cumsum(0)
    summed_length = summed_length.roll(1)
    summed_length[0] = 0
    indexes = summed_length.repeat_interleave(flattened_lengths)
    indexes = indexes + torch.arange(len(indexes), device=indexes.device)
    return flattened_buffer[indexes]

def encode_ntokens(cdf_int, encode_input, output_buffer, output_lengths) -> torch.Tensor:
    """
    Input:
        cdf_int: int16 tensor on GPU with shape [nlayers, nchannels, Lp]
        encode_input: int8 tensor on GPU with shape [nlayers, ntokens, nchannels]
        output_buffer: uint8 tensor on GPU with shape [nlayers, nchannels, BUFFER_SIZE]
        output_lengths: int32 tensor on GPU with shape [nlayers, nchannels]
    Returns:
        byte_tensor: the byte tensor
    """
    torchac_cuda.encode_fast_new(
            cdf_int,
            encode_input,
            output_buffer,
            output_lengths,
    )
    byte_tensor = collect_bytes(output_buffer, output_lengths)
    return byte_tensor

class CacheGenWrapper:
    def __init__(
        self, 
        quantized_keys: torch.Tensor,
        quantized_values: torch.Tensor,
        meta_data: List,
    ):
        assert quantized_keys.max() <= 127 and quantized_values.max() <= 127, "Quantized values must be in the range of [0, 127]"

        B, H, S, D = quantized_keys.shape
        quantized_keys = quantized_keys.transpose(1, 2).reshape(B, S, H * D)
        quantized_values = quantized_values.transpose(1, 2).reshape(B, S, H * D)
        self.quantized_keys = quantized_keys.to(torch.int8)
        self.quantized_values = quantized_values.to(torch.int8)
        self.meta_data = meta_data

    def compress(self) -> CacheGenGPUEncoderOutput:

        encode_input = torch.cat((self.quantized_keys, self.quantized_values), dim=0)
        nlayers, ntokens, nchannels = encode_input.shape

        max_value = max(self.quantized_keys.max(), self.quantized_values.max())
        new_cdf_key = torchac_cuda.calculate_cdf(self.quantized_keys, max_value)
        new_cdf_value = torchac_cuda.calculate_cdf(self.quantized_values, max_value)
        cdf_int = torch.cat([new_cdf_key, new_cdf_value])

        output_buffer = torch.zeros(
                (nlayers, nchannels, 256), 
                dtype=torch.uint8, 
                device=encode_input.device)
        output_lengths = torch.zeros(
                (nlayers, nchannels), 
                dtype=torch.int32, 
                device=encode_input.device)

        data_chunks = []
        for i in range(0, ntokens, 256):
            start = i
            end = min(i + 256, ntokens)
            bytestream = encode_ntokens(
                cdf_int,
                encode_input[:, start:end, :],
                output_buffer,
                output_lengths
            )
            data_chunks.append(CacheGenGPUBytestream(
                bytestream = bytestream, 
                bytestream_lengths = output_lengths.clone(),
                ntokens = end - start,
            ))


        return CacheGenGPUEncoderOutput(
                data_chunks,
                cdf_int,
                meta_data = self.meta_data,
            )


    def decompress(self, compressed: CacheGenGPUEncoderOutput) -> torch.Tensor:
        return compressed.tensor