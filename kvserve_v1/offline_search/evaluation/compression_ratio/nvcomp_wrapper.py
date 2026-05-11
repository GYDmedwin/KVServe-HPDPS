import torch
import cupy as cp
from nvidia import nvcomp
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class CompressedTensor:
    """
    自定义的数据类，用于打包压缩数据和重建所需的元数据
    """
    buffer: object          # nvcomp 的压缩对象
    original_shape: Tuple   # 原始形状，例如 (1, 32, 1024, 128)
    original_dtype: torch.dtype  # 原始类型，例如 torch.bfloat16
    original_device: torch.device     # 原始tensor的设备

@dataclass
class PackedData:
    compressed_tensor: CompressedTensor
    meta_data: List

@dataclass
class TensorData:
    tensor: torch.Tensor
    meta_data: List

class NVCompWrapper:
    def __init__(self, algorithm="ANS", **kwargs):
        self.codec = nvcomp.Codec(algorithm=algorithm, **kwargs)

    def compress(self, keys_tensor: torch.Tensor, values_tensor: torch.Tensor) -> CompressedTensor:
        """
        输入 PyTorch Tensor，返回包含元数据的压缩包
        """
        # 1. 记录元数据
        tensor = torch.cat([keys_tensor, values_tensor], dim=0)
        shape = tensor.shape
        dtype = tensor.dtype
        device = tensor.device
        
        # 2. 转换为扁平的 uint8 视图 (兼容性最强的方式)
        # 必须 contiguous 否则 nvcomp 会报错
        tensor_bytes = tensor.flatten().view(torch.uint8).contiguous()
        
        # # Save tensor_bytes to a binary file
        # with open("tensor_bytes.bin", "wb") as f:
        #     f.write(tensor_bytes.cpu().numpy().tobytes())
        
        # 3. 包装并压缩
        nv_array = nvcomp.as_array(tensor_bytes)
        comp_buffer = self.codec.encode(nv_array)
        
        # 4. 将 nvcomp buffer 转换为可序列化的 bytes
        comp_tensor = torch.as_tensor(comp_buffer, device=device, dtype=torch.uint8)
        # comp_bytes = cp.asarray(comp_buffer).tobytes()

        return CompressedTensor(
            buffer=comp_tensor,
            original_shape=shape,
            original_dtype=dtype,
            original_device=device
        )

    def decompress(self, compressed: CompressedTensor) -> torch.Tensor:
        """
        输入压缩包，自动重建回原始 Tensor
        """
        # 1. 解压 (得到扁平的 uint8 字节流)
        # 将 bytes 恢复为 nvcomp array
        # comp_cupy = cp.frombuffer(compressed.buffer, dtype=cp.uint8)
        comp_buffer_nv = nvcomp.as_array(compressed.buffer)

        # 这里的 decode 返回的是 nvcomp 的 byte array
        decomp_buffer = self.codec.decode(comp_buffer_nv)
        
        # 2. 转换为 PyTorch Tensor (uint8)
        # 借助 CuPy 进行零拷贝转换
        # decomp_cupy = cp.asarray(decomp_buffer)
        decomp_torch = torch.as_tensor(decomp_buffer, device=compressed.original_device, dtype=torch.uint8)
        
        # 3. 恢复类型和形状
        # 解释二进制位为原始 dtype
        reconstructed = decomp_torch.view(compressed.original_dtype)
        
        # 恢复维度
        reconstructed = reconstructed.reshape(compressed.original_shape)
        
        return reconstructed

def to_device(data, device):
    """递归将张量移动到指定设备，支持 list / dict / tuple / 嵌套结构"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device) for x in data)
    else:
        # 非张量/容器类型（如 int, str, None）原样返回
        return data

