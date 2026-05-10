import torch
import cupy as cp
from nvidia import nvcomp
from dataclasses import dataclass
from typing import Tuple

@dataclass
class CompressedTensor:
    """
    自定义的数据类，用于打包压缩数据和重建所需的元数据
    """
    buffer: object          # nvcomp 的压缩对象
    original_shape: Tuple   # 原始形状，例如 (1, 32, 1024, 128)
    original_dtype: torch.dtype  # 原始类型，例如 torch.bfloat16
    original_device: torch.device     # 原始tensor的设备

class NVCompWrapper:
    def __init__(self, algorithm="LZ4", **kwargs):
        self.codec = nvcomp.Codec(algorithm=algorithm, **kwargs)

    def compress(self, tensor: torch.Tensor) -> CompressedTensor:
        """
        输入 PyTorch Tensor，返回包含元数据的压缩包
        """
        # 1. 记录元数据
        shape = tensor.shape
        dtype = tensor.dtype
        device = tensor.device
        
        # 2. 转换为扁平的 uint8 视图 (兼容性最强的方式)
        # 必须 contiguous 否则 nvcomp 会报错
        tensor_bytes = tensor.flatten().view(torch.uint8).contiguous()
        
        # 3. 包装并压缩
        nv_array = nvcomp.as_array(tensor_bytes)
        comp_buffer = self.codec.encode(nv_array)
        
        return CompressedTensor(
            buffer=comp_buffer,
            original_shape=shape,
            original_dtype=dtype,
            original_device=device
        )

    def decompress(self, compressed: CompressedTensor) -> torch.Tensor:
        """
        输入压缩包，自动重建回原始 Tensor
        """
        # 1. 解压 (得到扁平的 uint8 字节流)
        # 这里的 decode 返回的是 nvcomp 的 byte array
        decomp_buffer = self.codec.decode(compressed.buffer)
        
        # 2. 转换为 PyTorch Tensor (uint8)
        # 借助 CuPy 进行零拷贝转换
        decomp_cupy = cp.asarray(decomp_buffer)
        decomp_torch = torch.as_tensor(decomp_cupy, device='cuda')
        
        # 3. 恢复类型和形状
        # 解释二进制位为原始 dtype
        reconstructed = decomp_torch.view(compressed.original_dtype)
        
        # 恢复维度
        reconstructed = reconstructed.reshape(compressed.original_shape)
        
        return reconstructed

# ================= 使用示例 =================
if __name__ == "__main__":
    # 准备数据
    original = torch.randn(2, 4096, dtype=torch.bfloat16, device='cuda')
    
    # 初始化封装器
    wrapper = NVCompWrapper("ANS", data_type="|u1")
    
    print(f"原始 Tensor: {original.shape}, {original.dtype}")
    
    # 1. 压缩 (Compress) -> 得到一个对象，里面包含了数据和 Shape 信息
    packed_data = wrapper.compress(original)
    print(f"压缩包大小: {packed_data.buffer.buffer_size} bytes")
    print(f"压缩包元数据: Shape={packed_data.original_shape}")
    
    # ... 此时你可以传输这个 packed_data 对象 ...
    
    # 2. 解压 (Decompress) -> 直接吐出还原好的 Tensor
    recon = wrapper.decompress(packed_data)
    print(f"重建 Tensor: {recon.shape}, {recon.dtype}")
    
    # 验证
    print(f"是否相等: {torch.equal(original, recon)}")