import math
import torch
import fast_hadamard_transform
from typing import List, Tuple, Optional

class HadamardKVTransform:
    """
    Vectorized Hadamard Transform for KV Cache.
    Maintains tensor shape: [bsz, num_heads, seq_len, head_dim].
    """

    def __init__(self, base_seed: int = 0xC0FEBABE):
        self.base_seed = base_seed

    def _get_seed(self, layer_idx: int, head_idx: int) -> int:
        return self.base_seed ^ (layer_idx << 16) ^ head_idx

    def _pow2_chunks(self, dim: int) -> List[int]:
        """Split dim into a list of descending powers-of-two."""
        chunks: List[int] = []
        remaining = dim
        while remaining > 0:
            chunk = 1 << (remaining.bit_length() - 1)
            chunks.append(chunk)
            remaining -= chunk
        return chunks

    def _fwht_in_chunks(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply FWHT to the last dimension of an arbitrary shaped tensor.
        If head_dim is not a power of two, it splits into chunks.
        
        Args:
            x: Tensor of shape [..., head_dim]
        """
        # x.shape[-1] 是 head_dim
        chunks = self._pow2_chunks(x.shape[-1])
        outputs = []
        start = 0
        for size in chunks:
            # narrow 操作是 zero-copy 的，非常快
            part = x.narrow(-1, start, size)
            
            # fast_hadamard_transform 库通常支持任意前导维度 (Batch dimensions)
            # scale=1.0/sqrt(size) 保证变换是正交的 (Orthogonal)
            transformed = fast_hadamard_transform.hadamard_transform(
                part.contiguous(), scale=1.0 / math.sqrt(size)
            )
            outputs.append(transformed)
            start += size
        
        # 将切分处理后的结果拼接回原形状
        return torch.cat(outputs, dim=-1)

    def get_rademacher_signs(self, 
                             num_heads: int, 
                             head_dim: int, 
                             layer_idx: int, 
                             device: torch.device, 
                             dtype: torch.dtype) -> torch.Tensor:
        """
        Generate the Rademacher sign tensor for ALL heads in a layer.
        
        Returns:
            signs: Tensor of shape [1, num_heads, 1, head_dim] for broadcasting.
        """
        signs_list = []
        
        # 虽然这里有一个循环，但只循环 num_heads 次 (例如 32 次)，
        # 且只生成 1D 向量，开销极小。
        # 相比于原来的对整个 KV cache 进行循环，这完全是可以接受的。
        gen = torch.Generator(device="cpu") # 使用 CPU 生成器保证跨设备确定性
        
        for h in range(num_heads):
            seed = self._get_seed(layer_idx, h)
            gen.manual_seed(seed)
            
            # 生成 {0, 1} -> 转为 {-1, 1}
            s = torch.randint(0, 2, (head_dim,), generator=gen, dtype=torch.int8)
            s = s.float().mul_(2).sub_(1)
            signs_list.append(s)
            
        # Stack heads -> [num_heads, head_dim]
        signs = torch.stack(signs_list).to(device=device, dtype=dtype)
        
        # Reshape for broadcasting: [1, num_heads, 1, head_dim]
        # 这样可以直接与 [bsz, num_heads, seq_len, head_dim] 相乘
        return signs.view(1, num_heads, 1, head_dim)

    def transform(self, kv: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Apply Hadamard transform to the KV tensor in-place or out-of-place.
        
        Args:
            kv: Input tensor [bsz, num_heads, seq_len, head_dim]
            layer_idx: Current layer index for seed generation
            
        Returns:
            transformed_kv: Tensor of same shape [bsz, num_heads, seq_len, head_dim]
        """
        assert kv.dim() == 4, f"Expected 4D tensor [bsz, heads, seq, dim], got {kv.shape}"
        bsz, num_heads, seq_len, head_dim = kv.shape

        # 1. 获取所有 Heads 的随机符号向量 (Shape: [1, num_heads, 1, head_dim])
        # 如果需要极致性能，可以将这个 signs 缓存起来，避免每次 forward 都重新生成
        signs = self.get_rademacher_signs(num_heads, head_dim, layer_idx, kv.device, kv.dtype)

        # 2. Element-wise 乘法 (利用广播机制)
        # [bsz, H, S, D] * [1, H, 1, D] -> [bsz, H, S, D]
        signed_kv = kv * signs

        # 3. Apply FWHT (Batched)
        # 库函数通常会自动处理 batch 维度，只要对最后一维操作即可
        output = self._fwht_in_chunks(signed_kv)

        return output

    def inverse(self, transformed_kv: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Inverse Hadamard transform.
        Logic: Inverse(H * S * x) = S * Inverse(H) * (H * S * x) 
               Since H is symmetric orthogonal (scaled), H^{-1} is proportional to H.
               And S is its own inverse (1/1=1, 1/-1=-1).
               So, steps are: 1. FWHT, 2. Multiply by S.
        """
        assert transformed_kv.dim() == 4
        bsz, num_heads, seq_len, head_dim = transformed_kv.shape

        # 1. Apply FWHT first (Hadamard matrix is symmetric)
        rotated_back = self._fwht_in_chunks(transformed_kv)

        # 2. Multiply by signs
        signs = self.get_rademacher_signs(num_heads, head_dim, layer_idx, transformed_kv.device, transformed_kv.dtype)
        original_kv = rotated_back * signs

        return original_kv

if __name__ == "__main__":
    # 模拟数据
    bsz, num_heads, seq_len, head_dim = 2, 32, 128, 64
    layer_idx = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建 KV Cache [bsz, heads, seq, dim]
    kv_cache = torch.randn(bsz, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    
    # 初始化变换类
    h_transform = HadamardKVTransform(base_seed=0x1234)
    
    # 1. 正向变换
    kv_transformed = h_transform.transform(kv_cache, layer_idx)
    
    print(f"原始形状: {kv_cache.shape}")
    print(f"变换后形状: {kv_transformed.shape}") # 应该保持不变
    
    # 2. 逆变换 (验证)
    kv_restored = h_transform.inverse(kv_transformed, layer_idx)
    
    # 验证误差
    diff = (kv_cache - kv_restored).abs().max()
    print(f"最大还原误差: {diff.item()}")