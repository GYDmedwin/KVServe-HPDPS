import math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn


class AffineMatrix(nn.Module):
    """
    Affine transform manager for all layers.
    Supports loading parameters for multiple layers and applying
    transform/inverse by specifying layer_idx.
    """

    def __init__(
        self,
        head_dim: int,
        mode: str = "diag",
        learnable_clip: bool = True,
        clip_init: float = 4.0,
        params_path: Optional[str] = None,
        device: str = "cuda"
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.mode = mode
        assert mode in ("diag", "full")
        self.learnable_clip = learnable_clip
        self.clip_init = clip_init
        self.device = device
        
        # Storage for layer parameters
        # We use a dict to store parameters for each layer
        # Keys are layer_idx, values are dicts of tensors (on device)
        self.layer_params: Dict[int, Dict[str, torch.Tensor]] = {}
        
        if params_path:
            self.load_parameters(params_path)

    def load_parameters(self, path: str):
        try:
            loaded = torch.load(path, map_location=self.device)
            print(f"Successfully loaded parameters from {path}")
        except FileNotFoundError:
            print(f"Warning: Parameter file {path} not found.")
            return

        # Expected format: {layer_idx: {"state_dict": {...}}} 
        for layer_idx, data in loaded.items():
            # Handle potential string keys for layer_idx
            try:
                idx = int(layer_idx)
            except ValueError:
                continue
                
            if isinstance(data, dict) and "state_dict" in data:
                self.layer_params[idx] = data["state_dict"]
            else:
                # Fallback if structure is different
                self.layer_params[idx] = data
                
        # print(f"Loaded parameters for layers: {sorted(list(self.layer_params.keys()))}")

    def _get_layer_params(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        if layer_idx not in self.layer_params:
            # Fallback: initialize default parameters on the fly
            # This allows usage without loading params (random/default init)
            # Or if the specific layer is missing in the loaded params
            return self._init_default_params()
        return self.layer_params[layer_idx]

    def _init_default_params(self) -> Dict[str, torch.Tensor]:
        params = {}
        if self.mode == "diag":
            params["log_scale"] = torch.zeros(self.head_dim, device=self.device)
        else:
            # Initialize identity matrix for full mode
            linear_weight = torch.eye(self.head_dim, device=self.device)
            params["linear.weight"] = linear_weight
            
        if self.learnable_clip:
            init = math.log(math.exp(self.clip_init) - 1)  # inverse softplus
            params["clip_k"] = torch.full((self.head_dim,), init, dtype=torch.float32, device=self.device)
            params["clip_v"] = torch.full((self.head_dim,), init, dtype=torch.float32, device=self.device)
        return params

    def transform(self, x: torch.Tensor, kind: str, layer_idx: int) -> torch.Tensor:
        """
        Apply affine transform to K/V.
        x: [..., head_dim]
        kind: "k" or "v" (for clip selection)
        layer_idx: current layer index
        """
        params = self._get_layer_params(layer_idx)
        
        if self.mode == "diag":
            log_scale = params["log_scale"]
            log_scale = torch.clamp(log_scale, min=-5.0, max=5.0)
            # Ensure scale is on same device as x
            scale = torch.exp(log_scale).to(x)
            x = x * scale
        else:
            weight = params["linear.weight"]
            x = torch.matmul(x, weight.t().to(x))
            
        if self.learnable_clip:
            if kind == "k":
                clip_param = params["clip_k"]
            else:
                clip_param = params["clip_v"]
            
            alpha = torch.nn.functional.softplus(clip_param).to(x)
            x = torch.clamp(x, -alpha, alpha)
            
        return torch.nan_to_num(x)

    def inverse(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        params = self._get_layer_params(layer_idx)
        
        if self.mode == "diag":
            log_scale = params["log_scale"]
            log_scale = torch.clamp(log_scale, min=-5.0, max=5.0)
            inv_scale = torch.exp(-log_scale).to(x)
            out = torch.nan_to_num(x * inv_scale)
            return out
        
        # full matrix inverse
        weight = params["linear.weight"].to(torch.float64) # Higher precision for inverse
        inv_w = torch.linalg.inv(weight).to(x) # Cast back to x's dtype/device
        return torch.matmul(x, inv_w.t())

    # Backward-compatible aliases (updated signature)
    def forward_transform(self, x: torch.Tensor, kind: str, layer_idx: int) -> torch.Tensor:
        return self.transform(x, kind, layer_idx)

    def inverse_transform(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.inverse(x, layer_idx)


class IdentityQuantizer:
    """Placeholder quantizer: passthrough."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

# Usage example :
'''
from kv_quant.transform.kv_affine import AffineMatrix
import torch

# 初始化
affine = AffineMatrix(head_dim=128, mode="diag", params_path="/path/to/affine_params.pt", device="cuda")

# 假设 k,v 形状 [bsz, n_heads, seq, head_dim]
k_t = affine.transform(k, "k", layer_idx=0)
v_t = affine.transform(v, "v", layer_idx=0)

# 这里接入你的量化/反量化
k_q = quantize_dequantize_k(k_t)
v_q = quantize_dequantize_v(v_t)

k_rec = affine.inverse(k_q, layer_idx=0)
v_rec = affine.inverse(v_q, layer_idx=0)
'''

if __name__ == "__main__":
    # 模拟数据
    bsz, num_heads, seq_len, head_dim = 1, 8, 128, 128
    layer_idx = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建 KV Cache [bsz, heads, seq, dim]
    kv_cache = torch.randn(bsz, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    
    # 假设有一个参数文件 (这里使用默认值演示，如果文件存在则加载)
    params_file = "affine_params_v2.pt"
    
    # 初始化变换类
    affine = AffineMatrix(head_dim=128, mode="diag", params_path=params_file, device=device)
    
    # 1. 正向变换
    print(f"Transforming layer {layer_idx}...")
    kv_transformed = affine.transform(kv_cache, "k", layer_idx=layer_idx)
    
    # 2. 逆变换 (验证)
    kv_restored = affine.inverse(kv_transformed, layer_idx=layer_idx)
    
    # 验证误差
    diff = (kv_cache - kv_restored).abs().max()
    print(f"最大还原误差: {diff.item()}")
