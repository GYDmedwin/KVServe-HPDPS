import torch
import pandas as pd
import numpy as np
from typing import List, Tuple
from collections import deque
from test_hadamard import HadamardKVTransform
from test_affine import AffineMatrix

LOW_PRESS_RATIO = 0.5

key = torch.load("../data/k.pt")
value = torch.load("../data/v.pt")

df = pd.read_csv("../duo_config/Meta-Llama-3.1-8B-Instruct_scores.csv", header=None).dropna()
scores = torch.tensor(df.values, dtype=torch.float32)

pruned_num = round(scores.numel() * LOW_PRESS_RATIO)
scores_mask = torch.zeros_like(scores, dtype=torch.bool).to('cuda:0')

flat_indices = torch.argsort(scores.flatten())[:pruned_num]
multi_indices = torch.unravel_index(flat_indices, scores.shape)
scores_mask[multi_indices] = True

def quantization(
    max_value: int,
    tensor: torch.Tensor, 
    axis: List[int], 
) -> Tuple[torch.Tensor, dict]:
    """
    Corrected quantization implementation.
    Maps [min, max] -> [0, max_value]
    """
    if tensor.numel() == 0:
        return tensor, None

    # 1. 计算 Min/Max
    # 注意：对于 fp16/bf16，min/max 可能会有数值稳定性问题，但在 Python 层无法避免
    max_ = torch.amax(tensor, dim=axis, keepdim=True)
    min_ = torch.amin(tensor, dim=axis, keepdim=True)

    # 2. 计算 Scale (添加 eps 防止除零)
    # 此时 scale 代表每个整数 step 对应的浮点距离
    scale = (max_ - min_).clamp_(min=1e-5).div_(max_value - 1) 
    
    # 3. 量化核心逻辑
    # 公式： Q = clamp(round((x - min) / scale), 0, max_val)
    # 性能优化：先做减法再除，减少一次除法运算，且数值更稳
    # 显存优化：尽可能利用 PyTorch 的广播机制
    tensor_sub = tensor.sub(min_) # 此时 tensor_sub >= 0
    
    # 使用 reciprocal 进行乘法通常比除法快一点点，但这里为了清晰用除法
    quant_float = tensor_sub.div_(scale) # In-place division to save memory if tensor_sub is not needed
    
    # 必须 clamp 防止精度溢出导致的回绕
    quantized_tensor = quant_float.round_().clamp_(0, max_value).to(torch.uint8)

    # 4. Metadata
    # 修正：不要存 min_ints (它会溢出)，直接存 min_ (作为 Zero Point 的基准)
    # 反量化公式将变为: dequant = quantized * scale + min_
    meta_data = {
        "min_val": min_.to(tensor.dtype),  # 存浮点型的最小值
        "quant_scale": scale.to(tensor.dtype),
    }
    
    return quantized_tensor, meta_data

def dequantization(
    quantized_tensor: torch.Tensor,
    meta_data: dict,
) -> torch.Tensor:
    """
    Corrected dequantization.
    Formula: Real = Quantized * Scale + Min_Value
    """
    if meta_data is None:
        return quantized_tensor

    # 1. 获取元数据
    # 注意：这里对应的是修正版 _quantize 中的 "min_val"
    min_val = meta_data["min_val"]
    quant_scale = meta_data["quant_scale"]

    # 2. 类型转换 (Cast)
    # 将 uint8 转换为目标浮点类型 (fp16/bf16/fp32)
    # 这一步会申请显存，无法避免
    quant_float = quantized_tensor.to(quant_scale.dtype)

    # 3. 反量化计算
    # 此时执行的是: result = x * s + b
    # 这种形式不仅逻辑直观，而且利用了广播机制 (Broadcasting)
    dequantized_tensor = quant_float * quant_scale + min_val

    return dequantized_tensor


hadamard_transform = HadamardKVTransform(0x3333)
affine_transform = AffineMatrix(head_dim=128, mode="diag", params_path="affine_params_v2.pt", device="cuda")
low_quantized_key_list = deque()
low_quantized_value_list = deque()
high_quantized_key_list = deque()
high_quantized_value_list = deque()
for i in range(len(key)):
    # 获取当前层的mask，形状为 [heads]
    # scores_mask[i] 是一个布尔张量，例如 [True, True, False, ...]
    layer_mask = scores_mask[i]

    key[i] = key[i].to('cuda:0')
    value[i] = value[i].to('cuda:0')
    # h_key = hadamard_transform.transform(key[i], i)
    # h_value = hadamard_transform.transform(value[i], i)
    # h_key = affine_transform.transform(key[i], "k", i)
    # h_value = affine_transform.transform(value[i], "v", i)
    h_key = key[i]
    h_value = value[i]
    # 使用布尔索引来分离 "low" (mask为True) 和 "high" (mask为False) 的头
    # low_keys 的形状会是 [num_true_heads, seq_len, head_dim]
    low_keys = h_key[:, layer_mask, :, :]
    low_keys = quantization(4, low_keys, [2])
    low_values = h_value[:, layer_mask, :, :]
    low_values = quantization(4, low_values, [1, 3])

    # high_keys 的形状会是 [num_false_heads, seq_len, head_dim]
    # 使用 '~' 操作符来反转布尔mask，选取为False的头
    high_keys = h_key[:, ~layer_mask, :, :]
    high_keys = quantization(6, high_keys, [2])
    high_values = h_value[:, ~layer_mask, :, :]
    high_values = quantization(6, high_values, [1, 3])

    # 将分离出的张量添加到对应的列表中
    low_quantized_key_list.append(low_keys)
    low_quantized_value_list.append(low_values)

    high_quantized_key_list.append(high_keys)
    high_quantized_value_list.append(high_values)


real_key_list = []
real_value_list = []
for i in range(len(key)):
    low_keys_dequantized = dequantization(*low_quantized_key_list.popleft())
    low_values_dequantized = dequantization(*low_quantized_value_list.popleft())
    high_keys_dequantized = dequantization(*high_quantized_key_list.popleft())
    high_values_dequantized = dequantization(*high_quantized_value_list.popleft())

    # --- 拼接解量化后的K/V缓存 ---
    # 1. 创建一个与原始层形状、类型、设备都相同的空张量作为容器
    reconstructed_key_layer = torch.empty_like(key[i])
    reconstructed_value_layer = torch.empty_like(value[i])

    # 2. 获取当前层的布尔掩码
    layer_mask = scores_mask[i]

    # 3. 使用布尔掩码将 dequantized 张量放置到正确的位置
    #    'low' 组对应 mask 中为 True 的位置
    reconstructed_key_layer[:, layer_mask, :, :] = low_keys_dequantized
    reconstructed_value_layer[:, layer_mask, :, :] = low_values_dequantized
    
    #    'high' 组对应 mask 中为 False 的位置 (通过 '~' 反转mask)
    reconstructed_key_layer[:, ~layer_mask, :, :] = high_keys_dequantized
    reconstructed_value_layer[:, ~layer_mask, :, :] = high_values_dequantized

    # reconstructed_key_layer = hadamard_transform.inverse(reconstructed_key_layer, i)
    # reconstructed_value_layer = hadamard_transform.inverse(reconstructed_value_layer, i)
    # reconstructed_key_layer = affine_transform.inverse(reconstructed_key_layer, i)
    # reconstructed_value_layer = affine_transform.inverse(reconstructed_value_layer, i)

    # 4. 将重建好的完整层张量添加到列表中
    real_key_list.append(reconstructed_key_layer)
    real_value_list.append(reconstructed_value_layer)

# 循环结束后, 将列表中的所有层张量堆叠成一个大的张量
# dim=0 表示在新的第0维上堆叠，恢复原始的 [layers, batch, heads, ...] 形状
real_key = torch.stack(real_key_list, dim=0)
real_value = torch.stack(real_value_list, dim=0)

original_key = torch.stack(key, dim=0)
original_value = torch.stack(value, dim=0)

print(torch.nn.functional.cosine_similarity(real_key.flatten(), original_key.flatten(), dim=0))
print(torch.nn.functional.cosine_similarity(real_value.flatten(), original_value.flatten(), dim=0))


