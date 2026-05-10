import torch
import gc
import pandas as pd
import numpy as np
from datasets import load_dataset
from typing import List, Tuple
from collections import deque
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import argparse

BASE_MODEL_DIR = "/root/workspace/models"
LOW_PRESS_RATIO = 0.5

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
        return tensor.to(torch.uint8), None

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
        return quantized_tensor.to(torch.bfloat16)

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

def main(args):
    # Unpack args
    INPUT_LENGTH = args.input_length
    WARMUP_ITER = args.warmup_iter
    BENCHMARK_ITER = args.benchmark_iter
    MODEL_NAME = args.model_name

    # 1. Load Model and Tokenizer
    print("Loading model and tokenizer...")
    # It's assumed you are logged in to huggingface-cli
    # load dataset and sample a text
    dataset = load_dataset("Xnhyacinth/LongBench", "2wikimqa", split="test").to_pandas()
    # Select the longest data based on the 'length' field
    max_length_idx = dataset['length'].idxmax()
    text = dataset.loc[max_length_idx, "context"]
    tokenizer = AutoTokenizer.from_pretrained(f"{BASE_MODEL_DIR}/{MODEL_NAME}")
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    current_length = input_ids.shape[1]

    if current_length > INPUT_LENGTH:
        inputs["input_ids"] = input_ids[:, :INPUT_LENGTH]
        inputs["attention_mask"] = inputs["attention_mask"][:, :INPUT_LENGTH]
    elif current_length < INPUT_LENGTH:
        num_repeats = (INPUT_LENGTH + current_length - 1) // current_length
        inputs["input_ids"] = input_ids.repeat(1, num_repeats)[:, :INPUT_LENGTH]
        inputs["attention_mask"] = inputs["attention_mask"].repeat(1, num_repeats)[:, :INPUT_LENGTH]
    inputs = inputs.to("cuda")
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        f"{BASE_MODEL_DIR}/{MODEL_NAME}",
        torch_dtype="auto",
        device_map="auto",
        use_cache=True,
        # output_attentions=True,
        attn_implementation="flash_attention_2",
    )
    # evaluate the model and run the generation

    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True)
        past_key_values = outputs.past_key_values

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    original_size = len(pickle.dumps(past_key_values)) / 1024 / 1024 / 1024
    key_list = [kv[0].to("cuda:1") for kv in past_key_values]
    value_list = [kv[1].to("cuda:1") for kv in past_key_values]
    print(f"Got past_key_values for {len(key_list)} layers.")

    # 3. Load scores and create mask
    try:
        df = pd.read_csv(f"../../duo_config/{MODEL_NAME}_scores.csv", header=None).dropna()
    except FileNotFoundError:
        print("Warning: score file not found. Generating a random mask for demonstration.")
        # Assuming model has 32 layers and 32 heads for Llama-3.1-8B
        num_layers = len(key_list)
        num_heads = key_list[0].shape[1]
        df = pd.DataFrame(np.random.rand(num_layers, num_heads))

    scores = torch.tensor(df.values, dtype=torch.float32).to('cuda:1')
    pruned_num = round(scores.numel() * LOW_PRESS_RATIO)
    scores_mask = torch.zeros_like(scores, dtype=torch.bool).to('cuda:1')
    flat_indices = torch.argsort(scores.flatten())[:pruned_num]
    multi_indices = torch.unravel_index(flat_indices, scores.shape)
    scores_mask[multi_indices] = True
    
    # 4. Prepare inputs for quantization before benchmarking to exclude data prep from timing
    print("Preparing quantization inputs...")
    quant_inputs = []
    for i in range(len(key_list)):
        layer_mask = scores_mask[i].to('cuda:1')
        k, v = key_list[i], value_list[i]
        
        low_keys = k[:, layer_mask, :, :]
        low_values = v[:, layer_mask, :, :]
        high_keys = k[:, ~layer_mask, :, :]
        high_values = v[:, ~layer_mask, :, :]
        quant_inputs.append((low_keys, low_values, high_keys, high_values))

    # 5. Warmup
    print(f"Warming up for {WARMUP_ITER} iterations...")
    for _ in range(WARMUP_ITER):
        # Quantize
        quantized_outputs = []
        for low_keys, low_values, high_keys, high_values in quant_inputs:
            high_keys_q = quantization(16, high_keys, [2])
            high_values_q = quantization(16, high_values, [1, 3])            
            low_keys_q = quantization(8, low_keys, [2])
            low_values_q = quantization(8, low_values, [1, 3])
            quantized_outputs.append((low_keys_q, low_values_q, high_keys_q, high_values_q))
        
        # Dequantize
        for low_keys_q, low_values_q, high_keys_q, high_values_q in quantized_outputs:
            _ = dequantization(*low_keys_q)
            _ = dequantization(*low_values_q)
            _ = dequantization(*high_keys_q)
            _ = dequantization(*high_values_q)
    torch.cuda.synchronize()

    # 6. Benchmarking
    print(f"Benchmarking for {BENCHMARK_ITER} iterations...")
    quant_total_time = 0
    dequant_total_time = 0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    quantized_outputs = []
    torch.cuda.synchronize()
    for _ in range(BENCHMARK_ITER):
        torch.cuda.synchronize()
        start_event.record()
        for low_keys, low_values, high_keys, high_values in quant_inputs:
            high_keys_q = quantization(16, high_keys, [2])
            high_values_q = quantization(16, high_values, [1, 3])            
            low_keys_q = quantization(8, low_keys, [2])
            low_values_q = quantization(8, low_values, [1, 3])
        end_event.record()
        # end_event.synchronize()
        torch.cuda.synchronize()
        quant_total_time += start_event.elapsed_time(end_event)

    for low_keys, low_values, high_keys, high_values in quant_inputs:
        high_keys_q = quantization(16, high_keys, [2])
        high_values_q = quantization(16, high_values, [1, 3])        
        low_keys_q = quantization(8, low_keys, [2])
        low_values_q = quantization(8, low_values, [1, 3])
        quantized_outputs.append((low_keys_q, low_values_q, high_keys_q, high_values_q))

    for _ in range(BENCHMARK_ITER):
        torch.cuda.synchronize()
        start_event.record()
        for low_keys_q, low_values_q, high_keys_q, high_values_q in quantized_outputs:
            _ = dequantization(*low_keys_q)
            _ = dequantization(*low_values_q)
            _ = dequantization(*high_keys_q)
            _ = dequantization(*high_values_q)
        end_event.record()
        # end_event.synchronize()
        torch.cuda.synchronize()
        dequant_total_time += start_event.elapsed_time(end_event)

    avg_quant_time = quant_total_time / BENCHMARK_ITER / 1000
    avg_dequant_time = dequant_total_time / BENCHMARK_ITER / 1000
    
    print("\n--- Benchmark Results ---")
    print(f"Input length: {INPUT_LENGTH}")
    print(f"Original size: {original_size:.4f} GB")
    print(f"Prefill time: {avg_quant_time:.4f} s")
    print(f"Decode time: {avg_dequant_time:.4f} s")
    print(f"Prefill throughput: {original_size / avg_quant_time:.2f} GB/s")
    print(f"Decode throughput: {original_size / avg_dequant_time:.2f} GB/s")
    print("-------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantizer Speed Benchmark")
    parser.add_argument("--input_length", type=int, default=1024, help="Input sequence length")
    parser.add_argument("--warmup_iter", type=int, default=100, help="Number of warmup iterations")
    parser.add_argument("--benchmark_iter", type=int, default=200, help="Number of benchmark iterations")
    parser.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct", help="Model name")
    args = parser.parse_args()
    main(args)


