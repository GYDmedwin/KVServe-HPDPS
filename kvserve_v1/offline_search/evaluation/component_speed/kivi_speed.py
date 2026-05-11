import torch
import gc
import pandas as pd
import numpy as np
from datasets import load_dataset
from typing import List, Tuple, Optional, Dict, Any
import time
import pickle
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= Configuration =================
NBITS = 2
AXIS_KEY = 2    # 3 for Head Dim (Per-channel), 2 for Seq Len (Per-token)
AXIS_VALUE = 3  # 3 for Head Dim, 2 for Seq Len
Q_GROUP_SIZE = 32
BASE_MODEL_DIR = "/root/workspace/models"
# =================================================

def pack_to_float32(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Pack uint8 tensor to float32 container using bit packing.
    """
    assert bits in [2, 4], "Only 2-bit or 4-bit supported"
    
    # 1. Calculate pack ratio
    pack_ratio = 32 // bits 
    
    # 2. Flatten and convert to int32 for bitwise ops
    flat_tensor = tensor.flatten().to(torch.int32)
    
    # 3. Padding if needed
    num_elements = flat_tensor.numel()
    padding = (pack_ratio - (num_elements % pack_ratio)) % pack_ratio
    if padding > 0:
        flat_tensor = torch.nn.functional.pad(flat_tensor, (0, padding), value=0)
    
    # 4. Reshape to [N, pack_ratio]
    reshaped = flat_tensor.view(-1, pack_ratio)
    
    # 5. Shift values
    # Little Endian packing: first element at lowest bits
    shift_vals = torch.arange(0, 32, bits, device=tensor.device, dtype=torch.int32)
    
    # 6. Bitwise packing
    shifted = reshaped << shift_vals
    packed_int32 = torch.sum(shifted, dim=1).to(torch.int32)
    
    # 7. Reinterpret as float32
    packed_float32 = packed_int32.view(torch.float32)
    
    return packed_float32

def unpack_from_float32(packed_tensor: torch.Tensor, bits: int, shape: torch.Size) -> torch.Tensor:
    """
    Unpack float32 container back to uint8 tensor.
    """
    # 1. Reinterpret as int32
    packed_int32 = packed_tensor.view(torch.int32)
    
    # 2. Prepare unshift
    shift_vals = torch.arange(0, 32, bits, device=packed_tensor.device, dtype=torch.int32)
    
    # 3. Expand and shift right
    # packed_int32: [N] -> [N, 1]
    # shift_vals: [pack_ratio]
    # Broadcast -> [N, pack_ratio]
    unpacked = (packed_int32.unsqueeze(-1) >> shift_vals) & ((1 << bits) - 1)
    
    # 4. Flatten
    flat_tensor = unpacked.flatten()
    
    # 5. Remove padding (slice to original size)
    original_numel = shape.numel()
    flat_tensor = flat_tensor[:original_numel]
    
    # 6. Reshape to original shape and cast to uint8
    return flat_tensor.view(shape).to(torch.uint8)

def kivi_quantize(
    tensor: torch.Tensor, 
    axis: int, 
    nbits: int = NBITS,
    q_group_size: int = Q_GROUP_SIZE
) -> Tuple[torch.Tensor, dict]:
    """
    KIVI quantization implementation including packing.
    """
    assert axis in [2, 3], "axis should be 2 or 3"
    if tensor.numel() == 0:
        return tensor, None

    batch_size, num_heads, seq_len, head_dim = tensor.shape
    
    # Reshape based on grouping
    if axis == 2:
        # Check divisibility
        if seq_len % q_group_size != 0:
             # Truncate for speed test purposes if not divisible
             new_len = (seq_len // q_group_size) * q_group_size
             tensor = tensor[:, :, :new_len, :]
             seq_len = new_len
        
        tensor = tensor.reshape(batch_size, num_heads, seq_len // q_group_size, q_group_size, head_dim)
        calc_axis = -2
    elif axis == 3:
        # Check divisibility
        if head_dim % q_group_size != 0:
            raise ValueError(f"Head dim {head_dim} not divisible by group size {q_group_size}")

        tensor = tensor.reshape(batch_size, num_heads, seq_len, head_dim // q_group_size, q_group_size)
        calc_axis = -1

    max_value = 2 ** nbits
    # 1. 计算 Min/Max
    max_ = torch.amax(tensor, dim=calc_axis, keepdim=True)
    min_ = torch.amin(tensor, dim=calc_axis, keepdim=True)

    # 2. 计算 Scale (添加 eps 防止除零)
    scale = (max_ - min_).clamp_(min=1e-5).div_(max_value - 1) 
    
    # 3. 量化核心逻辑
    tensor_sub = tensor.sub(min_)
    quant_float = tensor_sub.div_(scale)
    quantized_tensor = quant_float.round_().clamp_(0, max_value).to(torch.uint8)

    # 4. 打包 (Packing)
    packed_tensor = pack_to_float32(quantized_tensor, nbits)

    # 5. Metadata
    meta_data = {
        "min_val": min_.to(tensor.dtype),
        "quant_scale": scale.to(tensor.dtype),
        "quantized_shape": quantized_tensor.shape # Store shape for unpacking
    }
    
    return packed_tensor, meta_data

def kivi_dequantize(
    packed_tensor: torch.Tensor,
    meta_data: dict,
    axis: int,
    nbits: int = NBITS,
    q_group_size: int = Q_GROUP_SIZE
) -> torch.Tensor:
    """
    KIVI dequantization including unpacking.
    """
    if meta_data is None:
        return packed_tensor

    # 1. 获取元数据
    min_val = meta_data["min_val"]
    quant_scale = meta_data["quant_scale"]
    quantized_shape = meta_data["quantized_shape"]

    # 2. 解包 (Unpacking)
    quantized_tensor = unpack_from_float32(packed_tensor, nbits, quantized_shape)

    # 3. 类型转换 (Cast)
    quant_float = quantized_tensor.to(quant_scale.dtype)

    # 4. 反量化计算
    dequantized_tensor = quant_float * quant_scale + min_val

    # 5. Reshape back
    if axis == 2:
        batch_size, num_heads, num_groups, group_size, head_dim = dequantized_tensor.shape
        dequantized_tensor = dequantized_tensor.reshape(batch_size, num_heads, num_groups * group_size, head_dim)
    elif axis == 3:
        batch_size, num_heads, seq_len, num_groups, group_size = dequantized_tensor.shape
        dequantized_tensor = dequantized_tensor.reshape(batch_size, num_heads, seq_len, num_groups * group_size)

    return dequantized_tensor

def main(args):
    # Unpack args
    INPUT_LENGTH = args.input_length
    WARMUP_ITER = args.warmup_iter
    BENCHMARK_ITER = args.benchmark_iter
    MODEL_NAME = args.model_name

    # 1. Load data, Model and Tokenizer
    print("Loading model and tokenizer...")
    try:
        dataset = load_dataset("Xnhyacinth/LongBench", "2wikimqa", split="test").to_pandas()
        max_length_idx = dataset['length'].idxmax()
        text = dataset.loc[max_length_idx, "context"]
    except Exception as e:
        print(f"Dataset load failed: {e}. Using dummy text.")
        text = "Hello world " * 1000

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

    model = AutoModelForCausalLM.from_pretrained(
        f"{BASE_MODEL_DIR}/{MODEL_NAME}",
        torch_dtype="auto",
        device_map="auto",
        use_cache=True,
        attn_implementation="flash_attention_2",
    )
    
    # 2. Perform a forward pass to get past_key_values
    print("Performing a forward pass to get past_key_values...")
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True)
        past_key_values = outputs.past_key_values

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Extract keys and values
    key_list = [kv[0].to("cuda:0") for kv in past_key_values]
    value_list = [kv[1].to("cuda:0") for kv in past_key_values]
    num_layers = len(key_list)
    print(f"Got past_key_values for {num_layers} layers.")
    
    original_size_gb = (sum(k.numel() * k.element_size() for k in key_list) + 
                        sum(v.numel() * v.element_size() for v in value_list)) / 1024**3

    # 3. Warmup
    print(f"Warming up for {WARMUP_ITER} iterations...")
    
    # Ensure divisible length for speed test
    b, h, s, d = key_list[0].shape
    if AXIS_KEY == 2 and s % Q_GROUP_SIZE != 0:
        new_s = (s // Q_GROUP_SIZE) * Q_GROUP_SIZE
        print(f"Truncating sequence length from {s} to {new_s} for divisibility by {Q_GROUP_SIZE}")
        key_list = [k[:, :, :new_s, :].contiguous() for k in key_list]
        value_list = [v[:, :, :new_s, :].contiguous() for v in value_list]
        s = new_s
    elif AXIS_VALUE == 2 and s % Q_GROUP_SIZE != 0:
        new_s = (s // Q_GROUP_SIZE) * Q_GROUP_SIZE
        print(f"Truncating sequence length from {s} to {new_s} for divisibility by {Q_GROUP_SIZE}")
        key_list = [k[:, :, :new_s, :].contiguous() for k in key_list]
        value_list = [v[:, :, :new_s, :].contiguous() for v in value_list]
        s = new_s

    # Update original size after truncation
    original_size_gb = (sum(k.numel() * k.element_size() for k in key_list) + 
                        sum(v.numel() * v.element_size() for v in value_list)) / 1024**3

    for _ in range(WARMUP_ITER):
        # Warmup Quantize
        q_keys = []
        q_values = []
        for i in range(num_layers):
            k_q, k_meta = kivi_quantize(key_list[i], axis=AXIS_KEY)
            v_q, v_meta = kivi_quantize(value_list[i], axis=AXIS_VALUE)
            q_keys.append((k_q, k_meta))
            q_values.append((v_q, v_meta))
        
        # Warmup Dequantize
        for i in range(num_layers):
            _ = kivi_dequantize(q_keys[i][0], q_keys[i][1], axis=AXIS_KEY)
            _ = kivi_dequantize(q_values[i][0], q_values[i][1], axis=AXIS_VALUE)
            
    torch.cuda.synchronize()

    # 4. Benchmarking
    print(f"Benchmarking for {BENCHMARK_ITER} iterations...")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # --- Benchmark Quantize ---
    quant_time = 0
    
    for _ in range(BENCHMARK_ITER):
        torch.cuda.synchronize()
        start_event.record()
        
        q_results = []
        for i in range(num_layers):
            k_q, k_meta = kivi_quantize(key_list[i], axis=AXIS_KEY)
            v_q, v_meta = kivi_quantize(value_list[i], axis=AXIS_VALUE)
            q_results.append(((k_q, k_meta), (v_q, v_meta)))
            
        end_event.record()
        torch.cuda.synchronize()
        quant_time += start_event.elapsed_time(end_event)

    # Store one set of quantized results for dequant benchmark
    quantized_data = []
    for i in range(num_layers):
        k_q, k_meta = kivi_quantize(key_list[i], axis=AXIS_KEY)
        v_q, v_meta = kivi_quantize(value_list[i], axis=AXIS_VALUE)
        quantized_data.append(((k_q, k_meta), (v_q, v_meta)))

    avg_quant_time = quant_time / BENCHMARK_ITER / 1000

    # --- Benchmark Dequantize ---
    dequant_time = 0
    
    for _ in range(BENCHMARK_ITER):
        torch.cuda.synchronize()
        start_event.record()
        
        for i in range(num_layers):
            (k_q, k_meta), (v_q, v_meta) = quantized_data[i]
            _ = kivi_dequantize(k_q, k_meta, axis=AXIS_KEY)
            _ = kivi_dequantize(v_q, v_meta, axis=AXIS_VALUE)
            
        end_event.record()
        torch.cuda.synchronize()
        dequant_time += start_event.elapsed_time(end_event)
        
    avg_dequant_time = dequant_time / BENCHMARK_ITER / 1000

    print("\n--- Benchmark Results ---")
    print(f"Input length: {INPUT_LENGTH}")
    print(f"Original size: {original_size_gb:.4f} GB")
    print(f"Prefill time: {avg_quant_time:.4f} s")
    print(f"Decode time: {avg_dequant_time:.4f} s")
    print(f"Prefill throughput: {original_size_gb / avg_quant_time:.2f} GB/s")
    print(f"Decode throughput: {original_size_gb / avg_dequant_time:.2f} GB/s")
    print("-------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KIVI Speed Benchmark")
    parser.add_argument("--input_length", type=int, default=1024, help="Input sequence length")
    parser.add_argument("--warmup_iter", type=int, default=100, help="Number of warmup iterations")
    parser.add_argument("--benchmark_iter", type=int, default=200, help="Number of benchmark iterations")
    parser.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct", help="Model name")
    args = parser.parse_args()
    main(args)
