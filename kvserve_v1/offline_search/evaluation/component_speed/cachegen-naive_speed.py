import torch
import gc
import pandas as pd
import numpy as np
from datasets import load_dataset
from typing import List, Tuple
from collections import deque
import time
import pickle
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

QUANTIZATION_LEVEL = 1 # 1: aggressive, 2: moderate, 3: conservative
BASE_MODEL_DIR = "/root/workspace/models"

def cachegen_quantize(
    max_value: int,
    tensor: torch.Tensor,
    axis: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantizes a key/value using the CacheGen method."""
    if tensor.numel() == 0:
        return tensor, None

    MAX = (max_value // 2 - 1)
    # Clamp to prevent division by zero for empty tensors
    max1 = torch.amax(torch.abs(tensor), dim=axis, keepdim=True).clamp_(min=1e-5)
    factor = MAX / max1
    xq = torch.round(tensor * factor + MAX).to(torch.int8)

    return xq, max1

def cachegen_dequantize(
    max_value: int,
    quantized_tensor: torch.Tensor,
    max1: torch.Tensor,
) -> torch.Tensor:
    """Dequantizes back the tensor that was quantized by `cachegen_quantize()`"""
    if max1 is None:
        return quantized_tensor
        
    C = (max_value // 2 - 1)
    t = quantized_tensor - C
    t = t / C
    t = t.to(max1.dtype) * max1
    return t

def make_cachegen_bins(num_layers: int, quantization_level: int):
    """
    Generates quantization max_values for each layer based on CacheGen's strategy.
    """
    layer_max_value = []
    
    # Default values from CacheGenCacheConfig
    high_max_value = 32
    mid_max_value = 16
    low_max_value = 12

    # Layer boundary definitions from CacheGen
    key_first_boundary = int(num_layers * 0.32)
    key_second_boundary = int(num_layers * 0.63)
    val_first_boundary = max(2, int(num_layers * 0.15))

    for i in range(num_layers):
        # Key quantization strategy
        if quantization_level == 1:
            k_max = mid_max_value if i < key_second_boundary else low_max_value
        elif quantization_level == 2:
            k_max = high_max_value if i < key_first_boundary else mid_max_value
        else:
            k_max = high_max_value

        # Value quantization strategy
        if quantization_level == 1:
            v_max = mid_max_value if i < val_first_boundary else low_max_value
        elif quantization_level == 2:
            v_max = high_max_value if i < val_first_boundary else mid_max_value
        else:
            v_max = high_max_value

        layer_max_value.append({"k": k_max, "v": v_max})

    return layer_max_value

def main(args):
    # Unpack args
    INPUT_LENGTH = args.input_length
    WARMUP_ITER = args.warmup_iter
    BENCHMARK_ITER = args.benchmark_iter
    MODEL_NAME = args.model_name

    # 1. Load data, Model and Tokenizer
    print("Loading model and tokenizer...")
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
    original_size = len(pickle.dumps(past_key_values)) / 1024 / 1024 / 1024
    key_list = [kv[0].to("cuda:0") for kv in past_key_values]
    value_list = [kv[1].to("cuda:0") for kv in past_key_values]
    num_layers = len(key_list)
    print(f"Got past_key_values for {num_layers} layers.")

    # 3. Generate quantization bins for each layer
    layer_max_values = make_cachegen_bins(num_layers, QUANTIZATION_LEVEL)
    
    # 4. Prepare inputs for quantization before benchmarking
    # This is simpler as we don't split heads, we just need the tensors and their layer-specific max_values
    quant_inputs = []
    for i in range(num_layers):
        k, v = key_list[i], value_list[i]
        max_values = layer_max_values[i]
        quant_inputs.append((k, v, max_values))

    # 5. Warmup
    print(f"Warming up for {WARMUP_ITER} iterations...")
    for _ in range(WARMUP_ITER):
        quantized_outputs = []
        for k, v, max_values in quant_inputs:
            k_q, k_meta = cachegen_quantize(max_values["k"], k, axis=[2])
            v_q, v_meta = cachegen_quantize(max_values["v"], v, axis=[1, 3])
            quantized_outputs.append(((k_q, k_meta), (v_q, v_meta), max_values))
        
        for (k_q, k_meta), (v_q, v_meta), max_values in quantized_outputs:
            _ = cachegen_dequantize(max_values["k"], k_q, k_meta)
            _ = cachegen_dequantize(max_values["v"], v_q, v_meta)
    torch.cuda.synchronize()

    # 6. Benchmarking
    print(f"Benchmarking for {BENCHMARK_ITER} iterations...")
    quant_total_time = 0
    dequant_total_time = 0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # --- Quantization timing ---
    quantized_outputs = []
    
    for _ in range(BENCHMARK_ITER):
        torch.cuda.synchronize()
        start_event.record()
        for k, v, max_values in quant_inputs:
            k_q, k_meta = cachegen_quantize(max_values["k"], k, axis=[2])
            v_q, v_meta = cachegen_quantize(max_values["v"], v, axis=[1, 3])
        end_event.record()
        torch.cuda.synchronize()
        quant_total_time += start_event.elapsed_time(end_event)
    for k, v, max_values in quant_inputs:
        k_q, k_meta = cachegen_quantize(max_values["k"], k, axis=[2])
        v_q, v_meta = cachegen_quantize(max_values["v"], v, axis=[1, 3])
        quantized_outputs.append(((k_q, k_meta), (v_q, v_meta), max_values))

    # --- Dequantization timing ---
    for _ in range(BENCHMARK_ITER):
        torch.cuda.synchronize()
        start_event.record()
        for (k_q, k_meta), (v_q, v_meta), max_values in quantized_outputs:
            _ = cachegen_dequantize(max_values["k"], k_q, k_meta)
            _ = cachegen_dequantize(max_values["v"], v_q, v_meta)            
        end_event.record()
        torch.cuda.synchronize()
        dequant_total_time += start_event.elapsed_time(end_event)

    avg_quant_time = quant_total_time / BENCHMARK_ITER / 1000
    avg_dequant_time = dequant_total_time / BENCHMARK_ITER / 1000
    
    print("\n--- Benchmark Results ---")
    # print(f"Quantization Level: {QUANTIZATION_LEVEL}")
    print(f"Input length: {INPUT_LENGTH}")
    print(f"Original size: {original_size:.4f} GB")
    print(f"Prefill time: {avg_quant_time:.4f} s")
    print(f"Decode time: {avg_dequant_time:.4f} s")
    print(f"Prefill throughput: {original_size / avg_quant_time:.2f} GB/s")
    print(f"Decode throughput: {original_size / avg_dequant_time:.2f} GB/s")
    print("-------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CacheGen Speed Benchmark")
    parser.add_argument("--input_length", type=int, default=1024, help="Input sequence length")
    parser.add_argument("--warmup_iter", type=int, default=100, help="Number of warmup iterations")
    parser.add_argument("--benchmark_iter", type=int, default=200, help="Number of benchmark iterations")
    parser.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct", help="Model name")
    args = parser.parse_args()
    main(args)


