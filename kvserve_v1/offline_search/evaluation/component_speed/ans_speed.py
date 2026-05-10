import torch
import gc
import pandas as pd
import numpy as np
from datasets import load_dataset
from typing import List, Tuple
import time
import sys
import os
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
import cupy as cp
from nvidia import nvcomp
import argparse

# Add path for nvcomp_wrapper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
try:
    from Infer_Comm.evaluation.compression_ratio.nvcomp_wrapper import NVCompWrapper
except ImportError:
    # Fallback if the path resolution fails or dependency missing, though user says it exists
    print("Error importing nvcomp_wrapper. Please ensure the path is correct.")
    sys.exit(1)

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
    max_ = torch.amax(tensor, dim=axis, keepdim=True)
    min_ = torch.amin(tensor, dim=axis, keepdim=True)

    # 2. 计算 Scale (添加 eps 防止除零)
    scale = (max_ - min_).clamp_(min=1e-5).div_(max_value - 1) 
    
    # 3. 量化核心逻辑
    tensor_sub = tensor.sub(min_)
    quant_float = tensor_sub.div_(scale)
    quantized_tensor = quant_float.round_().clamp_(0, max_value).to(torch.uint8)

    meta_data = {
        "min_val": min_.to(tensor.dtype),
        "quant_scale": scale.to(tensor.dtype),
    }
    
    return quantized_tensor, meta_data

def main(args):
    # Unpack args
    INPUT_LENGTH = args.input_length
    WARMUP_ITER = args.warmup_iter
    BENCHMARK_ITER = args.benchmark_iter
    MODEL_NAME = args.model_name

    # 1. Load Model and Tokenizer
    print("Loading model and tokenizer...")
    dataset = load_dataset("Xnhyacinth/LongBench", "2wikimqa", split="test").to_pandas()
    max_length_idx = dataset['length'].idxmax()
    text = dataset.loc[max_length_idx, "context"]
    tokenizer = AutoTokenizer.from_pretrained(f"{BASE_MODEL_DIR}/{MODEL_NAME}")
    inputs = tokenizer(text, return_tensors="pt")
    current_length = inputs["input_ids"].shape[1]
    if current_length > INPUT_LENGTH:
        inputs["input_ids"] = inputs["input_ids"][:, :INPUT_LENGTH]
        inputs["attention_mask"] = inputs["attention_mask"][:, :INPUT_LENGTH]
    elif current_length < INPUT_LENGTH:
        num_repeats = (INPUT_LENGTH + current_length - 1) // current_length
        inputs["input_ids"] = inputs["input_ids"].repeat(1, num_repeats)[:, :INPUT_LENGTH]
        inputs["attention_mask"] = inputs["attention_mask"].repeat(1, num_repeats)[:, :INPUT_LENGTH]
    inputs = inputs.to("cuda:0")
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        f"{BASE_MODEL_DIR}/{MODEL_NAME}",
        torch_dtype="auto",
        device_map="auto",
        use_cache=True,
        attn_implementation="flash_attention_2",
    )

    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True)
        past_key_values = outputs.past_key_values
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    key_list = [kv[0].to("cuda:0") for kv in past_key_values]
    value_list = [kv[1].to("cuda:0") for kv in past_key_values]
    print(f"Got past_key_values for {len(key_list)} layers.")

    # 3. Load scores and create mask
    try:
        df = pd.read_csv(f"../../duo_config/{MODEL_NAME}_scores.csv", header=None).dropna()
    except FileNotFoundError:
        print("Warning: score file not found. Generating a random mask for demonstration.")
        num_layers = len(key_list)
        num_heads = key_list[0].shape[1]
        df = pd.DataFrame(np.random.rand(num_layers, num_heads))

    scores = torch.tensor(df.values, dtype=torch.float32).to('cuda:0')
    pruned_num = round(scores.numel() * LOW_PRESS_RATIO)
    scores_mask = torch.zeros_like(scores, dtype=torch.bool).to('cuda:0')
    flat_indices = torch.argsort(scores.flatten())[:pruned_num]
    multi_indices = torch.unravel_index(flat_indices, scores.shape)
    scores_mask[multi_indices] = True
    
    # 4. Prepare inputs for nvcomp
    print("Preparing quantization and nvcomp inputs...")
    key_blocks = []
    value_blocks = []

    for i in range(len(key_list)):
        layer_mask = scores_mask[i].to('cuda:0')
        k, v = key_list[i], value_list[i]
        
        low_keys = k[:, layer_mask, :, :]
        low_values = v[:, layer_mask, :, :]
        high_keys = k[:, ~layer_mask, :, :]
        high_values = v[:, ~layer_mask, :, :]
        
        # Quantize to get uint8 data
        high_keys_q, _ = quantization(16, high_keys, [2])
        high_values_q, _ = quantization(16, high_values, [1, 3])            
        low_keys_q, _ = quantization(8, low_keys, [2])
        low_values_q, _ = quantization(8, low_values, [1, 3])
        
        # Concatenate to form a layer block (Batch, Heads, Tokens, Channels)
        # Note: dim 1 is heads. 
        keys_block = torch.cat([low_keys_q, high_keys_q], dim=1)
        values_block = torch.cat([low_values_q, high_values_q], dim=1)
        
        key_blocks.append(keys_block)
        value_blocks.append(values_block)

    # Concatenate all layers along batch dimension (dim 0) as per custom_cr.py approach
    # Shape: (Batch * Layers, Heads, Tokens, Channels)
    all_keys = torch.cat(key_blocks, dim=0)
    all_values = torch.cat(value_blocks, dim=0)
    
    print(f"Data prepared. Keys shape: {all_keys.shape}, Values shape: {all_values.shape}, Dtype: {all_keys.dtype}")

    # Init NVComp
    nvcomp_wrapper = NVCompWrapper("ANS", data_type="|u1")

    # 5. Warmup
    print(f"Warming up for {WARMUP_ITER} iterations...")
    
    # Prepare data for core benchmark warmup
    tensor = torch.cat([all_keys, all_values], dim=0)
    tensor_bytes = tensor.flatten().view(torch.uint8).contiguous()
    original_size = len(pickle.dumps(tensor_bytes)) / 1024 / 1024 / 1024
    # nv_array = nvcomp.as_array(tensor_bytes)
    nv_array = cp.asarray(tensor_bytes)

    # Generate one compressed instance for decode input
    compressed_ref = nvcomp_wrapper.compress(all_keys, all_values)
    comp_cupy = cp.frombuffer(compressed_ref.buffer, dtype=cp.uint8)
    # comp_buffer_nv = nvcomp.as_array(comp_cupy)
    comp_buffer_nv = cp.asarray(comp_cupy)
    # print(f"Total size of compressed data: {len(pickle.dumps(tensor_bytes)) / 1024 / 1024} bytes")

    for _ in range(WARMUP_ITER):
        # Core codec
        # Encode
        _ = nvcomp_wrapper.codec.encode(nv_array)
        # Decode
        _ = nvcomp_wrapper.codec.decode(comp_buffer_nv)
            
    torch.cuda.synchronize()

    # 6. Benchmarking
    print(f"Benchmarking for {BENCHMARK_ITER} iterations...")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # --- Benchmark Core Encode ---
    # Pre-prepare inputs
    # (Reuse tensor_bytes/nv_array from warmup section)
    
    compress_core_time = 0
    torch.cuda.synchronize()

    for _ in range(BENCHMARK_ITER):
        torch.cuda.synchronize()
        start_event.record()
        
        _ = nvcomp_wrapper.codec.encode(nv_array)
        
        end_event.record()
        # end_event.synchronize()
        torch.cuda.synchronize()
        compress_core_time += start_event.elapsed_time(end_event)

    # --- Benchmark Core Decode ---
    # Pre-prepare inputs
    # (Reuse comp_buffer_nv from warmup section)

    decompress_core_time = 0
    torch.cuda.synchronize()

    for _ in range(BENCHMARK_ITER):
        torch.cuda.synchronize()
        start_event.record()
        
        _ = nvcomp_wrapper.codec.decode(comp_buffer_nv)
        
        end_event.record()
        # end_event.synchronize()
        torch.cuda.synchronize()
        decompress_core_time += start_event.elapsed_time(end_event)


    avg_core_comp = compress_core_time / BENCHMARK_ITER / 1000
    avg_core_decomp = decompress_core_time / BENCHMARK_ITER / 1000
    
    print("\n--- Benchmark Results ---")
    print(f"Input length: {INPUT_LENGTH}")
    print(f"Original size: {original_size:.4f} GB")
    print(f"Prefill time: {avg_core_comp:.4f} s")
    print(f"Decode time: {avg_core_decomp:.4f} s")
    print(f"Prefill throughput: {original_size / avg_core_comp:.2f} GB/s")
    print(f"Decode throughput: {original_size / avg_core_decomp:.2f} GB/s")
    print("-------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NVComp Speed Benchmark")
    parser.add_argument("--input_length", type=int, default=1024, help="Input sequence length")
    parser.add_argument("--warmup_iter", type=int, default=100, help="Number of warmup iterations")
    parser.add_argument("--benchmark_iter", type=int, default=200, help="Number of benchmark iterations")
    parser.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct", help="Model name")
    args = parser.parse_args()
    main(args)
