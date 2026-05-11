import torch
import gc
import sys
import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from typing import List, Tuple
from collections import deque
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import argparse

# Add project root to sys.path
# Assuming file is in Infer_Comm/evaluation/component_speed/
# We need to add path to 'mxy' folder to import Infer_Comm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

try:
    from offline_search.src.cache.cache_utils import HadamardTransform
except ImportError:
    # Fallback just in case
    print("Could not import HadamardTransform from offline_search.src.cache.cache_utils. Checking path...")
    print(sys.path)
    raise

BASE_MODEL_DIR = "/root/workspace/models"

def main(args):
    # Unpack args
    INPUT_LENGTH = args.input_length
    WARMUP_ITER = args.warmup_iter
    BENCHMARK_ITER = args.benchmark_iter
    MODEL_NAME = args.model_name

    # 1. Load Model and Tokenizer
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

    # load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            f"{BASE_MODEL_DIR}/{MODEL_NAME}",
            torch_dtype="auto",
            device_map="auto",
            use_cache=True,
            attn_implementation="flash_attention_2",
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

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
    print(f"Got past_key_values for {len(key_list)} layers.")

    # 2. Prepare inputs for Hadamard Transform
    print("Preparing Hadamard inputs...")
    kv_inputs = []
    for i in range(len(key_list)):
        k, v = key_list[i], value_list[i]
        kv_inputs.append((k, v, i))

    hadamard_transformer = HadamardTransform(0x3333)

    # 3. Warmup
    print(f"Warming up for {WARMUP_ITER} iterations...")
    for _ in range(WARMUP_ITER):
        # Transform
        transformed_outputs = []
        for k, v, idx in kv_inputs:
            k_trans = hadamard_transformer.transform(k, idx)
            v_trans = hadamard_transformer.transform(v, idx)
            transformed_outputs.append((k_trans, v_trans, idx))
        
        # Inverse
        for k_trans, v_trans, idx in transformed_outputs:
            _ = hadamard_transformer.inverse(k_trans, idx)
            _ = hadamard_transformer.inverse(v_trans, idx)
            
    torch.cuda.synchronize()

    # 4. Benchmarking
    print(f"Benchmarking for {BENCHMARK_ITER} iterations...")
    transform_total_time = 0
    inverse_total_time = 0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Pre-generate outputs for inverse benchmark
    pre_transformed_inputs = []
    for k, v, idx in kv_inputs:
        k_trans = hadamard_transformer.transform(k, idx)
        v_trans = hadamard_transformer.transform(v, idx)
        pre_transformed_inputs.append((k_trans, v_trans, idx))

    # Measure Transform
    for _ in range(BENCHMARK_ITER):
        torch.cuda.synchronize()
        start_event.record()
        for k, v, idx in kv_inputs:
             _ = hadamard_transformer.transform(k, idx)
             _ = hadamard_transformer.transform(v, idx)
        end_event.record()
        torch.cuda.synchronize()
        transform_total_time += start_event.elapsed_time(end_event)

    # Measure Inverse
    for _ in range(BENCHMARK_ITER):
        torch.cuda.synchronize()
        start_event.record()
        for k_trans, v_trans, idx in pre_transformed_inputs:
             _ = hadamard_transformer.inverse(k_trans, idx)
             _ = hadamard_transformer.inverse(v_trans, idx)
        end_event.record()
        torch.cuda.synchronize()
        inverse_total_time += start_event.elapsed_time(end_event)

    avg_transform_time = transform_total_time / BENCHMARK_ITER / 1000
    avg_inverse_time = inverse_total_time / BENCHMARK_ITER / 1000
    
    print("\n--- Benchmark Results ---")
    print(f"Input length: {INPUT_LENGTH}")
    print(f"Original size: {original_size:.4f} GB")
    print(f"Prefill time: {avg_transform_time:.4f} s")
    print(f"Decode time: {avg_inverse_time:.4f} s")
    print(f"Prefill throughput: {original_size / avg_transform_time:.2f} GB/s")
    print(f"Decode throughput: {original_size / avg_inverse_time:.2f} GB/s")
    print("-------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hadamard Speed Benchmark")
    parser.add_argument("--input_length", type=int, default=1024, help="Input sequence length")
    parser.add_argument("--warmup_iter", type=int, default=100, help="Number of warmup iterations")
    parser.add_argument("--benchmark_iter", type=int, default=200, help="Number of benchmark iterations")
    parser.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct", help="Model name")
    args = parser.parse_args()
    main(args)

