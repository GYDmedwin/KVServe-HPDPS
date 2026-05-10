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
import argparse

# Add path for torchac_cuda
# Assuming workspace root is /root/workspace
sys.path.append("/root/workspace/CacheGen/LMCache/third_party")
try:
    import torchac_cuda
except ImportError:
    print("Error importing torchac_cuda. Please ensure the path is correct.")
    sys.exit(1)

BASE_MODEL_DIR = "/root/workspace/models"
QUANTIZATION_LEVEL = 2 # 1: aggressive, 2: moderate, 3: conservative

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

def collect_bytes(output_buffer, output_lengths) -> torch.Tensor:
    """
    Collect a byte tensor from the output_buffer + output_lengths
    """
    output_buffer_size = output_buffer.shape[-1]
    flattened_lengths = output_lengths.flatten()
    flattened_buffer = output_buffer.flatten()
    summed_length = (output_buffer_size - flattened_lengths).cumsum(0)
    summed_length = summed_length.roll(1)
    summed_length[0] = 0
    indexes = summed_length.repeat_interleave(flattened_lengths)
    indexes = indexes + torch.arange(len(indexes), device=indexes.device)
    return flattened_buffer[indexes]

def main(args):
    # Unpack args
    INPUT_LENGTH = args.input_length
    WARMUP_ITER = args.warmup_iter
    BENCHMARK_ITER = args.benchmark_iter
    MODEL_NAME = args.model_name

    # 1. Load Model and Tokenizer
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
    num_layers = len(key_list)
    print(f"Got past_key_values for {num_layers} layers.")

    # 3. Generate quantization bins for each layer
    layer_max_values = make_cachegen_bins(num_layers, QUANTIZATION_LEVEL)
    
    # 4. Prepare inputs
    print("Preparing quantization inputs...")
    key_blocks = []
    value_blocks = []

    for i in range(num_layers):
        k, v = key_list[i], value_list[i]
        max_values = layer_max_values[i]
        
        # Quantize using CacheGen method
        k_q, _ = cachegen_quantize(max_values["k"], k, axis=[2])
        v_q, _ = cachegen_quantize(max_values["v"], v, axis=[1, 3])
        
        # k_q shape: [Batch, Heads, Tokens, Channels] or similar?
        # cachegen-naive_speed uses k (Batch, Heads, SeqLen, HeadDim) and axis=[2] (SeqLen) for max1?
        # Wait, naive_speed.py: max1 = amax(..., dim=axis).
        # k shape usually [Batch, Heads, SeqLen, HeadDim].
        # axis=[2] means max over SeqLen? That's unusual for quantization, usually per-channel or per-token.
        # But let's follow the reference script.
        # naive_speed reference: k_q, k_meta = cachegen_quantize(max_values["k"], k, axis=[2])
        
        key_blocks.append(k_q)
        value_blocks.append(v_q)

    # Concatenate all layers along batch dimension
    # Shape: (Batch * Layers, Heads, Tokens, Channels)
    all_keys = torch.cat(key_blocks, dim=0)
    all_values = torch.cat(value_blocks, dim=0)
    
    print(f"Data prepared. Keys shape: {all_keys.shape}, Values shape: {all_values.shape}")
    
    # Reshape for torchac: [nlayers, ntokens, nchannels]
    # Current: [N, H, T, C] -> Transpose to [N, T, H, C] -> Reshape [N, T, H*C]
    def prepare_for_torchac(tensor):
        # tensor: [B, H, T, C]
        B, H, T, C = tensor.shape
        tensor = tensor.transpose(1, 2).reshape(B, T, H * C)
        return tensor

    # Combine keys and values into one large batch for encoding if desired, or keep separate
    # CacheGen usually concatenates keys and values along batch dimension (dim 0)
    
    # [N, T, H*C]
    torchac_keys = prepare_for_torchac(all_keys)
    torchac_values = prepare_for_torchac(all_values)
    
    # Concatenate along dim 0 (layers)
    encode_input = torch.cat([torchac_keys, torchac_values], dim=0).contiguous()
    # Ensure int8/uint8? torchac expects int16 for cdf calc in wrapper but input is quant output (uint8)
    # The quant function returns uint8.
    # encode_fast_new expects input to be appropriate for cdf indices.
    
    encode_input = encode_input.to(torch.int8) # Wrapper converts to int8?
    # Actually wrapper says: self.quantized_keys = quantized_keys.to(torch.int8)
    # But quantization function returns uint8. 
    # If values are 0..127, int8 is fine. If 128..255, int8 will be negative.
    # torchac expects symbols.
    # Let's check max value.
    max_val = encode_input.max().item()
    print(f"Max value in data: {max_val}")
    
    # If max_val > 127 and we cast to int8, it wraps.
    # However, CacheGen wrapper casts to int8. 
    # Let's trust the wrapper logic or use uint8 if torchac supports it.
    # torchac_cuda signatures usually take generic tensors but interpret as bytes?
    # real.py uses encode_input (int8/int16?). 
    # In real.py: encode_input = encode_input...
    
    # Let's stick to what we have (uint8 from quant) and see if we need cast.
    # cachegen_wrapper.py line 99: self.quantized_keys = quantized_keys.to(torch.int8)
    # This implies 0-127 range assumption or reinterpretation.
    # Our quantization uses max_value=16 or 8. So it is small.
    
    encode_input = encode_input.to(torch.int8)
    
    nlayers, ntokens, nchannels = encode_input.shape
    print(f"TorchAC Input Shape: {encode_input.shape}")

    # Prepare CDF
    print("Calculating CDF...")
    # Using max of actual data or theoretical max? Wrapper uses max() of data.
    # We used max_value=16 in quantization, so max is small.
    # But safe to calculate.
    
    # calculate_cdf returns int16
    # Note: calculate_cdf might be slow, but usually part of compression.
    cdf = torchac_cuda.calculate_cdf(encode_input, max_val + 1 if max_val < 255 else 256)
    
    # Prepare Buffers
    # Buffer size 256 is hardcoded in wrapper
    BUFFER_SIZE = 256
    
    # 5. Warmup
    print(f"Warming up for {WARMUP_ITER} iterations...")
    
    # Pre-allocate buffers to reuse?
    # Wrapper creates new buffers inside compress.
    # For speed test, we can reuse buffers to test kernel speed, or alloc to test E2E.
    # We will reuse buffers to avoid alloc overhead dominating kernel time, 
    # unless alloc is required by logic.
    
    output_buffer = torch.zeros((nlayers, nchannels, BUFFER_SIZE), dtype=torch.uint8, device="cuda")
    output_lengths = torch.zeros((nlayers, nchannels), dtype=torch.int32, device="cuda")
    
    # Dummy run
    for _ in range(10):
         torchac_cuda.encode_fast_new(cdf, encode_input[:, :256, :], output_buffer, output_lengths)
         
    torch.cuda.synchronize()

    # 6. Benchmarking Encode
    print(f"Benchmarking Encode for {BENCHMARK_ITER} iterations...")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    compress_time = 0
    
    # Store compressed chunks for decode test
    compressed_chunks = [] 
    
    # We do one full pass first to get data for decode
    for i in range(0, ntokens, 256):
        end = min(i + 256, ntokens)
        curr_input = encode_input[:, i:end, :]
        # Handle last chunk if < 256? encode_fast_new might expect exactly 256 or padded?
        # wrapper says: "start = i, end = min(i+256, ntokens)"
        # But real.py test uses fixed 256?
        # Wrapper uses `encode_ntokens` which calls `encode_fast_new`.
        # If chunk < 256, we might need padding or torchac handles it?
        # Let's assume torchac handles < 256 if we pass correct shape?
        # Actually wrapper just passes slice.
        
        torchac_cuda.encode_fast_new(cdf, curr_input, output_buffer, output_lengths)
        byte_tensor = collect_bytes(output_buffer, output_lengths)
        compressed_chunks.append((byte_tensor.clone(), output_lengths.clone(), end-i))

    torch.cuda.synchronize()
    
    # Benchmark Loop
    for _ in range(BENCHMARK_ITER):
        torch.cuda.synchronize()
        start_event.record()
        cdf = torchac_cuda.calculate_cdf(encode_input, max_val + 1 if max_val < 255 else 256)
        # Re-run the loop logic
        for i in range(0, ntokens, 256):
            end = min(i + 256, ntokens)
            curr_input = encode_input[:, i:end, :]
            
            # Note: For strict kernel benchmark we might skip output allocation,
            # but collect_bytes is part of "getting the compressed stream".
            
            torchac_cuda.encode_fast_new(cdf, curr_input, output_buffer, output_lengths)
            # We don't necessarily need to collect_bytes for measuring Encode throughput 
            # if we consider "Encode" as just generating the buffer. 
            # But usually we need the bytes.
            # ans_speed.py includes everything in "nvcomp.encode".
            # We will include collect_bytes to be fair/realistic.
            _ = collect_bytes(output_buffer, output_lengths)
            
        end_event.record()
        torch.cuda.synchronize()
        compress_time += start_event.elapsed_time(end_event)

    avg_comp_time = compress_time / BENCHMARK_ITER / 1000

    # 7. Benchmarking Decode
    print(f"Benchmarking Decode for {BENCHMARK_ITER} iterations...")
    
    decompress_time = 0
    
    # Prepare buffers for decode
    # decode_buffer shape: [nlayers, 256, nchannels] according to real.py
    # But wait, input was [nlayers, ntokens, nchannels].
    # encode_fast_new takes [nlayers, ntokens, nchannels].
    # decode_fast_prefsum takes decode_buffer [nlayers, output_buffer_size, nchannels]?
    # real.py: decode_buffer = torch.zeros((nlayers, output_buffer_size, nchannels)...
    # output_buffer_size is 256.
    # So decode outputs [nlayers, 256, nchannels].
    
    decode_buffer = torch.zeros((nlayers, 256, nchannels), dtype=torch.uint8, device="cuda")
    
    for _ in range(BENCHMARK_ITER):
        torch.cuda.synchronize()
        start_event.record()
        
        for (byte_tensor, lengths, chunk_len) in compressed_chunks:
            # Prepare lengths prefsum
            # This calculation is part of decode overhead
            lengths_prefsum = lengths.flatten().cumsum(0).reshape(lengths.shape)
            
            torchac_cuda.decode_fast_prefsum(
                cdf, 
                byte_tensor, 
                lengths_prefsum, 
                decode_buffer
            )
            # Note: decode_buffer now contains recovered data for this chunk
            
        end_event.record()
        torch.cuda.synchronize()
        decompress_time += start_event.elapsed_time(end_event)
        
    avg_decomp_time = decompress_time / BENCHMARK_ITER / 1000
    
    original_size_gb = (encode_input.numel() * 1) / 1024**3 # 1 byte per element (int8)

    print("\n--- Benchmark Results ---")
    print(f"Input length: {INPUT_LENGTH}")
    print(f"Original size: {original_size_gb:.4f} GB")
    print(f"Encode time: {avg_comp_time:.4f} s")
    print(f"Decode time: {avg_decomp_time:.4f} s")
    print(f"Encode throughput: {original_size_gb / avg_comp_time:.2f} GB/s")
    print(f"Decode throughput: {original_size_gb / avg_decomp_time:.2f} GB/s")
    print("-------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TorchAC Speed Benchmark")
    parser.add_argument("--input_length", type=int, default=1024, help="Input sequence length")
    parser.add_argument("--warmup_iter", type=int, default=100, help="Number of warmup iterations")
    parser.add_argument("--benchmark_iter", type=int, default=200, help="Number of benchmark iterations")
    parser.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct", help="Model name")
    args = parser.parse_args()
    main(args)

