import json
import logging
import os
import sys
import pandas as pd
import torch
import argparse
import gc
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from offline_search.src.cache.cachegen_utils import CacheGenCacheConfig, CacheGenCache
from nvcomp_wrapper import CompressedTensor, PackedData, TensorData, to_device, NVCompWrapper
from cachegen_wrapper import CacheGenWrapper, reshape_tensor


BASE_MODEL_PATH = "/root/workspace/models"
BASE_CONFIG_PATH = "/root/workspace/Infer_Comm/duo_config"


parser = argparse.ArgumentParser(description="Evaluate kv cache compression ratio.")
parser.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct", help="Name of the model to evaluate.")
parser.add_argument("--quantization_level", type=int, default=2, help="Quantization level in cachegen to choose.")
parser.add_argument("--high_max_value", type=int, default=32, help="High max value in quantization.")
parser.add_argument("--mid_max_value", type=int, default=16, help="Mid max value in quantization.")
parser.add_argument("--low_max_value", type=int, default=12, help="Low max value in quantization.")
parser.add_argument("--input_length", type=int, default=10240, help="Desired input length for the model.")

args = parser.parse_args()


print("Evaluating the model...")

# load dataset and sample a text
dataset = load_dataset("Xnhyacinth/LongBench", "qasper", split="test").to_pandas()
# Select the longest data based on the 'length' field
max_length_idx = dataset['length'].idxmax()
text = dataset.loc[max_length_idx, "context"]

# load model config
model_config = AutoConfig.from_pretrained(f"{BASE_MODEL_PATH}/{args.model_name}")

# load tokenizer and encode the text
tokenizer = AutoTokenizer.from_pretrained(f"{BASE_MODEL_PATH}/{args.model_name}")
# truncate the input length if specified
inputs = tokenizer(text, return_tensors="pt")
if args.input_length is not None:
    input_ids = inputs["input_ids"]
    current_length = input_ids.shape[1]

    if current_length > args.input_length:
        inputs["input_ids"] = input_ids[:, :args.input_length]
        inputs["attention_mask"] = inputs["attention_mask"][:, :args.input_length]
    elif current_length < args.input_length:
        num_repeats = (args.input_length + current_length - 1) // current_length
        inputs["input_ids"] = input_ids.repeat(1, num_repeats)[:, :args.input_length]
        inputs["attention_mask"] = inputs["attention_mask"].repeat(1, num_repeats)[:, :args.input_length]
inputs = inputs.to("cuda")
print(f"Input length: {inputs['input_ids'].shape[1]}")

# load custom cache config
cache_config = CacheGenCacheConfig(
    model_layers=model_config.num_hidden_layers,
    quantization_level=args.quantization_level,
    high_max_value=args.high_max_value,
    mid_max_value=args.mid_max_value,
    low_max_value=args.low_max_value,
)
past_key_values = CacheGenCache(cache_config=cache_config)

# load model
model = AutoModelForCausalLM.from_pretrained(
    f"{BASE_MODEL_PATH}/{args.model_name}",
    torch_dtype="auto",
    device_map="auto",
    use_cache=True,
    # output_attentions=True,
    attn_implementation="flash_attention_2",
)

# evaluate the model and run the generation

model.eval()
outputs = model.generate(
    **inputs,
    max_new_tokens=1,
    return_dict_in_generate=True,
    past_key_values=past_key_values,
)
# release model memory
# del model, inputs, outputs
# gc.collect()
# torch.cuda.empty_cache()

# config the original tensors and meta data
device = "cuda"
meta_data = []
num_layers = model_config.num_hidden_layers
num_heads = model_config.num_key_value_heads
head_dim = model_config.hidden_size // model_config.num_attention_heads
current_idx = 0
to_compressed_key_tensors = None
to_compressed_value_tensors = None
original_kv_shape = (1, num_heads, inputs["input_ids"].shape[1], head_dim)
original_key_tensors = [torch.empty(original_kv_shape, dtype=model_config.torch_dtype, device=device) for _ in range(num_layers)]
original_value_tensors = [torch.empty(original_kv_shape, dtype=model_config.torch_dtype, device=device) for _ in range(num_layers)]

for i in range(num_layers):
    keys_quantized, keys_meta_data = past_key_values._quantized_key_cache[i]
    values_quantized, values_meta_data = past_key_values._quantized_value_cache[i]

    # concat the original tensors and meta data in each layer
    keys_quantized = to_device(keys_quantized, device)
    values_quantized = to_device(values_quantized, device)

    if to_compressed_key_tensors is None and to_compressed_value_tensors is None:
        batch_size, heads, tokens, channels = keys_quantized.shape
        
        # 预分配最终形状
        final_shape = (num_layers*batch_size, heads, tokens, channels)
        
        # 使用 empty 分配（不初始化值，速度快）
        to_compressed_key_tensors = torch.empty(final_shape, dtype=keys_quantized.dtype, device=device)
        to_compressed_value_tensors = torch.empty(final_shape, dtype=values_quantized.dtype, device=device)

    # 4. 将当前层数据“填入”大 Tensor 的对应位置
    # 使用切片赋值，避免复制整个大 Tensor
    end_idx = current_idx + keys_quantized.shape[0]
    to_compressed_key_tensors[current_idx:end_idx, ...] = keys_quantized
    to_compressed_value_tensors[current_idx:end_idx, ...] = values_quantized
    current_idx = end_idx
    
    # 5. 收集元数据并释放临时层块
    meta_data.append([keys_meta_data, values_meta_data])
    # del keys_quantized, values_quantized, keys_meta_data, values_meta_data
    # gc.collect()
    # torch.cuda.empty_cache()

# move the meta data to the device
meta_data = to_device(meta_data, device)

# calculate the original size
original_size = (len(pickle.dumps(original_key_tensors)) + len(pickle.dumps(original_value_tensors))) / 1024 / 1024

# load nvcomp wrapper
# nvcomp_wrapper = NVCompWrapper("ANS", data_type="|u1")
# # compress the original tensors and pack them
# compressed_data = nvcomp_wrapper.compress(to_compressed_key_tensors, to_compressed_value_tensors)
# packed_data = PackedData(compressed_data, meta_data)

# load cachegen wrapper
cachegen_wrapper = CacheGenWrapper(
    quantized_keys=to_compressed_key_tensors,
    quantized_values=to_compressed_value_tensors,
    meta_data=meta_data,
)
# compress the original tensors and pack them
packed_data = cachegen_wrapper.compress()
# calculate the compressed size
compressed_size = len(pickle.dumps(packed_data)) / 1024 / 1024
print(f"\nTotal Original size: {original_size:.2f} MB\nTotal Compressed size: {compressed_size:.2f} MB\nTotal Compression ratio: {original_size / compressed_size:.4f}")

