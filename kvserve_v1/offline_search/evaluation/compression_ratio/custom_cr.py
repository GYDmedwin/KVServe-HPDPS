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
from Infer_Comm.src.cache.cache_utils import CustomCacheConfig, CustomCache
from nvcomp_wrapper import NVCompWrapper, CompressedTensor, PackedData, TensorData, to_device
from cachegen_wrapper import CacheGenWrapper


BASE_MODEL_PATH = "/root/workspace/models"
BASE_CONFIG_PATH = "/root/workspace/Infer_Comm/duo_config"


parser = argparse.ArgumentParser(description="Evaluate kv cache compression ratio.")
parser.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct", help="Name of the model to evaluate.")
parser.add_argument("--transform_type", type=str, default="none", choices=["none", "hadamard"], help="Type of transform to use.")
parser.add_argument("--heads_selection", type=float, default=0.5, help="Heads selection ratio.")
parser.add_argument("--high_key_max_value", type=int, default=16, help="Max value for high key.")
parser.add_argument("--high_value_max_value", type=int, default=16, help="Max value for high value.")
parser.add_argument("--low_key_max_value", type=int, default=8, help="Max value for low key.")
parser.add_argument("--low_value_max_value", type=int, default=8, help="Max value for low value.")
parser.add_argument("--axis_key", type=int, nargs='+', default=[2], help="Axis for key.")
parser.add_argument("--axis_value", type=int, nargs='+', default=[1, 3], help="Axis for value.")
parser.add_argument("--input_length", type=int, default=10240, help="Desired input length for the model.")

args = parser.parse_args()


print("Evaluating the model...")

# load scores
df = pd.read_csv(f"{BASE_CONFIG_PATH}/{args.model_name}_scores.csv", header=None).dropna()
scores = torch.tensor(df.values, dtype=torch.float32)

# load dataset and sample a text
dataset = load_dataset("Xnhyacinth/LongBench", "2wikimqa", split="test").to_pandas()
# Select the longest data based on the 'length' field
max_length_idx = dataset['length'].idxmax()
text = dataset.loc[max_length_idx, "context"]

# load model config
model_config = AutoConfig.from_pretrained(f"{BASE_MODEL_PATH}/{args.model_name}")

# load tokenizer and encode the text
tokenizer = AutoTokenizer.from_pretrained(f"{BASE_MODEL_PATH}/{args.model_name}")
# truncate the input length if specified
inputs = tokenizer(text, return_tensors="pt")
print(inputs['input_ids'].shape)
inputs['input_ids'] = inputs['input_ids'][:, :args.input_length]
# if args.input_length is not None:
#     input_ids = inputs["input_ids"]
#     current_length = input_ids.shape[1]
#     print(f"Current length: {current_length}")

#     if current_length > args.input_length:
#         inputs["input_ids"] = input_ids[:, :args.input_length]
#         inputs["attention_mask"] = inputs["attention_mask"][:, :args.input_length]
#     elif current_length < args.input_length:
#         num_repeats = (args.input_length + current_length - 1) // current_length
#         inputs["input_ids"] = input_ids.repeat(1, num_repeats)[:, :args.input_length]
#         inputs["attention_mask"] = inputs["attention_mask"].repeat(1, num_repeats)[:, :args.input_length]
inputs = inputs.to("cuda")

# load custom cache config
cache_config = CustomCacheConfig(
    transform_type=args.transform_type,
    scores=scores,
    heads_selection=args.heads_selection,
    high_key_max_value=args.high_key_max_value,
    high_value_max_value=args.high_value_max_value,
    low_key_max_value=args.low_key_max_value,
    low_value_max_value=args.low_value_max_value,
    axis_key=args.axis_key,
    axis_value=args.axis_value,
)
past_key_values = CustomCache(cache_config=cache_config)

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
# take the original tensors and meta data in each layer
# layer_compression_ratios = []
for i in range(num_layers):
    low_keys_quantized, low_keys_meta_data = past_key_values._low_quantized_key_cache.popleft()
    low_values_quantized, low_values_meta_data = past_key_values._low_quantized_value_cache.popleft()
    high_keys_quantized, high_keys_meta_data = past_key_values._high_quantized_key_cache.popleft()
    high_values_quantized, high_values_meta_data = past_key_values._high_quantized_value_cache.popleft()

    # concat the original tensors and meta data in each layer
    keys_layer_block = torch.cat([
        to_device(low_keys_quantized, device),
        to_device(high_keys_quantized, device),
    ], dim=1)
    values_layer_block = torch.cat([
        to_device(low_values_quantized, device),
        to_device(high_values_quantized, device),
    ], dim=1)

    # Calculate compression ratio for the current layer
    # layer_meta_data = [low_keys_meta_data, high_keys_meta_data, low_values_meta_data, high_values_meta_data]
    
    # Original layer data size
    # original_layer_size = len(pickle.dumps(original_key_tensors[i])) + len(pickle.dumps(original_value_tensors[i]))

    # Compressed layer data size
    # compressed_layer_tensor = nvcomp_wrapper.compress(current_layer_block)
    # packed_layer_data = PackedData(compressed_layer_tensor, layer_meta_data)
    # compressed_layer_size = len(pickle.dumps(packed_layer_data))

    # if compressed_layer_size > 0:
    #     layer_compression_ratios.append(original_layer_size / compressed_layer_size)

    # del layer_parts, low_keys_quantized, high_keys_quantized, low_values_quantized, high_values_quantized # layer_meta_data, compressed_layer_tensor, packed_layer_data
    # gc.collect()
    # torch.cuda.empty_cache()

    if to_compressed_key_tensors is None and to_compressed_value_tensors is None:
        batch_size, heads, tokens, channels = keys_layer_block.shape

        # 预分配最终形状
        final_shape = (batch_size*num_layers, heads, tokens, channels)
        
        # 使用 empty 分配（不初始化值，速度快）
        to_compressed_key_tensors = torch.empty(final_shape, dtype=keys_layer_block.dtype, device=device)
        to_compressed_value_tensors = torch.empty(final_shape, dtype=values_layer_block.dtype, device=device)
    
    # 4. 将当前层数据“填入”大 Tensor 的对应位置
    # 使用切片赋值，避免复制整个大 Tensor
    end_idx = current_idx + keys_layer_block.shape[0]
    to_compressed_key_tensors[current_idx:end_idx, ...] = keys_layer_block
    to_compressed_value_tensors[current_idx:end_idx, ...] = values_layer_block
    
    # 更新索引
    current_idx = end_idx
    
    # 5. 收集元数据并释放临时层块
    meta_data.append([low_keys_meta_data, high_keys_meta_data, low_values_meta_data, high_values_meta_data])
    # del current_layer_block
    # gc.collect()
    # torch.cuda.empty_cache()

# calculate the average compression ratio
# if layer_compression_ratios:
#     avg_compression_ratio = sum(layer_compression_ratios) / len(layer_compression_ratios)
#     print(f"\nAverage layer compression ratio: {avg_compression_ratio:.4f}")

# move the meta data to the device
meta_data = to_device(meta_data, device)

# calculate the original size
original_size = (len(pickle.dumps(original_key_tensors)) + len(pickle.dumps(original_value_tensors))) / 1024 / 1024

# load nvcomp wrapper
nvcomp_wrapper = NVCompWrapper("ANS", data_type="|u1")
# compress the original tensors and pack them
compressed_data = nvcomp_wrapper.compress(to_compressed_key_tensors, to_compressed_value_tensors)
packed_data = PackedData(compressed_data, meta_data)

# # load cachegen wrapper
# cachegen_wrapper = CacheGenWrapper(
#     quantized_keys=to_compressed_key_tensors,
#     quantized_values=to_compressed_value_tensors,
#     meta_data=meta_data,
# )
# # compress the original tensors and pack them
# packed_data = cachegen_wrapper.compress()

# calculate the compressed size
compressed_size = len(pickle.dumps(packed_data)) / 1024 / 1024
print(f"\nTotal Original size: {original_size:.2f} MB\nTotal Compressed size: {compressed_size:.2f} MB\nTotal Compression ratio: {original_size / compressed_size:.4f}")


