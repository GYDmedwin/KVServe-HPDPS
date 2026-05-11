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
from offline_search.src.cache.duoattn_utils import DuoAttentionCacheConfig, DuoAttentionCache
from nvcomp_wrapper import CompressedTensor, PackedData, TensorData, to_device


BASE_MODEL_PATH = "/home/bingxing2/home/scx9kvs/mxy/models"
BASE_CONFIG_PATH = "/home/bingxing2/home/scx9kvs/mxy/Infer_Comm/duo_config"


parser = argparse.ArgumentParser(description="Evaluate kv cache compression ratio.")
parser.add_argument("--model_name", type=str, default="Meta-Llama-3.1-8B-Instruct", help="Name of the model to evaluate.")
parser.add_argument("--heads_selection", type=float, default=0.8, help="Heads selection ratio.")
parser.add_argument("--sink_size", type=int, default=128, help="Sink size.")
parser.add_argument("--recent_size", type=int, default=256, help="Recent size.")
parser.add_argument("--input_length", type=int, default=1024, help="Desired input length for the model.")

args = parser.parse_args()

print("Evaluating the model...")

# load scores
df = pd.read_csv(f"{BASE_CONFIG_PATH}/{args.model_name}_scores.csv", header=None).dropna()
scores = torch.tensor(df.values, dtype=torch.float32)

# load dataset and sample a text
dataset = load_dataset("Xnhyacinth/LongBench", "multi_news", split="test").to_pandas()
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

# load custom cache config
cache_config = DuoAttentionCacheConfig(
    scores=scores,
    heads_selection=args.heads_selection,
    sink_size=args.sink_size,
    recent_size=args.recent_size,
)
past_key_values = DuoAttentionCache(cache_config=cache_config)

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
original_kv_shape = (1, num_heads, args.input_length, head_dim)
original_key_tensors = [torch.empty(original_kv_shape, dtype=model_config.torch_dtype, device=device) for _ in range(num_layers)]
original_value_tensors = [torch.empty(original_kv_shape, dtype=model_config.torch_dtype, device=device) for _ in range(num_layers)]
compressed_key_tensors = []
compressed_value_tensors = []

for i in range(num_layers):
    key_cache_layer = past_key_values.key_cache[i]
    value_cache_layer = past_key_values.value_cache[i]
    masked_indices = past_key_values._build_masking(key_cache_layer, i)

    if masked_indices is None:
        # No tokens were dropped for this layer, so we keep the entire cache.
        compressed_key_tensors.append(key_cache_layer)
        compressed_value_tensors.append(value_cache_layer)
    else:
        # Some tokens were dropped. We create a mask to select the elements to keep.
        # Create a boolean mask of shape (batch, heads, seq_len) initialized to True (keep all).
        keep_mask = torch.ones(key_cache_layer.shape[:-1], dtype=torch.bool, device=key_cache_layer.device)
        
        # Set the positions of the masked indices to False (drop them).
        keep_mask[masked_indices] = False
        
        # Expand the mask to include the head_dim dimension.
        expanded_keep_mask = keep_mask.unsqueeze(-1).expand_as(key_cache_layer)
        
        # Apply the mask to get only the elements that are kept.
        # This will result in a 1D tensor containing the remaining data, which is sufficient for size calculation.
        kept_keys = key_cache_layer[expanded_keep_mask]
        kept_values = value_cache_layer[expanded_keep_mask]
        
        compressed_key_tensors.append(kept_keys)
        compressed_value_tensors.append(kept_values)

        # del key_cache_layer, value_cache_layer, keep_mask, expanded_keep_mask, kept_keys, kept_values
        # gc.collect()
        # torch.cuda.empty_cache()

# Calculate original and compressed sizes in MB using pickle, which accounts for tensor metadata.
original_size = (len(pickle.dumps(original_key_tensors)) + len(pickle.dumps(original_value_tensors))) / 1024 / 1024
compressed_size = (len(pickle.dumps(compressed_key_tensors)) + len(pickle.dumps(compressed_value_tensors))) / 1024 / 1024

print(f"\nTotal Original size: {original_size:.2f} MB\nTotal Compressed size: {compressed_size:.2f} MB\nTotal Compression ratio: {original_size / compressed_size:.4f}")

