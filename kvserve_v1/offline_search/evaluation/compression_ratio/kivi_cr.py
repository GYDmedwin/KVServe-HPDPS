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
from offline_search.src.cache.kivi_utils import KIVICacheConfig, KIVICache
from nvcomp_wrapper import CompressedTensor, PackedData, TensorData, to_device


BASE_MODEL_PATH = "/root/workspace/models"
# BASE_CONFIG_PATH = "/home/bingxing2/home/scx9kvs/mxy/Infer_Comm/duo_config"


parser = argparse.ArgumentParser(description="Evaluate kv cache compression ratio.")
parser.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct", help="Name of the model to evaluate.")
parser.add_argument("--nbits", type=int, default=2, help="Number of quantization bits.")
parser.add_argument("--axis_key", type=int, default=2, help="Axis for key.")
parser.add_argument("--axis_value", type=int, default=3, help="Axis for value.")
parser.add_argument("--q_group_size", type=int, default=32, help="Size of the quantization group.")
parser.add_argument("--residual_length", type=int, default=128, help="Length of the residual cache.")
parser.add_argument("--input_length", type=int, default=10240, help="Desired input length for the model.")

args = parser.parse_args()

def pack_to_float32(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """
    将 uint8 tensor 压缩打包存储到 float32 tensor 中。
    
    Args:
        tensor: 输入的 uint8 tensor (包含 4-bit 或 2-bit 数据)
        bits: 每个元素占用的位数 (支持 4 或 2)
    
    Returns:
        packed_tensor: dtype 为 float32 的压缩 tensor
    """
    assert bits in [2, 4], "目前只支持 2-bit 或 4-bit 压缩"
    
    # 1. 计算打包比例
    # 32位容器 / 4位 = 8 个数
    # 32位容器 / 2位 = 16 个数
    pack_ratio = 32 // bits 
    
    # 2. Flatten 拉平
    flat_tensor = tensor.flatten().to(torch.int32) # 转换为 int32 以便进行位移操作
    
    # 3. Padding (如果长度不能被整除)
    num_elements = flat_tensor.numel()
    padding = (pack_ratio - (num_elements % pack_ratio)) % pack_ratio
    if padding > 0:
        flat_tensor = torch.nn.functional.pad(flat_tensor, (0, padding), value=0)
    
    # 4. Reshape 为 (新长度, pack_ratio)
    # 例如 4-bit: [N, 8]
    reshaped = flat_tensor.view(-1, pack_ratio)
    
    # 5. 构造位移量 (Shift Vector)
    # 4-bit: [0, 4, 8, 12, 16, 20, 24, 28]
    # 注意：通常低位放第一个数还是高位放第一个数取决于你的解码习惯。
    # 这里采用 Little Endian 风格：第一个数在最低位。
    shift_vals = torch.arange(0, 32, bits, device=tensor.device, dtype=torch.int32)
    
    # 6. 执行位打包
    # 利用广播机制：reshaped << shift_vals
    shifted = reshaped << shift_vals
    
    # 按行进行 Bitwise OR (或者 Sum，因为位不重叠，Sum 等效于 OR)
    packed_int32 = torch.sum(shifted, dim=1).to(torch.int32)
    
    # 7. Reinterpret Cast (关键步骤)
    # 将 int32 的二进制位直接看作 float32，不改变位本身
    packed_float32 = packed_int32.view(torch.float32)
    
    return packed_float32

print("Evaluating the model...")

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
cache_config = KIVICacheConfig(
    nbits=args.nbits,
    axis_key=args.axis_key,
    axis_value=args.axis_value,
    q_group_size=args.q_group_size,
    residual_length=args.residual_length,
)
past_key_values = KIVICache(cache_config=cache_config)

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
compressed_key_tensors = None
compressed_value_tensors = None

for i in range(num_layers):
    keys_quantized, keys_meta_data = past_key_values._quantized_key_cache[i]
    values_quantized, values_meta_data = past_key_values._quantized_value_cache[i]

    # concat the original tensors and meta data in each layer
    keys_quantized = pack_to_float32(to_device(keys_quantized, device), args.nbits).unsqueeze(0)
    values_quantized = pack_to_float32(to_device(values_quantized, device), args.nbits).unsqueeze(0)

    if compressed_key_tensors is None and compressed_value_tensors is None:
        _, channels = keys_quantized.shape
        
        # 预分配最终形状
        final_shape = (num_layers, channels)
        
        # 使用 empty 分配（不初始化值，速度快）
        compressed_key_tensors = torch.empty(final_shape, dtype=keys_quantized.dtype, device=device)
        compressed_value_tensors = torch.empty(final_shape, dtype=values_quantized.dtype, device=device)

    # 4. 将当前层数据“填入”大 Tensor 的对应位置
    # 使用切片赋值，避免复制整个大 Tensor
    compressed_key_tensors[i, ...] = keys_quantized
    compressed_value_tensors[i, ...] = values_quantized
    
    # 5. 收集元数据并释放临时层块
    meta_data.append([keys_meta_data, values_meta_data])
    # del keys_quantized, values_quantized, keys_meta_data, values_meta_data
    # gc.collect()
    # torch.cuda.empty_cache()

# move the meta data to the device
meta_data = to_device(meta_data, device)

original_size = (len(pickle.dumps(original_key_tensors)) + len(pickle.dumps(original_value_tensors))) / 1024 / 1024
# pack the original tensors and meta data
compressed_data = TensorData(torch.cat([compressed_key_tensors, compressed_value_tensors], dim=0), meta_data)
# calculate the compressed size
compressed_size = (len(pickle.dumps(compressed_data)) + len(pickle.dumps(past_key_values.key_cache)) + len(pickle.dumps(past_key_values.value_cache))) / 1024 / 1024
print(f"\nTotal Original size: {original_size:.2f} MB\nTotal Compressed size: {compressed_size:.2f} MB\nTotal Compression ratio: {original_size / compressed_size:.4f}")

