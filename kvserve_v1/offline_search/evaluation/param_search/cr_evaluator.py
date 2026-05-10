import os
import sys
import torch
import pandas as pd
import numpy as np
import pickle
import gc
import argparse
import random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from collections import deque


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from Infer_Comm.src.cache.cache_utils import CustomCacheConfig, CustomCache
from Infer_Comm.evaluation.compression_ratio.nvcomp_wrapper import NVCompWrapper, CompressedTensor, PackedData, TensorData, to_device

# 硬编码路径 (参考自 custom_cr.py)
BASE_MODEL_PATH = "/root/workspace/models"
BASE_CONFIG_PATH = "/root/workspace/Infer_Comm/duo_config"

class CompressionEvaluator:
    def __init__(self, model_name="Qwen2.5-7B-Instruct", task="2wikimqa", device="cuda"):
        
        print(f"Loading model {model_name} and preparing data... (This happens only once)")
        self.device = device
        self.model_name = model_name
        self.task = task
        
        # 1. 加载 Scores
        scores_path = f"{BASE_CONFIG_PATH}/{model_name}_scores.csv"
        df = pd.read_csv(scores_path, header=None).dropna()
        self.scores = torch.tensor(df.values, dtype=torch.float32)
        
        # 2. 加载并处理输入数据
        # 遍历所有 tasks，每个 task 采样 limit 条数据
        self.inputs = None
        tokenizer = AutoTokenizer.from_pretrained(f"{BASE_MODEL_PATH}/{model_name}")

        # load dataset and sample a text
        dataset = load_dataset("Xnhyacinth/LongBench", self.task, split="test").to_pandas()
        # Select the longest data based on the 'length' field
        max_length_idx = dataset['length'].idxmax()
        text = dataset.loc[max_length_idx, "context"]        
        self.inputs = tokenizer(text, return_tensors="pt").to(device)
                
        # 3. 加载模型
        self.model_config = AutoConfig.from_pretrained(f"{BASE_MODEL_PATH}/{model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            f"{BASE_MODEL_PATH}/{model_name}",
            torch_dtype="auto",
            device_map="auto",
            use_cache=True,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        
        print("Initialization Complete. Ready for Evaluation.")

    def evaluate(self, params):
        """
        params: dict 包含 heads_selection, high_key_max_value 等参数
        """
        
        # 动态计算当前 input 的原始大小
        current_length = self.inputs["input_ids"].shape[1]
        print(f"Current length: {current_length}")
        num_layers = self.model_config.num_hidden_layers
        num_heads = self.model_config.num_key_value_heads
        head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads
        original_kv_shape = (1, num_heads, current_length, head_dim)
        
        # 创建占位 tensor 计算 pickle 大小
        original_key_tensors = [torch.empty(original_kv_shape, dtype=self.model_config.torch_dtype, device=self.device) for _ in range(num_layers)]
        original_value_tensors = [torch.empty(original_kv_shape, dtype=self.model_config.torch_dtype, device=self.device) for _ in range(num_layers)]
        
        original_size_mb = (len(pickle.dumps(original_key_tensors)) + len(pickle.dumps(original_value_tensors))) / 1024 / 1024
        del original_key_tensors, original_value_tensors

        # 1. 设置 Cache Config
        cache_config = CustomCacheConfig(
            transform_type=params["transform_type"],
            scores=self.scores,
            heads_selection=params["heads_selection"],
            high_key_max_value=params["high_key_max_value"],
            high_value_max_value=params["high_value_max_value"],
            low_key_max_value=params["low_key_max_value"],
            low_value_max_value=params["low_value_max_value"],
            axis_key=params["axis_key"],
            axis_value=params["axis_value"],
        )
        past_key_values = CustomCache(cache_config=cache_config)
        
        # 2. 执行推理 (Pre-fill only)
        with torch.no_grad():
            outputs = self.model(
                **self.inputs,
                # max_new_tokens=1,
                return_dict_in_generate=True,
                past_key_values=past_key_values,
            )
        
        # 3. 计算压缩后大小
        # 初始化 nvcomp wrapper
        nvcomp_wrapper = NVCompWrapper("ANS", data_type="|u1")
        
        meta_data = []
        num_layers = self.model_config.num_hidden_layers
        current_idx = 0
        to_compressed_key_tensors = None
        to_compressed_value_tensors = None
        # 遍历层并收集压缩数据
        for i in range(num_layers):
            low_keys_quantized, low_keys_meta_data = past_key_values._low_quantized_key_cache.popleft()
            low_values_quantized, low_values_meta_data = past_key_values._low_quantized_value_cache.popleft()
            high_keys_quantized, high_keys_meta_data = past_key_values._high_quantized_key_cache.popleft()
            high_values_quantized, high_values_meta_data = past_key_values._high_quantized_value_cache.popleft()

            keys_layer_block = torch.cat([
                to_device(low_keys_quantized, self.device),
                to_device(high_keys_quantized, self.device),
            ], dim=1)
            values_layer_block = torch.cat([
                to_device(low_values_quantized, self.device),
                to_device(high_values_quantized, self.device),
            ], dim=1)
            
            if to_compressed_key_tensors is None and to_compressed_value_tensors is None:
                batch_size, heads, tokens, channels = keys_layer_block.shape

                # 预分配最终形状
                final_shape = (batch_size*num_layers, heads, tokens, channels)
                
                # 使用 empty 分配（不初始化值，速度快）
                to_compressed_key_tensors = torch.empty(final_shape, dtype=keys_layer_block.dtype, device=self.device)
                to_compressed_value_tensors = torch.empty(final_shape, dtype=values_layer_block.dtype, device=self.device)
            
            end_idx = current_idx + keys_layer_block.shape[0]
            to_compressed_key_tensors[current_idx:end_idx, ...] = keys_layer_block
            to_compressed_value_tensors[current_idx:end_idx, ...] = values_layer_block
            
            # 更新索引
            current_idx = end_idx
            
            meta_data.append([low_keys_meta_data, high_keys_meta_data, low_values_meta_data, high_values_meta_data])
            
            # 清理显存
            del keys_layer_block, values_layer_block, low_keys_quantized, high_keys_quantized, low_values_quantized, high_values_quantized
        
        # 压缩与打包
        meta_data = to_device(meta_data, self.device)
        compressed_data = nvcomp_wrapper.compress(to_compressed_key_tensors, to_compressed_value_tensors)
        packed_data = PackedData(compressed_data, meta_data)
        
        compressed_size_mb = len(pickle.dumps(packed_data)) / 1024 / 1024
        compression_ratio = original_size_mb / compressed_size_mb
        print(f"original size: {original_size_mb}MB, compressed size: {compressed_size_mb}MB")
                
        # 清理本次推理产生的资源
        del outputs, past_key_values, to_compressed_key_tensors, to_compressed_value_tensors, packed_data, compressed_data, meta_data, nvcomp_wrapper
        gc.collect()
        torch.cuda.empty_cache()
            
        return round(compression_ratio, 4)

# params = {
#     "transform_type": "hadamard",
#     "heads_selection": 0.9,
#     "high_key_max_value": 16,
#     "high_value_max_value": 12,
#     "low_key_max_value": 8,
#     "low_value_max_value": 4,
#     "axis_key": [2],
#     "axis_value": [1, 3],
# }
# cr_evaluator = CompressionEvaluator(model_name="Qwen2.5-7B-Instruct", task="hotpotqa")
# cr_val = cr_evaluator.evaluate(params)
# import json
# with open('config3.json', 'w', encoding='utf-8') as f:
#     json.dump({
#         "heads_selection": params["heads_selection"], 
#         "high_key_max_value": params["high_key_max_value"], 
#         "high_value_max_value": params["high_value_max_value"], 
#         "low_key_max_value": params["low_key_max_value"], 
#         "low_value_max_value": params["low_value_max_value"], 
#         "tasks": cr_evaluator.tasks,
#         "compression_ratio_list": cr_val}, 
#         f, ensure_ascii=False, indent=4)
# print(f"Compression ratio: {cr_val}")