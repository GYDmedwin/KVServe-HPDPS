import os
import sys
import torch
import pandas as pd
import numpy as np
import pickle
import gc
import argparse
import random
import lm_eval
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from collections import deque
from contextlib import contextmanager
from lm_eval.tasks import TaskManager, get_task_dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from Infer_Comm.evaluation.lm_eval import lm_wrapper, lm_evaluator

# 硬编码路径 (参考自 custom_cr.py)
BASE_MODEL_PATH = "/root/workspace/models"
BASE_CONFIG_PATH = "/root/workspace/Infer_Comm/duo_config"
TASK_TO_CHAT_TEMPLATE = {
    "longbench_lcc": False,
    "longbench_lcc_e": False,
    "longbench_repobench-p": False,
    "longbench_repobench-p_e": False,
    "longbench_multi_news": True,
    "longbench_multi_news_e": True,
    "longbench_gov_report": True,
    "longbench_gov_report_e": True,
    "longbench_2wikimqa": True,
    "longbench_2wikimqa_e": True,
    "longbench_hotpotqa": True,
    "longbench_hotpotqa_e": True,
    "longbench_qasper": True,
    "longbench_qasper_e": True,
    "longbench_multifieldqa_en": True,
    "longbench_multifieldqa_en_e": True,
    "longbench_trec": False,
    "gsm8k_cot_llama": True,
    "gsm8k_cot": False,
    "mbpp_instruct": True,
    "humaneval_instruct": True,
}
TASK_TO_LIMIT = {
    # "longbench_multi_news": 40,
    # "longbench_qasper": 200,
    # "gsm8k_cot_llama": 500,
    # "humaneval_instruct": 200,
    "longbench_multi_news": 500,
    "longbench_qasper": 500,
    "longbench_2wikimqa": 500,
    "longbench_hotpotqa": 500,
    "longbench_multifieldqa_en": 500,
    "gsm8k_cot_llama": 2000,
    "humaneval_instruct": 200,
}

TASK_TO_BATCH_SIZE = {
    "longbench_multi_news": 8,
    "longbench_qasper": 8,
    "longbench_2wikimqa": 8,
    "longbench_hotpotqa": 8,
    "longbench_multifieldqa_en": 8,
    "gsm8k_cot_llama": 50,
    "humaneval_instruct": 32,
}
@contextmanager
def suppress_fd_stderr():
    """
    在 OS 层面屏蔽 stderr (文件描述符 2)。
    这可以屏蔽 subprocess (如 git) 和 C 扩展库的直接输出。
    """
    # 打开空设备
    devnull = os.open(os.devnull, os.O_WRONLY)
    # 备份原本的 stderr 文件描述符 (通常是 2)
    original_stderr_fd = os.dup(2)
    
    try:
        # 将 stderr (2) 重定向到 devnull
        sys.stderr.flush()
        os.dup2(devnull, 2)
        yield
    finally:
        # 恢复原本的 stderr
        os.dup2(original_stderr_fd, 2)
        # 关闭临时打开的文件描述符
        os.close(devnull)
        os.close(original_stderr_fd)

class AccuracyEvaluator:
    def __init__(self, model_name="Meta-Llama-3.1-8B-Instruct", tasks=["longbench_hotpotqa"], limit=100, batch_size=1, device="cuda", random_seed=42):
        print(f"Loading model {model_name} and preparing data... (This happens only once)")
        self.device = device
        self.model_name = model_name
        self.tasks = tasks
        self.limit = limit
        self.batch_size = batch_size
        self.random_seed = random_seed

        # 0. 预处理采样：均匀随机选择数据
        self.samples = {}
        try:
            task_manager = TaskManager()
            all_tasks = get_task_dict(self.tasks, task_manager)
            
            # 设置随机种子
            random.seed(self.random_seed)

            for task_name, task_obj in all_tasks.items():
                if hasattr(task_obj, 'test_docs') and task_obj.has_test_docs():
                    docs = task_obj.test_docs()
                elif hasattr(task_obj, 'validation_docs') and task_obj.has_validation_docs():
                    docs = task_obj.validation_docs()
                else:
                    docs = task_obj.training_docs()
                
                # 获取数据集大小
                try:
                    n_samples = len(docs)
                except:
                    # 如果不支持 len()，尝试转换成 list
                    all_docs = list(docs)
                    n_samples = len(all_docs)
                
                if n_samples <= TASK_TO_LIMIT[task_name]:
                    self.samples[task_name] = list(range(n_samples))
                else:
                    # 均匀分段采样
                    step = n_samples / TASK_TO_LIMIT[task_name]
                    selected = []
                    for i in range(TASK_TO_LIMIT[task_name]):
                        start = int(i * step)
                        end = int((i + 1) * step)
                        if end <= start:
                            end = start + 1
                        if end > n_samples:
                            end = n_samples
                        
                        # 在区间 [start, end) 内随机选一个
                        if start < end:
                            selected.append(random.randrange(start, end))
                        else:
                            selected.append(start)
                    self.samples[task_name] = selected
            print(f"Selected samples indices for tasks: { {k: len(v) for k,v in self.samples.items()} }")
        except Exception as e:
            print(f"Warning: Failed to prepare samples in __init__: {e}")
            # Fallback logic or just re-raise if critical
            raise e

        # 1. 加载 Scores
        scores_path = f"{BASE_CONFIG_PATH}/{model_name}_scores.csv"
        df = pd.read_csv(scores_path, header=None).dropna()
        self.scores = torch.tensor(df.values, dtype=torch.float32)
        # 2. 加载默认分数
        model_args = {
            "pretrained": f"{BASE_MODEL_PATH}/{self.model_name}",
            "device_map": "auto",
            "parallelize": True,
            "attn_implementation": "flash_attention_2",
            "cache_type": "default",
        }
        default_values = []
        with suppress_fd_stderr():
            for task_name in self.tasks:
                eval_task = [task_name]
                default_results = lm_evaluator.simple_evaluate(
                    model="my_custom_model",
                    model_args=model_args,
                    tasks=eval_task,
                    device="cuda",
                    batch_size=TASK_TO_BATCH_SIZE[task_name],
                    apply_chat_template=TASK_TO_CHAT_TEMPLATE,
                    confirm_run_unsafe_code=True,
                    samples={task_name: self.samples[task_name]},
                )
                for results in default_results:
                    for task_name, metrics in results['results'].items():
                        # print(f"Task: {task_name}")
                        for metric_name, value in metrics.items():
                            if not "_stderr" in metric_name and metric_name != "alias":
                                # 格式化浮点数，保留4位小数
                                # print(f"  {metric_name}: {value:.4f}")
                                default_values.append(round(value, 4))
                                # break
        self.default_scores = np.array(default_values)
        del default_results
        gc.collect()
        torch.cuda.empty_cache()        

    def evaluate(self, params):
        """
        params: dict 包含 heads_selection, high_key_max_value 等参数
        """

        # 1. 设置 Cache Config
        model_args = {
            "pretrained": f"{BASE_MODEL_PATH}/{self.model_name}",
            "device_map": "auto",
            "parallelize": True,
            "attn_implementation": "flash_attention_2",

            "transform_type": params["transform_type"],
            "scores": self.scores,
            "heads_selection": params["heads_selection"],
            "high_key_max_value": params["high_key_max_value"],
            "high_value_max_value": params["high_value_max_value"],
            "low_key_max_value": params["low_key_max_value"],
            "low_value_max_value": params["low_value_max_value"],
            "axis_key": list(params["axis_key"]),
            "axis_value": list(params["axis_value"]),

            "cache_type": "custom",
        }
        custom_values = []
        with suppress_fd_stderr():
            for task_name in self.tasks:
                eval_task = [task_name]
                custom_results = lm_evaluator.simple_evaluate(
                    model="my_custom_model",
                    model_args=model_args,
                    tasks=eval_task,
                    device="cuda",
                    batch_size=TASK_TO_BATCH_SIZE[task_name],
                    
                    # --- (可选) ---
                    apply_chat_template=TASK_TO_CHAT_TEMPLATE,
                    confirm_run_unsafe_code=True,
                    samples={task_name: self.samples[task_name]},
                    # num_fewshot=5,
                )

                for results in custom_results:
                    for task_name, metrics in results['results'].items():
                        # print(f"Task: {task_name}")
                        for metric_name, value in metrics.items():
                            if not "_stderr" in metric_name and metric_name != "alias":
                                # 格式化浮点数，保留4位小数
                                # print(f"  {metric_name}: {value:.4f}")
                                custom_values.append(round(value, 4))
                                # break
        custom_scores = np.array(custom_values)
        avg_score = (custom_scores / self.default_scores).mean() * 100
        print(f"Custom scores: {custom_scores}")
        print(f"Default scores: {self.default_scores}")
        print(f"Avg score: {avg_score}")
        del custom_results
        gc.collect()
        torch.cuda.empty_cache()
        return round(avg_score, 2)

params = {
    "transform_type": "hadamard",
    "heads_selection": 0.1,
    "high_key_max_value": 8,
    "high_value_max_value": 6,
    "low_key_max_value": 8,
    "low_value_max_value": 4,
    "axis_key": [2],
    "axis_value": [1, 3],
}
eva = AccuracyEvaluator(tasks=["longbench_2wikimqa", "longbench_hotpotqa", "longbench_multifieldqa_en"], model_name="Qwen2.5-7B-Instruct", limit=500, batch_size=50)
eva.evaluate(params)

# gsm8k: batch:40, limit:400
# humaneval: batch:32, limit:200
# longbench_multi_news: batch:8, limit:40
# longbench_qasper: batch:8, limit:200

# Custom scores: [0.8203 0.8309 0.8293 0.2336 0.2336 0.4347 0.4347]
# Default scores: [0.8324 0.8423 0.8232 0.2394 0.2394 0.4333 0.4333]
# Avg score: 99.10495745889702

# Custom scores: [0.4571 0.4571 0.5498 0.5498 0.5075 0.5075]
# Default scores: [0.4649 0.4649 0.5767 0.5767 0.5195 0.5195]
# Avg score: 97.11594539737969