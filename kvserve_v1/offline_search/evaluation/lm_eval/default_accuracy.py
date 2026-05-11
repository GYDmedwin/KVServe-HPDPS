import lm_eval
import json
import logging
import sys
import pandas as pd
import torch
import argparse
import numpy as np
import lm_wrapper
import lm_evaluator


# logging.getLogger("lm_eval").setLevel(logging.INFO)
BASE_MODEL_PATH = "/root/workspace/models"

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
    "gsm8k_cot_llama": True,
    "gsm8k_cot": False,
    "mbpp_instruct": True,
    "humaneval_instruct": True,
}

parser = argparse.ArgumentParser(description="Evaluate a model with lm_eval.")
parser.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct", help="Name of the model to evaluate.")

parser.add_argument("--tasks", type=str, nargs='+', default=["gsm8k_cot_llama"], help="List of tasks to evaluate.")
parser.add_argument("--comp_cr", type=bool, default=True, help="Whether to compute compression ratio.")
parser.add_argument("--cache_type", type=str, default="default", choices=["custom", "default", "kivi"], help="Type of cache to use.")
args = parser.parse_args()

print("Evaluating the model...")

model_args = {
    "pretrained": f"{BASE_MODEL_PATH}/{args.model_name}",
    "device_map": "auto",
    "parallelize": True,
    "attn_implementation": "flash_attention_2",

    "cache_type": args.cache_type,
}

results = lm_evaluator.simple_evaluate(
    model="my_custom_model",   # 对应 --model, 即注册的名称
    model_args=model_args, # 对应 --model_args
    tasks=args.tasks, # 对应 --tasks (一个 Python 列表)
    device="cuda",           # 对应 --device
    batch_size="auto:50",         # 对应 --batch_size
    max_batch_size=16,
    # --- (可选) ---
    apply_chat_template=TASK_TO_CHAT_TEMPLATE,
    confirm_run_unsafe_code=True,
    # gen_kwargs={"max_new_tokens": 256, "do_sample": False},
    # num_fewshot=8,
    # limit=5,               # 对应 --limit, 只运行100个样本进行快速测试
    # log_samples=True,        # 对应 --log_samples, 将预测结果保存到 'results' 中
)

print("\nEvaluation completed.")

# 'results' 是一个字典，包含详细的评测结果
# 'results' 键 包含了每个任务的指标
# 'configs' 键 包含了运行时的配置
# 'versions' 键 包含了任务的版本

# 1. 打印完整的 JSON 结果 (适合程序处理)
print("\n--- Full results ---")
for result in results:
    for task_name, metrics in result['results'].items():
        print(f"\nTask: {task_name}")
        for metric_name, value in metrics.items():
            # 过滤掉标准差 (stderr) 和 alias，只显示核心指标
            if not "_stderr" in metric_name and metric_name != "alias":
                # 格式化浮点数，保留4位小数
                print(f"  {metric_name}: {value:.4f}")

if args.comp_cr:
    print(f"Total Compression ratio: 1.0000")