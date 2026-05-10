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
    "gsm8k_cot_llama": True,
    "gsm8k_cot": False,
    "mbpp_instruct": True,
    "humaneval_instruct": True,
}

parser = argparse.ArgumentParser(description="Evaluate a model with lm_eval.")
parser.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct", help="Name of the model to evaluate.")
parser.add_argument("--transform_type", type=str, default="none", choices=["none", "hadamard", "affine"], help="Type of transform to use.")
parser.add_argument("--heads_selection", type=float, default=0.1, help="Heads selection ratio.")
parser.add_argument("--high_key_max_value", type=int, default=8, help="Max value for high key.")
parser.add_argument("--high_value_max_value", type=int, default=16, help="Max value for high value.")
parser.add_argument("--low_key_max_value", type=int, default=8, help="Max value for low key.")
parser.add_argument("--low_value_max_value", type=int, default=4, help="Max value for low value.")
parser.add_argument("--axis_key", type=int, nargs='+', default=[2], help="Axis for key.")
parser.add_argument("--axis_value", type=int, nargs='+', default=[1, 3], help="Axis for value.")
parser.add_argument("--tasks", type=str, nargs='+', default=["humaneval_instruct"], help="List of tasks to evaluate.")
parser.add_argument("--comp_cr", type=bool, default=True, help="Whether to compute compression ratio.")
parser.add_argument("--cache_type", type=str, default="custom", choices=["custom", "default"], help="Type of cache to use.")
args = parser.parse_args()


print("Evaluating the model...")

df = pd.read_csv(f"{BASE_CONFIG_PATH}/{args.model_name}_scores.csv", header=None).dropna()
scores = torch.tensor(df.values, dtype=torch.float32)

compression_ratio_list = []
model_args = {
    "pretrained": f"{BASE_MODEL_PATH}/{args.model_name}",
    "device_map": "auto",
    "parallelize": True,
    "attn_implementation": "flash_attention_2",

    "transform_type": args.transform_type,
    "scores": scores,
    "heads_selection": args.heads_selection,
    "high_key_max_value": args.high_key_max_value,
    "high_value_max_value": args.high_value_max_value,
    "low_key_max_value": args.low_key_max_value,
    "low_value_max_value": args.low_value_max_value,
    "axis_key": args.axis_key,
    "axis_value": args.axis_value,

    "cache_type": args.cache_type,
    "comp_cr": args.comp_cr,
    "cr_list": compression_ratio_list,
}

results = lm_evaluator.simple_evaluate(
    model="my_custom_model",   # 对应 --model, 即注册的名称
    model_args=model_args, # 对应 --model_args
    tasks=args.tasks, # 对应 --tasks (一个 Python 列表)
    device="cuda",           # 对应 --device
    batch_size="1",         # 对应 --batch_size
    
    # --- (可选) ---
    apply_chat_template=TASK_TO_CHAT_TEMPLATE,
    confirm_run_unsafe_code=True,
    # num_fewshot=5,
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
    print(f"Total Compression ratio: {np.mean(compression_ratio_list):.4f}")