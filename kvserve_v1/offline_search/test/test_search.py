import itertools
import subprocess
import re
import json
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm
import warnings
import sys
import os
import torch
import gc

# ====================================================
# [新] 导入刚刚创建的评估器
# ====================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from Infer_Comm.evaluation.compression_ratio.compression_evaluator import CompressionEvaluator
except ImportError as e:
    print(f"Warning: Could not import CompressionEvaluator. Please check sys.path. Error: {e}")
    exit(1)

# ================= 配置区域 =================

# 1. 文件路径 (ACC 脚本保持原样)
ACC_SCRIPT = "/home/bingxing2/home/scx9kvs/mxy/Infer_Comm/evaluation/lm_eval/custom_accuracy.py"

# 2. 目标设置
BASELINE_ACC = 0.5397 
ACC_TOLERANCE = 0.025
TARGET_ACC_THRESHOLD = BASELINE_ACC - ACC_TOLERANCE
PRUNING_EPSILON = 0.3 

# 3. 搜索空间 (保持不变)
SEARCH_SPACE = {
    "heads_selection": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
    "high_key_max_value": [16, 12, 10, 8, 6],
    "high_value_max_value": [16, 12, 10, 8, 6],
    "low_key_max_value": [8, 6, 4, 2],
    "low_value_max_value": [8, 6, 4, 2],
    "axis_key_options": [(2,), (1, 3)],
    "axis_value_options": [(1, 3), (2,)]
}

# ================= 工具函数 =================

def is_valid_config(params):
    try:
        if not (0 <= params["heads_selection"] <= 1): return False
        if params["high_key_max_value"] < params["low_key_max_value"]: return False
        if params["high_value_max_value"] < params["low_value_max_value"]: return False
        for k in ["high_key_max_value", "high_value_max_value", "low_key_max_value", "low_value_max_value"]:
            if not (0 < params[k] < 256): return False
        if not all(0 <= x <= 3 for x in params["axis_key"]): return False
        if not all(0 <= x <= 3 for x in params["axis_value"]): return False
        return True
    except KeyError: return False

def build_cmd_args(params):
    """仅供 run_acc 使用构造命令行参数"""
    cmd = [
        "--heads_selection", str(params["heads_selection"]),
        "--high_key_max_value", str(params["high_key_max_value"]),
        "--high_value_max_value", str(params["high_value_max_value"]),
        "--low_key_max_value", str(params["low_key_max_value"]),
        "--low_value_max_value", str(params["low_value_max_value"]),
    ]
    cmd.append("--axis_key")
    cmd.extend([str(x) for x in params["axis_key"]])
    cmd.append("--axis_value")
    cmd.extend([str(x) for x in params["axis_value"]])
    return cmd

def run_cr(evaluator, params):
    """
    [优化] 使用内存中的 evaluator 直接计算 CR，无需启动新进程
    """
    try:
        # 直接调用函数
        cr = evaluator.evaluate(params)
        return cr
    except Exception as e:
        print(f"Error running CR evaluation: {e}")
        return 0.0

def run_acc(params):
    """
    Acc 仍然使用 subprocess，保持隔离性
    """
    print(f"   >>> Running Evaluation (Subprocess)...")
    cmd = ["python", ACC_SCRIPT] + build_cmd_args(params)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        match = re.search(r"score,none:\s*([\d\.]+)", result.stdout)
        if match:
            acc = float(match.group(1))
            print(f"   >>> Result Accuracy: {acc:.4f}")
            return acc
        else:
            return 0.0
    except subprocess.CalledProcessError as e:
        print(f"Error running ACC: {e}")
        return 0.0

# ================= 贝叶斯优化类 (保持不变) =================

class ConstraintAwareBO:
    def __init__(self, candidate_pool_df):
        self.candidates = candidate_pool_df
        self.observed_indices = [] 
        self.observed_accs = []
        self.skipped_indices = set()
        
        self.feature_cols = [
            "heads_selection", 
            "high_key_max_value", "high_value_max_value",
            "low_key_max_value", "low_value_max_value",
            "axis_key_id", "axis_value_id"
        ]
        
        kernel = Matern(nu=1.5, length_scale=1.0) + WhiteKernel(noise_level=1e-4)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)

    def fit(self):
        if not self.observed_indices: return
        X = self.candidates.iloc[self.observed_indices][self.feature_cols].values
        y = np.array(self.observed_accs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(X, y)

    def propose_next(self, threshold):
        X_all = self.candidates[self.feature_cols].values
        mu, sigma = self.gp.predict(X_all, return_std=True)
        with np.errstate(divide='ignore'):
            z_scores = (mu - threshold) / sigma
        prob_feasible = norm.cdf(z_scores)
        acquisition_scores = self.candidates["cr"].values * prob_feasible
        
        for idx in self.observed_indices: acquisition_scores[idx] = -1.0
        for idx in self.skipped_indices: acquisition_scores[idx] = -1.0
            
        best_idx = np.argmax(acquisition_scores)
        if acquisition_scores[best_idx] == -1.0: return None, 0.0, 0.0 
        return best_idx, prob_feasible[best_idx], mu[best_idx]

# ================= 主流程 =================

def main():
    # ====================================================
    # [关键步骤] 初始化评估器，加载模型一次
    # ====================================================
    print("=== Initializing Compression Evaluator (Loading Model...) ===")
    evaluator = CompressionEvaluator()
    print("=== Model Loaded Successfully ===")

    print("\n=== Phase 1: Generating & Profiling Search Space (Fast) ===")
    
    keys = ["heads_selection", "high_key_max_value", "high_value_max_value", "low_key_max_value", "low_value_max_value", "axis_key", "axis_value"]
    values = [SEARCH_SPACE["heads_selection"], SEARCH_SPACE["high_key_max_value"], SEARCH_SPACE["high_value_max_value"], SEARCH_SPACE["low_key_max_value"], SEARCH_SPACE["low_value_max_value"], SEARCH_SPACE["axis_key_options"], SEARCH_SPACE["axis_value_options"]]
    
    raw_combinations = list(itertools.product(*values))
    valid_data = [dict(zip(keys, c)) for c in raw_combinations if is_valid_config(dict(zip(keys, c)))]
    df = pd.DataFrame(valid_data)
    
    if len(df) == 0:
        print("Error: No valid configurations.")
        return

    key_map = {v: i for i, v in enumerate(sorted(list(set(df["axis_key"]))))}
    val_map = {v: i for i, v in enumerate(sorted(list(set(df["axis_value"]))))}
    df["axis_key_id"] = df["axis_key"].apply(lambda x: key_map[x])
    df["axis_value_id"] = df["axis_value"].apply(lambda x: val_map[x])

    print(f"Evaluating CR for {len(df)} configurations (In-Memory Speedup!)...")
    crs = []
    # 这里通过内存调用，速度会飞快
    for i, row in df.iterrows():
        if i % 5 == 0: print(f"\rProcessing {i}/{len(df)}...", end="")
        crs.append(run_cr(evaluator, row.to_dict()))
    print(f"\rProcessing {len(df)}/{len(df)}... Done.")
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()
    
    df["cr"] = crs
    df = df[df["cr"] > 0].reset_index(drop=True)
    df.to_csv("search_space_cr_final.csv", index=False)
    
    # ================= Phase 2 (逻辑不变) =================
    print("\n=== Phase 2: Bayesian Optimization Search ===")
    print(f"Target Accuracy: >= {TARGET_ACC_THRESHOLD:.4f}")
    
    bo = ConstraintAwareBO(df)
    best_feasible_cr = 0.0
    best_config = None

    # ----------------- Cold Start -----------------
    initial_indices = np.random.choice(df.index, min(3, len(df)), replace=False)
    print("--- Cold Start ---")
    for idx in initial_indices:
        params = df.iloc[idx].to_dict()
        acc = run_acc(params)
        bo.observed_indices.append(idx)
        bo.observed_accs.append(acc)
        
        print(f"Testing Initial Config (ID:{idx}): CR={params['cr']:.4f} -> Acc={acc:.4f}")
        if acc >= TARGET_ACC_THRESHOLD:
            if params['cr'] > best_feasible_cr:
                best_feasible_cr = params['cr']
                best_config = params
                # Batch Pruning logic
                cutoff_cr = best_feasible_cr - PRUNING_EPSILON
                prune_indices = df[df['cr'] < cutoff_cr].index.tolist()
                new_skips = [x for x in prune_indices if x not in bo.skipped_indices and x not in bo.observed_indices]
                bo.skipped_indices.update(new_skips)
                if new_skips:
                    print(f"   [Pruning] Skipped {len(new_skips)} configs with CR < {cutoff_cr:.4f}")
        elif acc <= TARGET_ACC_THRESHOLD - ACC_TOLERANCE:
            cutoff_cr = params['cr'] + 2 * PRUNING_EPSILON
            prune_indices = df[df['cr'] > cutoff_cr].index.tolist()
            new_skips = [x for x in prune_indices if x not in bo.skipped_indices and x not in bo.observed_indices]
            bo.skipped_indices.update(new_skips)
            if new_skips:
                print(f"   [Pruning] Skipped {len(new_skips)} configs with CR > {cutoff_cr:.4f}")        

    # ----------------- BO Loop -----------------
    MAX_ITER = 100
    for i in range(MAX_ITER):
        total_configs = len(df)
        visited_count = len(set(bo.observed_indices) | bo.skipped_indices)
        remaining_count = total_configs - visited_count
        print(f"\n--- BO Iteration {i+1}/{MAX_ITER} | Remaining: {remaining_count}/{total_configs} ---")
        
        bo.fit()
        result = bo.propose_next(TARGET_ACC_THRESHOLD)
        next_idx, prob, pred_acc = result
        if next_idx is None:
            print("⚠️  No more candidates.")
            break        
        next_params = df.iloc[next_idx].to_dict()
        
        print(f"Proposing Config (ID:{next_idx}): CR={next_params['cr']:.4f} | Pred Acc={pred_acc:.4f}")
        
        # Single Point Pruning
        if best_feasible_cr > 0 and next_params['cr'] <= best_feasible_cr:
            print(f"⏭️  SKIPPING (Pruned). Reason: CR <= Best {best_feasible_cr:.4f}")
            bo.skipped_indices.add(next_idx)
            continue 
        
        real_acc = run_acc(next_params)
        bo.observed_indices.append(next_idx)
        bo.observed_accs.append(real_acc)
        
        if real_acc >= TARGET_ACC_THRESHOLD:
            print("✅ Configuration is FEASIBLE.")
            if next_params['cr'] > best_feasible_cr:
                best_feasible_cr = next_params['cr']
                best_config = next_params
                print(f"🎉 NEW BEST! Max CR: {best_feasible_cr:.4f}")
                # Batch Pruning
                cutoff_cr = best_feasible_cr - PRUNING_EPSILON
                prune_indices = df[df['cr'] < cutoff_cr].index.tolist()
                new_skips = [x for x in prune_indices if x not in bo.skipped_indices and x not in bo.observed_indices]
                bo.skipped_indices.update(new_skips)
                if new_skips:
                    print(f"🧹 [Batch Pruning] Skipped {len(new_skips)} low-CR configs.")
            else:
                print(f"   (CR <= Best)")
        else:
            print(f"❌ INFEASIBLE (Acc {real_acc:.4f}).")
            if real_acc <= TARGET_ACC_THRESHOLD - ACC_TOLERANCE:
                cutoff_cr = next_params['cr'] + 2 * PRUNING_EPSILON
                prune_indices = df[df['cr'] > cutoff_cr].index.tolist()
                new_skips = [x for x in prune_indices if x not in bo.skipped_indices and x not in bo.observed_indices]
                bo.skipped_indices.update(new_skips)
                if new_skips:
                    print(f"🧹 [Batch Pruning] Skipped {len(new_skips)} high-CR configs.")
                continue


    print("\n================ Search Finished ================")
    if best_config:
        final = {k: v for k, v in best_config.items() if k not in ["axis_key_id", "axis_value_id", "cr"]}
        print(json.dumps(final, indent=2, default=str))
        with open("best_config_fast.json", "w") as f:
            json.dump(final, f, indent=4, default=str)

if __name__ == "__main__":
    main()