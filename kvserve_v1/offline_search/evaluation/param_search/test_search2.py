import itertools
import ast
import logging
import json
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
import warnings
import sys
import os
import io
import torch
import gc
from contextlib import redirect_stdout, redirect_stderr
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from Infer_Comm.evaluation.param_search.cr_evaluator import CompressionEvaluator
from Infer_Comm.evaluation.param_search.acc_evaluator import AccuracyEvaluator

# ================= 配置区域 =================

# 2. 目标设置
BASELINE_ACC = 100
ACC_TOLERANCE = 8
TARGET_ACC_THRESHOLD = BASELINE_ACC - ACC_TOLERANCE
PRUNING_EPSILON = 0.2
MAX_ITER = 100
EXPLORATION_WEIGHT = 0.5
WHETHER_TO_EXPLORE = True
# 3. 搜索空间
SEARCH_SPACE = {
    "transform_type": ["hadamard"],
    "heads_selection": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
    "high_key_max_value": [16, 12, 10, 8, 6, 5],
    "high_value_max_value": [16, 12, 10, 8, 6, 5],
    "low_key_max_value": [8, 6, 4, 3],
    "low_value_max_value": [8, 6, 4, 3],
    "axis_key_options": [(2,)],
    "axis_value_options": [(1, 3)]
}
MODEL_NAME = "Qwen2.5-7B-Instruct"
TASK_TO_SEARCH = ["longbench_2wikimqa"]
CACHE_CSV_PATH = "hadamard_search_space.csv"
FINAL_JSON_PATH = f"new_search_hadamard_tolerance_{ACC_TOLERANCE}_pareto.json"
# ALL_RESULTS_PATH = f"hadamard_tolerance_{ACC_TOLERANCE}_all_history.json"

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

def silent_call(func, *args, **kwargs):
    logging.disable(logging.CRITICAL)
    temp_stdout = io.StringIO()
    temp_stderr = io.StringIO()
    try:
        with redirect_stdout(temp_stdout), redirect_stderr(temp_stderr):
            return func(*args, **kwargs)
    finally:
        logging.disable(logging.NOTSET)

def run_cr(evaluator, params):
    logging.info(f"   >>> Running Compression Evaluation ")
    try:
        cr = evaluator.evaluate(params)
        return cr
    except Exception as e:
        logging.error(f"Error running CR evaluation: {e}")
        return 0.0

def run_acc(evaluator, params):
    logging.info(f"   >>> Running Accuracy Evaluation ")
    try:
        acc = evaluator.evaluate(params)
        logging.info(f"   >>> Result Accuracy: {acc:.4f}")
        return acc
    except Exception as e:
        logging.error(f"Error running ACC evaluation: {e}")
        return 0.0

def bigger_pruning_select_by(df, current_params, current_idx, isolation_columns):
    """剪掉比当前点更激进(CR更高)的点 - 用于当前点不可行时"""
    isolation_mask = pd.Series(True, index=df.index)
    for col in isolation_columns:
        if col in df.columns:
            isolation_mask &= (df[col] == current_params[col])

    max_value = max(current_params['high_key_max_value'], current_params['high_value_max_value'], current_params['low_key_max_value'], current_params['low_value_max_value'])
    min_value = min(current_params['high_key_max_value'], current_params['high_value_max_value'], current_params['low_key_max_value'], current_params['low_value_max_value'])
    
    # 只有当参数跨度不是太极端时才剪枝，避免误伤
    if max_value - min_value < 8:
        # CR 比当前大 (更激进)
        weaker_mask = pd.Series(True, index=df.index)
        weaker_mask &= (df['cr'] >= current_params['cr'] + 2 * PRUNING_EPSILON)
    else:
        return []
    
    not_self = (df.index != current_idx)    
    final_mask = isolation_mask & weaker_mask & not_self
    return df[final_mask].index.tolist()

def smaller_pruning_select_by(df, current_params, current_idx, isolation_columns):
    """
    [新增] 剪掉比当前点更保守（CR更低）的点 - 用于当前点可行时
    逻辑：当前点已经可行了，比它压缩率还低的点肯定也可行，但由于压缩率低，不具备Pareto优势，故剪枝。
    """
    isolation_mask = pd.Series(True, index=df.index)
    for col in isolation_columns:
        if col in df.columns:
            isolation_mask &= (df[col] == current_params[col])

    # CR 比当前小 (更保守)
    # 使用 epsilon 确保不会剪掉 CR 非常接近的点
    conservative_mask = pd.Series(True, index=df.index)
    conservative_mask &= (df['cr'] <= current_params['cr'] - PRUNING_EPSILON)
    
    not_self = (df.index != current_idx)    
    final_mask = isolation_mask & conservative_mask & not_self
    return df[final_mask].index.tolist()

def process_one_hot_encoding(df, encode_cols, current_feature_cols):
    final_feature_cols = list(current_feature_cols)
    for col in encode_cols:
        if col not in df.columns: continue
        try:
            unique_vals = sorted(list(set(df[col])))
        except TypeError:
            temp_series = df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)
            unique_vals = sorted(list(set(temp_series)))

        val_map = {v: i for i, v in enumerate(unique_vals)}
        id_col_name = f"{col}_id"
        def get_id(x):
            key = tuple(x) if isinstance(x, list) else x
            return val_map[key]
        df[id_col_name] = df[col].apply(get_id)
        dummies = pd.get_dummies(df[id_col_name], prefix=col)
        df = pd.concat([df, dummies], axis=1)
        new_cols = list(dummies.columns)
        final_feature_cols.extend(new_cols)
    return df, final_feature_cols

# ================= 贝叶斯优化类 =================

class ConstraintAwareBO:
    def __init__(self, candidate_pool_df):
        self.candidates = candidate_pool_df
        self.observed_indices = [] 
        self.observed_accs = []
        self.skipped_indices = set()
        
        self.feature_cols = None
        
        kernel = Matern(nu=1.5, length_scale=1.0) + WhiteKernel(noise_level=1e-4)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)

        self.scaler = MinMaxScaler()
        self.scaler_fitted = False
        self.alpha = [random.random() for _ in range(MAX_ITER)]

    def _prepare_data(self):
        if not self.scaler_fitted:
            if not self.feature_cols: return
            X_all = self.candidates[self.feature_cols].values
            self.scaler.fit(X_all)
            self.scaler_fitted = True

    def fit(self):
        if not self.observed_indices: return
        self._prepare_data()
        X_raw = self.candidates.iloc[self.observed_indices][self.feature_cols].values
        X_scaled = self.scaler.transform(X_raw)
        y = np.array(self.observed_accs)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(X_scaled, y)

    def propose_next(self, threshold, iteration, max_iter, exploration_weight=0.5):
        """
        优化后的采集函数：
        1. 基础得分 = Pareto分数 * 可行性概率 (利用)
        2. 附加得分 = 归一化不确定性 * 探索权重 (探索)
        通过加法融合，允许算法探索那些"可能不可行但未知"的区域。
        """
        self._prepare_data()
        X_all_raw = self.candidates[self.feature_cols].values
        X_all_scaled = self.scaler.transform(X_all_raw)
        
        # 1. GP 预测
        mu, sigma = self.gp.predict(X_all_scaled, return_std=True)
        
        # 2. 动态 Beta (用于计算 UCB 中的乐观均值)
        # 既然我们在最后额外加了探索分，这里的 beta 可以稍微收敛一点，避免双重激进
        beta = 2.0 - 1.0 * (iteration / max_iter) # 2.0 -> 1.0
        
        ucb_acc = mu + beta * sigma
        
        # 3. 归一化 CR 和 Acc (保证都在 [0, 1])
        crs = self.candidates["cr"].values
        min_cr, max_cr = crs.min(), crs.max()
        norm_cr = (crs - min_cr) / (max_cr - min_cr + 1e-6)

        min_ucb, max_ucb = ucb_acc.min(), ucb_acc.max()
        norm_acc = (ucb_acc - min_ucb) / (max_ucb - min_ucb + 1e-6)

        # 4. 随机标量化权重 (Random Scalarization)
        
        if iteration < max_iter / 4:
            alpha = self.alpha[iteration] * 0.5
        else:
            alpha = self.alpha[iteration]
        scalarized_score = alpha * norm_cr + (1 - alpha) * norm_acc
        
        # 5. 计算可行性概率 (Prob Feasible)
        margin = 2.0
        with np.errstate(divide='ignore'):
            z_scores = (mu - (threshold - margin)) / sigma
        prob_feasible = norm.cdf(z_scores)
        
        # === 核心修改开始 ===
        
        # 6. 计算基础得分 (Base Score)
        # 范围 [0, 1]
        base_score = scalarized_score * prob_feasible
        
        # 7. 计算归一化探索分 (Normalized Exploration Score)
        # 将 sigma 归一化到 [0, 1]，保证量级一致
        min_sigma, max_sigma = sigma.min(), sigma.max()
        sigma_norm = (sigma - min_sigma) / (max_sigma - min_sigma + 1e-6)
        
        # 8. 最终融合
        # final = 利用分 + (权重 * 探索分)
        # 这样即使 prob_feasible 很小，只要 sigma 很大，总分依然可能胜出
        final_score = base_score + (exploration_weight * sigma_norm)
        
        # === 核心修改结束 ===

        # 9. 屏蔽
        mask_indices = list(set(self.observed_indices) | self.skipped_indices)
        final_score[mask_indices] = -np.inf

        best_idx = np.argmax(final_score)
        
        if final_score[best_idx] == -np.inf: 
            return None, 0.0, 0.0
        logging.info(f"Alpha: {alpha:.4f}, NormCR: {norm_cr[best_idx]:.4f}, NormAcc: {norm_acc[best_idx]:.4f}")
        return best_idx, prob_feasible[best_idx], mu[best_idx]
# ================= 主流程 =================

def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout, force=True)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("lm_eval").setLevel(logging.ERROR)

    logging.info("=== Phase 1: Generating & Profiling Search Space ===")
    
    keys = ["transform_type", "heads_selection", "high_key_max_value", "high_value_max_value", "low_key_max_value", "low_value_max_value", "axis_key", "axis_value"]
    values = [SEARCH_SPACE["transform_type"], SEARCH_SPACE["heads_selection"], SEARCH_SPACE["high_key_max_value"], SEARCH_SPACE["high_value_max_value"], SEARCH_SPACE["low_key_max_value"], SEARCH_SPACE["low_value_max_value"], SEARCH_SPACE["axis_key_options"], SEARCH_SPACE["axis_value_options"]]
    
    raw_combinations = list(itertools.product(*values))
    valid_data = [dict(zip(keys, c)) for c in raw_combinations if is_valid_config(dict(zip(keys, c)))]
    df_all_configs = pd.DataFrame(valid_data)

    if df_all_configs.empty: return

    # Cache loading (Simplified for brevity, similar to previous)
    df_to_evaluate = df_all_configs
    df_cached = pd.DataFrame()
    if os.path.exists(f"{MODEL_NAME}/{CACHE_CSV_PATH}"):
        try:
            logging.info(f"Loading CR cache...")
            df_cache = pd.read_csv(f"{MODEL_NAME}/{CACHE_CSV_PATH}")
            for col in ['axis_key', 'axis_value']:
                if col in df_cache.columns:
                    df_cache[col] = df_cache[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            df_cache = df_cache.drop_duplicates(subset=keys)
            df_merged = pd.merge(df_all_configs, df_cache[keys + ['cr']], on=keys, how='left')
            cached_mask = df_merged['cr'].notna()
            df_cached = df_merged[cached_mask].reset_index(drop=True)
            to_evaluate_indices = df_merged[df_merged['cr'].isna()].index
            df_to_evaluate = df_all_configs.iloc[to_evaluate_indices].reset_index(drop=True)
        except Exception: pass

    # Evaluation loop (Simplified)
    df_newly_evaluated = pd.DataFrame()
    if not df_to_evaluate.empty:
        logging.info(f"Evaluating CR for {len(df_to_evaluate)} configs...")
        cr_evaluator = silent_call(CompressionEvaluator, model_name=MODEL_NAME)        
        crs = []
        for i, (_, row) in enumerate(df_to_evaluate.iterrows()):
            if (i + 1) % 50 == 0: logging.info(f"Processing new CR... {i+1}/{len(df_to_evaluate)}")
            crs.append(silent_call(run_cr, cr_evaluator, row.to_dict()))
        del cr_evaluator
        gc.collect()
        torch.cuda.empty_cache()        
        df_newly_evaluated = df_to_evaluate.copy()
        df_newly_evaluated['cr'] = crs
        
        df_save = df_newly_evaluated[df_newly_evaluated['cr'] > 0].copy()
        if not df_save.empty:
            save_cols = keys + ['cr']
            os.makedirs(MODEL_NAME, exist_ok=True)
            file_exists = os.path.exists(f"{MODEL_NAME}/{CACHE_CSV_PATH}")
            df_save[save_cols].to_csv(f"{MODEL_NAME}/{CACHE_CSV_PATH}", mode='a', index=False, header=not file_exists)

    df = pd.concat([df_cached, df_newly_evaluated], ignore_index=True)
    if len(df) == 0: return
    df = df[df["cr"] > 0].reset_index(drop=True)

    acc_evaluator = silent_call(AccuracyEvaluator, model_name=MODEL_NAME, tasks=TASK_TO_SEARCH)

    base_feature_cols = ["heads_selection", "high_key_max_value", "high_value_max_value", "low_key_max_value", "low_value_max_value"]
    cols_to_encode = ["transform_type", "axis_key", "axis_value"]
    df, gp_feature_cols = process_one_hot_encoding(df, cols_to_encode, base_feature_cols)

    logging.info("=== Phase 2: Double-Pruning Bayesian Optimization Search ===")
    
    bo = ConstraintAwareBO(df)
    bo.feature_cols = gp_feature_cols 
    
    best_feasible_cr = 0.0
    best_config = None
    feasible_configs = []
    all_records = [] 
    config_id = 0
    
    # Cold Start
    if len(df) > 3:
        sorted_df = df.sort_values("cr")
        initial_indices = list(set([sorted_df.index[0], sorted_df.index[len(sorted_df)//2], sorted_df.index[-1]]))
    else:
        initial_indices = df.index.tolist()

    logging.info("--- Cold Start ---")
    for idx in initial_indices:
        params = df.iloc[idx].to_dict()
        acc = silent_call(run_acc, acc_evaluator, params)
        bo.observed_indices.append(idx)
        bo.observed_accs.append(acc)
        
        record = params.copy()
        record['accuracy'] = acc
        record['is_feasible'] = (acc >= TARGET_ACC_THRESHOLD)
        all_records.append(record)
        
        if acc >= TARGET_ACC_THRESHOLD:
            feasible_configs.append(record)
            if params['cr'] > best_feasible_cr:
                best_feasible_cr = params['cr']
                best_config = params
            
            # # [COLD START PRUNING] Feasible -> Prune Conservative (Lower CR)
            # skipped = smaller_pruning_select_by(df, params, idx, ["axis_key", "axis_value"])
            # new_skips = [x for x in skipped if x not in bo.skipped_indices and x not in bo.observed_indices]
            # if new_skips:
            #     bo.skipped_indices.update(new_skips)
            #     logging.info(f"   [Pruning] Skipped {len(new_skips)} conservative configs (Acc OK but Low CR)")

        elif acc <= TARGET_ACC_THRESHOLD - ACC_TOLERANCE:
            # [COLD START PRUNING] Infeasible -> Prune Aggressive (Higher CR)
            # skipped = bigger_pruning_select_by(df, params, idx, ["axis_key", "axis_value"])
            # new_skips = [x for x in skipped if x not in bo.skipped_indices and x not in bo.observed_indices]
            # if new_skips:
            #     bo.skipped_indices.update(new_skips)
            #     logging.info(f"   [Pruning] Skipped {len(new_skips)} risky configs (Acc likely Bad)")
            continue

    # BO Loop
    i = 0
    while(i < MAX_ITER):
        total_configs = len(df)
        visited_count = len(set(bo.observed_indices) | bo.skipped_indices)
        remaining_count = total_configs - visited_count
        logging.info(f"--- BO Iteration {i+1}/{MAX_ITER} | Remaining: {remaining_count}/{total_configs} ---")
        
        bo.fit()
        current_weight = max(0.1, EXPLORATION_WEIGHT * (0.99 ** i))
        result = bo.propose_next(TARGET_ACC_THRESHOLD, iteration=i, max_iter=MAX_ITER, exploration_weight=current_weight)
        next_idx, prob, pred_acc = result
        
        if next_idx is None:
            logging.warning("⚠️  No more candidates.")
            break        
        
        next_params = df.iloc[next_idx].to_dict()
        logging.info(f"Proposing (ID:{next_idx}): CR={next_params['cr']:.4f} | Pred Acc={pred_acc:.4f}")
        logging.info(f"   TT={next_params['transform_type']}, HS={next_params['heads_selection']}, MV={next_params['high_key_max_value']} {next_params['high_value_max_value']} {next_params['low_key_max_value']} {next_params['low_value_max_value']}, AK={next_params['axis_key']}, AV={next_params['axis_value']}")
        real_acc = silent_call(run_acc, acc_evaluator, next_params)
        bo.observed_indices.append(next_idx)
        bo.observed_accs.append(real_acc)
        
        record = next_params.copy()
        record['accuracy'] = real_acc
        record['is_feasible'] = (real_acc >= TARGET_ACC_THRESHOLD)
        all_records.append(record)
        
        if real_acc >= TARGET_ACC_THRESHOLD:
            feasible_configs.append(record)
            logging.info(f"✅ FEASIBLE. Acc={real_acc:.4f}")
            if next_params['cr'] > best_feasible_cr:
                best_feasible_cr = next_params['cr']
                best_config = next_params
                logging.info(f"🎉 NEW MAX CR: {best_feasible_cr:.4f}")
            
            # [KEY OPTIMIZATION] If Feasible -> Prune Lower CR
            skipped = smaller_pruning_select_by(df, next_params, next_idx, ["axis_key", "axis_value"])
            new_skips = [x for x in skipped if x not in bo.skipped_indices and x not in bo.observed_indices]
            if new_skips:
                bo.skipped_indices.update(new_skips)
                logging.info(f"🧹 [Conservative Pruning] Skipped {len(new_skips)} dominated configs")

        else:
            logging.info(f"❌ INFEASIBLE. Acc={real_acc:.4f}")
            if real_acc <= TARGET_ACC_THRESHOLD - ACC_TOLERANCE:
                # [EXISTING LOGIC] If Infeasible -> Prune Higher CR
                skipped = bigger_pruning_select_by(df, next_params, next_idx, ["axis_key", "axis_value"])
                new_skips = [x for x in skipped if x not in bo.skipped_indices and x not in bo.observed_indices]
                if new_skips:
                    bo.skipped_indices.update(new_skips)
                    logging.info(f"🧹 [Aggressive Pruning] Skipped {len(new_skips)} risky configs")

        i += 1

    # os.makedirs(MODEL_NAME, exist_ok=True)
    # with open(f"{MODEL_NAME}/{ALL_RESULTS_PATH}", "w") as f:
    #     json.dump(all_records, f, indent=4, default=str)
    
    if feasible_configs:
        feasible_configs.sort(key=lambda x: x['cr'], reverse=True)
        with open(f"{MODEL_NAME}/{FINAL_JSON_PATH}", "w") as f:
            json.dump(feasible_configs, f, indent=4, default=str)
        logging.info(f"Saved {len(feasible_configs)} feasible configs to {MODEL_NAME}/{FINAL_JSON_PATH}")

if __name__ == "__main__":
    main()