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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from offline_search.evaluation.param_search.cr_evaluator import CompressionEvaluator
from offline_search.evaluation.param_search.acc_evaluator import AccuracyEvaluator

# ================= 配置区域 =================

# 2. 目标设置
BASELINE_ACC = 100
ACC_TOLERANCE = 3
TARGET_ACC_THRESHOLD = BASELINE_ACC - ACC_TOLERANCE
PRUNING_EPSILON = 0.2
MAX_ITER = 5
EXPLORATION_WEIGHT = 1
SEED = 42
WHETHER_TO_EXPLORE = True
# 3. 搜索空间 (保持不变)
SEARCH_SPACE = {
    "transform_type": ["hadamard"],
    "heads_selection": [0.3, 0.5, 0.7, 0.9], 
    "high_key_max_value": [12, 10, 8],
    "high_value_max_value": [12, 10, 8],
    "low_key_max_value": [6, 4],
    "low_value_max_value": [6, 4],
    "axis_key_options": [(2,)],
    "axis_value_options": [(1, 3)]
}
MODEL_NAME = "Qwen2.5-7B-Instruct"
TASK_TO_SEARCH = ["longbench_2wikimqa"]
DATASET_LIMIT = 5
CACHE_CSV_PATH = "search_space.csv"
FINAL_JSON_PATH = f"tolerance_{ACC_TOLERANCE}_results.json"

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
    # 禁用 logging 输出 (CRITICAL 及以下级别都不显示)
    logging.disable(logging.CRITICAL)
    
    temp_stdout = io.StringIO()
    temp_stderr = io.StringIO()
    try:
        with redirect_stdout(temp_stdout), redirect_stderr(temp_stderr):
            return func(*args, **kwargs)
    finally:
        # 恢复 logging
        logging.disable(logging.NOTSET)

def run_cr(evaluator, params):
    logging.info(f"   >>> Running Compression Evaluation ")
    try:
        # 直接调用函数
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

def bigger_pruning_select_by(df, current_params, current_idx, current_iteration, isolation_columns):
    if current_iteration < MAX_ITER / 5:
        return []
    
    # 1. 构建隔离掩码 (Isolation Mask)
    # 逻辑：找出 DataFrame 中所有与 current_params 在 isolation_columns 上值完全相同的行
    isolation_mask = pd.Series(True, index=df.index)
    for col in isolation_columns:
        if col in df.columns:
            isolation_mask &= (df[col] == current_params[col])

    # 2. 构建条件掩码 (Weaker Mask)
    # 逻辑：定义参数的单调性。
    # 过滤掉压缩力度更大的参数
    max_value = max(current_params['high_key_max_value'], current_params['high_value_max_value'], current_params['low_key_max_value'], current_params['low_value_max_value'])
    min_value = min(current_params['high_key_max_value'], current_params['high_value_max_value'], current_params['low_key_max_value'], current_params['low_value_max_value'])
    
    if max_value - min_value < 8 or current_iteration > MAX_ITER / 2:
        weaker_mask = pd.Series(True, index=df.index)
        weaker_mask &= (df['cr'] >= current_params['cr'] + 2 * PRUNING_EPSILON)
    else:
        return []
    
    # 3. 排除自己 (Not Self)
    not_self = (df.index != current_idx)    

    # 4. 合并所有掩码
    final_mask = isolation_mask & weaker_mask & not_self
    
    # 5. 返回满足条件的索引
    return df[final_mask].index.tolist()

def smaller_pruning_select_by(df, current_params, current_idx, current_iteration, isolation_columns):
    if current_iteration <= MAX_ITER / 5:
        return []
    
    isolation_mask = pd.Series(True, index=df.index)
    for col in isolation_columns:
        if col in df.columns:
            isolation_mask &= (df[col] == current_params[col])

    # CR 比当前小 (更保守)
    # 使用 epsilon 确保不会剪掉 CR 非常接近的点
    conservative_mask = pd.Series(True, index=df.index)
    conservative_mask &= (df['cr'] <= current_params['cr'] - PRUNING_EPSILON / 2)
    
    not_self = (df.index != current_idx)    
    final_mask = isolation_mask & conservative_mask & not_self
    return df[final_mask].index.tolist()

def process_one_hot_encoding(df, encode_cols, current_feature_cols):
    """
    通用独热编码处理函数。
    
    Args:
        df (pd.DataFrame): 搜索空间的 DataFrame。
        encode_cols (list): 需要进行独热编码的列名列表，例如 ['axis_key', 'axis_value']。
        current_feature_cols (list): 当前用于 GP 模型的数值特征列列表。
        
    Returns:
        df (pd.DataFrame): 包含新生成 One-Hot 列的 DataFrame。
        final_feature_cols (list): 更新后的特征列列表（加入了 One-Hot 列）。
    """
    # 复制一份特征列表，避免修改原对象
    final_feature_cols = list(current_feature_cols)
    
    for col in encode_cols:
        if col not in df.columns:
            print(f"[Warning] Column '{col}' not found in DataFrame. Skipping.")
            continue
            
        # 1. 建立映射 (Value -> Integer ID)
        # 为了处理 list 等不可哈希类型，先统一转为 tuple (如果是 list 的话)
        # 如果已经是 tuple 或 int/str，这步操作是安全的
        try:
            # 尝试直接获取唯一值
            unique_vals = sorted(list(set(df[col])))
        except TypeError:
            # 如果报错（通常是因为列里存的是 list，不可哈希），则先临时转为 tuple
            temp_series = df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)
            unique_vals = sorted(list(set(temp_series)))

        val_map = {v: i for i, v in enumerate(unique_vals)}
        
        # 2. 生成 ID 列 (例如 axis_key -> axis_key_id)
        # 使用 apply 而不是 map，避开 pandas 的 tuple 索引 bug
        id_col_name = f"{col}_id"
        
        # 注意：这里要确保查表时用的 key 类型和 val_map 里的 key 一致
        # 如果上面做了 list->tuple 转换，这里查表也要转
        def get_id(x):
            key = tuple(x) if isinstance(x, list) else x
            return val_map[key]
            
        df[id_col_name] = df[col].apply(get_id)
        
        # 3. 生成 One-Hot 列
        # prefix=col 会生成如 "axis_key_0", "axis_key_1" 这样的列名
        dummies = pd.get_dummies(df[id_col_name], prefix=col)
        
        # 4. 拼接到原 DF
        df = pd.concat([df, dummies], axis=1)
        
        # 5. 更新特征列表
        # 将新生成的 dummy 列名加入 feature_cols
        new_cols = list(dummies.columns)
        final_feature_cols.extend(new_cols)
        
        # [可选] 将 ID 列也加入，方便后续剪枝逻辑使用（如果你的剪枝逻辑依赖 _id 列）
        # 如果剪枝逻辑用原始列（如 axis_key）判断，则不需要加 ID 列到 feature_cols
        # 这里只返回给 GP 用的 numerical columns
        
    return df, final_feature_cols

def check_early_stopping(df, bo):
    """
    检查是否满足早退条件：
    1. 剩余搜索空间的 CR 最大值与最小值之差 <= 2 * PRUNING_EPSILON
    2. 连续失败次数 (consecutive_fail_count) > MAX_ITER / 10
    """
    # 找出剩余未探索的索引
    visited = set(bo.observed_indices) | bo.skipped_indices
    remaining_mask = ~df.index.isin(visited)
    
    if not remaining_mask.any():
        return False
        
    remaining_crs = df.loc[remaining_mask, 'cr']
    max_cr = remaining_crs.max()
    min_cr = remaining_crs.min()
    diff = max_cr - min_cr
    
    if diff > 2 * PRUNING_EPSILON + PRUNING_EPSILON / 2:
        bo.consecutive_fail_count = 0
        return False
    elif bo.consecutive_fail_count > MAX_ITER / 10:
        logging.info(f"🛑 Early Stopping Triggered: Remaining CR Range [{min_cr:.4f}, {max_cr:.4f}] (Diff={diff:.4f} <= {2 * PRUNING_EPSILON + PRUNING_EPSILON / 2:.4f}) AND Consecutive Failures ({bo.consecutive_fail_count} > {MAX_ITER / 10})")
        return True
        
    return False

# ================= 贝叶斯优化类 (保持不变) =================

class ConstraintAwareBO:
    def __init__(self, candidate_pool_df, seed=42):
        self.candidates = candidate_pool_df
        self.observed_indices = [] 
        self.observed_accs = []
        self.skipped_indices = set()
        self.seed = seed
        self.feature_cols = None
        self.consecutive_fail_count = 0
        
        kernel = Matern(nu=1.5, length_scale=1.0) + WhiteKernel(noise_level=1e-4)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True, random_state=self.seed)

        self.scaler = MinMaxScaler()
        self.scaler_fitted = False

    def _prepare_data(self):
        """辅助函数：确保在第一次使用前 scaler 已经 fit 好了"""
        # 只有当 feature_cols 被赋值后，才能进行 fit
        if not self.scaler_fitted:
            if not self.feature_cols:
                return # 还没准备好
            
            # 对整个搜索空间的所有特征进行数据归一化
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
            # 喂给 GP 的是归一化后的数据
            self.gp.fit(X_scaled, y)

    def propose_next(self, threshold, exploration_weight=0.1):
        self._prepare_data()
        X_all_raw = self.candidates[self.feature_cols].values
        X_all_scaled = self.scaler.transform(X_all_raw)
        mu, sigma = self.gp.predict(X_all_scaled, return_std=True)
        
        with np.errstate(divide='ignore'):
            z_scores = (mu - threshold) / sigma
        prob_feasible = norm.cdf(z_scores)
        
        # 基础分：期望压缩率
        base_score = self.candidates["cr"].values * prob_feasible
        
        # 探索分：不确定性 (Sigma) * 权重
        norm_sigma = sigma / sigma.max() if sigma.max() > 0 else sigma
        exploration_score = norm_sigma * exploration_weight * self.candidates["cr"].mean()
        if not WHETHER_TO_EXPLORE: exploration_score = 0
        
        acquisition_scores = base_score + exploration_score
        
        # 屏蔽已观测和剪枝的点
        for idx in self.observed_indices: acquisition_scores[idx] = -1.0
        for idx in self.skipped_indices: acquisition_scores[idx] = -1.0

        best_idx = np.argmax(acquisition_scores)
        if acquisition_scores[best_idx] == -1.0: return None, 0.0, 0.0 
        return best_idx, prob_feasible[best_idx], mu[best_idx]

# ================= 主流程 =================

def main():
    # ====================================================
    # Logging Config
    # ====================================================
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True
    )
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("lm_eval").setLevel(logging.ERROR)

    logging.info("=== Phase 1: Generating & Profiling Search Space ===")
    
    keys = ["transform_type", "heads_selection", "high_key_max_value", "high_value_max_value", "low_key_max_value", "low_value_max_value", "axis_key", "axis_value"]
    values = [SEARCH_SPACE["transform_type"], SEARCH_SPACE["heads_selection"], SEARCH_SPACE["high_key_max_value"], SEARCH_SPACE["high_value_max_value"], SEARCH_SPACE["low_key_max_value"], SEARCH_SPACE["low_value_max_value"], SEARCH_SPACE["axis_key_options"], SEARCH_SPACE["axis_value_options"]]
    
    raw_combinations = list(itertools.product(*values))
    valid_data = [dict(zip(keys, c)) for c in raw_combinations if is_valid_config(dict(zip(keys, c)))]
    df_all_configs = pd.DataFrame(valid_data)

    if df_all_configs.empty:
        logging.error("Error: No valid configurations generated from search space.")
        return

    df_to_evaluate = df_all_configs
    df_cached = pd.DataFrame()

    if os.path.exists(f"results/{MODEL_NAME}/{CACHE_CSV_PATH}"):
        try:
            logging.info(f"Found cache: results/{MODEL_NAME}/{CACHE_CSV_PATH}. Loading cached CR values.")
            df_cache = pd.read_csv(f"results/{MODEL_NAME}/{CACHE_CSV_PATH}")
            
            # Ensure tuple columns from CSV (read as strings) are converted back to tuples
            for col in ['axis_key', 'axis_value']:
                if col in df_cache.columns:
                    df_cache[col] = df_cache[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            
            # Deduplicate cache just in case to avoid merge explosion
            df_cache = df_cache.drop_duplicates(subset=keys)

            # Left merge to find which configs are new (will have NaN for 'cr')
            df_merged = pd.merge(df_all_configs, df_cache[keys + ['cr']], on=keys, how='left')

            # Split into already cached and to-be-evaluated
            cached_mask = df_merged['cr'].notna()
            df_cached = df_merged[cached_mask].reset_index(drop=True)
            
            to_evaluate_indices = df_merged[df_merged['cr'].isna()].index
            df_to_evaluate = df_all_configs.iloc[to_evaluate_indices].reset_index(drop=True)
            
            logging.info(f"{len(df_cached)} configs found in cache. {len(df_to_evaluate)} new configs to evaluate.")
            
        except Exception as e:
            logging.warning(f"Could not load or parse cache file 'results/{MODEL_NAME}/{CACHE_CSV_PATH}'. Re-evaluating all. Error: {e}")
            df_to_evaluate = df_all_configs
            df_cached = pd.DataFrame()
    
    df_newly_evaluated = pd.DataFrame()
    if not df_to_evaluate.empty:
        logging.info(f"Evaluating CR for {len(df_to_evaluate)} configurations...")
        logging.info("=== Initializing Compression Evaluator ===")
        cr_evaluator = silent_call(CompressionEvaluator, model_name=MODEL_NAME)        
        crs = []
        for i, (_, row) in enumerate(df_to_evaluate.iterrows()):
            if (i + 1) % 10 == 0: logging.info(f"Processing new CR... {i+1}/{len(df_to_evaluate)}")
            cr_val = silent_call(run_cr, cr_evaluator, row.to_dict())
            crs.append(cr_val)

        logging.info(f"Finished evaluating {len(df_to_evaluate)} new configurations.")

        # 显式释放资源
        del cr_evaluator
        gc.collect()
        torch.cuda.empty_cache()        
        
        df_newly_evaluated = df_to_evaluate.copy()
        df_newly_evaluated['cr'] = crs
        
        # Append new valid results to the cache file immediately
        df_save = df_newly_evaluated[df_newly_evaluated['cr'] > 0].copy()
        if not df_save.empty:
            save_cols = keys + ['cr']
            os.makedirs(f"results/{MODEL_NAME}", exist_ok=True)
            file_exists = os.path.exists(f"results/{MODEL_NAME}/{CACHE_CSV_PATH}")
            df_save[save_cols].to_csv(f"results/{MODEL_NAME}/{CACHE_CSV_PATH}", mode='a', index=False, header=not file_exists)
            logging.info(f"Appended {len(df_save)} new configurations to results/{MODEL_NAME}/{CACHE_CSV_PATH}")
            # return

    # Combine cached and newly evaluated results
    df = pd.concat([df_cached, df_newly_evaluated], ignore_index=True)

    logging.info("=== Initializing Accuracy Evaluator ===")
    acc_evaluator = silent_call(AccuracyEvaluator, model_name=MODEL_NAME, tasks=TASK_TO_SEARCH, limit=DATASET_LIMIT, random_seed=SEED)

    # The rest of the processing happens on the combined dataframe
    if len(df) == 0:
        logging.error("Error: No valid configurations found or calculated.")
        return

    # Filter out invalid CRs for internal usage
    df = df[df["cr"] > 0].reset_index(drop=True)
    
    logging.info("Processing One-Hot Encodings for GP Model...")

    # 1. 定义基础特征列 (数值型参数)
    base_feature_cols = [
        "heads_selection", 
        "high_key_max_value", "high_value_max_value",
        "low_key_max_value", "low_value_max_value"
    ]
    
    # 2. 定义需要 One-Hot 编码的非连续列
    # 只要 SEARCH_SPACE 里有的非连续列，都加到这里
    cols_to_encode = ["transform_type", "axis_key", "axis_value"]
    
    # 3. 调用 process_one_hot_encoding 
    # 这个函数会自动生成 ID 列 (axis_key_id) 和 One-Hot 列 (axis_key_0, axis_key_1...)
    # 并返回更新后的 df 和特征列表
    df, gp_feature_cols = process_one_hot_encoding(df, cols_to_encode, base_feature_cols)
    
    logging.info(f"Features for GP Model: {gp_feature_cols}")

    # ================= Phase 2: 贝叶斯优化 =================
    logging.info("=== Phase 2: Bayesian Optimization Search ===")
    
    logging.info(f"Target Accuracy: >= {TARGET_ACC_THRESHOLD:.4f}")
    
    bo = ConstraintAwareBO(df, seed=SEED)
    bo.feature_cols = gp_feature_cols 
    
    best_feasible_cr = 0.0
    best_config = None
    
    # ----------------- Cold Start -----------------
    feasible_configs = []
    config_id = 0
    if len(df) > 3:
        # 按 CR 排序
        sorted_df = df.sort_values("cr")
        idx_min = sorted_df.index[0]                  # CR 最小 (最安全)
        idx_max = sorted_df.index[-1]                 # CR 最大 (最危险)
        idx_mid = sorted_df.index[len(sorted_df)//2]  # CR 中位数 (边界探索)
        # 严格顺序: Max -> Mid -> Min，并去重
        candidates = [idx_max, idx_mid, idx_min]
        selected_indices = []
        seen = set()
        for idx in candidates:
            if idx not in seen:
                selected_indices.append(idx)
                seen.add(idx)
        
        initial_indices = np.array(selected_indices)
    else:
        initial_indices = np.random.choice(df.index, min(3, len(df)), replace=False)
    logging.info("--- Cold Start ---")
    for idx in initial_indices:
        params = df.iloc[idx].to_dict()
        acc = silent_call(run_acc, acc_evaluator, params)
        bo.observed_indices.append(idx)
        bo.observed_accs.append(acc)
        
        logging.info(f"Testing Initial Config (ID:{idx}): CR={params['cr']:.4f} -> Acc={acc:.4f}")
        if acc >= TARGET_ACC_THRESHOLD:
            # Record feasible config
            record = params.copy()
            record['accuracy'] = acc
            record['config_id'] = config_id
            config_id += 1
            feasible_configs.append(record)

            if params['cr'] > best_feasible_cr:
                best_feasible_cr = params['cr']
                best_config = params
                # Batch Pruning logic
                # cutoff_cr = best_feasible_cr - PRUNING_EPSILON / 2
                # prune_indices = df[df['cr'] < cutoff_cr].index.tolist()
                # new_skips = [x for x in prune_indices if x not in bo.skipped_indices and x not in bo.observed_indices]
                # bo.skipped_indices.update(new_skips)
                # if new_skips:
                #     logging.info(f"   [Pruning] Skipped {len(new_skips)} configs with CR < {cutoff_cr:.4f}")
        elif acc <= TARGET_ACC_THRESHOLD - ACC_TOLERANCE:
            # condition_indices = bigger_pruning_select_by(df, params, idx, ["axis_key", "axis_value"])
            # new_skips = [x for x in condition_indices if x not in bo.skipped_indices and x not in bo.observed_indices]
            # bo.skipped_indices.update(new_skips)
            # if new_skips:
            #     logging.info(f"   [Pruning] Conditionally Skipped {len(new_skips)} configs with CR > {(params['cr'] + 2 * PRUNING_EPSILON):.4f}")        
            continue

    # ----------------- BO Loop -----------------
    i = 0
    while(i < MAX_ITER):
        total_configs = len(df)
        visited_count = len(set(bo.observed_indices) | bo.skipped_indices)
        remaining_count = total_configs - visited_count
        logging.info(f"--- BO Iteration {i+1}/{MAX_ITER} | Remaining: {remaining_count}/{total_configs} ---")
        
        bo.fit()
        # 探索权重衰减
        if i < MAX_ITER / 5:
            current_weight = max(0.1, EXPLORATION_WEIGHT * (0.99 ** i))
        else:
            current_weight = max(0.1, EXPLORATION_WEIGHT * (0.99 ** i) / 2)
        result = bo.propose_next(TARGET_ACC_THRESHOLD, current_weight)
        next_idx, prob, pred_acc = result
        if next_idx is None:
            logging.warning("⚠️  No more candidates.")
            break        
        next_params = df.iloc[next_idx].to_dict()
        
        logging.info(f"Proposing Config (ID:{next_idx}): CR={next_params['cr']:.4f} | Pred Acc={pred_acc:.4f}")
        
        # Single Point Pruning
        # if best_feasible_cr > 0 and next_params['cr'] <= best_feasible_cr:
        #     logging.info(f"⏭️  SKIPPING (Pruned). Reason: CR <= Best {best_feasible_cr:.4f}")
        #     bo.skipped_indices.add(next_idx)
        #     continue 
        
        logging.info(f"   TT={next_params['transform_type']}, HS={next_params['heads_selection']}, MV={next_params['high_key_max_value']} {next_params['high_value_max_value']} {next_params['low_key_max_value']} {next_params['low_value_max_value']}, AK={next_params['axis_key']}, AV={next_params['axis_value']}")
        real_acc = silent_call(run_acc, acc_evaluator, next_params)
        bo.observed_indices.append(next_idx)
        bo.observed_accs.append(real_acc)
        
        if real_acc >= TARGET_ACC_THRESHOLD:
            bo.consecutive_fail_count = 0
            # Record feasible config
            record = next_params.copy()
            record['accuracy'] = real_acc
            record['config_id'] = config_id
            config_id += 1
            feasible_configs.append(record)

            logging.info(f"✅ Configuration is FEASIBLE. Acc={real_acc:.4f} > {TARGET_ACC_THRESHOLD:.4f}")
            if next_params['cr'] > best_feasible_cr:
                best_feasible_cr = next_params['cr']
                best_config = next_params
                logging.info(f"🎉 NEW BEST! Max CR: {best_feasible_cr:.4f}")
                skipped_indices = smaller_pruning_select_by(df, next_params, next_idx, i, ["axis_key", "axis_value"])
                new_skips = [x for x in skipped_indices if x not in bo.skipped_indices and x not in bo.observed_indices]
                bo.skipped_indices.update(new_skips)
                if new_skips:
                    logging.info(f"🧹 [Batch Pruning] Skipped {len(new_skips)} dominated configs")
            else:
                logging.info(f"⏭️ (CR <= Best)")
                skipped_indices = smaller_pruning_select_by(df, next_params, next_idx, i, ["axis_key", "axis_value"])
                new_skips = [x for x in skipped_indices if x not in bo.skipped_indices and x not in bo.observed_indices]
                bo.skipped_indices.update(new_skips)
                if new_skips:
                    logging.info(f"🧹 [Batch Pruning] Skipped {len(new_skips)} dominated configs")
                # continue
        else:
            if i > MAX_ITER / 2:
                bo.consecutive_fail_count += 1
            logging.info(f"❌ INFEASIBLE (Acc={real_acc:.4f} < {TARGET_ACC_THRESHOLD:.4f})")
            if real_acc <= TARGET_ACC_THRESHOLD - ACC_TOLERANCE:
                skipped_indices = bigger_pruning_select_by(df, next_params, next_idx, i, ["axis_key", "axis_value"])
                new_skips = [x for x in skipped_indices if x not in bo.skipped_indices and x not in bo.observed_indices]
                bo.skipped_indices.update(new_skips)
                if new_skips:
                    logging.info(f"🧹 [Batch Pruning] Conditionally Skipped {len(new_skips)} configs with CR > {(next_params['cr'] + 2 * PRUNING_EPSILON):.4f}")

        if check_early_stopping(df, bo):
            break

        i += 1

    logging.info("================ Search Finished ================")
    os.makedirs(f"results/{MODEL_NAME}", exist_ok=True)
    if feasible_configs:
        logging.info(f"Found {len(feasible_configs)} feasible configurations.")
        if best_config:
             logging.info(f"Best Config (CR={best_feasible_cr:.4f}):")
             logging.info(json.dumps(best_config, indent=2, default=str))

        with open(f"results/{MODEL_NAME}/{FINAL_JSON_PATH}", "w") as f:
            json.dump(feasible_configs, f, indent=4, default=str)
        logging.info(f"Saved all feasible configurations to results/{MODEL_NAME}/{FINAL_JSON_PATH}")
    elif best_config:
        final = {k: v for k, v in best_config.items()}
        logging.info(json.dumps(final, indent=2, default=str))
        with open(f"results/{MODEL_NAME}/{FINAL_JSON_PATH}", "w") as f:
            json.dump(final, f, indent=4, default=str)

if __name__ == "__main__":
    main()