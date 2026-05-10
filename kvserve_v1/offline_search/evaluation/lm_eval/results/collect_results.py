import re
import os
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION (配置项)
# ==========================================

# 1. 要统计的 Cache 类型列表 
# (对应 logs/all_scripts_log/{type}_job_accuracy.log)
CACHE_TYPES = [
    'default', 
    'cachegen', 
    'kivi', 
    'duoattn',
    'custom_qwen_none',
    'custom_qwen_hadamard'
]

# 2. 指定要统计的模型名称 
# (需与 Log 中 Model: 后的名称完全一致，忽略大小写空白差异可调整代码，这里默认精确匹配)
TARGET_MODEL = 'Qwen2.5-7B-Instruct' 
os.makedirs(os.path.join(os.path.dirname(__file__), f"{TARGET_MODEL}"), exist_ok=True)

# 3. 输出 CSV 文件路径
OUTPUT_CSV_PATH = os.path.join(os.path.dirname(__file__), f"{TARGET_MODEL}", "summary_results.csv")

# ==========================================
# REGEX PATTERNS
# ==========================================
# 1. 主日志匹配模式
PATTERN_JOB_START = re.compile(r"Starting job \d+/\d+ on Physical GPU")
PATTERN_MODEL_TASK = re.compile(r"Model:\s*(.+?),\s*Task:\s*(.+)")
PATTERN_LOG_FILE = re.compile(r"Log file:\s*(.+)")

# 2. 子日志匹配模式
# 匹配 Compression Ratio
PATTERN_CR = re.compile(r"Total Compression ratio:\s*([\d\.]+)")

# 匹配 Accuracy
# 例如: "  score,none: 0.4539" 或 "  pass@1,create_test: 0.6646"
# 使用更通用的 [^\s:]+ 来匹配键名，以支持 @ 等特殊字符，同时避免匹配含空格的行(如 Total Compression ratio)
PATTERN_SCORE_GENERIC = re.compile(r"^\s*([^\s:]+)\s*:\s*(\d+\.\d+)", re.MULTILINE)

# 辅助标记：通常结果出现在 "--- Full results ---" 之后
PATTERN_RESULTS_HEADER = re.compile(r"--- Full results ---")

# ==========================================
# FUNCTIONS
# ==========================================

def parse_child_log(log_path):
    """
    读取子日志文件，提取 Accuracy 和 Compression Ratio
    """
    accuracy = np.nan
    cr = np.nan

    if not os.path.exists(log_path):
        # 简单的相对路径处理尝试
        if not os.path.isabs(log_path):
             # 这里的处理视具体运行环境而定，暂且保持原样
             pass
        print(f"[Warning] Log file not found: {log_path}")
        return accuracy, cr

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # --- 提取 Accuracy ---
            results_content = content
            match_header = PATTERN_RESULTS_HEADER.search(content)
            if match_header:
                results_content = content[match_header.end():]
            
            matches_score = PATTERN_SCORE_GENERIC.findall(results_content)
            
            if matches_score:
                try:
                    # 取第一个匹配到的分数作为主要指标
                    # findall 返回元组列表 [('key', 'value'), ...]，取第一个元组的第二个元素(数值)
                    val = float(matches_score[0][1])
                    accuracy = round(val * 100, 2)
                except (ValueError, IndexError):
                    pass
            
            # --- 提取 CR ---
            match_cr = PATTERN_CR.findall(content)
            if match_cr:
                try:
                    val = float(match_cr[-1]) # 取最后一个
                    cr = round(val, 2)
                except ValueError:
                    pass
                    
    except Exception as e:
        print(f"[Error] Reading {log_path}: {e}")

    return accuracy, cr

def parse_main_log(main_log_path):
    """
    解析主日志文件，返回任务列表
    """
    jobs = []
    
    if not os.path.exists(main_log_path):
        print(f"[Warning] Main log file not found at {main_log_path}")
        return jobs

    with open(main_log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_job = {}
    config_lines = []
    is_collecting_config = False

    for line in lines:
        line = line.strip()
        
        # 遇到新任务开始
        if PATTERN_JOB_START.search(line):
            # 保存上一个任务（如果有）
            if current_job:
                current_job['config'] = "; ".join(config_lines)
                jobs.append(current_job)
            
            # 重置
            current_job = {}
            config_lines = []
            is_collecting_config = False
            continue

        # 匹配 Model 和 Task
        match_mt = PATTERN_MODEL_TASK.search(line)
        if match_mt:
            current_job['model'] = match_mt.group(1).strip()
            current_job['task'] = match_mt.group(2).strip()
            is_collecting_config = True 
            continue

        # 匹配 Log file
        match_log = PATTERN_LOG_FILE.search(line)
        if match_log:
            current_job['log_path'] = match_log.group(1).strip()
            is_collecting_config = False 
            continue
        
        # 收集 config 信息 (位于 Model 和 Log file 之间)
        if is_collecting_config and line and not line.startswith("-----"):
            config_lines.append(line)

    # 保存最后一个任务
    if current_job:
        current_job['config'] = "; ".join(config_lines)
        jobs.append(current_job)
        
    return jobs

def main():
    all_results = []

    print(f"Target Model: {TARGET_MODEL}")
    print(f"Processing Cache Types: {CACHE_TYPES}")
    
    for cache_type in CACHE_TYPES:
        # 构建主日志路径
        log_filename = f"{cache_type}_job_accuracy.log"
        main_log_path = os.path.join(os.path.dirname(__file__), f'../logs/all_scripts_log/{log_filename}')
        
        print(f"\n--- Processing {cache_type} ---")
        print(f"Reading main log from: {main_log_path}")
        
        jobs = parse_main_log(main_log_path)
        print(f"Found {len(jobs)} jobs total.")
        
        processed_count = 0
        for job in jobs:
            model = job.get('model', 'Unknown')
            
            # 过滤模型
            if model != TARGET_MODEL:
                continue
                
            task = job.get('task', 'Unknown')
            config = job.get('config', '')
            log_path = job.get('log_path', '')
            
            # 获取详细数据
            accuracy = np.nan
            cr = np.nan
            if log_path:
                accuracy, cr = parse_child_log(log_path)
            
            # 格式化结果: accuracy/compression_ratio
            if np.isnan(accuracy) and np.isnan(cr):
                combined_res = np.nan
            else:
                acc_str = f"{accuracy:.2f}" if not np.isnan(accuracy) else "NaN"
                cr_str = f"{cr:.2f}" if not np.isnan(cr) else "NaN"
                combined_res = f"{acc_str}/{cr_str}"

            all_results.append({
                'cache_type': cache_type,
                'config': config,
                'task': task,
                'result': combined_res
            })
            processed_count += 1
            
        print(f"Matched {processed_count} jobs for model {TARGET_MODEL}.")

    # === 数据汇总与透视 ===
    if not all_results:
        print("\nNo results found matching criteria.")
        return

    df = pd.DataFrame(all_results)
    
    # 透视表：将 task 转为列，填充 result
    df_pivot = df.pivot_table(
        index=['cache_type', 'config'], 
        columns='task', 
        values='result', 
        aggfunc='first'
    ).reset_index()
    
    # 调整列顺序
    # 期望: cache_type, config, <tasks...>
    cols = df_pivot.columns.tolist()
    
    base_cols = ['cache_type', 'config']
    for c in base_cols:
        if c in cols: cols.remove(c)
    
    # task 列按字母顺序排序
    task_cols = sorted(cols)
    
    # 最终列顺序
    final_cols = base_cols + task_cols
    df_final = df_pivot[final_cols]
    
    # 排序：按照 CACHE_TYPES 的顺序对 cache_type 列进行排序
    df_final['cache_type'] = pd.Categorical(df_final['cache_type'], categories=CACHE_TYPES, ordered=True)
    df_final = df_final.sort_values(['cache_type', 'config'])

    # 保存
    print(f"\nSaving summary to: {OUTPUT_CSV_PATH}")
    df_final.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
    print("Done.")
    # print(df_final.head())

if __name__ == "__main__":
    main()
