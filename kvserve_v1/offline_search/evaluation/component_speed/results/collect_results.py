import argparse
import re
import csv
import os
import sys

def get_log_value(content, pattern, type_cast=float):
    """根据正则从日志内容中提取数值，提取失败则返回 NAN"""
    match = re.search(pattern, content)
    if match:
        try:
            return type_cast(match.group(1))
        except ValueError:
            return "NAN"
    return "NAN"

def parse_sub_log(file_path):
    """解析具体的子日志文件（如 len_1024.log）"""
    data = {
        "input_length": "NAN",
        "original_size": "NAN",
        "prefill_time": "NAN",
        "decode_time": "NAN",
        "prefill_throughput": "NAN",
        "decode_throughput": "NAN"
    }
    
    if not os.path.exists(file_path):
        # 如果文件不存在，直接返回全是NAN的数据
        return data
        
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # 根据日志格式提取数据
        # 示例: Input length: 1024
        data["input_length"] = get_log_value(content, r"Input length:\s+(\d+)", int)
        # 示例: Original size: 0.1250 GB
        data["original_size"] = get_log_value(content, r"Original size:\s+([\d\.]+)\s+GB")
        # 示例: Prefill time: 0.0074 s
        data["prefill_time"] = get_log_value(content, r"Prefill time:\s+([\d\.]+)\s+s")
        # 示例: Decode time: 0.0021 s
        data["decode_time"] = get_log_value(content, r"Decode time:\s+([\d\.]+)\s+s")
        # 示例: Prefill throughput: 16.99 GB/s
        data["prefill_throughput"] = get_log_value(content, r"Prefill throughput:\s+([\d\.]+)\s+GB/s")
        # 示例: Decode throughput: 58.21 GB/s
        data["decode_throughput"] = get_log_value(content, r"Decode throughput:\s+([\d\.]+)\s+GB/s")
        
    except Exception as e:
        print(f"Error reading sub-log {file_path}: {e}", file=sys.stderr)
        
    return data

def main():
    BASE_LOG_DIR = "/root/workspace/Infer_Comm/evaluation/component_speed/logs"
    logs = [
        "cachegen_speed.log",
        "hadamard_speed.log",
        "quantizer_speed.log",
        "ans_speed.log",
        "bitcomp_speed.log",
    ]
    model = "Qwen2.5-32B-Instruct"
    results = []
    
    # 定义CSV表头（包含单位）
    headers = [
        "type", 
        "input_length", 
        "original_size(GB)", 
        "prefill_time(s)", 
        "decode_time(s)", 
        "prefill_throughput(GB/s)", 
        "decode_throughput(GB/s)"
    ]

    for main_log in logs:
        # 1. 确定 Type 名称 (从文件名提取，去除 _speed.log 或 .log)
        basename = os.path.basename(main_log)
        if "_speed.log" in basename:
            log_type = basename.replace("_speed.log", "")
        else:
            log_type = os.path.splitext(basename)[0]
        
        if not os.path.exists(os.path.join(BASE_LOG_DIR, main_log)):
            print(f"Warning: Main log file '{os.path.join(BASE_LOG_DIR, main_log)}' not found.", file=sys.stderr)
            continue

        print(f"Processing main log: {os.path.join(BASE_LOG_DIR, main_log)} (Type: {log_type})")

        # 2. 读取主日志寻找子日志路径
        with open(os.path.join(BASE_LOG_DIR, main_log), 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if "Log file:" in line:
                    # 提取路径: Log file: /path/to/xxx.log
                    sub_log_path = line.split("Log file:")[1].strip()
                    
                    # 3. 过滤模型
                    # 检查路径中是否包含模型名称 (通常路径结构包含模型名)
                    if model in sub_log_path:
                        # 4. 解析子日志
                        sub_data = parse_sub_log(os.path.join(BASE_LOG_DIR, sub_log_path))
                        
                        row = {
                            "type": log_type,
                            "input_length": sub_data["input_length"],
                            "original_size(GB)": sub_data["original_size"],
                            "prefill_time(s)": sub_data["prefill_time"],
                            "decode_time(s)": sub_data["decode_time"],
                            "prefill_throughput(GB/s)": sub_data["prefill_throughput"],
                            "decode_throughput(GB/s)": sub_data["decode_throughput"]
                        }
                        results.append(row)

    # 排序：先按 type 字母序，再按 input_length 数字大小
    def sort_key(x):
        try:
            length = int(x['input_length']) if x['input_length'] != "NAN" else -1
            return (x['type'], length)
        except:
            return (x['type'], 0)
    
    results.sort(key=sort_key)

    # 5. 写入 CSV
    output_file = f"{model}_speed.csv"
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"Successfully saved statistics to: {output_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()