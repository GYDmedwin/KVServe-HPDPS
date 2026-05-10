import json
import os
import sys

# ================= 配置区域 =================

# 输入文件路径 (请修改为您实际的 JSON 文件路径)
# 默认指向您最近浏览的目录下的文件，请确认
INPUT_FILE = "/home/bingxing2/home/scx9kvs/mxy/Infer_Comm/evaluation/param_search/Meta-Llama-3.1-8B-Instruct/none_merged_qasper2.json"

# 在此处填入您测得的组件速度 (单位: ms)
# 这些值将用于计算每个配置的总 Latency
# 请根据 quantizer_speed.py 和 hadamard_speed.py 的输出结果修改以下数值
HADAMARD = 19.635
HEAD_LEVEL = 31.2174
LAYER_LEVEL = 28.4696
ANS = 3.7565
BITCOMP = 2.2395

# ===========================================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} does not exist.")
        print(f"Current working directory: {os.getcwd()}")
        return

    print(f"Reading from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    # 统一处理为列表
    is_list = isinstance(data, list)
    if not is_list:
        data = [data]
    
    print(f"Found {len(data)} configurations.")
    lat = HEAD_LEVEL + BITCOMP
    for i, config in enumerate(data):
        config["latency"] = round(lat, 4)
        # 可选：打印部分日志
        # print(f"Config {i}: Type={config.get('transform_type')}, Quantized={config.get('high_key_max_value')<256} -> Latency={lat} ms")

    # 如果原输入不是列表，改回字典 (视需求而定，通常 param search 结果是列表)
    if not is_list:
        data = data[0]

    # 确保输出目录存在
    os.makedirs(os.path.dirname(INPUT_FILE), exist_ok=True)

    print(f"Writing to {INPUT_FILE}...")
    with open(INPUT_FILE, 'w') as f:
        json.dump(data, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()

