import json
import os

# ==========================================
# CONFIGURATION (配置项)
# ==========================================

# 1. 要合并的 JSON 文件列表 (文件名即可，假设在同一目录下)
MODEL_NAME = 'Meta-Llama-3.1-8B-Instruct'
JSON_FILES = [
    # 'none_tolerance_3_qasper.json',
    # 'none_tolerance_5_qasper.json',
    # 'hadamard_tolerance_3_qasper.json',
    # 'hadamard_tolerance_5_qasper.json',
    'cachegen_merged_qasper.json',
    'cachegen_merged_qasper2.json',
    'hadamard_merged_qasper.json',
    'hadamard_merged_qasper2.json',
    'none_merged_qasper.json',
    'none_merged_qasper2.json',
]

# 2. 输出文件名
OUTPUT_FILE = '3d_merged_qasper.json'

# ==========================================
# FUNCTIONS
# ==========================================

def get_unique_key(item):
    """
    生成用于判断唯一性的键。
    排除 config_id，因为需要重新排序。
    将列表类型转换为 tuple 以便哈希。
    """
    key_dict = item.copy()
    if 'config_id' in key_dict:
        del key_dict['config_id']
    
    # 为了保证字典顺序一致导致的 hash 一致，可以排序 items
    # 但由于包含 list (如 axis_key)，需要特殊处理
    sorted_items = []
    for k, v in sorted(key_dict.items()):
        if isinstance(v, list):
            v = tuple(v)
        sorted_items.append((k, v))
    
    return tuple(sorted_items)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    merged_data = []
    seen_configs = set()
    
    print(f"Start merging {len(JSON_FILES)} files...")

    for json_file in JSON_FILES:
        file_path = os.path.join(current_dir, f"{MODEL_NAME}", json_file)
        if not os.path.exists(file_path):
            print(f"[Warning] File not found: {file_path}")
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if not isinstance(data, list):
                    print(f"[Warning] File content is not a list: {json_file}")
                    continue
                
                print(f"Processing {json_file}: found {len(data)} items.")
                
                for item in data:
                    # 生成唯一键
                    unique_key = get_unique_key(item)
                    
                    if unique_key not in seen_configs:
                        seen_configs.add(unique_key)
                        # 先添加进去，最后统一重新分配 config_id
                        merged_data.append(item)
                    # else:
                    #     print(f"Duplicate found in {json_file}, skipping.")
                        
        except Exception as e:
            print(f"[Error] Failed to process {json_file}: {e}")

    # 重新分配 config_id
    print(f"Total unique items: {len(merged_data)}")
    print("Re-indexing config_id...")
    
    for idx, item in enumerate(merged_data):
        item['config_id'] = idx

    # 写入结果文件
    output_path = os.path.join(current_dir, f"{MODEL_NAME}", OUTPUT_FILE)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved merged data to: {output_path}")
    except Exception as e:
        print(f"[Error] Failed to write output file: {e}")

if __name__ == "__main__":
    main()

