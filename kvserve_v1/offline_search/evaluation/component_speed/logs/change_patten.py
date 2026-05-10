import glob
import os
import sys

# ================= 配置区域 =================

# 定义要处理的文件模式列表
# 支持通配符，如 "*.log", "logs/**/*.log"
# 注意：路径是相对于脚本运行目录的
FILE_PATTERNS = [
    "./**/hadamard_speed/*.log",
    # "*.log",
]

# 定义替换规则列表
# 格式: ("旧字符串", "新字符串")
REPLACEMENTS = [
    ("Transform time", "Prefill time"),
    ("Transform throughput", "Prefill throughput"),
    ("Inverse time", "Decode time"),
    ("Inverse throughput", "Decode throughput"),
    # ("旧字符串", "新字符串"),
]

# ===========================================

def replace_in_file(file_path, replacements):
    """
    在文件中执行字符串替换。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = content
        modified = False
        for old_str, new_str in replacements:
            if old_str in new_content:
                new_content = new_content.replace(old_str, new_str)
                modified = True
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated: {file_path}")
        else:
            # print(f"No changes: {file_path}")
            pass
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    if not FILE_PATTERNS:
        print("未定义文件模式 (FILE_PATTERNS)。请在脚本配置区域添加。")
        return

    if not REPLACEMENTS:
        print("未定义替换规则 (REPLACEMENTS)。请在脚本配置区域添加。")
        return

    all_files = set()
    for pattern in FILE_PATTERNS:
        # 处理通配符
        # 如果是递归查找 (例如 **/*.log)，需要 recursive=True
        is_recursive = '**' in pattern
        expanded = glob.glob(pattern, recursive=is_recursive)
        
        # 如果 glob 没有匹配到且文件本身存在（可能没用通配符），直接添加
        if not expanded and os.path.exists(pattern):
             expanded = [pattern]
        
        for f in expanded:
            if os.path.isfile(f):
                all_files.add(os.path.abspath(f))
    
    if not all_files:
        print(f"根据配置的模式未找到匹配的文件: {FILE_PATTERNS}")
        print(f"当前工作目录: {os.getcwd()}")
        return

    print(f"正在处理 {len(all_files)} 个文件...")
    print(f"替换规则: {REPLACEMENTS}")
    
    for file_path in sorted(list(all_files)):
        replace_in_file(file_path, REPLACEMENTS)
    
    print("完成。")

if __name__ == "__main__":
    main()
