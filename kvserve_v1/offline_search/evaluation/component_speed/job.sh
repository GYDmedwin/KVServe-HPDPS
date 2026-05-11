#!/bin/bash

# ##################################################
# ## 📜 1. 变量设置区 (Variable Setup Area)
# ##################################################

# --- 1.1 固定参数 ---
# 脚本名称（不含后缀），例如: quantizer_speed 或 cachegen_speed
SCRIPT_NAME="quantizer_speed" 

# 执行命令前缀
COMMAND_TO_RUN="python ./${SCRIPT_NAME}.py"

# 固定运行参数
WARMUP_ITER=100
BENCHMARK_ITER=200

# --- 1.2 可变参数 ---
# 排列组合参数，脚本将为这些数组的元素创建所有可能的组合
param_array_1_input_lengths=($(seq 1024 1024 32768)) # 1024 to 32768, step 1024
param_array_2_model_names=("Qwen2.5-32B-Instruct")

# --- 1.3 日志设置 ---
# 基础日志目录
BASE_LOG_DIR="/root/workspace/Infer_Comm/evaluation/component_speed/logs"

# ##################################################
# ## 🧬 2. 任务排列组合区 (Task Combination Area)
# ##################################################

jobs=() # 用于存储所有任务组合的数组

# 使用嵌套循环创建所有参数组合
for length in "${param_array_1_input_lengths[@]}"; do
    for model in "${param_array_2_model_names[@]}"; do
        # 格式: "input_length model_name"
        jobs+=("$length $model")
    done
done

total_jobs=${#jobs[@]}
echo "Total number of jobs to run: $total_jobs"

# ##################################################
# ## 🚀 3. 脚本执行区 (Script Execution Area)
# ##################################################

# 顺序执行任务
job_number=0
for job_params in "${jobs[@]}"; do
    job_number=$((job_number + 1))
    
    # --- 3.1 解析任务参数 ---
    read -r current_length current_model <<< "$job_params"
    
    # --- 3.2 准备日志目录 ---
    # 路径格式: logs/model_name/script_name
    LOG_DIR="${BASE_LOG_DIR}/${current_model}/${SCRIPT_NAME}"
    mkdir -p "$LOG_DIR" # 确保目录存在
    
    # 日志文件名
    log_file="${LOG_DIR}/len_${current_length}.log"
    
    # --- 3.3 打印任务信息 ---
    echo "---------------------------------"
    echo "Starting job $job_number/$total_jobs"
    echo "  Script: $SCRIPT_NAME"
    echo "  Params: Length=$current_length, Model=$current_model"
    echo "  Log file: $log_file"
    
    # --- 3.4 执行脚本 ---
    # 顺序执行，不需要后台运行 (&)，会自动等待上一个任务结束
    # 所有 GPU 可见，不设置 CUDA_VISIBLE_DEVICES
    
    $COMMAND_TO_RUN \
        --warmup_iter "$WARMUP_ITER" \
        --benchmark_iter "$BENCHMARK_ITER" \
        --input_length "$current_length" \
        --model_name "$current_model" \
        > "$log_file" 2>&1
        
    echo "Job $job_number completed."
done

echo "---------------------------------"
echo "All jobs completed. Check the '$BASE_LOG_DIR' directory for logs."
