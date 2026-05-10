#!/bin/bash

# ##################################################
# ## 1. 变量设置区 (Variable Setup Area)
# ##################################################
#
# 在这里定义你的所有固定参数和可变参数
#
# --- 1.1 固定参数 (在所有任务中保持不变) ---
ACC_COMMAND_TO_RUN="python custom_accuracy.py" # 你要执行的主脚本
CR_COMMAND_TO_RUN="python ../compression_ratio/custom_cr.py"
CACHE_TYPE="custom"
COMP_CR="True" # 固定参数：是否记录压缩率 True/False


# --- 1.2 可变参数 (用于排列组合) ---
# 脚本将为这些数组的元素创建所有可能的组合
param_model_names=("Meta-Llama-3.1-8B-Instruct")

# --- 聚合参数组合 ---
# 这里将原有的 param_transform_types, param_heads_selections, param_kv_configs, param_axis_keys, param_axis_values, param_tasks 聚合在一起
# 格式: "transform_type|heads_selection|kv_config|axis_key|axis_value|task"
# 注意：必须使用 '|' 作为分隔符，因为后续脚本逻辑依赖此分隔符
param_combinations=(
    # "hadamard|0.7|10 6 8 4|2|1 3|gsm8k_cot_llama"
    # "hadamard|0.3|8 6 8 6|2|1 3|gsm8k_cot_llama"
    # "hadamard|0.1|10 6 8 3|2|1 3|gsm8k_cot_llama"
    # "hadamard|0.2|6 8 6 4|2|1 3|humaneval_instruct"
    # "hadamard|0.1|6 8 6 3|2|1 3|humaneval_instruct"
    # "hadamard|0.4|12 5 6 4|2|1 3|humaneval_instruct"
    # "hadamard|0.9|10 10 4 4|2|1 3|longbench_multi_news"
    # "hadamard|0.7|10 8 4 3|2|1 3|longbench_multi_news"
    # "hadamard|0.9|12 8 4 4|2|1 3|longbench_multi_news"
    # "hadamard|0.8|12 8 6 4|2|1 3|longbench_qasper"
    "hadamard|0.8|12 8 6 4|2|1 3|longbench_2wikimqa"
    "hadamard|0.8|12 8 6 4|2|1 3|longbench_hotpotqa"
    "hadamard|0.8|12 8 6 4|2|1 3|longbench_multifieldqa_en"
    "hadamard|0.9|10 10 4 4|2|1 3|longbench_gov_report"
    "hadamard|0.7|10 8 4 3|2|1 3|longbench_gov_report"
    "hadamard|0.9|12 8 4 4|2|1 3|longbench_gov_report"    
)


# --- 1.3 日志设置 (可选) ---
LOG_ROOT_DIRECTORY="/home/bingxing2/home/scx9kvs/mxy/Infer_Comm/evaluation/lm_eval/logs"
# 子目录将在执行区根据 cache_type 和 model_name 创建

# ##################################################
# ## 2. 设备检测区 (Device Detection Area)
# ##################################################
#
# 检测可用的 GPU，支持 CUDA_VISIBLE_DEVICES 环境变量
#
declare -a physical_gpu_ids # 存储物理 GPU ID 的数组 (例如: 0, 1, 4, 7)
declare -a queue_indices    # 存储队列索引的数组 (始终为: 0, 1, 2, ...)

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # 如果设置了 CUDA_VISIBLE_DEVICES，则只使用这些 GPU
    IFS=',' read -r -a physical_gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
    num_gpus=${#physical_gpu_ids[@]}
    echo "Using GPUs specified in CUDA_VISIBLE_DEVICES: ${physical_gpu_ids[*]} ($num_gpus GPUs)"
else
    # 否则，检测所有可用的 GPU
    num_gpus=$(nvidia-smi --list-gpus | wc -l)
    if [ $num_gpus -eq 0 ]; then
        echo "Error: No GPUs found."
        exit 1
    fi
    # 物理 GPU ID 就是 0, 1, ..., n-1
    physical_gpu_ids=($(seq 0 $((num_gpus-1)) ))
    echo "Found $num_gpus GPUs: ${physical_gpu_ids[*]}"
fi

# 队列索引始终是 0 到 n-1，用于管理 pids 数组
queue_indices=($(seq 0 $((num_gpus-1)) ))

# ##################################################
# ## 3. 任务排列组合区 (Task Combination Area)
# ##################################################
#
# 根据第1区设置的可变参数，生成所有任务组合
#
jobs=() # 用于存储所有任务组合的数组
DELIMITER="|" # 使用一个在参数中不会存在的分隔符

# 使用嵌套循环创建所有参数组合
for model_name in "${param_model_names[@]}"; do
    for combination in "${param_combinations[@]}"; do
        # 将一组参数作为一个单独的字符串添加到 'jobs' 数组中
        # combination 内部已经包含了 transform_type|heads_selection|kv_config|axis_key|axis_value|task (以 | 分隔)
        # 最终格式: model_name|transform_type|heads_selection|kv_config|axis_key|axis_value|task
        jobs+=("$model_name$DELIMITER$combination")
    done
done

total_jobs=${#jobs[@]}
echo "Total number of jobs to run: $total_jobs"

# ##################################################
# ## 4. 队列检测区 (Queue Detection/Management Area)
# ##################################################
#
# 定义管理 GPU 任务队列的函数和数据结构
#

# 使用关联数组 (Associative Array) 来存储每个 *队列索引* 对应的进程 PID
# pids[queue_index] = PID
declare -A pids
for q_idx in "${queue_indices[@]}"; do
    pids[$q_idx]=0 # 0 表示该队列索引 (GPU) 空闲
done

# 函数：查找一个空闲的 GPU 队列索引
# 这个函数会循环检测，直到找到一个空闲的 GPU，然后返回其 *队列索引* (0, 1, 2...)
find_free_gpu_queue_index() {
    while true; do
        for i in "${queue_indices[@]}"; do
            pid=${pids[$i]}
            
            # 一个 GPU 被认为是空闲的，如果:
            # 1. 它的 PID 记录为 0 (从未运行过任务)
            # 2. 它的 PID 对应的进程已经不存在 (ps -p $pid 失败)
            if [[ $pid -eq 0 ]] || ! ps -p $pid > /dev/null; then
                echo $i    # 返回空闲的 *队列索引*
                return    # 退出函数
            fi
        done
        # 如果没有找到空闲 GPU，等待1秒钟再重新检测
        sleep 1
    done
}

# ##################################################
# ## 5. 脚本执行区 (Script Execution Area)
# ##################################################
#
# 遍历所有任务，将它们分配到空闲的 GPU 上执行
#
job_number=0
for job_params in "${jobs[@]}"; do
    job_number=$((job_number + 1))
    
    # --- 5.1 解析任务参数 ---
    # !!! 重要: 'read' 后的变量名必须与第3区中 'jobs+=' 的参数顺序一致
    IFS="$DELIMITER" read -r current_model_name current_transform_type current_heads_selection current_kv_config current_axis_key current_axis_value current_task <<< "$job_params"
    
    # 从 kv_config 字符串中解析出四个单独的值
    read -r high_k high_v low_k low_v <<< "$current_kv_config"

    # --- 5.2 查找空闲 GPU ---
    # 'find_free_gpu_queue_index' 函数会阻塞，直到找到一个空闲的 *队列索引*
    gpu_queue_index=$(find_free_gpu_queue_index)
    
    # --- 5.3 映射到物理 GPU ID ---
    # 根据队列索引，从 'physical_gpu_ids' 数组中获取真实的 GPU ID
    physical_gpu_id=${physical_gpu_ids[$gpu_queue_index]}
    
    # --- 5.4 准备日志文件 (可选) ---
    # 创建日志子目录
    LOG_DIRECTORY="$LOG_ROOT_DIRECTORY/$CACHE_TYPE/$current_model_name"
    mkdir -p "$LOG_DIRECTORY"
    
    # 为了文件名合法，将参数中的空格替换为下划线
    ak_str=$(echo "$current_axis_key" | tr ' ' '_')
    av_str=$(echo "$current_axis_value" | tr ' ' '_')
    task_str=$(echo "$current_task" | tr ' ' '_')
    
    # 根据所有可变参数创建唯一的日志文件名
    log_file="$LOG_DIRECTORY/${current_transform_type}-hs${current_heads_selection}-kv_${high_k}_${high_v}_${low_k}_${low_v}-ak_${ak_str}-av_${av_str}-t_${task_str}.log"
    
    # --- 5.5 打印任务信息 ---
    echo "---------------------------------"
    echo "Starting job $job_number/$total_jobs on Physical GPU $physical_gpu_id (Queue Index: $gpu_queue_index)"
    echo "  Model: $current_model_name, Task: $current_task"
    echo "  TF: $current_transform_type"
    echo "  HS: $current_heads_selection, KV Config: high_k=$high_k, high_v=$high_v, low_k=$low_k, low_v=$low_v"
    echo "  Axis Key: $current_axis_key, Axis Value: $current_axis_value"
    echo "  Log file: $log_file"
    
    # --- 5.6 执行脚本 ---
    # ( ... ) & 将命令放入子 shell 中后台执行
    (
        
        # 设置当前脚本命令可见cuda
        export CUDA_VISIBLE_DEVICES=$physical_gpu_id
        
        $ACC_COMMAND_TO_RUN \
            --cache_type "$CACHE_TYPE" \
            --model_name "$current_model_name" \
            --transform_type "$current_transform_type" \
            --heads_selection "$current_heads_selection" \
            --high_key_max_value "$high_k" \
            --high_value_max_value "$high_v" \
            --low_key_max_value "$low_k" \
            --low_value_max_value "$low_v" \
            --axis_key $current_axis_key \
            --axis_value $current_axis_value \
            --tasks $current_task \
            --comp_cr "$COMP_CR" \
            > "$log_file" 2>&1

        # $CR_COMMAND_TO_RUN \
        #     --model_name "$current_model_name" \
        #     --transform_type "$current_transform_type" \
        #     --heads_selection "$current_heads_selection" \
        #     --high_key_max_value "$high_k" \
        #     --high_value_max_value "$high_v" \
        #     --low_key_max_value "$low_k" \
        #     --low_value_max_value "$low_v" \
        #     --axis_key $current_axis_key \
        #     --axis_value $current_axis_value \
        #     >> "$log_file" 2>&1
    ) &
    
    # --- 5.7 记录新任务的 PID ---
    # $! 是 Bash 中最后一个后台进程的 PID
    # 使用 *队列索引* 作为 pids 数组的键
    pids[$gpu_queue_index]=$!
done

# ##################################################
# ## 6. 最终等待区 (Final Wait Area)
# ##################################################
#
# 所有任务都已分配，现在等待所有正在运行的后台任务完成
#
echo "---------------------------------"
echo "All $total_jobs jobs have been assigned. Waiting for running jobs to complete..."

while true; do
    all_done=1 # 假设所有任务都完成了
    
    # 检查所有 *队列索引* 上的 PID
    for i in "${queue_indices[@]}"; do
        pid=${pids[$i]}
        # 如果 PID 不为 0 且该进程仍在运行
        if [[ $pid -ne 0 ]] && ps -p $pid > /dev/null; then
            all_done=0 # 标记为 "未完成"
            break      # 无需检查其他 GPU，直接跳出内层循环
        fi
    done

    # 如果 all_done 标记仍然为 1，说明所有进程都已结束
    if [[ $all_done -eq 1 ]]; then
        break # 跳出 "while true" 循环
    fi
    
    # 等待 5 秒后再次检查
    sleep 5
done

echo "All jobs completed. Check the '$LOG_ROOT_DIRECTORY' directory for logs."