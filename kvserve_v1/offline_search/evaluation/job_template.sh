#!/bin/bash

# ##################################################
# ## 📜 1. 变量设置区 (Variable Setup Area)
# ##################################################
#
# 在这里定义你的所有固定参数和可变参数
#
# --- 1.1 固定参数 (在所有任务中保持不变) ---
COMMAND_TO_RUN="python my_main_script.py" # 你要执行的主脚本
FIXED_ARG_1="--model /path/to/model"
FIXED_ARG_2="--dataset my_dataset"

# --- 1.2 可变参数 (用于排列组合) ---
# 脚本将为这些数组的元素创建所有可能的组合
param_array_1_learning_rates=(0.01 0.001)
param_array_2_batch_sizes=(32 64)
param_array_3_modes=("train" "eval")

# --- 1.3 日志设置 (可选) ---
LOG_DIRECTORY="run_logs"
mkdir -p $LOG_DIRECTORY # 确保日志目录存在

# ##################################################
# ## 💻 2. 设备检测区 (Device Detection Area)
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
# ## 🧬 3. 任务排列组合区 (Task Combination Area)
# ##################################################
#
# 根据第1区设置的可变参数，生成所有任务组合
#
jobs=() # 用于存储所有任务组合的数组

# 使用嵌套循环创建所有参数组合
for lr in "${param_array_1_learning_rates[@]}"; do
    for bs in "${param_array_2_batch_sizes[@]}"; do
        for mode in "${param_array_3_modes[@]}"; do
            # 将一组参数作为一个单独的字符串添加到 'jobs' 数组中
            # 格式: "参数1 参数2 参数3 ..."
            jobs+=("$lr $bs $mode")
        done
    done
done

total_jobs=${#jobs[@]}
echo "Total number of jobs to run: $total_jobs"

# ##################################################
# ## 🚦 4. 队列检测区 (Queue Detection/Management Area)
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
# ## 🚀 5. 脚本执行区 (Script Execution Area)
# ##################################################
#
# 遍历所有任务，将它们分配到空闲的 GPU 上执行
#
job_number=0
for job_params in "${jobs[@]}"; do
    job_number=$((job_number + 1))
    
    # --- 5.1 解析任务参数 ---
    # !!! 重要: 'read' 后的变量名必须与第3区中 'jobs+=' 的参数顺序一致
    read -r current_lr current_bs current_mode <<< "$job_params"
    
    # --- 5.2 查找空闲 GPU ---
    # 'find_free_gpu_queue_index' 函数会阻塞，直到找到一个空闲的 *队列索引*
    gpu_queue_index=$(find_free_gpu_queue_index)
    
    # --- 5.3 映射到物理 GPU ID ---
    # 根据队列索引，从 'physical_gpu_ids' 数组中获取真实的 GPU ID
    physical_gpu_id=${physical_gpu_ids[$gpu_queue_index]}
    
    # --- 5.4 准备日志文件 (可选) ---
    # 在日志名中使用物理 GPU ID，更易读
    log_file="$LOG_DIRECTORY/job_${job_number}_gpu${physical_gpu_id}_${current_lr}_${current_bs}_${current_mode}.log"
    
    # --- 5.5 打印任务信息 ---
    echo "---------------------------------"
    echo "Starting job $job_number/$total_jobs on Physical GPU $physical_gpu_id (Queue Index: $gpu_queue_index)"
    echo "  Params: LR=$current_lr, BS=$current_bs, Mode=$current_mode"
    echo "  Log file: $log_file"
    
    # --- 5.6 执行脚本 ---
    # ( ... ) & 将命令放入子 shell 中后台执行
    (
        
        # 设置当前脚本命令可见cuda
        CUDA_VISIBLE_DEVICES=$physical_gpu_id
        
        $COMMAND_TO_RUN \
            $FIXED_ARG_1 \
            $FIXED_ARG_2 \
            --learning_rate "$current_lr" \
            --batch_size "$current_bs" \
            --mode "$current_mode" \
            > "$log_file" 2>&1
    ) &
    
    # --- 5.7 记录新任务的 PID ---
    # $! 是 Bash 中最后一个后台进程的 PID
    # 使用 *队列索引* 作为 pids 数组的键
    pids[$gpu_queue_index]=$!
done

# ##################################################
# ## ⏳ 6. 最终等待区 (Final Wait Area)
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

echo "All jobs completed. Check the '$LOG_DIRECTORY' directory for logs."