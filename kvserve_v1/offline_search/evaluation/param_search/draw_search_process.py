import re
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.9,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

LOG_FILE_NAME = 'qwen_hadamard_search_tolerance_3_qasper.log'
INPUT_LOG_PATH = os.path.join(os.path.dirname(__file__), 'logs_new', LOG_FILE_NAME)
OUTPUT_IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'logs_new', f'{LOG_FILE_NAME.replace(".log", "")}_process.pdf')

# 自定义 Y 轴刻度列表 (左图)
# 刻度之间将以均匀间隔显示
CUSTOM_LEFT_Y_TICKS = [5, 6, 7, 8, 9, 12, 15, 20]

# 正则匹配模式
PATTERN_ITERATION_REMAINING = re.compile(r"--- BO Iteration (\d+)/\d+ \| Remaining: (\d+)/(\d+) ---")
PATTERN_PROPOSING = re.compile(r"Proposing Config \(ID:\d+\): CR=([\d\.]+)")
PATTERN_INFEASIBLE = re.compile(r"❌ INFEASIBLE")
PATTERN_FEASIBLE = re.compile(r"✅ Configuration is FEASIBLE")

# 绘图颜色和样式
COLOR_FEASIBLE = '#4daf4a'   # Green
COLOR_INFEASIBLE = '#e41a1c' # Red
COLOR_LINE = '#377eb8'       # Blue
MARKER_FEASIBLE = 'o'
MARKER_INFEASIBLE = 'x'
LINE_STYLE = '-'

FIGURE_SIZE = (12, 5)
FONT_SIZE = 12
LINE_WIDTH = 2

def parse_log(log_path):
    """
    解析 Log 文件，返回两个列表：
    1. iterations_data: [{'iter': 1, 'cr': 20.9, 'is_feasible': False}, ...]
    2. remaining_data: [{'iter': 1, 'remaining': 3966}, ...]
    3. best_cr_final
    """
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return [], [], None

    iterations_data = []
    remaining_data = []
    
    current_iter = None
    current_remaining = None
    current_cr = None
    
    best_cr_final = 0.0

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 1. 匹配 Iteration 和 Remaining
            match_iter = PATTERN_ITERATION_REMAINING.search(line)
            if match_iter:
                current_iter = int(match_iter.group(1))
                current_remaining = int(match_iter.group(2))
                remaining_data.append({'iter': current_iter, 'remaining': current_remaining})
                continue

            # 2. 匹配 Proposing CR
            match_prop = PATTERN_PROPOSING.search(line)
            if match_prop:
                current_cr = float(match_prop.group(1))
                continue

            # 3. 匹配 Feasible / Infeasible
            if current_iter is not None and current_cr is not None:
                if PATTERN_FEASIBLE.search(line):
                    iterations_data.append({
                        'iter': current_iter,
                        'cr': current_cr,
                        'is_feasible': True
                    })
                    best_cr_final = max(best_cr_final, current_cr)
                    current_cr = None 
                elif PATTERN_INFEASIBLE.search(line):
                    iterations_data.append({
                        'iter': current_iter,
                        'cr': current_cr,
                        'is_feasible': False
                    })
                    current_cr = None

    return iterations_data, remaining_data, best_cr_final

def transform_to_equidistant(values, ticks):
    """
    将原始值映射到基于 ticks 的等间距坐标上。
    ticks 中的每个间隔将被映射为 1 的距离。
    例如 ticks=[5, 10, 20]
    5 -> 0
    10 -> 1
    20 -> 2
    7.5 (5和10中间) -> 0.5
    """
    if not ticks:
        return values
    
    transformed = []
    sorted_ticks = sorted(ticks)
    
    for v in values:
        # 找到 v 所在的区间
        if v <= sorted_ticks[0]:
            # 低于最小值，线性外推 (假设斜率与第一个区间相同)
            # slope = 1 / (sorted_ticks[1] - sorted_ticks[0])
            # mapped = 0 - (sorted_ticks[0] - v) * slope
            # 简单处理：直接截断或映射到0
            transformed.append(0) 
        elif v >= sorted_ticks[-1]:
            # 高于最大值
            transformed.append(len(sorted_ticks) - 1)
        else:
            # 在区间内
            for i in range(len(sorted_ticks) - 1):
                if sorted_ticks[i] <= v <= sorted_ticks[i+1]:
                    # 线性插值
                    ratio = (v - sorted_ticks[i]) / (sorted_ticks[i+1] - sorted_ticks[i])
                    transformed.append(i + ratio)
                    break
    return transformed

def main():
    print(f"Reading log from: {INPUT_LOG_PATH}")
    iter_data, rem_data, best_cr = parse_log(INPUT_LOG_PATH)
    
    if not iter_data:
        print("No iteration data found.")
        return

    # 构建 Remaining Map 以便查找
    rem_map = {d['iter']: d['remaining'] for d in rem_data}

    # 准备绘图数据 (左图: CR)
    iters = [d['iter'] for d in iter_data]
    raw_crs = [d['cr'] for d in iter_data]
    
    # 应用坐标变换
    plot_crs = transform_to_equidistant(raw_crs, CUSTOM_LEFT_Y_TICKS)
    
    # 拆分 Feasible/Infeasible 并变换
    feasible_iters = [d['iter'] for d in iter_data if d['is_feasible']]
    raw_feasible_crs = [d['cr'] for d in iter_data if d['is_feasible']]
    plot_feasible_crs = transform_to_equidistant(raw_feasible_crs, CUSTOM_LEFT_Y_TICKS)
    
    infeasible_iters = [d['iter'] for d in iter_data if not d['is_feasible']]
    raw_infeasible_crs = [d['cr'] for d in iter_data if not d['is_feasible']]
    plot_infeasible_crs = transform_to_equidistant(raw_infeasible_crs, CUSTOM_LEFT_Y_TICKS)
    
    # 最佳 CR 变换
    plot_best_cr = transform_to_equidistant([best_cr], CUSTOM_LEFT_Y_TICKS)[0] if best_cr > 0 else -1

    # 准备绘图数据 (右图: Remaining)
    rem_iters = [d['iter'] for d in rem_data]
    rem_counts = [d['remaining'] for d in rem_data]
    
    feasible_rem_iters = []
    feasible_rem_counts = []
    infeasible_rem_iters = []
    infeasible_rem_counts = []
    
    for d in iter_data:
        it = d['iter']
        if it in rem_map:
            cnt = rem_map[it]
            if d['is_feasible']:
                feasible_rem_iters.append(it)
                feasible_rem_counts.append(cnt)
            else:
                infeasible_rem_iters.append(it)
                infeasible_rem_counts.append(cnt)

    # 开始绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    
    # --- 左图: Iteration vs CR (Transformed Y) ---
    ax1.plot(iters, plot_crs, color=COLOR_LINE, linestyle=LINE_STYLE, alpha=0.9, zorder=1, linewidth=LINE_WIDTH)
    
    if feasible_iters:
        ax1.scatter(feasible_iters, plot_feasible_crs, color=COLOR_FEASIBLE, marker=MARKER_FEASIBLE, zorder=2)
    if infeasible_iters:
        ax1.scatter(infeasible_iters, plot_infeasible_crs, color=COLOR_INFEASIBLE, marker=MARKER_INFEASIBLE, zorder=2)
    
    if best_cr > 0:
        ax1.axhline(y=plot_best_cr, color=COLOR_FEASIBLE, linestyle='--', alpha=0.9, zorder=1, linewidth=LINE_WIDTH)

    # 设置自定义 Y 轴刻度
    if CUSTOM_LEFT_Y_TICKS:
        ax1.set_yticks(range(len(CUSTOM_LEFT_Y_TICKS)))
        ax1.set_yticklabels(CUSTOM_LEFT_Y_TICKS)
        ax1.set_ylim(-0.5, len(CUSTOM_LEFT_Y_TICKS) - 0.5)

    ax1.set_xlabel('Iteration', fontweight='bold', fontsize=FONT_SIZE)
    ax1.set_ylabel('Compression Ratio (CR)', fontweight='bold', fontsize=FONT_SIZE)
    ax1.set_title('Exploration Process', fontweight='bold', fontsize=FONT_SIZE)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- 右图: Iteration vs Remaining Space ---
    ax2.plot(rem_iters, rem_counts, color=COLOR_LINE, linestyle=LINE_STYLE, alpha=0.9, zorder=1, linewidth=LINE_WIDTH)
    
    # 绘制 Feasible Points (Remaining)
    if feasible_rem_iters:
        ax2.scatter(feasible_rem_iters, feasible_rem_counts, color=COLOR_FEASIBLE, marker=MARKER_FEASIBLE, zorder=2)
    
    # 绘制 Infeasible Points (Remaining)
    if infeasible_rem_iters:
        ax2.scatter(infeasible_rem_iters, infeasible_rem_counts, color=COLOR_INFEASIBLE, marker=MARKER_INFEASIBLE, zorder=2)
    
    ax2.set_xlabel('Iteration', fontweight='bold', fontsize=FONT_SIZE)
    ax2.set_ylabel('Remaining Search Space', fontweight='bold', fontsize=FONT_SIZE)
    ax2.set_title('Pruning Process', fontweight='bold', fontsize=FONT_SIZE)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # --- 添加阶段分隔线和标记 (1-24: 探索, 25+: 剪枝) ---
    max_x = max(iters) if iters else 0
    split_x = 24.5
    min_x = 0.5  # Start boundary
    
    if max_x > split_x:
        # 统一标注的垂直位置 (Axes 坐标, 0-1)
        y_pos_arrow = 0.87
        
        for ax in (ax1, ax2):
            # 绘制竖虚线 (Phase 1 Start, Split, Phase 2 End)
            ax.axvline(x=min_x, color='#666666', linestyle='--', linewidth=1.5, alpha=0.8)
            ax.axvline(x=split_x, color='#666666', linestyle='--', linewidth=1.5, alpha=0.8)
            ax.axvline(x=max_x + 0.5, color='#666666', linestyle='--', linewidth=1.5, alpha=0.8)
            
            # --- 阶段 1 标记 ---
            mid_1 = (min_x + split_x) / 2
            
            # 1. 绘制文字 (zorder=10 确保在上层, alpha=1.0 不透明背景遮挡箭头)
            ax.text(mid_1, y_pos_arrow, '①', transform=ax.get_xaxis_transform(),
                    ha='center', va='center', fontsize=16, fontweight='bold', color='#333333',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=1.0, pad=4), zorder=10)
            
            # 2. 绘制双向箭头 (与文字同一高度)
            ax.annotate('', xy=(min_x, y_pos_arrow), xytext=(split_x, y_pos_arrow),
                        xycoords=ax.get_xaxis_transform(), textcoords=ax.get_xaxis_transform(),
                        arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.5), zorder=9)

            # --- 阶段 2 标记 ---
            end_x = max_x + 0.5
            mid_2 = (split_x + end_x) / 2
            
            # 1. 绘制文字
            ax.text(mid_2, y_pos_arrow, '②', transform=ax.get_xaxis_transform(),
                    ha='center', va='center', fontsize=16, fontweight='bold', color='#333333',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=1.0, pad=4), zorder=10)
            
            # 2. 绘制双向箭头
            ax.annotate('', xy=(split_x, y_pos_arrow), xytext=(end_x, y_pos_arrow),
                        xycoords=ax.get_xaxis_transform(), textcoords=ax.get_xaxis_transform(),
                        arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.5), zorder=9)

    # --- Legend ---
    # Create dummy handles for legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLOR_LINE, lw=LINE_WIDTH, alpha=0.9, label='Process Trace'),
        Line2D([0], [0], color=COLOR_FEASIBLE, lw=LINE_WIDTH, alpha=0.9, linestyle='--', label='Best CR Found'),        
        Line2D([0], [0], marker=MARKER_FEASIBLE, color='w', label='Feasible',
               markerfacecolor=COLOR_FEASIBLE, markersize=8),
        Line2D([0], [0], marker=MARKER_INFEASIBLE, color='w', label='Infeasible',
               markerfacecolor=COLOR_INFEASIBLE, markeredgecolor=COLOR_INFEASIBLE, markersize=6),
        # Add phases description
        Line2D([0], [0], marker='o', color='w', label='① Exploration Phase',
               markerfacecolor='none', markeredgecolor='none', markersize=0),
        Line2D([0], [0], marker='o', color='w', label='② Exploitation Phase',
               markerfacecolor='none', markeredgecolor='none', markersize=0),
    ]
    
    # Place legend centered above the whole figure
    # 使用 frameon=True 加上边框，edgecolor 设置边框颜色
    # Adjusted ncol to accommodate new items, or let it flow
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
               ncol=3, frameon=True, edgecolor='black', fancybox=False, shadow=False, prop={'weight': 'bold'})

    # Adjust layout
    plt.tight_layout()
    # Reserve space at top for legend
    plt.subplots_adjust(top=0.82)

    # Save
    plt.savefig(OUTPUT_IMAGE_PATH)
    print(f"Plot saved to: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    main()
