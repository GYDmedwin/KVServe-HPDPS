import json
import matplotlib.pyplot as plt
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# ==========================================
# CONFIGURATION (配置项)
# ==========================================
plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.9,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# 输入数据文件路径
MODEL_NAME = 'Meta-Llama-3.1-8B-Instruct'
# 假设 JSON 文件在 MODEL_NAME 目录下
INPUT_JSON_PATH = os.path.join(os.path.dirname(__file__), f"{MODEL_NAME}", "3d_merged_qasper.json")

# 输出图片保存路径
OUTPUT_IMAGE_PATH = os.path.join(os.path.dirname(__file__), f"{MODEL_NAME}", "3d_pareto_front_qasper.pdf")

# JSON 中的字段名称
KEY_ACCURACY = 'accuracy'
KEY_CR = 'cr'
KEY_LATENCY = 'latency'
KEY_ID = 'config_id'

# 绘图设置
X_LABEL = 'Compression Ratio'
Y_LABEL = 'Relative Accuracy (%)'
Z_LABEL = 'Latency (ms)'
PLOT_TITLE = f'{MODEL_NAME} 3D Pareto Frontier'
FIGURE_SIZE = (12, 10)

# Latency 聚类阈值 (如果两个 latency 差值小于此值，视为同一层)
LATENCY_CLUSTERING_THRESHOLD = 0.3 

# 颜色设置
COLOR_PARETO_LINE = '#e41a1c'
COLOR_SURFACE_BASE = '#377eb8'

# ==========================================
# FUNCTIONS
# ==========================================

def get_pareto_frontier_2d(points):
    """
    计算 2D Pareto Frontier (Accuracy vs CR).
    假设 X (Accuracy) 和 Y (CR) 都是越大越好。
    """
    if not points:
        return []
        
    # 1. 根据 X 轴 (Accuracy) 从大到小排序
    # 如果 X 相同，则按 Y 从大到小排
    sorted_points = sorted(points, key=lambda p: (p['x'], p['y']), reverse=True)
    
    pareto_front = []
    current_max_y = -float('inf')
    
    for p in sorted_points:
        # 如果当前点的 Y 值大于目前遇到的最大 Y 值，说明该点在 Pareto 前沿上
        if p['y'] > current_max_y:
            pareto_front.append(p)
            current_max_y = p['y']
    
    # 为了绘图连线顺滑，最后按 X 从小到大排序返回
    return sorted(pareto_front, key=lambda p: p['x'])

def group_latencies(points, threshold):
    """
    将点根据 Latency 进行聚类/分组
    返回: 
    - levels: list of dict, 每个元素包含该 level 的 'z_index', 'avg_latency', 'points'
    """
    # 提取所有 latency 并排序
    latencies = sorted([p['z'] for p in points])
    
    if not latencies:
        return []

    # 简单的聚类：如果相邻差 > threshold，则切分
    groups = []
    if latencies:
        current_group = [latencies[0]]
        for l in latencies[1:]:
            if l - current_group[-1] > threshold:
                # 结束当前组，开启新组 (但在排序数组中，l 肯定大于 current_group[-1]，这里要看能否合并)
                # 实际上应该跟当前组的平均值或第一个值比，或者只要跟上一个值比断层
                # 既然是排序的，直接看 diff
                groups.append(current_group)
                current_group = [l]
            else:
                # 也可以选择跟 group mean 比较，这里简单跟上一个元素比会有链式效应，
                # 但考虑到物理含义（几个等级），通常是有明显间隙的。
                # 更加稳健的方式：如果 (l - current_group[0]) > threshold，则开新组
                if (l - current_group[0]) > threshold:
                     groups.append(current_group)
                     current_group = [l]
                else:
                    current_group.append(l)
        if current_group:
            groups.append(current_group)
    
    # 计算每个组的范围和平均值
    levels = []
    for i, g in enumerate(groups):
        min_l = min(g)
        max_l = max(g)
        avg_l = sum(g) / len(g)
        
        # 找到属于该组的点
        # 为了避免边界问题，我们使用 min_l 和 max_l 稍微放宽一点，或者直接重新遍历分配
        # 这里直接分配
        level_points = [p for p in points if min_l <= p['z'] <= max_l]
        
        levels.append({
            'index': i,
            'avg_latency': avg_l,
            'min_latency': min_l,
            'max_latency': max_l,
            'points': level_points,
            'label': f"{avg_l:.2f}"
        })
    
    return levels

def main():
    # 1. 读取数据
    if not os.path.exists(INPUT_JSON_PATH):
        print(f"Error: File not found at {INPUT_JSON_PATH}")
        return

    print(f"Reading data from {INPUT_JSON_PATH}...")
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. 提取绘图所需数据
    points = []
    for entry in data:
        if KEY_ACCURACY in entry and KEY_CR in entry and KEY_LATENCY in entry:
            points.append({
                'x': entry[KEY_CR],
                'y': entry[KEY_ACCURACY],
                'z': entry[KEY_LATENCY],
                'id': entry.get(KEY_ID, 'N/A')
            })
    
    if not points:
        print("No valid data points found.")
        return

    # 3. 分组 Latency
    # 使用用户建议的阈值（比如 1 或 2，根据数据范围调整）
    # 之前看到数据有 32.2 和 30.7，差值约 1.5。如果要把它们分开，阈值要小。
    # 如果要合并，阈值要大。用户说“json数据中的latency只有大概六个数值”。
    # 我们可以先打印一下所有 latency 的分布情况
    all_z = sorted([p['z'] for p in points])
    print(f"Latency range: {min(all_z):.2f} - {max(all_z):.2f}")
    
    # 这里使用 2.0 作为初始尝试，如果分出来太多组，可能需要调大
    levels = group_latencies(points, LATENCY_CLUSTERING_THRESHOLD)
    print(f"Identified {len(levels)} latency levels:")
    for l in levels:
        print(f"  Level {l['index']}: Avg {l['avg_latency']:.2f} (Range {l['min_latency']:.2f}-{l['max_latency']:.2f}), Points: {len(l['points'])}")

    # 4. 准备 3D 绘图
    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_subplot(111, projection='3d')
    
    # 存储每一层的 Pareto 数据用于连接
    # structure: list of dict {'x': [], 'y': [], 'z_index': int, 'points': []}
    pareto_levels = []

    # Filter valid levels first to ensure sequential indexing for plotting
    valid_levels_raw = []
    for level in levels:
        bin_points = level['points']
        if not bin_points:
            continue
        pareto_2d = get_pareto_frontier_2d(bin_points)
        if not pareto_2d:
            continue
        valid_levels_raw.append((level, pareto_2d))

    for idx, (level, pareto_2d) in enumerate(valid_levels_raw):
        # 这里使用 idx (0, 1, 2...) 作为 Z 轴绘图坐标，实现等距分布
        z_plot_val = idx
        
        xs = [p['x'] for p in pareto_2d]
        ys = [p['y'] for p in pareto_2d]
        zs = [z_plot_val] * len(pareto_2d) 
        
        # 绘制该层的 Pareto 线 (点)
        ax.plot(xs, ys, zs, c=COLOR_PARETO_LINE, marker='o', markersize=4, alpha=0.9, zorder=10)
        
        # 标注 Pareto 点 ID
        for p_data, x, y, z in zip(pareto_2d, xs, ys, zs):
            ax.text(x, y, z, str(p_data['id']), fontsize=6, zorder=20, color='black')

        pareto_levels.append({
            'x': xs,
            'y': ys,
            'z': zs, 
            'z_val': z_plot_val, # scalar
            'raw_latency': level['avg_latency']
        })

    # 5. 连接相邻层 (相邻两个桶的 pareto 连接成一面)
    # pareto_levels 已经按 index 排序（因为 levels 是按 latency 排序生成的）
    
    # 使用 colormap 区分不同层级的面，或者统一颜色
    # Fix deprecation warning: cm.get_cmap -> plt.get_cmap or matplotlib.colormaps
    cmap = plt.get_cmap('viridis', len(pareto_levels))

    for i in range(len(pareto_levels) - 1):
        level_curr = pareto_levels[i]
        level_next = pareto_levels[i+1]
        
        # 准备合并的点集用于生成曲面
        # 注意：这里只连接 level_curr 和 level_next
        
        # 当前层点
        curr_x = level_curr['x']
        curr_y = level_curr['y']
        curr_z = level_curr['z']
        
        # 下一层点
        next_x = level_next['x']
        next_y = level_next['y']
        next_z = level_next['z']
        
        # 合并
        surf_x = curr_x + next_x
        surf_y = curr_y + next_y
        surf_z = curr_z + next_z
        
        # 绘制曲面
        # alpha 透明度设置低一点以便能看到后面的层
        # 使用 trisurf
        # Fix: Ensure at least 3 points for triangulation
        if len(surf_x) >= 3:
            try:
                ax.plot_trisurf(surf_x, surf_y, surf_z, color=cmap(i), alpha=0.4, shade=True)
            except Exception as e:
                print(f"Error plotting surface between level {i} and {i+1}: {e}")
        else:
            # 如果点不够，画线连接对应点或最近点
            # 简单起见，这里可以画线连接两个平面的中心或者什么都不做
            print(f"Skipping surface between level {i} and {i+1} due to insufficient points ({len(surf_x)}).")

    # 6. 设置坐标轴
    ax.set_xlabel(X_LABEL, fontweight='bold', labelpad=10)
    ax.set_ylabel(Y_LABEL, fontweight='bold', labelpad=10)
    ax.set_zlabel(Z_LABEL, fontweight='bold', labelpad=10)
    ax.set_title(PLOT_TITLE, fontweight='bold')
    
    # 自定义 Z 轴刻度
    # 获取所有存在的 level index 和 对应的 label
    z_indices = [pl['z_val'] for pl in pareto_levels]
    z_labels = [f"{pl['raw_latency']:.1f}" for pl in pareto_levels]
    
    ax.set_zticks(z_indices)
    ax.set_zticklabels(z_labels)
    
    # 反转 Z 轴
    # 此时 Z 轴坐标是 0, 1, 2... 
    # Index 0 对应最小 Latency (levels[0])
    # Index Max 对应最大 Latency
    # invert_zaxis() 后：
    # 屏幕上方: Index 0 (Min Latency, "Best" if smaller is better)
    # 屏幕下方: Index Max (Max Latency)
    # 符合 "Latency 轴是反向的" (数值小的在上面)
    ax.invert_zaxis()

    # 调换 Accuracy 轴方向 (Y轴)
    # 根据用户需求 "acc轴请调换一下方向"
    ax.invert_yaxis()

    # 调整视角
    ax.view_init(elev=25, azim=-45)

    # 保存图片
    # Fix: tight_layout warning
    # plt.tight_layout() # sometimes causes issues in 3D
    # plt.subplots_adjust(left=0, right=1, bottom=0, top=1) # Alternative if needed, but tight_layout is usually fine if margins allow
    try:
        plt.tight_layout()
    except UserWarning:
        pass # Ignore warning if tight_layout fails
    
    plt.savefig(OUTPUT_IMAGE_PATH)
    print(f"3D Plot saved to: {OUTPUT_IMAGE_PATH}")
    plt.savefig(OUTPUT_IMAGE_PATH.replace('.pdf', '.png'))

if __name__ == "__main__":
    main()

