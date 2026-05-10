import json
import matplotlib.pyplot as plt
import os
import numpy as np
import itertools

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
INPUT_JSON_PATH = os.path.join(os.path.dirname(__file__), f"{MODEL_NAME}", "3d_merged_qasper.json")

# 输出图片保存路径
OUTPUT_IMAGE_PATH = os.path.join(os.path.dirname(__file__), f"{MODEL_NAME}", "3d_pareto_front_qasper_2d_projection.pdf")

# JSON 中的字段名称
KEY_ACCURACY = 'accuracy'
KEY_CR = 'cr'
KEY_LATENCY = 'latency'
KEY_ID = 'config_id'

# 绘图设置
# 参考二维代码的轴设置: X=Relative Accuracy, Y=Compression Ratio
X_LABEL = 'Relative Accuracy (%)'
Y_LABEL = 'Compression Ratio'
PLOT_TITLE = f'{MODEL_NAME} Pareto Frontier by Latency'
FIGURE_SIZE = (10, 6)

# Latency 聚类阈值
LATENCY_CLUSTERING_THRESHOLD = 0.3 

# 标记和颜色
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
# 使用 colormap 生成颜色
import matplotlib.cm as cm

# ==========================================
# FUNCTIONS
# ==========================================

def get_pareto_frontier(points):
    """
    计算 Pareto Frontier。
    假设两个维度都是越大越好 (Maximize X, Maximize Y)。
    points: list of dict, e.g. [{'x': 90, 'y': 5, 'id': 1}, ...]
    """
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
            # 如果 gap 大于阈值，分新组
            if l - current_group[-1] > threshold:
                # 额外的健壮性检查：如果当前值跟组内第一个值差距过大，也可以考虑切分，
                # 但这里主要遵循相邻差
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
        level_points = [p for p in points if min_l <= p['z'] <= max_l]
        
        levels.append({
            'index': i,
            'avg_latency': avg_l,
            'min_latency': min_l,
            'max_latency': max_l,
            'points': level_points,
            'label': f"{avg_l:.2f} ms"
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
    # 结构: [{'x': accuracy, 'y': cr, 'z': latency, 'id': config_id}, ...]
    points = []
    for entry in data:
        if KEY_ACCURACY in entry and KEY_CR in entry and KEY_LATENCY in entry:
            points.append({
                'x': entry[KEY_ACCURACY],
                'y': entry[KEY_CR],
                'z': entry[KEY_LATENCY],
                'id': entry.get(KEY_ID, 'N/A')
            })
    
    if not points:
        print("No valid data points found.")
        return

    # 3. 分组 Latency
    levels = group_latencies(points, LATENCY_CLUSTERING_THRESHOLD)
    print(f"Identified {len(levels)} latency levels.")

    # 4. 绘图
    plt.figure(figsize=FIGURE_SIZE)
    
    # 生成颜色
    # 使用 viridis 或 jet 等 colormap，根据 levels 数量采样
    # colors = cm.viridis(np.linspace(0, 1, len(levels)))
    # 或者使用 qualitative colormap 如 tab10 如果组数不多
    cmap = plt.get_cmap('tab10')
    
    # 标记循环器
    marker_cycle = itertools.cycle(MARKERS)

    for i, level in enumerate(levels):
        pts = level['points']
        if not pts:
            continue
            
        # 计算 Pareto 前沿
        pareto_points = get_pareto_frontier(pts)
        if not pareto_points:
            continue
            
        px = [p['x'] for p in pareto_points]
        py = [p['y'] for p in pareto_points]
        
        # 获取颜色和标记
        color = cmap(i % 10)
        marker = next(marker_cycle)
        label = f"Latency ≈ {level['label']}"
        
        # 绘制 Pareto 线和点
        plt.plot(px, py, c=color, linestyle='--', linewidth=1.5, alpha=0.8)
        plt.scatter(px, py, c=[color], marker=marker, s=30, label=label, zorder=5)
        
        # 标注 ID (可选，如果点太多可能会重叠)
        for p in pareto_points:
             plt.annotate(str(p['id']), 
                         (p['x'], p['y']),
                         textcoords="offset points", 
                         xytext=(3, 3), 
                         ha='left', 
                         fontsize=6,
                         color=color)

    # 装饰图表
    plt.title(PLOT_TITLE, fontweight='bold')
    plt.xlabel(X_LABEL, fontweight='bold')
    plt.ylabel(Y_LABEL, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 图例
    plt.legend(prop={'weight': 'normal', 'size': 8}, loc='best', title="Latency Groups")
    
    # 保存
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_PATH)
    print(f"Plot saved to: {OUTPUT_IMAGE_PATH}")
    
    # 保存 PNG 预览
    plt.savefig(OUTPUT_IMAGE_PATH.replace('.pdf', '.png'))

if __name__ == "__main__":
    main()

