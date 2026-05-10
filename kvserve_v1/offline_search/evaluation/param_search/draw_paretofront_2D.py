import json
import matplotlib.pyplot as plt
import os

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
# 输入数据文件路径 (建议使用绝对路径或相对于脚本的路径)
MODEL_NAME = 'Llama-3.1-8B-Instruct'
INPUT_JSON_PATH = os.path.join(os.path.dirname(__file__), f"{MODEL_NAME}", "hadamard_tolerance_3_all_tasks.json")

# 输出图片保存路径
OUTPUT_IMAGE_PATH = os.path.join(os.path.dirname(__file__), f"{MODEL_NAME}", "hadamard_pareto_front_all_tasks.pdf")

# JSON 中的字段名称
KEY_ACCURACY = 'accuracy'
KEY_CR = 'cr'
KEY_ID = 'config_id'

# 绘图设置
X_LABEL = 'Relative Accuracy (%)'
Y_LABEL = 'Compression Ratio'
PLOT_TITLE = f'{MODEL_NAME} Pareto Frontier'
FIGURE_SIZE = (8, 5)
FONT_SIZE_LABEL = 12

# 是否显示 Pareto 点上的 ID
SHOW_ID_LABELS = True

# 颜色设置
COLOR_ALL_POINTS = '#868686'      # 普通点的颜色
COLOR_PARETO_LINE = '#e41a1c'      # Pareto线的颜色
COLOR_PARETO_POINTS = '#e41a1c'    # Pareto点的颜色

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

def main():
    # 1. 读取数据
    if not os.path.exists(INPUT_JSON_PATH):
        print(f"Error: File not found at {INPUT_JSON_PATH}")
        return

    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. 提取绘图所需数据
    # 结构: [{'x': accuracy, 'y': cr, 'id': config_id}, ...]
    points = []
    for entry in data:
        if KEY_ACCURACY in entry and KEY_CR in entry:
            points.append({
                'x': entry[KEY_ACCURACY],
                'y': entry[KEY_CR],
                'id': entry.get(KEY_ID, 'N/A')
            })
    
    if not points:
        print("No valid data points found.")
        return

    # 3. 计算 Pareto Frontier
    pareto_points = get_pareto_frontier(points)
    
    # 4. 绘图
    plt.figure(figsize=FIGURE_SIZE)
    
    # 区分 Pareto 点和非 Pareto 点
    pareto_ids = set(p['id'] for p in pareto_points)
    non_pareto_points = [p for p in points if p['id'] not in pareto_ids]
    
    non_pareto_x = [p['x'] for p in non_pareto_points]
    non_pareto_y = [p['y'] for p in non_pareto_points]
    
    pareto_x = [p['x'] for p in pareto_points]
    pareto_y = [p['y'] for p in pareto_points]
    
    # 绘制 Pareto 前沿 (线 + 点)
    # 使用 plot 同时指定 marker 和 linestyle，图例会自动显示为"点在线上"的样式
    plt.plot(pareto_x, pareto_y, c=COLOR_PARETO_LINE, linestyle='--', linewidth=2, 
             marker='o', markersize=7, label='Pareto Frontier', zorder=5)

    # 标注 Pareto 点 ID
    if SHOW_ID_LABELS:
        for p in pareto_points:
            plt.annotate(str(p['id']), 
                         (p['x'], p['y']),
                         textcoords="offset points", 
                         xytext=(3, 6), 
                         ha='center', 
                         fontsize=8,
                         color=COLOR_PARETO_POINTS)

    # 3. 绘制没有在线上的点 (空心圆)
    if non_pareto_x:
        plt.scatter(non_pareto_x, non_pareto_y, facecolors='none', edgecolors=COLOR_ALL_POINTS, 
                   linewidths=1.5, label='Trials')

    # 装饰图表
    plt.title(PLOT_TITLE, fontweight='bold', fontsize=FONT_SIZE_LABEL)
    plt.xlabel(X_LABEL, fontweight='bold', fontsize=FONT_SIZE_LABEL)
    plt.ylabel(Y_LABEL, fontweight='bold', fontsize=FONT_SIZE_LABEL)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 修改图例位置到左下角
    plt.legend(prop={'weight': 'bold'}, loc='lower left')
    
    # 保存
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_PATH)
    print(f"Plot saved to: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    main()
