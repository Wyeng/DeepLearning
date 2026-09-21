import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# --- 1. 准备数据：生成一个心形轨迹 (Target x1) ---
def get_heart_points(n):
    t = np.linspace(0, 2 * np.pi, n)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return np.stack([x, y], axis=1) * 0.1


n_points = 200
x1 = get_heart_points(n_points)  # 终点：心形
x0 = np.random.randn(n_points, 2) * 2  # 起点：随机噪声

# --- 2. 设置画布 ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_title("Flow Matching: Straight Path (x0 -> x1)")
scatter = ax.scatter([], [], c='red', s=10)
line_plots = [ax.plot([], [], 'gray', alpha=0.2, lw=0.5)[0] for _ in range(n_points)]


def init():
    scatter.set_offsets(np.empty((0, 2)))
    return [scatter] + line_plots


# --- 3. 动画核心逻辑：线性插值 (Flow Matching 的本质) ---
def update(frame):
    t = frame / 50.0  # 时间从 0 到 1
    # Flow Matching 公式: xt = (1-t)*x0 + t*x1
    xt = (1 - t) * x0 + t * x1

    scatter.set_offsets(xt)

    # 绘制淡淡的运动轨迹
    for i in range(n_points):
        path = np.vstack([x0[i], xt[i]])
        line_plots[i].set_data(path[:, 0], path[:, 1])

    return [scatter] + line_plots


# 创建动画
ani = FuncAnimation(fig, update, frames=51, init_func=init, blit=True, interval=50)

plt.show()