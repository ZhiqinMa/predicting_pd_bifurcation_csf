import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
lag_time = np.linspace(0, 10, 100)  # 滞后时间
auto_corr1 = np.exp(-0.1 * lag_time)  # 自关联函数1的连续数据
auto_corr2 = np.exp(-0.2 * lag_time)  # 自关联函数2的连续数据
auto_corr3 = np.exp(-0.3 * lag_time)  # 自关联函数3的连续数据

# 特定数据点，例如每十个点取一个
indices = np.arange(0, 100, 10)
lag_points = lag_time[indices]
points1 = auto_corr1[indices]
points2 = auto_corr2[indices]
points3 = auto_corr3[indices]

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制线
ax.plot(lag_time, auto_corr1, label='ACF 1 Line', color='blue', linewidth=2, linestyle='--')
ax.plot(lag_time, auto_corr2, label='ACF 2 Line', color='red', linewidth=2, linestyle='-.')
ax.plot(lag_time, auto_corr3, label='ACF 3 Line', color='green', linewidth=2, linestyle='-')

# 绘制符号
ax.scatter(lag_points, points1, label='ACF 1 Points', color='blue', marker='o', s=100)  # s控制符号大小
ax.scatter(lag_points, points2, label='ACF 2 Points', color='red', marker='s', s=100)
ax.scatter(lag_points, points3, label='ACF 3 Points', color='green', marker='^', s=100)

# 设置图例
ax.legend(loc='upper right', fontsize=12, frameon=True)

# 标题和轴标签
ax.set_title('Autocorrelation Function vs. Lag Time', fontsize=16, fontweight='bold', family='serif')
ax.set_xlabel('Lag Time', fontsize=14, fontweight='bold', family='serif')
ax.set_ylabel('Autocorrelation Function', fontsize=14, fontweight='bold', family='serif')

# 设置网格
ax.grid(True, linestyle=':', linewidth=0.5)

# 设置轴范围
ax.set_xlim(0, 10)
ax.set_ylim(0, 1)

# 调整字体样式
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# 显示图形
plt.show()
