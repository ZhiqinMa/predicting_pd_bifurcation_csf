import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
lag_time = np.linspace(0, 10, 100)  # 滞后时间
auto_corr1 = np.exp(-0.1 * lag_time)  # 自关联函数1
auto_corr2 = np.exp(-0.2 * lag_time)  # 自关联函数2
auto_corr3 = np.exp(-0.3 * lag_time)  # 自关联函数3

# 创建主图形和子图
fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

# 绘制自关联函数1
axs[0].plot(lag_time, auto_corr1, label='ACF 1', color='blue', marker='o', markersize=6, linewidth=2)
axs[0].set_title('Autocorrelation Function 1', fontsize=14)
axs[0].set_ylabel('ACF 1', fontsize=12)
axs[0].grid(True)

# 绘制自关联函数2
axs[1].plot(lag_time, auto_corr2, label='ACF 2', color='red', marker='s', markersize=6, linewidth=2)
axs[1].set_title('Autocorrelation Function 2', fontsize=14)
axs[1].set_ylabel('ACF 2', fontsize=12)
axs[1].grid(True)

# 绘制自关联函数3
axs[2].plot(lag_time, auto_corr3, label='ACF 3', color='green', marker='^', markersize=6, linewidth=2)
axs[2].set_title('Autocorrelation Function 3', fontsize=14)
axs[2].set_ylabel('ACF 3', fontsize=12)
axs[2].grid(True)

# 设置共享X轴标签
for ax in axs:
    ax.set_xlabel('Lag Time', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)

# 调整子图间的间距
fig.tight_layout(pad=3.0)

# 显示图形
plt.show()
