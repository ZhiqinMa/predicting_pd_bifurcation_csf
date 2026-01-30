import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
lag_time = np.linspace(0, 10, 100)  # 滞后时间
auto_corr1 = np.exp(-0.1 * lag_time)  # 自关联函数1
auto_corr2 = np.exp(-0.2 * lag_time)  # 自关联函数2
auto_corr3 = np.exp(-0.3 * lag_time)  # 自关联函数3

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制三条线和符号
plt.plot(lag_time, auto_corr1, label='ACF 1', color='blue', marker='o', markersize=8, linewidth=2)
plt.plot(lag_time, auto_corr2, label='ACF 2', color='red', marker='s', markersize=8, linewidth=2)
plt.plot(lag_time, auto_corr3, label='ACF 3', color='green', marker='^', markersize=8, linewidth=2)

# 设置图例
plt.legend(loc='upper right', fontsize=12)

# 标题和轴标签
plt.title('Autocorrelation Function vs. Lag Time', fontsize=16, fontweight='bold')
plt.xlabel('Lag Time', fontsize=14, fontweight='bold')
plt.ylabel('Autocorrelation Function', fontsize=14, fontweight='bold')

# 设置网格
plt.grid(True)

# 设置轴范围
plt.xlim(0, 10)
plt.ylim(0, 1)

# 显示图形
plt.show()
