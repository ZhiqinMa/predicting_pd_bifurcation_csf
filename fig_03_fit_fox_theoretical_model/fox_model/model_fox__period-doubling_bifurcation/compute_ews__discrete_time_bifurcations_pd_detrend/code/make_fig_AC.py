# -------------- 01: Import libraries --------------
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


# -------------- 02: Load model data --------------
# Load df_plot data
df_plot_AC_compute = pd.read_csv('../output/data/df_lags_AC__compute__bif_pd_lags_5.csv')
df_plot_AC_fit = pd.read_csv('../output/data/df_lags_AC__fit__bif_pd_lags_5.csv')
# print(df_plot_AC_compute)
# print(df_plot_AC_fit)

# -------------- 03: Plotting --------------
time_far = 250
time_middle = 375
time_near = 500

title_fontsize = 16
label_fontsize = 14
tick_labelsize = 12

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(7, 4))

# 绘制符号
ax.scatter(df_plot_AC_compute['tau'], df_plot_AC_compute[f'time_{time_far}'], color='tab:green', marker='o', s=100)  # s控制符号大小
ax.scatter(df_plot_AC_compute['tau'], df_plot_AC_compute[f'time_{time_middle}'], color='tab:orange', marker='s', s=100)
ax.scatter(df_plot_AC_compute['tau'], df_plot_AC_compute[f'time_{time_near}'], color='tab:red', marker='^', s=100)

# 绘制线
ax.plot(df_plot_AC_fit['tau'], df_plot_AC_fit[f'time_{time_far}_fit'], label=f'Time$=${time_far}; $R^2=0.95$; $\lambda=-0.57$', color='tab:green', linewidth=2, linestyle='-')
ax.plot(df_plot_AC_fit['tau'], df_plot_AC_fit[f'time_{time_middle}_fit'], label=f'Time$=${time_middle}; $R^2=0.99$; $\lambda=-0.68$', color='tab:orange', linewidth=2, linestyle='-')
ax.plot(df_plot_AC_fit['tau'], df_plot_AC_fit[f'time_{time_near}_fit'], label=f'Time$=${time_near}; $R^2=1.00$; $\lambda=-0.85$', color='tab:red', linewidth=2, linestyle='-')

# 设置图例
# ax.legend(loc='upper right', fontsize=label_fontsize, frameon=False)
legend = ax.legend(loc='upper center', fontsize=tick_labelsize, frameon=False)
# 将图例字体颜色设置为对应的线条颜色
plt.setp(legend.get_texts()[0], color='tab:green')
plt.setp(legend.get_texts()[1], color='tab:orange')
plt.setp(legend.get_texts()[2], color='tab:red')

# 标题和轴标签
# ax.set_title('Autocorrelation Function vs. Lag Time', fontsize=title_fontsize, fontweight='bold', family='serif')
ax.set_xlabel(r'$\tau$', fontsize=label_fontsize)
ax.set_ylabel(r'AC($\tau$)', fontsize=label_fontsize)

# # 设置网格
# ax.grid(True, linestyle=':', linewidth=0.5)

# 设置轴范围
ax.set_xlim(-0.2, 4.2)
ax.set_ylim(-1.1, 1.4)

# 调整字体样式
plt.rcParams.update({'font.size': tick_labelsize, 'font.family': 'serif'})

plt.tight_layout()
plt.subplots_adjust(wspace=0.0, hspace=0.0)
fig.savefig(f'../output/figures/fig3.png', dpi=1200)
fig.savefig('../output/figures/fig3.eps', format='eps', dpi=1200)
plt.show()