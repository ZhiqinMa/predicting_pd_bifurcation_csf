# -------------- 01: Import libraries --------------
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import string  # 用于生成 a, b, c... 标签


# -------------- 02: Load model data --------------
# Load df_plot data
df_plot = pd.read_csv("../output/df_plot.csv")

# Select type of models
# model_types = ['pd', 'ns', 'fold']
model_types = ['pd', 'flip', 'henon']

# [用户修改] 定义3个条件对应的列标题
model_titles = ['Fox model', 'Ricker model', r'Henon map model']


# -------------- 03: Plotting Configuration --------------
plot_columns = [
    ('state', 'smoothing', 'State Over Time', 'State'),
    ('variance', None, 'Variance Over Time', 'Variance'),
    ('ac1', None, 'AC1 Over Time', 'Lag-1 AC'),
    ('dev', None, '|DEV| Over Time', '|DEV|'),
    ('de', None, 'Dominant Eigenvalues', 'DE-AC'),
]

# Each row corresponds to one model, and each column corresponds to one variable = 每一行对应一个模型，每一列对应一个变量
subplot_Kendall_tau_texts = [
    # Row 1 (State)
    ['', '', ''],
    # Row 2 (Variance)
    ['Kendall tau = 0.88', 'Kendall tau = 0.85', 'Kendall tau = 0.85'],
    # Row 3 (AC1)
    ['Kendall tau = -0.95', 'Kendall tau = -0.95', 'Kendall tau = -0.84'],
    # Row 4 (DEV)
    ['Kendall tau = 0.91', 'Kendall tau = 0.91', 'Kendall tau = 0.81'],
    # Row 5 (DE)
    ['Kendall tau = -1.00', 'Kendall tau = -1.00', 'Kendall tau = -0.99']
]

# Define parameters
time_marker = 500
line_width = 2.0
label_fontsize = 12
tick_labelsize = 10
title_fontsize = 14
color_main = 'tab:blue'
color_secondary = 'black'  # 原代码中第二个变量的颜色


# -------------- 04: Create Subplots --------------
nrows = len(plot_columns)
# sharex='col': The same column shares the x-axis; ensure alignment within the columns,
# but allow the x-axis ranges of different columns to be different
# = 同一列共享x轴（确保列内对齐，但允许列间 X 轴范围不同）
fig, axes = plt.subplots(nrows, 3, figsize=(14, 5), sharex='col')

# Flatten axes irrelevant here since we need 2D indexing, but we can iterate efficiently.
# Loop through rows (Metrics)
for row_idx, (col_main, col_sub, _, ylabel) in enumerate(plot_columns):
    # Loop through columns (Conditions)
    for col_idx, model_type in enumerate(model_types):
        ax = axes[row_idx, col_idx]

        # Filter data
        df_subset = df_plot[df_plot['model_type'] == model_type].reset_index(drop=True)
        # Safety check: if data is empty, skip plotting
        if df_subset.empty:
            print(f"Warning: No data found for model_type='{model_type}'")
            continue

        # 1、Plot Main Variable
        ax.plot(df_subset.index, df_subset[col_main], label=col_main, color=color_main, linewidth=line_width)
        # Plot Secondary Variable (if exists)
        if col_sub and col_sub in df_subset.columns:
            ax.plot(df_subset.index, df_subset[col_sub], label=col_sub, color=color_secondary)

        # 2. Set Kendall Tau text = 设置 Kendall Tau 文本
        tau_text = subplot_Kendall_tau_texts[row_idx][col_idx]
        if tau_text:
            # Place uniformly at the lower left position, in bold font. = 统一放置在左下位置，bold字体
            ax.text(0.025, 0.25, tau_text, transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='bottom', ha='left')

        # 3. Set the sub-figure numbers (a, b, c...) o) = 设置子图编号 (a, b, c ... o)
        subplot_num = row_idx * 3 + col_idx  # Calculate the current sub-image number: row * 3 + col = 计算当前是第几个子图: row * 3 + col
        subplot_label = string.ascii_lowercase[subplot_num]  # Generate a, b, c... = 生成 a, b, c...
        ax.text(0.015, 0.95, subplot_label, transform=ax.transAxes,
                fontsize=label_fontsize, fontweight='bold', va='top', ha='left')

        # 4. Set different X-axis ranges based on the column index = 根据列索引设置不同的 X 轴范围
        if col_idx == 2:  # 第三列 (Henon)
            x_limits = [-5, 775]
        else:  # 第一列和第二列
            x_limits = [-5, 675]
        ax.set_xlim(x_limits)

        # 5. Set vertical dashed lines and shadows = 设置垂直虚线和阴影
        ax.axvline(x=time_marker, color='tab:gray', linestyle='--', lw=1.5)
        # Fill the shadow up to the current maximum 'x' value = 阴影填充至当前设定的最大'x'值
        ax.axvspan(time_marker, x_limits[1], color='tab:gray', alpha=0.3)

        # 6. Set the format of axis labels = 设置轴标签格式
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)

        # Only display the Y-axis labels in the first column = 只在第一列显示 Y 轴标签
        if col_idx == 0:
            ax.set_ylabel(ylabel, fontsize=label_fontsize)
            # Align the position of Y-axis labels = 对齐 Y 轴标签位置
            ax.yaxis.set_label_coords(-0.15, 0.5)

        # Only display the X-axis label on the last row = 只在最后一行显示 X 轴标签
        if row_idx == nrows - 1:
            ax.set_xlabel('Time', fontsize=label_fontsize)

        # Only display the column titles on the first row. = 只在第一行显示列标题
        if row_idx == 0:
            ax.set_title(model_titles[col_idx], fontsize=title_fontsize, pad=10)

        ## 7. Arrow annotation = 箭头注释
        # 这里的坐标 (x, y) 是相对于子图框的比例：
        # (0,0) 是左下角, (1,1) 是右上角。
        # 0.02 到 0.35 表示箭头占据子图宽度的约 33%
        # y=0.12 表示在底部稍微偏上的位置（避开刻度线）
        if row_idx == 0 and col_idx == 0:
            ax.annotate('',
                        xy=(0.00, 0.12),  # 箭头起点 (左侧)
                        xytext=(0.39, 0.12),  # 箭头终点 (右侧)
                        xycoords='axes fraction',  # 使用相对坐标系
                        textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='<->', color='black', lw=1.5),
                        annotation_clip=False)
        if row_idx == 0 and col_idx == 1:
            ax.annotate('',
                        xy=(0.00, 0.12),  # 箭头起点 (左侧)
                        xytext=(0.39, 0.12),  # 箭头终点 (右侧)
                        xycoords='axes fraction',  # 使用相对坐标系
                        textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='<->', color='black', lw=1.5),
                        annotation_clip=False)
        if row_idx == 0 and col_idx == 2:
            ax.annotate('',
                        xy=(0.00, 0.12),  # 箭头起点 (左侧)
                        xytext=(0.34, 0.12),  # 箭头终点 (右侧)
                        xycoords='axes fraction',  # 使用相对坐标系
                        textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='<->', color='black', lw=1.5),
                        annotation_clip=False)


# -------------- 05: Layout & Save --------------
plt.tight_layout()
# wspace 控制列间距，hspace 控制行间距。设为 0 可以紧挨着，但因为要共享轴，适当留一点点或设为0看效果
plt.subplots_adjust(wspace=0.15, hspace=0.0)

print("Saving figures...")
fig.savefig('../output/fig2.png', dpi=600)
fig.savefig('../output/fig2.pdf', dpi=600)
# fig.savefig('../output/fig2.eps', format='eps', dpi=600)
plt.show()