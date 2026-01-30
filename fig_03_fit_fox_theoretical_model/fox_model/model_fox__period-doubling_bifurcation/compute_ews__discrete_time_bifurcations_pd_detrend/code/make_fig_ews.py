# -------------- 01: Import libraries --------------
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

# -------------- 02: Load model data --------------
# Load df_plot data
df_plot = pd.read_csv('../output/data/df_ews__bif_pd_lags_5.csv')

# -------------- 03: Plotting --------------
plot_columns = [
    ('state', 'smoothing', 'State Over Time', 'State'),
    ('variance', None, 'Variance Over Time', 'Variance'),
    # ('variance_rolling_avg', None, 'Rolling Average Variance Over Time', 'Rolling Average Variance'),
    ('ac1', None, 'AC1 Over Time', 'AC(1)'),
    # ('ac1_rolling_avg', None, 'Rolling Average AC1 Over Time', 'Rolling Average AC1'),
    ('dominant_eigenvalue', None, 'Dominant_eigenvalue Over Time', '$\lambda$'),
    # ('dominant_eigenvalue_rolling_avg', None, 'Rolling Average dominant eigenvalue Over Time', 'Dominant Eigenvalue')
]

# Define labels for each subplot
subplot_labels = ['a', 'b', 'c', 'd']
# Define Kendall \tau for each subplot
subplot_Kendall_tau_texts = ['', 'Kendall tau = 0.86', 'Kendall tau = -0.89', 'Kendall tau = -1.0']
# Define the time for the vertical dashed line
time_marker = 500
# Define x-axis limits
x_limits = [-5, 675]

# Style configuration
line_width = 2.0
# colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple']
colors = ['tab:blue', 'tab:blue', 'tab:blue', 'tab:blue']
label_fontsize = 14
title_fontsize = 16
tick_labelsize = 12

# Plotting subplots
fig, axes = plt.subplots(4, 1, figsize=(7, 4), sharex=True)
# col: Primary column name to plot.
# col2: Secondary column name to plot (if any).
# title: Title of the subplot.
# ylabel: Label for the y-axis of the subplot.
for ax, (col, col2, title, ylabel), label, subtext, color in zip(axes, plot_columns, subplot_labels, subplot_Kendall_tau_texts, colors):
    ax.plot(df_plot.index, df_plot[col], label=col, color=color, linewidth=line_width)
    if col2:
        # ax.plot(df_plot.index, df_plot[col2], label=col2, color='tab:gray')
        ax.plot(df_plot.index, df_plot[col2], label=col2, color='black')
    # Add bold subplot text
    # ax.text(0.1, 0.95, subtext)
    ax.text(0.25, 0.4, subtext, transform=ax.transAxes, fontsize=label_fontsize, fontweight='bold', va='top', ha='left')
    # ax.set_title(title)
    ax.set_xlim(x_limits)  # Set x-axis limits
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.yaxis.set_label_coords(-0.1, 0.5)  # # Align y-labels at the same x position for all subplots
    # ax.legend()
    ax.label_outer()  # 只保留外部的x轴标签和y轴标签
    # Add bold subplot label
    ax.text(0.01, 0.95, label, transform=ax.transAxes, fontsize=label_fontsize, fontweight='bold', va='top', ha='left')
    ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)

    # Draw a vertical dashed line at time=500
    ax.axvline(x=time_marker, color='tab:gray', linestyle='--', lw=1.5)
    # Shade the region to the right of the dashed line
    # ax.axvspan(time_marker, x_limits[1], color='gray', alpha=0.3)
    ax.axvspan(time_marker, x_limits[1], color='tab:gray', alpha=0.3)

# # Left arrow
# axes[0].annotate(
#     '', xy=(249, 110), xytext=(0, 110),
#     arrowprops=dict(arrowstyle='<->', color='tab:blue', lw=1.5),
#     annotation_clip=False
# )

# Left arrow
axes[0].annotate(
    '', xy=(249, 105), xytext=(-4, 105),
    arrowprops=dict(arrowstyle='<->', shrinkA=1, shrinkB=1, color='black', connectionstyle='angle3'),
    annotation_clip=True
)

# 最后一个子图显示x轴标签
axes[-1].set_xlabel('Time', fontsize=label_fontsize)

plt.tight_layout()
plt.subplots_adjust(wspace=0.0, hspace=0.0)
fig.savefig(f'../output/figures/fig2.png', dpi=1200)
fig.savefig('../output/figures/fig2.pdf', dpi=1200)
fig.savefig('../output/figures/fig2.eps', format='eps', dpi=1200)
plt.show()
