import time
start_time = time.time()

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# plt.style.use('seaborn-v0_8-bright')
# print(plt.style.available)
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

""""
make fig. 1
"""


def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        # ax.text(0.5, 0.5, "ax%d" % (i + 1), va="center", ha="center")
        # ax.tick_params(labelbottom=False, labelleft=False)
        # ax.label_outer()
        # ax.tight_layout()
        pass

# ------------------------------------ 01: Load data ------------------------------------
# ------------- For ax1: -------------
# Setting different values of lambda
# lambdas_dp = [-0.125, -0.5, -0.75, -0.875]
lambdas_dp = [-0.125, -0.50,-0.75]

# Define the radius
R = 1

# Generate data points for the circle using the parametric equation
theta = np.linspace(0, 2*np.pi, 100)
x = R * np.cos(theta)
y = R * np.sin(theta)

# Create a DataFrame to display the data points
df_eigenvalues = pd.DataFrame({'theta': theta, 'x': x, 'y': y})


# ------------- For ax2: -------------
tau_dp = np.arange(0, 17, 1)  # Range of time delays tau from 0 to 45

# Calculate ACF(tau) = lambda ** |tau| for each lambda
data_AC_pd = {f'lambda={lam}': lam ** abs(tau_dp) for lam in lambdas_dp}

# Create a DataFrame to store the data
df_AC_dp = pd.DataFrame(data_AC_pd, index=tau_dp)
df_AC_dp.index.name = 'tau'


# ------------- For ax3: -------------
lambdas_hopf = [-0.75, -0.5, -0.125]


# ------------- For ax4: -------------
tau_hopf = np.arange(0, 17, 1)  # Range of time delays tau from 0 to 45
omega = 0.5
# Calculate AC(tau) = exp(lambda * |tau|) for each lambda
data_AC_hopf = {f'lambda={lam}': np.exp(lam * abs(tau_hopf)) * np.cos(omega * tau_hopf) for lam in lambdas_hopf}

# Create a DataFrame to store the data
df_AC_hopf = pd.DataFrame(data_AC_hopf, index=tau_hopf)
df_AC_hopf.index.name = 'tau'

# ------------------------------------ 02: Create GridSpec ------------------------------------
# 设置全局字体和大小
plt.rcParams.update({'font.size': 8, 'font.family': 'Arial'})

# Create a figure with GridSpec
fig = plt.figure(figsize=(8, 3.3), layout="constrained", dpi=600)  # 创建一个8x6英寸的图，600 dpi
gs = GridSpec(1, 3, figure=fig)

# Create subplots
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1:3])
# ax3 = fig.add_subplot(gs[1, 0])
# ax4 = fig.add_subplot(gs[1, 1:3])

# ------------------------------------ 02-1: Plot on ax1 ------------------------------------
ax1.axhline(0, color='black', linewidth=1.5)
ax1.axvline(0, color='black', linewidth=1.5)

ax1.plot(df_eigenvalues['x'], df_eigenvalues['y'], linestyle='dashdot', color='tab:blue',  linewidth=1.0)
# ax1.plot(df_eigenvalues['x'], df_eigenvalues['y'], linestyle='dashdot', color='tab:brown',  linewidth=1.0)

# Create a custom colormap from white to red
colors = ["tab:green", "tab:orange", "tab:red"]
# colors = ["tab:gray", "tab:orange", "tab:red"]
# colors = ["tab:blue", "tab:orange", "tab:red"]

# Mark specific points = 标记特定的点
for i, lambda_value in enumerate(lambdas_dp):
    ax1.plot(lambda_value, 0, marker='o', markersize=5, color=colors[i])  # Mark eigenvalues on the real axis
# Set the title, labels and limits of the plot
ax1.set_title("Period-doubling \n bifurcation", fontsize=12)

ax1.set_xlim(-R-0.15, R+0.15)
ax1.set_ylim(-R-0.15, R+0.15)
# Set the aspect ratio of the plot to be equal
ax1.set_aspect('equal', 'box')
# Display the plot

# Move Im label to the top and Re label to the right
ax1.xaxis.set_label_position("top")
ax1.xaxis.tick_top()
ax1.yaxis.set_label_position("right")
ax1.yaxis.tick_right()

# Set labels
ax1.set_xlabel('Im', fontsize=10, labelpad=5, horizontalalignment='center')
ax1.set_ylabel('Re', fontsize=10, rotation=0, labelpad=8, verticalalignment='center')

# Remove axis numbers
ax1.set_xticks([])
ax1.set_yticks([])

# Add subplot label
ax1.text(0.1, 1.1, 'a', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

# Add text annotations at the four intersection points
ax1.text(R+0.01, -0.12, '1', fontsize=8, ha='left', va='center')
ax1.text(-R-0.012, -0.12, '-1', fontsize=8, ha='right', va='center')
# ax1.text(lambdas[1]+0.012, -0.12, lambdas[1], fontsize=8, ha='right', va='center')
ax1.text(-0.08, R+0.01, '1', fontsize=8, ha='center', va='bottom')
ax1.text(-0.11, -R-0.01, '-1', fontsize=8, ha='center', va='top')

ax1.text(-0.4, 0.34, r'$\lambda$', color='tab:blue', va='center', ha='right', fontsize=12)
ax2.text(8, 0.95, r'$ACF(\tau) = \lambda^{|\tau|}$', weight='bold', va='center', ha='center', fontsize=10)
# ax3.text(0.5, 0.25, r'$\lambda = \mu \pm i \omega$',  weight='bold', va='center', ha='center', fontsize=10)
# ax4.text(8, 0.95, r'$ACF(\tau) = e^{\mu |\tau|} \cos \omega \tau$', weight='bold', va='center', ha='center', fontsize=10)

ax1.text(-0.05, -0.2, r'Re$(\lambda) \rightarrow -1$', color='tab:blue', va='center', ha='right')
ax1.text(-0.05, -0.4, r'Im$(\lambda) = 0$', color='tab:blue', va='center', ha='right')

# ax3.text(0.05, -0.2, r'$Im(\lambda_{1,2}) \ne 0$', weight='bold', va='center', ha='left')
# ax3.text(0.05, -0.4, r'$Re(\lambda_{1,2}) \rightarrow 0$', weight='bold', va='center', ha='left')


# Add a horizontal arrow
ax1.annotate('', xy=(-R+0.03, 0.25), xytext=(0, 0.25), arrowprops=dict(edgecolor='black', facecolor='tab:blue', shrink=0.05, width=1, headwidth=7))
# ax1.annotate('', xy=(-R+0.03, 0.25), xytext=(0, 0.25), arrowprops=dict(shrink=0.05, width=1, headwidth=7))

# Remove borders
for spine in ax1.spines.values():
    spine.set_visible(False)


# ------------------------------------ 02-2: Plot on ax2 ------------------------------------
ax2.axhline(0, color='black', linewidth=1.0, linestyle='--')
# ax2.axvline(0, color='black', linewidth=1.5)

# Create a custom colormap from white to red
colors = ["tab:green", "tab:orange", "tab:red"]
# colors = ["tab:gray", "tab:orange", "tab:red"]
# colors = ["tab:blue", "tab:orange", "tab:red"]

for i, key in enumerate(data_AC_pd):
    label = key.replace("lambda", r"\lambda").replace("=", r"\mathrm{=}")
    label = r"$" + label + r"$"
    # print(label)
    ax2.plot(tau_dp, data_AC_pd[key], 'o-', label=label, color=colors[i])
# ax2.set_title(r"Lag-$\tau$ autocorrelation, $\rho(\tau)$")  # Using LaTeX for tau
ax2.set_title(r"Lag-$\tau$ autocorrelation function", fontsize=12)
ax2.set_xlabel(r"$\tau$", fontsize=10)  # Using LaTeX for tau in x-axis label
ax2.set_ylabel(r"$ACF(\tau$)", fontsize=10)  # Using LaTeX for tau in y-axis label
ax2.legend(fontsize='large', frameon=False)
# ax2.grid(True)

# Add subplot label
ax2.text(-0.08, 1.1, 'b', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

# # 移动x轴到y=0的位置
# # 隐藏上方和右方的边框
# ax2.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
#
# # 将下方的边框（通常用作x轴）移动到y=0的位置
# ax2.spines['bottom'].set_position(('data', 0))
#
# # 保留左边的边框作为y轴
# ax2.spines['left'].set_position(('data', 0))
#
# # 设置刻度位置
# ax2.xaxis.set_ticks_position('bottom')
# ax2.yaxis.set_ticks_position('left')


# # ------------------------------------ 02-3: Plot on ax3 ------------------------------------
# ax3.axhline(0, color='black', linewidth=1.5)
# ax3.axvline(0, color='black', linewidth=1.5)
#
# ax3.plot(df_eigenvalues['x'], df_eigenvalues['y'], linestyle='dashdot', color='tab:blue',  linewidth=1.0)
#
# # Create a custom colormap from white to red
# colors = ["tab:green", "tab:orange", "tab:red"]
# # colors = ["tab:gray", "tab:orange", "tab:red"]
# # colors = ["tab:blue", "tab:orange", "tab:red"]
#
# # Mark specific points = 标记特定的点
# for i, lambda_value in enumerate(lambdas_hopf):
#     ax3.plot(lambda_value, 0.5, marker='o', markersize=5, color=colors[i])  # Mark eigenvalues on the real axis
#     ax3.plot(lambda_value, -0.5, marker='o', markersize=5, color=colors[i])  # Mark eigenvalues on the real axis
# # Set the title, labels and limits of the plot
# ax3.set_title("Hopf bifurcation", fontsize=12)
# ax3.set_xlim(-R-0.15, R+0.15)
# ax3.set_ylim(-R-0.15, R+0.15)
# # Set the aspect ratio of the plot to be equal
# ax3.set_aspect('equal', 'box')
# # Display the plot
#
# # Move Im label to the top and Re label to the right
# ax3.xaxis.set_label_position("top")
# ax3.xaxis.tick_top()
# ax3.yaxis.set_label_position("right")
# ax3.yaxis.tick_right()
#
# # Set labels
# ax3.set_xlabel('Im', labelpad=5, horizontalalignment='center')
# ax3.set_ylabel('Re', rotation=0, labelpad=8, verticalalignment='center')
#
# # Remove axis numbers
# ax3.set_xticks([])
# ax3.set_yticks([])
#
# # Add subplot label
# ax3.text(0.1, 1.1, 'c', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
#
# # Add text annotations at the four intersection points
# ax3.text(R+0.01, -0.12, '1', fontsize=8, ha='left', va='center')
# ax3.text(-R-0.012, -0.12, '-1', fontsize=8, ha='right', va='center')
# # ax1.text(lambdas[1]+0.012, -0.12, lambdas[1], fontsize=8, ha='right', va='center')
# ax3.text(-0.08, R+0.01, '1', fontsize=8, ha='center', va='bottom')
# ax3.text(-0.11, -R-0.01, '-1', fontsize=8, ha='center', va='top')
#
# # Add a horizontal arrow
# ax3.annotate('', xytext=(-R, 0.25), xy=(0, 0.25),
#              arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=7))
#
#
# # Remove borders
# for spine in ax3.spines.values():
#     spine.set_visible(False)
#
#
#
# # ------------------------------------ 02-4: Plot on ax4 ------------------------------------
# ax4.axhline(0, color='black', linewidth=1.0, linestyle='--')
# # ax4.axhline(0, color='black', linewidth=1.5)
#
# # Create a custom colormap from white to red
# colors = ["tab:green", "tab:orange", "tab:red"]
# # colors = ["tab:gray", "tab:orange", "tab:red"]
# # colors = ["tab:blue", "tab:orange", "tab:red"]
#
# for i, key in enumerate(data_AC_hopf):
#     ax4.plot(tau_hopf, data_AC_hopf[key], 'o-', label=key, color=colors[i])
# # ax2.set_title(r"Lag-$\tau$ autocorrelation, $\rho(\tau)$")  # Using LaTeX for tau
# ax4.set_title("Hopf bifurcation", fontsize=12)
# ax4.set_xlabel(r"$\tau$")  # Using LaTeX for tau in x-axis label
# ax4.set_ylabel(r"$ACF(\tau$)")  # Using LaTeX for tau in y-axis label
# ax4.legend()
#
# # Add subplot label
# ax4.text(-0.08, 1.1, 'd', transform=ax4.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')




# fig.suptitle("GridSpec")
format_axes(fig)

# Adjust the subplot layout to make sure the plot stays centered
fig.tight_layout()

plt.savefig('../output/figures/fig1.png', dpi=600)
plt.savefig('../output/figures/fig1.eps', format='eps', dpi=600)
# plt.show()
