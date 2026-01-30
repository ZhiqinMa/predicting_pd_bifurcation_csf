# -*- coding: utf-8 -*-
"""
Created on June 2024

compute ews (var, ac1, and dominant eigenvalue/$\lambda$) for oscillatory bifurcation: Period-doubling (pd)

@author: Zhiqin Ma
https://orcid.org/0000-0002-5809-464X
Note：ewstools packages require python=3.8 above
"""

# -------------- 01: Import libraries --------------

# Start timer to record execution time of notebook
import time
start_time = time.time()

import numpy as np
np.random.seed(0)  # Set seed for reproducibility
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import ewstools


# Make export directory if doens't exist
def create_folder_os(path: str) -> None:
    """Create a directory if it doesn't exist using os module."""
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Folder '{path}' created successfully.")
    except Exception as e:
        print(f"Failed to create folder '{path}'. Error: {e}")

# Call the function to create the folder
sub_folder_data = '../output/data'
sub_folder_figures = '../output/figures'
create_folder_os(sub_folder_data)
create_folder_os(sub_folder_figures)


# -------------- 02: User-defined functions --------------
def lagsAC_transpose(df_ews: pd.DataFrame) -> pd.DataFrame:
    """Transpose the DataFrame containing lagged autocorrelation values."""
    # 筛选出以 'ac' 开头的列
    ac_columns = [col for col in df_ews.columns if col.startswith('ac')]
    # print(ac_columns)

    # 对这些列名进行排序
    ac_columns_sorted = sorted(ac_columns, key=lambda x: int(x[2:]))
    # print('ac_columns_sorted:')
    # print(ac_columns_sorted)

    # 创建新的 DataFrame 仅包含排序后的 'ac' 列
    df_lagsAC = df_ews[ac_columns_sorted]
    # print('df_lagsAC: ')
    # print(df_lagsAC)

    # Calculate the 'dominant_eigenvalue' col average value of the sliding window
    window_size = 0.15
    # Get absolute size of rollling window if given as a proportion
    if 0 < window_size <= 1:
        abs_window = int(window_size * len(df_lagsAC.dropna()))
    else:
        abs_window = window_size
    # print('rollling window:', abs_window, ' ', 'len(df_lagsAC):', len(df_lagsAC.dropna()))

    # The minimum acceptable number of observations to calculate the statistic = 可以接受的最小观测值数量来计算统计量
    df_lagsAC[ac_columns_sorted] = df_lagsAC[ac_columns_sorted].dropna().rolling(window=abs_window, min_periods=1).mean()
    # print('df_lagsAC_rolling: ')
    # print(df_lagsAC)

    # 转置 DataFrame
    df_lagsAC_tran = df_lagsAC.transpose()  # ordf_t = df.T

    # 修改行索引
    df_lagsAC_tran.index = [int(idx[2:]) for idx in df_lagsAC_tran.index]
    # 设置新的行索引名
    df_lagsAC_tran.index.name = 'tau'
    # 修改列名以反映时间点
    df_lagsAC_tran.columns = [f'time_{int(col)}' for col in df_lagsAC_tran.columns]
    # print(df_lagsAC_tran)

    return df_lagsAC_tran


# -------------- 03: Load model data --------------
# Load df_plot data
df_plot = pd.read_csv('../raw_data/df_plot.csv')

# Set bifurcation type: type_bif = 'pd', 'ns', 'fold', 'tc', 'pf'.
type_bif = 'pd'

# Select column data
df = df_plot[df_plot["model"] == type_bif].set_index('time')
series = df['state']

# print(series)   # 查看数据
# series.plot(); plt.show()   # 画图


# -------------- 04: Set up TimeSeries object and detrend --------------
transition = 500.0
ts = ewstools.TimeSeries(series, transition=transition)

# Detrend
ts.detrend(method='Lowess', span=0.25)

# print(ts.state)                                               # 查看数据
# ts.state[['state', 'smoothing']].plot(); plt.show()           # 画图
# ts.state.to_csv('../output_data/test_data.csv')               # 保存查看数据


# -------------- 05: Compute ews (variance and ac1) over a rolling window --------------
rw = 0.5
ts.compute_var(rolling_window=rw)
ts.compute_auto(rolling_window=rw, lag=1)

# Sava ews of var and ac1 to DataFrame
df_ews_var_ac = ts.state.join(ts.ews)


# -------------- 06: Compute the lags of autocorrelation over a rolling window --------------
# # method 1
# lags = int(rw * transition * 0.1)

# or method 2
multiple = 1        # Usually choose an integer multiple of the period
period = 4          # The minimum period of a sine function
lags = (multiple * period) + 1
# Check up
print('lags: ', lags)

# compute lag-\tau autocorrelation
for lag in np.arange(lags):
    ts.compute_auto(rolling_window=rw, lag=lag)

# Drop NaN values
df_ews = ts.ews.dropna()
# print(df_ews)

# Transpose DataFrame of ews_ac
df_lags_AC = lagsAC_transpose(df_ews)
# print(df_lags_AC)

# Export AC DataFrame
# df_lags_AC.to_csv('../output_data/df_lags_AC__compute__bif_{}_lags_{}.csv'.format(type_bif, lags))
df_lags_AC.to_csv(f'../output/data/df_lags_AC__compute__bif_{type_bif}_lags_{lags}.csv')


# -------------- 07: Computing autocorrelation EWS - dominant eigenvalue (T/R/$\lambda$) --------------
from scipy.optimize import curve_fit

# # 定义阻尼正弦波函数
# def damped_sine(x, T, A, omega, tc, y0):
#     """Damped sine wave function for fitting."""
#     return A * np.exp(-x / T) * np.sin((np.pi * x - np.pi * tc) / omega) + y0


# # 定义阻尼余弦波函数
# def damped_sine(x, T, A, omega, tc, y0):
#     return A * np.exp(-x / T) * np.cos((np.pi * x - np.pi * tc) / omega) + y0


# # 定义衰减函数
# def damped_sine(x, T, A, y0):
#     return A * np.exp(-x / T) + y0


# --------------------------------


# # 定义e指数衰减函数：连续系统-->Fold,TC,PF
# def fun_model(x, R, A):
#     return np.exp(R * abs(x))

# # 定义e指数衰减余弦波函数：连续系统-->Hopf
# def fun_model(x, R, A, omega):
#     return np.exp(R * abs(x)) * np.cos(omega * x)

# 定义指数衰减函数：离散系统-->Fold,TC,PF-->PD(Flip)
def fun_model(x, R, A):
    return A * R ** abs(x)

# # 定义指数衰减余弦波函数：离散系统-->NS
# def fun_model(x, R, A, Phi):
#     return R ** abs(x) * np.cos(Phi * x)


# copy data to be fitted
x = df_lags_AC.index.values
# print(x)

# empty list
list_lags_AC_fit = []
list_lags_AC_fit_colname = []
list_dominant_eigenvalue = []

for col in df_lags_AC.columns[::1]:
    y = df_lags_AC[col].values
    # print(y)
    print('col: ', int(col[5:]))

    try:
        # 使用curve_fit函数拟合数据
        # popt, pcov = curve_fit(damped_sine, x, y)
        popt, pcov = curve_fit(fun_model, x, y, maxfev=1000000)
        # 计算参数的标准偏差
        if pcov is not None:
            perr = np.sqrt(np.diag(pcov))
        else:
            perr = np.array([np.inf] * len(popt))  # 如果 pcov 是无效的，则标准偏差为无限
    except ValueError as e:
        print(e)
        continue

    # Calculate the quality of fit=计算拟合质量R^2
    y_pred = fun_model(x, *popt)

    # Append the fit result
    list_lags_AC_fit.append(y_pred)
    list_lags_AC_fit_colname.append('{}_fit'.format(col))

    # calculate R^2
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print('R^2: ', r_squared)
    print("--------------------")

    # Extract specific values from parameter data = 从参数数据中提取特定值
    dominant_eigenvalue = popt[0]
    dominant_eigenvalue_stderr = perr[0]
    print('dominant_eigenvalue: ', dominant_eigenvalue)
    # print('dominant_eigenvalue_stderr: ', dominant_eigenvalue_stderr)
    print('\n')

    # Append the result to the list
    list_dominant_eigenvalue.append({
        'time': int(col[5:]),
        'dominant_eigenvalue': dominant_eigenvalue,
        'R_squared': r_squared,
    })

    # # Plotting the fitting results at each time moment
    # plt.plot(x, y, '+', label='Data')
    # plt.plot(x, y_pred, label='Fitted Curve')
    # plt.legend()
    # plt.title(f'Compute and fit over lags -- time: {int(col[5:])}')
    # plt.xlabel('lags')
    # plt.ylabel('AC')
    # plt.show()

# Plotting the fitting result at the final time moment
plt.plot(x, y, '+', label='Data')
plt.plot(x, y_pred, label='Fitted Curve')
plt.legend()
plt.title(f'Compute and fit over lags -- time: {int(col[5:])}')
plt.xlabel('lags')
plt.ylabel('AC')
plt.savefig(f'../output/figures/df_lags_AC__fit_and_compute__bif_{type_bif}_lags_{lags}.png', dpi=300)
plt.show()


# Save fit lag-tau AC values
df_lags_AC_fit = pd.DataFrame(list_lags_AC_fit, index=list_lags_AC_fit_colname)
# transpose DataFrame
df_lags_AC_fit_T = df_lags_AC_fit.transpose()
# set index name
df_lags_AC_fit_T.index.name = 'tau'
# Export fit lag-tau AC DataFrame to csv file
df_lags_AC_fit_T.to_csv('../output/data/df_lags_AC__fit__bif_{}_lags_{}.csv'.format(type_bif, lags))

# Save ews of dominant eigenvalue to dataframe
df_ews_dominant_eigenvalue = pd.DataFrame(list_dominant_eigenvalue)
# print('df_ews_dominant_eigenvalue:')
# print(df_ews_dominant_eigenvalue)
# # Export ews of dominant eigenvalue to csv file
# df_ews_dominant_eigenvalue.to_csv('../output/data/df_ews__dominant_eigenvalue__bif_{}_lags_{}.csv'.format(type_bif, lags), index=False)

# Save full ews dataframe
df_ews_var_ac_rt = df_ews_var_ac.join(df_ews_dominant_eigenvalue.set_index('time'))
# print('df_ews_var_ac_rt:')
# print(df_ews_var_ac_rt)

# Calculate the 'dominant_eigenvalue' col average value of the sliding window
window_size = 0.15
# Get absolute size of rollling window if given as a proportion
if 0 < window_size <= 1:
    abs_window = int(window_size * len(df_ews_var_ac_rt.dropna()))
else:
    abs_window = window_size
print('rollling window:', abs_window, ' ', 'len(df_ews_var_ac_rt):',  len(df_ews_var_ac_rt.dropna()))

# # Calculate the average value of the sliding window (bad: Less data points of abs_window)
# df_ews_var_ac_rt['variance_rolling_avg'] = df_ews_var_ac_rt['variance'].dropna().rolling(window=abs_window).mean()
# df_ews_var_ac_rt['ac1_rolling_avg'] = df_ews_var_ac_rt['ac1'].dropna().rolling(window=abs_window).mean()
# df_ews_var_ac_rt['dominant_eigenvalue_rolling_avg'] = df_ews_var_ac_rt['dominant_eigenvalue'].dropna().rolling(window=abs_window).mean()

# # or The minimum acceptable number of observations to calculate the statistic = 可以接受的最小观测值数量来计算统计量
# df_ews_var_ac_rt['variance_rolling_avg'] = df_ews_var_ac_rt['variance'].dropna().rolling(window=abs_window, min_periods=1).mean()
# df_ews_var_ac_rt['ac1_rolling_avg'] = df_ews_var_ac_rt['ac1'].dropna().rolling(window=abs_window, min_periods=1).mean()
# df_ews_var_ac_rt['dominant_eigenvalue_rolling_avg'] = df_ews_var_ac_rt['dominant_eigenvalue'].dropna().rolling(window=abs_window, min_periods=1).mean()

# or Use of exponentially weighted moving averages = 使用指数加权移动平均
df_ews_var_ac_rt['variance_rolling_avg'] = df_ews_var_ac_rt['variance'].dropna().ewm(span=abs_window, adjust=False).mean()
df_ews_var_ac_rt['ac1_rolling_avg'] = df_ews_var_ac_rt['ac1'].dropna().ewm(span=abs_window, adjust=False).mean()
df_ews_var_ac_rt['dominant_eigenvalue_rolling_avg'] = df_ews_var_ac_rt['dominant_eigenvalue'].dropna().ewm(span=abs_window, adjust=False).mean()

# Export full ews to csv file
df_ews_var_ac_rt.to_csv('../output/data/df_ews__bif_{}_lags_{}.csv'.format(type_bif, lags), index=True)


# -------------- 08: Measure the trend of the EWS with Kendall tau --------------
# added 'dominant_eigenvalue' column to 'ts.ews'
ts.ews['dominant_eigenvalue'] = df_ews_var_ac_rt['dominant_eigenvalue']
# added 'variance_rolling_avg', 'ac1_rolling_avg', and 'dominant_eigenvalue_rolling_avg' columns to 'ts.ews'
ts.ews['variance_rolling_avg'] = df_ews_var_ac_rt['variance_rolling_avg']
ts.ews['ac1_rolling_avg'] = df_ews_var_ac_rt['ac1_rolling_avg']
ts.ews['dominant_eigenvalue_rolling_avg'] = df_ews_var_ac_rt['dominant_eigenvalue_rolling_avg']

# # compute the kendall tau values: e.g., ts.compute_ktau(tmin=450, tmax=500)
# ts.compute_ktau()
ts.compute_ktau(tmin=450, tmax=500)
# ts.compute_ktau(tmin=450, tmax=500)

# create DataFrame with two columns named "Key" and "Value"
df_ktau = pd.DataFrame(list(ts.ktau.items()), columns=['ews', 'ktau'])
# Export ews dataframe
df_ktau.to_csv('../output/data/df_ews__ktau_bif_{}_lags_{}.csv'.format(type_bif, lags), index=False)


# -------------- 09: Plotting --------------
# Plotting = 绘图
fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(14, 10), sharex=True)

# 绘制状态变量随时间的变化
ax1.plot(df_ews_var_ac_rt.index, df_ews_var_ac_rt['state'], label='State', color='blue')
ax1.plot(df_ews_var_ac_rt.index, df_ews_var_ac_rt['smoothing'], label='smoothing', color='black')
ax1.set_title('State Over Time')
ax1.set_xlabel('Time')
ax1.set_ylabel('State')
ax1.legend()

# 绘制方差随时间的变化
ax2.plot(df_ews_var_ac_rt.index, df_ews_var_ac_rt['variance'], label='variance', color='blue')
ax2.set_title('Variance Over Time')
ax2.set_xlabel('Time')
ax2.set_ylabel('variance')
ax2.legend()

# 绘制平滑方差随时间的变化
ax3.plot(df_ews_var_ac_rt.index, df_ews_var_ac_rt['variance_rolling_avg'], label='variance rolling average', color='blue')
ax3.set_title('Rolling average variance Over Time')
ax3.set_xlabel('Time')
ax3.set_ylabel('Rolling average variance')
ax3.legend()

# 绘制滞后1自关联函数随时间的变化
ax4.plot(df_ews_var_ac_rt.index, df_ews_var_ac_rt['ac1'], label='ac1', color='blue')
ax4.set_title('ac1 Over Time')
ax4.set_xlabel('Time')
ax4.set_ylabel('ac1')
ax4.legend()

# 绘制平滑滞后1自相关函数随时间的变化
ax5.plot(df_ews_var_ac_rt.index, df_ews_var_ac_rt['ac1_rolling_avg'], label='ac1 rolling average', color='blue')
ax5.set_title('Rolling average ac1 Over Time')
ax5.set_xlabel('Time')
ax5.set_ylabel('Rolling average ac1')
ax5.legend()

# 绘制弛豫时间随时间的变化
ax6.plot(df_ews_var_ac_rt.index, df_ews_var_ac_rt['dominant_eigenvalue'], label='dominant eigenvalue', color='blue')
ax6.set_title('dominant eigenvalue Over Time')
ax6.set_xlabel('Time')
ax6.set_ylabel('Dominant eigenvalue')
ax6.legend()

# 绘制平滑弛豫时间随时间的变化
ax7.plot(df_ews_var_ac_rt.index, df_ews_var_ac_rt['dominant_eigenvalue_rolling_avg'], label='dominant eigenvalue rolling average', color='blue')
ax7.set_title('Rolling average dominant eigenvalue Over Time')
ax7.set_xlabel('Time')
ax7.set_ylabel('Rolling average dominant eigenvalue')
ax7.legend()

plt.tight_layout()  # 自动调整子图的间隔
fig.savefig('../output/figures/df_ews__bif_{}_lags_{}.png'.format(type_bif, lags), dpi=300)
plt.show()


# # or Plotting 2
# fig, axes = plt.subplots(4, 1, figsize=(7, 4), sharex=True)
# plot_columns = [
#     ('state', 'smoothing', 'State Over Time', 'State'),
#     ('variance', None, 'Variance Over Time', 'Variance'),
#     # ('variance_rolling_avg', None, 'Rolling Average Variance Over Time', 'Rolling Average Variance'),
#     ('ac1', None, 'AC1 Over Time', 'AC1'),
#     # ('ac1_rolling_avg', None, 'Rolling Average AC1 Over Time', 'Rolling Average AC1'),
#     # ('dominant_eigenvalue', None, 'Dominant_eigenvalue Over Time', 'Dominant Eigenvalue'),
#     ('dominant_eigenvalue_rolling_avg', None, 'Rolling Average dominant eigenvalue Over Time', '$\lambda$')
# ]
#
# # Define labels for each subplot
# subplot_labels = ['a', 'b', 'c', 'd']
#
# # col: Primary column name to plot.
# # col2: Secondary column name to plot (if any).
# # title: Title of the subplot.
# # ylabel: Label for the y-axis of the subplot.
# for ax, (col, col2, title, ylabel), label in zip(axes, plot_columns, subplot_labels):
#     ax.plot(df_ews_var_ac_rt.index, df_ews_var_ac_rt[col], label=col, color='blue')
#     if col2:
#         ax.plot(df_ews_var_ac_rt.index, df_ews_var_ac_rt[col2], label=col2, color='black')
#     # ax.set_title(title)
#     ax.set_ylabel(ylabel)
#     ax.yaxis.set_label_coords(-0.1, 0.5)
#     # ax.legend()
#     ax.label_outer()  # 只保留外部的x轴标签和y轴标签
#     # Add bold subplot label
#     ax.text(0.01, 0.95, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
#
#
# # Left arrow
# axes[0].annotate(
#     '', xy=(250, 120), xytext=(0, 120),
#     arrowprops=dict(arrowstyle='<->', color='black', lw=1.5),
#     annotation_clip=False
# )
#
# # 最后一个子图显示x轴标签
# axes[-1].set_xlabel('Time')
#
# plt.tight_layout()
# plt.subplots_adjust(wspace=0, hspace=0)
# fig.savefig(f'../output/figures/df_ews__bif_{type_bif}_lags_{lags}_2.png', dpi=1200)
# plt.show()



# -------------- 10: Statistical running time --------------
# Stop timer
end_time = time.time()
print('Notebook took {:.1f}s to run'.format(end_time-start_time))