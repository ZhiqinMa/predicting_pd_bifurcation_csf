"""
Created on May 1, 2024 16:04:19

-Compute EWS (var, ac1, dev,and de-ac) rolling over chick heart data

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

import os
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import ewstools_user



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

list_ews = []

# list_tsid = np.arange(1, 24)
list_tsid = np.arange(1, 24)
for tsid in list_tsid:
    print('tsid:', tsid)
    # Create export directories if they don't exist
    base_dir = '../output_single'
    sub_dir = os.path.join(base_dir, '02_dominant_eigenvalue')
    tsid_dir = os.path.join(sub_dir, f'tsid{tsid}')

    try:
        os.makedirs(tsid_dir, exist_ok=True)
        print(f"Directories '{tsid_dir}' created successfully.")
    except Exception as e:
        print(f"Failed to create directories '{tsid_dir}'. Error: {e}")


# -------------- 02: User-defined functions --------------
    def lagsAC_transpose(df_ews: pd.DataFrame) -> pd.DataFrame:
        """Transpose the DataFrame containing lagged autocorrelation values"""
        # 筛选出以 'ac' 开头的列
        ac_columns = [col for col in df_ews.columns if col.startswith('ac')]
        # print(ac_columns)

        # 对这些列名进行排序
        ac_columns_sorted = sorted(ac_columns, key=lambda x: int(x[2:]))
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

        # # The minimum acceptable number of observations to calculate the statistic = 可以接受的最小观测值数量来计算统计量
        # df_lagsAC[ac_columns_sorted] = df_lagsAC[ac_columns_sorted].dropna().rolling(window=abs_window, min_periods=1).mean()

        # or Use of exponentially weighted moving averages = 使用指数加权移动平均
        df_lagsAC[ac_columns_sorted] = df_lagsAC[ac_columns_sorted].dropna().ewm(span=abs_window, adjust=False).mean()
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
    type_bif = 'pd'

    # Load in trajectory data
    df_traj = pd.read_csv('../output_single/01_ews/df_ews_pd_rolling.csv')
    df_traj_pd = df_traj
    # print(df_traj_pd)

    # Load in transition times
    df_transition = pd.read_csv('../raw_data/df_transitions.csv')
    # print(df_transition)


    # select col data
    df_spec = df_traj_pd[df_traj_pd['tsid'] == tsid].set_index('Beat number')
    series = df_spec['state'].iloc[:]
    # series = df_spec['residuals'].iloc[:]
    # series = df_spec['variance'].iloc[:].dropna()

    # print(series)

    # print(series_may)                     # 查看数据
    # series.plot(); plt.show()             # 画图


# -------------- 04: Set up TimeSeries object and detrend --------------
    transition = df_transition[df_transition['tsid']==tsid]['transition'].iloc[0]
    # print('transition: ', transition)

    # Set up TimeSeries object and detrend
    ts = ewstools_user.TimeSeries(series, transition=transition)

    bw = 20
    # # Detrend
    ts.detrend(method='Gaussian', bandwidth=bw)

    # print(ts.state)                                                       # 查看数据
    # ts.state[['state', 'smoothing']].plot(); plt.show()                   # 画图


# -------------- 05: Compute ews( varian and ac1) ews over a rolling window --------------
    rw = 0.5
    ts.compute_var(rolling_window=rw)
    ts.compute_auto(rolling_window=rw, lag=1)
    ts.compute_dev(rolling_window=rw, roll_offset=1)

    # Sava ews of var and ac1 to DataFrame
    df_ews_var_ac = ts.state.join(ts.ews)


# -------------- 06: Compute the lags of autocorrelation over a rolling window --------------
    # # method 1
    # lags = int(rw * transition * 0.15)

    # or method 2
    multiple = 1                            # Usually choose an integer multiple of the period
    period = 4                              # The minimum period of a sine function
    lags = (multiple * period) + 1

    # # or method 3
    # lags = 9
    print('lags: ', lags)

    for lag in np.arange(lags):
        ts.compute_auto(rolling_window=rw, lag=lag)

    # Drop NaN values
    df_ews = ts.ews.dropna()
    # print(df_ews)

    # transpose DataFrame of ews_ac
    df_lags_AC = lagsAC_transpose(df_ews)
    # print(df_lags_AC)

    # # Export AC dataframe
    df_lags_AC.to_csv(f'../output_single/02_dominant_eigenvalue/tsid{tsid}/df_lags_AC__compute__tsid_{tsid}_bif_{type_bif}_lags_{lags}.csv')

    # print('--------------')
    # print("Tau Indices:")
    # print(df_lags_AC.index)
    # print("Values of time_209 column:")
    # print(df_lags_AC['time_209'])


# -------------- 07: Computing autocorrelation EWS - dominant eigenvalue (R/$\lambda$) --------------
    from scipy.optimize import curve_fit

    # 定义指数衰减函数: 离散系统-->Fold,TC,PF-->PD(Flip)
    def fun_model(x, R, A):
        return R ** abs(x)
        # return A * R ** abs(x)

    # # 定义指数衰减余弦波函数：离散系统-->NS
    # def fun_model(x, R, A, phi):
    #     return R ** abs(x) * np.cos(phi * x)
    #     return A * R ** abs(x) * np.cos(phi * x)

    # # 定义e指数衰减函数：连续系统-->Fold,TC,PF
    # def fun_model(x, R, A):
    #     return np.exp(R * abs(x))

    # # 定义e指数衰减余弦波函数：连续系统-->Hopf
    # def fun_model(x, R, A, omega):
    #     return np.exp(R * abs(x)) * np.cos(omega * x)


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

        # Fitting data using the ‘curve_fit’ = 使用’curve_fit‘拟合数据
        try:
            # popt, pcov = curve_fit(damped_sine, x, y)
            popt, pcov = curve_fit(fun_model, x, y, maxfev=1000000)
            # 计算参数的标准偏差
            if pcov is not None:
                perr = np.sqrt(np.diag(pcov))
            else:
                perr = np.array([np.inf] * len(popt))  # 如果 pcov 是无效的，则标准偏差为无限
        except ValueError as e:
            print(e)

        # Calculate the quality of fit = 计算拟合质量R^2
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
    plt.savefig(f'../output_single/02_dominant_eigenvalue/tsid{tsid}/df_lags_AC__fit_and_compute__tsid_{tsid}_bif_{type_bif}_lags_{lags}.png')
    plt.show()


    # Save fit lag-tau AC values
    df_lags_AC_fit = pd.DataFrame(list_lags_AC_fit, index=list_lags_AC_fit_colname)
    # Transpose DataFrame
    df_lags_AC_fit_T = df_lags_AC_fit.transpose()
    # Set index name
    df_lags_AC_fit_T.index.name = 'tau'
    # Export fit lag-tau AC DataFrame to csv file
    df_lags_AC_fit_T.to_csv('../output_single/02_dominant_eigenvalue/tsid{}/df_lags_AC__fit__tsid_{}_bif_{}_lags_{}.csv'.format(tsid, tsid, type_bif, lags))

    # Save ews of dominant eigenvalue to dataframe
    df_ews_dominant_eigenvalue = pd.DataFrame(list_dominant_eigenvalue)
    # print('df_ews_dominant_eigenvalue:')
    # print(df_ews_dominant_eigenvalue)
    # # Export ews of dominant eigenvalue to csv file
    # df_ews_dominant_eigenvalue.to_csv('../output_single/02_dominant_eigenvalue/tsid{}/df_ews__dominant_eigenvalue__tsid_{}_bif_{}_lags_{}.csv'.format(tsid, tsid, type_bif, lags), index=False)

    # Save full ews dataframe
    df_ews_var_ac_de = df_ews_var_ac.join(df_ews_dominant_eigenvalue.set_index('time'))
    # print('df_ews_var_ac_de:')
    # print(df_ews_var_ac_de)

    # Calculate the 'dominant_eigenvalue' col average value of the sliding window
    window_size = 0.15
    # Get absolute size of rollling window if given as a proportion
    if 0 < window_size <= 1:
        abs_window = int(window_size * len(df_ews_var_ac_de.dropna()))
    else:
        abs_window = window_size
    print('rollling window:', abs_window, ' ', 'len(df_ews_var_ac_de):', len(df_ews_var_ac_de.dropna()))

    # # Calculate the average value of the sliding window (bad: Less data points of abs_window)
    # df_ews_var_ac_de['variance_rolling_avg'] = df_ews_var_ac_de['variance'].dropna().rolling(window=abs_window).mean()
    # df_ews_var_ac_de['ac1_rolling_avg'] = df_ews_var_ac_de['ac1'].dropna().rolling(window=abs_window).mean()
    # df_ews_var_ac_de['dominant_eigenvalue_rolling_avg'] = df_ews_var_ac_de['dominant_eigenvalue'].dropna().rolling(window=abs_window).mean()

    # or The minimum acceptable number of observations to calculate the statistic = 可以接受的最小观测值数量来计算统计量
    df_ews_var_ac_de['variance_rolling_avg'] = df_ews_var_ac_de['variance'].dropna().rolling(window=abs_window, min_periods=1).mean()
    df_ews_var_ac_de['ac1_rolling_avg'] = df_ews_var_ac_de['ac1'].dropna().rolling(window=abs_window, min_periods=1).mean()
    df_ews_var_ac_de['dominant_eigenvalue_rolling_avg'] = df_ews_var_ac_de['dominant_eigenvalue'].dropna().rolling(window=abs_window, min_periods=1).mean()

    # # or Use of exponentially weighted moving averages = 使用指数加权移动平均
    # df_ews_var_ac_de['variance_rolling_avg'] = df_ews_var_ac_de['variance'].dropna().ewm(span=abs_window, adjust=False).mean()
    # df_ews_var_ac_de['ac1_rolling_avg'] = df_ews_var_ac_de['ac1'].dropna().ewm(span=abs_window, adjust=False).mean()
    # df_ews_var_ac_de['dominant_eigenvalue_rolling_avg'] = df_ews_var_ac_de['dominant_eigenvalue'].dropna().ewm(span=abs_window, adjust=False).mean()

    # add "tsid" column
    df_ews_var_ac_de['tsid'] = tsid
    list_ews.append(df_ews_var_ac_de)
    # Export full ews to csv file
    df_ews_var_ac_de.to_csv('../output_single/02_dominant_eigenvalue/tsid{}/df_ews__tsid_{}_bif_{}_lags_{}.csv'.format(tsid, tsid, type_bif, lags), index=True)


# -------------- 08: Measure the trend of the EWS with Kendall tau --------------
    # added 'dominant_eigenvalue' column to 'ts.ews'
    ts.ews['dominant_eigenvalue'] = df_ews_var_ac_de['dominant_eigenvalue']
    # added 'variance_rolling_avg', 'ac1_rolling_avg', and 'dominant_eigenvalue_rolling_avg' columns to 'ts.ews'
    ts.ews['variance_rolling_avg'] = df_ews_var_ac_de['variance_rolling_avg']
    ts.ews['ac1_rolling_avg'] = df_ews_var_ac_de['ac1_rolling_avg']
    ts.ews['dominant_eigenvalue_rolling_avg'] = df_ews_var_ac_de['dominant_eigenvalue_rolling_avg']

    # # compute the kendall tau values: e.g., ts.compute_ktau(tmin=450, tmax=500)
    # ts.compute_ktau()
    ts.compute_ktau(tmin=(transition - transition * 0.02), tmax=transition)

    # create DataFrame with two columns named "Key" and "Value"
    df_ktau = pd.DataFrame(list(ts.ktau.items()), columns=['ews', 'ktau'])
    # Export ews dataframe
    df_ktau.to_csv(f'../output_single/02_dominant_eigenvalue/tsid{tsid}/df_ews__ktau__tsid_{tsid}_bif_{type_bif}_lags_{lags}.csv', index=False)


# -------------- 09: Plotting --------------
    # Plotting = 绘图
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(14, 10), sharex=True)

    # 绘制状态变量随时间的变化
    ax1.plot(df_ews_var_ac_de.index, df_ews_var_ac_de['state'], label='State', color='blue')
    # ax1.plot(df_ews_var_ac_de.index, df_ews_var_ac_de['smoothing'], label='smoothing', color='black')
    ax1.set_title('State Over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('State')
    ax1.legend()

    # 绘制方差随时间的变化
    ax2.plot(df_ews_var_ac_de.index, df_ews_var_ac_de['variance'], label='variance', color='blue')
    ax2.set_title('Variance Over Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('variance')
    ax2.legend()

    # 绘制平滑方差随时间的变化
    ax3.plot(df_ews_var_ac_de.index, df_ews_var_ac_de['variance_rolling_avg'], label='variance rolling average', color='blue')
    ax3.set_title('Rolling average variance Over Time')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Rolling average variance')
    ax3.legend()

    # 绘制滞后1自关联函数随时间的变化
    ax4.plot(df_ews_var_ac_de.index, df_ews_var_ac_de['ac1'], label='ac1', color='blue')
    ax4.set_title('ac1 Over Time')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('ac1')
    ax4.legend()

    # 绘制平滑滞后1自相关函数随时间的变化
    ax5.plot(df_ews_var_ac_de.index, df_ews_var_ac_de['ac1_rolling_avg'], label='ac1 rolling average', color='blue')
    ax5.set_title('Rolling average ac1 Over Time')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Rolling average ac1')
    ax5.legend()

    # 绘制弛豫时间随时间的变化
    ax6.plot(df_ews_var_ac_de.index, df_ews_var_ac_de['dominant_eigenvalue'], label='dominant eigenvalue', color='blue')
    ax6.set_title('Dominant eigenvalue Over Time')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Dominant eigenvalue')
    ax6.legend()

    # 绘制平滑弛豫时间随时间的变化
    ax7.plot(df_ews_var_ac_de.index, df_ews_var_ac_de['dominant_eigenvalue_rolling_avg'],
             label='dominant eigenvalue rolling average', color='blue')
    ax7.set_title('Rolling average dominant eigenvalue Over Time')
    ax7.set_xlabel('Time')
    ax7.set_ylabel('Rolling average dominant eigenvalue')
    ax7.legend()

    plt.tight_layout()  # 自动调整子图的间隔
    fig.savefig(f'../output_single/02_dominant_eigenvalue/tsid{tsid}/df_ews__tsid_{tsid}_bif_{type_bif}_lags_{lags}.png')
    plt.show()

    # # or Plotting 2
    # fig, axes = plt.subplots(7, 1, figsize=(14, 10), sharex=True)
    # plot_columns = [
    #     ('state', 'smoothing', 'State Over Time', 'State'),
    #     ('variance', None, 'Variance Over Time', 'Variance'),
    #     ('variance_rolling_avg', None, 'Rolling Average Variance Over Time', 'Rolling Average Variance'),
    #     ('ac1', None, 'AC1 Over Time', 'AC1'),
    #     ('ac1_rolling_avg', None, 'Rolling Average AC1 Over Time', 'Rolling Average AC1'),
    #     ('dominant_eigenvalue', None, 'Dominant Eigenvalue Over Time', 'Dominant Eigenvalue'),
    #     ('dominant_eigenvalue_rolling_avg', None, 'Rolling Average Dominant Eigenvalue Over Time', 'Rolling Average Dominant Eigenvalue')
    # ]
    # # col: Primary column name to plot.
    # # col2: Secondary column name to plot (if any).
    # # title: Title of the subplot.
    # # ylabel: Label for the y-axis of the subplot.
    # for ax, (col, col2, title, ylabel) in zip(axes, plot_columns):
    #     ax.plot(df_ews_var_ac_de.index, df_ews_var_ac_de[col], label=col, color='blue')
    #     if col2:
    #         ax.plot(df_ews_var_ac_de.index, df_ews_var_ac_de[col2], label=col2, color='black')
    #     ax.set_title(title)
    #     ax.set_ylabel(ylabel)
    #     ax.legend()
    #
    # plt.xlabel('Time')
    # plt.tight_layout()
    # fig.savefig(f'../output_single/02_dominant_eigenvalue/tsid{tsid}/df_ews__tsid_{tsid}_bif_{type_bif}_lags_{lags}.png')
    # plt.show()


# -------------- 10: Statistical running time --------------
    # Stop timer
    end_time = time.time()
    print('Notebook took {:.1f}s to run'.format(end_time - start_time))

df_ews_pd = pd.concat(list_ews)
# Export full ews to csv file
df_ews_pd.to_csv('../output/data/df_ews_pd_rolling.csv', index=True)

