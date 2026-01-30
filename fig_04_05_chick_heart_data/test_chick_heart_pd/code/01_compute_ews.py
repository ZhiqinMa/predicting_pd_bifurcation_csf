# -*- coding: utf-8 -*-
"""
Created on May 1, 2024 16:04:19

-Compute EWS (variance and lag-1 autocorrelation) rolling over chick heart data

@author: Zhiqin Ma

Note：ewstools packages require python=3.8 above
"""

import time
start_time = time.time()

import os
import pandas as pd
import ewstools_user
import numpy as np
# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objs as go
# import warnings
# warnings.filterwarnings('ignore')


# Make export directory if doens't exist
try:
    os.mkdir('../output_single')
except:
    print('../output_single directory already exists!')

try:
    os.mkdir('../output_single/01_ews')
except:
    print('../output_single/01_ews directory already exists!')

np.random.seed(0)

# Load in trajectory data
df_traj = pd.read_csv('../raw_data/df_chick.csv')
df_traj_pd = df_traj[df_traj['type'] == 'pd']
# print(df_traj_pd)

# Load in transition times
df_transition = pd.read_csv('../raw_data/df_transitions.csv')
# print(df_transition)

# -------------
# Compute EWS for period-doubling trajectories
# --------------
# EWS parameters
rw = 0.5        # rolling window
bw = 20         # Gaussian band width (# beats)

list_ews = []   # empty list
list_ktau = []   # empty list
list_tsid = df_traj_pd['tsid'].unique()
# list_tsid = [1]

# Loop through each record
for tsid in list_tsid:
    # Filter and prepare specific trajectory data
    df_spec = df_traj_pd[df_traj_pd['tsid']==tsid].set_index('Beat number')
    # Extract transition value
    transition = df_transition[df_transition['tsid']==tsid]['transition'].iloc[0]
    # Extract IBI series
    series = df_spec['IBI (s)'].iloc[:]

    # Create TimeSeries object (new)
    ts = ewstools_user.TimeSeries(data=series, transition=transition)
    # ts.detrend(method='Lowess', span=50)
    ts.detrend(method='Gaussian', bandwidth=bw)

    # Compute EWS metrics
    ts.compute_var(rolling_window=rw)
    ts.compute_auto(rolling_window=rw, lag=1)
    ts.compute_dev(rolling_window=rw, roll_offset=1)
    # ts.compute_de(rolling_window=rw, fit_method='lmfit', user_model='discrete_exp_decay')   # method：1
    ts.compute_de(rolling_window=rw, user_model='discrete_exp_decay')  # method：2

    # Compute kendall tau
    ts.compute_ktau()

    # Prepare EWS dataframe
    data_ews = pd.concat([ts.state, ts.ews], axis=1)    # merge dataframe by columns
    # add new columns
    data_ews['tsid'] = tsid
    # Add to list
    list_ews.append(data_ews)

    # Prepare Kendall tau dataframe
    data_ktau = pd.DataFrame([ts.ktau])      # Create 'ktau' DataFrame
    # add new key and value
    data_ktau['tsid'] = tsid
    list_ktau.append(data_ktau)

# Concatenate dataframes
df_ews = pd.concat(list_ews)
df_ktau = pd.concat(list_ktau, ignore_index=True)

# Export ews dataframe
df_ews.to_csv('../output_single/01_ews/df_ews_pd_rolling.csv')
df_ktau.to_csv('../output_single/01_ews/df_ktau_pd_rolling.csv', index=False)

# Time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print('\nRan in {:.2f}s'.format(time_taken))

print('\n'"---------------------------- 01 Completed compute ews ----------------------------")



