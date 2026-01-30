# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:38:29 2022

- Compute kendall tau at fixed evaluation points in chick heart data

@author: tbury
@author: Zhiqin Ma: https://orcid.org/0000-0002-5809-464X
"""

import time

start_time = time.time()

import numpy as np

np.random.seed(0)
import pandas as pd
import ewstools_self

# eval_pts = np.arange(0.64, 1.01, 0.04) #  percentage of way through pre-transition time series
# eval_pts = np.arange(0.61, 0.81, 0.01) #  percentage of way through pre-transition time series
# eval_pts = np.arange(0.81, 1.01, 0.01) #  percentage of way through pre-transition time series
# eval_pts = np.arange(0.71, 1.01, 0.01) #  percentage of way through pre-transition time series

# eval_pts = np.arange(0.51, 0.71, 0.01) #  percentage of way through pre-transition time series # good: a
# eval_pts = np.arange(0.51, 1.01, 0.01) #  percentage of way through pre-transition time series # good: b
eval_pts = np.arange(0.71, 0.91, 0.01) #  percentage of way through pre-transition time series # good: b

# EWS parameters
rw = 0.5  # rolling window
bw = 20  # Gaussian band width (# beats)

# Load in trajectory data
df = pd.read_csv('../raw_data/df_chick.csv')
df_pd = df[df['type'] == 'pd']
df_null = df[df['type'] == 'neutral']

# Load in transition times
df_transition = pd.read_csv('../raw_data/df_transitions.csv')
df_transition.set_index('tsid', inplace=True)

# --------------
# period-doubling trajectories
# ---------------
print('-------- Simulate period-doubling trajectories and compute EWS --------')

list_ktau = []

list_tsid = df_pd['tsid'].unique()
for tsid in list_tsid:

    df_spec = df_pd[df_pd['tsid'] == tsid].set_index('Beat number')
    transition = df_transition.loc[tsid]['transition']
    series = df_spec['IBI (s)']

    # Compute EWS
    ts = ewstools_self.TimeSeries(series, transition=transition)
    # ts.detrend(method='Lowess', span=50)
    ts.detrend(method='Gaussian', bandwidth=bw)

    ts.compute_var(rolling_window=rw)
    ts.compute_auto(rolling_window=rw, lag=1)
    ts.compute_de(rolling_window=rw, lags=5, fun_model=ewstools_self.TimeSeries.discrete_exp_decay)

    for eval_pt in eval_pts:
        eval_time = transition * eval_pt

        # Compute kendall tau at evaluation points
        ts.compute_ktau(tmin=0, tmax=eval_time)
        dic_ktau = ts.data_ktau
        dic_ktau['eval_time'] = eval_time
        dic_ktau['tsid'] = tsid
        list_ktau.append(dic_ktau)

    print('Complete for pd tsid={}'.format(tsid))

df_ktau_forced = pd.DataFrame(list_ktau)

# -------------
# null trajectories
# -------------
print('-------- Simulate null trajectories and compute EWS --------')

list_ktau = []

list_tsid = df_null['tsid'].unique()

for tsid in list_tsid:

    df_spec = df_null[df_null['tsid'] == tsid].set_index('Beat number')
    series = df_spec['IBI (s)']

    # Compute EWS
    ts = ewstools_self.TimeSeries(series)
    # ts.detrend(method='Lowess', span=50)
    ts.detrend(method='Gaussian', bandwidth=bw)

    ts.compute_var(rolling_window=rw)
    ts.compute_auto(rolling_window=rw, lag=1)
    ts.compute_de(rolling_window=rw, lags=5, fun_model=ewstools_self.TimeSeries.discrete_exp_decay)

    for eval_pt in eval_pts:
        eval_time = eval_pt * series.index[-1]

        # Compute kendall tau at evaluation points
        ts.compute_ktau(tmin=0, tmax=eval_time)
        dic_ktau = ts.data_ktau
        dic_ktau['eval_time'] = eval_time
        dic_ktau['tsid'] = tsid
        list_ktau.append(dic_ktau)

    print('Complete for null tsid={}'.format(tsid))

df_ktau_null = pd.DataFrame(list_ktau)

# Export data
df_ktau_forced.to_csv('../output/data/df_ktau_pd_fixed_sub_b.csv', index=False)
df_ktau_null.to_csv('../output/data/df_ktau_null_fixed_sub_b.csv', index=False)

# Time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print('Ran in {:.2f}s'.format(time_taken))

print("---------- Successful Test EWS in chick heart data: test_chick_heart ----------")
