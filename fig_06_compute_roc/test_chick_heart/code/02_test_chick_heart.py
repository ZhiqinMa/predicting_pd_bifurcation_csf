# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:38:29 2022

- Compute DL and kendall tau at fixed evaluation points in chick heart data

@author: tbury
"""


import time
start_time = time.time()

import sys

import numpy as np
import pandas as pd

import ewstools_user

import os

# # Parse command line arguments
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--model_sims', type=int, help='Total number of model simulations',
#                     default=50)
# parser.add_argument('--use_inter_classifier', type=bool, help='Use the intermediate classifier as opposed to the hard saved classifier', default=True)
#
# args = parser.parse_args()
# model_sims = args.model_sims
# use_inter_classifier = True if args.use_inter_classifier == True else False
# print('use_inter_classifier=', use_inter_classifier)


np.random.seed(0)
# eval_pts = np.arange(0.64, 1.01, 0.01) #  percentage of way through pre-transition time series
eval_pts = np.arange(0.51, 0.71, 0.01) #  percentage of way through pre-transition time series # good

# EWS parameters
rw = 0.5    # rolling window
bw = 20     # Gaussian band width (# beats)

# Load in trajectory data
df = pd.read_csv('../input_data/df_chick.csv')
df_pd = df[df['type'] == 'pd']
df_null = df[df['type'] == 'neutral']

# Load in transition times
df_transition = pd.read_csv('../output/df_transitions.csv')
df_transition.set_index('tsid', inplace=True)


#--------------
# period-doubling trajectories
#---------------
list_ktau = []

list_tsid = df_pd['tsid'].unique()
for tsid in list_tsid:
    
    df_spec = df_pd[df_pd['tsid']==tsid].set_index('Beat number')
    transition = df_transition.loc[tsid]['transition']
    series = df_spec['IBI (s)']
    
    # Compute EWS
    ts = ewstools_user.TimeSeries(series, transition=transition)
    # ts.detrend(method='Lowess', span=50)
    ts.detrend(method='Gaussian', bandwidth=bw)
    
    ts.compute_var(rolling_window=rw)
    ts.compute_auto(rolling_window=rw, lag=1)
    ts.compute_dev(rolling_window=rw, roll_offset=1)
    ts.compute_de(rolling_window=rw, user_model='discrete_exp_decay')  # method：2

    for eval_pt in eval_pts:
        
        eval_time = transition*eval_pt
        
        # Compute kendall tau at evaluation points
        ts.compute_ktau(tmin=0, tmax=eval_time)
        dic_ktau = ts.ktau
        dic_ktau['eval_time'] = eval_time
        dic_ktau['tsid'] = tsid
        list_ktau.append(dic_ktau)

    print('Complete for pd tsid={}'.format(tsid))


df_ktau_forced = pd.DataFrame(list_ktau)



#-------------
# null trajectories
#-------------
print('Simulate null trajectories and compute EWS')

list_ktau = []
list_dl_preds = []

list_tsid = df_null['tsid'].unique()

for tsid in list_tsid:
    
    df_spec = df_null[df_null['tsid']==tsid].set_index('Beat number')
    series = df_spec['IBI (s)']    
    
    # Compute EWS
    ts = ewstools_user.TimeSeries(series)
    # ts.detrend(method='Lowess', span=50)
    ts.detrend(method='Gaussian', bandwidth=bw)
    
    ts.compute_var(rolling_window=rw)
    ts.compute_auto(rolling_window=rw, lag=1)
    ts.compute_dev(rolling_window=rw, roll_offset=1)
    ts.compute_de(rolling_window=rw, user_model='discrete_exp_decay')  # method：2

    for eval_pt in eval_pts:
        
        eval_time = eval_pt*series.index[-1]
        
        # Compute kendall tau at evaluation points
        ts.compute_ktau(tmin=0, tmax=eval_time)
        dic_ktau = ts.ktau
        dic_ktau['eval_time'] = eval_time
        dic_ktau['tsid'] = tsid
        list_ktau.append(dic_ktau)
    
    print('Complete for null tsid={}'.format(tsid))
        
df_ktau_null = pd.DataFrame(list_ktau)


# Export data
df_ktau_forced.to_csv('../output/df_ktau_pd_fixed.csv', index=False)
df_ktau_null.to_csv('../output/df_ktau_null_fixed.csv', index=False)


# Time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print('Ran in {:.2f}s'.format(time_taken))

print("---------- Successful Test EWS and DL in chick heart data: test_chick_heart ----------")


