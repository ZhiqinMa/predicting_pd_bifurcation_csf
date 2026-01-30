# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:38:29 2022

Test DL classifier on Fox model with period-doubling bifrcation

Single pipeline to:
    - run simulations of Fox model - sweep over different noise and rof values
    - compute EWS and kenall tau at single point
    - compute DL predictions at single point

@author: tbury
"""

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_sims', type=int, help='Total number of model simulations', 
                    default=2500)
parser.add_argument('--use_inter_classifier', type=bool, help='Use the intermediate classifier as opposed to the hard saved classifier', default=True)

args = parser.parse_args()
model_sims = args.model_sims
use_inter_classifier = True if args.use_inter_classifier==True else False
print('use_inter_classifier=', use_inter_classifier)

import time
start_time = time.time()

import numpy as np
import pandas as pd

import funs_fox as funs

import ewstools_user
import ewstools_user_null

# from compute_roc.test_fox.code import ewstools_user

print('Running for {} sims'.format(model_sims))

np.random.seed(1)
eval_pt_start = 0.0     # percentage of way through pre-transition time series
eval_pt_end = 0.7       # percentage of way through pre-transition time series

sigma_vals = [0.00625, 0.0125, 0.025, 0.05, 0.1]
rof_vals = [100/500, 100/400, 100/300, 100/200, 100/100]
id_vals = np.arange(int(model_sims/25))  # number of simulations at each combo of rof and sigma

# EWS parameters
rw = 0.5        # rolling window
span = 0.25     # Lowess span


#--------------
# forced trajectories
#---------------
print('Simulate forced trajectories and compute EWS')
list_ktau_forced = []
list_ktau_null = []


for rof in rof_vals:
    for sigma in sigma_vals:
        for id_val in id_vals:
         
            s_forced, transition, s_null = funs.sim_rate_forcing(sigma, rof)

            # Compute EWS for forced trajectory
            ts = ewstools_user.TimeSeries(s_forced, transition=transition)
            ts.detrend(method='Lowess', span=span)
            ts.compute_var(rolling_window=rw)
            ts.compute_auto(rolling_window=rw, lag=1)
            ts.compute_dev(rolling_window=rw, roll_offset=10)
            # ts.compute_de(rolling_window=rw, fit_method='lmfit', user_model='discrete_exp_decay')   # method：1
            ts.compute_de(rolling_window=rw, user_model='discrete_exp_decay')  # method：2
            # ts.compute_ktau(tmin=0, tmax=transition*eval_pt)
            ts.compute_ktau(tmin=transition*eval_pt_start, tmax=transition * eval_pt_end)
            dic_ktau = ts.ktau
            dic_ktau['sigma'] = sigma
            dic_ktau['rof'] = rof
            dic_ktau['id'] = id_val
            list_ktau_forced.append(dic_ktau)

        
            # Compute EWS for null trajectory
            ts = ewstools_user_null.TimeSeries(s_null)
            ts.detrend(method='Lowess', span=span)
            ts.compute_var(rolling_window=rw)
            ts.compute_auto(rolling_window=rw, lag=1)
            ts.compute_dev(rolling_window=rw, roll_offset=10)
            # ts.compute_de(rolling_window=rw, fit_method='lmfit', user_model='discrete_exp_decay')   # method：1
            ts.compute_de(rolling_window=rw, user_model='discrete_exp_decay')  # method：2
            # ts.compute_ktau(tmin=0, tmax=transition*eval_pt)
            ts.compute_ktau(tmin=transition * eval_pt_start, tmax=transition * eval_pt_end)
            dic_ktau = ts.ktau
            dic_ktau['sigma'] = sigma
            dic_ktau['rof'] = rof
            dic_ktau['id'] = id_val
            list_ktau_null.append(ts.ktau)
            
    print('Complete for rof={}'.format(rof))

df_ktau_forced = pd.DataFrame(list_ktau_forced)
df_ktau_null = pd.DataFrame(list_ktau_null)

# Export data
# df_ktau_forced.to_csv('output/df_ktau_forced.csv', index=False)
# df_ktau_null.to_csv('output/df_ktau_null.csv', index=False)

df_ktau_forced.to_csv('../output/df_ktau_forced.csv', index=False)
df_ktau_null.to_csv('../output/df_ktau_null.csv', index=False)

# Export time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print('Script took {:.2f} seconds'.format(time_taken))

print("---------- Successful Test DL classifier and EWS on Fox model: test_fox ----------")


