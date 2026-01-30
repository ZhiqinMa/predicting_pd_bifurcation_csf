# -*- coding: utf-8 -*-
"""
Generate data for fig 2 - EWS in a sample of model simulations
"""

import time
start_time = time.time()

import numpy as np
import pandas as pd

import ewstools_user

import sys
sys.path.append('../models/test_fox')
sys.path.append('../models/test_westerhoff')
sys.path.append('../models/test_ricker_fold')
sys.path.append('../models/test_ricker_flip')
sys.path.append('../models/test_kot')
sys.path.append('../models/test_lorenz')
sys.path.append('../models/test_henon')

import funs_fox, funs_westerhoff, funs_ricker_fold, funs_ricker_flip, funs_kot, funs_lorenz, funs_henon

# EWS paramers
span = 0.25
rw = 0.5


#----------
# Simulate models and compute EWS
#----------

## 1、Fox period-doubling model
np.random.seed(0)
type_bif = 'pd'
sigma = 0.05
s_forced, transition, s_null = funs_fox.sim_rate_forcing(sigma)

# Compute EWS
ts_pd = ewstools_user.TimeSeries(s_forced, transition=transition)
ts_pd.detrend(method='Lowess', span=span)
ts_pd.compute_var(rolling_window=rw)
ts_pd.compute_auto(rolling_window=rw, lag=1)

# Get |DEV| and de predictions
ts_pd.compute_dev(rolling_window=rw)
# ts_pd.compute_de(rolling_window=rw, fit_method='lmfit', user_model='discrete_exp_decay')   # method：1
ts_pd.compute_de(rolling_window=rw, user_model='discrete_exp_decay')     # method：2
print('EWS computed for Fox model')

# Compute the kendall tau values
ts_pd.compute_ktau(tmin=400, tmax=500)
# Create DataFrame with two columns named "Key" and "Value"
df_ktau = pd.DataFrame(list(ts_pd.ktau.items()), columns=['ews', 'ktau'])
# Export ews dataframe
df_ktau.to_csv('../output/df_ktau_bif_{}.csv'.format(type_bif), index=False)
print('kendall tau computed for Fox model')





## 2、Westerhoff NS model
np.random.seed(2)
type_bif = 'ns'
sigma = 0.1
s_forced, transition, s_null = funs_westerhoff.sim_rate_forcing(sigma)

# Compute EWS
ts_ns = ewstools_user.TimeSeries(s_forced, transition=transition)
ts_ns.detrend(method='Lowess', span=span)
ts_ns.compute_var(rolling_window=rw)
ts_ns.compute_auto(rolling_window=rw, lag=1)

# Get |DEV| and de predictions
ts_ns.compute_dev(rolling_window=rw)
# ts_ns.compute_de(rolling_window=rw, fit_method='lmfit', user_model='discrete_exp_cosine_decay')   # method：1
ts_ns.compute_de(rolling_window=rw, user_model='discrete_exp_cosine_decay')     # method：2
print('EWS computed for Westerhoff model')

# Compute the kendall tau values
ts_ns.compute_ktau(tmin=300, tmax=500)
# Create DataFrame with two columns named "Key" and "Value"
df_ktau = pd.DataFrame(list(ts_ns.ktau.items()), columns=['ews', 'ktau'])
# Export ews dataframe
df_ktau.to_csv('../output/df_ktau_bif_{}.csv'.format(type_bif), index=False)
print('kendall tau computed for Westerhoff model')
        


## 3、Ricker fold model
np.random.seed(0)
type_bif = 'fold'
sigma = 0.1
s_forced, transition, s_null = funs_ricker_fold.sim_rate_forcing(sigma)

# Compute EWS
ts_fold = ewstools_user.TimeSeries(s_forced, transition=transition)
ts_fold.detrend(method='Lowess', span=span)
ts_fold.compute_var(rolling_window=rw)
ts_fold.compute_auto(rolling_window=rw, lag=1)

# Get |DEV| and de predictions
ts_fold.compute_dev(rolling_window=rw)
# ts_fold.compute_de(rolling_window=rw, fit_method='lmfit', user_model='discrete_exp_decay')  # method：1
ts_fold.compute_de(rolling_window=rw, user_model='discrete_exp_decay')     # method：2
print('EWS computed for Ricker model')

# Compute the kendall tau values
ts_fold.compute_ktau(tmin=300, tmax=500)
# Create DataFrame with two columns named "Key" and "Value"
df_ktau = pd.DataFrame(list(ts_fold.ktau.items()), columns=['ews', 'ktau'])
# Export ews dataframe
df_ktau.to_csv('../output/df_ktau_bif_{}.csv'.format(type_bif), index=False)
print('kendall tau computed for Ricker Fold model')




## 4、Kot transcritical model
np.random.seed(2)
type_bif = 'tc'
sigma = 0.005
s_forced, transition, s_null = funs_kot.sim_rate_forcing(sigma)

# Compute EWS
ts_tc = ewstools_user.TimeSeries(s_forced, transition=transition)
ts_tc.detrend(method='Lowess', span=span)
ts_tc.compute_var(rolling_window=rw)
ts_tc.compute_auto(rolling_window=rw, lag=1)

# Get |DEV| and de predictions
ts_tc.compute_dev(rolling_window=rw)
# ts_tc.compute_de(rolling_window=rw, fit_method='lmfit', user_model='discrete_exp_decay')    # method：1
ts_tc.compute_de(rolling_window=rw, user_model='discrete_exp_decay')     # method：2
print('EWS computed for Kot model')

# Compute the kendall tau values
ts_tc.compute_ktau(tmin=300, tmax=500)
# Create DataFrame with two columns named "Key" and "Value"
df_ktau = pd.DataFrame(list(ts_tc.ktau.items()), columns=['ews', 'ktau'])
# Export ews dataframe
df_ktau.to_csv('../output/df_ktau_bif_{}.csv'.format(type_bif), index=False)
print('kendall tau computed for Kot model')




## 5、Lorenz pitchfork model
np.random.seed(0)
type_bif = 'pf'
sigma = 0.005
s_forced, transition, s_null = funs_lorenz.sim_rate_forcing(sigma)

# Compute EWS
ts_pf = ewstools_user.TimeSeries(s_forced, transition=transition)
ts_pf.detrend(method='Lowess', span=span)
ts_pf.compute_var(rolling_window=rw)
ts_pf.compute_auto(rolling_window=rw, lag=1)

# Get |DEV| and de predictions
ts_pf.compute_dev(rolling_window=rw)
# ts_pf.compute_de(rolling_window=rw, fit_method='lmfit', user_model='discrete_exp_decay')  # method：1
ts_pf.compute_de(rolling_window=rw, user_model='discrete_exp_decay')     # method：2
print('EWS computed for Lorenz model')

# Compute the kendall tau values
ts_pf.compute_ktau(tmin=300, tmax=500)
# Create DataFrame with two columns named "Key" and "Value"
df_ktau = pd.DataFrame(list(ts_pf.ktau.items()), columns=['ews', 'ktau'])
# Export ews dataframe
df_ktau.to_csv('../output/df_ktau_bif_{}.csv'.format(type_bif), index=False)
print('kendall tau computed for Lorenz model')



## 6、Ricker flip model
np.random.seed(0)
type_bif = 'flip'
sigma = 0.1
s_forced, transition, s_null = funs_ricker_flip.sim_rate_forcing(sigma)

# Compute EWS
ts_flip = ewstools_user.TimeSeries(s_forced, transition=transition)
ts_flip.detrend(method='Lowess', span=span)
ts_flip.compute_var(rolling_window=rw)
ts_flip.compute_auto(rolling_window=rw, lag=1)

# Get |DEV| and de predictions
ts_flip.compute_dev(rolling_window=rw)
# ts_flip.compute_de(rolling_window=rw, fit_method='lmfit', user_model='discrete_exp_decay')  # method：1
ts_flip.compute_de(rolling_window=rw, user_model='discrete_exp_decay')     # method：2
print('EWS computed for Ricker Flip model')

# Compute the kendall tau values
# ts_flip.compute_ktau(tmin=400, tmax=500)
ts_flip.compute_ktau(tmin=300, tmax=500)
# Create DataFrame with two columns named "Key" and "Value"
df_ktau = pd.DataFrame(list(ts_flip.ktau.items()), columns=['ews', 'ktau'])
# Export ews dataframe
df_ktau.to_csv('../output/df_ktau_bif_{}.csv'.format(type_bif), index=False)
print('kendall tau computed for Ricker Flip model')



## 7、Henon map model
np.random.seed(0)
type_bif = 'henon'
sigma = 0.1
s_forced, transition, s_null = funs_henon.sim_rate_forcing(sigma)

# Compute EWS
ts_henon = ewstools_user.TimeSeries(s_forced, transition=transition)
ts_henon.detrend(method='Lowess', span=span)
ts_henon.compute_var(rolling_window=rw)
ts_henon.compute_auto(rolling_window=rw, lag=1)

# Get |DEV| and de predictions
ts_henon.compute_dev(rolling_window=rw)
# ts_henon.compute_de(rolling_window=rw, fit_method='lmfit', user_model='discrete_exp_decay')  # method：1
ts_henon.compute_de(rolling_window=rw, user_model='discrete_exp_decay')     # method：2
print('EWS computed for Henon map model')

# Compute the kendall tau values
# ts_flip.compute_ktau(tmin=400, tmax=500)
ts_henon.compute_ktau(tmin=300, tmax=500)
# Create DataFrame with two columns named "Key" and "Value"
df_ktau = pd.DataFrame(list(ts_henon.ktau.items()), columns=['ews', 'ktau'])
# Export ews dataframe
df_ktau.to_csv('../output/df_ktau_bif_{}.csv'.format(type_bif), index=False)
print('kendall tau computed for Henon map model')




#--------
# Collect data and save
#--------

df_pd = ts_pd.state.join(ts_pd.ews)
df_pd['model_type'] = 'pd'

df_ns = ts_ns.state.join(ts_ns.ews)
df_ns['model_type'] = 'ns'

df_fold = ts_fold.state.join(ts_fold.ews)
df_fold['model_type'] = 'fold'

df_tc = ts_tc.state.join(ts_tc.ews)
df_tc['model_type'] = 'tc'

df_pf = ts_pf.state.join(ts_pf.ews)
df_pf['model_type'] = 'pf'

df_flip = ts_flip.state.join(ts_flip.ews)
df_flip['model_type'] = 'flip'

df_henon = ts_henon.state.join(ts_henon.ews)
df_henon['model_type'] = 'henon'


df_plot = pd.concat([df_pd, df_ns, df_fold, df_tc, df_pf, df_flip, df_henon])
df_plot.to_csv('../output/df_plot.csv')


# Export time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print('Script took {:.2f}s'.format(time_taken))


















