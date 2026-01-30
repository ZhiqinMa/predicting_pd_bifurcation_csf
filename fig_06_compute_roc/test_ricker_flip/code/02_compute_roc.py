##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:58:49 2020

Compute ROC curves for EWS and DL predictions.

@author: Thomas M. Bury
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import scipy.stats as stats


#-------------
# Import predictions
#–------------

df_ktau_forced = pd.read_csv('../output/df_ktau_forced.csv')
df_ktau_null = pd.read_csv('../output/df_ktau_null.csv')

#----------------
# compute ROC curves
#----------------
print('Compute ROC curves')

df_ktau_forced['truth_value'] = 1
df_ktau_null['truth_value'] = 0

df_ktau = pd.concat([df_ktau_forced, df_ktau_null])
# print(df_ktau)



def roc_compute(truth_vals, indicator_vals):
    
    # Compute ROC curve and threhsolds using sklearn
    fpr, tpr, thresholds = metrics.roc_curve(truth_vals,indicator_vals)
    
    # Compute AUC (area under curve)
    auc = metrics.auc(fpr, tpr)
    
    # Put into a DF
    dic_roc = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': auc}
    df_roc = pd.DataFrame(dic_roc)

    return df_roc

# Initiliase list for ROC dataframes for predicting May fold bifurcation
list_roc = []


# Assign indicator and truth values for variance
indicator_vals = df_ktau['variance']
truth_vals = df_ktau['truth_value']
df_roc = roc_compute(truth_vals, indicator_vals)
df_roc['ews'] = 'Variance'
list_roc.append(df_roc)

# Assign indicator and truth values for lag-1 AC
indicator_vals = -df_ktau['ac1']
truth_vals = df_ktau['truth_value']
df_roc = roc_compute(truth_vals, indicator_vals)
df_roc['ews'] = 'Lag-1 AC'
list_roc.append(df_roc)

# Assign indicator and truth values for |DEV|
indicator_vals = df_ktau['dev']
truth_vals = df_ktau['truth_value']
df_roc = roc_compute(truth_vals, indicator_vals)
df_roc['ews'] = '|DEV|'
list_roc.append(df_roc)

# Assign indicator and truth values for de
indicator_vals = -df_ktau['de']
truth_vals = df_ktau['truth_value']
df_roc = roc_compute(truth_vals, indicator_vals)
df_roc['ews'] = 'DE-AC'
list_roc.append(df_roc)

# Concatenate roc dataframes
df_roc_full = pd.concat(list_roc, ignore_index=True)

# Export ROC data
filepath = '../output/df_roc.csv'
df_roc_full.to_csv(filepath, index=False,)

print('Exported ROC data to {}'.format(filepath))




# # TEMP work
# # Plot a histogram with error bars of the weights
# df_plot = df_dl.query('truth_value==1')[['1','2','3','4','5']]
# df_plot.boxplot()

print("---------- Successful Test EWS on Ricker_flip model: compute_roc ----------")


