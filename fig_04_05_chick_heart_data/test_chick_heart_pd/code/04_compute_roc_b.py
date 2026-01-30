# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:38:29 2022

- Compute ROC curve data using predictions at evaluation points

@author: tbury
@author: Zhiqin Ma: https://orcid.org/0000-0002-5809-464X
"""


import sys

import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import os
import plotly.express as px

import sklearn.metrics as metrics

# Import df predictions
df_ktau_forced = pd.read_csv('../output/data/df_ktau_pd_fixed_sub_b.csv')
df_ktau_null = pd.read_csv('../output/data/df_ktau_null_fixed_sub_b.csv')


#----------------
# compute ROC curves
#----------------
print('Compute ROC curves')

df_ktau_forced['truth_value'] = 1
df_ktau_null['truth_value'] = 0

# concat DataFrame
df_ktau = pd.concat([df_ktau_forced, df_ktau_null])


def roc_compute(truth_vals, indicator_vals):

    # Compute ROC curve and threhsolds using sklearn
    fpr, tpr, thresholds = metrics.roc_curve(truth_vals, indicator_vals)
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

# Assign indicator and truth values for dominant eigenvalue
indicator_vals = -df_ktau['de']
truth_vals = df_ktau['truth_value']
df_roc = roc_compute(truth_vals, indicator_vals)
df_roc['ews'] = 'de'
list_roc.append(df_roc)

# Concatenate roc dataframes
df_roc_full = pd.concat(list_roc, ignore_index=True)

# Export ROC data
filepath = '../output/data/df_roc_sub_b.csv'
df_roc_full.to_csv(filepath, index=False,)

print('Exported ROC data to {}'.format(filepath))

print("---------- Successful Test EWS and DL in chick heart data: compute_roc ----------")



