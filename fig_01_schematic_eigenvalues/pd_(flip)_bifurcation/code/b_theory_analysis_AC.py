# -*- coding: utf-8 -*-
"""
Created on July 5, 2024

Theory analysis the lag-\tau autocorrelation

@author: Zhiqin Ma
https://orcid.org/0000-0002-5809-464X
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Use the non-interactive 'TkAgg' backend
import matplotlib.pyplot as plt

# Create the new folder
base_dir = '../output'
sub_dir_data = os.path.join(base_dir, 'data')
sub_dir_data_b = os.path.join(sub_dir_data, 'b')
sub_dir_figures = os.path.join(base_dir, 'figures')
try:
    os.makedirs(sub_dir_data, exist_ok=True)
    os.makedirs(sub_dir_data_b, exist_ok=True)
    os.makedirs(sub_dir_figures, exist_ok=True)
    print(f"Directories '{sub_dir_data}' created successfully.")
    print(f"Directories '{sub_dir_data_b}' created successfully.")
    print(f"Directories '{sub_dir_figures}' created successfully.")
except Exception as e:
    print(f"Failed to create directories '{sub_dir_data}'. Error: {e}")
    print(f"Failed to create directories '{sub_dir_data_b}'. Error: {e}")
    print(f"Failed to create directories '{sub_dir_figures}'. Error: {e}")


# Select the type of bifurcation
type_bif = 'pd'

# Setting different values of lambda
lambdas = [-0.125, -0.5, -0.75, -0.875]
# lambdas = [-0.125, -0.5, -0.875]
# lambdas = [-0.125, -0.5, -0.75]

tau = np.arange(0, 30, 1)  # Range of time delays tau from 0 to 45

# Calculate AC(tau) = exp(lambda * |tau|) for each lambda
data_AC = {f'lambda={lam}': lam ** abs(tau) for lam in lambdas}
# data = {f'lambda={lam}': (-(1+lam))**abs(tau) for lam in lambdas}
# print(data)

# Create a DataFrame to store the data
df_AC = pd.DataFrame(data_AC, index=tau)
df_AC.index.name = 'tau'

# Save data to a CSV file
csv_path = f'../output/data/b/data_autocorrelation__bif_{type_bif}.csv'
df_AC.to_csv(csv_path)

# Plot AC(tau) as a function of tau
plt.figure(figsize=(8, 5))
for key in data_AC:
    plt.plot(tau, data_AC[key], 'o-', label=key)

plt.title(r"Lag-$\tau$ autocorrelation, $\rho(\tau)$")  # Using LaTeX for tau
plt.xlabel(r"$\tau$")  # Using LaTeX for tau in x-axis label
plt.ylabel(r"$\rho(\tau)$")  # Using LaTeX for tau in y-axis label
plt.legend()
plt.grid(True)

# Save the plot
plot_path = f'../output/figures/plot_autocorrelation__bif_{type_bif}.png'
plt.savefig(plot_path)
plt.show()

# Print paths where files are saved
print(f'Data saved to: {csv_path}')
print(f'Plot saved to: {plot_path}')
