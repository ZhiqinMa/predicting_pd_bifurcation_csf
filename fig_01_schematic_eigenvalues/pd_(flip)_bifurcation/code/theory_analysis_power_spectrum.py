# -*- coding: utf-8 -*-
"""
Created on July 5, 2024

Theory analysis the power spectrum

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
sub_dir_figures = os.path.join(base_dir, 'figures')
try:
    os.makedirs(sub_dir_data, exist_ok=True)
    os.makedirs(sub_dir_figures, exist_ok=True)
    print(f"Directories '{sub_dir_data}' created successfully.")
    print(f"Directories '{sub_dir_figures}' created successfully.")
except Exception as e:
    print(f"Failed to create directories '{sub_dir_data}'. Error: {e}")
    print(f"Failed to create directories '{sub_dir_figures}'. Error: {e}")


# Select the type of bifurcation
type_bif = 'fold_tc_pf'
# Setting different values of lambda
lambdas = [-0.125, -0.5, -0.75, -0.875]

sigma = 1.0
omega = np.linspace(-5.3, 5.3, 1061)  # Range of omega from -1.1 to 1.1


# Calculate AC(tau) = exp(lambda * |tau|) for each lambda
data = {f'lambda={lam}': (sigma**2/(2.0 * np.pi))*(1.0/(1.0 + lam**2 - 2.0 * lam * np.cos(omega))) for lam in lambdas}
# print(data)

# Create a DataFrame to store the data
df = pd.DataFrame(data, index=omega)
df.index.name = 'omega'

# Save data to a CSV file
csv_path = f'../output/data/data_power_spectrum__bif_{type_bif}.csv'
df.to_csv(csv_path)

# Plot AC(tau) as a function of tau
plt.figure(figsize=(8, 5))
for key in data:
    plt.plot(omega, data[key], label=key)

plt.title(r"power spectrum, $S(\omega)$")  # Using LaTeX for tau
plt.xlabel(r"$\omega$")  # Using LaTeX for tau in x-axis label
plt.ylabel(r"S$(\omega)$")  # Using LaTeX for tau in y-axis label
plt.legend()
plt.grid(True)

# Save the plot
plot_path = f'../output/figures/plot_power_spectrum__bif_{type_bif}.png'
plt.savefig(plot_path)
plt.show()

# Print paths where files are saved
print(f'Data saved to: {csv_path}')
print(f'Plot saved to: {plot_path}')
