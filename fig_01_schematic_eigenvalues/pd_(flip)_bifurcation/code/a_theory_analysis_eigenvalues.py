# -*- coding: utf-8 -*-
"""
Created on July 5, 2024

Theory analysis the eigenvalues

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
sub_dir_data_a = os.path.join(sub_dir_data, 'a')
sub_dir_figures = os.path.join(base_dir, 'figures')
try:
    os.makedirs(sub_dir_data, exist_ok=True)
    os.makedirs(sub_dir_data_a, exist_ok=True)
    os.makedirs(sub_dir_figures, exist_ok=True)
    print(f"Directories '{sub_dir_data}' created successfully.")
    print(f"Directories '{sub_dir_data_a}' created successfully.")
    print(f"Directories '{sub_dir_figures}' created successfully.")
except Exception as e:
    print(f"Failed to create directories '{sub_dir_data}'. Error: {e}")
    print(f"Failed to create directories '{sub_dir_data_a}'. Error: {e}")
    print(f"Failed to create directories '{sub_dir_figures}'. Error: {e}")


# Select the type of bifurcation
type_bif = 'pd'
# Setting different values of lambda
lambdas = [-0.125, -0.5, -0.75, -0.875]

# Define the radius
R = 1

# Generate data points for the circle using the parametric equation
theta = np.linspace(0, 2*np.pi, 100)
x = R * np.cos(theta)
y = R * np.sin(theta)

# Create a DataFrame to display the data points
data = pd.DataFrame({'theta': theta, 'x': x, 'y': y})
# Save data to a CSV file
csv_path = f'../output/data/a/data_eigenvalues__bif_{type_bif}.csv'
data.to_csv(csv_path, index=False)


# Create a figure and axis
fig, ax = plt.subplots()
# Plot the circle using the data points
ax.plot(x, y)
# Mark specific points = 标记特定的点
for lambda_value in lambdas:
    ax.plot(lambda_value, 0, marker='o', markersize=8)  # Mark eigenvalues on the real axis


# Set the title, labels and limits of the plot
ax.set_xlim(-R-0.15, R+0.15)
ax.set_ylim(-R-0.15, R+0.15)
# Set the aspect ratio of the plot to be equal
ax.set_aspect('equal', 'box')
# Display the plot
ax.axhline(0, color='black', linewidth=1.5)
ax.axvline(0, color='black', linewidth=1.5)

# Remove borders
for spine in ax.spines.values():
    spine.set_visible(False)

# Move Im label to the top and Re label to the right
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

# Set labels
ax.set_xlabel('Im', labelpad=10)
ax.set_ylabel('Re', rotation=0, labelpad=10)

# Remove axis numbers
ax.set_xticks([])
ax.set_yticks([])

# Save the plot
plot_path = f'../output/figures/plot_eigenvalues__bif_{type_bif}.png'
fig.savefig(plot_path)
plt.show()

# Print paths where files are saved
print(f'Data saved to: {csv_path}')
print(f'Plot saved to: {plot_path}')
