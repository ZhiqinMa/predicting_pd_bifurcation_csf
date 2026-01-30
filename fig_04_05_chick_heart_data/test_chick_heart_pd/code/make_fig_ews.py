# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 19:39:06 2021
Created on Mon Aug  5  2024

Make a figure that includes:
- Trajectory and smoothing
- Variance
- Lag-1 AC
- |DEV|
- DE-AC

For each period-doubling trajectory in chick-heart data

@author: tbury
@author: Zhiqin Ma: https://orcid.org/0000-0002-5809-464X
"""


import time
start_time = time.time()

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load in transition times
df_transitions = pd.read_csv("../raw_data/df_transitions.csv")
df_transitions.set_index("tsid", inplace=True)

# Load in EWS data
# df_ews = pd.read_csv("../output/data/df_ews_pd_rolling.csv")
df_ews = pd.read_csv("../output_single/01_ews/df_ews_pd_rolling.csv")

# Load in ktau data
df_ktau = pd.read_csv("../output_single/01_ews/df_ktau_pd_rolling.csv")

# Colour scheme
# cols = px.colors.qualitative.D3   # blue, orange, green, red, purple, brown
cols = px.colors.qualitative.Plotly   # blue, red, green, purple, orange, cyan, pink, light green
col_grays = px.colors.sequential.gray
dic_colours = {
    "state": "gray",
    "smoothing": col_grays[2],
    "variance": cols[0],
    "ac1": cols[4],
    "dev": cols[2],
    "de": cols[3],
}

# cols = px.colors.qualitative.Set1   # red, blue, green, purple, orange
# dic_colours = {
#     "state": "gray",
#     "smoothing": col_grays[2],
#     "variance": cols[1],
#     "ac1": cols[2],
#     "dev": cols[0],
#     "de": cols[3],
# }


# # tab:green -> RGB(44, 160, 44) -> #2CA02C
# # tab:blue -> RGB(31, 119, 180) -> #1F77B4
# # tab:orange -> RGB(255, 127, 14) -> #FF7F0E
# # tab:red -> RGB(214, 39, 40) -> #D62728
# dic_colours = {
#     "state": "gray",
#     "smoothing": col_grays[2],
#     "variance": "#2CA02C",
#     "ac1": "#1F77B4",
#     "dev": "#FF7F0E",
#     "de": "#D62728",
# }

linewidth = 1.2
opacity = 0.5


def make_grid_figure(df_ews, df_ktau, letter_label, title, transition=False):
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0,
    )

    # --------------
    # Trace for trajectory and EWS and kendall tau value
    # --------------

    # Trace for trajectory
    fig.add_trace(
        go.Scatter(
            x=df_ews["Beat number"],
            y=df_ews["state"],
            marker_color=dic_colours["state"],
            showlegend=False,
            line={"width": linewidth},
        ),
        row=1,
        col=1,
    )

    # Trace for smoothing
    fig.add_trace(
        go.Scatter(
            x=df_ews["Beat number"],
            y=df_ews["smoothing"],
            marker_color=dic_colours["smoothing"],
            showlegend=False,
            line={"width": linewidth},
        ),
        row=1,
        col=1,
    )

    # Trace for variance
    fig.add_trace(
        go.Scatter(
            x=df_ews["Beat number"],
            y=df_ews["variance"],
            marker_color=dic_colours["variance"],
            showlegend=False,
            line={"width": linewidth},
        ),
        row=2,
        col=1,
    )

    # Trace for lag-1 AC
    fig.add_trace(
        go.Scatter(
            x=df_ews["Beat number"],
            y=df_ews["ac1"],
            marker_color=dic_colours["ac1"],
            showlegend=False,
            line={"width": linewidth},
        ),
        row=3,
        col=1,
    )

    # Trace for |DEV|
    fig.add_trace(
        go.Scatter(
            x=df_ews["Beat number"],
            y=df_ews["dev"],
            marker_color=dic_colours["dev"],
            showlegend=False,
            line={"width": linewidth},
        ),
        row=4,
        col=1,
    )

    # Trace for dominant_eigenvalue (de)
    fig.add_trace(
        go.Scatter(
            x=df_ews["Beat number"],
            y=df_ews["de"],
            marker_color=dic_colours["de"],
            showlegend=False,
            line={"width": linewidth},
        ),
        row=5,
        col=1,
    )

    # --------------
    # Add vertical line where transition occurs
    # --------------

    if transition:
        # Add vertical lines where transitions occur
        list_shapes = []

        #  Make line for start of transition transition
        shape = {
            "type": "line",
            "x0": transition,
            "y0": 0,
            "x1": transition,
            "y1": 1,
            "xref": "x",
            "yref": "paper",
            "line": {"width": 2, "dash": "dot"},
        }

        # Add shape to list
        list_shapes.append(shape)

        # fig["layout"].update(shapes=list_shapes)
        fig.update_layout(shapes=list_shapes)

    # --------------
    # Add labels and titles and text
    # ----------------------

    list_annotations = []

    label_annotation = dict(
        # x=sum(xrange)/2,
        x=0.03,
        y=1,
        text="<b>{}</b>".format(letter_label),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color="black", size=16),
    )
    list_annotations.append(label_annotation)

    title_annotation = dict(
        # x=sum(xrange)/2,
        x=0.65,
        y=1,
        text=title,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color="black", size=14),
    )
    list_annotations.append(title_annotation)

    # Add variance Kendall tau text
    var_annotation = dict(
        x=0,
        y=1,
        xref="x2 domain",
        yref="y2 domain",
        text=f"Kendall tau = {round(df_ktau['variance'].iloc[0], 2):.2f}",
        showarrow=False,
        font=dict(size=10, color=dic_colours["variance"]),
    )
    list_annotations.append(var_annotation)

    # Add ac1 Kendall tau text
    ac1_annotation = dict(
        x=0,
        y=1,
        xref="x3 domain",
        yref="y3 domain",
        text=f"Kendall tau = {round(df_ktau['ac1'].iloc[0], 2):.2f}",
        showarrow=False,
        font=dict(size=10, color=dic_colours["ac1"]),
    )
    list_annotations.append(ac1_annotation)

    # Add |DEV| Kendall tau text = 添加|DEV| Kendall tau文本
    dev_annotation = dict(
        x=0,
        y=1,
        xref="x4 domain",
        yref="y4 domain",
        text=f"Kendall tau = {round(df_ktau['dev'].iloc[0], 2):.2f}",
        showarrow=False,
        font=dict(size=10, color=dic_colours["dev"]),
    )
    list_annotations.append(dev_annotation)

    # Add DE-AC Kendall tau text添加DE-AC Kendall tau文本
    de_annotation = dict(
        x=0,
        y=1,
        xref="x5 domain",
        yref="y5 domain",
        text=f"Kendall tau = {round(df_ktau['de'].iloc[0], 2):.2f}",
        showarrow=False,
        font=dict(size=10, color=dic_colours["de"]),
    )
    list_annotations.append(de_annotation)

    # fig["layout"].update(annotations=list_annotations)
    fig.update_layout(annotations=list_annotations)

    # -------
    # Axes properties
    # ---------

    # Let x range go 15% beyond transition
    tstart = df_ews["Beat number"].iloc[0]
    tend = tstart + 1.15 * (transition - tstart)

    fig.update_xaxes(
        title={"text": "Beat number", "standoff": 5},
        ticks="outside",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        range=[tstart, tend],
        row=5,
        col=1,
    )

    # Global y axis properties
    fig.update_yaxes(
        showline=True,
        ticks="outside",
        linecolor="black",
        mirror=True,
        showgrid=False,
        automargin=False,
    )

    # Global x axis properties
    fig.update_xaxes(
        showline=True,
        linecolor="black",
        mirror=False,
        showgrid=False,
    )

    fig.update_xaxes(mirror=True, row=1, col=1)

    fig.update_yaxes(
        title={
            "text": "IBI (s)",
            "standoff": 30,
        },
        tickformat=".1f",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title={
            "text": "Variance",
            "standoff": 30,
        },
        tickformat=".4f",
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title={
            "text": "Lag-1 AC",
            "standoff": 30,
        },
        tickformat=".2f",
        row=3,
        col=1,
    )

    fig.update_yaxes(
        title={
            "text": "|DEV|",
            "standoff": 30,
        },
        tickformat=".2f",
        # range=[-0.05, 1.07],
        row=4,
        col=1,
    )

    fig.update_yaxes(
        title={
            "text": "DE-AC",
            "standoff": 30,
        },
        tickformat=".2f",
        row=5,
        col=1,
    )

    fig.update_layout(
        height=400,
        width=200,
        margin={"l": 60, "r": 5, "b": 20, "t": 10},
        font=dict(size=12, family="Times New Roman"),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
    )

    fig.update_traces(mode="lines")

    return fig


# -------------
# Make ind figs for period-doubling trajectories
# -----------
import string

list_tsid = df_ews["tsid"].unique()

list_letter_labels = string.ascii_lowercase[: len(list_tsid)]

for i, tsid in enumerate(list_tsid):
    letter_label = list_letter_labels[i % 12]
    df_ews_spec = df_ews[df_ews["tsid"] == tsid]
    df_ktau_spec = df_ktau[df_ktau['tsid'] == tsid]
    transition = df_transitions["transition"].loc[tsid]

    # Title
    # title = 'tsid={}'.format(tsid)
    title = ""
    fig = make_grid_figure(
        df_ews_spec, df_ktau_spec, letter_label, title, transition=transition
    )
    # Export as png
    fig.write_image("img_{}.png".format(tsid), format='png', scale=8)
    print("Exported image {}".format(tsid))


# -----------
# Combine images into single png - part 1
# -----------

from PIL import Image

list_img = []
filename_png = "../output/figures/fig4_1.png"

# Load images
for tsid in np.arange(1, 13):
    img = Image.open("img_{}.png".format(tsid))
    list_img.append(img)

# Get height and width of individual panels
ind_height = list_img[0].height
ind_width = list_img[0].width

# Create frame
dst = Image.new("RGB", (6 * ind_width, 2 * ind_height), (255, 255, 255))

# Paste in images
i = 0
for y in np.arange(0, 2) * ind_height:
    for x in np.arange(0, 6) * ind_width:
        dst.paste(list_img[i], (x, y))
        i += 1

# Save as PNG
dst.save(filename_png)

# -----------
# Combine images into single png - part 2
# -----------

list_img = []
filename_png = "../output/figures/fig4_2.png"

for tsid in np.arange(13, 24):
    img = Image.open("img_{}.png".format(tsid))
    list_img.append(img)

# Get height and width of individual panels
ind_height = list_img[0].height
ind_width = list_img[0].width

# Create frame
dst = Image.new("RGB", (6 * ind_width, 2 * ind_height), (255, 255, 255))

# Paste in images
i = 0
for y in np.arange(0, 2) * ind_height:
    for x in np.arange(0, 6) * ind_width:
        try:
            dst.paste(list_img[i], (x, y))
            i += 1
        except:
            pass

# Save as PNG
dst.save(filename_png)


# Remove temp images
import os

for i in range(1, 24):
    try:
        os.remove("img_{}.png".format(i))
    except:
        pass


# Time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print("Ran in {:.2f}s".format(time_taken))


# # Export time taken for script to run
# end_time = time.time()
# time_taken = end_time - start_time
# path = 'time_make_fig.txt'
# with open(path, 'w') as f:
#     f.write('{:.2f}'.format(time_taken))

print("--------- Successful Make Fig. 4_1 and 4_2 - EWS in chick heart period-doubling ---------")