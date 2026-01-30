# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 19:03:01 2021
Created on Mon Aug  6  2024

Make fig of ROC curves

@author: Thoams M. Bury
@author: Zhiqin Ma: https://orcid.org/0000-0002-5809-464X

"""


import time
start_time = time.time()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Import PIL for image tools
from PIL import Image

# -----------
# General fig params
# ------------

# Colour scheme
# cols = px.colors.qualitative.D3 # blue, orange, green, red, purple, brown
cols = px.colors.qualitative.Plotly  # blue, red, green, purple, orange, cyan, pink, light green
col_grays = px.colors.sequential.gray

dic_colours = {
    "state": "gray",
    "smoothing": col_grays[2],
    "variance": cols[0],
    "ac": cols[4],
    "dev": cols[2],
    "de": cols[3],
}

# Pixels to mm
mm_to_pixel = 96 / 25.4  # 96 dpi, 25.4mm in an inch

# Nature width of single col fig : 89mm
# Nature width of double col fig : 183mm

# Get width of single panel in pixels
fig_width = 183 * mm_to_pixel / 2  # 2 panels wide
fig_height = fig_width


font_size = 10
font_family = "Times New Roman"
font_size_letter_label = 14
font_size_auc_text = 10


# AUC annotations
x_auc = 0.98
y_auc = 0.6
x_N = 0.18
y_N = 0.05
y_auc_sep = 0.065

linewidth = 0.7
linewidth_axes = 0.5
tickwidth = 0.5
linewidth_axes_inset = 0.5

axes_standoff = 0


# Scale up factor on image export
scale = 8  # default dpi=72 - nature=300-600


def make_roc_figure(df_roc, letter_label, title="", text_N=""):
    """Make ROC figure (no inset)"""

    fig = go.Figure()

    # DE-AC plot
    df_trace = df_roc[df_roc["ews"] == "DE-AC"]
    auc_de = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["de"],
            ),
        )
    )

    # |DEV| plot
    df_trace = df_roc[df_roc["ews"] == "|DEV|"]
    auc_dev = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["dev"],
            ),
        )
    )

    # Lag-1  AC plot
    df_trace = df_roc[df_roc["ews"] == "Lag-1 AC"]
    auc_ac = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["ac"],
            ),
        )
    )

    # Variance plot
    df_trace = df_roc[df_roc["ews"] == "Variance"]
    auc_var = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["variance"],
            ),
        )
    )


    # Line y=x
    fig.add_trace(
        go.Scatter(
            x=np.linspace(0, 1, 100),
            y=np.linspace(0, 1, 100),
            showlegend=False,
            line=dict(
                color="black",
                dash="dot",
                width=linewidth,
            ),
        )
    )

    # --------------
    # Add labels and titles
    # ----------------------

    list_annotations = []

    label_annotation = dict(
        # x=sum(xrange)/2,
        x=0.02,
        y=1,
        text="<b>{}</b>".format(letter_label),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color="black",
            size=font_size_letter_label,
        ),
    )

    annotation_auc_de = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc,
        text="A<sub>DE-AC</sub>={:.2f}".format(auc_de),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            # color="black",
            color=dic_colours["de"],
            size=font_size_auc_text,
        ),
    )

    annotation_auc_dev = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc - y_auc_sep,
        text="A<sub>|DEV|</sub>={:.2f}".format(auc_dev),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            # color="black",
            color=dic_colours["dev"],
            size=font_size_auc_text,
        ),
    )

    annotation_auc_ac = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc - 2 * y_auc_sep,
        text="A<sub>AC</sub>={:.2f}".format(auc_ac),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            # color="black",
            color=dic_colours["ac"],
            size=font_size_auc_text,
        ),
    )

    annotation_auc_var = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc - 3 * y_auc_sep,
        text="A<sub>Var</sub>={:.2f}".format(auc_var),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            # color="black",
            color=dic_colours["variance"],
            size=font_size_auc_text,
        ),
    )

    annotation_N = dict(
        # x=sum(xrange)/2,
        x=x_N,
        y=y_N,
        text=text_N,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color="black",
            size=font_size_auc_text,
        ),
    )
    title_annotation = dict(
        # x=sum(xrange)/2,
        x=0.5,
        y=1,
        text=title,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color="black", size=font_size),
    )

    list_annotations.append(label_annotation)
    list_annotations.append(annotation_auc_de)
    list_annotations.append(annotation_auc_dev)
    list_annotations.append(annotation_auc_var)
    list_annotations.append(annotation_auc_ac)
    list_annotations.append(annotation_N)
    # list_annotations.append(title_annotation)

    fig["layout"].update(annotations=list_annotations)

    # -------------
    # General layout properties
    # --------------

    # X axes properties
    fig.update_xaxes(
        title=dict(
            text="False positive",
            standoff=axes_standoff,
        ),
        range=[-0.04, 1.04],
        ticks="outside",
        tickwidth=tickwidth,
        tickvals=np.arange(0, 1.1, 0.2),
        showline=True,
        linewidth=linewidth_axes,
        linecolor="black",
        mirror=False,
    )

    # Y axes properties
    fig.update_yaxes(
        title=dict(
            text="True positive",
            standoff=axes_standoff,
        ),
        range=[-0.04, 1.04],
        ticks="outside",
        tickvals=np.arange(0, 1.1, 0.2),
        tickwidth=tickwidth,
        showline=True,
        linewidth=linewidth_axes,
        linecolor="black",
        mirror=False,
    )

    # Overall properties
    fig.update_layout(
        legend=dict(x=0.6, y=0),
        width=fig_width,
        height=fig_height,
        margin=dict(l=30, r=5, b=15, t=5),
        font=dict(size=font_size, family=font_family),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
    )

    return fig



# -------
# Fox period-doubling
# --------
df_roc = pd.read_csv("../test_fox/output/df_roc.csv")
df_ktau_forced = pd.read_csv("../test_fox/output/df_ktau_forced.csv")

fig_roc = make_roc_figure(df_roc, "a", text_N="N={}".format(len(df_ktau_forced) * 2))
fig_roc.write_image("output/temp_roc_fox_pd.png", scale=scale)


# -------
# ricker period-doubling
# --------
df_roc = pd.read_csv("../test_ricker_flip/output/df_roc.csv")
df_ktau_forced = pd.read_csv("../test_ricker_flip/output/df_ktau_forced.csv")

fig_roc = make_roc_figure(df_roc, "b", text_N="N={}".format(len(df_ktau_forced) * 2))
fig_roc.write_image("output/temp_roc_ricker_flip.png", scale=scale)


# -------
# henon period-doubling
# --------
df_roc = pd.read_csv("../test_henon/output/df_roc.csv")
df_ktau_forced = pd.read_csv("../test_henon/output/df_ktau_forced.csv")

fig_roc = make_roc_figure(df_roc, "c", text_N="N={}".format(len(df_ktau_forced) * 2))
fig_roc.write_image("output/temp_roc_henon_pd.png", scale=scale)


# -------
# Heart data
# --------
df_roc = pd.read_csv("../test_chick_heart/output/df_roc.csv")
df_ktau_forced = pd.read_csv("../test_chick_heart/output/df_ktau_pd_fixed.csv")

fig_roc = make_roc_figure(df_roc, "d", text_N="N={}".format(len(df_ktau_forced) * 2))
fig_roc.write_image("output/temp_roc_heart.png", scale=scale)


# ------------
# Combine ROC plots
# ------------

# # Early or late predictions
# timing = 'late'

list_filenames = [
    "temp_roc_fox_pd",
    "temp_roc_ricker_flip",
    "temp_roc_henon_pd",
    "temp_roc_heart",
]
list_filenames = ["output/{}.png".format(s) for s in list_filenames]

list_img = []
for filename in list_filenames:
    img = Image.open(filename)
    list_img.append(img)

# Get heght and width of individual panels
ind_height = list_img[0].height
ind_width = list_img[0].width


# Create frame
dst = Image.new("RGB", (2 * ind_width, 2 * ind_height), (255, 255, 255))

# Paste in images
i = 0
for y in np.arange(2) * ind_height:
    for x in np.arange(2) * ind_width:
        dst.paste(list_img[i], (x, y))
        i += 1


dpi = 96 * 8  # (default dpi) * (scaling factor)
dst.save("output/fig6.png", dpi=(dpi, dpi))

# Remove temporary images
import os

for filename in list_filenames:
    try:
        os.remove(filename)
    except:
        pass



# Time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print("Ran in {:.2f}s".format(time_taken))


print("--------- Successful Make Figure 6 - ROC curves ---------")