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
cols = (px.colors.qualitative.Plotly)  # blue, red, green, purple, orange, cyan, pink, light green
col_grays = px.colors.sequential.gray

# dic_colours = {
#     "state": "gray",
#     "smoothing": col_grays[2],
#     "de": cols[0],
#     "variance": cols[1],
#     "ac": cols[2],
#     "dl_fold": cols[3],
#     "dl_hopf": cols[4],
#     "dl_branch": cols[5],
#     "dl_null": "black",
# }

# tab:green -> RGB(44, 160, 44) -> #2CA02C
# tab:blue -> RGB(31, 119, 180) -> #1F77B4
# tab:orange -> RGB(255, 127, 14) -> #FF7F0E
# tab:red -> RGB(214, 39, 40) -> #D62728
dic_colours = {
    "state": "gray",
    "smoothing": col_grays[2],
    "variance": "#2CA02C",
    "ac": "#FF7F0E",
    "de": "#D62728",
}

# Pixels to mm
mm_to_pixel = 96 / 25.4  # 96 dpi, 25.4mm in an inch

# Nature width of single col fig : 89mm
# Nature width of double col fig : 183mm

# Get width of single panel in pixels
fig_width = 183 * mm_to_pixel / 3  # 3 panels wide
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

    # de plot
    df_trace = df_roc[df_roc["ews"] == "de"]
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

    annotation_auc_dl = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc,
        text="A<sub>λ</sub>={:.2f}".format(auc_de),
        # text="A<sub>DE</sub>={:.2f}".format(auc_de),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=dic_colours["de"],
            size=font_size_auc_text,
        ),
    )

    annotation_auc_var = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc - y_auc_sep,
        text="A<sub>Var</sub>={:.2f}".format(auc_var),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=dic_colours["variance"],
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
            color=dic_colours["ac"],
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
    list_annotations.append(annotation_auc_dl)
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
# Heart data
# --------
df_roc = pd.read_csv("../output/data/df_roc_sub_a.csv")
df_ktau_forced = pd.read_csv("../output/data/df_ktau_pd_fixed_sub_a.csv")

fig_roc = make_roc_figure(df_roc, "a", text_N="N={}".format(len(df_ktau_forced) * 2))
fig_roc.write_image("../output/figures/fig5_roc_sub_a.pdf", scale=scale)
fig_roc.write_image("../output/figures/fig5_roc_sub_a.png", scale=scale)


# Time taken for script to run
end_time = time.time()
time_taken = end_time - start_time
print("Ran in {:.2f}s".format(time_taken))


print("--------- Successful Make Figure 3 - ROC curves ---------")