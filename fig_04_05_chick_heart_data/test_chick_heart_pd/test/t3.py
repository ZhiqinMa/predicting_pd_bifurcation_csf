import plotly.graph_objects as go

fig = go.Figure()

fig.add_annotation(
    x=0.5,
    y=0.5,
    text='居中文本',
    showarrow=False,
    font=dict(size=50, color='darkblue'),
    xref='paper',
    yref='paper'
)

fig.update_layout(
    xaxis=dict(showgrid=False, visible=False),
    yaxis=dict(showgrid=False, visible=False),
    plot_bgcolor='lightyellow'
)
fig.show()