import plotly.graph_objects as go

fig = go.Figure()

# 添加散点
fig.add_trace(go.Scatter(
    x=[1, 2, 3],
    y=[2, 1, 4],
    mode='markers'
))

# 添加多个文本注释
annotations = [
    dict(x=1, y=2, text='重要点', showarrow=True, arrowhead=2, ax=-50, ay=-40),
    dict(x=2, y=1, text='参考点', font=dict(size=20, color='red'), bgcolor='lightyellow'),
    dict(x=3, y=4, text='最大值', bordercolor='black', borderwidth=2, borderpad=4)
]

# 添加标题注释
fig.add_annotation(
    x=0.5, y=1.1,
    text='图表标题注释',
    showarrow=False,
    xref='paper', yref='paper',
    font=dict(size=24)
)

fig.update_layout(annotations=annotations, title='独立文本注释')
fig.show()