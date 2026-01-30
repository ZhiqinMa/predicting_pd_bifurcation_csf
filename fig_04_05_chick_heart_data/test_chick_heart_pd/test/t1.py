import plotly.graph_objects as go

# 示例数据
x = [1, 2, 3, 4]
y = [10, 11, 12, 13]
text = ['点 A', '点 B', '点 C', '点 D']

fig = go.Figure(data=go.Scatter(
    x=x,
    y=y,
    mode='markers+text',  # 同时显示标记和文本
    text=text,
    textposition='top center',  # 文本位置 (top, bottom, left, right, 或组合)
    textfont=dict(size=18, color='firebrick'),  # 文本样式
    marker=dict(size=12, color='royalblue')
))

fig.update_layout(
    title='带文本标签的散点图',
    xaxis_title='X 轴',
    yaxis_title='Y 轴'
)
fig.show()