import numpy as np
import plotly.express as px


def plot_route(x, y, start, end):
    color = np.zeros_like(x)
    color[start:end] = 1

    fig = px.scatter(x=x, y=y, color=color,
                     title=f"start:{start}, end={end}",
                     width=600, height=600)
    fig.update(layout_coloraxis_showscale=False)

    # axis equal
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return fig
