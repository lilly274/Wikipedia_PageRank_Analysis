from datetime import datetime
import dash
import pandas as pd
from _plotly_utils.colors import sample_colorscale
from dash import dcc, Input, Output, State
from dash import callback
import plotly.graph_objects as go
import requests
import networkx as nx
import networkx.algorithms.community as community_graph
import numpy as np
import layout


app = dash.Dash(__name__)


def empty_figure():

    fig = go.Figure()

    fig.update_layout(
        xaxis=dict(
            visible=False,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            visible=False,
            showgrid=False,
            zeroline=False
        ),
        margin=dict(l=20, r=20, t=20, b=20)
    )

    return fig


def build_layout(n_clicks, topic, depth, edge_number):

    # API
    response = requests.get(
        "http://127.0.0.1:8000/analyze",
        params={
            "topic": topic,
            "depth": depth,
            "edge_number": edge_number,
        }
    )
    data = response.json()

    # Graph
    G = nx.DiGraph()
    for edge in data["edges"]:
        G.add_edge(edge["source"], edge["target"])

    pagerank = nx.pagerank(G, alpha=0.85)
    communities = list(community_graph.greedy_modularity_communities(G))

    cluster_centers = {
        i: np.random.uniform(-1, 1, size=2)
        for i in range(len(communities))
    }

    # Nodes
    node_x, node_y, labels, raw_sizes, node_color, texts = [], [], [], [], [], []

    pos = {}
    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i
            sub_pos = nx.spring_layout(
                G.subgraph(comm),
                center=cluster_centers[i],
                scale=0.3
            )
            pos.update(sub_pos)

    for node in G.nodes():
        x, y = pos[node]
        pr = pagerank[node]
        node_x.append(x)
        node_y.append(y)
        labels.append(node)
        raw_sizes.append(pagerank[node])
        node_color.append(community_map.get(node, 0))
        texts.append(
            f"<b>{node}</b><br>"
            f"PageRank: {pr:.4f}<br>"
        )

    # Size normalization
    min_size, max_size = 15, 50
    min_pr, max_pr = min(raw_sizes), max(raw_sizes)
    sizes = [
        min_size + (s - min_pr) / (max_pr - min_pr) * (max_size - min_size)
        for s in raw_sizes
    ]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hovertext=texts,
        hoverinfo="text",
        text=labels,
        marker=dict(
            size=sizes,
            color=node_color,
            colorscale="Bluered",
            line=dict(width=1, color="black")
        )
    )

    # Edges
    edge_x, edge_y, annotations = [], [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        annotations.append(
            dict(
                ax=x0,
                ay=y0,
                x=x1,
                y=y1,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=0.6,
                arrowcolor="rgba(120,120,120,0.6)"
            )
        )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="rgba(150,150,150,0.4)"),
        hoverinfo="none"
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        annotations=annotations,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # Table definition
    df = pd.DataFrame({
        "Name": list(G.nodes()),
        "PageRank": [pagerank[n] for n in G.nodes()],
        "Eingehende Kanten": [G.in_degree(n) for n in G.nodes()],
        "Ausgehende Kanten": [G.out_degree(n) for n in G.nodes()],
        "Community": [community_map[n] for n in G.nodes()]
    })

    df["Rang"] = df["PageRank"].rank(
        ascending=False,
        method="dense"
    ).astype(int)

    df = df.sort_values("Rang")

    return fig, df.to_dict("records"),


# Layout
app.layout = layout.layout()


@callback(
    Output("download-csv", "data"),
    Input("export_csv", "n_clicks"),
    State("node-table", "data"),
    prevent_initial_call=True
)
def export_table_to_csv(table_data):
    if not table_data:
        return None

    df = pd.DataFrame(table_data)
    today = datetime.strftime(datetime.now(), "%Y-%m-%d")

    return dcc.send_data_frame(
        df.to_csv,
        f"Wikipedia-Page-Rank-Uebersicht-{today}.csv",
        index=False,
        encoding="utf-8"
    )


@app.callback(
    Output("graph", "figure"),
    Output("node-table", "data"),
    Input("submit", "n_clicks"),
    Input("submit_depth", "n_clicks"),
    Input("submit_edge", "n_clicks"),
    State("input-topic", "value"),
    State("depth", "value"),
    State("edge", "value"),
)
def update(n_clicks, d_clicks, e_clicks, topic, depth, edge_number):
    if n_clicks == 0:
        return empty_figure(), []
    depth = depth or 2
    edge_number = edge_number or 10
    return build_layout(n_clicks, topic, depth,  edge_number)


@app.callback(
    Output("node-table", "style_data_conditional"),
    Input("node-table", "data"),
    prevent_initial_call=True
)
def color_names_by_community(rows):
    if not rows:
        return []

    communities = sorted({row["Community"] for row in rows})

    colors = sample_colorscale(
        "Bluered", # or RdBu, Viridis, Bluered, Plasma
        [i / max(len(communities) - 1, 1) for i in range(len(communities))]
    )

    styles = []
    for comm, color in zip(communities, colors):
        styles.append({
            "if": {
                "filter_query": f"{{Community}} = {comm}",
                "column_id": "Name"
            },
            "color": color,
            "fontWeight": "500"
        })

    return styles


if __name__ == "__main__":
    app.run(debug=True)
