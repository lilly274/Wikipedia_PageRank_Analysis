'''
import dash
import pandas as pd
from dash import dcc, html, Input, Output, State
from dash import dash_table
import plotly.graph_objects as go
import requests
import networkx as nx
import networkx.algorithms.community as community_graph
import math

app = dash.Dash(__name__)
G = nx.DiGraph()
pagerank = nx.pagerank(G, alpha=0.85)

def build_figure(n_clicks, topic):
    if n_clicks == 0:
        return go.Figure()

    response = requests.get(
        "http://127.0.0.1:8000/analyze",
        params={"topic": topic}
    )
    data = response.json()

    # -------- GRAPH --------
    for edge in data["edges"]:
        G.add_edge(edge["source"], edge["target"])

    pos = nx.spring_layout(G, k=0.8, seed=42)
    communities = list(community_graph.greedy_modularity_communities(G))

    # -------- EDGES --------
    edge_x = []
    edge_y = []

    for edge in data["edges"]:
        x0, y0 = pos[edge["source"]]
        x1, y1 = pos[edge["target"]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="rgba(150,150,150,0.4)"),
        hoverinfo="none"
    )

    # -------- NODES --------
    x_nodes = []
    y_nodes = []
    labels = []
    raw_sizes = []

    for node in data["nodes"]:
        x, y = pos[node["id"]]
        x_nodes.append(x)
        y_nodes.append(y)
        labels.append(node["id"])

        # logarithmische Dämpfung
        raw_sizes.append(math.log(node["pagerank"] + 1e-6))

    # Determine Colors for communities
    node_x, node_y, node_color = [], [], []
    for i, comm in enumerate(communities):
        for node in comm:
            node_x.append(pos[node][0])
            node_y.append(pos[node][1])
            node_color.append(i)

    # -------- SIZE NORMALIZATION --------
    min_size = 10
    max_size = 50

    min_pr = min(raw_sizes)
    max_pr = max(raw_sizes)

    sizes = [
        min_size + (s - min_pr) / (max_pr - min_pr) * (max_size - min_size)
        for s in raw_sizes
    ]

    node_trace = go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode="markers+text",
        text=labels,
        marker=dict(
            size=sizes,
            color=node_color,
            colorscale='Viridis',
            line=dict(width=1, color="black")
        ),
        textposition="top center"
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig

#####################################################

df = pd.DataFrame({
    "Name": list(G.nodes()),
    "PageRank": [pagerank[n] for n in G.nodes()]
})

#####################################################

app.layout = html.Div([
    html.H1("Wikipedia PageRank Analyse"),
    html.Plaintext(
        "Diese Visualisierung zeigt die relative Bedeutung von Knoten in einem gerichteten Netzwerk anhand des PageRank-Algorithmus am Beispiel von Wikipedia. Ziel ist es,\n"
        "reputationsbasierte Wichtigkeit sichtbar zu machen und strukturelle Rollen im Netzwerk zu identifizieren."),
    dcc.Input(
        id="input-topic",
        type="text",
        value="Bündnis 90/Die Grünen"
    ),
    html.Button("Analyse starten", id="submit", n_clicks=0),

    dcc.Graph(id="graph"),

    html.H4("Knotenübersicht"),
    dash_table.DataTable(
        id="node-table",
        columns=[{"Name": c, "PageRank": c} for c in df.columns],
        data=df.to_dict("records"),
        style_table={"width": "60%"}
    )
])

@app.callback(
    Output("graph", "figure"),
    Output("node-table", "data"),
    Input("submit", "n_clicks"),
    State("input-topic", "value")
)
def update(n_clicks, topic):
    fig = build_figure(n_clicks, topic)
    return fig, df.to_dict("records")

if __name__ == "__main__":
    app.run(debug=True)
'''

import dash
import pandas as pd
from dash import dcc, html, Input, Output, State
from dash import dash_table
import plotly.graph_objects as go
import requests
import networkx as nx
import networkx.algorithms.community as community_graph
import math
import numpy as np

app = dash.Dash(__name__)

def build_layout(n_clicks, topic):
    if n_clicks == 0:
        return go.Figure(), []

    # ---------- API ----------
    response = requests.get(
        "http://127.0.0.1:8000/analyze",
        params={"topic": topic}
    )
    data = response.json()

    # ---------- GRAPH ----------
    G = nx.DiGraph()
    for edge in data["edges"]:
        G.add_edge(edge["source"], edge["target"])

    pagerank = nx.pagerank(G, alpha=0.85)
    pos = nx.spring_layout(G, seed=42)
    communities = list(community_graph.greedy_modularity_communities(G))

    cluster_centers = {
        i: np.random.uniform(-1, 1, size=2)
        for i in range(len(communities))
    }

    # ---------- NODES ----------
    node_x, node_y, labels, raw_sizes, node_color = [], [], [], [], []

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
        node_x.append(x)
        node_y.append(y)
        labels.append(node)
        raw_sizes.append(pagerank[node])
        node_color.append(community_map.get(node, 0))

    # Size normalization
    min_size, max_size = 6, 22
    min_pr, max_pr = min(raw_sizes), max(raw_sizes)
    sizes = [
        min_size + (s - min_pr) / (max_pr - min_pr) * (max_size - min_size)
        for s in raw_sizes
    ]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hovertext=labels,
        hoverinfo="text",
        text=labels,
        marker=dict(
            size=sizes,
            color=node_color,
            colorscale="Viridis",
            line=dict(width=1, color="black")
        )
    )

    # ---------- EDGES ----------
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="rgba(150,150,150,0.4)"),
        hoverinfo="none"
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # ---------- TABLE ----------
    df = pd.DataFrame({
        "Name": list(G.nodes()),
        "PageRank": [pagerank[n] for n in G.nodes()],
        "Eingehende Kanten": [G.in_degree(n) for n in G.nodes()],
        "Ausgehende Kanten": [G.out_degree(n) for n in G.nodes()]
    })

    df["Rang"] = df["PageRank"].rank(
        ascending=False,
        method="dense"
    ).astype(int)

    df = df.sort_values("Rang")

    return fig, df.to_dict("records")

#--------------------------------------------------------------------
app.layout = html.Div([
    html.H1("Wikipedia PageRank Analyse"),
    html.Plaintext(
        "Diese Visualisierung zeigt die relative Bedeutung von Knoten in einem gerichteten Netzwerk anhand des PageRank-Algorithmus am Beispiel von Wikipedia. Ziel ist es,\n"
        "reputationsbasierte Wichtigkeit sichtbar zu machen und strukturelle Rollen im Netzwerk zu identifizieren."
    ),
    dcc.Input(
        id="input-topic",
        type="text",
        value="Bündnis 90/Die Grünen",
        style={"width": "400px"}
    ),
    html.Button("Analyse starten", id="submit", n_clicks=0),

    dcc.Graph(id="graph"),

    html.H4("Knotenübersicht"),
    dash_table.DataTable(
        id="node-table",
        columns=[
            {"name": "Rang", "id": "Rang"},
            {"name": "Name", "id": "Name"},
            {"name": "PageRank", "id": "PageRank", "type": "numeric", "format": {"specifier": ".4f"}},
            {"name": "Eingehende Kanten", "id": "Eingehende Kanten"},
            {"name": "Ausgehende Kanten", "id": "Ausgehende Kanten"}
        ],
        style_table={"width": "80%"},
        style_cell={"textAlign": "center"}
    )
])

@app.callback(
    Output("graph", "figure"),
    Output("node-table", "data"),
    Input("submit", "n_clicks"),
    State("input-topic", "value")
)
def update(n_clicks, topic):
    return build_layout(n_clicks, topic)

if __name__ == "__main__":
    app.run(debug=True)
