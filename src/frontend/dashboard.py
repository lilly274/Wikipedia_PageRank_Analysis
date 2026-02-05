from datetime import datetime

import dash
import pandas as pd
from dash import dcc, html, Input, Output, State
from dash import dash_table, callback
import plotly.graph_objects as go
import requests
import networkx as nx
import networkx.algorithms.community as community_graph
import math
import numpy as np

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
    if n_clicks == 0:
        return empty_figure(), []

    # ---------- API ----------
    response = requests.get(
        "http://127.0.0.1:8000/analyze",
        params={
            "topic": topic,
            "depth": depth,
            "edge_number": edge_number,
        }
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
            colorscale="Viridis",
            line=dict(width=1, color="black")
        )
    )

    # ---------- EDGES ----------
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

app.layout = html.Div(
    style={
        "fontFamily": "Inter, system-ui, -apple-system, BlinkMacSystemFont",
        "backgroundColor": "#f5f7fb",
        "padding": "40px"
    },
    children=[

        # ---------- HEADER ----------
        html.Div(
            style={
                "padding": "30px 40px",
                "borderRadius": "14px",
                "marginBottom": "30px"
            },
            children=[
                html.H1(
                    "Wikipedia PageRank Analyse",
                    style={
                        "marginBottom": "10px",
                        "fontWeight": "600",
                        "letterSpacing": "-0.5px"
                    }
                ),
                html.P(
                    "Diese Visualisierung zeigt die relative Bedeutung von Artikeln "
                    "in einem gerichteten Wikipedia-Netzwerk auf Basis des PageRank-Algorithmus. "
                    "Ziel ist es, reputationsbasierte Wichtigkeit und strukturelle Rollen sichtbar zu machen.",
                    style={
                        "color": "#555",
                        "maxWidth": "900px",
                        "lineHeight": "1.6"
                    }
                )
            ]
        ),

        # ---------- GRAPH ----------
        html.Div(
            style={
                "background": "white",
                "padding": "24px 28px",
                "borderRadius": "14px",
                "boxShadow": "0 10px 25px rgba(0,0,0,0.06)",
                "marginBottom": "30px"
            },
            children=[

                # ---------- TOP BAR ----------
                # ---------- TOP BAR ----------
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "minmax(320px, 1.4fr) minmax(280px, 1fr)",
                        "gap": "24px",
                        "alignItems": "start",
                        "marginBottom": "20px"
                    },
                    children=[

                        # LEFT: FILTER
                        html.Div(
                            children=[
                                html.Label(
                                    "Filter",
                                    style={
                                        "fontWeight": "500",
                                        "marginBottom": "6px",
                                        "display": "block"
                                    }
                                ),
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "gap": "12px"
                                    },
                                    children=[
                                        dcc.Input(
                                            id="input-topic",
                                            type="text",
                                            value="Bündnis 90/Die Grünen",
                                            style={
                                                "flex": "1",
                                                "minWidth": "0",  # 🔑 extrem wichtig
                                                "padding": "11px 12px",
                                                "borderRadius": "10px",
                                                "border": "1px solid #ddd",
                                                "fontSize": "14px"
                                            }
                                        ),
                                        html.Button(
                                            "Analyse starten",
                                            id="submit",
                                            n_clicks=0,
                                            style={
                                                "padding": "11px 18px",
                                                "borderRadius": "10px",
                                                "border": "none",
                                                "background": "linear-gradient(135deg, #4f46e5, #6366f1)",
                                                "color": "white",
                                                "fontWeight": "600",
                                                "cursor": "pointer",
                                                "whiteSpace": "nowrap"
                                            }
                                        ),
                                    ]
                                )
                            ]
                        ),

                        # RIGHT: DEPTH + EDGE
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "1fr 1fr",
                                "gap": "16px"
                            },
                            children=[

                                # DEPTH
                                html.Div(children=[
                                    html.Label(
                                        "Knotentiefe",
                                        style={
                                            "fontWeight": "500",
                                            "marginBottom": "6px",
                                            "display": "block"
                                        }
                                    ),
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "gap": "8px"
                                        },
                                        children=[
                                            dcc.Input(
                                                id="depth",
                                                type="number",
                                                value=2,
                                                min=1,
                                                step=1,
                                                style={
                                                    "flex": "1",
                                                    "minWidth": "0",
                                                    "padding": "11px 12px",
                                                    "borderRadius": "10px",
                                                    "border": "1px solid #ddd",
                                                    "fontSize": "14px"
                                                }
                                            ),
                                            html.Button(
                                                "Start",
                                                id="submit_depth",
                                                n_clicks=0,
                                                style={
                                                    "padding": "11px 14px",
                                                    "borderRadius": "10px",
                                                    "border": "1px solid #e5e7eb",
                                                    "background": "#f8fafc",
                                                    "color": "#374151",
                                                    "fontWeight": "500",
                                                    "cursor": "pointer",
                                                    "whiteSpace": "nowrap"
                                                }
                                            ),
                                        ]
                                    )
                                ]),

                                # EDGE
                                html.Div(children=[
                                    html.Label(
                                        "Max. Kanten",
                                        style={
                                            "fontWeight": "500",
                                            "marginBottom": "6px",
                                            "display": "block"
                                        }
                                    ),
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "gap": "8px"
                                        },
                                        children=[
                                            dcc.Input(
                                                id="edge",
                                                type="number",
                                                value=10,
                                                min=1,
                                                step=1,
                                                style={
                                                    "flex": "1",
                                                    "minWidth": "0",
                                                    "padding": "11px 12px",
                                                    "borderRadius": "10px",
                                                    "border": "1px solid #ddd",
                                                    "fontSize": "14px"
                                                }
                                            ),
                                            html.Button(
                                                "Start",
                                                id="submit_edge",
                                                n_clicks=0,
                                                style={
                                                    "padding": "11px 14px",
                                                    "borderRadius": "10px",
                                                    "border": "1px solid #e5e7eb",
                                                    "background": "#f8fafc",
                                                    "color": "#374151",
                                                    "fontWeight": "500",
                                                    "cursor": "pointer",
                                                    "whiteSpace": "nowrap"
                                                }
                                            ),
                                        ]
                                    )
                                ]),
                            ]
                        )
                    ]
                ),

                # ---------- GRAPH ----------
                dcc.Graph(
                    id="graph",
                    style={"height": "650px"}
                )
            ]
        ),

        # ---------- TABLE ----------
        html.Div(
            style={
                "background": "white",
                "padding": "25px",
                "borderRadius": "14px",
                "boxShadow": "0 10px 25px rgba(0,0,0,0.06)"
            },
            children=[
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "marginBottom": "15px"
                    },
                    children=[
                        html.H4(
                            "Knotenübersicht",
                            style={"margin": "0"}
                        ),
                        html.Button(
                            "Export CSV",
                            id="export_csv",
                            n_clicks=0,
                            style={
                                "padding": "11px 14px",
                                "borderRadius": "10px",
                                "border": "1px solid #e5e7eb",
                                "background": "#f8fafc",
                                "color": "#374151",
                                "fontWeight": "500",
                                "cursor": "pointer",
                                "whiteSpace": "nowrap"
                            }
                        )
                    ]
                ),
                dash_table.DataTable(
                    id="node-table",
                    columns=[
                        {"name": "Rang", "id": "Rang"},
                        {"name": "Name", "id": "Name"},
                        {
                            "name": "PageRank",
                            "id": "PageRank",
                            "type": "numeric",
                            "format": {"specifier": ".4f"}
                        },
                        {"name": "Eingehende Kanten", "id": "Eingehende Kanten"},
                        {"name": "Ausgehende Kanten", "id": "Ausgehende Kanten"}
                    ],
                    style_table={
                        "width": "100%",
                        "overflowX": "auto"
                    },
                    style_cell={
                        "textAlign": "center",
                        "padding": "10px",
                        "fontSize": "13px"
                    },
                    style_header={
                        "backgroundColor": "#f0f2f8",
                        "fontWeight": "600",
                        "border": "none"
                    },
                    style_data={
                        "borderBottom": "1px solid #eee"
                    }
                ),
                dcc.Download(id="download-csv")
            ]
        )
    ]
)

@callback(
    Output("download-csv", "data"),
    Input("export_csv", "n_clicks"),
    State("node-table", "data"),
    prevent_initial_call=True
)
def export_table_to_csv(n_clicks, table_data):
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

if __name__ == "__main__":
    app.run(debug=True)
