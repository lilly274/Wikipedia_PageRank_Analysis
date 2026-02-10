from dash import html, dcc, dash_table
from dash.html import Div


def layout() -> Div:
    return html.Div(
        style=_create_sytle(),
        children=[
            _create_header(),
            _create_graph(),
            _create_table()
        ]
    )


def _create_sytle() -> dict[str, str]:
    return {
            "fontFamily": "Inter, system-ui, -apple-system, BlinkMacSystemFont",
            "backgroundColor": "#f5f7fb",
            "padding": "40px"
        }


def _create_header() -> Div:
    return html.Div(
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
                            "font-size": "18px",
                            "color": "#555",
                            "maxWidth": "1200px",
                            "lineHeight": "1.6"
                        }
                    ),
                    html.P(
                        "Die praphische Visualisierung wird durch eine tabllarische Sicht relevanter Metriken ergänzt. "
                        "Zu diesen gehören: Rang, Bezeichnung, Pagerank, eingehende Kanten und ausgehende Kanten",
                        style={
                            "font-size": "18px",
                            "color": "#555",
                            "maxWidth": "1200px",
                            "lineHeight": "1.6"
                        }
                    )
                ]
            )


def _create_graph() -> Div:
    return html.Div(
                style={
                    "background": "white",
                    "padding": "24px 28px",
                    "borderRadius": "14px",
                    "boxShadow": "0 10px 25px rgba(0,0,0,0.06)",
                    "marginBottom": "30px"
                },
                children=[

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
                    dcc.Graph(
                        id="graph",
                        style={"height": "650px"}
                    )
                ]
            )


def _create_table() -> Div:
    return html.Div(
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
                            "padding": "10px 12px",
                            "fontSize": "13px",
                            "fontFamily": "Inter, system-ui",
                            "color": "#1f2937",
                            "border": "none",
                        },
                        style_header={
                            "backgroundColor": "#f3f4f6",
                            "fontWeight": "600",
                            "color": "#374151",
                            "borderBottom": "1px solid #e5e7eb",
                            "textTransform": "uppercase",
                            "fontSize": "12px",
                            "letterSpacing": "0.04em"
                        },
                        style_data={
                            "borderBottom": "1px solid #f1f1f1"
                        },
                        style_data_conditional=[

                        ],
                    ),
                    dcc.Download(id="download-csv")
                ]
            )