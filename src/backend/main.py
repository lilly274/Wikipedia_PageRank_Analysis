from fastapi import FastAPI
from .logic import get_wikipedia_data

app = FastAPI(title="Wikipedia PageRank API")

@app.get("/analyze")
def analyze(topic: str = "Deutscher Bundestag"):
    graph, scores = get_wikipedia_data(topic)

    nodes = [
        {"id": node, "pagerank": float(scores[node])}
        for node in graph.nodes()
    ]

    edges = [
        {"source": u, "target": v}
        for u, v in graph.edges()
    ]

    return {"nodes": nodes, "edges": edges}
