import wikipediaapi
import networkx as nx


def get_wikipedia_data(start_node: str, depth: int, edge_number: int):
    wiki = wikipediaapi.Wikipedia(
        user_agent="PolitikAnalyseBot (example@example.com)",
        language="de"
    )

    graph = nx.DiGraph()

    def crawl(page_title: str, current_depth: int):
        if current_depth > depth:
            return

        page = wiki.page(page_title)
        if not page.exists():
            return

        links = list(page.links.keys())[:edge_number]

        for link in links:
            graph.add_edge(page_title, link)
            crawl(link, current_depth + 1)

    crawl(page_title=start_node, current_depth=1)

    pagerank_scores = nx.pagerank(graph)
    return graph, pagerank_scores
