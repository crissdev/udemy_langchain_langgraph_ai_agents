from langchain_core.runnables.graph import Graph


def visualize_graph(graph: Graph, filename: str):
    with open(filename, mode="wb") as f:
        f.write(graph.draw_mermaid_png(background_color="white"))
        print(f"Graph written to {filename}")
