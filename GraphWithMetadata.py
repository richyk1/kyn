import rustworkx as rx


class GraphWithMetadata:
    def __init__(self, graph: rx.PyDiGraph, metadata=None):
        self.graph = graph
        self.metadata = metadata or {}

    def __getattr__(self, name):
        return getattr(self.graph, name)
