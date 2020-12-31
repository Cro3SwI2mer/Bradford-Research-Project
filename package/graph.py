import numpy as np
import pandas as pd

# TODO:
#  1. Write a Class (into another .py file) which divided on clusters nodes of a weighted graph


class Graph:
    def __init__(self, nodes: list, distance_matrix: np.array) -> None:
        self.nodes = nodes
        self.distance_matrix = distance_matrix

    def __nodes_combinations(self, tuple_format: bool) -> tuple or list:
        args = [self.nodes, self.nodes]
        result = [[]]
        for arg in args:
            result = [x + [y] for x in result for y in arg]
        list_of_all_pairs_of_nodes = [tuple(sorted(elem)) for elem in result if elem[0] != elem[1]]
        if tuple_format:
            return sorted(list(set(list_of_all_pairs_of_nodes)))
        else:
            return [list(elem) for elem in sorted(list(set(list_of_all_pairs_of_nodes)))]

    def dataframe_format(self) -> pd.DataFrame:
        graph = pd.DataFrame(self.distance_matrix.T)
        graph.columns, graph.index = self.nodes, self.nodes
        return graph

    def dict_format(self, method: str) -> dict:
        graph = dict.fromkeys(self.nodes)
        for node in graph.keys():
            vertexes = self.nodes.copy()
            vertexes.remove(node)
            graph[node] = dict.fromkeys(vertexes)
        edges = Graph.__nodes_combinations(self, tuple_format=True)
        graph_df = Graph.dataframe_format(self)
        for edge in edges:
            node_1, node_2 = edge[0], edge[1]
            if method == 'directed':
                graph[node_1][node_2], graph[node_2][node_1] = graph_df[node_1][node_2], graph_df[node_2][node_1]
            elif method == 'min':
                graph[node_1][node_2], graph[node_2][node_1] = min(graph_df[node_1][node_2], graph_df[node_2][node_1]),\
                                                               min(graph_df[node_1][node_2], graph_df[node_2][node_1])
            elif method == 'mean':
                graph[node_1][node_2], graph[node_2][node_1] = (graph_df[node_1][node_2] + graph_df[node_2][node_1])/2,\
                                                               (graph_df[node_1][node_2] + graph_df[node_2][node_1])/2
            elif method == 'max':
                graph[node_1][node_2], graph[node_2][node_1] = max(graph_df[node_1][node_2], graph_df[node_2][node_1]),\
                                                               max(graph_df[node_1][node_2], graph_df[node_2][node_1])
            else:
                raise ValueError("Wrong value of variable 'method'")
        return graph


__all__ = ['Graph']
