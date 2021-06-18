import numpy as np

# TODO:
#  1. Write Metrics class in an appropriate way


class Metric:
    def __init__(self, clusters: dict, matrix: np.array):
        self.clusters = clusters
        self.matrix = matrix

    def fit(self, metric: str, approach: str):
        inside_edges = []
        for cluster in self.clusters.keys():
            edges = []
            for v_i in self.clusters[cluster]:
                for v_j in self.clusters[cluster]:
                    edges.append(self.matrix[v_i][v_j])
            inside_edges.append(edges)
        if metric == 'inside':
            if approach == 'avg':
                metrics = [sum(edges) / len(edges) for edges in inside_edges]
                return sum(metrics) / len(metrics)
            elif approach == 'sum':
                metrics = [sum(edges) for edges in inside_edges]
                return sum(metrics) / len(metrics)
        elif metric == 'outside':
            if approach == 'avg':
                all_edges_sum = sum([sum(row) for row in self.matrix])
                inside_edges_sum = sum([sum(row) for row in inside_edges])
                all_edges_number = self.matrix.shape[0] * (self.matrix.shape[0] + 1) / 2
                inside_edges_number = sum([len(row) for row in inside_edges]) / 2
                # print(all_edges_sum, inside_edges_sum, all_edges_number, inside_edges_number)
                return (all_edges_sum - inside_edges_sum) / (all_edges_number - inside_edges_number)
            elif approach == 'sum':
                pass
