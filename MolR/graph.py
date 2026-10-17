from dataclasses import dataclass

import torch

attribute_names = ['element', 'charge', 'aromatic', 'hcount']


@dataclass
class Graph:
    """Bi-directed graph with self-loops. Edges run src -> dst; node_counts holds nodes per batched graph."""
    feature: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    node_counts: torch.Tensor

    def num_nodes(self):
        return self.feature.shape[0]

    def to(self, device):
        return Graph(*(t.to(device) for t in (self.feature, self.src, self.dst, self.node_counts)))

    @staticmethod
    def from_dgl(graph):
        src, dst = graph.edges()
        return Graph(graph.ndata['feature'], src.long(), dst.long(), graph.batch_num_nodes())


def networkx_to_graph(raw_graph, feature_encoder):
    n = len(raw_graph.nodes)
    features = [[feature_encoder[j].get(raw_graph.nodes[i][j], feature_encoder[j]['unknown']) for j in attribute_names]
                for i in range(n)]
    edges = {(s, t) for s, t in raw_graph.edges} | {(t, s) for s, t in raw_graph.edges}
    edges |= {(i, i) for i in range(n)}
    src, dst = zip(*sorted(edges))
    return Graph(torch.tensor(features), torch.tensor(src), torch.tensor(dst), torch.tensor([n]))


def batch_graphs(graphs):
    counts = torch.cat([g.node_counts for g in graphs])
    sizes = torch.tensor([g.num_nodes() for g in graphs])
    offsets = torch.cumsum(sizes, 0) - sizes
    return Graph(
        torch.cat([g.feature for g in graphs]),
        torch.cat([g.src + o for g, o in zip(graphs, offsets.tolist())]),
        torch.cat([g.dst + o for g, o in zip(graphs, offsets.tolist())]),
        counts,
    )
