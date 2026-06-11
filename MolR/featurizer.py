import dgl
import torch
import pickle
import pysmiles
import numpy as np
from dataclasses import dataclass
from importlib.resources import files
from MolR.model import GNN
from dgl.dataloading import GraphDataLoader
from MolR.data_processing import networkx_to_dgl


def model_resource_path(model):
    return str(files('MolR').joinpath('models', model)) + '/'


def cuda_device_for_gpu(gpu):
    if gpu is None or str(gpu).lower() in {'', 'none', 'cpu'}:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device(f'cuda:{int(gpu)}')
    return torch.device('cpu')


def load_feature_encoder(model):
    path = model_resource_path(model)
    with open(path + 'feature_enc.pkl', 'rb') as f:
        return pickle.load(f)


@dataclass
class GraphBuildResult:
    index: int
    smiles: str
    graph: object | None
    error: str | None = None


def smiles_to_graph(smiles, feature_encoder):
    try:
        raw_graph = pysmiles.read_smiles(
            smiles,
            zero_order_bonds=False,
            infer_bare_bracket_hcount=True,
            ignore_invalid_ez=True,
        )
        return networkx_to_dgl(raw_graph, feature_encoder), None
    except Exception as exc:
        return None, f"parse failed: {type(exc).__name__}: {exc}"


def build_graph_result(index, smiles, feature_encoder):
    graph, error = smiles_to_graph(str(smiles), feature_encoder)
    return GraphBuildResult(
        index=index,
        smiles=str(smiles),
        graph=graph,
        error=error,
    )


class GraphDataset(dgl.data.DGLDataset):
    def __init__(self, model, smiles_list, gpu):
        self.path = model_resource_path(model)
        self.smiles_list = smiles_list
        self.gpu = gpu
        self.parsed = []
        self.graphs = []
        self.failures = []
        super().__init__(name='graph_dataset')

    def process(self):
        feature_encoder = load_feature_encoder(self.path.rstrip('/').split('/')[-1])
        for i, smiles in enumerate(self.smiles_list):
            result = build_graph_result(i, smiles, feature_encoder)
            if result.graph is None:
                self.failures.append(result)
                continue
            self.graphs.append(result.graph)
            self.parsed.append(i)
        device = cuda_device_for_gpu(self.gpu)
        if device.type == 'cuda':
            self.graphs = [graph.to(device) for graph in self.graphs]

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)

class MolEFeaturizer(object):
    def __init__(self, model='gcn_1024', gpu=0):
        self.model_name = model
        self.gpu = gpu
        self.path_to_model = model_resource_path(model)

        with open(self.path_to_model + 'hparams.pkl', 'rb') as f:
            hparams = pickle.load(f)
        self.mole = GNN(hparams['gnn'], hparams['layer'], hparams['feature_len'], hparams['dim'])
        self.dim = hparams['dim']

        device = cuda_device_for_gpu(gpu)
        self.device = device
        self.mole.load_state_dict(torch.load(self.path_to_model + 'model.pt', map_location=device, weights_only=True))
        self.mole = self.mole.to(device)

    def transform(self, smiles_list, batch_size=None):
        data = GraphDataset(self.model_name, smiles_list, self.gpu)
        dataloader = GraphDataLoader(data, batch_size=batch_size if batch_size else len(smiles_list))
        all_embeddings = np.zeros((len(smiles_list), self.dim), dtype=float)
        flags = np.zeros(len(smiles_list), dtype=bool)
        res = []
        with torch.no_grad():
            self.mole.eval()
            for graphs in dataloader:
                if self.device.type == 'cuda':
                    graphs = graphs.to(self.device)
                graph_embeddings = self.mole(graphs)
                res.append(graph_embeddings)
            if res:
                res = torch.cat(res, dim=0).cpu().numpy()
                all_embeddings[data.parsed, :] = res
                flags[data.parsed] = True
        return all_embeddings, flags

    def transform_graphs(self, graph_records, batch_size=None):
        graphs = [record.graph for record in graph_records]
        parsed = [record.index for record in graph_records]
        if self.device.type == 'cuda':
            graphs = [graph.to(self.device) for graph in graphs]
        dataloader = GraphDataLoader(graphs, batch_size=batch_size if batch_size else len(graphs))
        embeddings = []
        with torch.no_grad():
            self.mole.eval()
            for graphs_batch in dataloader:
                if self.device.type == 'cuda':
                    graphs_batch = graphs_batch.to(self.device)
                embeddings.append(self.mole(graphs_batch))
        if not embeddings:
            return np.zeros((0, self.dim), dtype=float), []
        return torch.cat(embeddings, dim=0).cpu().numpy(), parsed


def example_usage():
    model = MolEFeaturizer(path_to_model='../saved/gcn_1024')
    embeddings, flags = model.transform(['C', 'CC', 'ccc'])
    print(embeddings)
    print(flags)


if __name__ == '__main__':
    example_usage()
