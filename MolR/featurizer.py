import torch
import pickle
import pysmiles
import numpy as np
from dataclasses import dataclass
from importlib.resources import files
from MolR.model import GNN
from MolR.graph import batch_graphs, networkx_to_graph


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
        return networkx_to_graph(raw_graph, feature_encoder), None
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
        encoder = load_feature_encoder(self.model_name)
        results = [build_graph_result(i, smiles, encoder) for i, smiles in enumerate(smiles_list)]
        parsed_results = [r for r in results if r.graph is not None]
        all_embeddings = np.zeros((len(smiles_list), self.dim), dtype=float)
        flags = np.zeros(len(smiles_list), dtype=bool)
        embeddings, parsed = self.transform_graphs(parsed_results, batch_size)
        if parsed:
            all_embeddings[parsed, :] = embeddings
            flags[parsed] = True
        return all_embeddings, flags

    def transform_graphs(self, graph_records, batch_size=None):
        graphs = [record.graph for record in graph_records]
        parsed = [record.index for record in graph_records]
        batch_size = batch_size or len(graphs)
        embeddings = []
        with torch.no_grad():
            self.mole.eval()
            for i in range(0, len(graphs), batch_size):
                batch = batch_graphs(graphs[i:i + batch_size]).to(self.device)
                embeddings.append(self.mole(batch))
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
