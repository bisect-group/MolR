from pathlib import Path

import numpy as np
import pytest

from MolR.featurizer import MolEFeaturizer, batch_graphs, build_graph_result, load_feature_encoder

REFERENCE = np.load(Path(__file__).parent / 'reference_embeddings.npz')
MODELS = ['gcn_1024', 'gat_1024', 'sage_1024', 'tag_1024']


@pytest.mark.parametrize('model_name', MODELS)
def test_embeddings_match_dgl_reference(model_name):
    encoder = load_feature_encoder(model_name)
    graphs = [build_graph_result(i, s, encoder).graph for i, s in enumerate(REFERENCE['smiles'])]
    featurizer = MolEFeaturizer(model=model_name, gpu=None)
    featurizer.mole.eval()
    expected = REFERENCE[model_name]
    embeddings = featurizer.mole(batch_graphs(graphs)).detach().numpy()
    # float32 sum pooling over 1024 dims: compare relative to the largest value
    assert np.abs(embeddings - expected).max() < 1e-5 * np.abs(expected).max()
    single = featurizer.mole(graphs[3]).detach().numpy()
    assert np.abs(single - expected[3]).max() < 1e-5 * np.abs(expected).max()
