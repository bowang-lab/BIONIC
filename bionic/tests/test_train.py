import json
from pathlib import Path

import torch
import pytest
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from bionic.train import Trainer
from bionic.utils.common import Device

torch.manual_seed(42 * 42 - 42 + 4 * 2)
config_path = Path(__file__).resolve().parents[0] / "config" / "mock_config.json"
with config_path.open() as f:
    mock_config = json.load(f)


class TestTrain:
    def test_trainer_completes(self):
        trainer = Trainer(mock_config)
        trainer.train()
        trainer.forward()

    def test_network_batching_completes(self):
        mock_config_with_batching = mock_config.copy()
        mock_config_with_batching["sample_size"] = 1
        trainer = Trainer(mock_config_with_batching)
        trainer.train()
        trainer.forward()

    def test_one_layer_completes(self):
        mock_config_one_layer = mock_config.copy()
        mock_config_one_layer["gat_shapes"]["n_layers"] = 1
        trainer = Trainer(mock_config_one_layer)
        trainer.train()
        trainer.forward()

    def test_two_layer_completes(self):
        mock_config_two_layer = mock_config.copy()
        mock_config_two_layer["gat_shapes"]["n_layers"] = 2
        trainer = Trainer(mock_config_two_layer)
        trainer.train()
        trainer.forward()

    def test_svd_approximation_completes(self):
        mock_config_svd = mock_config.copy()
        mock_config_svd["svd_dim"] = 3
        trainer = Trainer(mock_config_svd)
        trainer.train()
        trainer.forward()

    def test_shared_encoder_completes(self):
        mock_config_shared_encoder = mock_config.copy()
        mock_config_shared_encoder["shared_encoder"] = True
        trainer = Trainer(mock_config_shared_encoder)
        trainer.train()
        trainer.forward()

    def test_trainer_completes_on_cpu(self):
        Device._device = "cpu"
        assert Device() == "cpu"
        trainer = Trainer(mock_config)
        trainer.train()
        trainer.forward()

    def test_supervised_trainer_completes(self):
        mock_config_with_labels = mock_config.copy()
        mock_config_with_labels["label_names"] = ["bionic/tests/inputs/mock_labels.json"]
        mock_config_with_labels["lambda"] = 0.95
        trainer = Trainer(mock_config_with_labels)
        trainer.train()
        trainer.forward()

    def test_supervised_trainer_with_multiple_labels_completes(self):
        mock_config_with_multiple_labels = mock_config.copy()
        mock_config_with_multiple_labels["label_names"] = [
            "bionic/tests/inputs/mock_labels.json",
            "bionic/tests/inputs/mock_labels.json",
        ]
        mock_config_with_multiple_labels["lambda"] = 0.95
        trainer = Trainer(mock_config_with_multiple_labels)
        trainer.train()
        trainer.forward()

    def test_supervised_trainer_with_network_batching_completes(self):
        mock_config_with_labels_and_batching = mock_config.copy()
        mock_config_with_labels_and_batching["label_names"] = [
            "bionic/tests/inputs/mock_labels.json"
        ]
        mock_config_with_labels_and_batching["lambda"] = 0.95
        mock_config_with_labels_and_batching["sample_size"] = 1
        trainer = Trainer(mock_config_with_labels_and_batching)
        trainer.train()
        trainer.forward()

    @pytest.fixture(autouse=True)
    def test_connected_regions_are_similar(self):
        """Tests that connected nodes have more similar features than disconnected nodes.

        In the mock datasets, the input graphs consist of two, four-node cliques,
        i.e. A, B, C, D form a clique and E, F, G, H form a clique. Here, A and B should
        have similar features, whereas A and E, and B and E, should not. This test runs after
        each of the above tests, ensuring both error free `Trainer` execution and error free
        embeddings.
        """
        yield
        mock_features_path = (
            Path(__file__).resolve().parents[0] / "outputs" / "mock_integration_features.tsv"
        )
        mock_features = pd.read_csv(mock_features_path, sep="\t", index_col=0)
        print(mock_features)
        abc_features = mock_features.loc[["A", "B", "E"]].values

        dot = 1 - squareform(pdist(abc_features, metric="cosine"))  # pairwise cosine similarity
        print(dot)
        assert dot[0, 1] > dot[0, 2]  # cos_sim(A, B) > cos_sim(A, E)
        assert dot[0, 1] > dot[1, 2]  # cos_sim(A, B) > cos_sim(B, E)
