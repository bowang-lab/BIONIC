import json
from pathlib import Path

import torch
import pytest
import numpy as np
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
        mock_config_svd["svd_dim"] = 7
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
        i.e. A, B, C, D form a clique and E, F, G, H form a clique. Here, the average pairwise
        similarity within each clique should be greater than between cliques. This test runs
        after each of the above tests, ensuring both error free `Trainer` execution and error 
        free embeddings.
        """
        yield
        mock_features_path = (
            Path(__file__).resolve().parents[0] / "outputs" / "mock_integration_features.tsv"
        )
        mock_features = pd.read_csv(mock_features_path, sep="\t", index_col=0)

        dot = 1 - squareform(pdist(mock_features, metric="cosine"))  # pairwise cosine similarity
        within_clus_1 = np.mean(dot[:4, :4][np.triu_indices(4, k=1)])
        within_clus_2 = np.mean(dot[4:, 4:][np.triu_indices(4, k=1)])
        between_clus = np.mean(dot[:4, 4:])

        assert within_clus_1 > between_clus
        assert within_clus_2 > between_clus
