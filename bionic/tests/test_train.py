import json
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
from bionic.train import Trainer
from bionic.utils.common import Device

config_path = Path(__file__).resolve().parents[0] / "config" / "mock_config.json"
with config_path.open() as f:
    mock_config = json.load(f)


class TestTrain:
    def test_trainer_completes(self):
        trainer = Trainer(mock_config)
        trainer.train()
        trainer.forward()

    def test_trainer_completes_on_cpu(self):
        Device._device = "cpu"
        assert Device() == "cpu"
        trainer = Trainer(mock_config)
        trainer.train()
        trainer.forward()

    def test_connected_regions_are_similar(self):
        """Tests that connected nodes have more similar features than disconnected nodes.

        In the mock datasets, the input graphs connect pairs of sequentially labelled nodes,
        i.e. A <-> B, C <-> D, E <-> F, etc. Here, A and B should have similar features,
        whereas A and C should not.
        """
        mock_features_path = (
            Path(__file__).resolve().parents[0] / "outputs" / "mock_integration_features.tsv"
        )
        mock_features = pd.read_csv(mock_features_path, sep="\t", index_col=0)
        abc_features = mock_features.loc[["A", "B", "C"]].values

        dot = abc_features @ abc_features.T
        assert dot[0, 1] > dot[0, 2]  # A dot B > A dot C
        assert dot[0, 1] > dot[1, 2]  # A dot B > B dot C
