import json
import pytest
from pathlib import Path
from bionic.utils.preprocessor import Preprocessor

import numpy as np

config_path = Path(__file__).resolve().parents[1] / "config" / "mock_config.json"
with config_path.open() as f:
    mock_config = json.load(f)

# Update `mock_config` "names" parameter to take this file's path into account
mock_config["names"] = [
    Path(__file__).resolve().parents[1] / "inputs" / Path(name).name
    for name in mock_config["names"]
]


class TestPreprocessor:
    def test_final_networks_have_same_nodes_and_node_ordering(self):
        p = Preprocessor(mock_config["names"])
        p.process()
        for G1 in p.graphs:
            for G2 in p.graphs:
                assert (np.array(G1.nodes()) == np.array(G2.nodes())).all()
