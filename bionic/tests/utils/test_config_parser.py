import json
from pathlib import Path

import pytest
from bionic.utils.config_parser import ConfigParser

config_path = Path(__file__).resolve().parents[1] / "config" / "mock_config.json"
with open(config_path, "r") as f:
    mock_config = json.load(f)


class TestConfigParser:
    def test_config_is_properly_imported(self):
        cp = ConfigParser(mock_config, "out_name")
        assert isinstance(cp.config, dict)

    def test_out_name_is_provided(self):
        with pytest.raises(Exception):
            cp = ConfigParser(mock_config, None)

    def test_out_name_equals_what_is_provided(self):
        cp = ConfigParser(mock_config, "out_name")
        assert cp._defaults["out_name"] == "out_name"

    def test_names_field_is_provided(self):
        mock_config_without_names = mock_config.copy()
        del mock_config_without_names["names"]
        with pytest.raises(Exception):
            cp = ConfigParser(mock_config_without_names, "out_name")

    def test_asterisk_in_names_fetches_filenames(self):
        mock_config_asterisk_names = mock_config.copy()
        mock_config_asterisk_names["names"] = "*"
        cp = ConfigParser(mock_config_asterisk_names, "out_name")
        params = cp.parse()
        assert isinstance(params.names, list)

    def test_config_fields_replace_defaults(self):
        cp = ConfigParser(mock_config, "out_name")
        params = cp.parse()
        assert params.batch_size == 4096
        assert not params.save_model
        assert params.sample_size == 2
        assert params.gat_shapes["n_layers"] == 3
