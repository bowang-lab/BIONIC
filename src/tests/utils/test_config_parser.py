import sys
import json
import pathlib

path = f"{pathlib.Path(__file__).parent.parent.absolute()}/"
sys.path.append(path)

import pytest
from utils.config_parser import ConfigParser

mock_config = json.load(open(f"{path}config/mock_config.json", "r"))


class TestConfigParser:
    def test_config_is_properly_imported(self):
        cp = ConfigParser(mock_config, "out_name")
        assert isinstance(cp.config, dict)

    def test_out_name_is_provided(self):
        with pytest.raises(Exception):
            cp = ConfigParser(mock_config, None)

    def test_out_name_equals_what_is_provided(self):
        cp = ConfigParser(mock_config, "out_name")
        assert cp.defaults["out_name"] == "out_name"

    def test_names_field_is_provided(self):
        mock_config_without_names = mock_config.copy()
        del mock_config_without_names["names"]
        cp = ConfigParser(mock_config_without_names, "out_name")
        with pytest.raises(Exception):
            cp.parse()

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
