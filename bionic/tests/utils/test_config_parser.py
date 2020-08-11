import json
from pathlib import Path

import pytest
from bionic.utils.config_parser import ConfigParser

config_path = Path(__file__).resolve().parents[1] / "config" / "mock_config.json"
with config_path.open() as f:
    mock_config = json.load(f)

config_no_outname_path = (
    Path(__file__).resolve().parents[1] / "config" / "mock_config_no_outname.json"
)
with config_no_outname_path.open() as f:
    mock_config_no_outname = json.load(f)


class TestConfigParser:
    def test_config_is_properly_imported(self):
        cp = ConfigParser(config_path)
        assert isinstance(cp.config, dict)

    def test_out_name_is_provided(self):
        mock_config_without_outname = mock_config.copy()
        del mock_config_without_outname["out_name"]
        with pytest.raises(ValueError):
            cp = ConfigParser(mock_config_without_outname)

    def test_out_name_equals_what_is_provided(self):
        cp = ConfigParser(mock_config)
        assert cp.config["out_name"] == Path("path/to/out_name")

    def test_out_name_adopts_config_file_name(self):
        cp = ConfigParser(config_no_outname_path)
        assert (
            cp.config["out_name"]
            == config_no_outname_path.parent / config_no_outname_path.stem
        )

    def test_names_field_is_provided(self):
        mock_config_without_names = mock_config.copy()
        del mock_config_without_names["names"]
        with pytest.raises(ValueError):
            cp = ConfigParser(mock_config_without_names)

    def test_asterisk_in_names_fetches_filenames(self):
        mock_config_asterisk_names = mock_config.copy()
        mock_config_asterisk_names["names"] = "*"
        cp = ConfigParser(mock_config_asterisk_names)
        params = cp.parse()
        assert isinstance(params.names, list)

    def test_config_fields_replace_defaults(self):
        cp = ConfigParser(mock_config)
        params = cp.parse()
        assert params.batch_size == 4096
        assert not params.save_model
        assert params.sample_size == 2
        assert params.gat_shapes["n_layers"] == 3
