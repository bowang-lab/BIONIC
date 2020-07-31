import os
import json
from pathlib import Path
from typing import Dict, Union
from argparse import Namespace


class DefaultConfig:
    def __init__(self):
        """Defines the default BIONIC config parameters.
        """

        self._defaults = {
            "names": None,  # Filenames of input networks
            "out_name": None,  # Name of output feature, model and network weight files
            "delimiter": " ",  # Delimiter for network input files
            "epochs": 3000,  # Number of training epochs
            "batch_size": 2048,  # Number of genes/proteins in each batch
            "sample_size": 0,  # Number of networks to batch at once (0 will batch all networks) TODO
            "learning_rate": 0.0005,  # Adam optimizer learning rate
            "embedding_size": 512,  # Dimensionality of output integrated features
            "svd_dim": 0,  # Dimensionality of network SVD approximation (0 will not perform SVD) TODO
            "initialization": "xavier",  # Method used to initialize BIONIC weights
            "gat_shapes": {
                "dimension": 64,  # Dimension of each GAT layer
                "n_heads": 10,  # Number of attention heads for each GAT layer
                "n_layers": 2,  # Number of GAT layers for each input network
            },
            "save_network_scales": False,  # Whether to save internal learned network feature scaling
            "save_model": True,  # Whether to save the trained model or not
            "load_pretrained_model": False,  # Whether to load a pretrained model TODO
            "use_tensorboard": False,  # Whether to output tensorboard data
            "plot_loss": True,  # Whether to plot loss curves
        }


class ConfigParser(DefaultConfig):
    def __init__(self, config: Union[str, dict], out_name: Union[str, None] = None):
        """Parses and validates a user supplied config

        Args:
            config (Union[str, dict]): Name of config file or preloaded config
                dictionary.
            out_name (Union[str, None], optional): Name to use for BIONIC output files.
                Inferred from config file name if None. Defaults to None.
        """

        super().__init__()
        self._out_name = out_name
        self._defaults["out_name"] = self._out_name
        self._required = {"names"}  # Fields required in config file
        self.config = config

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: Union[str, dict]) -> None:

        # Check if the config is already loaded and validate `out_name` exists if so.
        if isinstance(config, dict):
            if "out_name" not in config:
                if self._out_name is None:
                    raise Exception(
                        "Output file name `out_name` must be provided if `config` is a dictionary"
                    )
            else:
                self._out_name = config["out_name"]
        else:
            if self._out_name is None:
                self._out_name = config.replace(".json", "")
            config = json.load(open("config/" + config, "r"))

        # Validate required parameters are present in `config`
        if len(self._required.intersection(set(config.keys()))) != len(self._required):
            missing_params = "`, `".join(self._required - set(config.keys()))
            raise Exception(
                f"Required parameter(s) `{missing_params}` not found in provided config file."
            )

        self._config = config

    def _get_param(self, param, default):
        if param in self.config:
            if param == "names" and self._config["names"] == "*":
                return [
                    f for f in os.listdir("inputs") if os.path.isfile(f"inputs/{f}")
                ]
            return self.config[param]
        else:
            return default

    def parse(self) -> Namespace:
        """Parses config file.
        
        Overrides config defaults with user provided params and returns them
        namespaced.

        Returns:
            Namespace: A Namespace object containing parsed BIONIC parameters.
        """

        parsed_params = {
            param: self._get_param(param, default)
            for param, default in self._defaults.items()
        }
        namespace = Namespace(**parsed_params)
        return namespace
