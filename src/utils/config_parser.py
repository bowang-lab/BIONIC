import os
import json
import types
from argparse import Namespace


class ConfigParser:
    def __init__(self, config, out_name=None):

        # Check if the config is already loaded.
        if isinstance(config, dict):
            if "out_name" not in config:
                if out_name is None:
                    raise Exception("Output file name `out_name` must be provided if `config` is a dictionary")
            else:
                out_name = config["out_name"]
        else:
            if out_name is None:
                out_name = config.replace(".json", "")
            config = json.load(open("config/" + config, "r"))
        self._config = config

        # Default BIONIC parameter values
        self._defaults = {
            "names": None,  # Filenames of input networks
            "out_name": out_name,  # Name of output feature, model and network weight files
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

        # Fields required in config file
        self._required = {"names"}

    @property
    def config(self):
        return self._config

    @property
    def defaults(self):
        return self._defaults

    @property
    def required(self):
        return self._required

    def _get_param(self, param, default):
        if param in self._config:
            if param == "names" and self._config["names"] == "*":
                return [
                    f
                    for f in os.listdir("inputs")
                    if os.path.isfile(f"inputs/{f}")
                ]
            return self._config[param]
        else:
            if param in self._required:
                raise Exception(
                    f"{param} is a required parameter and was not found in the provided config file."
                )

            return default

    def parse(self):
        parsed_params = {
            param: self._get_param(param, default)
            for param, default in self._defaults.items()
        }
        namespace = Namespace(**parsed_params)
        return namespace
