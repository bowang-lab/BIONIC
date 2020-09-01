import typer
from pathlib import Path
from bionic.train import Trainer

app = typer.Typer()


# TODO: add config params to cli
@app.command("bionic")
def train(config_path: Path):
    """Integrates networks using BIONIC.

    All relevant parameters for the model should be specified in a `.json` config file.
    
    See https://github.com/bowang-lab/BIONIC/blob/master/README.md for details on writing
    the config file, as well as usage tips.
    """
    trainer = Trainer(config_path)
    trainer.train()
    trainer.forward()


def main():
    app()
