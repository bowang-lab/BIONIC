import time
import typer
from pathlib import Path
from .train import Trainer
from .utils.common import create_time_taken_string

app = typer.Typer()


# TODO: add config params to cli
@app.command("bionic")
def train(config_path: Path):
    """Integrates networks using BIONIC.

    All relevant parameters for the model should be specified in a `.json` config file.

    See https://github.com/bowang-lab/BIONIC/blob/master/README.md for details on writing
    the config file, as well as usage tips.
    """
    time_start = time.time()
    trainer = Trainer(config_path)
    trainer.train()
    trainer.forward()
    time_end = time.time()
    typer.echo(create_time_taken_string(time_start, time_end))


def main():
    app()
