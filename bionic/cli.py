import typer
from pathlib import Path
from typing import Optional, Union
from bionic.train import train

app = typer.Typer()


@app.command("bionic")
# TODO: figure out how to handle various paths
def train_(config_path: Path):
    train(config_path)


def main():
    app()
