import typer
from pathlib import Path
from bionic.train import train

app = typer.Typer()


@app.command("bionic")
def train_(config_path: Path):
    train(config_path)


def main():
    app()
