import typer
from typing import Optional
from bionic.train import train

app = typer.Typer()

@app.command("bionic")
# TODO: figure out how to handle various paths
def train_(config_path: str, out_name: Optional[Union[str, None]] = None):
    train(config_path, out_name)

def main():
    app()