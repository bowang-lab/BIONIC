import typer
from pathlib import Path
from typing import Union


def extend_path(path: Path, extension: str) -> Path:
    return path.parent / (path.stem + extension)


def cyan(string, **kwargs):
    return typer.style(string, fg=typer.colors.CYAN, bold=True, **kwargs)


def magenta(string, **kwargs):
    return typer.style(string, fg=typer.colors.MAGENTA, bold=True, **kwargs)
