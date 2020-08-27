import typer
from pathlib import Path
from typing import Union

import torch


def extend_path(path: Path, extension: str) -> Path:
    return path.parent / (path.stem + extension)


def cyan(string: str, **kwargs) -> str:
    return typer.style(string, fg=typer.colors.CYAN, bold=True, **kwargs)


def magenta(string: str, **kwargs) -> str:
    return typer.style(string, fg=typer.colors.MAGENTA, bold=True, **kwargs)


class Device:
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    def __new__(cls):
        return cls._device
