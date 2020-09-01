import typer
from pathlib import Path

import torch


def extend_path(path: Path, extension: str) -> Path:
    """Extends a path by adding an extension to the stem.

    Args:
        path (Path): Full path.
        extension (str): Extension to add. This will replace the current extension.

    Returns:
        Path: New path with extension.
    """
    return path.parent / (path.stem + extension)


def cyan(string: str, **kwargs) -> str:
    return typer.style(string, fg=typer.colors.CYAN, bold=True, **kwargs)


def magenta(string: str, **kwargs) -> str:
    return typer.style(string, fg=typer.colors.MAGENTA, bold=True, **kwargs)


class Device:
    """Returns the currently used device by calling `Device()`.

    Returns:
        str: Either "cuda" or "cpu".
    """

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    def __new__(cls) -> str:
        return cls._device
