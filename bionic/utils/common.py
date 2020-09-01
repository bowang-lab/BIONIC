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


def create_time_taken_string(time_start: float, time_end: float) -> str:
    time_taken_seconds = time_end - time_start
    if time_taken_seconds < 60:
        return f"Time taken: {magenta(f'{time_taken_seconds:.2f}')} seconds"
    elif time_taken_seconds < 3600:
        return f"Time taken: {magenta(f'{time_taken_seconds/60:.2f}')} minutes"
    else:
        return f"Time taken: {magenta(f'{time_taken_seconds/3600:.2f}')} hours"


class Device:
    """Returns the currently used device by calling `Device()`.

    Returns:
        str: Either "cuda" or "cpu".
    """

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    def __new__(cls) -> str:
        return cls._device
