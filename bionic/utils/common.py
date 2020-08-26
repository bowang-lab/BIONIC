from pathlib import Path


def extend_path(path: Path, extension: str):
    return path.parent / (path.stem + extension)
