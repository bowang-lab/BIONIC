import sys
import pathlib

path = f"{pathlib.Path(__file__).parent.parent.absolute()}/"
sys.path.append(path)

import pytest


class TestPreprocessor:
    def test_(self):
        pass
