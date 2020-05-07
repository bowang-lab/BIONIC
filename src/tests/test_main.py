import sys
import pathlib
sys.path.append(f'{pathlib.Path(__file__).parent.parent.absolute()}/')

import pytest
from main import main

base_config = {

}

class TestMain:
    def test_test(self):
        assert True