import sys
import os
print(os.path.abspath(__file__ + "/../../../src/"))
sys.path.append(os.path.abspath(__file__ + "/../../../src/"))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import types
import importlib
import traceback
import pytest

def test_launch():
    pass