import os
from pathlib import Path


def hello_world():
    return "Hello World!"


ROOT_DIRECTORY = Path(os.path.dirname(os.path.realpath(__file__)))

print(ROOT_DIRECTORY.parent.parent.absolute())
