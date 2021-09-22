import os
import sys
from pathlib import Path


def running_in_blender():
    return Path(sys.argv[0]).stem.lower() == 'blender'


def loaded_as_addon():
    return int(os.environ.get('NO_BPY', '0')) == 0
