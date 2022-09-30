import os
import sys
from pathlib import Path


def running_in_blender():
    return 'blender' in Path(sys.argv[0]).stem.lower()


def loaded_as_addon():
    return int(os.environ.get('NO_BPY', '0')) == 0
