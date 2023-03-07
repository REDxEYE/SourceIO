import os
import sys
from pathlib import Path


def running_in_blender():
    exe_name = Path(sys.argv[0]).stem.lower()
    return "blender" in exe_name or "bforartists" in exe_name


def loaded_as_addon():
    return int(os.environ.get('NO_BPY', '0')) == 0
