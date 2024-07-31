import os
import sys
from pathlib import Path


def running_in_blender():
    exe_name = Path(sys.argv[0]).stem.lower()
    return "blender" in exe_name or "bforartists" in exe_name


def loaded_as_addon():
    force_no_bpy = int(os.environ.get('NO_BPY', '0')) == 1
    if force_no_bpy:
        return False
    try:
        import bpy
        # Used to check if we are running from python with blender as module
        return bpy.app.sdl.supported
    except ImportError:
        return False
