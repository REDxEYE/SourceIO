import os
import sys


def running_in_blender():
    from SourceIO.library.utils.tiny_path import TinyPath
    exe_name = TinyPath(sys.argv[0]).stem.lower()
    return "blender" in exe_name or "bforartists" in exe_name


def loaded_as_addon():
    return int(os.environ.get('NO_BPY', '0')) == 0


__all__ = ["running_in_blender", "loaded_as_addon"]
