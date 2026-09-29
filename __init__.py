import sys
from pathlib import Path

if "SourceIO" not in sys.modules:
    sys.modules['SourceIO'] = sys.modules[Path(__file__).parent.stem]

if sys.version_info < (3, 11, 0):
    raise Exception("SourceIO requires python 3.11+ or Blender 4.2.0+")

from SourceIO.library import loaded_as_addon, running_in_blender

try:
    import bpy
    # 4.2 LTS is the oldest Blender bundling Python 3.11, which is the stable-ABI
    # floor the bundled pylib is built against; 4.0/4.1 ship 3.10 and cannot load it.
    if bpy.app.version < (4, 2, 0):
        raise Exception("SourceIO only support blender 4.2.0 and above")
except ImportError:
    bpy = ...

bl_info = {
    "name": "SourceIO",
    "author": "RED_EYE, ShadelessFox, Syborg64",
    "version": (5, 5, 5),
    "blender": (4, 2, 0),
    "location": "File > Import > SourceEngine assets",
    "description": "GoldSrc/Source1/Source2 Engine assets(.mdl, .bsp, .vmt, .vtf, .vmdl_c, .vwrld_c, .vtex_c)"
                   "Notice that you cannot delete this addon via blender UI, remove it manually from addons folder",
    "category": "Import-Export"
}

import warnings
warnings.simplefilter("always", DeprecationWarning)

from SourceIO.library import loaded_as_addon, running_in_blender

if running_in_blender() and loaded_as_addon():
    from SourceIO.blender_bindings.bindings import register, unregister
    if __name__ == "__main__":
        register()
