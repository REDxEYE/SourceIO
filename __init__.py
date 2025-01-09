import sys
import warnings
from pathlib import Path

bl_info = {
    "name": "SourceIO",
    "author": "RED_EYE, ShadelessFox, Syborg64",
    "version": (5, 4, 6),
    "blender": (4, 0, 0),
    "location": "File > Import > SourceEngine assets",
    "description": "GoldSrc/Source1/Source2 Engine assets(.mdl, .bsp, .vmt, .vtf, .vmdl_c, .vwrld_c, .vtex_c)"
                   "Notice that you cannot delete this addon via blender UI, remove it manually from addons folder",
    "category": "Import-Export"
}
warnings.simplefilter("always", DeprecationWarning)
if "SourceIO" not in sys.modules:
    sys.modules['SourceIO'] = sys.modules[Path(__file__).parent.stem]

from SourceIO.library import loaded_as_addon, running_in_blender

if running_in_blender() and loaded_as_addon():
    import bpy

    if bpy.app.version >= (4, 0, 0):
        print("SourceIO only support blender 4.X.X")

    from SourceIO.blender_bindings.bindings import register, unregister

    if __name__ == "__main__":
        register()
