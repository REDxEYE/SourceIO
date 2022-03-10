import os

from .library import running_in_blender, loaded_as_addon

NO_BPY = int(os.environ.get('NO_BPY', '0'))

bl_info = {
    "name": "SourceIO",
    "author": "RED_EYE, ShadelessFox, Syborg64",
    "version": (4, 0, 2),
    "blender": (2, 80, 0),
    "location": "File > Import-Export > SourceEngine assets",
    "description": "GoldSrc/Source1/Source2 Engine assets(.mdl, .bsp, .vmt, .vtf, .vmdl_c, .vwrld_c, .vtex_c)"
                   "Notice that you cannot delete this addon via blender UI, remove it manually from addons folder",
    "category": "Import-Export"
}
if running_in_blender() and loaded_as_addon():
    from .blender_bindings.bindings import register, unregister

    if __name__ == "__main__":
        register()
