from .library import loaded_as_addon, running_in_blender

bl_info = {
    "name": "SourceIO",
    "author": "RED_EYE, ShadelessFox, Syborg64",
    "version": (5, 2, 0),
    "blender": (3, 1, 0),
    "location": "File > Import-Export > SourceEngine assets",
    "description": "GoldSrc/Source1/Source2 Engine assets(.mdl, .bsp, .vmt, .vtf, .vmdl_c, .vwrld_c, .vtex_c)"
                   "Notice that you cannot delete this addon via blender UI, remove it manually from addons folder",
    "category": "Import-Export"
}
if running_in_blender() and loaded_as_addon():
    from .blender_bindings.bindings import register, unregister

    if __name__ == "__main__":
        register()
