import platform

import bpy

from SourceIO.library.utils.singleton import SingletonMeta
from SourceIO.library.utils.tiny_path import TinyPath

from .attributes import register_props, unregister_props
from .operators.flex_operators import classes as flex_classes
from .operators.goldsrc_operators import SOURCEIO_OT_GBSPImport
from .operators.shared_operators import shared_classes
from .operators.source1_operators import (SOURCEIO_OT_BSPImport,
                                          SOURCEIO_OT_DMXImporter,
                                          SOURCEIO_OT_MDLImport)
from .operators.source1_operators import (SOURCEIO_OT_SkyboxImport,
                                          SOURCEIO_OT_VMTImport,
                                          # SOURCEIO_OT_VTFExport,
                                          SOURCEIO_OT_VTFImport)
from .operators.source2_operators import (SOURCEIO_OT_VMAPImport,
                                          SOURCEIO_OT_VMATImport,
                                          SOURCEIO_OT_VMDLImport,
                                          SOURCEIO_OT_VPHYSImport,
                                          SOURCEIO_OT_VPK_VMAPImport,
                                          SOURCEIO_OT_VTEXImport,
                                          SOURCEIO_OT_DMXCameraImport)
from .ui.export_nodes import register_nodes, unregister_nodes
from .utils.bpy_utils import is_blender_4_1


custom_icons = {}


# noinspection PyPep8Naming
class SourceIO_MT_Menu(bpy.types.Menu):
    bl_label = "Source Engine Assets"
    bl_idname = "IMPORT_MT_sourceio"

    def draw(self, context):
        crowbar_icon = custom_icons["main"]["crowbar_icon"]
        bsp_icon = custom_icons["main"]["bsp_icon"]
        vtf_icon = custom_icons["main"]["vtf_icon"]
        vmt_icon = custom_icons["main"]["vmt_icon"]
        model_doc_icon = custom_icons["main"]["model_doc_icon"]
        vmat_icon = custom_icons["main"]["vmat_icon"]
        vtex_icon = custom_icons["main"]["vtex_icon"]
        vwrld_icon = custom_icons["main"]["vwrld_icon"]
        layout = self.layout

        layout.operator(SOURCEIO_OT_MDLImport.bl_idname, text="GoldSrc/Source model (.mdl)",
                        icon_value=crowbar_icon.icon_id)
        layout.separator()

        layout.operator(SOURCEIO_OT_GBSPImport.bl_idname, text="GoldSrc map (.bsp)",
                        icon_value=bsp_icon.icon_id)
        layout.operator(SOURCEIO_OT_BSPImport.bl_idname, text="Source map (.bsp)",
                        icon_value=bsp_icon.icon_id)
        layout.operator(SOURCEIO_OT_VTFImport.bl_idname, text="Source texture (.vtf)",
                        icon_value=vtf_icon.icon_id)
        layout.operator(SOURCEIO_OT_SkyboxImport.bl_idname, text="Source Skybox (.vmt)",
                        icon_value=vtf_icon.icon_id)
        layout.operator(SOURCEIO_OT_VMTImport.bl_idname, text="Source material (.vmt)",
                        icon_value=vmt_icon.icon_id)
        # layout.operator(SOURCEIO_OT_DMXImporter.bl_idname, text="[!!!WIP!!!] SFM session (.dmx) [!!!WIP!!!]")
        layout.separator()

        layout.operator(SOURCEIO_OT_VMDLImport.bl_idname, text="Source2 model (.vmdl_c)",
                        icon_value=model_doc_icon.icon_id)
        layout.operator(SOURCEIO_OT_VPHYSImport.bl_idname, text="Source2 physics (.vphys_c)",
                        icon_value=model_doc_icon.icon_id)
        layout.operator(SOURCEIO_OT_VMAPImport.bl_idname, text="Source2 map (.vmap_c)",
                        icon_value=vwrld_icon.icon_id)
        layout.operator(SOURCEIO_OT_VPK_VMAPImport.bl_idname, text="Source2 packed map (.vpk)",
                        icon_value=vwrld_icon.icon_id)
        layout.operator(SOURCEIO_OT_VTEXImport.bl_idname, text="Source2 texture (.vtex_c)",
                        icon_value=vtex_icon.icon_id)
        layout.operator(SOURCEIO_OT_VMATImport.bl_idname, text="Source2 material (.vmat_c)",
                        icon_value=vmat_icon.icon_id)
        layout.separator()
        # layout.operator(SourceIO_OP_VPKBrowserLoader.bl_idname, text="[!!!WIP!!!]Browse new VPK (.vpk)",
        #                 icon_value=bsp_icon.icon_id)
        # layout.operator(SourceIO_OP_VPKBrowser.bl_idname, text="[!!!WIP!!!]Browse already open VPK (.vpk)",
        #                 icon_value=bsp_icon.icon_id)
        # layout.separator()
        # layout.menu(SourceIOUtils_MT_Menu.bl_idname)


class SourceIOUtils_MT_Menu(bpy.types.Menu):
    bl_label = "Source Engine Utils"
    bl_idname = "IMPORT_MT_sourceio_utils"

    def draw(self, context):
        layout = self.layout
        layout.operator(SOURCEIO_OT_DMXCameraImport.bl_idname, text="Valve camera(.dmx)")


def menu_import(self, context):
    source_io_icon = custom_icons["main"]["sourceio_icon"]
    self.layout.menu(SourceIO_MT_Menu.bl_idname, icon_value=source_io_icon.icon_id)


def load_icon(loader, filename, name):
    script_path = TinyPath(__file__).parent
    icon_path = script_path / 'icons' / filename
    loader.load(name, str(icon_path), 'IMAGE')


def register_custom_icon():
    import bpy.utils.previews
    pcoll = bpy.utils.previews.new()
    load_icon(pcoll, 'sourceio_icon.png', "sourceio_icon")
    load_icon(pcoll, 'crowbar_icon.png', "crowbar_icon")
    load_icon(pcoll, 'bsp_icon.png', "bsp_icon")
    load_icon(pcoll, 'vtf_icon.png', "vtf_icon")
    load_icon(pcoll, 'vmt_icon.png', "vmt_icon")
    load_icon(pcoll, 'model_doc_icon.png', "model_doc_icon")
    load_icon(pcoll, 'vmat_icon.png', "vmat_icon")
    load_icon(pcoll, 'vtex_icon.png', "vtex_icon")
    load_icon(pcoll, 'vwrld_icon.png', "vwrld_icon")
    custom_icons["main"] = pcoll


def unregister_custom_icon():
    import bpy.utils.previews
    for pcoll in custom_icons.values():
        bpy.utils.previews.remove(pcoll)
    custom_icons.clear()


classes = [
    # GoldSrc
    SOURCEIO_OT_GBSPImport,
    # Source1 stuff

    SOURCEIO_OT_MDLImport,
    SOURCEIO_OT_BSPImport,
    SOURCEIO_OT_DMXImporter,

    # Source2 stuff
    SOURCEIO_OT_DMXCameraImport,
    SOURCEIO_OT_VMDLImport,
    SOURCEIO_OT_VTEXImport,
    SOURCEIO_OT_VPHYSImport,
    SOURCEIO_OT_VMATImport,
    SOURCEIO_OT_VPK_VMAPImport,
    SOURCEIO_OT_VMAPImport,

    # Addon tools
    # SourceIOPreferences,
    SourceIO_MT_Menu,
    SourceIOUtils_MT_Menu,

    SOURCEIO_OT_VTFImport,
    SOURCEIO_OT_VMTImport,
    SOURCEIO_OT_SkyboxImport,

    # *vpk_classes,
    *shared_classes,
    *flex_classes,
]
if is_blender_4_1():
    from .operators.dragndrop import (
        IMAGE_FH_vtf_import,
        IMAGE_FH_vtex_import,
        OBJECT_FH_mdl_import,
        MATERIAL_FH_vmt_import,
        OBJECT_FH_bsp_import,
        OBJECT_FH_vmap_import,
        OBJECT_FH_vmap_vpk_import,
        MATERIAL_FH_vmat_import,
    )

    classes.append(IMAGE_FH_vtf_import)
    classes.append(IMAGE_FH_vtex_import)
    classes.append(OBJECT_FH_mdl_import)
    classes.append(MATERIAL_FH_vmt_import)
    classes.append(OBJECT_FH_bsp_import)
    classes.append(OBJECT_FH_vmap_import)
    classes.append(OBJECT_FH_vmap_vpk_import)
    classes.append(MATERIAL_FH_vmat_import)

register_, unregister_ = bpy.utils.register_classes_factory(classes)


is_windows = platform.system() == "Windows"
def register():
    # Taken from https://github.com/lasa01/Plumber/blob/master/plumber/__init__.py
    # if is_windows and False:
    #     # Check if the extension module was renamed on the last unregister,
    #     # and either rename it back or delete it if the addon was updated with a newer extension module
    #     ext_path = TinyPath(__file__).parent.parent / "library/utils/rustlib/windows_x64/rustlib.pyd"
    #     unloaded_ext_path = TinyPath(__file__).parent.parent.parent / "rustlib.pyd.unloaded"
    #     if unloaded_ext_path.exists():
    #         if ext_path.exists():
    #             try:
    #                 os.remove(unloaded_ext_path)
    #             except OSError:
    #                 print("[SourceIO] [WARN] old files remaining, restart Blender to finish post-update clean up")
    #         else:
    #             os.rename(unloaded_ext_path, ext_path)


    register_custom_icon()
    register_()
    register_nodes()
    register_props()
    bpy.types.TOPBAR_MT_file_import.append(menu_import)

    # if is_vtflib_supported():
    #     from ..library.source1.vtf import VTFLib
    #     from .operators.source1_operators import export
    #     bpy.types.IMAGE_MT_image.append(export)

def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)

    # Taken from https://github.com/lasa01/Plumber/blob/master/plumber/__init__.py
    # if is_windows and False:
    #     # Rename the extension module to allow updating the addon without restarting Blender,
    #     # since the extension module will stay open and can't be overwritten even if the addon is unloaded
    #     ext_path = TinyPath(__file__).parent.parent / "library/utils/rustlib/windows_x64/rustlib.pyd"
    #     unloaded_ext_path = TinyPath(__file__).parent.parent.parent / "rustlib.pyd.unloaded"
    #     try:
    #         os.rename(ext_path, unloaded_ext_path)
    #     except OSError:
    #         pass

    # if is_vtflib_supported():
    #     from .operators.source1_operators import export
    #     bpy.types.IMAGE_MT_image.remove(export)

    unregister_nodes()
    unregister_props()
    SingletonMeta.cleanup()

    unregister_custom_icon()
    unregister_()
