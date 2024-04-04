import bpy

from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4_1

if is_blender_4_1():
    # noinspection PyPep8Naming
    class IMAGE_FH_vtf_import(bpy.types.FileHandler):
        bl_idname = "IMAGE_FH_vtf_import"
        bl_label = "File handler for vtf texture import"
        bl_import_operator = "sourceio.vtf"
        bl_file_extensions = ".vtf"

        @classmethod
        def poll_drop(cls, context):
            return context.region and context.region.type == 'WINDOW'

    # noinspection PyPep8Naming
    class IMAGE_FH_vtex_import(bpy.types.FileHandler):
        bl_idname = "IMAGE_FH_vtex_import"
        bl_label = "File handler for vtf texture import"
        bl_import_operator = "sourceio.vtex"
        bl_file_extensions = ".vtex_c"

        @classmethod
        def poll_drop(cls, context):
            return context.region and context.region.type == 'WINDOW'

    # noinspection PyPep8Naming
    class OBJECT_FH_mdl_import(bpy.types.FileHandler):
        bl_idname = "OBJECT_FH_mdl_import"
        bl_label = "File handler for vtf texture import"
        bl_import_operator = "sourceio.mdl"
        bl_file_extensions = ".mdl;.md3"

        @classmethod
        def poll_drop(cls, context):
            return (context.region and context.region.type == 'WINDOW'
                    and context.area and context.area.ui_type == 'VIEW_3D')

    # noinspection PyPep8Naming
    class MATERIAL_FH_vmt_import(bpy.types.FileHandler):
        bl_idname = "MATERIAL_FH_vmt_import"
        bl_label = "File handler for vmt material import"
        bl_import_operator = "sourceio.vmt"
        bl_file_extensions = ".vmt"

        @classmethod
        def poll_drop(cls, context):
            return (context.region and context.region.type == 'WINDOW')

    # noinspection PyPep8Naming
    class OBJECT_FH_bsp_import(bpy.types.FileHandler):
        bl_idname = "OBJECT_FH_bsp_import"
        bl_label = "File handler for BPS map import"
        bl_import_operator = "sourceio.bsp"
        bl_file_extensions = ".bsp"

        @classmethod
        def poll_drop(cls, context):
            return (context.region and context.region.type == 'WINDOW')

    # noinspection PyPep8Naming
    class OBJECT_FH_vmap_import(bpy.types.FileHandler):
        bl_idname = "OBJECT_FH_vmap_import"
        bl_label = "File handler for VMAP map import"
        bl_import_operator = "sourceio.vmap"
        bl_file_extensions = ".vmap_c"

        @classmethod
        def poll_drop(cls, context):
            return (context.region and context.region.type == 'WINDOW')

    # noinspection PyPep8Naming
    class OBJECT_FH_vmap_vpk_import(bpy.types.FileHandler):
        bl_idname = "OBJECT_FH_vmap_vpk_import"
        bl_label = "File handler for VMAP vpk map import"
        bl_import_operator = "sourceio.vmap_vpk"
        bl_file_extensions = ".vpk"

        @classmethod
        def poll_drop(cls, context):
            return (context.region and context.region.type == 'WINDOW')

    # noinspection PyPep8Naming
    class MATERIAL_FH_vmat_import(bpy.types.FileHandler):
        bl_idname = "MATERIAL_FH_vmat_import"
        bl_label = "File handler for VMAP map import"
        bl_import_operator = "sourceio.vmat"
        bl_file_extensions = ".vmat_c"

        @classmethod
        def poll_drop(cls, context):
            return (context.region and context.region.type == 'WINDOW')
