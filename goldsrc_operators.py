from pathlib import Path

import bpy
from bpy.props import StringProperty, CollectionProperty, BoolProperty, FloatProperty

from .goldsrc.bsp.import_bsp import BSP
from .goldsrc.bsp.mgr import GoldSrcContentManager
from .goldsrc.mdl.import_mdl import import_model
from .utilities.math_utilities import HAMMER_UNIT_TO_METERS


class GBSPImport_OT_operator(bpy.types.Operator):
    """Load GoldSrc BSP"""
    bl_idname = "source_io.gbsp"
    bl_label = "Import GoldSrc BSP file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    filter_glob: StringProperty(default="*.bsp", options={'HIDDEN'})

    scale: FloatProperty(name="World scale", default=HAMMER_UNIT_TO_METERS, precision=6)
    use_hd: BoolProperty(name="Load HD models", default=False, subtype='UNSIGNED')

    def execute(self, context):

        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        for n, file in enumerate(self.files):
            print(f"Loading {n}/{len(self.files)}")
            content_manager = GoldSrcContentManager()
            content_manager.use_hd = self.use_hd
            bsp = BSP(directory / file.name)
            bsp.scale = self.scale
            bsp.load_map()
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


class GMDLImport_OT_operator(bpy.types.Operator):
    """Load GoldSrc MDL"""
    bl_idname = "source_io.gmdl"
    bl_label = "Import GoldSrc MDL file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    filter_glob: StringProperty(default="*.mdl", options={'HIDDEN'})
    scale: FloatProperty(name="World scale", default=HAMMER_UNIT_TO_METERS, precision=6)

    def execute(self, context):

        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        for n, file in enumerate(self.files):
            print(f"Loading {n}/{len(self.files)}")
            texture_file = (directory / file.name).with_name(Path(file.name).stem + 't.mdl')
            import_model(directory / file.name, texture_file if texture_file.exists() else None, self.scale)
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}
