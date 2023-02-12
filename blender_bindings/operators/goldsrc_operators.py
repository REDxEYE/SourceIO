from pathlib import Path

import bpy
from bpy.props import (BoolProperty, CollectionProperty, FloatProperty,
                       StringProperty)

from ...library.global_config import GoldSrcConfig
from ...library.utils import FileBuffer
from ...library.utils.math_utilities import SOURCE1_HAMMER_UNIT_TO_METERS
from ...logger import SLoggingManager
from ..goldsrc import import_model
from ..goldsrc.bsp.import_bsp import BSP

logger = SLoggingManager().get_logger("GoldSrc::Operators")


class SOURCEIO_OT_GBSPImport(bpy.types.Operator):
    """Load GoldSrc BSP"""
    bl_idname = "sourceio.gbsp"
    bl_label = "Import GoldSrc BSP file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    filter_glob: StringProperty(default="*.bsp", options={'HIDDEN'})

    scale: FloatProperty(name="World scale", default=SOURCE1_HAMMER_UNIT_TO_METERS, precision=6)
    use_hd: BoolProperty(name="Load HD models", default=False, subtype='UNSIGNED')
    single_collection: BoolProperty(name="Load everything into 1 collection", default=False, subtype='UNSIGNED')

    def execute(self, context):

        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        for n, file in enumerate(self.files):
            logger.info(f"Loading {n}/{len(self.files)}")
            content_manager = GoldSrcConfig()
            content_manager.use_hd = self.use_hd
            bsp = BSP(directory / file.name, scale=self.scale, single_collection=self.single_collection)
            bsp.load_map()
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


class SOURCEIO_OT_GMDLImport(bpy.types.Operator):
    """Load GoldSrc MDL"""
    bl_idname = "sourceio.gmdl"
    bl_label = "Import GoldSrc MDL file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    filter_glob: StringProperty(default="*.mdl", options={'HIDDEN'})
    scale: FloatProperty(name="World scale", default=SOURCE1_HAMMER_UNIT_TO_METERS, precision=6)

    def execute(self, context):

        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        for n, file in enumerate(self.files):
            logger.info(f"Loading {n}/{len(self.files)}")
            texture_file = (directory / file.name).with_name(Path(file.name).stem + 't.mdl')
            import_model(Path(file.name).stem,
                         FileBuffer(directory / file.name),
                         FileBuffer(texture_file) if texture_file.exists() else None,
                         self.scale)
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}
