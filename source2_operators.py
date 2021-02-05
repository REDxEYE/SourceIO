from pathlib import Path

import bpy
from bpy.props import StringProperty, BoolProperty, CollectionProperty, EnumProperty, FloatProperty

from .source2.resouce_types.valve_model import ValveCompiledModel
from .source2.resouce_types.valve_texture import ValveCompiledTexture
from .source2.resouce_types.valve_material import ValveCompiledMaterial
from .source2.resouce_types.valve_world import ValveCompiledWorld
from .source_shared.content_manager import ContentManager
from .utilities.math_utilities import HAMMER_UNIT_TO_METERS


# noinspection PyUnresolvedReferences
class VMDLImport_OT_operator(bpy.types.Operator):
    """Load Source2 VMDL"""
    bl_idname = "source_io.vmdl"
    bl_label = "Import Source2 VMDL file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    invert_uv: BoolProperty(name="invert UV?", default=True)
    import_anim: BoolProperty(name="Import animations", default=False)
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)

    filter_glob: StringProperty(default="*.vmdl_c", options={'HIDDEN'})

    def execute(self, context):

        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        ContentManager().scan_for_content(directory)
        for n, file in enumerate(self.files):
            print(f"Loading {n + 1}/{len(self.files)}")
            model = ValveCompiledModel(str(directory / file.name))
            model.load_mesh(self.invert_uv)
            model.load_attachments()
            if self.import_anim:
                model.load_animations()
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


class VWRLDImport_OT_operator(bpy.types.Operator):
    """Load Source2 VWRLD"""
    bl_idname = "source_io.vwrld"
    bl_label = "Import Source2 VWRLD file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    filter_glob: StringProperty(default="*.vwrld_c", options={'HIDDEN'})

    invert_uv: BoolProperty(name="invert UV?", default=True)
    scale: FloatProperty(name="World scale", default=HAMMER_UNIT_TO_METERS, precision=6)

    def execute(self, context):

        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        for n, file in enumerate(self.files):
            print(f"Loading {n}/{len(self.files)}")
            ContentManager().scan_for_content((directory.parent / file.name).with_suffix('.vpk'))
            world = ValveCompiledWorld(directory / file.name, invert_uv=self.invert_uv, scale=self.scale)
            world.load(file.name)
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


# noinspection PyUnresolvedReferences
class VMATImport_OT_operator(bpy.types.Operator):
    """Load Source2 material"""
    bl_idname = "source_io.vmat"
    bl_label = "Import Source2 VMDL file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    flip: BoolProperty(name="Flip texture", default=True)
    split_alpha: BoolProperty(name="Extract alpha texture", default=True)
    filter_glob: StringProperty(default="*.vmat_c", options={'HIDDEN'})

    def execute(self, context):

        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        ContentManager().scan_for_content(directory)
        for n, file in enumerate(self.files):

            print(f"Loading {n + 1}/{len(self.files)}")
            material = ValveCompiledMaterial(str(directory / file.name))
            material.load()
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


class VTEXImport_OT_operator(bpy.types.Operator):
    """Load Source Engine VTF texture"""
    bl_idname = "source_io.vtex"
    bl_label = "Import VTEX"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype='FILE_PATH', )
    flip: BoolProperty(name="Flip texture", default=True)
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    filter_glob: StringProperty(default="*.vtex_c", options={'HIDDEN'})

    def execute(self, context):
        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        for file in self.files:
            texture = ValveCompiledTexture(str(directory / file.name))
            texture.load(self.flip)
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}
