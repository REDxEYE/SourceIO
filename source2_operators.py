from pathlib import Path

import bpy
from bpy.props import StringProperty, BoolProperty, CollectionProperty, EnumProperty, FloatProperty

from .source2.resouce_types.valve_model import ValveModel
from .source2.resouce_types.valve_texture import ValveTexture
from .source2.resouce_types.valve_material import ValveMaterial
from .source2.resouce_types.valve_world import ValveWorld




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
        for n, file in enumerate(self.files):
            print(f"Loading {n+1}/{len(self.files)}")
            model = ValveModel(str(directory / file.name))
            model.load_mesh(self.invert_uv)
            model.load_attachments()
            if self.import_anim:
                model.load_animations()
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


# noinspection PyUnresolvedReferences
class VWRLDImport_OT_operator(bpy.types.Operator):
    """Load Source2 VWRLD"""
    bl_idname = "source_io.vwrld"
    bl_label = "Import Source2 VWRLD file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    filter_glob: StringProperty(default="*.vwrld_c", options={'HIDDEN'})

    invert_uv: BoolProperty(name="invert UV?", default=True)
    scale: FloatProperty(name="World scale", default=0.0328083989501312)  # LifeForLife suggestion

    use_placeholders: BoolProperty(name="Use placeholders instead of objects", default=False)
    load_static: BoolProperty(name="Load static meshes", default=True)
    load_dynamic: BoolProperty(name="Load entities", default=True)

    def execute(self, context):

        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        for n, file in enumerate(self.files):
            print(f"Loading {n}/{len(self.files)}")
            world = ValveWorld(str(directory / file.name), self.invert_uv, self.scale)
            if self.load_static:
                try:
                    world.load_static()
                except KeyboardInterrupt:
                    print("Skipped static assets")
            if self.load_dynamic:
                world.load_entities(self.use_placeholders)
        print("Hey @LifeForLife, everything is imported as you wanted!!")
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
        for n, file in enumerate(self.files):
            print(f"Loading {n + 1}/{len(self.files)}")
            material = ValveMaterial(str(directory / file.name))
            material.load(self.flip, self.split_alpha)
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
    load_alpha: BoolProperty(default=False, name='Load alpha into separate image')
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    filter_glob: StringProperty(default="*.vtex_c", options={'HIDDEN'})

    def execute(self, context):
        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        for file in self.files:
            texture = ValveTexture(str(directory / file.name))
            texture.load(self.flip,self.load_alpha)
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


