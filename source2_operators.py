from pathlib import Path

import bpy
from bpy.props import StringProperty, BoolProperty, CollectionProperty, EnumProperty, FloatProperty

from .source2.resouce_types.valve_model import ValveModel
from .source2.resouce_types.valve_texture import ValveTexture
from .source2.resouce_types.valve_material import ValveMaterial
from .source2.resouce_types.valve_world import ValveWorld

from .utilities.path_utilities import backwalk_file_resolver


# noinspection PyUnresolvedReferences
class VMDLImporter_OT_operator(bpy.types.Operator):
    """Load Source2 VMDL"""
    bl_idname = "source_io.vmdl"
    bl_label = "Import Source2 VMDL file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    invert_uv: BoolProperty(name="invert UV?", default=True)
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)

    filter_glob: StringProperty(default="*.vmdl_c", options={'HIDDEN'})

    def execute(self, context):

        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        for n, file in enumerate(self.files):
            print(f"Loading {n}/{len(self.files)}")
            model = ValveModel(str(directory / file.name))
            model.load_mesh(self.invert_uv)
            model.load_attachments()
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


# noinspection PyUnresolvedReferences
class VWRLDImporter_OT_operator(bpy.types.Operator):
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
class VMATImporter_OT_operator(bpy.types.Operator):
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


class VTEXImporter_OT_operator(bpy.types.Operator):
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


class LoadPlaceholder_OT_operator(bpy.types.Operator):
    bl_idname = "source_io.load_placeholder"
    bl_label = "Load placeholder"
    bl_options = {'UNDO'}

    def execute(self, context):
        for obj in context.selected_objects:

            if obj.get("entity_data", None):
                custom_prop_data = obj['entity_data']

                model_path = backwalk_file_resolver(custom_prop_data['parent_path'],
                                                    Path(custom_prop_data['prop_path'] + "_c"))
                if model_path:

                    collection = bpy.data.collections.get(custom_prop_data['type'],
                                                          None) or bpy.data.collections.new(
                        name=custom_prop_data['type'])
                    try:
                        bpy.context.scene.collection.children.link(collection)
                    except:
                        pass

                    model = ValveModel(model_path)
                    model.load_mesh(True, parent_collection=collection,
                                    skin_name=custom_prop_data.get("skin_id", 'default'))
                    for ob in model.objects:  # type:bpy.types.Object
                        ob.location = obj.location
                        ob.rotation_mode = "XYZ"
                        ob.rotation_euler = obj.rotation_euler
                        ob.scale = obj.scale
                    bpy.data.objects.remove(obj)
                else:
                    self.report({'INFO'}, f"Model '{custom_prop_data['prop_path']}_c' not found!")
        return {'FINISHED'}


class ChangeSkin_OT_operator(bpy.types.Operator):
    bl_idname = "source_io.select_skin"
    bl_label = "Change skin"
    bl_options = {'UNDO'}

    skin_name: bpy.props.StringProperty(name="skin_name", default="default")

    def execute(self, context):
        obj = context.active_object
        skin_material = obj['skin_groups'][self.skin_name]
        current_material = obj['skin_groups'][obj['active_skin']]

        mat_name = Path(skin_material).stem
        current_mat_name = Path(current_material).stem
        mat = bpy.data.materials.get(mat_name, None) or bpy.data.materials.new(name=mat_name)

        for n, obj_mat in enumerate(obj.data.materials):
            if obj_mat.name == current_mat_name:
                print(obj_mat.name, "->", mat.name)
                obj.data.materials[n] = mat
                break

        obj['active_skin'] = self.skin_name

        return {'FINISHED'}


class SourceIOUtils_PT_panel(bpy.types.Panel):
    bl_label = "SourceIO utils"
    bl_idname = "source_io.utils"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SourceIO"

    @classmethod
    def poll(cls, context):
        obj = context.active_object  # type:bpy.types.Object
        if obj:
            return obj.type in ["EMPTY", 'MESH']
        else:
            return False

    def draw(self, context):
        self.layout.label(text="SourceIO stuff")
        obj = context.active_object  # type:bpy.types.Object
        if obj.get("entity_data", None):
            self.layout.operator('source_io.load_placeholder')
        if obj.get("skin_groups", None):
            self.layout.label(text="Skins")
            box = self.layout.box()
            for skin, _ in obj['skin_groups'].items():
                row = box.row()
                op = row.operator('source_io.select_skin', text=skin)
                op.skin_name = skin
                if skin == obj['active_skin']:
                    row.enabled = False
