from pathlib import Path

import bpy
from bpy.props import StringProperty, BoolProperty, CollectionProperty, EnumProperty, FloatProperty

from .source1.new_model_import import import_model, import_materials
from .source2.resouce_types.valve_model import ValveModel
from .utilities.path_utilities import backwalk_file_resolver


class LoadPlaceholder_OT_operator(bpy.types.Operator):
    bl_idname = "source_io.load_placeholder"
    bl_label = "Load placeholder"
    bl_options = {'UNDO'}

    def execute(self, context):
        for obj in context.selected_objects:

            if obj.get("entity_data", None):
                custom_prop_data = obj['entity_data']
                model_type = Path(custom_prop_data['prop_path']).suffix
                collection = bpy.data.collections.get(custom_prop_data['type'],
                                                      None) or bpy.data.collections.new(
                    name=custom_prop_data['type'])
                if model_type in ['.vmdl_c', '.vmdl_c']:
                    model_path = backwalk_file_resolver(custom_prop_data['parent_path'],
                                                        Path(custom_prop_data['prop_path']).with_suffix('.vmdl_c'))

                    if model_path:

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
                elif model_type == '.mdl':
                    model_path = backwalk_file_resolver(custom_prop_data['parent_path'],
                                                        Path(custom_prop_data['prop_path']))
                    if model_path:
                        vvd = backwalk_file_resolver(model_path.parent, model_path.with_suffix('.vvd'))
                        vtx = backwalk_file_resolver(model_path.parent, Path(model_path.stem + '.dx90.vtx'))
                        mdl, vvd, vtx, armature = import_model(model_path, vvd, vtx, None, False, collection, True)
                        armature.location = obj.location
                        armature.rotation_mode = "XYZ"
                        armature.rotation_euler = obj.rotation_euler
                        armature.scale = obj.scale
                        import_materials(mdl)

                        bpy.data.objects.remove(obj)
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
