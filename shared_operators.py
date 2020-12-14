from pathlib import Path

import bpy
from bpy.props import StringProperty, BoolProperty, CollectionProperty, EnumProperty, FloatProperty

from .source1.content_manager import ContentManager
from .source1.new_model_import import import_model, import_materials
from .source2.resouce_types.valve_model import ValveModel
from .utilities.path_utilities import backwalk_file_resolver


class LoadPlaceholder_OT_operator(bpy.types.Operator):
    bl_idname = "source_io.load_placeholder"
    bl_label = "Load placeholder"
    bl_options = {'UNDO'}

    def execute(self, context):
        content_manager = ContentManager()
        for name, sub in bpy.context.scene.get('content_manager_data', {}).items():
            if name not in content_manager.sub_managers:
                print(f'Registering cached sub manager for {name}:{sub}')
                content_manager.scan_for_content(sub)

        for obj in context.selected_objects:
            print(obj.name, obj.get("entity_data", None))
            if obj.get("entity_data", None):
                custom_prop_data = obj['entity_data']
                model_type = Path(custom_prop_data['prop_path']).suffix
                collection = bpy.data.collections.get(custom_prop_data['type'],
                                                      None) or bpy.data.collections.new(
                    name=custom_prop_data['type'])
                if model_type in ['.vmdl_c', '.vmdl_c']:
                    mld_file = backwalk_file_resolver(custom_prop_data['parent_path'],
                                                      Path(custom_prop_data['prop_path']).with_suffix('.vmdl_c'))

                    if mld_file:

                        try:
                            bpy.context.scene.collection.children.link(collection)
                        except:
                            pass

                        model = ValveModel(mld_file)
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
                    prop_path = Path(custom_prop_data['prop_path'])
                    mld_file = content_manager.find_file(prop_path)
                    if mld_file:
                        vvd_file = content_manager.find_file(prop_path.with_suffix('.vvd'))
                        vtx_file = content_manager.find_file(prop_path.parent / f'{prop_path.stem}.dx90.vtx')
                        mdl, vvd, vtx, armature = import_model(mld_file, vvd_file, vtx_file, None, False, collection,
                                                               True)
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
            entiry_data = obj['entity_data']
            entity_raw_data = entiry_data.get('entity',{})
            box = self.layout.box()
            if entity_raw_data.get('classname', False):
                row = box.row()
                row.label(text='Prop type:')
                row.label(text=entity_raw_data['classname'])
            box.label(text=entiry_data['prop_path'])
            box.operator('source_io.load_placeholder')
        if obj.get("skin_groups", None):
            self.layout.label(text="Skins")
            box = self.layout.box()
            for skin, _ in obj['skin_groups'].items():
                row = box.row()
                op = row.operator('source_io.select_skin', text=skin)
                op.skin_name = skin
                if skin == obj['active_skin']:
                    row.enabled = False
