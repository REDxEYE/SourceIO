import math
from itertools import chain
from pathlib import Path

import bpy

from ..utils.utils import get_or_create_collection
from ..source1.bsp.import_bsp import BPSPropCache
from ..source1.mdl.model_loader import import_model_from_files
from ..source1.mdl.v49.import_mdl import import_materials
from ..source1.mdl.v49.import_mdl import put_into_collections as s1_put_into_collections
from ..source2.vmdl.loader import put_into_collections as s2_put_into_collections, ValveCompiledModelLoader

from ...library.source1.vtf import is_vtflib_supported
from ...library.source2.resource_types.vmdl.model import ValveCompiledModel
from ...library.shared.content_providers.content_manager import ContentManager
from ...library.utils.path_utilities import find_vtx_cm


def get_parent(collection):
    for pcoll in bpy.data.collections:
        if collection.name in pcoll.children:
            return pcoll
    return bpy.context.scene.collection


class ChangeSkin_OT_LoadEntity(bpy.types.Operator):
    bl_idname = "sourceio.load_placeholder"
    bl_label = "Load Entity"
    bl_options = {'UNDO'}

    def execute(self, context):
        content_manager = ContentManager()
        content_manager.deserialize(bpy.context.scene.get('content_manager_data', {}))
        unique_material_names = True

        for obj in context.selected_objects:
            print(f'Loading {obj.name}')
            if obj.get("entity_data", None):
                custom_prop_data = obj['entity_data']
                if 'prop_path' not in custom_prop_data:
                    continue
                model_type = Path(custom_prop_data['prop_path']).suffix
                parent = get_parent(obj.users_collection[0])
                collection = get_or_create_collection(custom_prop_data['type'], parent)
                if model_type == '.vmdl_c':
                    vmld_file = content_manager.find_file(custom_prop_data['prop_path'])
                    if vmld_file:
                        # skin = custom_prop_data.get('skin', None)
                        model = ValveCompiledModelLoader(vmld_file)
                        model.load_mesh(True)
                        container = model.container
                        if container.armature:
                            armature = container.armature
                            armature.location = obj.location
                            armature.rotation_mode = "XYZ"
                            armature.rotation_euler = obj.rotation_euler
                            armature.scale = obj.scale
                        else:
                            for ob in chain(container.objects,
                                            container.physics_objects):  # type:bpy.types.Object
                                ob.location = obj.location
                                ob.rotation_mode = "XYZ"
                                ob.rotation_euler = obj.rotation_euler
                                ob.scale = obj.scale

                            # if skin:
                            #     if str(skin) in ob['skin_groups']:
                            #         skin = str(skin)
                            #         skin_materials = ob['skin_groups'][skin]
                            #         current_materials = ob['skin_groups'][ob['active_skin']]
                            #         print(skin_materials, current_materials)
                            #         for skin_material, current_material in zip(skin_materials, current_materials):
                            #             swap_materials(ob, skin_material[-63:], current_material[-63:])
                            #         ob['active_skin'] = skin
                            #     else:
                            #         print(f'Skin {skin} not found')
                        master_collection = s2_put_into_collections(container, Path(model.name).stem, collection,
                                                                    False)
                        entity_data_holder = bpy.data.objects.new(Path(model.name).stem + '_ENT', None)
                        entity_data_holder['entity_data'] = {}
                        entity_data_holder['entity_data']['entity'] = obj['entity_data']['entity']
                        if container.armature:
                            entity_data_holder.parent = container.armature
                        elif container.objects:
                            entity_data_holder.parent = container.objects[0]
                        elif container.physics_objects:
                            entity_data_holder.parent = container.physics_objects[0]
                        else:
                            entity_data_holder.location = obj.location
                            entity_data_holder.rotation_euler = obj.rotation_euler
                            entity_data_holder.scale = obj.scale

                        master_collection.objects.link(entity_data_holder)
                        bpy.data.objects.remove(obj)
                    else:
                        self.report({'INFO'}, f"Model '{custom_prop_data['prop_path']}' not found!")
                elif model_type == '.mdl':
                    prop_path = Path(custom_prop_data['prop_path'])

                    container = BPSPropCache().get_object(prop_path)

                    if container is None:
                        mld_file = content_manager.find_file(prop_path)
                        vvd_file = content_manager.find_file(prop_path.with_suffix('.vvd'))
                        vvc_file = content_manager.find_file(prop_path.with_suffix('.vvc'))
                        vtx_file = find_vtx_cm(prop_path, content_manager)
                        model_container = import_model_from_files(prop_path, mld_file, vvd_file, vtx_file, vvc_file,
                                                                  1.0, False, True,
                                                                  unique_material_names=unique_material_names)
                    else:
                        model_container = container.clone()
                    if model_container is None:
                        continue
                    entity_data_holder = bpy.data.objects.new(model_container.mdl.header.name, None)
                    entity_data_holder['entity_data'] = {}
                    entity_data_holder['entity_data']['entity'] = obj['entity_data']['entity']

                    master_collection = s1_put_into_collections(model_container, prop_path.stem, collection, False)
                    master_collection.objects.link(entity_data_holder)

                    if model_container.armature is not None:
                        armature = model_container.armature
                        armature.rotation_mode = "XYZ"
                        entity_data_holder.parent = armature

                        bpy.context.view_layer.update()
                        armature.parent = obj.parent
                        armature.matrix_world = obj.matrix_world.copy()
                        armature.rotation_euler[2] += math.radians(90)
                    else:
                        if model_container.objects:
                            entity_data_holder.parent = model_container.objects[0]
                        else:
                            entity_data_holder.location = obj.location
                            entity_data_holder.rotation_euler = obj.rotation_euler
                            entity_data_holder.scale = obj.scale
                        for mesh_obj in model_container.objects:
                            mesh_obj.rotation_mode = "XYZ"
                            bpy.context.view_layer.update()
                            mesh_obj.parent = obj.parent
                            mesh_obj.matrix_world = obj.matrix_world.copy()

                    for mesh_obj in model_container.objects:
                        mesh_obj['prop_path'] = custom_prop_data['prop_path']
                    if is_vtflib_supported():
                        if container is None:
                            import_materials(model_container.mdl, unique_material_names=unique_material_names)
                    skin = custom_prop_data.get('skin', None)
                    if skin:
                        for model in model_container.objects:
                            if str(skin) in model['skin_groups']:
                                skin = str(skin)
                                skin_materials = model['skin_groups'][skin]
                                current_materials = model['skin_groups'][model['active_skin']]
                                print(skin_materials, current_materials)
                                for skin_material, current_material in zip(skin_materials, current_materials):
                                    if unique_material_names:
                                        skin_material = f"{Path(model_container.mdl.header.name).stem}_{skin_material[-63:]}"[
                                                        -63:]
                                        current_material = f"{Path(model_container.mdl.header.name).stem}_{current_material[-63:]}"[
                                                           -63:]
                                    else:
                                        skin_material = skin_material[-63:]
                                        current_material = current_material[-63:]

                                    swap_materials(model, skin_material, current_material)
                                model['active_skin'] = skin
                            else:
                                print(f'Skin {skin} not found')

                    bpy.data.objects.remove(obj)
        return {'FINISHED'}


class SOURCEIO_OT_ChangeSkin(bpy.types.Operator):
    bl_idname = "sourceio.select_skin"
    bl_label = "Change skin"
    bl_options = {'UNDO'}

    skin_name: bpy.props.StringProperty(name="skin_name", default="default")

    def execute(self, context):
        obj = context.active_object
        if obj.get('model_type', False):
            model_type = obj['model_type']
            if model_type == 's1':
                self.handle_s1(obj)
            elif model_type == 's2':
                self.handle_s2(obj)
            else:
                self.handle_s2(obj)

        obj['active_skin'] = self.skin_name

        return {'FINISHED'}

    def handle_s1(self, obj):
        prop_path = Path(obj['prop_path'])
        skin_materials = obj['skin_groups'][self.skin_name]
        current_materials = obj['skin_groups'][obj['active_skin']]
        unique_material_names = obj['unique_material_names']
        for skin_material, current_material in zip(skin_materials, current_materials):
            if unique_material_names:
                skin_material = f"{prop_path.stem}_{skin_material[-63:]}"[-63:]
                current_material = f"{prop_path.stem}_{current_material[-63:]}"[-63:]
            else:
                skin_material = skin_material[-63:]
                current_material = current_material[-63:]

            swap_materials(obj, skin_material, current_material)

    def handle_s2(self, obj):
        skin_material = obj['skin_groups'][self.skin_name]
        current_material = obj['skin_groups'][obj['active_skin']]

        mat_name = Path(skin_material).stem
        current_mat_name = Path(current_material).stem
        swap_materials(obj, mat_name, current_mat_name)


def swap_materials(obj, new_material_name, target_name):
    mat = bpy.data.materials.get(new_material_name, None) or bpy.data.materials.new(name=new_material_name)
    print(f'Swapping {target_name} with {new_material_name}')
    for n, obj_mat in enumerate(obj.data.materials):
        print(target_name, obj_mat.name)
        if obj_mat.name == target_name:
            print(obj_mat.name, "->", mat.name)
            obj.data.materials[n] = mat
            break


class UITools:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SourceIO"


class SOURCEIO_PT_Utils(UITools, bpy.types.Panel):
    bl_label = "SourceIO utils"
    bl_idname = "sourceio.utils"

    def draw(self, context):
        pass
        # self.layout.label(text="SourceIO Utils")

    @classmethod
    def poll(cls, context):
        obj: bpy.types.Object = context.active_object
        return obj and (obj.get("entity_data", None) or obj.get("skin_groups", None))


class SOURCEIO_PT_Placeholders(UITools, bpy.types.Panel):
    bl_label = 'Placeholders loading'
    bl_idname = 'sourceio.placeholders'
    bl_parent_id = "sourceio.utils"

    @classmethod
    def poll(cls, context):
        obj: bpy.types.Object = context.active_object
        if not obj and not context.selected_objects:
            obj = context.selected_objects[0]
        return obj and obj.get("entity_data", None)

    def draw(self, context):
        self.layout.label(text="Entity loading")
        obj: bpy.types.Object = context.active_object
        if obj.get("entity_data", None):
            entiry_data = obj['entity_data']
            entity_raw_data = entiry_data.get('entity', {})
            row = self.layout.row()
            row.label(text=f'Total selected entities:')
            row.label(text=str(len([obj for obj in context.selected_objects if 'entity_data' in obj])))
            if entiry_data.get('prop_path', False):
                box = self.layout.box()
                box.operator('sourceio.load_placeholder')
            box = self.layout.box()
            for k, v in entity_raw_data.items():
                row = box.row()
                row.label(text=f'{k}:')
                row.label(text=str(v))


class SOURCEIO_PT_SkinChanger(UITools, bpy.types.Panel):
    bl_label = 'Model skins'
    bl_idname = 'sourceio.skin_changer'
    bl_parent_id = "sourceio.utils"

    @classmethod
    def poll(cls, context):
        obj = context.active_object  # type:bpy.types.Object
        return obj and obj.get("skin_groups", None) is not None

    def draw(self, context):
        self.layout.label(text="Model skins")
        obj = context.active_object  # type:bpy.types.Object
        if obj.get("skin_groups", None):
            self.layout.label(text="Skins")
            box = self.layout.box()
            for skin, _ in obj['skin_groups'].items():
                row = box.row()
                op = row.operator('sourceio.select_skin', text=skin)
                op.skin_name = skin
                if skin == obj['active_skin']:
                    row.enabled = False


class SOURCEIO_PT_Scene(bpy.types.Panel):
    bl_label = 'SourceIO configuration'
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"
    bl_default_closed = True

    def draw(self, context):
        layout = self.layout
        layout.label(text="SourceIO configuration")
        box = layout.box()
        box.label(text='Mounted folders')
        box2 = box.box()
        for mount_name, mount in ContentManager().content_providers.items():
            box2.label(text=f'{mount_name}: {mount.root}')
