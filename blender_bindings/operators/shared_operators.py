import traceback
from hashlib import md5
from itertools import chain
from pathlib import Path
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty,
                       FloatProperty, StringProperty)
from bpy.types import (Panel,
                       Operator,
                       AddonPreferences,
                       PropertyGroup)
from idprop.types import *
import bpy

from ..source1.mdl.v44.import_mdl import import_static_animations
from ..utils.resource_utils import deserialize_mounted_content, serialize_mounted_content
from ...library.shared.content_providers.content_manager import ContentManager
from ...library.source2 import CompiledModelResource
from ...library.utils.path_utilities import find_vtx_cm
from ..source1.mdl import FileImport
from ..source1.mdl import put_into_collections as s1_put_into_collections
from ..source1.mdl.model_loader import import_model_from_files
from ..source1.mdl.v49.import_mdl import import_materials
from ..source2.vmdl_loader import load_model
from ..source2.vmdl_loader import \
    put_into_collections as s2_put_into_collections
from ..utils.utils import get_or_create_collection, find_layer_collection


def get_parent(collection):
    for pcoll in bpy.data.collections:
        if collection.name in pcoll.children:
            return pcoll
    return bpy.context.scene.collection


def get_collection(model_path: Path, *other_args):
    md_ = md5(model_path.as_posix().encode("ascii"))
    for key in other_args:
        if key:
            md_.update(key.encode("ascii"))
    key = md_.hexdigest()
    cache = bpy.context.scene.get("INSTANCE_CACHE", {})
    if key in cache:
        return cache[key]


def add_collection(model_path: Path, collection: bpy.types.Collection, *other_args):
    md_ = md5(model_path.as_posix().encode("ascii"))
    for key in other_args:
        if key:
            md_.update(key.encode("ascii"))
    key = md_.hexdigest()
    cache = bpy.context.scene.get("INSTANCE_CACHE", {})
    cache[key] = collection.name
    bpy.context.scene["INSTANCE_CACHE"] = cache


# noinspection PyPep8Naming
class SourceIO_OT_LoadEntity(Operator):
    bl_idname = "sourceio.load_placeholder"
    bl_label = "Load Entity"
    bl_options = {'UNDO'}

    use_bvlg: BoolProperty(default=True)

    def execute(self, context):
        content_manager = ContentManager()
        deserialize_mounted_content(content_manager)
        unique_material_names = True
        use_collections = context.scene.use_instances
        replace_entity = context.scene.replace_entity and not use_collections
        master_instance_collection = get_or_create_collection("MASTER_INSTANCES_DO_NOT_EDIT",
                                                              bpy.context.scene.collection)
        master_instance_lcollection = find_layer_collection(bpy.context.view_layer.layer_collection,
                                                            master_instance_collection.name)
        master_instance_lcollection.exclude = True
        # master_instance_collection.hide_viewport = True
        # master_instance_collection.hide_render = True
        if not use_collections:
            master_instance_collection = get_or_create_collection("MODELS",
                                                                  bpy.context.scene.collection)
        win = bpy.context.window_manager

        win.progress_begin(0, len(context.selected_objects))
        for n, obj in enumerate(context.selected_objects):
            print(f'Loading {obj.name}')
            win.progress_update(n)
            if obj.get("entity_data", None):
                custom_prop_data = obj['entity_data']
                prop_path = custom_prop_data.get('prop_path', None)
                if prop_path is None or custom_prop_data.get("imported", False):
                    continue
                model_type = Path(prop_path).suffix
                parent = get_parent(obj.users_collection[0])
                if model_type == '.vmdl_c':

                    instance_collection = get_collection(Path(prop_path))
                    if instance_collection and use_collections:
                        collection = bpy.data.collections.get(instance_collection, None)
                        if collection is not None:
                            obj.instance_type = 'COLLECTION'
                            obj.instance_collection = collection
                            obj["entity_data"]["prop_path"] = None
                            obj["entity_data"]["imported"] = True
                            continue

                    vmld_file = content_manager.find_file(prop_path)
                    if vmld_file:
                        # skin = custom_prop_data.get('skin', None)
                        model_resource = CompiledModelResource.from_buffer(vmld_file, Path(prop_path))
                        container = load_model(model_resource, custom_prop_data["scale"], lod_mask=1)
                        if replace_entity:
                            s2_put_into_collections(container, model_resource.name, parent)
                        else:
                            s2_put_into_collections(container, model_resource.name, master_instance_collection)
                        obj["entity_data"]["prop_path"] = None
                        obj["entity_data"]["imported"] = True

                        if use_collections:
                            add_collection(Path(prop_path), container.collection)

                            obj.instance_type = 'COLLECTION'
                            obj.instance_collection = container.collection
                            continue
                        if replace_entity:
                            if container.armature:
                                container.armature.location = obj.location
                                container.armature.rotation_mode = "XYZ"
                                container.armature.rotation_euler = obj.rotation_euler
                                container.armature.scale = obj.scale
                                container.armature.name = obj.name
                            else:
                                for ob in chain(container.objects,
                                                container.physics_objects):  # type:bpy.types.Object
                                    ob.location = obj.location
                                    ob.rotation_mode = "XYZ"
                                    ob.rotation_euler = obj.rotation_euler
                                    ob.scale = obj.scale
                            bpy.data.objects.remove(obj)
                        else:
                            if container.armature:
                                container.armature.parent = obj
                            else:
                                for o in container.objects:
                                    o.parent = obj
                                for o in container.physics_objects:
                                    o.parent = obj
                    else:
                        self.report({'INFO'}, f"Model '{prop_path}' not found!")
                elif model_type == '.mdl':
                    default_anim = custom_prop_data["entity"].get("defaultanim", None)
                    prop_path = Path(prop_path)

                    instance_collection = get_collection(prop_path, default_anim)
                    if instance_collection and use_collections:
                        collection = bpy.data.collections.get(instance_collection, None)
                        if collection is not None:
                            obj.instance_type = 'COLLECTION'
                            obj.instance_collection = collection
                            obj["entity_data"]["prop_path"] = None
                            obj["entity_data"]["imported"] = True
                            continue

                    mdl_file = content_manager.find_file(prop_path)
                    vvd_file = content_manager.find_file(prop_path.with_suffix('.vvd'))
                    vvc_file = content_manager.find_file(prop_path.with_suffix('.vvc'))
                    phy_file = content_manager.find_file(prop_path.with_suffix('.phy'))
                    vtx_file = find_vtx_cm(prop_path, content_manager)
                    if mdl_file is None or vvd_file is None or vtx_file is None:
                        self.report({"WARNING"}, f"Failed to find mdl/vvd/vtx file for {obj.name}({prop_path}) prop")
                        continue
                    file_list = FileImport(mdl_file, vvd_file, vtx_file,
                                           vvc_file if vvc_file else None,
                                           phy_file if phy_file else None)
                    if not file_list.is_valid():
                        self.report({"WARNING"},
                                    f"Mdl file for {obj.name}({prop_path}) prop is invalid. Too small file or missing file")
                        continue
                    model_container = import_model_from_files(prop_path, file_list, 1.0, False, True,
                                                              unique_material_names=unique_material_names)
                    if model_container is None:
                        continue
                    import_materials(model_container.mdl, unique_material_names=unique_material_names,
                                     use_bvlg=context.scene.use_bvlg)

                    s1_put_into_collections(model_container, prop_path.stem, master_instance_collection, False)

                    if default_anim is not None and model_container.armature is not None:
                        try:
                            import_static_animations(content_manager, model_container.mdl, default_anim,
                                                     model_container.armature, 1.0)
                        except RuntimeError:
                            self.report({"WARNING"}, "Failed to load animation")
                            traceback.print_exc()

                    obj["entity_data"]["prop_path"] = None
                    obj["entity_data"]["imported"] = True
                    if use_collections:
                        add_collection(prop_path, model_container.collection, default_anim)

                        obj.instance_type = 'COLLECTION'
                        obj.instance_collection = model_container.collection
                        continue

                    if replace_entity:
                        if model_container.armature:
                            model_container.armature.location = obj.location
                            model_container.armature.rotation_mode = "XYZ"
                            model_container.armature.rotation_euler = obj.rotation_euler
                            model_container.armature.scale = obj.scale
                            model_container.armature.name = obj.name
                            model_container.armature["entity_data"] = obj["entity_data"]
                            model_container.armature["entity_data"]["prop_path"] = None
                            model_container.armature["entity_data"]["imported"] = True
                        else:
                            for ob in model_container.objects:  # type:bpy.types.Object
                                ob.location = obj.location
                                ob.rotation_mode = "XYZ"
                                ob.rotation_euler = obj.rotation_euler
                                ob.scale = obj.scale
                                ob["entity_data"] = obj["entity_data"]
                                ob["entity_data"]["prop_path"] = None
                                ob["entity_data"]["imported"] = True
                        bpy.data.objects.remove(obj)
                    else:
                        if model_container.armature:
                            model_container.armature.parent = obj
                        else:
                            for o in model_container.objects:
                                o.parent = obj

                    # entity_data_holder = bpy.data.objects.new(model_container.mdl.header.name, None)
                    # entity_data_holder['entity_data'] = {}
                    # entity_data_holder['entity_data']['entity'] = obj['entity_data']['entity']
                    #
                    # master_collection = s1_put_into_collections(model_container, prop_path.stem, collection, False)
                    # master_collection.objects.link(entity_data_holder)
                    #
                    # if model_container.armature is not None:
                    #     armature = model_container.armature
                    #     armature.rotation_mode = "XYZ"
                    #     entity_data_holder.parent = armature
                    #
                    #     bpy.context.view_layer.update()
                    #     armature.parent = obj.parent
                    #     armature.matrix_world = obj.matrix_world.copy()
                    #     armature.rotation_euler[2] += math.radians(90)
                    # else:
                    #     if model_container.objects:
                    #         entity_data_holder.parent = model_container.objects[0]
                    #     else:
                    #         entity_data_holder.location = obj.location
                    #         entity_data_holder.rotation_euler = obj.rotation_euler
                    #         entity_data_holder.scale = obj.scale
                    #     for mesh_obj in model_container.objects:
                    #         mesh_obj.rotation_mode = "XYZ"
                    #         bpy.context.view_layer.update()
                    #         mesh_obj.parent = obj.parent
                    #         mesh_obj.matrix_world = obj.matrix_world.copy()
                    #
                    # for mesh_obj in model_container.objects:
                    #     mesh_obj['prop_path'] = prop_path
                    # if container is None:
                    #     import_materials(model_container.mdl, unique_material_names=unique_material_names)
                    # skin = custom_prop_data.get('skin', None)
                    # if skin:
                    #     for model in model_container.objects:
                    #         if str(skin) in model['skin_groups']:
                    #             skin = str(skin)
                    #             skin_materials = model['skin_groups'][skin]
                    #             current_materials = model['skin_groups'][model['active_skin']]
                    #             print(skin_materials, current_materials)
                    #             for skin_material, current_material in zip(skin_materials, current_materials):
                    #                 if unique_material_names:
                    #                     skin_material = f"{Path(model_container.mdl.header.name).stem}_{skin_material[-63:]}"[
                    #                                     -63:]
                    #                     current_material = f"{Path(model_container.mdl.header.name).stem}_{current_material[-63:]}"[
                    #                                        -63:]
                    #                 else:
                    #                     skin_material = skin_material[-63:]
                    #                     current_material = current_material[-63:]
                    #
                    #                 swap_materials(model, skin_material, current_material)
                    #             model['active_skin'] = skin
                    #         else:
                    #             print(f'Skin {skin} not found')
                    #
                    # bpy.data.objects.remove(obj)

        win.progress_end()

        return {'FINISHED'}


# noinspection PyPep8Naming
class SOURCEIO_OT_ChangeSkin(Operator):
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


# noinspection PyPep8Naming
class SOURCEIO_PT_Utils(UITools, Panel):
    bl_label = "SourceIO utils"
    bl_idname = "SOURCEIO_PT_Utils"

    def draw(self, context):
        pass
        # self.layout.label(text="SourceIO Utils")

    @classmethod
    def poll(cls, context):
        obj: bpy.types.Object = context.active_object
        return obj and (obj.get("entity_data", None) or obj.get("skin_groups", None))


# noinspection PyPep8Naming
class SOURCEIO_PT_EntityLoader(UITools, Panel):
    bl_label = 'Entity loader'
    bl_idname = 'SOURCEIO_PT_EntityLoader'
    bl_parent_id = "SOURCEIO_PT_Utils"

    @classmethod
    def poll(cls, context):
        obj: bpy.types.Object = context.active_object
        if not obj and not context.selected_objects:
            obj = context.selected_objects[0]
        return obj and obj.get("entity_data", None)

    def draw(self, context):
        self.layout.label(text="Entity loading")
        layout = self.layout.box()
        layout.prop(context.scene, "use_bvlg")
        layout.prop(context.scene, "use_instances")
        if not context.scene.use_instances:
            layout.prop(context.scene, "replace_entity")
        obj: bpy.types.Object = context.active_object
        if obj is None and context.selected_objects:
            obj = context.selected_objects[0]
        if obj.get("entity_data", None):
            entity_data = obj['entity_data']
            row = self.layout.row()
            row.label(text=f'Total selected entities:')
            row.label(text=str(len([obj for obj in context.selected_objects if 'entity_data' in obj])))
            if entity_data.get('prop_path', False):
                box = self.layout.box()
                box.operator('sourceio.load_placeholder')


# noinspection PyPep8Naming
class SOURCEIO_PT_EntityInfo(UITools, Panel):
    bl_label = 'Entity Info'
    bl_idname = 'SOURCEIO_PT_EntityInfo'
    bl_parent_id = "SOURCEIO_PT_Utils"

    @classmethod
    def poll(cls, context):
        obj: bpy.types.Object = context.active_object
        if not obj and not context.selected_objects:
            obj = context.selected_objects[0]
        return obj and obj.get("entity_data", None)

    def draw(self, context):
        self.layout.label(text="Entity info")
        obj: bpy.types.Object = context.active_object
        if obj is None and context.selected_objects:
            obj = context.selected_objects[0]
        if obj.get("entity_data", None):
            entity_data = obj['entity_data']
            entity_raw_data = entity_data.get('entity', {})
            box = self.layout.box()
            for k, v in entity_raw_data.items():
                row = box.row()
                row.label(text=f'{k}:')
                if isinstance(v, IDPropertyArray):
                    row.label(text=str(v.to_list()))
                elif isinstance(v, IDPropertyGroup):
                    row.label(text=str(v.to_dict()))
                else:
                    row.label(text=str(v))


class SOURCEIO_PT_SkinChanger(UITools, Panel):
    bl_label = 'Model skins'
    bl_idname = 'SOURCEIO_PT_SkinChanger'
    bl_parent_id = "SOURCEIO_PT_Utils"

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


class SOURCEIO_UL_MountedResource(PropertyGroup):
    name: StringProperty(
        name="Name",
        description="A name for this resource",
        default="Untitled")

    path: StringProperty(
        name="Path",
        description="Path to the resource",
        default="",
        subtype='FILE_PATH')

    hash: StringProperty(
        name="Hash",
        description="Hash of the resource",
        default="")


class SOURCEIO_OT_NewResource(Operator):
    bl_idname = "sourceio.new_resource"
    bl_label = "Add New Resource"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        cm = ContentManager()
        serialize_mounted_content(cm)
        cm.clean()
        new_resource = context.scene.mounted_resources.add()
        new_resource.path = self.filepath
        new_resource.name = Path(self.filepath).name
        cm.scan_for_content(Path(self.filepath))
        deserialize_mounted_content(cm)
        serialize_mounted_content(cm)

        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class SOURCEIO_OT_DeleteResource(Operator):
    bl_idname = "sourceio.delete_resource"
    bl_label = "Delete Resource"

    @classmethod
    def poll(cls, context):
        return context.scene.mounted_resources_index >= 0

    def execute(self, context):
        resources = context.scene.mounted_resources
        index = context.scene.mounted_resources_index

        resources.remove(index)

        if index > 0:
            context.scene.mounted_resources_index = index - 1

        return {'FINISHED'}


class SOURCEIO_OT_CleanResources(Operator):
    bl_idname = "sourceio.clean_resources"
    bl_label = "Clean All Resources"

    def execute(self, context):
        resources = context.scene.mounted_resources

        # Remove all resources
        for i in range(len(resources)):
            resources.remove(0)

        return {'FINISHED'}


class SOURCEIO_UL_ResourcesList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.prop(item, "name", text="", emboss=False, icon='FILE_TICK')
            layout.prop(item, "path", text="")
        elif self.layout_type in {'GRID'}:
            layout.alignment = 'CENTER'
            layout.label(text="", icon_value=icon)


class SOURCEIO_PT_ResourcesPanel(Panel):
    bl_idname = "SOURCEIO_PT_resources"
    bl_label = "Mounted Resources"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Resources"

    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.template_list("SOURCEIO_UL_ResourcesList", "", context.scene, "mounted_resources", context.scene,
                          "mounted_resources_index")

        col = row.column(align=True)
        col.operator("sourceio.new_resource", icon='ADD', text="")
        col.operator("sourceio.delete_resource", icon='REMOVE', text="")


class SOURCEIO_OT_ResourceMove(bpy.types.Operator):
    bl_idname = "sourceio.move_resource"
    bl_label = "Move Resource"
    bl_options = {'REGISTER', 'UNDO'}

    direction: bpy.props.EnumProperty(
        items=(
            ('UP', "Up", ""),
            ('DOWN', "Down", ""),
        )
    )

    @classmethod
    def poll(cls, context):
        return context.scene.mounted_resources_index >= 0

    def move_index(self):
        # Move index of an item render queue while clamping it
        index = bpy.context.scene.mounted_resources_index
        list_length = len(bpy.context.scene.mounted_resources) - 1  # (index starts at 0)
        new_index = index + (-1 if self.direction == 'UP' else 1)
        bpy.context.scene.mounted_resources_index = max(0, min(new_index, list_length))

    def execute(self, context):
        resources = context.scene.mounted_resources
        index = context.scene.mounted_resources_index
        direction = self.direction

        neighbor_index = index - 1 if direction == 'UP' else index + 1
        resources.move(neighbor_index, index)

        self.move_index()

        return {'FINISHED'}


class SOURCEIO_PT_Scene(Panel):
    bl_label = 'SourceIO configuration'
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"
    bl_default_closed = True

    def draw(self, context):
        layout = self.layout
        layout.label(text="SourceIO configuration")
        layout.prop(context.scene, "TextureCachePath")

        box = layout.box()
        box.label(text='Mounted Resources')

        row = box.row()
        row.template_list("SOURCEIO_UL_ResourcesList", "", context.scene, "mounted_resources", context.scene,
                          "mounted_resources_index")

        col = row.column(align=True)
        col.operator("sourceio.new_resource", icon='ADD', text="")
        col.operator("sourceio.delete_resource", icon='REMOVE', text="")
        col.operator("sourceio.clean_resources", icon='X', text="")
        col.separator()
        col.operator("sourceio.move_resource", icon='TRIA_UP', text="").direction = 'UP'
        col.operator("sourceio.move_resource", icon='TRIA_DOWN', text="").direction = 'DOWN'


shared_classes = (
    SOURCEIO_UL_MountedResource,
    SOURCEIO_OT_NewResource,
    SOURCEIO_OT_DeleteResource,
    SOURCEIO_OT_CleanResources,
    SOURCEIO_PT_ResourcesPanel,
    SOURCEIO_UL_ResourcesList,
    SOURCEIO_OT_ResourceMove,
    SOURCEIO_PT_Scene,
    SOURCEIO_PT_Utils,
    SOURCEIO_PT_EntityLoader,
    SOURCEIO_PT_EntityInfo,
    SOURCEIO_PT_SkinChanger,
    SourceIO_OT_LoadEntity,
    SOURCEIO_OT_ChangeSkin,
)
