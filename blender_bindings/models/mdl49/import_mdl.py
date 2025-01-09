import warnings
from collections import defaultdict

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

from SourceIO.blender_bindings.models.common import merge_meshes
from SourceIO.blender_bindings.models.mdl44.import_mdl import create_armature
from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.blender_bindings.utils.bpy_utils import add_material, is_blender_4_1, get_or_create_material
from SourceIO.library.models.mdl.structs.header import StudioHDRFlags
from SourceIO.library.models.mdl.v44.vertex_animation_cache import preprocess_vertex_animation
from SourceIO.library.models.mdl.v49.flex_expressions import *
from SourceIO.library.models.mdl.v49.mdl_file import MdlV49
from SourceIO.library.models.vtx.v7.vtx import Vtx
from SourceIO.library.models.vvd import Vvd
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.utils.common import get_slice
from SourceIO.library.utils.path_utilities import path_stem, collect_full_material_names
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('Source1::ModelLoader')


def import_model(content_manager: ContentManager, mdl: MdlV49, vtx: Vtx, vvd: Vvd,
                 scale=1.0, create_drivers=False, load_refpose=False):
    full_material_names = collect_full_material_names([mat.name for mat in mdl.materials], mdl.materials_paths,
                                                      content_manager)

    objects = []
    bodygroups = defaultdict(list)
    attachments = []
    desired_lod = 0
    all_vertices = vvd.lod_data[desired_lod]

    static_prop = mdl.header.flags & StudioHDRFlags.STATIC_PROP != 0
    armature = None
    vertex_anim_cache = preprocess_vertex_animation(mdl, vvd)

    if not static_prop:
        armature = create_armature(mdl, scale, load_refpose)

    for vtx_body_part, body_part in zip(vtx.body_parts, mdl.body_parts):
        for vtx_model, model in zip(vtx_body_part.models, body_part.models):

            if model.vertex_count == 0:
                continue
            object_name = model.name
            mesh_name = f'{mdl.header.name}_{body_part.name}_{object_name}_MESH'

            mesh_data = bpy.data.meshes.new(mesh_name)
            mesh_obj = bpy.data.objects.new(object_name, mesh_data)
            if getattr(mdl, 'material_mapper', None):
                material_mapper = mdl.material_mapper
                true_skin_groups = {str(n): list(map(lambda a: material_mapper.get(a.material_pointer), group)) for (n, group) in enumerate(mdl.skin_groups)}
                for key, value in true_skin_groups.items():
                    while None in value:
                        value.remove(None)
                try:
                    mesh_obj['skin_groups'] = true_skin_groups
                except:
                    mesh_obj['skin_groups'] = {str(n): list(map(lambda a: a.name, group)) for (n, group) in enumerate(mdl.skin_groups)}
            else:
                mesh_obj['skin_groups'] = {str(n): list(map(lambda a: a.name, group)) for (n, group) in enumerate(mdl.skin_groups)}
            mesh_obj['active_skin'] = '0'
            mesh_obj['model_type'] = 's1'

            objects.append(mesh_obj)
            bodygroups[body_part.name].append(mesh_obj)
            mesh_obj['prop_path'] = path_stem(mdl.header.name)

            model_vertices = get_slice(all_vertices, model.vertex_offset, model.vertex_count)
            vtx_vertices, indices_array, material_indices_array = merge_meshes(model, vtx_model.model_lods[desired_lod])

            indices_array = np.array(indices_array, dtype=np.uint32)
            vertices = model_vertices[vtx_vertices]
            vertices_vertex = vertices['vertex']

            mesh_data.from_pydata(vertices_vertex * scale, [], np.flip(indices_array).reshape((-1, 3)))
            mesh_data.update()

            mesh_data.polygons.foreach_set("use_smooth", np.ones(len(mesh_data.polygons), np.uint32))
            mesh_data.normals_split_custom_set_from_vertices(vertices['normal'])
            if not is_blender_4_1():
                mesh_data.use_auto_smooth = True

            material_remapper = np.zeros((material_indices_array.max() + 1,), dtype=np.uint32)
            for mat_id in np.unique(material_indices_array):
                mat_name = mdl.materials[mat_id].name
                material = get_or_create_material(mat_name, full_material_names[mat_name])
                material_remapper[mat_id] = add_material(material, mesh_obj)

            mesh_data.polygons.foreach_set('material_index', material_remapper[material_indices_array[::-1]])

            vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
            mesh_data.loops.foreach_get('vertex_index', vertex_indices)

            uv_data = mesh_data.uv_layers.new()
            uvs = vertices['uv']
            uvs[:, 1] = 1 - uvs[:, 1]
            uv_data.data.foreach_set('uv', uvs[vertex_indices].flatten())

            if vvd.extra_data:
                for extra_type, extra_data in vvd.extra_data.items():
                    extra_data = extra_data.reshape((-1, 2))
                    extra_uv = get_slice(extra_data, model.vertex_offset, model.vertex_count)
                    extra_uv = extra_uv[vtx_vertices]
                    uv_data = mesh_data.uv_layers.new(name=extra_type.name)
                    extra_uv[:, 1] = 1 - extra_uv[:, 1]
                    uv_data.data.foreach_set('uv', extra_uv[vertex_indices].flatten())

            if not static_prop:
                modifier = mesh_obj.modifiers.new(
                    type="ARMATURE", name="Armature")
                modifier.object = armature
                mesh_obj.parent = armature

                weight_groups = {bone.name: mesh_obj.vertex_groups.new(name=bone.name) for bone in mdl.bones}

                for n, (bone_indices, bone_weights) in enumerate(zip(vertices['bone_id'], vertices['weight'])):
                    for bone_index, weight in zip(bone_indices, bone_weights):
                        if weight > 0:
                            bone_name = mdl.bones[bone_index].name
                            weight_groups[bone_name].add([n], weight, 'REPLACE')

                flexes = []
                for mesh in model.meshes:
                    if mesh.flexes:
                        flexes.extend([(mdl.flex_names[flex.flex_desc_index], flex) for flex in mesh.flexes])

                if flexes:
                    mesh_obj.shape_key_add(name='base')
                    for flex_name, flex_desc in flexes:
                        vertex_animation = vertex_anim_cache[flex_name]
                        flex_delta = get_slice(vertex_animation["pos"], model.vertex_offset, model.vertex_count)
                        flex_delta = flex_delta[vtx_vertices] * scale
                        model_vertices = get_slice(all_vertices['vertex'], model.vertex_offset, model.vertex_count)
                        model_vertices = model_vertices[vtx_vertices] * scale

                        if create_drivers and flex_desc.partner_index:
                            partner_name = mdl.flex_names[flex_desc.partner_index]
                            partner_shape_key = (mesh_data.shape_keys.key_blocks.get(partner_name, None) or
                                                 mesh_obj.shape_key_add(name=partner_name))
                            shape_key = (mesh_data.shape_keys.key_blocks.get(flex_name, None) or
                                         mesh_obj.shape_key_add(name=flex_name))

                            balance = model_vertices[:, 0]
                            balance_width = (model_vertices.max() - model_vertices.min()) * (1 - (99.3 / 100))
                            balance = np.clip((-balance / balance_width / 2) + 0.5, 0, 1)

                            flex_vertices = (flex_delta * balance[:, None]) + model_vertices
                            shape_key.data.foreach_set("co", flex_vertices.reshape(-1))

                            p_balance = 1 - balance
                            p_flex_vertices = (flex_delta * p_balance[:, None]) + model_vertices
                            partner_shape_key.data.foreach_set("co", p_flex_vertices.reshape(-1))
                        else:
                            shape_key = mesh_data.shape_keys.key_blocks.get(flex_name, None) or mesh_obj.shape_key_add(
                                name=flex_name)

                            shape_key.data.foreach_set("co", (flex_delta + model_vertices).ravel())
                    if create_drivers:
                        create_flex_drivers(mesh_obj, mdl)
                mesh_data.validate()
    if mdl.attachments:
        attachments = create_attachments(mdl, armature if not static_prop else objects[0], scale)

    return ModelContainer(objects, bodygroups, [], attachments, armature, None)


def create_flex_drivers(obj, mdl: MdlV49):
    from ...operators.flex_operators import SourceIO_PG_FlexController
    if not obj.data.shape_keys:
        return
    all_exprs = mdl.rebuild_flex_rules()
    data = obj.data
    shape_key_block = data.shape_keys

    def _parse_simple_flex(missing_flex_name: str):
        flexes = missing_flex_name.split('_')
        if not all(flex in data.flex_controllers for flex in flexes):
            return None
        return Combo([FetchController(flex) for flex in flexes]), [(flex, 'fetch2') for flex in flexes]

    st = '\n    '

    for flex_controller_ui in mdl.flex_ui_controllers:
        cont: SourceIO_PG_FlexController = data.flex_controllers.add()

        if flex_controller_ui.nway_controller:
            nway_cont: SourceIO_PG_FlexController = data.flex_controllers.add()
            nway_cont.stereo = False
            multi_controller = next(filter(lambda a: a.name == flex_controller_ui.nway_controller, mdl.flex_controllers)
                                    )
            nway_cont.name = flex_controller_ui.nway_controller
            nway_cont.set_from_controller(multi_controller)

        if flex_controller_ui.stereo:
            left_controller = next(
                filter(lambda a: a.name == flex_controller_ui.left_controller, mdl.flex_controllers)
            )
            right_controller = next(
                filter(lambda a: a.name == flex_controller_ui.right_controller, mdl.flex_controllers)
            )
            cont.stereo = True
            cont.name = flex_controller_ui.name
            assert left_controller.max == right_controller.max
            assert left_controller.min == right_controller.min
            cont.set_from_controller(left_controller)
        else:
            controller = next(filter(lambda a: a.name == flex_controller_ui.controller, mdl.flex_controllers))
            cont.stereo = False
            cont.name = flex_controller_ui.name
            cont.set_from_controller(controller)
    blender_py_file = """
import bpy

def rclamped(val, a, b, c, d):
    if ( a == b ):
        return d if val >= b else c;
    return c + (d - c) * min(max((val - a) / (b - a), 0.0), 1.0)
    
def clamp(val, a, b):
    return min(max(val, a), b)

def nway(multi_value, flex_value, x, y, z, w):
    if multi_value <= x or multi_value >= w:  # outside of boundaries
        multi_value = 0.0
    elif multi_value <= y:
        multi_value = rclamped(multi_value, x, y, 0.0, 1.0)
    elif multi_value >= z:
        multi_value = rclamped(multi_value, z, w, 1.0, 0.0)
    else:
        multi_value = 1.0
    return multi_value * flex_value


def combo(*values):
    val = values[0]
    for v in values[1:]:
        val*=v
    return val
    
def dom(dm, *values):
    val = 1
    for v in values:
        val *= v
    return val * (1 - dm)

def lower_eyelid_case(eyes_up_down,close_lid_v,close_lid):
    if eyes_up_down > 0.0:
        return (1.0 - eyes_up_down) * (1.0 - close_lid_v) * close_lid
    else:
        return  (1.0 - close_lid_v) * close_lid

def upper_eyelid_case(eyes_up_down,close_lid_v,close_lid):
    if eyes_up_down > 0.0:
        return (1.0 + eyes_up_down) * close_lid_v * close_lid
    else:
        return  close_lid_v * close_lid


bpy.app.driver_namespace["combo"] = combo
bpy.app.driver_namespace["dom"] = dom
bpy.app.driver_namespace["nway"] = nway
bpy.app.driver_namespace["rclamped"] = rclamped

    """

    def normalize_name(name):
        return name.replace("-", "_").replace(" ", "_")

    for flex_name, (expr, inputs) in all_exprs.items():
        normalized_flex_name = normalize_name(flex_name)
        driver_name = f"{normalized_flex_name}_driver"
        if driver_name in globals():
            continue

        input_definitions = []
        for inp in inputs:
            input_name = inp[0]
            normalized_input_name = normalize_name(inp[0])
            if inp[1] in ('fetch1', '2WAY1', '2WAY0', 'NWAY', 'DUE'):
                if 'left_' in input_name:
                    input_definitions.append(
                        f'{normalized_input_name} = obj_data.flex_controllers["{input_name.replace("left_", "")}"].value_left')
                elif 'right_' in input_name:
                    input_definitions.append(
                        f'{normalized_input_name} = obj_data.flex_controllers["{input_name.replace("right_", "")}"].value_right')
                else:
                    input_definitions.append(
                        f'{normalized_input_name} = obj_data.flex_controllers["{inp[0]}"].value')
            elif inp[1] == 'fetch2':
                input_definitions.append(
                    f'{normalized_input_name} = obj_data.shape_keys.key_blocks["{input_name}"].value')
            else:
                raise NotImplementedError(f'"{inp[1]}" is not supported')
        print(f"{flex_name} = {expr}")
        template_function = f"""
def {driver_name}(obj_data):
    {st.join(input_definitions)}
    return {expr}
bpy.app.driver_namespace["{driver_name}"] = {driver_name}

"""
        blender_py_file += template_function

    for shape_key in shape_key_block.key_blocks:

        flex_name = shape_key.name
        normalized_flex_name = normalize_name(flex_name)

        if flex_name == 'base':
            continue
        if flex_name not in all_exprs:
            warnings.warn(f'Rule for {flex_name} not found! Generating basic rule.')
            expr, inputs = _parse_simple_flex(flex_name) or (None, None)
            if not expr or not inputs:
                warnings.warn(f'Failed to generate basic rule for {flex_name}!')
                cont: SourceIO_PG_FlexController = data.flex_controllers.add()
                cont.name = flex_name
                cont.mode = 1
                cont.value_min = 0
                cont.value_max = 1
                template_function = f"""
def {normalized_flex_name}_driver(obj_data):
    return obj_data.flex_controllers["{flex_name}"].value
bpy.app.driver_namespace["{normalized_flex_name}_driver"] = {normalized_flex_name}_driver

                                """
                blender_py_file += template_function
            else:
                template_function = f"""
def {normalized_flex_name}_driver(obj_data):
    {st.join([i[0] for i in inputs])}
    return {expr}
bpy.app.driver_namespace["{normalized_flex_name}_driver"] = {normalized_flex_name}_driver

                """
                blender_py_file += template_function

        shape_key.driver_remove("value")
        fcurve = shape_key.driver_add("value")
        fcurve.modifiers.remove(fcurve.modifiers[0])

        driver = fcurve.driver
        driver.type = 'SCRIPTED'
        driver.expression = f"{normalized_flex_name}_driver(obj_data)"
        var = driver.variables.new()
        var.name = 'obj_data'
        var.targets[0].id_type = 'OBJECT'
        var.targets[0].id = obj
        var.targets[0].data_path = f"data"

    driver_file = bpy.data.texts.new(f'{mdl.header.name}.py')
    driver_file.write(blender_py_file)
    driver_file.use_module = True


def create_attachments(mdl: MdlV49, armature: bpy.types.Object, scale):
    attachments = []
    for attachment in mdl.attachments:
        empty = bpy.data.objects.new(attachment.name, None)
        pos = Vector(attachment.pos) * scale
        rot = Euler(attachment.rot)

        empty.matrix_basis.identity()
        empty.scale *= scale
        empty.location = pos
        empty.rotation_euler = rot

        if armature.type == 'ARMATURE':
            modifier = empty.constraints.new(type="CHILD_OF")
            modifier.target = armature
            modifier.subtarget = mdl.bones[attachment.parent_bone].name
            modifier.inverse_matrix.identity()

        attachments.append(empty)

    return attachments


def import_animations(cm: ContentProvider, mdl: MdlV49, armature: bpy.types.Object,
                      scale: float):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.scene.collection.objects.link(armature)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    if not armature.animation_data:
        armature.animation_data_create()

    for n, anim in enumerate(mdl.anim_descs):
        animation_data = mdl.animations[n]
        action = bpy.data.actions.new(anim.name)
        action.use_fake_user = True
        armature.animation_data.action = action
        curve_per_bone = {}

        for bone in mdl.bones:
            bone_name = bone.name
            bl_bone = armature.pose.bones.get(bone_name)
            bl_bone.rotation_mode = 'QUATERNION'
            bone_string = f'pose.bones["{bone_name}"].'
            group = action.groups.new(name=bone_name)
            pos_curves = []
            rot_curves = []
            for i in range(3):
                pos_curve = action.fcurves.new(data_path=bone_string + "location", index=i)
                pos_curve.keyframe_points.add(anim.frame_count)
                pos_curve.auto_smoothing = "CONT_ACCEL"
                pos_curves.append(pos_curve)
                pos_curve.group = group
            for i in range(4):
                rot_curve = action.fcurves.new(data_path=bone_string + "rotation_quaternion", index=i)
                rot_curve.keyframe_points.add(anim.frame_count)
                rot_curve.auto_smoothing = "CONT_ACCEL"
                rot_curves.append(rot_curve)
                rot_curve.group = group
            curve_per_bone[bone_name] = pos_curves, rot_curves
        for bone_id, bone in enumerate(mdl.bones):
            for frame_id in range(anim.frame_count):
                bl_bone = armature.pose.bones.get(bone.name)
                pos_curves, rot_curves = curve_per_bone[bone.name]
                anim_data = animation_data[frame_id, bone_id]
                bl_bone.matrix_basis.identity()
                pos = Vector(anim_data["pos"]) * scale
                x, y, z, w = anim_data["rot"]
                rot = Quaternion((w, x, y, z))
                mat = Matrix.Translation(pos) @ rot.to_matrix().to_4x4()

                if bl_bone.parent:

                    mat = bl_bone.parent.matrix @ mat if bl_bone.parent else mat
                    bl_bone.matrix = mat
                    pos, rot = bl_bone.location, bl_bone.rotation_quaternion
                    for i in range(3):
                        pos_curves[i].keyframe_points[frame_id].co = (frame_id, (pos[i]))

                    for i in range(4):
                        rot_curves[i].keyframe_points[frame_id].co = (frame_id, (rot[i]))
                else:
                    mat = bl_bone.matrix.inverted() @ mat
                    pos, rot, scl = mat.decompose()
                    for i in range(3):
                        pos_curves[i].keyframe_points[frame_id].co = (frame_id, (pos[i]))

                    for i in range(4):
                        rot_curves[i].keyframe_points[frame_id].co = (frame_id, (rot[i]))
                bl_bone.matrix = Matrix.Identity(4)
        for pos_curves, rot_curves in curve_per_bone.values():
            for curve in rot_curves + pos_curves:
                curve.update()
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.scene.collection.objects.unlink(armature)
