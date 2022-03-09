import warnings
from pathlib import Path
from typing import BinaryIO, Iterable, Sized, Union, Optional

import bpy

from bpy.props import FloatProperty

import numpy as np
from mathutils import Vector, Matrix, Euler, Quaternion

from .....logger import SLoggingManager
from .....library.shared.content_providers.content_manager import ContentManager

from .....library.source1.vvd import Vvd
from .....library.source1.vvc import Vvc

from .....library.source1.vtx.v7.structs.mesh import Mesh as VtxMesh
from .....library.source1.vtx.v7.structs.model import ModelLod as VtxModel
from .....library.source1.vtx.v7.vtx import Vtx

from .....library.source1.mdl.v49.flex_expressions import *
from .....library.source1.mdl.v49.mdl_file import MdlV49
from .....library.source1.mdl.structs.header import StudioHDRFlags
from .....library.source1.mdl.structs.model import ModelV49
from .....library.source1.mdl.v44.vertex_animation_cache import VertexAnimationCache

from ....shared.model_container import Source1ModelContainer
from ....material_loader.material_loader import Source1MaterialLoader
from ....material_loader.shaders.source1_shader_base import Source1ShaderBase
from ....utils.utils import get_material, get_new_unique_collection
from .....library.utils.math_utilities import euler_to_quat

log_manager = SLoggingManager()
logger = log_manager.get_logger('Source1::ModelLoader')


def collect_full_material_names(mdl: MdlV49):
    content_manager = ContentManager()
    full_mat_names = {}
    for material_path in mdl.materials_paths:
        for material in mdl.materials:
            real_material_path = content_manager.find_material(Path(material_path) / material.name)
            if real_material_path is not None:
                full_mat_names[material] = str(Path(material_path) / material.name)
    return full_mat_names


def merge_strip_groups(vtx_mesh: VtxMesh):
    indices_accumulator = []
    vertex_accumulator = []
    vertex_offset = 0
    for strip_group in vtx_mesh.strip_groups:
        indices_accumulator.append(np.add(strip_group.indexes, vertex_offset))
        vertex_accumulator.append(strip_group.vertexes['original_mesh_vertex_index'].reshape(-1))
        vertex_offset += sum(strip.vertex_count for strip in strip_group.strips)
    return np.hstack(indices_accumulator), np.hstack(vertex_accumulator), vertex_offset


def merge_meshes(model: ModelV49, vtx_model: VtxModel):
    vtx_vertices = []
    acc = 0
    mat_arrays = []
    indices_array = []
    # TODO: Merge clamped mesh into most fitting model.
    for n, (vtx_mesh, mesh) in enumerate(zip(vtx_model.meshes, model.meshes)):

        if not vtx_mesh.strip_groups:
            continue

        vertex_start = mesh.vertex_index_start
        indices, vertices, offset = merge_strip_groups(vtx_mesh)
        indices = np.add(indices, acc)
        mat_array = np.full(indices.shape[0] // 3, mesh.material_index)
        mat_arrays.append(mat_array)
        vtx_vertices.extend(np.add(vertices, vertex_start))
        indices_array.append(indices)
        acc += offset

    return vtx_vertices, np.hstack(indices_array), np.hstack(mat_arrays)


def get_slice(data: [Iterable, Sized], start, count=None):
    if count is None:
        count = len(data) - start
    return data[start:start + count]


def create_armature(mdl: MdlV49, scale=1.0):
    model_name = Path(mdl.header.name).stem
    armature = bpy.data.armatures.new(f"{model_name}_ARM_DATA")
    armature_obj = bpy.data.objects.new(f"{model_name}_ARM", armature)
    armature_obj['MODE'] = 'SourceIO'
    armature_obj.show_in_front = True
    bpy.context.scene.collection.objects.link(armature_obj)

    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    bpy.ops.object.mode_set(mode='EDIT')
    bl_bones = []
    for bone in mdl.bones:
        bl_bone = armature.edit_bones.new(bone.name[-63:])
        bl_bones.append(bl_bone)

    for bl_bone, s_bone in zip(bl_bones, mdl.bones):
        if s_bone.parent_bone_index != -1:
            bl_parent = bl_bones[s_bone.parent_bone_index]
            bl_bone.parent = bl_parent
        bl_bone.tail = (Vector([0, 0, 1]) * scale) + bl_bone.head

    bpy.ops.object.mode_set(mode='POSE')
    for se_bone in mdl.bones:
        bl_bone = armature_obj.pose.bones.get(se_bone.name[-63:])
        pos = Vector(se_bone.position) * scale
        rot = Euler(se_bone.rotation)
        mat = Matrix.Translation(pos) @ rot.to_matrix().to_4x4()
        bl_bone.matrix_basis.identity()

        bl_bone.matrix = bl_bone.parent.matrix @ mat if bl_bone.parent else mat
    bpy.ops.pose.armature_apply()
    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.context.scene.collection.objects.unlink(armature_obj)
    return armature_obj


def import_model(mdl_file: Union[BinaryIO, Path],
                 vvd_file: Union[BinaryIO, Path],
                 vtx_file: Union[BinaryIO, Path],
                 vvc_file: Optional[Union[BinaryIO, Path]] = None,
                 scale=1.0, create_drivers=False, re_use_meshes=False, unique_material_names=False):
    mdl = MdlV49(mdl_file)
    mdl.read()

    full_material_names = collect_full_material_names(mdl)

    vvd = Vvd(vvd_file)
    vvd.read()
    if vvc_file is not None:
        vvc = Vvc(vvc_file)
        vvc.read()
    else:
        vvc = None
    vtx = Vtx(vtx_file)
    vtx.read()

    container = Source1ModelContainer(mdl, vvd, vtx)

    desired_lod = 0
    all_vertices = vvd.lod_data[desired_lod]

    static_prop = mdl.header.flags & StudioHDRFlags.STATIC_PROP != 0
    armature = None
    if mdl.flex_names:
        vac = VertexAnimationCache(mdl, vvd)
        vac.process_data()

    if not static_prop:
        armature = create_armature(mdl, scale)
        container.armature = armature

    for vtx_body_part, body_part in zip(vtx.body_parts, mdl.body_parts):
        for vtx_model, model in zip(vtx_body_part.models, body_part.models):

            if model.vertex_count == 0:
                continue
            mesh_name = f'{body_part.name}_{model.name}'
            used_copy = False
            if re_use_meshes and static_prop:
                mesh_obj_original = bpy.data.objects.get(mesh_name, None)
                mesh_data_original = bpy.data.meshes.get(f'{mdl.header.name}_{mesh_name}_MESH', False)
                if mesh_obj_original and mesh_data_original:
                    mesh_data = mesh_data_original.copy()
                    mesh_obj = mesh_obj_original.copy()
                    mesh_obj['skin_groups'] = mesh_obj_original['skin_groups']
                    mesh_obj['active_skin'] = mesh_obj_original['active_skin']
                    mesh_obj['model_type'] = 's1'
                    mesh_obj.data = mesh_data
                    used_copy = True
                else:
                    mesh_data = bpy.data.meshes.new(f'{mesh_name}_MESH')
                    mesh_obj = bpy.data.objects.new(mesh_name, mesh_data)
                    mesh_obj['skin_groups'] = {str(n): group for (n, group) in enumerate(mdl.skin_groups)}
                    mesh_obj['active_skin'] = '0'
                    mesh_obj['model_type'] = 's1'
            else:
                mesh_data = bpy.data.meshes.new(f'{mesh_name}_MESH')
                mesh_obj = bpy.data.objects.new(mesh_name, mesh_data)
                mesh_obj['skin_groups'] = {str(n): group for (n, group) in enumerate(mdl.skin_groups)}
                mesh_obj['active_skin'] = '0'
                mesh_obj['model_type'] = 's1'

            if not static_prop:
                modifier = mesh_obj.modifiers.new(
                    type="ARMATURE", name="Armature")
                modifier.object = armature
                mesh_obj.parent = armature
            container.objects.append(mesh_obj)
            container.bodygroups[body_part.name].append(mesh_obj)
            mesh_obj['unique_material_names'] = unique_material_names
            mesh_obj['prop_path'] = Path(mdl.header.name).stem

            if used_copy:
                continue

            model_vertices = get_slice(all_vertices, model.vertex_offset, model.vertex_count)
            vtx_vertices, indices_array, material_indices_array = merge_meshes(model, vtx_model.model_lods[desired_lod])

            indices_array = np.array(indices_array, dtype=np.uint32)
            vertices = model_vertices[vtx_vertices]
            vertices_vertex = vertices['vertex']

            mesh_data.from_pydata(vertices_vertex * scale, [], np.flip(indices_array).reshape((-1, 3)).tolist())
            mesh_data.update()

            mesh_data.polygons.foreach_set("use_smooth", np.ones(len(mesh_data.polygons), np.uint32))
            mesh_data.normals_split_custom_set_from_vertices(vertices['normal'])
            mesh_data.use_auto_smooth = True

            material_remapper = np.zeros((material_indices_array.max() + 1,), dtype=np.uint32)
            for mat_id in np.unique(material_indices_array):
                mat_name = mdl.materials[mat_id].name
                if unique_material_names:
                    mat_name = f"{Path(mdl.header.name).stem}_{mat_name[-63:]}"[-63:]
                else:
                    mat_name = mat_name[-63:]
                material_remapper[mat_id] = get_material(mat_name, mesh_obj)

            mesh_data.polygons.foreach_set('material_index', material_remapper[material_indices_array[::-1]].tolist())

            uv_data = mesh_data.uv_layers.new()

            vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
            mesh_data.loops.foreach_get('vertex_index', vertex_indices)
            uvs = vertices['uv']
            uvs[:, 1] = 1 - uvs[:, 1]
            uv_data.data.foreach_set('uv', uvs[vertex_indices].flatten())
            if vvc is not None:
                model_uvs2 = get_slice(vvc.secondary_uv, model.vertex_offset, model.vertex_count)
                uvs2 = model_uvs2[vtx_vertices]
                uv_data = mesh_data.uv_layers.new(name='UV2')
                uvs2[:, 1] = 1 - uvs2[:, 1]
                uv_data.data.foreach_set('uv', uvs2[vertex_indices].flatten())

                model_colors = get_slice(vvc.color_data, model.vertex_offset, model.vertex_count)
                colors = model_colors[vtx_vertices]

                vc = mesh_data.vertex_colors.new()
                vc.data.foreach_set('color', colors[vertex_indices].flatten())

            if not static_prop:
                weight_groups = {bone.name: mesh_obj.vertex_groups.new(name=bone.name) for bone in mdl.bones}

                for n, (bone_indices, bone_weights) in enumerate(zip(vertices['bone_id'], vertices['weight'])):
                    for bone_index, weight in zip(bone_indices, bone_weights):
                        if weight > 0:
                            bone_name = mdl.bones[bone_index].name
                            weight_groups[bone_name].add([n], weight, 'REPLACE')

            if not static_prop:
                flexes = []
                for mesh in model.meshes:
                    if mesh.flexes:
                        flexes.extend([(mdl.flex_names[flex.flex_desc_index], flex) for flex in mesh.flexes])
                if flexes:
                    mesh_obj.shape_key_add(name='base')
                for flex_name, flex_desc in flexes:
                    vertex_animation = vac.vertex_cache[flex_name]
                    flex_delta = get_slice(vertex_animation, model.vertex_offset, model.vertex_count)
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

                        shape_key.data.foreach_set("co", (flex_delta + model_vertices).reshape(-1))
                if create_drivers:
                    create_flex_drivers(mesh_obj, mdl)
    if mdl.attachments:
        attachments = create_attachments(mdl, armature if not static_prop else container.objects[0], scale)
        container.attachments.extend(attachments)

    return container


def put_into_collections(model_container: Source1ModelContainer, model_name,
                         parent_collection=None, bodygroup_grouping=False):
    static_prop = model_container.armature is None
    if not static_prop:
        master_collection = get_new_unique_collection(model_name, parent_collection or bpy.context.scene.collection)
    else:
        master_collection = parent_collection or bpy.context.scene.collection
    for bodygroup_name, meshes in model_container.bodygroups.items():
        if bodygroup_grouping:
            body_part_collection = get_new_unique_collection(bodygroup_name, master_collection)
        else:
            body_part_collection = master_collection

        for mesh in meshes:
            body_part_collection.objects.link(mesh)
    if model_container.armature:
        master_collection.objects.link(model_container.armature)

    if model_container.attachments:
        attachments_collection = get_new_unique_collection(model_name + '_ATTACHMENTS', master_collection)
        for attachment in model_container.attachments:
            attachments_collection.objects.link(attachment)
    return master_collection


def create_flex_drivers(obj, mdl: MdlV49):
    from ....operators.flex_operators import SourceIO_PG_FlexController
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
    for flex_name, (expr, inputs) in all_exprs.items():
        driver_name = f'{flex_name}_driver'.replace(' ', '_')
        if driver_name in globals():
            continue

        input_definitions = []
        for inp in inputs:
            input_name = inp[0]
            if inp[1] in ('fetch1', '2WAY1', '2WAY0', 'NWAY', 'DUE'):
                if 'left_' in input_name:
                    input_name = input_name.replace('left_', '')
                    input_definitions.append(
                        f'{inp[0].replace(" ", "_")} = obj_data.flex_controllers["{input_name}"].value_left')
                elif 'right_' in input_name:
                    input_name = input_name.replace('right_', '')
                    input_definitions.append(
                        f'{inp[0].replace(" ", "_")} = obj_data.flex_controllers["{input_name}"].value_right')
                else:
                    input_definitions.append(
                        f'{inp[0].replace(" ", "_")} = obj_data.flex_controllers["{inp[0]}"].value')
            elif inp[1] == 'fetch2':
                input_definitions.append(
                    f'{inp[0].replace(" ", "_")} = obj_data.shape_keys.key_blocks["{input_name}"].value')
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
def {flex_name.replace(' ', '_')}_driver(obj_data):
    return obj_data.flex_controllers["{flex_name}"].value
bpy.app.driver_namespace["{flex_name.replace(' ', '_')}_driver"] = {flex_name.replace(' ', '_')}_driver

                                """
                blender_py_file += template_function
            else:
                template_function = f"""
def {flex_name.replace(' ', '_')}_driver(obj_data):
    {st.join(inputs)}
    return {expr}
bpy.app.driver_namespace["{flex_name.replace(' ', '_')}_driver"] = {flex_name.replace(' ', '_')}_driver

                """
                blender_py_file += template_function

        shape_key.driver_remove("value")
        fcurve = shape_key.driver_add("value")
        fcurve.modifiers.remove(fcurve.modifiers[0])

        driver = fcurve.driver
        driver.type = 'SCRIPTED'
        driver.expression = f"{flex_name.replace(' ', '_')}_driver(obj_data)"
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


def import_materials(mdl: MdlV49, unique_material_names=False, use_bvlg=False):
    content_manager = ContentManager()
    for material in mdl.materials:

        if unique_material_names:
            mat_name = f"{Path(mdl.header.name).stem}_{material.name[-63:]}"[-63:]
        else:
            mat_name = material.name[-63:]
        material_eyeball = None
        for eyeball in mdl.eyeballs:
            if eyeball.material.name == material.name:
                material_eyeball = eyeball

        if bpy.data.materials.get(mat_name, False):
            if bpy.data.materials[mat_name].get('source1_loaded', False):
                logger.info(f'Skipping loading of {mat_name} as it already loaded')
                continue
        material_path = None
        for mat_path in mdl.materials_paths:
            material_path = content_manager.find_material(Path(mat_path) / material.name)
            if material_path:
                break
        if material_path:
            Source1ShaderBase.use_bvlg(use_bvlg)
            if material_eyeball is not None:
                pass
                # TODO: Syborg64 replace this with actual shader class
                # new_material = EyeShader(material_path, mat_name, material_eyeball)
                new_material = Source1MaterialLoader(material_path, mat_name)
            else:
                new_material = Source1MaterialLoader(material_path, mat_name)
            new_material.create_material()


def import_animations(mdl: MdlV49, armature, scale):
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    if not armature.animation_data:
        armature.animation_data_create()
    # for var_pos in ['XYZ', 'YXZ', ]:
    #     for var_rot in ['XYZ', 'XZY', 'YZX', 'ZYX', 'YXZ', 'ZXY', ]:
    for anim_desc in mdl.anim_descs:
        anim_name = anim_desc.name
        action = bpy.data.actions.new(anim_name)
        armature.animation_data.action = action
        curve_per_bone = {}

        for bone in mdl.bones:
            bone_name = bone.name
            bl_bone = armature.pose.bones.get(bone.name)
            bl_bone.rotation_mode = 'QUATERNION'
            bone_string = f'pose.bones["{bone_name}"].'
            group = action.groups.new(name=bone_name)
            pos_curves = []
            rot_curves = []
            for i in range(3):
                pos_curve = action.fcurves.new(data_path=bone_string + "location", index=i)
                pos_curve.keyframe_points.add(anim_desc.frame_count)
                pos_curves.append(pos_curve)
                pos_curve.group = group
            for i in range(4):
                rot_curve = action.fcurves.new(data_path=bone_string + "rotation_quaternion", index=i)
                rot_curve.keyframe_points.add(anim_desc.frame_count)
                rot_curves.append(rot_curve)
                rot_curve.group = group
            curve_per_bone[bone_name] = pos_curves, rot_curves
        for n, bone in enumerate(mdl.bones):
            for frame_index in range(anim_desc.frame_count):
                frame = frame_index
                section = anim_desc.find_section(frame)
                if not section:
                    continue

                frame -= section.first_frame
                data = section.anim_data

                frame = min(frame, section.frame_count - 1)

                track_id = n
                if mdl.bone_table_by_name:
                    track_id = mdl.bone_table_by_name[n]

                if track_id not in data.tracks:
                    continue
                track = data.tracks[track_id]
                pos, rot = track.get_pos_rot(mdl.reader, bone, frame)
                pos -= bone.position
                rot = Quaternion(rot).rotation_difference(euler_to_quat(np.asarray(bone.rotation)))
                pos_curves, rot_curves = curve_per_bone[bone.name]
                for i in range(3):
                    pos_curves[i].keyframe_points.add(1)
                    pos_curves[i].keyframe_points[-1].co = (frame_index, (pos[i]) * scale)

                for i in range(4):
                    rot_curves[i].keyframe_points.add(1)
                    rot_curves[i].keyframe_points[-1].co = (frame_index, (rot[i]))

        bpy.ops.object.mode_set(mode='OBJECT')
