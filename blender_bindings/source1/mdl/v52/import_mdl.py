from pathlib import Path

import bpy
import numpy as np

from .....library.source1.mdl.structs.header import StudioHDRFlags
from .....library.source1.mdl.v44.vertex_animation_cache import \
    VertexAnimationCache
from .....library.source1.mdl.v52.mdl_file import MdlV52
from .....library.source1.vtx import open_vtx
from .....library.source1.vtx.v7.vtx import Vtx
from .....library.source1.vvc import Vvc
from .....library.source1.vvd import Vvd
from .....logger import SLoggingManager
from ....shared.model_container import Source1ModelContainer
from ....utils.utils import add_material, is_blender_4_1
from .. import FileImport
from ..common import get_slice, merge_meshes
from ..v49.import_mdl import (collect_full_material_names, create_armature,
                              create_attachments, create_flex_drivers)

log_manager = SLoggingManager()
logger = log_manager.get_logger('Source1::ModelLoader')


def import_model(file_list: FileImport,
                 scale=1.0, create_drivers=False, re_use_meshes=False, unique_material_names=False, load_refpose=False):
    mdl = MdlV52.from_buffer(file_list.mdl_file)
    vvd = Vvd.from_buffer(file_list.vvd_file)
    vtx = open_vtx(file_list.vtx_file)

    full_material_names = collect_full_material_names(mdl)

    if file_list.vvc_file is not None:
        vvc = Vvc.from_buffer(file_list.vvc_file)
    else:
        vvc = None

    container = Source1ModelContainer(mdl, vvd, vtx, file_list)

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

            mesh_data.from_pydata(vertices_vertex * scale, [], np.flip(indices_array).reshape((-1, 3)))
            mesh_data.update()

            mesh_data.polygons.foreach_set("use_smooth", np.ones(len(mesh_data.polygons), np.uint32))
            mesh_data.normals_split_custom_set_from_vertices(vertices['normal'])
            if is_blender_4_1():
                pass
            else:
                mesh_data.use_auto_smooth = True

            material_remapper = np.zeros((material_indices_array.max() + 1,), dtype=np.uint32)
            for mat_id in np.unique(material_indices_array):
                mat_name = mdl.materials[mat_id].name
                if unique_material_names:
                    mat_name = f"{Path(mdl.header.name).stem}_{mat_name[-63:]}"[-63:]
                else:
                    mat_name = mat_name[-63:]
                material_remapper[mat_id] = add_material(mat_name, mesh_obj)

            mesh_data.polygons.foreach_set('material_index', material_remapper[material_indices_array[::-1]])

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
                    # wrinkle_cache = get_slice(vac.wrinkle_cache, model.vertex_offset, model.vertex_count)
                    # vc = mesh_data.vertex_colors.new(name=f'speed_and_wrinkle')
                    # vc.data.foreach_set('color', wrinkle_cache[vtx_vertices][vertex_indices].flatten().tolist())
                    mesh_obj.shape_key_add(name='base')
                for flex_name, flex_desc in flexes:
                    vertex_animation = vac.vertex_cache[flex_name]
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

                        shape_key.data.foreach_set("co", (flex_delta + model_vertices).reshape(-1))
                if create_drivers:
                    create_flex_drivers(mesh_obj, mdl)
    if mdl.attachments:
        attachments = create_attachments(mdl, armature if not static_prop else container.objects[0], scale)
        container.attachments.extend(attachments)

    return container
