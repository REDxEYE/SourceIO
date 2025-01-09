from collections import defaultdict

import bpy
import numpy as np

from SourceIO.blender_bindings.models.common import merge_meshes
from SourceIO.blender_bindings.models.mdl49.import_mdl import create_armature, create_attachments, create_flex_drivers
from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.blender_bindings.utils.bpy_utils import add_material, is_blender_4_1, get_or_create_material
from SourceIO.library.models.mdl.structs.header import StudioHDRFlags
from SourceIO.library.models.mdl.v44.vertex_animation_cache import preprocess_vertex_animation
from SourceIO.library.models.mdl.v52.mdl_file import MdlV52
from SourceIO.library.models.vtx.v7.vtx import Vtx
from SourceIO.library.models.vvc import Vvc
from SourceIO.library.models.vvd import Vvd
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.utils.common import get_slice
from SourceIO.library.utils.path_utilities import path_stem, collect_full_material_names
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('Source1::ModelLoader')


def import_model(content_provider: ContentProvider, mdl: MdlV52, vtx: Vtx, vvd: Vvd, vvc: Vvc,
                 scale=1.0, create_drivers=False, load_refpose=False):
    full_material_names = collect_full_material_names([mat.name for mat in mdl.materials], mdl.materials_paths,
                                                      content_provider)

    objects = []
    bodygroups = defaultdict(list)
    attachments = []
    desired_lod = 0
    all_vertices = vvd.lod_data[desired_lod]
    vertex_anim_cache = preprocess_vertex_animation(mdl, vvd)

    static_prop = mdl.header.flags & StudioHDRFlags.STATIC_PROP != 0
    armature = None

    if not static_prop:
        armature = create_armature(mdl, scale)

    for vtx_body_part, body_part in zip(vtx.body_parts, mdl.body_parts):
        for vtx_model, model in zip(vtx_body_part.models, body_part.models):

            if model.vertex_count == 0:
                continue
            mesh_name = f'{body_part.name}_{model.name}'

            mesh_data = bpy.data.meshes.new(f'{mesh_name}_MESH')
            mesh_obj = bpy.data.objects.new(mesh_name, mesh_data)
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

                        shape_key.data.foreach_set("co", (flex_delta + model_vertices).reshape(-1))
                if create_drivers:
                    create_flex_drivers(mesh_obj, mdl)
            mesh_data.validate()
    if mdl.attachments:
        attachments = create_attachments(mdl, armature if not static_prop else objects[0], scale)

    return ModelContainer(objects, bodygroups, [], attachments, armature, None)
