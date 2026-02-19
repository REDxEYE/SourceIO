from collections import defaultdict

import bpy
import numpy as np

from SourceIO.blender_bindings.models.common import merge_meshes, create_eyeballs
from SourceIO.blender_bindings.models.mdl49.import_mdl import create_armature, create_attachments, create_flex_drivers
from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.blender_bindings.utils.bpy_utils import add_material, is_blender_4_1, get_or_create_material
from SourceIO.blender_bindings.utils.fast_mesh import FastMesh
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
                 scale=1.0, create_drivers=False, load_refpose=False, *, debug_stereo_balance=False):
    full_material_names = collect_full_material_names([mat.name for mat in mdl.materials], mdl.materials_paths,
                                                      content_provider)
    [setattr(mat, 'bpy_material', get_or_create_material(mat.name, full_material_names[mat.name])) for mat in mdl.materials if mat.bpy_material is None]
    # ensure all MaterialV49 has its bpy_material counterpart

    objects = []
    bodygroups = defaultdict(list)
    attachments = []
    extra_stuff = []
    desired_lod = 0
    all_vertices = vvd.lod_data[desired_lod]
    vertex_anim_cache = preprocess_vertex_animation(mdl, vvd)

    static_prop = mdl.header.flags & StudioHDRFlags.STATIC_PROP != 0
    vert_anim_fixed_point_scale = mdl.header.vert_anim_fixed_point_scale if (mdl.header.flags & StudioHDRFlags.VERT_ANIM_FIXED_POINT_SCALE !=0 ) else 1/4096
    armature = None

    if not static_prop:
        armature = create_armature(mdl, scale)

    for vtx_body_part, body_part in zip(vtx.body_parts, mdl.body_parts):
        for vtx_model, model in zip(vtx_body_part.models, body_part.models):

            if model.vertex_count == 0:
                continue
            mesh_name = f'{body_part.name}_{model.name}'

            mesh_data = FastMesh.new(f'{mesh_name}_MESH')
            mesh_obj = bpy.data.objects.new(mesh_name, mesh_data)
            default_skin_groups = {str(n): list(map(lambda a: a.name, group)) for (n, group) in enumerate(mdl.skin_groups)}
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

            skin_groups = {str(n): list(map(lambda a: a.bpy_material, group)) for (n, group) in enumerate(mdl.skin_groups)}
            try:
                mesh_obj['skin_groups'] = skin_groups
            except:
                mesh_obj['skin_groups'] = default_skin_groups

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

                    if debug_stereo_balance:
                        # debug tool to get stereo flex balances. i'll leave it here just in case
                        side_right = mesh_obj.vertex_groups.new(name='blendright')
                        side_left = mesh_obj.vertex_groups.new(name='blendleft')
                        side_all = np.zeros(model.vertex_count, dtype=np.float32)

                        for flex_name, flex_desc in flexes:
                            vertex_animation = vertex_anim_cache[flex_name]
                            side = get_slice(vertex_animation['side'], model.vertex_offset, model.vertex_count).ravel()
                            side_all = np.maximum(side_all, side)
                        side_all = side_all[vtx_vertices] + 0.0

                        for n, vert in enumerate(vtx_vertices):
                            side_right.add([n], side_all[n], 'REPLACE')
                            side_left.add([n], 1-side_all[n], 'REPLACE')

                    for flex_name, flex_desc in flexes:
                        vertex_animation = vertex_anim_cache[flex_name]
                        flex_delta = get_slice(vertex_animation["pos"], model.vertex_offset, model.vertex_count)
                        flex_delta = flex_delta[vtx_vertices] * scale

                        side = get_slice(vertex_animation["side"], model.vertex_offset, model.vertex_count)
                        side = side[vtx_vertices] + 0.0
                        wrinkle = get_slice(vertex_animation["wrinkle"], model.vertex_offset, model.vertex_count)
                        wrinkle = wrinkle[vtx_vertices] + 0.0 # this will have to be explained to me :P
                        # model.vertex_count and vtx_vertices can differ in size, so doing something like this just makes it work?
                        # apparently vtx_vertices has duplicate indicies, which can be observed by turning it into a set.
                        # i'm just following here
                        # -hisanimations
                        
                        model_vertices = get_slice(all_vertices['vertex'], model.vertex_offset, model.vertex_count)
                        model_vertices = model_vertices[vtx_vertices] * scale

                        if flex_desc.partner_index:
                            partner_name = mdl.flex_names[flex_desc.partner_index]
                            flexes, sides = [flex_name, partner_name], [1-side, side] if not debug_stereo_balance else [1.0, 1.0]
                        else:
                            flexes, sides = [flex_name], [1.0]

                        for flex_name, side in zip(flexes, sides):
                            shape_key = mesh_data.shape_keys.key_blocks.get(flex_name, None) or mesh_obj.shape_key_add(
                                name=flex_name)
                            shape_key.data.foreach_set("co", (flex_delta*side + model_vertices).ravel())

                            if flex_desc.vertex_anim_type == 1:
                                mesh_data: bpy.types.Mesh
                                if wrinkle.max() > 0:
                                    wrinkle_name = f'WR.{flex_name}.S'
                                if wrinkle.min() < 0:
                                    wrinkle_name = f'WR.{flex_name}.C'

                                attr: bpy.types.Attribute = mesh_data.attributes.get(wrinkle_name, None) or mesh_data.attributes.new(wrinkle_name, 'FLOAT', 'POINT')
                                wrinkle_data = (abs(wrinkle) * vert_anim_fixed_point_scale) * side
                                attr.data.foreach_set('value', wrinkle_data.ravel())

                if create_drivers:
                    create_flex_drivers(mesh_obj, mdl)
            mesh_data.validate()

        if model.has_eyeballs:
            create_eyeballs(mdl, armature, mesh_obj, model, scale, extra_stuff)

    if mdl.attachments:
        attachments = create_attachments(mdl, armature if not static_prop else objects[0], scale)
    attachments.extend(extra_stuff)

    return ModelContainer(objects, bodygroups, [], attachments, armature, None)
