import logging
from struct import pack, unpack
from itertools import chain
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple, cast

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from ...library.shared.content_providers.content_manager import ContentManager
from ...library.source2 import (CompiledMaterialResource,
                                CompiledModelResource, CompiledMorphResource,
                                CompiledPhysicsResource, CompiledResource)
from ...library.source2.common import convert_normals, convert_normals_2
from ...library.source2.data_types.blocks.kv3_block import KVBlock
from ...library.source2.data_types.blocks.morph_block import MorphBlock
from ...library.source2.data_types.blocks.phys_block import PhysBlock
from ...library.source2.data_types.blocks.vertex_index_buffer import \
    VertexIndexBuffer
from ...library.source2.data_types.blocks.vertex_index_buffer.vertex_buffer import \
    VertexBuffer
from ...library.source2.data_types.keyvalues3.types import NullObject, Object
from ...library.source2.exceptions import MissingBlock
from ...library.source2.resource_types import CompiledMeshResource
from ...library.utils.math_utilities import SOURCE2_HAMMER_UNIT_TO_METERS
from ..material_loader.material_loader import Source2MaterialLoader
from ..shared.model_container import Source2ModelContainer
from ..utils.utils import (add_material, find_layer_collection,
                           get_new_unique_collection)
from .vmat_loader import load_material
from .vphy_loader import load_physics


def put_into_collections(model_container, model_name, parent_collection=None, bodygroup_grouping=False):
    model_container: Source2ModelContainer
    static_prop = model_container.armature is None
    if static_prop:
        master_collection = parent_collection or bpy.context.scene.collection
    else:
        master_collection = get_new_unique_collection(model_name, parent_collection or bpy.context.scene.collection)

    for group_name, objs in model_container.bodygroups.items():
        group_collection = get_new_unique_collection(group_name, master_collection)
        for obj in objs:
            group_collection.objects.link(obj)

    if model_container.armature:
        for obj in chain(model_container.objects, model_container.physics_objects):
            obj.parent = model_container.armature
        master_collection.objects.link(model_container.armature)

    if model_container.physics_objects:
        phys_collection = get_new_unique_collection(model_name + '_PHYSICS', master_collection)
        for phys in model_container.physics_objects:
            phys_collection.objects.link(phys)
        phys_lcollection = find_layer_collection(bpy.context.view_layer.layer_collection, phys_collection.name)
        phys_lcollection.exclude = True
    if model_container.attachments:
        attachments_collection = get_new_unique_collection(model_name + '_ATTACHMENTS', master_collection)
        for attachment in model_container.attachments:
            attachments_collection.objects.link(attachment)
    model_container.collection = master_collection
    return master_collection


def load_model(resource: CompiledModelResource, scale: float = SOURCE2_HAMMER_UNIT_TO_METERS,
               lod_mask: int = 255, import_physics: bool = False, import_attachments: bool = False):
    armature = create_armature(resource, scale)
    container = Source2ModelContainer(resource)

    if import_physics:
        physics_block = get_physics_block(resource)
        if physics_block is not None:
            objects = load_physics(physics_block, scale)
            container.physics_objects = objects

    container.armature = armature
    objects = create_meshes(resource, ContentManager(), container, scale, lod_mask, import_attachments)
    if armature:
        for obj in objects:
            modifier = obj.modifiers.new(type="ARMATURE", name="Armature")
            modifier.object = armature
    container.objects = objects

    return container


def clear_selection():
    for obj in bpy.context.selected_objects:
        obj.select_set(False)


def create_armature(resource: CompiledModelResource, scale: float):
    s2_bones = resource.get_bones()
    if not s2_bones:
        return None
    name = resource.name
    armature_obj = bpy.data.objects.new(name + "_ARM", bpy.data.armatures.new(name + "_ARM_DATA"))
    armature_obj['MODE'] = 'SourceIO'
    armature_obj.show_in_front = True
    bpy.context.scene.collection.objects.link(armature_obj)
    clear_selection()
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')
    armature = armature_obj.data

    for s2_bone in s2_bones:
        bl_bone = armature.edit_bones.new(name=s2_bone.name)
        bl_bone.tail = (Vector([0, 0, 1]) * scale) + bl_bone.head

        if s2_bone.parent:
            bl_bone.parent = armature.edit_bones.get(s2_bone.parent)

        bone_pos = s2_bone.pos
        bone_rot = s2_bone.rot

        bone_pos = Vector(bone_pos) * scale
        # noinspection PyTypeChecker
        bone_rot = Quaternion(bone_rot)
        mat = (Matrix.Translation(bone_pos) @ bone_rot.to_matrix().to_4x4())
        if bl_bone.parent:
            bl_bone.matrix = bl_bone.parent.matrix @ mat
        else:
            bl_bone.matrix = mat

    physics_block = get_physics_block(resource)
    if physics_block and physics_block["m_pFeModel"] and physics_block["m_pFeModel"]["m_TreeChildren"]:
        p_model_data = physics_block["m_pFeModel"]
        names = p_model_data["m_CtrlName"]

        for parent_pair in p_model_data["m_CtrlOffsets"]:
            parent = names[parent_pair["nCtrlParent"]]
            child = names[parent_pair["nCtrlChild"]]

            if child in armature.edit_bones:
                armature.edit_bones.get(child).parent = armature.edit_bones.get(parent)

    bpy.ops.object.mode_set(mode='OBJECT')
    # armature_obj.rotation_euler = Euler([math.radians(180), 0, math.radians(90)])

    bpy.context.scene.collection.objects.unlink(armature_obj)
    return armature_obj


def create_meshes(model_resource: CompiledModelResource, cm: ContentManager, container: Source2ModelContainer,
                  scale: float, lod_mask: int, import_attachments: bool) -> List[bpy.types.Object]:
    lod_mask = unpack("Q", pack("q", lod_mask))[0]
    data, = model_resource.get_data_block(block_name='DATA')
    ctrl, = model_resource.get_data_block(block_name='CTRL')
    group_masks = {}
    lod_count = len(data['m_lodGroupSwitchDistances'])

    object_groups = []
    if data['m_meshGroups']:
        for i, group in enumerate(data['m_meshGroups']):
            group_masks[2 ** i] = group
    else:
        group_masks[0xFFFF] = model_resource.name + '_ALL'
    for i, mesh in enumerate(data['m_refMeshes']):
        mesh_mask = int(data['m_refMeshGroupMasks'][i])
        lod_id = data['m_refLODGroupMasks'][i]
        if lod_id & lod_mask == 0:
            continue
        if isinstance(mesh, NullObject) or not mesh:
            # Embedded mesh
            mesh_info = ctrl['embedded_meshes'][i]
            sub_meshes = load_internal_mesh(model_resource, cm, container, scale, mesh_info, import_attachments)
        else:
            # External mesh
            mesh_resource = model_resource.get_child_resource(mesh, cm, CompiledMeshResource)
            sub_meshes = load_external_mesh(model_resource, cm, container, scale, i, mesh_resource, import_attachments)
        object_groups.extend(sub_meshes)
        for sub_mesh in sub_meshes:
            for lod in range(lod_count):
                if lod_id & (2 ** lod) > 0:
                    sub_mesh.name += f'_LOD{lod}'
                    break
            for group_mask, group in group_masks.items():
                if mesh_mask & group_mask > 0:
                    container.bodygroups[group].append(sub_mesh)
    return object_groups


def load_internal_mesh(model_resource: CompiledModelResource, cm: ContentManager, container: Source2ModelContainer,
                       scale: float,
                       mesh_info: Mapping[str, Any], import_attachments: bool):
    mesh_index = mesh_info['mesh_index']
    data_block: Optional[KVBlock] = model_resource.get_data_block(block_id=mesh_info['data_block'])
    vbib_block: Optional[VertexIndexBuffer] = model_resource.get_data_block(block_id=mesh_info['vbib_block'])
    morph_block: Optional[MorphBlock] = model_resource.get_data_block(block_id=mesh_info['morph_block'])
    if data_block and vbib_block:
        return create_mesh(model_resource, cm, container, data_block, vbib_block, morph_block, scale, mesh_index,
                           model_resource,
                           mesh_info['name'],
                           import_attachments)
    return None


def load_external_mesh(model_resource: CompiledModelResource, cm: ContentManager, container: Source2ModelContainer,
                       scale: float, mesh_id: int,
                       mesh_resource: CompiledMeshResource, import_attachments: bool):
    data_block, = mesh_resource.get_data_block(block_name='DATA')
    vbib_block, = mesh_resource.get_data_block(block_name='VBIB')
    if morph_set_path := data_block['m_morphSet']:
        morph_resource = mesh_resource.get_child_resource(morph_set_path, cm, CompiledMorphResource)
        morph_block, = morph_resource.get_data_block(block_name='DATA')
    else:
        morph_block, = mesh_resource.get_data_block(block_name='MRPH')
    if data_block and vbib_block:
        return create_mesh(model_resource, cm, container, data_block, vbib_block, morph_block, scale, mesh_id,
                           mesh_resource, import_attachments=import_attachments)
    return None


def _add_vertex_groups(model_resource: CompiledModelResource,
                       vertex_buffer: VertexBuffer,
                       mesh_id: int,
                       used_vertices: np.ndarray,
                       mesh_obj: bpy.types.Object):
    has_weights = vertex_buffer.has_attribute('BLENDWEIGHT')
    has_indicies = vertex_buffer.has_attribute('BLENDINDICES')

    if not has_weights and not has_indicies:
        return
    model_data_block, = model_resource.get_data_block(block_name='DATA')
    bones = model_data_block['m_modelSkeleton']['m_boneName']
    weight_groups = {bone: mesh_obj.vertex_groups.new(name=bone) for bone in bones}
    remap_table = np.asarray(model_data_block['m_remappingTable'][model_data_block['m_remappingTableStarts'][mesh_id]:],
                             np.uint32)
    if has_weights and has_indicies:
        weights_array = used_vertices["BLENDWEIGHT"] / 255
        indices_array = used_vertices["BLENDINDICES"].astype(np.uint32)
    else:
        indices_array = used_vertices["BLENDINDICES"].astype(np.uint32)
        weights_array = np.ones_like(indices_array, dtype=np.float32)
    remapped_indices = remap_table[indices_array]
    for n, bone_indices in enumerate(remapped_indices):
        weights = weights_array[n]
        for bone_index, weight in zip(bone_indices, weights):
            if weight > 0:
                bone_name = bones[bone_index]
                weight_groups[bone_name].add([n], weight, 'REPLACE')


def convert_to_float32(uv_array: np.ndarray):
    if uv_array.dtype == np.float32 or uv_array.dtype == np.float16:
        return uv_array
    dtype_info = np.iinfo(uv_array.dtype)
    dtype_min, dtype_max = dtype_info.min, dtype_info.max

    if uv_array.shape[1] == 4:
        uv_array = uv_array[:, :2]

    if dtype_info.kind == 'u':  # Unsigned type
        return (uv_array.astype(np.float32) - dtype_min) / (dtype_max - dtype_min)
    else:  # Signed type
        return (uv_array.astype(np.float32) - dtype_min) / (dtype_max - dtype_min) * 2 - 1


def create_mesh(model_resource: CompiledModelResource, cm: ContentManager, container: Source2ModelContainer,
                data_block: KVBlock, vbib_block: VertexIndexBuffer, morph_block: MorphBlock,
                scale: float, mesh_id: int,
                mesh_resource: CompiledResource, mesh_name: Optional[str] = None,
                import_attachments: bool = False) -> List[bpy.types.Object]:
    objects: List[Tuple[str, bpy.types.Object]] = []
    g_vertex_offset = 0
    if import_attachments:
        load_attachments(data_block["m_attachments"], container, scale)

    for scene_object in data_block['m_sceneObjects']:
        for draw_call in scene_object['m_drawCalls']:
            print(draw_call)
            assert len(draw_call['m_vertexBuffers']) == 1
            assert draw_call['m_nPrimitiveType'] == 'RENDER_PRIM_TRIANGLES'

            vertex_buffer_info = draw_call['m_vertexBuffers'][0]
            index_buffer_info = draw_call['m_indexBuffer']
            vertex_buffer = vbib_block.vertex_buffers[vertex_buffer_info['m_hBuffer']]
            index_buffer = vbib_block.index_buffers[index_buffer_info['m_hBuffer']]
            material_name = draw_call['m_material']
            material_resource: Optional[CompiledMaterialResource] = None
            if not isinstance(material_name, NullObject):
                material_resource = mesh_resource.get_child_resource(material_name, ContentManager(),
                                                                     CompiledMaterialResource)
            else:
                material_name = "NullMaterial"
            tint = draw_call.get("m_vTintColor", None)
            if material_resource:
                load_material(material_resource, Path(material_name), tint is not None)
                morph_supported = material_resource.get_int_property('F_MORPH_SUPPORTED', 0) == 1
                overlay = material_resource.get_int_property('F_OVERLAY', 0) == 1
                if not overlay:
                    data, = material_resource.get_data_block(block_name='DATA')
                    if data:
                        shader = data['m_shaderName']
                        overlay |= shader == "csgo_static_overlay.vfx"

            else:
                overlay = False
                morph_supported = bool(morph_block)
                logging.error(f'Failed to load material {material_name} for {mesh_resource.name}!')
            vertices = vertex_buffer.get_vertices()
            indices = index_buffer.get_indices()
            base_vertex = draw_call['m_nBaseVertex']
            vertex_count = draw_call['m_nVertexCount']
            start_index = draw_call['m_nStartIndex'] // 3
            index_count = draw_call['m_nIndexCount'] // 3

            part_indices = indices[start_index:start_index + index_count]
            used_vertices_ids, _, new_indices = np.unique(part_indices, return_index=True, return_inverse=True)

            used_vertices = vertices[base_vertex:][used_vertices_ids]
            del used_vertices_ids, part_indices

            material_stem = Path(material_name).stem
            model_name = mesh_name or mesh_resource.name
            mesh = bpy.data.meshes.new(f'{model_name}_{material_stem}_mesh')
            mesh_obj = bpy.data.objects.new(f'{model_name}_{material_stem}', mesh)

            positions = used_vertices['POSITION'] * scale
            if vertex_buffer.has_attribute('NORMAL'):
                normals = used_vertices['NORMAL']
                use_compressed_normal_tangent = draw_call.get('m_bUseCompressedNormalTangent', False)
                use_compressed_normal_tangent |= "COMPRESSED_NORMAL_TANGENT" in draw_call.get("m_nFlags", [])
                if use_compressed_normal_tangent:
                    if normals.dtype == np.uint32:
                        normals = convert_normals_2(normals)
                    else:
                        normals = convert_normals(normals)
            else:
                normals = None
            if overlay and normals is not None:
                positions += normals * 0.01

            mesh.from_pydata(positions, [], new_indices.reshape((-1, 3)).tolist())
            mesh.update()
            add_material(material_stem, mesh_obj)

            if data_block.get('m_materialGroups', None):
                default_skin = data_block['m_materialGroups'][0]

                if material_name in default_skin['m_materials']:
                    mat_id = default_skin['m_materials'].index(material_name)
                    mat_groups = {}
                    for skin_group in data_block['m_materialGroups']:
                        mat_groups[skin_group['m_name']] = skin_group['m_materials'][mat_id]

                    mesh_obj['active_skin'] = 'default'
                    mesh_obj['skin_groups'] = mat_groups
            else:
                mesh_obj['active_skin'] = 'default'
                mesh_obj['skin_groups'] = []
            mesh_obj['model_type'] = 'S2'

            vertex_indices = np.zeros((len(mesh.loops, )), dtype=np.uint32)
            mesh.loops.foreach_get('vertex_index', vertex_indices)
            if tint is not None:
                vertex_colors = mesh.vertex_colors.get('TINT', False) or mesh.vertex_colors.new(name='TINT')
                vertex_colors_data = vertex_colors.data
                tmp = np.ones((4,), np.float32)
                tmp[:3] = tint
                tint_data = np.full((vertex_count, 4), tmp, np.float32)
                vertex_colors_data.foreach_set('color', tint_data[vertex_indices].flatten())
            for uv_id in range(16):
                if uv_id == 0:
                    attrib_name = f"TEXCOORD"
                else:
                    attrib_name = f"TEXCOORD_{uv_id}"
                if vertex_buffer.has_attribute(attrib_name):
                    uv_layer = used_vertices[attrib_name].copy()
                    if uv_layer.shape[1] < 2:
                        continue
                    if uv_layer.shape[1] == 4:
                        uv_layer_0 = convert_to_float32(uv_layer[:, :2])
                        uv_layer_1 = convert_to_float32(uv_layer[:, 2:])
                        uv_layer_0[:, 1] = np.subtract(1, uv_layer_0[:, 1])
                        uv_layer_1[:, 1] = np.subtract(1, uv_layer_1[:, 1])

                        uv_data = mesh.uv_layers.new(name=attrib_name).data
                        uv_data.foreach_set('uv', uv_layer_0[vertex_indices].flatten())

                        uv_data = mesh.uv_layers.new(name=attrib_name + "_2").data
                        uv_data.foreach_set('uv', uv_layer_1[vertex_indices].flatten())
                        del uv_layer_0, uv_layer_1, uv_data

                    else:
                        uv_layer = convert_to_float32(uv_layer)
                        uv_layer[:, 1] = np.subtract(1, uv_layer[:, 1])

                        uv_data = mesh.uv_layers.new(name=attrib_name).data
                        uv_data.foreach_set('uv', uv_layer[vertex_indices].flatten())
                        del uv_layer, uv_data

            if vertex_buffer.has_attribute('NORMAL'):
                mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), np.uint32))
                normals = used_vertices['NORMAL']
                use_compressed_normal_tangent = draw_call.get('m_bUseCompressedNormalTangent', False)
                use_compressed_normal_tangent |= "COMPRESSED_NORMAL_TANGENT" in draw_call.get("m_nFlags", [])
                if use_compressed_normal_tangent:
                    if normals.dtype == np.uint32:
                        normals = convert_normals_2(normals)
                    else:
                        normals = convert_normals(normals)
                mesh.normals_split_custom_set_from_vertices(normals)
                mesh.use_auto_smooth = True

            if vertex_buffer.has_attribute('COLOR'):
                color = used_vertices['COLOR']
                vertex_colors = mesh.vertex_colors.get('COLOR', False) or mesh.vertex_colors.new(name='COLOR')
                vertex_colors_data = vertex_colors.data
                vertex_colors_data.foreach_set('color', color[vertex_indices].flatten())

            _add_vertex_groups(model_resource, vertex_buffer, mesh_id, used_vertices, mesh_obj)
            objects.append(mesh_obj)
            if morph_block and morph_supported:
                pos_bundle_id = morph_block.get_bundle_id('MORPH_BUNDLE_TYPE_POSITION_SPEED')
                if pos_bundle_id is None:
                    pos_bundle_id = morph_block.get_bundle_id('BUNDLE_TYPE_POSITION_SPEED')
                if pos_bundle_id is not None:
                    mesh_obj.shape_key_add(name='base')
                    for flex_name_ in morph_block['m_FlexDesc']:
                        flex_name = flex_name_['m_szFacs']
                        morph_data = morph_block.get_morph_data(flex_name, pos_bundle_id, cm)
                        if morph_data is None:
                            continue
                        flex_data = morph_data[:, :, :3].reshape((-1, 3))
                        flex_verts = flex_data[g_vertex_offset:g_vertex_offset + vertex_count]
                        if flex_verts.max() == 0.0 and flex_verts.min() == 0.0:
                            logging.debug(f'Skipping {flex_name!r} because flex delta is zero')
                            continue
                        shape = mesh_obj.shape_key_add(name=flex_name)

                        precomputed_data = np.add(flex_verts * scale, positions)
                        shape.data.foreach_set("co", precomputed_data.reshape(-1))
            g_vertex_offset += vertex_count

    return objects


def load_attachments(attachments_info: List[Object], container: Source2ModelContainer, scale: float):
    all_attachment = {}
    for attachment in attachments_info:
        if attachment['key'] not in all_attachment:
            all_attachment[attachment['key']] = attachment['value']

    for name, attachment in all_attachment.items():
        empty = bpy.data.objects.new(name, None)
        empty.empty_display_size = scale
        empty.matrix_basis.identity()

        if attachment['m_influenceNames'][0]:
            empty.parent = container.armature
            empty.parent_type = 'BONE'
            empty.parent_bone = attachment['m_influenceNames'][0]
        empty.location = Vector(attachment['m_vInfluenceOffsets'][0]) * scale
        empty.rotation_quaternion = Quaternion(attachment['m_vInfluenceRotations'][0])

        container.attachments.append(empty)


def get_physics_block(model_resource: CompiledModelResource) -> Optional[PhysBlock]:
    data: KVBlock = cast(KVBlock, model_resource.get_data_block(block_name='DATA')[0])

    cdata, = cast(Optional[KVBlock], model_resource.get_data_block(block_name='CTRL'))

    if 'm_refPhysicsData' in data and data['m_refPhysicsData']:
        for phys_file_path in data['m_refPhysicsData']:
            vphys = model_resource.get_child_resource(phys_file_path, ContentManager(), CompiledPhysicsResource)
            if not vphys:
                return None
            phys_data, = vphys.get_data_block(block_name="DATA")
            if phys_data is None:
                raise MissingBlock('Required block "DATA" is missing')
            return phys_data
    elif not cdata:
        return None
    elif 'embedded_physics' in cdata and cdata['embedded_physics']:
        block_id = cdata['embedded_physics']['phys_data_block']
        phys_data = model_resource.get_data_block(block_id=block_id)
        return phys_data
    return None

# class ValveCompiledModelLoader:
#     def __init__(self, path_or_file, re_use_meshes=False, scale=1.0):
#         super().__init__(path_or_file)
#         self.scale = scale
#         self.re_use_meshes = re_use_meshes
#         self.strip_from_name = ''
#         self.lod_collections = {}
#         self.materials = []
#         self.container = Source2ModelContainer(self)
#
#     def load_mesh(self, invert_uv, strip_from_name=''):
#         self.strip_from_name = strip_from_name
#
#         data_block = self.get_data_block(block_name='DATA')[0]
#
#         model_skeleton = data_block.data['m_modelSkeleton']
#         bone_names = model_skeleton['m_boneName']
#         if bone_names:
#             self.container.armature = self.build_armature()
#
#         self.build_meshes(self.container.armature, invert_uv)
#         self.load_materials()
#         self.load_physics()
#
#     def build_meshes(self, armature, invert_uv: bool = True):
#         content_manager = ContentManager()
#
#         data_block = self.get_data_block(block_name='DATA')[0]
#         use_external_meshes = len(self.get_data_block(block_name='CTRL')) == 0
#         if use_external_meshes:
#             for mesh_index, mesh_ref in enumerate(data_block.data['m_refMeshes']):
#                 if data_block.data['m_refLODGroupMasks'][mesh_index] & 1 == 0:
#                     continue
#                 mesh_ref_path = self.available_resources.get(mesh_ref, None)  # type:Path
#                 if mesh_ref_path:
#                     mesh_ref_file = content_manager.find_file(mesh_ref_path)
#                     if mesh_ref_file:
#                         mesh = ValveCompiledMesh(mesh_ref_file)
#                         self.available_resources.update(mesh.available_resources)
#                         mesh_data_block = mesh.get_data_block(block_name="DATA")[0]
#                         buffer_block = mesh.get_data_block(block_name="VBIB")[0]
#                         name = mesh_ref_path.stem
#                         vmorf_actual_path = mesh.available_resources.get(
#                             (mesh_data_block.data.get('m_morphSet', None) or
#                              mesh_data_block.data.get('m_pMorphSet', None)), None)
#                         morph_block: Optional = None
#                         if vmorf_actual_path:
#                             vmorf_path = content_manager.find_file(vmorf_actual_path)
#                             if vmorf_path is not None:
#                                 morph = ValveCompiledMorph(vmorf_path)
#                                 morph.read_block_info()
#                                 morph.check_external_resources()
#                                 morph_block = morph.get_data_block(block_name="DATA")[0]
#                         self.build_mesh(name, armature,
#                                         mesh_data_block, buffer_block, data_block, morph_block,
#                                         invert_uv, mesh_index)
#         else:
#             control_block = self.get_data_block(block_name="CTRL")[0]
#             e_meshes = control_block.data['embedded_meshes']
#             for e_mesh in e_meshes:
#                 name = e_mesh['name']
#                 name = name.replace(self.strip_from_name, "")
#                 data_block_index = e_mesh['data_block']
#                 mesh_index = e_mesh['mesh_index']
#                 if data_block.data['m_refLODGroupMasks'][mesh_index] & 1 == 0:
#                     continue
#
#                 buffer_block_index = e_mesh['vbib_block']
#                 morph_block_index = e_mesh['morph_block']
#
#                 mesh_data_block = self.get_data_block(block_id=data_block_index)
#                 buffer_block = self.get_data_block(block_id=buffer_block_index)
#                 morph_block = self.get_data_block(block_id=morph_block_index)
#
#                 self.build_mesh(name, armature,
#                                 mesh_data_block,
#                                 buffer_block,
#                                 data_block,
#                                 morph_block,
#                                 invert_uv,
#                                 mesh_index)
#
#     def build_mesh(self, name, armature,
#                    mesh_data_block: DATA,
#                    buffer_block: VBIB,
#                    data_block: DATA,
#                    morph_block: Optional[MRPH],
#                    invert_uv: bool,
#                    mesh_index: int
#                    ):
#
#         morphs_available = morph_block is not None and morph_block.read_morphs()
#         if morphs_available:
#             flex_trunc = bpy.data.texts.get(f"{name}_flexes", None) or bpy.data.texts.new(f"{name}_flexes")
#             for flex in morph_block.data['m_morphDatas']:
#                 if flex['m_name']:
#                     flex_trunc.write(f"{flex['m_name'][:63]}->{flex['m_name']}\n")
#
#         for scene in mesh_data_block.data["m_sceneObjects"]:
#             draw_calls = scene["m_drawCalls"]
#             global_vertex_offset = 0
#
#             for draw_call in draw_calls:
#
#                 material_ = draw_call.get('m_material', None) or draw_call.get('m_pMaterial', None)
#                 if isinstance(material_, int):
#                     material_ = self.available_resources.get(material_, str(material_))
#                 self.materials.append(material_)
#
#                 material_name = Path(material_).stem
#                 model_name = f"{name}_{material_name}_{draw_call['m_nStartIndex']}"
#                 used_copy = False
#                 mesh_obj = None
#                 if self.re_use_meshes:
#                     mesh_obj_original = bpy.data.objects.get(model_name, None)
#                     mesh_data_original = bpy.data.meshes.get(f'{model_name}_mesh', False)
#                     if mesh_obj_original and mesh_data_original:
#                         model_mesh = mesh_data_original.copy()
#                         mesh_obj = mesh_obj_original.copy()
#                         mesh_obj['skin_groups'] = mesh_obj_original['skin_groups']
#                         mesh_obj['active_skin'] = mesh_obj_original['active_skin']
#                         mesh_obj['model_type'] = 'S2'
#                         mesh_obj.data = model_mesh
#                         used_copy = True
#
#                 if not self.re_use_meshes or not used_copy:
#                     model_mesh = bpy.data.meshes.new(f'{model_name}_mesh')
#                     mesh_obj = bpy.data.objects.new(f'{model_name}', model_mesh)
#
#                 if data_block.data['m_materialGroups']:
#                     default_skin = data_block.data['m_materialGroups'][0]
#
#                     if material_ in default_skin['m_materials']:
#                         mat_id = default_skin['m_materials'].index(material_)
#                         mat_groups = {}
#                         for skin_group in data_block.data['m_materialGroups']:
#                             mat_groups[skin_group['m_name']] = skin_group['m_materials'][mat_id]
#
#                         mesh_obj['active_skin'] = 'default'
#                         mesh_obj['skin_groups'] = mat_groups
#                 else:
#                     mesh_obj['active_skin'] = 'default'
#                     mesh_obj['skin_groups'] = []
#
#                 material_name = Path(material_).stem
#                 mesh = mesh_obj.data
#
#                 self.container.objects.append(mesh_obj)
#
#                 if armature:
#                     modifier = mesh_obj.modifiers.new(
#                         type="ARMATURE", name="Armature")
#                     modifier.object = armature
#
#                 if used_copy:
#                     continue
#                 add_material(material_name, mesh_obj)
#
#                 base_vertex = draw_call['m_nBaseVertex']
#                 vertex_count = draw_call['m_nVertexCount']
#                 start_index = draw_call['m_nStartIndex'] // 3
#                 index_count = draw_call['m_nIndexCount'] // 3
#                 index_buffer = buffer_block.index_buffer[draw_call['m_indexBuffer']['m_hBuffer']]
#                 vertex_buffer = buffer_block.vertex_buffer[draw_call['m_vertexBuffers'][0]['m_hBuffer']]
#
#                 part_indices = index_buffer.indices[start_index:start_index + index_count]
#                 used_vertices_ids, _, new_indices = np.unique(part_indices, return_index=True, return_inverse=True)
#
#                 used_vertices = vertex_buffer.vertexes[base_vertex:][used_vertices_ids]
#
#                 positions = used_vertices['POSITION']
#
#                 mesh.from_pydata(positions * self.scale, [], new_indices.reshape((-1, 3)).tolist())
#                 mesh.update()
#                 n = 0
#                 for attrib in vertex_buffer.attributes:
#                     if 'TEXCOORD' in attrib.name.upper():
#                         uv_layer = used_vertices[attrib.name].copy()
#                         if uv_layer.dtype == np.uint16:
#                             uv_layer = uv_layer.astype(np.float32) / 65535
#                         elif uv_layer.dtype == np.int16:
#                             uv_layer = uv_layer.astype(np.float32) / 32767
#                         if uv_layer.shape[1] != 2:
#                             continue
#                         if invert_uv:
#                             uv_layer[:, 1] = np.subtract(1, uv_layer[:, 1])
#
#                         uv_data = mesh.uv_layers.new(name=attrib.name).data
#                         vertex_indices = np.zeros((len(mesh.loops, )), dtype=np.uint32)
#                         mesh.loops.foreach_get('vertex_index', vertex_indices)
#                         new_uv_data = uv_layer[vertex_indices]
#                         uv_data.foreach_set('uv', new_uv_data.flatten())
#                         n += 1
#                     if 'COLOR' in attrib.name.upper():
#                         color = used_vertices[attrib.name]
#                         vertex_indices = np.zeros((len(mesh.loops, )), dtype=np.uint32)
#                         mesh.loops.foreach_get('vertex_index', vertex_indices)
#                         vertex_colors = mesh.vertex_colors.get(attrib.name, False) or \
#                                         mesh.vertex_colors.new(name=attrib.name)
#                         vertex_colors_data = vertex_colors.data
#                         vertex_colors_data.foreach_set('color', color[vertex_indices].flatten())
#                 if armature:
#                     model_skeleton = data_block.data['m_modelSkeleton']
#                     bone_names = model_skeleton['m_boneName']
#                     remap_table = data_block.data['m_remappingTable']
#                     remap_table_starts = data_block.data['m_remappingTableStarts']
#                     remaps_start = remap_table_starts[mesh_index]
#                     new_bone_names = bone_names.copy()
#                     weight_groups = {bone: mesh_obj.vertex_groups.new(name=bone) for bone in new_bone_names}
#
#                     names = vertex_buffer.attribute_names
#                     if 'BLENDWEIGHT' in names and 'BLENDINDICES' in names:
#                         weights_array = used_vertices["BLENDWEIGHT"] / 255
#                         indices_array = used_vertices["BLENDINDICES"]
#                     elif 'BLENDINDICES' in names:
#                         indices_array = used_vertices["BLENDINDICES"]
#                         weights_array = np.ones_like(indices_array).astype(np.float32)
#                     else:
#                         weights_array = []
#                         indices_array = []
#
#                     for n, bone_indices in enumerate(indices_array):
#
#                         if len(weights_array) > 0:
#                             weights = weights_array[n]
#                             for bone_index, weight in zip(bone_indices, weights):
#                                 if weight > 0:
#                                     bone_name = new_bone_names[remap_table[remaps_start:][int(bone_index)]]
#                                     weight_groups[bone_name].add([n], weight, 'REPLACE')
#
#                         else:
#                             for bone_index in bone_indices:
#                                 bone_name = new_bone_names[remap_table[remaps_start:][int(bone_index)]]
#                                 weight_groups[bone_name].add([n], 1.0, 'REPLACE')
#
#                 mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), np.uint32))
#                 if 'NORMAL' in vertex_buffer.attribute_names:
#                     normals = used_vertices['NORMAL']
#                     if normals.dtype.char == 'B' and normals.shape[1] == 4:
#                         normals = convert_normals(normals)
#                     mesh.normals_split_custom_set_from_vertices(normals)
#                 mesh.use_auto_smooth = True
#
#                 if morphs_available:
#                     mesh_obj.shape_key_add(name='base')
#                     bundle_types = morph_block.data['m_bundleTypes']
#                     if bundle_types and isinstance(bundle_types[0], tuple):
#                         bundle_types = [b[0] for b in bundle_types]
#                     if 'MORPH_BUNDLE_TYPE_POSITION_SPEED' in bundle_types:
#                         bundle_id = bundle_types.index('MORPH_BUNDLE_TYPE_POSITION_SPEED')
#                     elif 'BUNDLE_TYPE_POSITION_SPEED' in bundle_types:
#                         bundle_id = bundle_types.index('BUNDLE_TYPE_POSITION_SPEED')
#                     else:
#                         bundle_id = -1
#                     if bundle_id != -1:
#                         for n, (flex_name, flex_data) in enumerate(morph_block.flex_data.items()):
#                             print(f"Importing {flex_name} {n + 1}/{len(morph_block.flex_data)}")
#                             if flex_name is None:
#                                 continue
#
#                             shape = mesh_obj.shape_key_add(name=flex_name)
#                             vertices = np.zeros((len(mesh.vertices) * 3,), dtype=np.float32)
#                             mesh.vertices.foreach_get('co', vertices)
#                             vertices = vertices.reshape((-1, 3))
#                             bundle_data = flex_data[bundle_id]
#                             pre_computed_data = np.add(
#                                 bundle_data[global_vertex_offset:global_vertex_offset + vertex_count][:, :3], vertices)
#                             shape.data.foreach_set("co", pre_computed_data.reshape((-1,)))
#                     # expressions = morph_block.rebuild_flex_expressions()
#                     # print()
#                 global_vertex_offset += vertex_count
#
#     def build_armature(self):
#         data_block = self.get_data_block(block_name='DATA')[0]
#         model_skeleton = data_block.data['m_modelSkeleton']
#         bone_names = model_skeleton['m_boneName']
#         bone_positions = model_skeleton['m_bonePosParent']
#         bone_rotations = model_skeleton['m_boneRotParent']
#         bone_parents = model_skeleton['m_nParent']
#
#         armature_obj = bpy.data.objects.new(self.name + "_ARM", bpy.data.armatures.new(self.name + "_ARM_DATA"))
#         armature_obj['MODE'] = 'SourceIO'
#         armature_obj.show_in_front = True
#         bpy.context.scene.collection.objects.link(armature_obj)
#         bpy.ops.object.select_all(action="DESELECT")
#         armature_obj.select_set(True)
#         bpy.context.view_layer.objects.active = armature_obj
#
#         armature_obj.rotation_euler = Euler([math.radians(180), 0, math.radians(90)])
#         armature = armature_obj.data
#
#         bpy.ops.object.mode_set(mode='EDIT')
#
#         bones = []
#         for bone_name in bone_names:
#             bl_bone = armature.edit_bones.new(name=bone_name)
#             bl_bone.tail = Vector([0, 0, 1]) + bl_bone.head * self.scale
#             bones.append((bl_bone, bone_name))
#
#         for n, bone_name in enumerate(bone_names):
#             bl_bone = armature.edit_bones.get(bone_name)
#             parent_id = bone_parents[n]
#             if parent_id != -1:
#                 bl_parent, parent = bones[parent_id]
#                 bl_bone.parent = bl_parent
#
#         bpy.ops.object.mode_set(mode='POSE')
#         for n, (bl_bone, bone_name) in enumerate(bones):
#             pose_bone = armature_obj.pose.bones.get(bone_name)
#             if pose_bone is None:
#                 print("Missing", bone_name, 'bone')
#             parent_id = bone_parents[n]
#             bone_pos = bone_positions[n]
#             bone_rot = bone_rotations[n]
#             bone_pos = Vector([bone_pos[1], bone_pos[0], -bone_pos[2]]) * self.scale
#             # noinspection PyTypeChecker
#             bone_rot = Quaternion([-bone_rot[3], -bone_rot[1], -bone_rot[0], bone_rot[2]])
#             mat = (Matrix.Translation(bone_pos) @ bone_rot.to_matrix().to_4x4())
#             pose_bone.matrix_basis.identity()
#
#             if parent_id != -1:
#                 parent_bone = armature_obj.pose.bones.get(bone_names[parent_id])
#                 pose_bone.matrix = parent_bone.matrix @ mat
#             else:
#                 pose_bone.matrix = mat
#         bpy.ops.pose.armature_apply()
#         bpy.ops.object.mode_set(mode='OBJECT')
#         bpy.ops.object.select_all(action="DESELECT")
#         armature_obj.select_set(True)
#         bpy.context.view_layer.objects.active = armature_obj
#         bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)
#         bpy.context.scene.collection.objects.unlink(armature_obj)
#         return armature_obj
#
#     def load_attachments(self):
#         all_attachment = {}
#         for block in self.get_data_block(block_name="MDAT"):
#             for attachment in block.data['m_attachments']:
#                 if attachment['key'] not in all_attachment:
#                     all_attachment[attachment['key']] = attachment['value']
#
#         for name, attachment in all_attachment.items():
#             empty = bpy.data.objects.new(name, None)
#             self.container.attachments.append(empty)
#             pos = attachment['m_vInfluenceOffsets'][0]
#             rot = Quaternion(attachment['m_vInfluenceRotations'][0])
#             empty.matrix_basis.identity()
#
#             if attachment['m_influenceNames'][0]:
#                 empty.parent = self.container.armature
#                 empty.parent_type = 'BONE'
#                 empty.parent_bone = attachment['m_influenceNames'][0]
#             empty.location = Vector([pos[1], pos[0], pos[2]]) * self.scale
#             empty.rotation_quaternion = rot
#
#     def load_animations(self):
#         if not self.get_data_block(block_name='CTRL'):
#             return
#         armature = self.container.armature
#         if armature:
#             if not armature.animation_data:
#                 armature.animation_data_create()
#
#             bpy.ops.object.select_all(action="DESELECT")
#             armature.select_set(True)
#             bpy.context.view_layer.objects.active = armature
#             bpy.ops.object.mode_set(mode='POSE')
#
#             ctrl_block = self.get_data_block(block_name='CTRL')[0]
#             embedded_anim = ctrl_block.data['embedded_animation']
#             agrp = self.get_data_block(block_id=embedded_anim['group_data_block'])
#             anim_data = self.get_data_block(block_id=embedded_anim['anim_data_block'])
#
#             animations = parse_anim_data(anim_data.data, agrp.data)
#             bone_array = agrp.data['m_decodeKey']['m_boneArray']
#
#             for animation in animations:
#                 print(f"Loading animation {animation.name}")
#                 action = bpy.data.actions.new(animation.name)
#                 armature.animation_data.action = action
#                 curve_per_bone = {}
#                 for bone in bone_array:
#                     bone_string = f'pose.bones["{bone["m_name"]}"].'
#                     group = action.groups.new(name=bone['m_name'])
#                     pos_curves = []
#                     rot_curves = []
#                     for i in range(3):
#                         pos_curve = action.fcurves.new(data_path=bone_string + "location", index=i)
#                         pos_curve.keyframe_points.add(len(animation.frames))
#                         pos_curves.append(pos_curve)
#                         pos_curve.group = group
#                     for i in range(4):
#                         rot_curve = action.fcurves.new(data_path=bone_string + "rotation_quaternion", index=i)
#                         rot_curve.keyframe_points.add(len(animation.frames))
#                         rot_curves.append(rot_curve)
#                         rot_curve.group = group
#                     curve_per_bone[bone['m_name']] = pos_curves, rot_curves
#
#                 for n, frame in enumerate(animation.frames):
#                     for bone_name, bone_data in frame.bone_data.items():
#                         bone_data = frame.bone_data[bone_name]
#                         pos_curves, rot_curves = curve_per_bone[bone_name]
#
#                         pos_type, pos = bone_data['Position']
#                         rot_type, rot = bone_data['Angle']
#
#                         bone_pos = Vector([pos[1], pos[0], -pos[2]]) * self.scale
#                         # noinspection PyTypeChecker
#                         bone_rot = Quaternion([-rot[3], -rot[1], -rot[0], rot[2]])
#
#                         bone = armature.pose.bones[bone_name]
#                         # mat = (Matrix.Translation(bone_pos) @ bone_rot.to_matrix().to_4x4())
#                         translation_mat = Matrix.Identity(4)
#
#                         if 'Position' in bone_data:
#                             if pos_type in ['CCompressedFullVector3',
#                                             'CCompressedAnimVector3',
#                                             'CCompressedStaticFullVector3']:
#                                 translation_mat = Matrix.Translation(bone_pos)
#                             # elif pos_type == "CCompressedDeltaVector3":
#                             # 'CCompressedStaticVector3',
#                             #     a, b, c = decompose(mat)
#                             #     a += bone_pos
#                             #     translation_mat = compose(a, b, c)
#                         rotation_mat = Matrix.Identity(4)
#                         if 'Angle' in bone_data:
#
#                             if rot_type in ['CCompressedAnimQuaternion',
#                                             'CCompressedFullQuaternion',
#                                             'CCompressedStaticQuaternion']:
#                                 rotation_mat = bone_rot.to_matrix().to_4x4()
#
#                         mat = translation_mat @ rotation_mat
#                         if bone.parent:
#                             bone.matrix = bone.parent.matrix @ mat
#                         else:
#                             bone.matrix = bone.matrix @ mat
#
#                         if 'Position' in bone_data:
#                             for i in range(3):
#                                 pos_curves[i].keyframe_points.add(1)
#                                 pos_curves[i].keyframe_points[-1].co = (n, bone.location[i])
#                         if 'Angle' in bone_data:
#                             for i in range(4):
#                                 rot_curves[i].keyframe_points.add(1)
#                                 rot_curves[i].keyframe_points[-1].co = (n, bone.rotation_quaternion[i])
#
#     def load_materials(self):
#         content_manager = ContentManager()
#         for material in self.materials:
#             print(f'Loading {material}')
#             file = self.available_resources.get(material, None)
#             if file:
#                 file = content_manager.find_file(file)
#                 if file:  # duh
#                     material = ValveCompiledMaterialLoader(file)
#                     material.load()
#
#     def load_physics(self):
#         data = self.get_data_block(block_name='DATA')[0]
#         cdata = self.get_data_block(block_name='CTRL')
#         if not cdata:
#             return
#         cdata = cdata[0]
#         if 'm_refPhysicsData' in data.data and data.data['m_refPhysicsData']:
#             for phys_file_path in data.data['m_refPhysicsData']:
#                 if phys_file_path not in self.available_resources:
#                     continue
#                 phys_file_path = self.available_resources[phys_file_path]
#                 phys_file = ContentManager().find_file(phys_file_path)
#                 if not phys_file:
#                     continue
#                 vphys = ValveCompiledPhysicsLoader(phys_file, self.scale)
#                 vphys.parse_meshes()
#                 phys_meshes = vphys.build_mesh()
#                 self.container.physics_objects.extend(phys_meshes)
#         elif 'embedded_physics' in cdata.data and cdata.data['embedded_physics']:
#             block_id = cdata.data['embedded_physics']['phys_data_block']
#             data = self.get_data_block(block_id=block_id)
#             vphys = ValveCompiledPhysicsLoader(None, self.scale)
#             vphys.data_block = data
#             vphys.parse_meshes()
#             phys_meshes = vphys.build_mesh()
#             self.container.physics_objects.extend(phys_meshes)
