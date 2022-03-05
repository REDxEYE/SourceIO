import math
from pathlib import Path
from typing import Optional

import bpy
from itertools import chain

import numpy as np
from mathutils import Quaternion, Vector, Matrix, Euler

from ..vmat.loader import ValveCompiledMaterialLoader
from ..vphys.loader import ValveCompiledPhysicsLoader
from ...shared.model_container import Source2ModelContainer
from ...utils.utils import get_material, get_new_unique_collection, find_layer_collection
from ....library.source2.common import convert_normals
from ....library.source2.data_blocks import MRPH, VBIB, DATA
from ....library.source2.utils.decode_animations import parse_anim_data
from ....library.source2.resource_types.vmorf.morph import ValveCompiledMorph
from ....library.shared.content_providers.content_manager import ContentManager
from ....library.source2.resource_types import ValveCompiledModel, ValveCompiledResource


def put_into_collections(model_container, model_name, parent_collection=None, bodygroup_grouping=False):
    model_container: Source2ModelContainer
    static_prop = model_container.armature is None and not model_container.physics_objects
    if static_prop:
        master_collection = parent_collection or bpy.context.scene.collection
    else:
        master_collection = get_new_unique_collection(model_name, parent_collection or bpy.context.scene.collection)
    for obj in model_container.objects:
        master_collection.objects.link(obj)

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
    return master_collection


class ValveCompiledModelLoader(ValveCompiledModel):
    def __init__(self, path_or_file, re_use_meshes=False, scale=1.0):
        super().__init__(path_or_file)
        self.scale = scale
        self.re_use_meshes = re_use_meshes
        self.strip_from_name = ''
        self.lod_collections = {}
        self.materials = []
        self.container = Source2ModelContainer(self)

    def load_mesh(self, invert_uv, strip_from_name=''):
        self.strip_from_name = strip_from_name

        data_block = self.get_data_block(block_name='DATA')[0]

        model_skeleton = data_block.data['m_modelSkeleton']
        bone_names = model_skeleton['m_boneName']
        if bone_names:
            self.container.armature = self.build_armature()

        self.build_meshes(self.container.armature, invert_uv)
        self.load_materials()
        self.load_physics()

    def build_meshes(self, armature, invert_uv: bool = True):
        content_manager = ContentManager()

        data_block = self.get_data_block(block_name='DATA')[0]
        use_external_meshes = len(self.get_data_block(block_name='CTRL')) == 0
        if use_external_meshes:
            for mesh_index, mesh_ref in enumerate(data_block.data['m_refMeshes']):
                if data_block.data['m_refLODGroupMasks'][mesh_index] & 1 == 0:
                    continue
                mesh_ref_path = self.available_resources.get(mesh_ref, None)  # type:Path
                if mesh_ref_path:
                    mesh_ref_file = content_manager.find_file(mesh_ref_path)
                    if mesh_ref_file:
                        mesh = ValveCompiledResource(mesh_ref_file)
                        self.available_resources.update(mesh.available_resources)
                        mesh.read_block_info()
                        mesh.check_external_resources()
                        mesh_data_block = mesh.get_data_block(block_name="DATA")[0]
                        buffer_block = mesh.get_data_block(block_name="VBIB")[0]
                        name = mesh_ref_path.stem
                        vmorf_actual_path = mesh.available_resources.get(mesh_data_block.data['m_morphSet'], None)
                        morph_block: Optional = None
                        if vmorf_actual_path:
                            vmorf_path = content_manager.find_file(vmorf_actual_path)
                            if vmorf_path is not None:
                                morph = ValveCompiledMorph(vmorf_path)
                                morph.read_block_info()
                                morph.check_external_resources()
                                morph_block = morph.get_data_block(block_name="DATA")[0]
                        self.build_mesh(name, armature,
                                        mesh_data_block, buffer_block, data_block, morph_block,
                                        invert_uv, mesh_index)
        else:
            control_block = self.get_data_block(block_name="CTRL")[0]
            e_meshes = control_block.data['embedded_meshes']
            for e_mesh in e_meshes:
                name = e_mesh['name']
                name = name.replace(self.strip_from_name, "")
                data_block_index = e_mesh['data_block']
                mesh_index = e_mesh['mesh_index']
                if data_block.data['m_refLODGroupMasks'][mesh_index] & 1 == 0:
                    continue

                buffer_block_index = e_mesh['vbib_block']
                morph_block_index = e_mesh['morph_block']

                mesh_data_block = self.get_data_block(block_id=data_block_index)
                buffer_block = self.get_data_block(block_id=buffer_block_index)
                morph_block = self.get_data_block(block_id=morph_block_index)

                self.build_mesh(name, armature,
                                mesh_data_block,
                                buffer_block,
                                data_block,
                                morph_block,
                                invert_uv,
                                mesh_index)

    def build_mesh(self, name, armature,
                   mesh_data_block: DATA,
                   buffer_block: VBIB,
                   data_block: DATA,
                   morph_block: Optional[MRPH],
                   invert_uv: bool,
                   mesh_index: int
                   ):

        morphs_available = morph_block is not None and morph_block.read_morphs()
        if morphs_available:
            flex_trunc = bpy.data.texts.get(f"{name}_flexes", None) or bpy.data.texts.new(f"{name}_flexes")
            for flex in morph_block.data['m_morphDatas']:
                if flex['m_name']:
                    flex_trunc.write(f"{flex['m_name'][:63]}->{flex['m_name']}\n")

        for scene in mesh_data_block.data["m_sceneObjects"]:
            draw_calls = scene["m_drawCalls"]
            global_vertex_offset = 0

            for draw_call in draw_calls:

                self.materials.append(draw_call['m_material'])

                material_name = Path(draw_call['m_material']).stem
                model_name = f"{name}_{material_name}_{draw_call['m_nStartIndex']}"
                used_copy = False
                mesh_obj = None
                if self.re_use_meshes:
                    mesh_obj_original = bpy.data.objects.get(model_name, None)
                    mesh_data_original = bpy.data.meshes.get(f'{model_name}_mesh', False)
                    if mesh_obj_original and mesh_data_original:
                        model_mesh = mesh_data_original.copy()
                        mesh_obj = mesh_obj_original.copy()
                        mesh_obj['skin_groups'] = mesh_obj_original['skin_groups']
                        mesh_obj['active_skin'] = mesh_obj_original['active_skin']
                        mesh_obj['model_type'] = 'S2'
                        mesh_obj.data = model_mesh
                        used_copy = True

                if not self.re_use_meshes or not used_copy:
                    model_mesh = bpy.data.meshes.new(f'{model_name}_mesh')
                    mesh_obj = bpy.data.objects.new(f'{model_name}', model_mesh)

                if data_block.data['m_materialGroups']:
                    default_skin = data_block.data['m_materialGroups'][0]

                    if draw_call['m_material'] in default_skin['m_materials']:
                        mat_id = default_skin['m_materials'].index(draw_call['m_material'])
                        mat_groups = {}
                        for skin_group in data_block.data['m_materialGroups']:
                            mat_groups[skin_group['m_name']] = skin_group['m_materials'][mat_id]

                        mesh_obj['active_skin'] = 'default'
                        mesh_obj['skin_groups'] = mat_groups
                else:
                    mesh_obj['active_skin'] = 'default'
                    mesh_obj['skin_groups'] = []

                material_name = Path(draw_call['m_material']).stem
                mesh = mesh_obj.data

                self.container.objects.append(mesh_obj)

                if armature:
                    modifier = mesh_obj.modifiers.new(
                        type="ARMATURE", name="Armature")
                    modifier.object = armature

                if used_copy:
                    continue
                get_material(material_name, mesh_obj)

                base_vertex = draw_call['m_nBaseVertex']
                vertex_count = draw_call['m_nVertexCount']
                start_index = draw_call['m_nStartIndex'] // 3
                index_count = draw_call['m_nIndexCount'] // 3
                index_buffer = buffer_block.index_buffer[draw_call['m_indexBuffer']['m_hBuffer']]
                vertex_buffer = buffer_block.vertex_buffer[draw_call['m_vertexBuffers'][0]['m_hBuffer']]

                part_indices = index_buffer.indices[start_index:start_index + index_count]
                used_vertices_ids, _, new_indices = np.unique(part_indices, return_index=True, return_inverse=True)

                used_vertices = vertex_buffer.vertexes[base_vertex:][used_vertices_ids]

                positions = used_vertices['POSITION']
                normals = used_vertices['NORMAL']

                if normals.dtype.char == 'B' and normals.shape[1] == 4:
                    normals = convert_normals(normals)

                mesh.from_pydata(positions * self.scale, [], new_indices.reshape((-1, 3)).tolist())
                mesh.update()
                n = 0
                for attrib in vertex_buffer.attributes:
                    if 'TEXCOORD' in attrib.name.upper():
                        uv_layer = used_vertices[attrib.name].copy()
                        if uv_layer.shape[1] != 2:
                            continue
                        if invert_uv:
                            uv_layer[:, 1] = np.subtract(1, uv_layer[:, 1])

                        uv_data = mesh.uv_layers.new(name=attrib.name).data
                        vertex_indices = np.zeros((len(mesh.loops, )), dtype=np.uint32)
                        mesh.loops.foreach_get('vertex_index', vertex_indices)
                        new_uv_data = uv_layer[vertex_indices]
                        uv_data.foreach_set('uv', new_uv_data.flatten())
                        n += 1
                if armature:
                    model_skeleton = data_block.data['m_modelSkeleton']
                    bone_names = model_skeleton['m_boneName']
                    remap_table = data_block.data['m_remappingTable']
                    remap_table_starts = data_block.data['m_remappingTableStarts']
                    remaps_start = remap_table_starts[mesh_index]
                    new_bone_names = bone_names.copy()
                    weight_groups = {bone: mesh_obj.vertex_groups.new(name=bone) for bone in new_bone_names}

                    names = vertex_buffer.attribute_names
                    if 'BLENDWEIGHT' in names and 'BLENDINDICES' in names:
                        weights_array = used_vertices["BLENDWEIGHT"] / 255
                        indices_array = used_vertices["BLENDINDICES"]
                    elif 'BLENDINDICES' in names:
                        indices_array = used_vertices["BLENDINDICES"]
                        weights_array = np.ones_like(indices_array).astype(np.float32)
                    else:
                        weights_array = []
                        indices_array = []

                    for n, bone_indices in enumerate(indices_array):

                        if len(weights_array) > 0:
                            weights = weights_array[n]
                            for bone_index, weight in zip(bone_indices, weights):
                                if weight > 0:
                                    bone_name = new_bone_names[remap_table[remaps_start:][int(bone_index)]]
                                    weight_groups[bone_name].add([n], weight, 'REPLACE')

                        else:
                            for bone_index in bone_indices:
                                bone_name = new_bone_names[remap_table[remaps_start:][int(bone_index)]]
                                weight_groups[bone_name].add([n], 1.0, 'REPLACE')

                mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), np.uint32))
                mesh.normals_split_custom_set_from_vertices(normals)
                mesh.use_auto_smooth = True
                if morphs_available:
                    mesh_obj.shape_key_add(name='base')
                    bundle_types = morph_block.data['m_bundleTypes']
                    if bundle_types and isinstance(bundle_types[0], tuple):
                        bundle_types = [b[0] for b in bundle_types]
                    if 'MORPH_BUNDLE_TYPE_POSITION_SPEED' in bundle_types:
                        bundle_id = bundle_types.index('MORPH_BUNDLE_TYPE_POSITION_SPEED')
                    elif 'BUNDLE_TYPE_POSITION_SPEED' in bundle_types:
                        bundle_id = bundle_types.index('BUNDLE_TYPE_POSITION_SPEED')
                    else:
                        bundle_id = -1
                    if bundle_id != -1:
                        for n, (flex_name, flex_data) in enumerate(morph_block.flex_data.items()):
                            print(f"Importing {flex_name} {n + 1}/{len(morph_block.flex_data)}")
                            if flex_name is None:
                                continue

                            shape = mesh_obj.shape_key_add(name=flex_name)
                            vertices = np.zeros((len(mesh.vertices) * 3,), dtype=np.float32)
                            mesh.vertices.foreach_get('co', vertices)
                            vertices = vertices.reshape((-1, 3))
                            bundle_data = flex_data[bundle_id]
                            pre_computed_data = np.add(
                                bundle_data[global_vertex_offset:global_vertex_offset + vertex_count][:, :3], vertices)
                            shape.data.foreach_set("co", pre_computed_data.reshape((-1,)))
                    expressions = morph_block.rebuild_flex_expressions()
                    print()
                global_vertex_offset += vertex_count

    def build_armature(self):
        data_block = self.get_data_block(block_name='DATA')[0]
        model_skeleton = data_block.data['m_modelSkeleton']
        bone_names = model_skeleton['m_boneName']
        bone_positions = model_skeleton['m_bonePosParent']
        bone_rotations = model_skeleton['m_boneRotParent']
        bone_parents = model_skeleton['m_nParent']

        armature_obj = bpy.data.objects.new(self.name + "_ARM", bpy.data.armatures.new(self.name + "_ARM_DATA"))
        armature_obj['MODE'] = 'SourceIO'
        armature_obj.show_in_front = True
        bpy.context.scene.collection.objects.link(armature_obj)
        bpy.ops.object.select_all(action="DESELECT")
        armature_obj.select_set(True)
        bpy.context.view_layer.objects.active = armature_obj

        armature_obj.rotation_euler = Euler([math.radians(180), 0, math.radians(90)])
        armature = armature_obj.data

        bpy.ops.object.mode_set(mode='EDIT')

        bones = []
        for bone_name in bone_names:
            bl_bone = armature.edit_bones.new(name=bone_name)
            bl_bone.tail = Vector([0, 0, 1]) + bl_bone.head * self.scale
            bones.append((bl_bone, bone_name))

        for n, bone_name in enumerate(bone_names):
            bl_bone = armature.edit_bones.get(bone_name)
            parent_id = bone_parents[n]
            if parent_id != -1:
                bl_parent, parent = bones[parent_id]
                bl_bone.parent = bl_parent

        bpy.ops.object.mode_set(mode='POSE')
        for n, (bl_bone, bone_name) in enumerate(bones):
            pose_bone = armature_obj.pose.bones.get(bone_name)
            if pose_bone is None:
                print("Missing", bone_name, 'bone')
            parent_id = bone_parents[n]
            bone_pos = bone_positions[n]
            bone_rot = bone_rotations[n]
            bone_pos = Vector([bone_pos[1], bone_pos[0], -bone_pos[2]]) * self.scale
            # noinspection PyTypeChecker
            bone_rot = Quaternion([-bone_rot[3], -bone_rot[1], -bone_rot[0], bone_rot[2]])
            mat = (Matrix.Translation(bone_pos) @ bone_rot.to_matrix().to_4x4())
            pose_bone.matrix_basis.identity()

            if parent_id != -1:
                parent_bone = armature_obj.pose.bones.get(bone_names[parent_id])
                pose_bone.matrix = parent_bone.matrix @ mat
            else:
                pose_bone.matrix = mat
        bpy.ops.pose.armature_apply()
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action="DESELECT")
        armature_obj.select_set(True)
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)
        bpy.context.scene.collection.objects.unlink(armature_obj)
        return armature_obj

    def load_attachments(self):
        all_attachment = {}
        for block in self.get_data_block(block_name="MDAT"):
            for attachment in block.data['m_attachments']:
                if attachment['key'] not in all_attachment:
                    all_attachment[attachment['key']] = attachment['value']

        for name, attachment in all_attachment.items():
            empty = bpy.data.objects.new(name, None)
            self.container.attachments.append(empty)
            pos = attachment['m_vInfluenceOffsets'][0]
            rot = Quaternion(attachment['m_vInfluenceRotations'][0])
            empty.matrix_basis.identity()

            if attachment['m_influenceNames'][0]:
                empty.parent = self.container.armature
                empty.parent_type = 'BONE'
                empty.parent_bone = attachment['m_influenceNames'][0]
            empty.location = Vector([pos[1], pos[0], pos[2]]) * self.scale
            empty.rotation_quaternion = rot

    def load_animations(self):
        if not self.get_data_block(block_name='CTRL'):
            return
        armature = self.container.armature
        if armature:
            if not armature.animation_data:
                armature.animation_data_create()

            bpy.ops.object.select_all(action="DESELECT")
            armature.select_set(True)
            bpy.context.view_layer.objects.active = armature
            bpy.ops.object.mode_set(mode='POSE')

            ctrl_block = self.get_data_block(block_name='CTRL')[0]
            embedded_anim = ctrl_block.data['embedded_animation']
            agrp = self.get_data_block(block_id=embedded_anim['group_data_block'])
            anim_data = self.get_data_block(block_id=embedded_anim['anim_data_block'])

            animations = parse_anim_data(anim_data.data, agrp.data)
            bone_array = agrp.data['m_decodeKey']['m_boneArray']

            for animation in animations:
                print(f"Loading animation {animation.name}")
                action = bpy.data.actions.new(animation.name)
                armature.animation_data.action = action
                curve_per_bone = {}
                for bone in bone_array:
                    bone_string = f'pose.bones["{bone["m_name"]}"].'
                    group = action.groups.new(name=bone['m_name'])
                    pos_curves = []
                    rot_curves = []
                    for i in range(3):
                        pos_curve = action.fcurves.new(data_path=bone_string + "location", index=i)
                        pos_curve.keyframe_points.add(len(animation.frames))
                        pos_curves.append(pos_curve)
                        pos_curve.group = group
                    for i in range(4):
                        rot_curve = action.fcurves.new(data_path=bone_string + "rotation_quaternion", index=i)
                        rot_curve.keyframe_points.add(len(animation.frames))
                        rot_curves.append(rot_curve)
                        rot_curve.group = group
                    curve_per_bone[bone['m_name']] = pos_curves, rot_curves

                for n, frame in enumerate(animation.frames):
                    for bone_name, bone_data in frame.bone_data.items():
                        bone_data = frame.bone_data[bone_name]
                        pos_curves, rot_curves = curve_per_bone[bone_name]

                        pos_type, pos = bone_data['Position']
                        rot_type, rot = bone_data['Angle']

                        bone_pos = Vector([pos[1], pos[0], -pos[2]]) * self.scale
                        # noinspection PyTypeChecker
                        bone_rot = Quaternion([-rot[3], -rot[1], -rot[0], rot[2]])

                        bone = armature.pose.bones[bone_name]
                        # mat = (Matrix.Translation(bone_pos) @ bone_rot.to_matrix().to_4x4())
                        translation_mat = Matrix.Identity(4)

                        if 'Position' in bone_data:
                            if pos_type in ['CCompressedFullVector3',
                                            'CCompressedAnimVector3',
                                            'CCompressedStaticFullVector3']:
                                translation_mat = Matrix.Translation(bone_pos)
                            # elif pos_type == "CCompressedDeltaVector3":
                            # 'CCompressedStaticVector3',
                            #     a, b, c = decompose(mat)
                            #     a += bone_pos
                            #     translation_mat = compose(a, b, c)
                        rotation_mat = Matrix.Identity(4)
                        if 'Angle' in bone_data:

                            if rot_type in ['CCompressedAnimQuaternion',
                                            'CCompressedFullQuaternion',
                                            'CCompressedStaticQuaternion']:
                                rotation_mat = bone_rot.to_matrix().to_4x4()

                        mat = translation_mat @ rotation_mat
                        if bone.parent:
                            bone.matrix = bone.parent.matrix @ mat
                        else:
                            bone.matrix = bone.matrix @ mat

                        if 'Position' in bone_data:
                            for i in range(3):
                                pos_curves[i].keyframe_points.add(1)
                                pos_curves[i].keyframe_points[-1].co = (n, bone.location[i])
                        if 'Angle' in bone_data:
                            for i in range(4):
                                rot_curves[i].keyframe_points.add(1)
                                rot_curves[i].keyframe_points[-1].co = (n, bone.rotation_quaternion[i])

    def load_materials(self):
        content_manager = ContentManager()
        for material in self.materials:
            print(f'Loading {material}')
            file = self.available_resources.get(material, None)
            if file:
                file = content_manager.find_file(file)
                if file:  # duh
                    material = ValveCompiledMaterialLoader(file)
                    material.load()

    def load_physics(self):
        data = self.get_data_block(block_name='DATA')[0]
        cdata = self.get_data_block(block_name='CTRL')
        if not cdata:
            return
        cdata = cdata[0]
        if 'm_refPhysicsData' in data.data and data.data['m_refPhysicsData']:
            for phys_file_path in data.data['m_refPhysicsData']:
                if phys_file_path not in self.available_resources:
                    continue
                phys_file_path = self.available_resources[phys_file_path]
                phys_file = ContentManager().find_file(phys_file_path)
                if not phys_file:
                    continue
                vphys = ValveCompiledPhysicsLoader(phys_file, self.scale)
                vphys.parse_meshes()
                phys_meshes = vphys.build_mesh()
                self.container.physics_objects.extend(phys_meshes)
        elif 'embedded_physics' in cdata.data and cdata.data['embedded_physics']:
            block_id = cdata.data['embedded_physics']['phys_data_block']
            data = self.get_data_block(block_id=block_id)
            vphys = ValveCompiledPhysicsLoader(None, self.scale)
            vphys.data_block = data
            vphys.parse_meshes()
            phys_meshes = vphys.build_mesh()
            self.container.physics_objects.extend(phys_meshes)
