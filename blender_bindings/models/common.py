import bpy
import numpy as np

from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.blender_bindings.utils.bpy_utils import get_new_unique_collection
from SourceIO.library.models.mdl.structs.model import Model
from SourceIO.library.models.vtx.v7.structs.lod import ModelLod as VtxModel
from SourceIO.library.models.vtx.v7.structs.mesh import Mesh as VtxMesh
from SourceIO.library.models.mdl import Mdl


def merge_strip_groups(vtx_mesh: VtxMesh):
    indices_accumulator = []
    vertex_accumulator = []
    vertex_offset = 0
    for strip_group in vtx_mesh.strip_groups:
        indices_accumulator.append(np.add(strip_group.indices, vertex_offset))
        vertex_accumulator.append(strip_group.vertexes['original_mesh_vertex_index'].reshape(-1))
        vertex_offset += sum(strip.vertex_count for strip in strip_group.strips)
    return np.hstack(indices_accumulator), np.hstack(vertex_accumulator), vertex_offset


def merge_meshes(model: Model, vtx_model: VtxModel):
    vtx_vertices = []
    acc = 0
    mat_arrays = []
    indices_array = []
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


def put_into_collections(model_container: ModelContainer, model_name,
                         parent_collection=None, bodygroup_grouping=False):
    master_collection = get_new_unique_collection(model_name, parent_collection or bpy.context.scene.collection)
    if model_container.bodygroups:
        for bodygroup_name, meshes in model_container.bodygroups.items():
            if bodygroup_grouping:
                body_part_collection = get_new_unique_collection(bodygroup_name, master_collection)
            else:
                body_part_collection = master_collection

            for mesh in meshes:
                body_collection = get_new_unique_collection(mesh.name, body_part_collection)
                body_collection.objects.link(mesh)
    else:
        for obj in model_container.objects:
            master_collection.objects.link(obj)
    if model_container.armature:
        master_collection.objects.link(model_container.armature)

    if model_container.attachments:
        attachments_collection = get_new_unique_collection(model_name + '_ATTACHMENTS', master_collection)
        for attachment in model_container.attachments:
            attachments_collection.objects.link(attachment)
    if model_container.physics_objects:
        physics_collection = get_new_unique_collection(model_name + '_PHYSICS', master_collection)
        for physics in model_container.physics_objects:
            physics_collection.objects.link(physics)
    model_container.master_collection = master_collection
    return master_collection

def create_eyeballs(mdl: Mdl, armature: bpy.types.Object, mesh_obj: bpy.types.Object, model: Model, scale: float, extra_stuff: list):
    from math import atan, radians
    from mathutils import Matrix, Vector

    eyeballs = model.eyeballs
    for mesh in model.meshes:
        if mesh.material_type != 1:
            continue
        eyeball = eyeballs[mesh.material_param]
        eyeball_name = eyeball.name or f'eye_{mesh.material_param}'
        forward = Vector(eyeball.forward)
        up = Vector(eyeball.up)
        eyeball_obj = bpy.data.objects.new(eyeball_name, None)
        eyeball_obj['forward_debug'] = forward
        eyeball_obj['up_debug'] = up
        eyeball_obj.show_in_front = True
        extra_stuff.append(eyeball_obj)

        eyeball_pos = Vector(eyeball.org) * scale
        eyeball_matrix_rotation = Matrix([up.cross(forward),
                                        forward,
                                        up])
        
        eyeball_obj.location = eyeball_pos
        eyeball_obj.rotation_mode = 'QUATERNION'
        eyeball_obj.scale = [scale]*3
        eyeball_obj.empty_display_type = 'SPHERE'

        con = eyeball_obj.constraints.new('CHILD_OF')
        con.target = armature
        con.subtarget = mdl.bones[eyeball.bone_index].name
        con.inverse_matrix.identity()
        eye_material = mdl.materials[mesh.material_index].bpy_material
        eye_material['target'] = eyeball_obj
        #eyeball_obj['user'] = eye_material

        bone_parent = armature.data.bones[mdl.bones[eyeball.bone_index].name]

        eyeball_obj.rotation_quaternion = bone_parent.matrix_local.to_quaternion().inverted() @ eyeball_matrix_rotation.to_quaternion() @ Matrix.Rotation(radians(-90), 3, 'Y').to_quaternion()

        locs, rots, scales = ['LOC_X', 'LOC_Y', 'LOC_Z'], ['ROT_W', 'ROT_X', 'ROT_Y', 'ROT_Z'], ['SCALE_X', 'SCALE_Y', 'SCALE_Z']

        prop_loc = eyeball_name + '_loc'
        prop_rot = eyeball_name + '_rot'
        prop_scale = eyeball_name + '_scale'
        mesh_obj[prop_loc] = [0.0]*3
        mesh_obj[prop_rot] = [0.0]*4 # quaternion
        mesh_obj[prop_scale] = [0.0]*3
        drivers_loc = mesh_obj.driver_add(f'["{prop_loc}"]')
        drivers_rot = mesh_obj.driver_add(f'["{prop_rot}"]')
        drivers_scale = mesh_obj.driver_add(f'["{prop_scale}"]')

        def get_obj_transforms_driver(drivers, transform_type, do_quaternion=False):
            for driver, transform_type in zip(drivers, transform_type):
                driver = driver.driver
                driver.type = 'AVERAGE'
                var = driver.variables.new()
                var.type = 'TRANSFORMS'
                var.targets[0].id = eyeball_obj
                if do_quaternion:
                    var.targets[0].rotation_mode = 'QUATERNION'
                var.targets[0].transform_type = transform_type
                
        get_obj_transforms_driver(drivers_loc, locs)
        get_obj_transforms_driver(drivers_rot, rots, True)
        get_obj_transforms_driver(drivers_scale, scales)

        mesh_obj[eyeball_name+'_iris_scale'] = 1/eyeball.iris_scale
        eyeball_obj.empty_display_size = 1/eyeball.iris_scale
        mesh_obj[eyeball_name+'_z_offset'] = atan(eyeball.z_offset)

        if (nodes := getattr(eye_material.node_tree, 'nodes', None)):
            if nodes.get('!EYE_LOC'):
                nodes['!EYE_LOC'].attribute_name = prop_loc
            if nodes.get('!EYE_ROT'):
                nodes['!EYE_ROT'].attribute_name = prop_rot
            if nodes.get('!EYE_LOC'):
                nodes['!EYE_SCALE'].attribute_name = prop_scale
            if nodes.get('!EYE_Z'):
                nodes['!EYE_Z'].attribute_name = eyeball_name + '_z_offset'
            if nodes.get('!EYE_IRIS_SCALE'):
                nodes['!EYE_IRIS_SCALE'].attribute_name = eyeball_name + '_iris_scale'