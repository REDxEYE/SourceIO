import bpy
import numpy as np

from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.library.models.mdl.v36 import MdlV36
from SourceIO.library.models.phy.phy import ConvexLeaf, Phy, CompactLedgetreeNode
from SourceIO.library.utils import Buffer
from SourceIO.library.utils.math_utilities import vector_transform_v
from SourceIO.library.utils.path_utilities import path_stem


def _collect_meshes(node: CompactLedgetreeNode, meshes: list[ConvexLeaf]):
    if node.convex_leaf is not None:
        meshes.append(node.convex_leaf)
    if node.left_node is not None:
        _collect_meshes(node.left_node, meshes)
    if node.right_node is not None:
        _collect_meshes(node.right_node, meshes)


def import_physics(phy: Phy, mdl: MdlV36, container: ModelContainer, scale: float = 1.0):
    mesh_name = path_stem(mdl.header.name)

    for i, solid in enumerate(phy.solids):
        meshes: list[ConvexLeaf] = []
        _collect_meshes(solid.compact_surface.root_tree, meshes)

        vertex_data = solid.compact_surface.vertices

        for j, mesh in enumerate(meshes):
            used_vertices_ids, _, new_indices = np.unique(mesh.triangles, return_index=True, return_inverse=True)
            vertices = vertex_data[used_vertices_ids]

            if mesh.has_children:
                continue

            if container.armature:
                matrix = mdl.bones[mesh.bone_id - 1].pose_to_bone.copy()
                matrix.T[:, 3] *= scale

                vertices = (vertices * 1 / 0.0254) * scale

                vertices = vector_transform_v(vertices, matrix)
            else:
                vertices = (vertices * 1 / 0.0254) * scale

            mesh_data = bpy.data.meshes.new(f'{mesh_name}_solid_{i}{j}_MESH')
            mesh_obj = bpy.data.objects.new(f'{mesh_name}_solid_{i}{j}', mesh_data)

            mesh_data.from_pydata(vertices.tolist(), [], new_indices.reshape((-1, 3)))
            mesh_data.update()
            if container.armature:
                bone = mdl.bones[mesh.bone_id - 1]
                weight_group = mesh_obj.vertex_groups.new(name=bone.name)
                for n in range(len(vertices)):
                    weight_group.add([n], 1, 'REPLACE')

                modifier = mesh_obj.modifiers.new(
                    type="ARMATURE", name="Armature")
                modifier.object = container.armature
                mesh_obj.parent = container.armature
            mesh_data.validate()
            container.physics_objects.append(mesh_obj)
