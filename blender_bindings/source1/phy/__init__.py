from pathlib import Path

import bpy
import numpy as np

from ....library.models.mdl.v36 import MdlV36
from ....library.models.phy.phy import ConvexLeaf, Phy, TreeNode
from ....library.utils import Buffer
from ....library.utils.math_utilities import vector_transform_v
from ...shared.model_container import ModelContainer
from ....library.utils.path_utilities import path_stem


def _collect_meshes(node: TreeNode, meshes: list[ConvexLeaf]):
    unique_vertices = set()
    if node.convex_leaf is not None:
        meshes.append(node.convex_leaf)
        unique_vertices.update(node.convex_leaf.unique_vertices)
    if node.left_node is not None:
        unique_vertices.update(_collect_meshes(node.left_node, meshes))
    if node.right_node is not None:
        unique_vertices.update(_collect_meshes(node.right_node, meshes))
    return unique_vertices


def import_physics(phy: Phy, phy_buffer: Buffer, mdl: MdlV36, container: ModelContainer, scale: float = 1.0):
    mesh_name = path_stem(mdl.header.name)

    for i, solid in enumerate(phy.solids):
        meshes: list[ConvexLeaf] = []
        vertex_count = len(_collect_meshes(solid.collision_model.root_tree, meshes))

        vertex_data = solid.collision_model.get_vertex_data(phy_buffer,
                                                            solid.collision_model.root_tree.convex_leaf,
                                                            vertex_count)
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
            container.physics_objects.append(mesh_obj)
