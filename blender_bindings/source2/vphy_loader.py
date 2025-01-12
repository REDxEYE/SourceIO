import bpy
import numpy as np
from mathutils import Matrix, Vector

from SourceIO.library.source2.blocks.phys_block import PhysBlock
from SourceIO.library.source2.keyvalues3.binary_keyvalues import BinaryBlob, TypedArray
from SourceIO.library.source2.utils.entity_keyvalues_keys import EntityKeyValuesKeys
from SourceIO.library.utils.math_utilities import SOURCE2_HAMMER_UNIT_TO_METERS


def generate_capsule_mesh(p1, p2, radius, segments=16):
    direction = np.array(p2, dtype=float) - np.array(p1, dtype=float)
    length = np.linalg.norm(direction)
    direction /= length
    rot = np.eye(3)
    if abs(direction[1]) < 0.9999:
        axis = np.cross([0, 1, 0], direction)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(np.dot([0, 1, 0], direction))
        s, c = np.sin(angle), np.cos(angle)
        rot = np.array([
            [c + axis[0] ** 2 * (1 - c), axis[0] * axis[1] * (1 - c) - axis[2] * s,
             axis[0] * axis[2] * (1 - c) + axis[1] * s],
            [axis[1] * axis[0] * (1 - c) + axis[2] * s, c + axis[1] ** 2 * (1 - c),
             axis[1] * axis[2] * (1 - c) - axis[0] * s],
            [axis[2] * axis[0] * (1 - c) - axis[1] * s, axis[2] * axis[1] * (1 - c) + axis[0] * s,
             c + axis[2] ** 2 * (1 - c)]
        ])

    total_vertices = segments * 2 + ((segments - 2) * segments + 1) * 2
    vertices = np.zeros((total_vertices, 3), np.float32)
    offset = 0
    for y in [0, length]:
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x, z = radius * np.cos(angle), radius * np.sin(angle)
            vertices[offset] = [x, y, z]
            offset += 1

    for i in range(1, segments - 1):
        angle_v = 0.5 * np.pi * i / (segments - 1)
        y_offset = radius * np.cos(angle_v)
        r = radius * np.sin(angle_v)
        for j in range(segments):
            angle_h = 2 * np.pi * j / segments
            x, z = r * np.cos(angle_h), r * np.sin(angle_h)
            vertices[offset] = [x, -y_offset, z]
            offset += 1
    vertices[offset] = [0, -radius, 0]
    offset += 1

    for i in range(1, segments - 1):  # Start from 1 to exclude the base ring
        angle_v = 0.5 * np.pi * i / (segments - 1)
        y_offset = radius * np.cos(angle_v)
        r = radius * np.sin(angle_v)
        for j in range(segments):
            angle_h = 2 * np.pi * j / segments
            x, z = r * np.cos(angle_h), r * np.sin(angle_h)
            vertices[offset] = [x, length + y_offset, z]
            offset += 1

    vertices[offset] = [0, length + radius, 0]
    offset += 1
    # Generate indices for each part
    indices1 = np.zeros((segments * 2, 3), dtype=int)
    for i1 in range(segments):
        j1 = (i1 + 1) % segments
        indices1[2 * i1] = [i1, j1, i1 + segments]
        indices1[2 * i1 + 1] = [i1 + segments, j1, j1 + segments]

    indices2 = np.zeros((segments * 2, 3), np.uint32)
    cylinder_top_indices = np.arange(segments)
    offset1 = (segments - 1) * segments
    hemisphere_bottom_indices = np.arange(offset1, offset1 + segments)
    for j2 in range(segments):
        k = (j2 + 1) % segments
        idx1, idx2, idx3 = cylinder_top_indices[j2], cylinder_top_indices[k], hemisphere_bottom_indices[j2]
        indices2[j2 * 2] = [idx1, idx2, idx3]
        idx4 = hemisphere_bottom_indices[k]
        indices2[j2 * 2 + 1] = [idx2, idx4, idx3]
    top_join_indices = np.array(indices2, dtype=np.uint32)

    indices3 = np.zeros((segments * 2, 3), np.uint32)
    cylinder_top_indices = np.arange(segments) + segments
    offset1 = (segments - 1) * segments + ((segments - 2) * segments + 1)
    hemisphere_bottom_indices = np.arange(offset1, offset1 + segments)
    for j2 in range(segments):
        k = (j2 + 1) % segments
        idx1, idx2, idx3 = cylinder_top_indices[j2], cylinder_top_indices[k], hemisphere_bottom_indices[j2]
        indices3[j2 * 2] = [idx1, idx2, idx3]
        idx4 = hemisphere_bottom_indices[k]
        indices3[j2 * 2 + 1] = [idx2, idx4, idx3]
    bot_join_indices = np.array(indices3, dtype=np.uint32)

    indices4 = []
    for i2 in range(0, segments - 3):
        for j3 in range(segments):
            k1 = (j3 + 1) % segments
            base_idx = i2 * segments
            idx_1, idx_, idx_3, idx_2 = base_idx + j3, base_idx + k1, base_idx + segments + j3, base_idx + segments + k1

            # Triangles
            indices4.extend([[idx_1, idx_3, idx_], [idx_3, idx_2, idx_]])
    top_vertex = (segments - 2) * segments
    for j3 in range(segments):
        k1 = (j3 + 1) % segments
        indices4.append([j3, top_vertex, j3 + 1 if k1 != 0 else j3 + 1 - segments])
    hemisphere_ind = np.array(indices4, dtype=np.uint32)

    max_idx = np.max(indices1)
    top_hemisphere_ind = hemisphere_ind + max_idx + 1
    max_idx = np.max(top_hemisphere_ind)
    bot_hemisphere_ind = hemisphere_ind + max_idx + 1

    indices = np.concatenate([indices1, top_hemisphere_ind, bot_hemisphere_ind, top_join_indices, bot_join_indices])
    vertices = np.dot(rot, vertices.T).T + p1
    return vertices, indices


def generate_sphere_mesh(radius, segments):
    vertices = []
    indices = []

    # Generate vertices
    for i in range(segments):
        angle_v = np.pi * i / (segments - 1)
        y = radius * np.cos(angle_v)
        r = radius * np.sin(angle_v)
        for j in range(segments):
            angle_h = 2 * np.pi * j / segments
            x, z = r * np.cos(angle_h), r * np.sin(angle_h)
            vertices.append([x, y, z])

    # Generate indices
    for i in range(segments - 1):
        for j in range(segments):
            k = (j + 1) % segments
            base_idx = i * segments
            idx1, idx2, idx3, idx4 = base_idx + j, base_idx + k, base_idx + segments + j, base_idx + segments + k

            # Triangles
            indices.extend([[idx1, idx3, idx2], [idx3, idx4, idx2]])

    return np.array(vertices), np.array(indices, dtype=np.uint32)


segments = 12


def load_physics(phys_block: PhysBlock, scale: float = SOURCE2_HAMMER_UNIT_TO_METERS):
    parts = phys_block["m_parts"]
    indices = phys_block["m_boneParents"]
    names = phys_block["m_boneNames"]
    matrices = phys_block["m_bindPose"]
    shapes = []

    if len(indices) != 0 and names and len(matrices) != 0:
        for parent, part, matrix in zip(indices, parts, matrices):
            bone_matrix = Matrix(matrix.reshape((3, 4))).to_4x4()

            shape = part["m_rnShape"]

            spheres = shape["m_spheres"]
            capsules = shape["m_capsules"]
            hulls = shape["m_hulls"]
            meshes = shape["m_meshes"]

            shape_name = names[parent]
            collision_attributes = phys_block.get("m_collisionAttributes", None)
            keys = EntityKeyValuesKeys()
            surface_properties = [keys.get(hsh) for hsh in phys_block.get("m_surfacePropertyHashes", [])] or None
            shapes.extend(generate_physics_shapes(shape_name, bone_matrix, scale, capsules, spheres, hulls, meshes,
                                                  collision_attributes, surface_properties))
    else:
        for part in parts:
            shape = part["m_rnShape"]

            spheres = shape["m_spheres"]
            capsules = shape["m_capsules"]
            hulls = shape["m_hulls"]
            meshes = shape["m_meshes"]
            shape_name = "physics_mesh"
            bone_matrix = Matrix.Identity(4)
            collision_attributes = phys_block["m_collisionAttributes"]
            keys = EntityKeyValuesKeys()
            surface_properties = [keys.get(hsh) for hsh in phys_block["m_surfacePropertyHashes"]]
            shapes.extend(generate_physics_shapes(shape_name, bone_matrix, scale, capsules, spheres, hulls, meshes,
                                                  collision_attributes, surface_properties))
    return shapes


def generate_physics_shapes(shape_name, bone_matrix, scale, capsules, spheres, hulls, meshes,
                            collision_attributes: list[dict] | None, surface_properties: list[str] | None):
    shapes = []
    for capsule_info in capsules:
        collision_attribute_index = capsule_info["m_nCollisionAttributeIndex"]
        surface_property_index = capsule_info["m_nSurfacePropertyIndex"]
        if "m_UserFriendlyName" in capsule_info:
            shape_name = capsule_info["m_UserFriendlyName"]
        capsule = capsule_info["m_Capsule"]
        capsule_start, capsule_end = capsule["m_vCenter"]
        radius = capsule["m_flRadius"]

        capsule_start = Vector(capsule_start)
        capsule_end = Vector(capsule_end)
        capsule_start = (bone_matrix @ capsule_start) * scale
        capsule_end = (bone_matrix @ capsule_end) * scale

        mesh_data = bpy.data.meshes.new(name=f'{shape_name}_mesh')
        mesh_obj = bpy.data.objects.new(name=shape_name, object_data=mesh_data)
        vertices, indices = generate_capsule_mesh(capsule_start, capsule_end,
                                                  radius * scale, segments)

        mesh_data.from_pydata(vertices, [], indices)
        # if collision_attributes and surface_properties:
        #     mesh_obj["entity_data"] = {"entity": {"collision_group": collision_attributes[collision_attribute_index],
        #                                           "surface_prop": surface_properties[surface_property_index]}}
        mesh_data.update()
        mesh_data.validate()

        shapes.append(mesh_obj)
    for sphere_info in spheres:
        collision_attribute_index = sphere_info["m_nCollisionAttributeIndex"]
        surface_property_index = sphere_info["m_nSurfacePropertyIndex"]
        if "m_UserFriendlyName" in sphere_info:
            shape_name = sphere_info["m_UserFriendlyName"]
        sphere = sphere_info["m_Sphere"]
        radius = sphere["m_flRadius"]
        center = Vector(sphere["m_vCenter"])
        center = (bone_matrix @ center) * scale

        mesh_data = bpy.data.meshes.new(name=f'{shape_name}_mesh')
        mesh_obj = bpy.data.objects.new(name=shape_name, object_data=mesh_data)
        sphere_vertices, sphere_indices = generate_sphere_mesh(radius * scale, segments)
        sphere_vertices += center
        mesh_data.from_pydata(sphere_vertices, [], sphere_indices)
        mesh_data.validate()
        # if collision_attributes and surface_properties:
        #     mesh_obj["entity_data"] = {"entity": {"collision_group": collision_attributes[collision_attribute_index],
        #                                           "surface_prop": surface_properties[surface_property_index]}}
        mesh_data.update()

        shapes.append(mesh_obj)
    for mesh_info in meshes:
        collision_attribute_index = mesh_info["m_nCollisionAttributeIndex"]
        surface_property_index = mesh_info["m_nSurfacePropertyIndex"]
        if "m_UserFriendlyName" in mesh_info:
            shape_name = mesh_info["m_UserFriendlyName"]
        mesh = mesh_info["m_Mesh"]
        vertex_data = mesh["m_Vertices"]
        indices_data = mesh["m_Triangles"]
        if isinstance(vertex_data, BinaryBlob):
            vertices = np.frombuffer(vertex_data, np.float32) * scale
        else:
            vertices = np.asarray(vertex_data, np.float32) * scale
        if isinstance(indices_data, BinaryBlob):
            indices = np.frombuffer(indices_data, np.uint32)
        elif isinstance(indices_data, TypedArray):
            indices = np.asarray([a["m_nIndex"] for a in indices_data], np.uint32)
        else:
            indices = np.asarray(indices_data, np.uint32)

        mesh_data = bpy.data.meshes.new(name=f'{shape_name}_mesh')
        mesh_obj = bpy.data.objects.new(name=shape_name, object_data=mesh_data)
        mesh_data.from_pydata(vertices.reshape((-1, 3)), [], indices.reshape((-1, 3)))
        mesh_data.validate()
        # if collision_attributes and surface_properties:
        #     mesh_obj["entity_data"] = {"entity": {"collision_group": collision_attributes[collision_attribute_index],
        #                                           "surface_prop": surface_properties[surface_property_index]}}
        mesh_data.update()

        shapes.append(mesh_obj)
    for hull_info in hulls:
        if "m_UserFriendlyName" in hull_info:
            shape_name = hull_info["m_UserFriendlyName"]
        hull = hull_info["m_Hull"]
        if "m_VertexPositions" in hull:
            vertex_data = hull["m_VertexPositions"]
            if isinstance(vertex_data, BinaryBlob):
                vertices = np.frombuffer(vertex_data, np.float32) * scale
            else:
                vertices = np.asarray(vertex_data, np.float32) * scale
        else:
            vertex_data = hull["m_Vertices"]
            if isinstance(vertex_data, BinaryBlob):
                vertices = np.frombuffer(vertex_data, np.float32) * scale
            else:
                vertices = np.asarray(vertex_data, np.float32) * scale
        edge_data = hull["m_Edges"]
        face_data = hull["m_Faces"]

        edge_dtype = np.dtype([
            ("next", np.uint8),
            ("twin", np.uint8),
            ("origin", np.uint8),
            ("face", np.uint8),
        ])
        if isinstance(edge_data, BinaryBlob):
            edges = np.frombuffer(edge_data, edge_dtype)
        else:
            edges = np.zeros(len(edge_data), edge_dtype)
            for i, edge in enumerate(edge_data):
                edges[i] = (edge["m_nNext"]), (edge["m_nTwin"]), (edge["m_nOrigin"]), (edge["m_nFace"])

        if isinstance(face_data, BinaryBlob):
            faces = np.frombuffer(face_data, np.uint8)
        else:
            faces = np.zeros(len(edge_data), np.uint8)
            for i, face in enumerate(face_data):
                faces[i] = face["m_nEdge"]

        indices = []
        for face in faces:
            start_edge = face
            edge = edges[start_edge]["next"]
            while start_edge != edge:
                next_edge = edges[edge]["next"]
                if next_edge == start_edge:
                    break
                indices.append((
                    edges[start_edge]["origin"],
                    edges[edge]["origin"],
                    edges[next_edge]["origin"],
                ))
                edge = next_edge

        mesh_data = bpy.data.meshes.new(name=f'{shape_name}_mesh')
        mesh_obj = bpy.data.objects.new(name=shape_name, object_data=mesh_data)
        mesh_data.from_pydata(vertices.reshape((-1, 3)), [], indices)
        mesh_data.validate()
        collision_attribute_index = hull_info["m_nCollisionAttributeIndex"]
        surface_property_index = hull_info["m_nSurfacePropertyIndex"]
        # if collision_attributes and surface_properties:
        #     mesh_obj["entity_data"] = {"entity": {"flag": hull["m_nFlags"] & 0xFFFFFFFF,
        #                                           "flag_hi": (hull["m_nFlags"] >> 32) & 0xFFFFFFFF,
        #                                           "collision_group": collision_attributes[collision_attribute_index],
        #                                           "surface_prop": surface_properties[surface_property_index]}}
        mesh_data.update()

        shapes.append(mesh_obj)
    return shapes
