import numpy as np

from SourceIO.library.shared.intermediate_data.common import Matrix4x4, Vector3
from SourceIO.library.shared.intermediate_data import Model, Bone, BoneFlags, Mesh, VertexAttributesName, \
    VertexAttribute, VertexAttributeType, build_numpy_vertex_buffer_type
from SourceIO.library.shared.intermediate_data.model import Material
from SourceIO.library.utils.pylib.mesh import SMDModel
from SourceIO.library.utils import Buffer, TinyPath


def rle_pairs(arr):
    arr = np.asarray(arr)

    changes = np.empty(len(arr), dtype=bool)
    changes[0] = True
    changes[1:] = arr[1:] != arr[:-1]

    values = arr[changes]
    counts = np.diff(np.r_[np.flatnonzero(changes), len(arr)])

    return values, counts


def load_smd(path: TinyPath, buffer: Buffer) -> Model:
    smd_model = SMDModel(buffer.read().decode('utf-8'))
    if smd_model.frame_count == 0:
        raise NotImplementedError("SMD with 0 animation frames is not supported yet.")

    frame = smd_model.skeleton.frames[0]
    bones: list[Bone] = []

    for node in smd_model.nodes:
        bone_def = frame[node.id]
        x, y, z = bone_def.pos
        rx, ry, rz = bone_def.rot

        translation = Matrix4x4.from_translation(Vector3(x, y, z))
        rotation = Matrix4x4.from_euler_angles(rx, ry, rz)

        bone = Bone(
            name=node.name,
            parent=smd_model.nodes[node.parent].name if node.parent != -1 else None,
            flags=BoneFlags.NO_BONE_FLAGS,
            transform=translation @ rotation
        )
        bones.append(bone)

    vertex_attributes = {
        VertexAttributesName.POSITION: VertexAttribute(VertexAttributesName.POSITION, VertexAttributeType.FLOAT, 3),
        VertexAttributesName.NORMAL: VertexAttribute(VertexAttributesName.NORMAL, VertexAttributeType.FLOAT, 3),
        VertexAttributesName.UV0: VertexAttribute(VertexAttributesName.UV0, VertexAttributeType.FLOAT, 2),
        VertexAttributesName.BONE_IND0: VertexAttribute(VertexAttributesName.BONE_IND0, VertexAttributeType.INT, 4),
        VertexAttributesName.BONE_WEIGHTS0: VertexAttribute(VertexAttributesName.BONE_WEIGHTS0,
                                                            VertexAttributeType.FLOAT, 4),
    }
    vertex_attributes_tmp = {
        VertexAttributesName.POSITION: VertexAttribute(VertexAttributesName.POSITION, VertexAttributeType.FLOAT, 3),
        VertexAttributesName.NORMAL: VertexAttribute(VertexAttributesName.NORMAL, VertexAttributeType.FLOAT, 3),
        VertexAttributesName.BONE_IND0: VertexAttribute(VertexAttributesName.BONE_IND0, VertexAttributeType.INT, 4),
        VertexAttributesName.BONE_WEIGHTS0: VertexAttribute(VertexAttributesName.BONE_WEIGHTS0,
                                                            VertexAttributeType.FLOAT, 4),
    }

    Vertex = build_numpy_vertex_buffer_type(vertex_attributes)
    VertexTmp = build_numpy_vertex_buffer_type(vertex_attributes_tmp)

    vertices = np.empty(smd_model.triangle_count * 3, dtype=Vertex)
    materials = {}
    position_attr = VertexAttributesName.POSITION.name
    normal_attr = VertexAttributesName.NORMAL.name
    uv0_attr = VertexAttributesName.UV0.name
    weight_attr = VertexAttributesName.BONE_WEIGHTS0.name
    bone_index_attr = VertexAttributesName.BONE_IND0.name
    polygon_material_indices = np.zeros((smd_model.triangle_count,), dtype=np.uint32)
    for t, triangle in enumerate(smd_model.triangles):
        if triangle.material not in materials:
            materials[triangle.material] = len(materials)
        material_id = materials[triangle.material]
        polygon_material_indices[t] = material_id
        for i in range(0, 3):
            vertices[t * 3 + i][position_attr] = triangle.vertices[i].pos
            vertices[t * 3 + i][normal_attr] = triangle.vertices[i].normal
            vertices[t * 3 + i][uv0_attr] = triangle.vertices[i].uv
            influences = triangle.vertices[i].weights
            bone_ids = [weight[0] for weight in influences]
            weights = [weight[1] for weight in influences]
            w_count = len(weights)
            vertices[t * 3 + i][bone_index_attr][:w_count] = bone_ids
            vertices[t * 3 + i][weight_attr][:w_count] = weights

    material_ranges = []

    unique_vertices_wuv, indices = np.unique(vertices, return_inverse=True)
    original_indices = indices.reshape(-1, 3)

    vertices_tmp = np.empty(len(unique_vertices_wuv), dtype=VertexTmp)
    for attr in vertex_attributes_tmp.values():
        attr_name = attr.usage.name
        vertices_tmp[attr_name] = unique_vertices_wuv[attr_name]

    unique_vertices, index, new_indices = np.unique(vertices_tmp, return_index=True, return_inverse=True)

    core_uv = unique_vertices_wuv[uv0_attr][new_indices.reshape(-1, 3)]
    core_faces = new_indices[original_indices]

    # core_faces = original_indices.reshape(-1, 3)

    # remap = np.zeros(max(index) + 2, dtype=np.uint32)
    # remap[index] = np.arange(len(index), dtype=np.uint32)
    # core_faces = remap[core_faces]

    for mat_id, count in zip(*rle_pairs(polygon_material_indices)):
        material_ranges.append((int(mat_id), int(count)))

    mesh = Mesh(path.stem, unique_vertices, {VertexAttributesName.UV0: core_uv}, core_faces, None,
                vertex_attributes_tmp, material_ranges)

    mat_list = [Material(name, name) for name in materials.keys()]
    model = Model.without_bodygroups(path.stem, [mesh], bones, [], mat_list, Matrix4x4())

    return model
