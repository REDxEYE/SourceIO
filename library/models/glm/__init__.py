from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import Sequence

import numpy as np

from SourceIO.library.utils import Buffer


@dataclass
class GLMHeader:
    ident: str
    version: int
    name: str
    anim_file_name: str
    bone_count: int
    lod_count: int
    lod_offset: int
    mesh_count: int
    mesh_hierarchy_offset: int
    end_offset: int

    @staticmethod
    def from_buffer(buffer: Buffer) -> GLMHeader:
        ident = buffer.read_fourcc()
        assert ident == "2LGM", f"Invalid GLM2 file ident: {ident}"
        version = buffer.read_uint32()
        assert version == 6, f"Unsupported GLM2 version: {version}"
        name = buffer.read_ascii_string(64)
        anim_file_name = buffer.read_ascii_string(64)
        anim_id, bone_count, lod_count, lod_offset, mesh_count, mesh_hierarchy_offset, end_offset = buffer.read_fmt(
            "7I")
        return GLMHeader(ident, version, name, anim_file_name,
                         bone_count, lod_count, lod_offset, mesh_count,
                         mesh_hierarchy_offset, end_offset
                         )


class GLMMeshFlags(IntFlag):
    TAG = 0x1
    OFF = 0x2


@dataclass
class GLMMeshHier:
    name: str
    flags: GLMMeshFlags
    material: str
    material_id: int
    parent_id: int
    children: Sequence[int]

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> GLMMeshHier:
        name = buffer.read_ascii_string(64)
        flags = GLMMeshFlags(buffer.read_uint32())
        material = buffer.read_ascii_string(64)
        material_id = buffer.read_int32()
        parent_id = buffer.read_int32()
        child_count = buffer.read_int32()
        children = buffer.read_fmt(f"{child_count}I")
        return cls(name, flags, material, material_id, parent_id, children)


GLMVertex = np.dtype([
    ("normal", np.float32, (3,)),
    ("position", np.float32, (3,)),
    ("packed_data", np.uint32, (1,)),
    ("weights", np.uint8, (4,)),
])

GLMVertexUnpacked = np.dtype([
    ("normal", np.float32, (3,)),
    ("position", np.float32, (3,)),
    ("bone_indices", np.int32, (4,)),
    ("bone_weights", np.float32, (4,)),
])


@dataclass
class GLMMesh:
    index: int
    vertices: np.ndarray
    uv: np.ndarray
    triangles: np.ndarray
    bones: np.ndarray
    next_offset: int

    @staticmethod
    def _unpack_glm_vertices(packed: np.ndarray) -> np.ndarray:
        """
        Convert an array of GLMVertex (with packed_data and byte weights)
        into GLMVertexUnpacked (bone_indices and bone_weights).
        Weight count = ((packed_data >> 30) & 3) + 1.
        For the first N-1 weights, use ((w8 | (overflow<<8)) / 1023).
        The Nth weight is 1 - sum(first N-1). Unused slots are zeroed.
        Bone indices are 5-bit fields starting at bit 0 in packed_data.
        """
        n = packed.shape[0]
        out = np.empty(n, dtype=GLMVertexUnpacked)

        out["normal"] = packed["normal"]
        out["position"] = packed["position"]

        pd = packed["packed_data"].reshape(-1).astype(np.uint32)
        numw = ((pd >> 30) & 0x3).astype(np.int32) + 1

        w8 = packed["weights"].astype(np.uint16)
        ov = np.empty_like(w8, dtype=np.uint16)
        for i in range(4):
            ov[:, i] = (((pd >> (20 + 2 * i)) & 0x3).astype(np.uint16) << 8)
        rec = (w8 | ov).astype(np.uint16)
        norm = rec.astype(np.float32) / 1023.0

        weights = np.zeros((n, 4), dtype=np.float32)
        mask_n2 = numw >= 2
        mask_n3 = numw >= 3
        mask_n4 = numw >= 4
        weights[:, 0] = np.where(mask_n2, norm[:, 0], 0.0)
        weights[:, 1] = np.where(mask_n3, norm[:, 1], 0.0)
        weights[:, 2] = np.where(mask_n4, norm[:, 2], 0.0)

        m = numw == 1
        weights[m, 0] = 1.0
        m = numw == 2
        weights[m, 1] = 1.0 - weights[m, 0]
        m = numw == 3
        weights[m, 2] = 1.0 - (weights[m, 0] + weights[m, 1])
        m = numw == 4
        weights[m, 3] = 1.0 - (weights[m, 0] + weights[m, 1] + weights[m, 2])

        out["bone_weights"] = weights

        indices = np.empty((n, 4), dtype=np.int32)
        for i in range(4):
            indices[:, i] = ((pd >> (5 * i)) & 0x1F).astype(np.int32)

        used = (np.arange(4)[None, :] < numw[:, None])
        indices[~used] = 0
        out["bone_indices"] = indices

        return out

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> GLMMesh:
        entry = buffer.tell()
        ident, index, header_offset, vertex_count, vertex_offset, triangle_count, triangle_offset, bone_count, bone_offset, next_offset = buffer.read_fmt(
            "10i")

        with buffer.read_from_offset(entry + vertex_offset):
            vertices = np.frombuffer(buffer.read(vertex_count * GLMVertex.itemsize), dtype=GLMVertex)

            uv = np.frombuffer(buffer.read(vertex_count * 2 * 4), dtype=np.float32).reshape((vertex_count, 2))
        with buffer.read_from_offset(entry + triangle_offset):
            triangles = np.frombuffer(buffer.read(triangle_count * 3 * 4), dtype=np.uint32).reshape(
                (triangle_count, 3)).reshape(-1, 3)
            triangles = triangles[:, ::-1]

        with buffer.read_from_offset(entry + bone_offset):
            bones = np.frombuffer(buffer.read(bone_count * 4), dtype=np.int32)

        vertices = cls._unpack_glm_vertices(vertices)
        return cls(index, vertices, uv, triangles, bones, entry + next_offset)


@dataclass
class GLMLod:
    meshes: Sequence[GLMMesh]
    next_offset: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, header: GLMHeader) -> GLMLod:
        entry = buffer.tell()
        next_offset = entry + buffer.read_uint32()
        mesh_offsets = buffer.read_fmt(f"{header.mesh_count}i")
        meshes = []
        for _ in mesh_offsets:
            mesh = GLMMesh.from_buffer(buffer)
            meshes.append(mesh)
            buffer.seek(mesh.next_offset)

        return cls(meshes, next_offset)


class GLMLodList(list[GLMLod]):

    @classmethod
    def from_buffer(cls, buffer: Buffer, header: GLMHeader) -> GLMLodList:
        lods = []
        for _ in range(header.lod_count):
            lod = GLMLod.from_buffer(buffer, header)
            lods.append(lod)
        return cls(lods)


@dataclass
class GLMModel:
    header: GLMHeader
    hier: Sequence[GLMMeshHier]
    lods: GLMLodList

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> GLMModel:
        header = GLMHeader.from_buffer(buffer)
        hier_base = buffer.tell()
        hier_offsets = buffer.read_fmt(f"{header.mesh_count}i")
        hier = []
        for mesh_offset in hier_offsets:
            buffer.seek(hier_base + mesh_offset)
            mesh = GLMMeshHier.from_buffer(buffer)
            hier.append(mesh)

        buffer.seek(header.lod_offset)
        lods = GLMLodList.from_buffer(buffer, header)

        return cls(header, hier, lods)


@dataclass
class GLAHeader:
    ident: str
    version: int
    name: str
    scale: float
    frame_count: int
    frame_offset: int
    bone_count: int
    compressed_bone_pool: int
    skeleton_offset: int
    end_offset: int

    @staticmethod
    def from_buffer(buffer: Buffer) -> GLAHeader:
        ident = buffer.read_fourcc()
        assert ident == "2LGA", f"Invalid GLA file ident: {ident}"
        version = buffer.read_uint32()
        assert version == 6, f"Unsupported GLA version: {version}"
        name = buffer.read_ascii_string(64)
        scale, frame_count, frame_offset, bone_count, composite_bone_pool, skeleton_offset, end_offset = buffer.read_fmt(
            "f6I")
        return GLAHeader(ident, version, name, scale,
                         frame_count, frame_offset, bone_count,
                         composite_bone_pool, skeleton_offset, end_offset
                         )


@dataclass
class GLABone:
    name: str
    flags: int
    parent_id: int
    matrix: np.ndarray
    matrix_inv: np.ndarray
    children: Sequence[int]

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> GLABone:
        name = buffer.read_ascii_string(64)
        flags = buffer.read_uint32()
        parent_id = buffer.read_int32()
        matrix = np.frombuffer(buffer.read(12 * 4), dtype=np.float32).reshape((3, 4))
        matrix_inv = np.frombuffer(buffer.read(12 * 4), dtype=np.float32).reshape((3, 4))
        child_count = buffer.read_int32()
        children = buffer.read_array("i", child_count)
        return cls(name, flags, parent_id, matrix, matrix_inv, children)


@dataclass
class BoneFrame:
    position: np.ndarray
    rotation: np.ndarray

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> BoneFrame:
        rw, rx, ry, rz = buffer.read_fmt("4H")
        tx, ty, tz = buffer.read_fmt("3H")

        quat = (np.array([rw, rx, ry, rz], dtype=np.float32) / 16383) - 2
        position = (np.array([tx, ty, tz], dtype=np.float32) / 64) - 512

        return cls(position, quat)


@dataclass
class GLASkeleton:
    header: GLAHeader
    bones: Sequence[GLABone]
    animation_frame_indices: np.ndarray
    animation_frames: Sequence[BoneFrame]

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> GLASkeleton:
        header = GLAHeader.from_buffer(buffer)
        bones_entry = buffer.tell()
        bone_offsets = buffer.read_array("I", header.bone_count)
        bones = []
        for offset in bone_offsets:
            buffer.seek(bones_entry + offset)
            bone = GLABone.from_buffer(buffer)
            bones.append(bone)

        bone_indices = np.frombuffer(buffer.read(header.bone_count * 3 * header.frame_count), dtype=np.uint8).reshape(
            (-1, 3))

        bone_indices = bone_indices[:, 0] + (bone_indices[:, 1] << 8) + (bone_indices[:, 2] << 16)
        bone_indices = bone_indices.reshape((header.frame_count, header.bone_count))
        buffer.align(4)
        assert header.compressed_bone_pool == buffer.tell(), f"Expected compressed bone pool at {header.compressed_bone_pool}, got {buffer.tell()}"
        compressed_bones = [BoneFrame.from_buffer(buffer) for _ in range(bone_indices.max() + 1)]
        return cls(header, bones, bone_indices, compressed_bones)
