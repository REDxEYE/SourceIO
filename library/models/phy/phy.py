from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.library.shared.types import Vector3, Vector4
from SourceIO.library.utils import Buffer, FileBuffer


@dataclass(slots=True)
class Header:
    size: int
    id: int
    solid_count: int
    checksum: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        size = buffer.read_uint32()
        ident = buffer.read_uint32()
        solid_count = buffer.read_uint32()
        checksum = buffer.read_uint32()
        return cls(size, ident, solid_count, checksum)


@dataclass(slots=True)
class CompactEdge:
    start_point: int
    opposite_index: int
    is_virtual: bool

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        bit_field = buffer.read_uint32()
        start_point_index = (bit_field & 0xFFFF)
        opposite_index = ((bit_field >> 16) & 0x7FFF)
        if opposite_index & 0x4000:
            opposite_index = opposite_index - 0x8000
        else:
            opposite_index = opposite_index
        is_virtual = (bit_field >> 31) != 0
        return cls(start_point_index, opposite_index, is_virtual)


@dataclass(slots=True)
class CompactTriangle:
    edges: list[CompactEdge]
    triangle_index: int
    pierce_index: int
    material_index: int
    is_virtual: bool

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        bit_field = buffer.read_uint32()

        edges = [CompactEdge.from_buffer(buffer), CompactEdge.from_buffer(buffer), CompactEdge.from_buffer(buffer)]

        triangle_index = (bit_field & 0xFFF)
        pierce_index = ((bit_field >> 12) & 0xFFF)
        material_index = ((bit_field >> 24) & 0x7F)
        is_virtual = (bit_field >> 31) != 0

        return cls(edges, triangle_index, pierce_index, material_index, is_virtual)

    @property
    def vertex_ids(self):
        return [e.start_point for e in self.edges]


@dataclass(slots=True)
class ConvexLeaf:
    is_root: bool
    _entry: int = 0
    vertex_offset: int = 0
    bone_id: int = 0
    flags: int = 0
    triangle_count: int = 0
    unused: int = 0
    compact_triangles: list[CompactTriangle] = field(default_factory=list)
    triangles: list = field(default_factory=list)
    unique_vertices: set = field(default_factory=set)

    @classmethod
    def from_buffer(cls, buffer: Buffer, is_root: bool):
        entry = buffer.tell()
        vertex_offset, bone_id, flags, triangle_count, unused = buffer.read_fmt(
            '3i2h')
        triangles = []
        unique_vertices = set()
        compact_triangles = []
        for _ in range(triangle_count):
            tri = CompactTriangle.from_buffer(buffer)
            triangles.append(tri.vertex_ids)
            unique_vertices.update(tri.vertex_ids)
            compact_triangles.append(tri)
        return cls(is_root, entry, vertex_offset, bone_id, flags, triangle_count,
                   unused, compact_triangles, triangles, unique_vertices)

    @property
    def has_children(self):
        return (self.flags >> 0) & 3

    @property
    def is_compact(self):
        return (self.flags >> 2) & 3

    @property
    def padding(self):
        return (self.flags >> 4) & 0xF

    @property
    def size_div_16(self):
        return (self.flags >> 8) & 0xFFFFFF

    @property
    def vertex_data_offset(self):
        return self._entry + self.vertex_offset


@dataclass(slots=True)
class CompactLedgetreeNode:
    center: Vector3[float]
    radius: float
    bbox_size: Vector3[int]
    free: int
    left_node: Optional['CompactLedgetreeNode'] = field(default=None)
    right_node: Optional['CompactLedgetreeNode'] = field(default=None)
    convex_leaf: ConvexLeaf | None = field(default=None)

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        entry_offset = buffer.tell()
        right_node_offset, compact_ledge_offset, *center, radius = buffer.read_fmt('2i4f')
        bbox_size = Vector3(*buffer.read_fmt('3B'))
        free = buffer.read_uint8()
        is_leaf = right_node_offset == 0
        convex_leaf: ConvexLeaf | None = None
        with buffer.save_current_offset():
            if compact_ledge_offset:
                with buffer.save_current_offset():
                    buffer.seek(entry_offset + compact_ledge_offset)
                    convex_leaf = ConvexLeaf.from_buffer(buffer, not is_leaf)
                if is_leaf:
                    return cls(center, radius, bbox_size, free, None, None, convex_leaf)
            left_node = CompactLedgetreeNode.from_buffer(buffer)
            with buffer.save_current_offset():
                buffer.seek(entry_offset + right_node_offset)
                right_node = CompactLedgetreeNode.from_buffer(buffer)
        return cls(center, radius, bbox_size, free, left_node, right_node, convex_leaf)


@dataclass(slots=True)
class CompactSurface:
    mass_center: Vector3[float]
    rotation_inertia: Vector3[float]
    upper_limit_radius: float
    max_factor_surface_deviation: int
    byte_size: int
    ledge_tree_root_offset: int
    root_tree: CompactLedgetreeNode

    unique_vertices: set[int]
    vertices: np.ndarray

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        entry = buffer.tell()
        mass_center = Vector3.from_buffer(buffer)
        rotation_inertia = Vector3.from_buffer(buffer)
        upper_limit_radius = buffer.read_float()
        bit_field = buffer.read_uint32()
        max_factor_surface_deviation = bit_field & 0xFF
        byte_size = bit_field >> 8
        assert byte_size >= 1
        ledge_tree_root_offset = buffer.read_uint32()
        dummies = buffer.read_fmt("2I")
        assert dummies[0] == 0
        assert dummies[1] == 0
        ivps_magic = buffer.read_fourcc()
        assert ivps_magic == 'IVPS'

        with buffer.read_from_offset(entry + ledge_tree_root_offset):
            ledge_tree_root = CompactLedgetreeNode.from_buffer(buffer)

        def traverse(node: CompactLedgetreeNode):
            unique_vertices_ = set()
            if node.convex_leaf is not None:
                unique_vertices_.update(node.convex_leaf.unique_vertices)
            if node.left_node is not None:
                unique_vertices_.update(traverse(node.left_node))
            if node.right_node is not None:
                unique_vertices_.update(traverse(node.right_node))
            return unique_vertices_

        unique_vertices = traverse(ledge_tree_root)
        with buffer.read_from_offset(ledge_tree_root.convex_leaf.vertex_data_offset):
            vertices = np.frombuffer(buffer.read(4 * 4 * len(unique_vertices)), np.float32).copy()
            vertices = vertices.reshape((-1, 4))[:, :3]
            y = vertices[:, 1].copy()
            z = vertices[:, 2].copy()
            vertices[:, 1] = z
            vertices[:, 2] = y

        return cls(mass_center, rotation_inertia, upper_limit_radius, max_factor_surface_deviation, byte_size,
                   ledge_tree_root_offset, ledge_tree_root, unique_vertices, vertices)

    @staticmethod
    def get_vertex_data(buffer: Buffer, convex_leaf: ConvexLeaf, vertex_count):
        with buffer.save_current_offset():
            buffer.seek(convex_leaf.vertex_data_offset)
            vertex_data = np.frombuffer(buffer.read(4 * 4 * vertex_count), np.float32).copy()
            vertex_data = vertex_data.reshape((-1, 4))[:, :3]

            y = vertex_data[:, 1].copy()
            z = vertex_data[:, 2].copy()
            vertex_data[:, 1] = z
            vertex_data[:, 2] = y

        return vertex_data


@dataclass(slots=True)
class CompactSurfaceHeader:
    size: int
    version: int
    type: int
    surface_size: int
    drag_axis_areas: Vector3[float]
    axis_map_size: int
    compact_surface: CompactSurface | None

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        size = buffer.read_uint32()
        ident = buffer.read_fourcc()
        assert ident == 'VPHY'
        version = buffer.read_uint16()
        model_type = buffer.read_uint16()
        surface_size = buffer.read_uint32()
        drag_axis_areas = buffer.read_fmt('3f')
        axis_map_size = buffer.read_uint32()
        return cls(size, version, model_type, surface_size, drag_axis_areas, axis_map_size, None)

    def end(self):
        return self.size + 4


@dataclass(slots=True)
class Phy:
    header: Header
    solids: list[CompactSurfaceHeader]
    kv: str

    @classmethod
    def from_filepath(cls, filepath: TinyPath):
        return cls.from_buffer(FileBuffer(filepath))

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        header = Header.from_buffer(buffer)
        buffer.seek(header.size)
        solids = []
        for _ in range(header.solid_count):
            solid_start = buffer.tell()
            surface_header = CompactSurfaceHeader.from_buffer(buffer)
            compact_surface = CompactSurface.from_buffer(buffer)
            buffer.seek(solid_start + surface_header.size + 4)
            surface_header.compact_surface = compact_surface
            solids.append(surface_header)
        # if solids:
        #     buffer.seek(solid_start + solids[-1].end())
        kv = buffer.read_ascii_string()
        return cls(header, solids, kv)
