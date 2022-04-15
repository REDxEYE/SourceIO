from typing import List, Optional

import numpy as np

from ...utils.byte_io_mdl import ByteIO
from ...shared.base import Base


class Header(Base):
    def __init__(self):
        self.size = 0
        self.id = 0
        self.solid_count = 0
        self.checksum = 0

    def read(self, reader: ByteIO):
        reader.begin_region('Header')
        self.size = reader.read_uint32()
        self.id = reader.read_uint32()
        self.solid_count = reader.read_uint32()
        self.checksum = reader.read_uint32()
        reader.end_region()


class TreeNode:
    def __init__(self):
        self._entry = 0
        self.right_node_offset = 0
        self.convex_offset = 0
        self.center = []
        self.radius = 0.0
        self.bbox_size = []

        self.left_node: Optional[TreeNode] = None
        self.right_node: Optional[TreeNode] = None
        self.convex_leaf: Optional[ConvexLeaf] = None

    @property
    def is_leaf(self):
        return self.right_node_offset == 0

    def read(self, reader: ByteIO):
        self._entry = reader.tell()
        reader.begin_region('TreeNode')
        self.right_node_offset, self.convex_offset, *self.center, self.radius = reader.read_fmt('2i4f')
        self.bbox_size = reader.read_fmt('4B')
        reader.end_region()
        with reader.save_current_pos():
            if self.convex_offset:
                with reader.save_current_pos():
                    reader.seek(self._entry + self.convex_offset)
                    self.convex_leaf = ConvexLeaf(not self.is_leaf)
                    self.convex_leaf.read(reader)
                if self.is_leaf:
                    return
            self.left_node = TreeNode()
            self.left_node.read(reader)
            with reader.save_current_pos():
                reader.seek(self._entry + self.right_node_offset)
                self.right_node = TreeNode()
                self.right_node.read(reader)


class ConvexTriangle:
    def __init__(self):
        self.pad = 0
        self.edges = []

    def read(self, reader: ByteIO):
        self.pad, *self.edges = reader.read_fmt('i6h')
        pass

    def get_vertex_id(self, index):
        return self.edges[index * 2]

    @property
    def vertex_ids(self):
        return self.edges[::2]


class ConvexLeaf:
    def __init__(self, root_collision):
        self.is_root = root_collision
        self._entry = 0
        self.vertex_offset = 0
        self.bone_id = 0
        self.flags = 0
        self.triangle_count = 0
        self.unused = 0

        self.triangles = []
        self.unique_vertices = set()

    def child_node(self, reader: ByteIO):
        if self.has_children:
            with reader.save_current_pos():
                reader.seek(self._entry + self.bone_id)
                child = TreeNode()
                child.read(reader)
            return child

        # struct {

    #       uint	has_chilren_flag:2;
    #       int		is_compact_flag:2;  // if false than compact ledge uses points outside this piece of memory
    #       uint	dummy:4;
    #       uint	size_div_16:24;
    #   };
    @property
    def has_children(self):
        return (self.flags >> 0) & 3

    @property
    def is_compact(self):
        return (self.flags >> 2) & 3

    @property
    def dummy(self):
        return (self.flags >> 4) & 15

    @property
    def size_div_16(self):
        return (self.flags >> 8) & 0xFF_FF_FF_FF

    @property
    def vertex_data_offset(self):
        return self._entry + self.vertex_offset

    def read(self, reader: ByteIO):
        self._entry = reader.tell()
        reader.begin_region('ConvexLeaf')
        self.vertex_offset, self.bone_id, self.flags, self.triangle_count, self.unused = reader.read_fmt('3i2h')
        triangles = []
        for _ in range(self.triangle_count):
            tri = ConvexTriangle()
            tri.read(reader)
            triangles.append(tri.vertex_ids)
            self.unique_vertices.update(tri.vertex_ids)
        self.triangles = triangles
        reader.end_region()


class CollisionModel:
    def __init__(self):
        self._entry = 0
        self.values = []
        self.surface = 0
        self.offset_tree = 0
        self.pad = []
        self.root_tree = TreeNode()

    def read(self, reader: ByteIO):
        self._entry = reader.tell()
        reader.begin_region('CollisionModel')
        self.values = reader.read_fmt('7f')
        self.surface, self.offset_tree, *self.pad = reader.read_fmt('4I')
        ivps_magic = reader.read_fourcc()
        assert ivps_magic == 'IVPS'
        reader.end_region()
        with reader.save_current_pos():
            reader.seek(self._entry + self.offset_tree)
            self.root_tree.read(reader)

    @staticmethod
    def get_vertex_data(reader: ByteIO, convex_leaf: ConvexLeaf, vertex_count):
        with reader.save_current_pos():
            reader.seek(convex_leaf.vertex_data_offset)
            reader.begin_region('VertexData')
            vertex_data = np.frombuffer(reader.read(4 * 4 * vertex_count), np.float32).copy()
            vertex_data = vertex_data.reshape((-1, 4))[:, :3]

            y = vertex_data[:, 1].copy()
            z = vertex_data[:, 2].copy()
            vertex_data[:, 1] = z
            vertex_data[:, 2] = y

            reader.end_region()
        return vertex_data


class SolidHeader:

    def __init__(self):
        self._entry = 0
        self.solid_size = 0
        self.id = ''
        self.version = 0
        self.type = 0
        self.size = 0
        self.areas = []
        self.axis_map_size = 0
        self.collision_model = CollisionModel()

    def read(self, reader: ByteIO):
        self._entry = reader.tell()
        reader.begin_region('SolidHeader')
        self.solid_size = reader.read_uint32()
        self.id = reader.read_fourcc()
        assert self.id == 'VPHY'
        self.version = reader.read_uint16()
        self.type = reader.read_uint16()
        self.size = reader.read_uint32()
        self.areas = reader.read_fmt('3f')
        self.axis_map_size = reader.read_uint32()
        reader.end_region()
        self.collision_model.read(reader)

    def next_solid(self, reader: ByteIO):
        with reader.save_current_pos():
            reader.seek(self._entry + self.solid_size + 4)
            solid = SolidHeader()
            solid.read(reader)
            return solid

    def end(self):
        return self._entry + self.solid_size + 4


class Phy(Base):
    def __init__(self, filepath):
        self.reader = ByteIO(filepath)
        self.header = Header()
        self.solids = []  # type:List[SolidHeader]
        self.kv = ''

    def read(self):
        reader = self.reader
        self.header.read(reader)
        reader.seek(self.header.size)
        solid = SolidHeader()
        solid.read(reader)
        self.solids.append(solid)
        for _ in range(self.header.solid_count - 1):
            solid = solid.next_solid(reader)
            self.solids.append(solid)
        reader.seek(solid.end())
        reader.begin_region('KV')
        self.kv = self.reader.read_ascii_string()
        reader.end_region()
        pass
