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
        self.size = reader.read_uint32()
        self.id = reader.read_uint32()
        self.solid_count = reader.read_uint32()
        self.checksum = reader.read_uint32()


class TreeNode:
    def __init__(self):
        self.__entry = 0
        self.right_node_offset = 0
        self.convex_offset = 0
        self.floats = []

        self.left_node: Optional[TreeNode] = None
        self.right_node: Optional[TreeNode] = None
        self.convex_leaf: Optional[ConvexLeaf] = None

    @property
    def is_leaf(self):
        return self.right_node_offset == 0

    def read(self, reader: ByteIO):
        self.__entry = reader.tell()
        self.right_node_offset, self.convex_offset, *self.floats = reader.read_fmt('2i5f')
        with reader.save_current_pos():
            if self.is_leaf:
                with reader.save_current_pos():
                    reader.seek(self.__entry + self.convex_offset)
                    self.convex_leaf = ConvexLeaf()
                    self.convex_leaf.read(reader)
                return
            self.left_node = TreeNode()
            self.left_node.read(reader)
            with reader.save_current_pos():
                reader.seek(self.__entry + self.right_node_offset)
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
    def __init__(self):
        self.__entry = 0
        self.vertex_offset = 0
        self.pad = []
        self.triangle_count = 0
        self.unused = 0

        self.triangles = []
        self.vertices = []

    def read(self, reader: ByteIO):
        self.__entry = reader.tell()
        self.vertex_offset, *self.pad, self.triangle_count, self.unused = reader.read_fmt('3i2h')
        with reader.save_current_pos():
            all_indices = set()
            triangles = []
            for _ in range(self.triangle_count):
                tri = ConvexTriangle()
                tri.read(reader)
                triangles.append(tri.vertex_ids)
                all_indices.update(tri.vertex_ids)
            self.triangles = np.array(triangles)
            self.triangles-=self.triangles.min()
            reader.seek(self.__entry + self.vertex_offset)
            self.vertices = np.frombuffer(reader.read(4 * 4 * len(all_indices)), np.float32)
            self.vertices = self.vertices.reshape((-1, 4))[:, :3]
            pass


class CollisionModel:
    def __init__(self):
        self.__entry = 0
        self.values = []
        self.surface = 0
        self.offset_tree = 0
        self.pad = []
        self.root_tree = TreeNode()

    def read(self, reader: ByteIO):
        self.__entry = reader.tell()
        self.values = reader.read_fmt('7f')
        self.surface, self.offset_tree, *self.pad = reader.read_fmt('4I')
        with reader.save_current_pos():
            reader.seek(self.__entry + self.offset_tree)
            self.root_tree.read(reader)


class SolidHeader:

    def __init__(self):
        self.__entry = 0
        self.solid_size = 0
        self.id = ''
        self.version = 0
        self.type = 0
        self.size = 0
        self.areas = []
        self.axis_map_size = 0
        self.collision_model = CollisionModel()
        self.ivps_data = b''

    def read(self, reader: ByteIO):
        self.__entry = reader.tell()
        self.solid_size = reader.read_uint32()
        self.id = reader.read_fourcc()
        self.version = reader.read_uint16()
        self.type = reader.read_uint16()
        self.size = reader.read_uint32()
        self.areas = reader.read_fmt('3f')
        self.axis_map_size = reader.read_uint32()
        self.collision_model.read(reader)
        self.ivps_data = reader.read(self.size - 44)

    def next_solid(self, reader: ByteIO):
        with reader.save_current_pos():
            reader.seek(self.__entry + self.solid_size + 4)
            solid = SolidHeader()
            solid.read(reader)
            return solid


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
        for _ in range(self.header.solid_count):
            solid = SolidHeader()
            solid.read(reader)
            self.solids.append(solid)
        self.kv = self.reader.read_ascii_string()
        pass
