from enum import IntEnum

from ....utilities.byte_io_mdl import ByteIO


class ColliderType(IntEnum):
    COLLIDE_POLY = 0
    COLLIDE_MOPP = 1
    COLLIDE_BALL = 2
    COLLIDE_VIRTUAL = 3


class IVPCompactSurface:
    def __init__(self):
        self.mass_center = []
        self.rotation_inertia = []
        self.upper_limit_radius = 0
        self.max_factor_surface_deviation = 0
        self.size = 0
        self.offset_ledge_tree_root = 0
        self.dummy = []
        self.id = ''

    def read(self, reader: ByteIO):
        self.mass_center = reader.read_fmt('3f')
        self.rotation_inertia = reader.read_fmt('3f')
        self.upper_limit_radius = reader.read_float()
        tmp = reader.read_uint32()
        self.max_factor_surface_deviation = tmp & 0xFF
        self.size = tmp >> 8
        self.offset_ledge_tree_root = reader.read_int32()
        self.dummy = reader.read_fmt('2I')
        self.id = reader.read_fourcc()


class ColliderHeader:
    def __init__(self):
        self.id = ''
        self.version = 0
        self.model_type = ColliderType(0)

    def read(self, reader: ByteIO):
        self.id = reader.read_fourcc()
        self.version = reader.read_int16()
        self.model_type = ColliderType(reader.read_int16())

    @classmethod
    def peek(cls, reader: ByteIO):
        self = cls()
        with reader.save_current_pos():
            self.id = reader.read_fourcc()
            self.version = reader.read_uint16()
            self.model_type = ColliderType(reader.read_int16())
        return self


class CompactSurfaceHeader(ColliderHeader):
    def __init__(self):
        super().__init__()
        self.surface_size = 0
        self.drag_axis_areas = []
        self.axis_map_size = 0
        self.surface = IVPCompactSurface()

    def read(self, reader: ByteIO):
        super().read(reader)
        self.surface_size = reader.read_uint32()
        self.drag_axis_areas = reader.read_fmt('3f')
        self.axis_map_size = reader.read_uint32()
        self.surface.read(reader)


class MoppHeader(ColliderHeader):
    def __init__(self):
        super().__init__()
        self.mopp_size = 0

    def read(self, reader: ByteIO):
        super().read(reader)
        self.mopp_size = reader.read_uint32()
