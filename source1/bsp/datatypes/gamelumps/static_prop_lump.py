from typing import List

from .....utilities.byte_io_mdl import ByteIO


class StaticProp:
    def __init__(self):
        self.origin = []
        self.rotation = []
        self.prop_type = 0
        self.first_leaf = 0
        self.leaf_count = 0
        self.solid = 0
        self.flags = 0
        self.skin = 0
        self.fade_min_dist = 0.0
        self.fade_max_dist = 0.0
        self.lighting_origin = []

        self.forced_fade_scale = 0.0

        self.min_dx_level = 0
        self.max_dx_level = 0

        self.min_cpu_level = 0
        self.max_cpu_level = 0
        self.min_gpu_level = 0
        self.max_gpu_level = 0

        self.diffuse_modulation = []
        self.disable_x360 = 0
        self.flags_ex = 0
        self.uniform_scale = 0.0

    def parse(self, reader: ByteIO, version: int):
        if version >= 4:
            self.origin = reader.read_fmt('3f')
            self.rotation = reader.read_fmt('3f')
            self.prop_type, self.first_leaf, self.leaf_count = reader.read_fmt('3H')
            self.solid, self.flags = reader.read_fmt('2B')
            self.skin = reader.read_int32()
            self.fade_min_dist, self.fade_max_dist = reader.read_fmt('2f')

            self.lighting_origin = reader.read_fmt('3f')
        if version >= 5:
            self.forced_fade_scale = reader.read_float()
        if version in [6, 7]:
            self.min_dx_level, self.max_dx_level = reader.read_fmt('2H')
        if version >= 8:
            self.min_cpu_level, self.max_cpu_level, self.min_gpu_level, self.max_gpu_level = reader.read_fmt('4B')
        if version >= 7:
            self.diffuse_modulation = reader.read_fmt('4B')
        if version in [9, 10]:
            self.disable_x360 = reader.read_uint32()
        if version > 10:
            self.flags_ex = reader.read_uint32()
        if version >= 11:
            reader.skip(4)
            self.uniform_scale = reader.read_float()


class StaticPropLump:
    def __init__(self, glump_info):
        from ..game_lump_header import GameLumpHeader
        self._glump_info: GameLumpHeader = glump_info
        self.model_names: List[str] = []
        self.leafs: List[int] = []
        self.static_props: List[StaticProp] = []

    def parse(self, reader: ByteIO):
        for _ in range(reader.read_int32()):
            self.model_names.append(reader.read_ascii_string(128))
        for _ in range(reader.read_int32()):
            self.leafs.append(reader.read_uint16())
        prop_count = reader.read_int32()
        for _ in range(prop_count):
            prop = StaticProp()
            prop.parse(reader, self._glump_info.version)
            self.static_props.append(prop)
