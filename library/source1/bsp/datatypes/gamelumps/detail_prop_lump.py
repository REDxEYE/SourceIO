from typing import List

from .....utils.file_utils import IBuffer


class DetailPropLump:
    def __init__(self, glump_info):
        from ..game_lump_header import GameLumpHeader
        self._glump_info: GameLumpHeader = glump_info
        self.model_names: List[str] = []
        self.sprites: List[DetailSprite] = []
        self.detail_props: List[DetailProp] = []

    def parse(self, reader: IBuffer, bsp: 'BSPFile'):
        for _ in range(reader.read_int32()):
            self.model_names.append(reader.read_ascii_string(128))
        if self._glump_info.version == 4:
            for _ in range(reader.read_int32()):
                prop = DetailSprite()
                prop.parse(reader, self._glump_info.version)
                self.sprites.append(prop)
            for _ in range(reader.read_int32()):
                prop = DetailProp()
                prop.parse(reader, self._glump_info.version)
                self.detail_props.append(prop)


class DetailSprite:

    def __init__(self):
        self.upper_left = []
        self.lower_right = []
        self.upper_left_uv = []
        self.lower_right_uv = []

    def parse(self, reader: IBuffer, version):
        self.upper_left = reader.read_fmt('2f')
        self.lower_right = reader.read_fmt('2f')
        self.upper_left_uv = reader.read_fmt('2f')
        self.lower_right_uv = reader.read_fmt('2f')


class DetailProp:
    def __init__(self):
        self.origin = []
        self.rotation = []
        self.model_id = 0
        self.leaf_id = 0
        self.lighting = []
        self.light_style = 0
        self.light_style_count = 0
        self.sway_amount = 0
        self.shape_angle = 0
        self.shape_size = 0
        self.orientation = 0
        self.type = 0
        self.scale = 0

    def parse(self, reader: IBuffer, version):
        self.origin = reader.read_fmt('3f')
        self.rotation = reader.read_fmt('3f')
        self.model_id, self.leaf_id = reader.read_fmt('2H')
        self.lighting = reader.read_fmt('4B')
        (self.light_style, self.light_style_count,
         self.sway_amount, self.shape_angle,
         self.shape_size, self.orientation,
         self.type, self.scale) = reader.read_fmt('I4BB3xB3xf')
