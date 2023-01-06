from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

from .....shared.types import Vector2, Vector3
from .....utils.file_utils import Buffer

if TYPE_CHECKING:
    from ...bsp_file import BSPFile
    from ..game_lump_header import GameLumpHeader


class DetailPropLump:
    def __init__(self, glump_info: 'GameLumpHeader'):
        self._glump_info = glump_info
        self.model_names: List[str] = []
        self.sprites: List[DetailSprite] = []
        self.detail_props: List[DetailProp] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        for _ in range(buffer.read_int32()):
            self.model_names.append(buffer.read_ascii_string(128))
        if self._glump_info.version == 4:
            for _ in range(buffer.read_int32()):
                self.sprites.append(DetailSprite.from_buffer(buffer, self._glump_info.version, bsp))
            for _ in range(buffer.read_int32()):
                self.detail_props.append(DetailProp.from_buffer(buffer, self._glump_info.version, bsp))


@dataclass(slots=True)
class DetailSprite:
    upper_left: Vector2[float]
    lower_right: Vector2[float]
    upper_left_uv: Vector2[float]
    lower_right_uv: Vector2[float]

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        return cls(buffer.read_fmt('2f'), buffer.read_fmt('2f'),
                   buffer.read_fmt('2f'), buffer.read_fmt('2f'))


@dataclass(slots=True)
class DetailProp:
    origin: Vector3[float]
    rotation: Vector3[float]
    model_id: int
    leaf_id: int
    lighting: Tuple[int, int, int, int]
    light_style: int
    light_style_count: int
    sway_amount: int
    shape_angle: int
    shape_size: int
    orientation: int
    type: int
    scale: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: 'BSPFile'):
        origin = buffer.read_fmt('3f')
        rotation = buffer.read_fmt('3f')
        model_id, leaf_id = buffer.read_fmt('2H')
        lighting = buffer.read_fmt('4B')
        (light_style, light_style_count,
         sway_amount, shape_angle,
         shape_size, orientation,
         detail_type, scale) = buffer.read_fmt('I4BB3xB3xf')
        return cls(origin, rotation, model_id, leaf_id, lighting, light_style, light_style_count, sway_amount,
                   shape_angle, shape_size, orientation, detail_type, scale)
