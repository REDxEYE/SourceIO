from dataclasses import dataclass
from enum import IntFlag

from .....library.utils import Buffer


class StudioHeaderFlags(IntFlag):
    EF_ROCKET = 1  # ! leave a trail
    EF_GRENADE = 2  # ! leave a trail
    EF_GIB = 4  # ! leave a trail
    EF_ROTATE = 8  # ! rotate (bonus items)
    EF_TRACER = 16  # ! green split trail
    EF_ZOMGIB = 32  # ! small blood trail
    EF_TRACER2 = 64  # ! orange split trail + rotate
    EF_TRACER3 = 128  # ! purple trail
    EF_NOSHADELIGHT = 256  # ! No shade lighting
    EF_HITBOXCOLLISIONS = 512  # ! Use hitbox collisions
    EF_FORCESKYLIGHT = 1024  # ! Forces the model to be lit by skybox lighting


@dataclass(slots=True)
class StudioHeader:
    version: int
    name: str
    file_size: int

    bone_count: int
    bone_offset: int

    bone_controllers_count: int
    bone_controllers_offset: int

    sequence_count: int
    sequence_offset: int

    sequence_groups_count: int
    sequence_groups_offset: int

    texture_count: int
    texture_offset: int
    texture_data_offset: int

    skin_ref_count: int
    skin_families_count: int
    skin_offset: int

    body_part_count: int
    body_part_offset: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        magic = buffer.read_fourcc()
        assert magic == 'IDST', 'Not a GoldSrc model'
        version = buffer.read_int32()
        assert version in [6, 10], f'MDL version {version} are not supported by GoldSrc importer'
        name = buffer.read_ascii_string(64)
        file_size = buffer.read_int32()
        hdr_data = buffer.read_fmt('14I')
        buffer.skip(14 * 4)
        return cls(version, name, file_size, *hdr_data)
