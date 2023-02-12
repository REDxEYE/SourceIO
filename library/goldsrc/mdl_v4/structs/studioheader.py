from dataclasses import dataclass
from enum import IntFlag

from ....utils import Buffer


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
    flags: StudioHeaderFlags
    bone_count: int
    body_part_count: int
    total_model_count: int
    sequence_count: int
    total_frame_count: int
    unk_1: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        magic = buffer.read_fourcc()
        assert magic == 'IDST', 'Not a GoldSrc model'
        version = buffer.read_int32()
        assert version == 4, f'MDL version {version} are not supported by GoldSrc importer'

        flags = StudioHeaderFlags(buffer.read_int32())
        return cls(version,flags,*buffer.read_fmt('6I'))
