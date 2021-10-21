from enum import IntFlag

from ....utils.byte_io_mdl import ByteIO


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


class StudioHeader:
    def __init__(self):
        self.magic = ''
        self.version = 0
        self.flags = StudioHeaderFlags(0)

        self.bone_count = 0
        self.body_part_count = 0
        self.unk_count = 0
        self.sequence_count = 0
        self.total_frame_count = 0
        self.unk_1 = 0

    def read(self, reader: ByteIO):
        self.magic = reader.read_fourcc()
        assert self.magic == 'IDST', 'Not a GoldSrc model'
        self.version = reader.read_int32()
        assert self.version == 4, f'MDL version {self.version} are not supported by GoldSrc importer'

        self.flags = StudioHeaderFlags(reader.read_int32())

        (self.bone_count,
         self.body_part_count,
         self.unk_count,
         self.sequence_count,
         self.total_frame_count,
         self.unk_1,
         ) = reader.read_fmt('6I')
