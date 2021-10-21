from enum import IntFlag

from .....library.utils.byte_io_mdl import ByteIO


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
        self.name = ''
        self.file_size = 0

        self.bone_count = 0
        self.bone_offset = 0

        self.bone_controllers_count = 0
        self.bone_controllers_offset = 0

        self.sequence_count = 0
        self.sequence_offset = 0

        self.sequence_groups_count = 0
        self.sequence_groups_offset = 0

        self.texture_count = 0
        self.texture_offset = 0
        self.texture_data_offset = 0

        self.skin_ref_count = 0
        self.skin_families_count = 0
        self.skin_offset = 0

        self.body_part_count = 0
        self.body_part_offset = 0

    def read(self, reader: ByteIO):
        self.magic = reader.read_fourcc()
        assert self.magic == 'IDST', 'Not a GoldSrc model'
        self.version = reader.read_int32()
        assert self.version in [6, 10], f'MDL version {self.version} are not supported by GoldSrc importer'
        self.name = reader.read_ascii_string(64)
        self.file_size = reader.read_int32()
        (
            self.bone_count, self.bone_offset,
            self.bone_controllers_count, self.bone_controllers_offset,
            self.sequence_count, self.sequence_offset,
            self.texture_count, self.texture_offset, self.texture_data_offset,
            self.skin_ref_count, self.skin_families_count, self.skin_offset,
            self.body_part_count, self.body_part_offset,
        ) = reader.read_fmt('14I')
        reader.skip(14 * 4)
