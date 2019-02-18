from SourceIO.byte_io_mdl import ByteIO
from SourceIO.data_structures.mdl_data import SourceMdlFileData
from SourceIO.mdl_readers.mdl_v49 import SourceMdlFile49


class SourceMdlFile10(SourceMdlFile49):
    def __init__(self, reader: ByteIO):
        super().__init__(reader)
        self.reader = reader
        self.file_data = SourceMdlFileDataV10()


class SourceMdlFileDataV10(SourceMdlFileData):

    def read_header00(self, reader: ByteIO):
        self.id = reader.read_fourcc()
        self.version = reader.read_int32()
        self.name = reader.read_ascii_string(64)
        self.file_size = reader.read_int32()
        self.actual_file_size = reader.size()

        self.eye_position.read(reader)

        self.hull_min_position.read(reader)
        self.hull_max_position.read(reader)

        self.view_bounding_box_min_position.read(reader)
        self.view_bounding_box_max_position.read(reader)

        self.flags = reader.read_int32()

        self.bone_count = reader.read_int32()
        self.bone_offset = reader.read_int32()

        self.bone_controller_count = reader.read_int32()
        self.bone_controller_offset = reader.read_int32()

        self.hitbox_set_count = reader.read_int32()
        self.hitbox_set_offset = reader.read_int32()

        self.local_sequence_count = reader.read_int32()
        self.local_sequence_offset = reader.read_int32()

        self.sequence_group_count = reader.read_int32()
        self.sequence_group_offset = reader.read_int32()
