from byte_io_mdl import ByteIO
from mdl_data import SourceMdlFileData
from mdl_readers.mdl_v49 import SourceMdlFile49


class SourceMdlFile53(SourceMdlFile49):
    # Super class call does not required here due to different __init__ behaviour
    # noinspection PyMissingConstructor
    def __init__(self, reader: ByteIO):
        self.reader = reader
        self.file_data = SourceMdlFileDataV53()
        self.VVD = self.file_data.vvd_data
        self.VTX = self.file_data.vtx_data

    def read_flex_controllers_ui(self):
        return


class SourceMdlFileDataV53(SourceMdlFileData):
    def __init__(self):
        super().__init__()
        self.name_copy_offset = 0
        self.vtx_offset = 0
        self.vvd_offset = 0
        self.vtx_data = None
        self.vvd_data = None

    def read(self, reader: ByteIO):
        super().read(reader)

    def read_header00(self, reader: ByteIO):
        self.id = reader.read_fourcc()
        self.version = reader.read_uint32()
        self.checksum = reader.read_uint32()

        self.name_copy_offset = reader.read_uint32()

        self.name = reader.read_ascii_string(64)
        self.file_size = reader.read_uint32()

    def read_header01(self, reader: ByteIO):
        super().read_header01(reader)
        reader.skip(16)
        self.vtx_offset = reader.read_uint32()
        self.vvd_offset = reader.read_uint32()
        print('VTF:{} VVD:{}'.format(self.vtx_offset, self.vvd_offset))
        if self.vvd_offset != 0 and self.vtx_offset != 0:
            with reader.save_current_pos():
                reader.seek(self.vtx_offset)
                self.vtx_data = ByteIO(byte_object=reader.read_bytes(-1))
                reader.seek(self.vvd_offset)
                self.vvd_data = ByteIO(byte_object=reader.read_bytes(-1))
