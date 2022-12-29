from ....utils import IBuffer


class Header:
    MAGIC = 0x55AA1234

    def __init__(self):
        self.magic = 0
        self.version = (0, 0)
        self.tree_size = 0

        self.file_data_section_size = 0
        self.archive_md5_section_size = 0
        self.other_md5_section_size = 0
        self.signature_section_size = 0

    def read(self, reader: IBuffer):
        self.magic = reader.read_uint32()
        assert self.magic == self.MAGIC, "Not a VPK file"

        self.version = reader.read_fmt('2H')
        self.tree_size = reader.read_uint32()

        if self.version[0] == 1:
            ...
        elif self.version[0] == 2 and self.version[1] == 0:
            self.file_data_section_size = reader.read_uint32()
            self.archive_md5_section_size = reader.read_uint32()
            self.other_md5_section_size = reader.read_uint32()
            self.signature_section_size = reader.read_uint32()
        elif self.version[0] == 2 and self.version[1] == 3:
            self.file_data_section_size = reader.read_uint32()
            pass
        else:
            raise NotImplementedError(f"Bad VPK version ({self.version})")

