from dataclasses import dataclass, field
from typing import Tuple, Optional

from ....utils import Buffer

VPK_MAGIC = 0x55AA1234


@dataclass
class Header:
    version: Tuple[int, int]
    tree_size: int
    file_data_section_size: Optional[int] = field(default=None)
    archive_md5_section_size: Optional[int] = field(default=None)
    other_md5_section_size: Optional[int] = field(default=None)
    signature_section_size: Optional[int] = field(default=None)

    @classmethod
    def from_buffer(cls, reader: Buffer):
        magic = reader.read_uint32()
        assert magic == VPK_MAGIC, "Not a VPK file"

        version = reader.read_fmt('2H')
        tree_size = reader.read_uint32()

        file_data_section_size = None
        archive_md5_section_size = None
        other_md5_section_size = None
        signature_section_size = None

        if version[0] == 1:
            ...
        elif version[0] == 2 and version[1] == 0:
            file_data_section_size = reader.read_uint32()
            archive_md5_section_size = reader.read_uint32()
            other_md5_section_size = reader.read_uint32()
            signature_section_size = reader.read_uint32()
        elif version[0] == 2 and version[1] == 3:
            file_data_section_size = reader.read_uint32()
            pass
        else:
            raise NotImplementedError(f"Bad VPK version ({version})")
        return cls(version, tree_size,
                   file_data_section_size, archive_md5_section_size, other_md5_section_size, signature_section_size)
