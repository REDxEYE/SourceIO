from dataclasses import dataclass, field
from typing import Optional

from ...utils import Buffer


@dataclass(slots=True)
class FileEntry:
    id: int
    name: str
    size: int
    crc: int
    offset: int = field(init=False)

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> Optional['FileEntry']:
        entry_id = buffer.read_uint32()
        if entry_id == 0:
            return None
        name = buffer.read_ascii_string()
        size, crc = buffer.read_fmt('IQ')
        return cls(entry_id, name, size, crc)

    def __repr__(self) -> str:
        return f'FileEntry "{self.name}":0x{self.crc:X} {self.offset}:{self.size}'
