from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from SourceIO.library.utils import Buffer, FileBuffer
from SourceIO.library.utils.exceptions import InvalidFileMagic
from SourceIO.library.utils.tiny_path import TinyPath


def check_gma(filepath: TinyPath):
    tmp = FileBuffer(filepath)
    ident = tmp.read(4)
    return ident == b'GMAD'


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


class GMA:
    def __init__(self, filepath: TinyPath):
        self.filepath = filepath
        self.buffer = FileBuffer(filepath)
        self.version = 0
        self.steam_id = b''
        self.timestamp = datetime(1970, 1, 1)
        self.required_content = ''
        self.addon_name = ''
        self.addon_description = ''
        self.addon_author = ''
        self.addon_version = 0
        self._content_offset = 0
        self.file_entries: dict[TinyPath, FileEntry] = {}

    def read(self):
        buffer = self.buffer
        magic = buffer.read(4)
        if magic != b"GMAD":
            raise InvalidFileMagic("Not a GMA file", b"GMAD", magic)
        self.version, self.steam_id, timestamp = buffer.read_fmt('BQQ')
        self.timestamp = timestamp
        if self.version > 1:
            self.required_content = buffer.read_ascii_string()
        self.addon_name = buffer.read_ascii_string()
        self.addon_description = buffer.read_ascii_string()
        self.addon_author = buffer.read_ascii_string()
        self.addon_version = buffer.read_uint32()

        offset = 0
        while True:
            entry = FileEntry.from_buffer(buffer)
            if entry is None:
                break
            entry.offset = offset
            offset += entry.size
            self.file_entries[TinyPath(entry.name.lower())] = entry
        self._content_offset = buffer.tell()

    def find_file(self, filename: TinyPath) -> Optional[Buffer]:
        filename = filename.as_posix().lower()
        if filename in self.file_entries:
            entry = self.file_entries[filename]
            data = self.buffer.slice(self._content_offset + entry.offset, entry.size)
            return data
        return None

    def has_file(self, filename: TinyPath) -> bool:
        filename = filename.as_posix().lower()
        return filename in self.file_entries
