import json
from datetime import datetime
from pathlib import Path
from typing import Union

from .file_entry import FileEntry
from ...utils.byte_io_mdl import ByteIO


def open_gma(filepath: Union[str, Path]):
    gma = GMA(Path(filepath))
    gma.read()
    return gma


class GMA:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.reader = ByteIO(filepath)
        self.version = 0
        self.steam_id = b''
        self.timestamp = datetime(1970, 1, 1)
        self.required_content = ''
        self.addon_name = ''
        self.addon_description = ''
        self.addon_author = ''
        self.addon_version = 0
        self._content_offset = 0
        self.file_entries = {}

    def read(self):
        reader = self.reader
        magic = reader.read(4)
        assert magic == b'GMAD'
        self.version, self.steam_id, timestamp = reader.read_fmt('>BQQ')
        self.timestamp = timestamp
        if self.version > 1:
            self.required_content = reader.read_ascii_string()
        self.addon_name = reader.read_ascii_string()
        self.addon_description = reader.read_ascii_string()
        self.addon_author = reader.read_ascii_string()
        self.addon_version = reader.read_uint32()

        offset = 0
        while True:
            entry = FileEntry()
            if not entry.read(reader):
                break
            entry.offset = offset
            offset += entry.size
            self.file_entries[entry.name] = entry
        self._content_offset = reader.tell()

    def find_file(self, filename):
        if filename in self.file_entries:
            entry = self.file_entries[filename]
            self.reader.seek(self._content_offset + entry.offset)
            data = self.reader.read(entry.size)
            return data
        return None
