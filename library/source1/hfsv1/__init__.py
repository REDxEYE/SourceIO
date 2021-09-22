from pathlib import Path
from typing import Dict

from .directory import Directory
from .file import File
from .index import Index
from ...utils.byte_io_mdl import ByteIO


# noinspection PyShadowingNames
class HFS:

    def __init__(self, hfs_path: Path):
        self.reader = reader = ByteIO(hfs_path)
        self.index = Index()
        self.entries: Dict[str, Directory] = {}
        reader.seek(-0x16, 2)
        if not self.index.read(reader):
            reader.seek(0)
            directory = Directory()
            directory.file = File()
            directory.file.read(reader, True)
            self.entries[directory.filename.lower()] = directory
        else:
            reader.seek(self.index.directory_offset)
            for _ in range(self.index.directory_count):
                directory = Directory()
                directory.read(reader)
                self.entries[directory.filename.lower()] = directory

    def get_file(self, path):
        path = Path(path).as_posix().lower()
        if path in self.entries:
            return self.entries[path].file.read_file()
        return None

    def has_file(self, path):
        path = Path(path).as_posix()
        return path in self.entries
