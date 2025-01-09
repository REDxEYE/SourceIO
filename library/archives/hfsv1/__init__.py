from SourceIO.library.utils import FileBuffer
from SourceIO.library.utils.tiny_path import TinyPath
from .file import File
from .index import Index


# noinspection PyShadowingNames
class HFS:

    def __init__(self, hfs_path: TinyPath):
        self.buffer = buffer = FileBuffer(hfs_path)
        self.entries: dict[str, File] = {}
        buffer.seek(-0x16, 2)
        self.index = Index.from_buffer(buffer)
        if self.index is None:
            # buffer.seek(0)
            # directory = Directory()
            # directory.file = File.from_buffer(buffer, True)
            file = File.from_buffer(buffer, True)
            self.entries[file.filename.lower()] = file
        else:
            buffer.seek(self.index.directory_offset)
            for _ in range(self.index.directory_count):
                buffer.skip(38)
                data_offset = buffer.read_uint32()
                buffer.seek(data_offset)
                file = File.from_buffer(buffer)
                self.entries[file.filename.lower()] = file

    def get_file(self, path):
        path = TinyPath(path).as_posix().lower()
        if path in self.entries:
            return self.entries[path].read_file(self.buffer)
        return None

    def has_file(self, path):
        path = TinyPath(path).as_posix()
        return path in self.entries
