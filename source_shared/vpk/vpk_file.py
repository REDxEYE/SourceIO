from pathlib import Path
from typing import Union, List, Dict

from ...utilities.byte_io_mdl import ByteIO

from .structs import *


class VPKFile:

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self.is_dir = self.filepath.stem[-3:] == 'dir'
        self.reader = ByteIO(self.filepath)
        self.header = Header()
        self.entries: Dict[str, List[Entry]] = {}
        self.archive_md5_entries: List[ArchiveMD5Entry] = []

        self.path_cache = []

        self.tree_hash = b''
        self.archive_md5_hash = b''
        self.file_hash = b''

        self.public_key = b''
        self.signature = b''

    def read(self):
        reader = self.reader
        self.header.read(reader)
        self.read_entries()
        if self.header.version == 2:
            reader.skip(self.header.file_data_section_size)
        self.read_archive_md5_section()

        assert self.header.other_md5_section_size == 48, f'Invalid size of other_md5_section {self.header.other_md5_section_size} bytes, should be 48 bytes'
        self.tree_hash = reader.read_bytes(16)
        self.archive_md5_hash = reader.read_bytes(16)
        self.file_hash = reader.read_bytes(16)

        if self.header.signature_section_size != 0:
            self.public_key = reader.read_bytes(reader.read_int32())
            self.signature = reader.read_bytes(reader.read_int32())

    def read_entries(self):
        reader = self.reader
        while 1:
            type_name = reader.read_ascii_string()
            if type_name not in self.entries:
                self.entries[type_name] = []
            if not type_name:
                break
            while 1:
                directory_name = reader.read_ascii_string()
                if not directory_name:
                    break
                while 1:
                    file_name = reader.read_ascii_string()
                    if not file_name:
                        break

                    entry = Entry(file_name, directory_name, type_name)
                    entry.read(reader)
                    self.path_cache.append(entry.full_path)

                    if reader.read_uint16() != 0xFFFF:
                        raise NotImplementedError('Invalid terminator')

                    if entry.preload_data_size > 0:
                        entry.preload_data = reader.read_bytes(entry.preload_data_size)

                    self.entries[type_name].append(entry)

    def read_archive_md5_section(self):
        reader = self.reader

        if self.header.archive_md5_section_size == 0:
            return

        entry_count = self.header.archive_md5_section_size // 28

        for _ in range(entry_count):
            md5_entry = ArchiveMD5Entry()
            md5_entry.read(reader)
            self.archive_md5_entries.append(md5_entry)

    def find_file(self, *, full_path: str = None,
                  file_type: str = None, directory: str = None, file_name: str = None):
        if full_path is not None:

            full_path = Path(full_path)
            if full_path.is_absolute():
                full_path = full_path.relative_to(self.filepath.parent)
            ext = Path(full_path).suffix.strip('./\\')
            for entry in self.entries[ext.lower()]:
                if entry.full_path == full_path:
                    return entry
        elif all([file_type, directory, file_name]):
            file_type = file_type.strip('./\\')
            directory = directory.strip('./\\')
            file_name = Path(file_name.strip('./\\')).stem
            for entry in self.entries[file_type]:
                if entry.directory_name == directory and entry.file_name == file_name:
                    return entry
        else:
            raise Exception("No valid parameters were given")

    def read_file(self, entry: Entry):
        if entry.archive_id == 0x7FFF:
            print("Internal file")
        else:
            target_archive_path = self.filepath.parent / f'{self.filepath.stem[:-3]}{entry.archive_id:03d}.vpk'
            target_archive = ByteIO(target_archive_path)
            target_archive.seek(entry.offset)
            reader = ByteIO(target_archive.read_bytes(entry.size))
            target_archive.close()
            del target_archive
            return reader
