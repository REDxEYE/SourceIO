from io import BytesIO
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

        self.path_cache = {}

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
            # self.read_archive_md5_section()
            reader.skip(self.header.archive_md5_section_size)

            assert self.header.other_md5_section_size == 48, \
                f'Invalid size of other_md5_section {self.header.other_md5_section_size} bytes, should be 48 bytes'
        if self.header.version == 2:
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
            if not type_name:
                break
            if type_name not in self.entries:
                self.entries[type_name] = []

            while 1:
                directory_name = reader.read_ascii_string()
                if not directory_name:
                    break
                while 1:
                    file_name = reader.read_ascii_string()
                    if not file_name:
                        break

                    full_path = Path(f'{directory_name}/{file_name}.{type_name}')
                    entry = Entry(full_path)
                    entry.read(reader)
                    self.path_cache[full_path] = entry

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

    def find_file(self, *, full_path: str = None, file_type: str = None, directory: str = None, file_name: str = None):
        if full_path is not None:
            full_path = Path(full_path)
            if full_path.is_absolute():
                full_path = full_path.relative_to(self.filepath.parent)

            # ext = Path(full_path).suffix.strip('./\\')
            # for entry in self.entries.get(ext.lower(), []):
            #     if entry.file_name == full_path:
            #         return entry
            # return None
        elif file_type and directory and file_name:
            file_type = file_type.strip('./\\')
            directory = directory.strip('./\\')
            file_name = Path(file_name.strip('./\\')).stem
            full_path = Path(f'{directory}/{file_name}.{file_type}')
        else:
            raise Exception("No valid parameters were given")
            # for entry in self.entries.get(file_type, []):
            #     if entry.directory_name == directory and entry.file_name == file_name:
            #         return entry
            # return None
        entry = self.path_cache.get(full_path,None)
        if entry:
            return entry


    def read_file(self, entry: Entry) -> BytesIO:
        if entry.archive_id == 0x7FFF:
            # print("Internal file")
            # raise NotImplementedError('Internal files are not supported.')
            reader = BytesIO(entry.preload_data)
            return reader
        else:
            target_archive_path = self.filepath.parent / f'{self.filepath.stem[:-3]}{entry.archive_id:03d}.vpk'
            print(f'Reading {entry.file_name} from {target_archive_path}')
            with open(target_archive_path, 'rb') as target_archive:
                target_archive.seek(entry.offset)
                reader = BytesIO(entry.preload_data + target_archive.read(entry.size))
                return reader
