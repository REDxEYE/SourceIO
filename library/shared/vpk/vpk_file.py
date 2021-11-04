from functools import lru_cache
from io import BytesIO
from pathlib import Path, WindowsPath, PosixPath, PurePath
from typing import Union, List, Dict

from .structs.entry import TitanfallEntry
from ...utils.byte_io_mdl import ByteIO
from .structs import *
from ...utils.thirdparty.lzham.lzham import LZHAM

class InvalidMagic(Exception):
    pass

def open_vpk(filepath: Union[str, Path]):
    from struct import unpack
    with open(filepath, 'rb') as f:
        magic, version_mj, version_mn = unpack('IHH', f.read(8))
    if magic != Header.MAGIC:
        raise InvalidMagic(f'Not a VPK file, expected magic: {Header.MAGIC}, got {magic}')
    if version_mj in [1, 2] and version_mn == 0:
        return VPKFile(filepath)
    elif version_mj == 2 and version_mn == 3 and LZHAM.lib is not None:
        return TitanfallVPKFile(filepath)
    else:
        raise NotImplementedError(f"Failed to find VPK handler for VPK:{version_mj}.{version_mn}. "
                                  f"LZHAM:{'Available' if LZHAM.lib else 'Unavailable'}")


class VPKFile:

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self.reader = ByteIO(self.filepath)
        self.header = Header()
        self.archive_md5_entries: List[ArchiveMD5Entry] = []

        self.entries: Dict[str, Entry] = {}
        self.tree_offset = 0
        self.tree_hash = b''
        self.archive_md5_hash = b''
        self.file_hash = b''

        self.public_key = b''
        self.signature = b''

        self._folders_in_current_dir = set()

    def read(self):
        reader = self.reader

        self.header.read(reader)
        entry = reader.tell()
        self.read_entries()
        self.reader.seek(entry + self.header.tree_size)
        if self.header.version == 2:
            reader.skip(self.header.file_data_section_size)
            reader.skip(self.header.archive_md5_section_size)

            assert self.header.other_md5_section_size == 48, \
                f'Invalid size of other_md5_section {self.header.other_md5_section_size} bytes, should be 48 bytes'
        if self.header.version == 2:
            self.tree_hash = reader.read(16)
            self.archive_md5_hash = reader.read(16)
            self.file_hash = reader.read(16)

            if self.header.signature_section_size != 0:
                self.public_key = reader.read(reader.read_int32())
                self.signature = reader.read(reader.read_int32())

    def read_entries(self):
        reader = self.reader
        self.tree_offset = reader.tell()
        while 1:
            type_name = reader.read_ascii_string()
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

                    full_path = f'{directory_name}/{file_name}.{type_name}'.lower()
                    self.entries[full_path] = Entry(full_path, reader.tell())
                    _, preload_size = reader.read_fmt('IH')
                    reader.skip(preload_size + 12)

    def read_archive_md5_section(self):
        reader = self.reader

        if self.header.archive_md5_section_size == 0:
            return

        entry_count = self.header.archive_md5_section_size // 28

        for _ in range(entry_count):
            md5_entry = ArchiveMD5Entry()
            md5_entry.read(reader)
            self.archive_md5_entries.append(md5_entry)

    @lru_cache(128)
    def find_file(self, full_path: Union[Path, str]):
        if isinstance(full_path, (Path, PurePath, PosixPath, WindowsPath)):
            full_path = full_path.as_posix().lower()
        else:
            full_path = Path(full_path).as_posix().lower()
        return self.entries.get(full_path, None)

    def read_file(self, entry: Entry) -> BytesIO:
        if not entry.loaded:
            entry.read(self.reader)
        if entry.archive_id == 0x7FFF:
            data = bytearray(entry.preload_data)
            with self.reader.save_current_pos():
                self.reader.seek(entry.offset + self.header.tree_size + self.tree_offset)
                data.extend(self.reader.read(entry.size))

            reader = BytesIO(data)
            return reader
        else:
            target_archive_path = self.filepath.parent / f'{self.filepath.stem[:-3]}{entry.archive_id:03d}.vpk'
            with open(target_archive_path, 'rb') as target_archive:
                target_archive.seek(entry.offset)
                reader = BytesIO(entry.preload_data + target_archive.read(entry.size))
                return reader

    def files_in_path(self, partial_path):
        if partial_path is None:
            self._folders_in_current_dir = set([Path(a).parts[0] for a in self.entries.keys()])
        else:
            partial_path = Path(partial_path).as_posix().lower()
            self._folders_in_current_dir.clear()
            for filepath in self.entries.keys():
                tmp = Path(filepath).parent.as_posix().lower()
                if tmp.startswith(partial_path):
                    self._folders_in_current_dir.add(Path(filepath).relative_to(partial_path).parts[0])
        yield from self._folders_in_current_dir


class TitanfallVPKFile(VPKFile):

    def read(self):
        reader = self.reader
        self.header.read(reader)
        entry = reader.tell()
        self.read_entries()
        self.reader.seek(entry + self.header.tree_size)

    def read_entries(self):
        reader = self.reader
        while 1:
            type_name = reader.read_ascii_string()
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
                    full_path = f'{directory_name}/{file_name}.{type_name}'.lower()
                    entry = self.entries[full_path] = TitanfallEntry(full_path, reader.tell())
                    entry.read(reader)

    def read_file(self, entry: TitanfallEntry) -> BytesIO:
        if not entry.loaded:
            entry.read(self.reader)
        if entry.archive_id == 0x7FFF:
            reader = BytesIO(entry.preload_data)
            return reader
        else:
            archive_name_base = self.filepath.stem[:-3]
            archive_name_base = 'client_' + archive_name_base.split('_', 1)[-1]
            target_archive_path = self.filepath.parent / f'{archive_name_base}{entry.archive_id:03d}.vpk'
            with open(target_archive_path, 'rb') as target_archive:

                buffer = entry.preload_data
                for block in entry.blocks:
                    target_archive.seek(block.offset)
                    block_data = target_archive.read(block.compressed_size)
                    if block.compressed_size == block.uncompressed_size:
                        buffer += block_data
                    else:
                        buffer += LZHAM.decompress_memory(block_data, block.uncompressed_size, 20, 1 << 0)
                reader = BytesIO(buffer)
                return reader
