from functools import lru_cache
from io import BytesIO
from pathlib import Path, WindowsPath, PosixPath, PurePath
from typing import Union, List, Dict

from .structs.entry import TitanfallEntry
from ...utils import FileBuffer, MemoryBuffer, IBuffer
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
        self.buffer = FileBuffer(self.filepath)
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
        buffer = self.buffer
        self.header.read(buffer)
        # entry = buffer.tell()
        self.read_entries()
        # buffer.seek(entry + self.header.tree_size)
        # if self.header.version == 2:
        #     buffer.skip(self.header.file_data_section_size)
        #     buffer.skip(self.header.archive_md5_section_size)
        #
        #     assert self.header.other_md5_section_size == 48, \
        #         f'Invalid size of other_md5_section {self.header.other_md5_section_size} bytes, should be 48 bytes'
        # if self.header.version == 2:
        #     self.tree_hash = buffer.read(16)
        #     self.archive_md5_hash = buffer.read(16)
        #     self.file_hash = buffer.read(16)
        #
        #     if self.header.signature_section_size != 0:
        #         self.public_key = buffer.read(buffer.read_int32())
        #         self.signature = buffer.read(buffer.read_int32())

    def read_entries(self):
        buffer = self.buffer
        self.tree_offset = buffer.tell()
        while 1:
            type_name = buffer.read_ascii_string()
            if not type_name:
                break
            while 1:
                directory_name = buffer.read_ascii_string()
                if not directory_name:
                    break
                while 1:
                    file_name = buffer.read_ascii_string()
                    if not file_name:
                        break

                    full_path = f'{directory_name}/{file_name}.{type_name}'.lower()
                    self.entries[full_path] = Entry(full_path, buffer.tell())
                    _, preload_size = buffer.read_fmt('IH')
                    buffer.skip(preload_size + 12)

    def read_archive_md5_section(self):
        buffer = self.buffer

        if self.header.archive_md5_section_size == 0:
            return

        entry_count = self.header.archive_md5_section_size // 28

        for _ in range(entry_count):
            md5_entry = ArchiveMD5Entry()
            md5_entry.read(buffer)
            self.archive_md5_entries.append(md5_entry)

    @lru_cache(128)
    def find_file(self, full_path: Union[Path, str]):
        if isinstance(full_path, (Path, PurePath, PosixPath, WindowsPath)):
            full_path = full_path.as_posix().lower()
        else:
            full_path = Path(full_path).as_posix().lower()
        return self.entries.get(full_path, None)

    def read_file(self, file_entry: Entry) -> IBuffer:
        if not file_entry.loaded:
            file_entry.read(self.buffer)
        if file_entry.archive_id == 0x7FFF:
            data = bytearray(file_entry.preload_data)
            with self.buffer.save_current_offset():
                self.buffer.seek(file_entry.offset + self.header.tree_size + self.tree_offset)
                data.extend(self.buffer.read(file_entry.size))

            reader = MemoryBuffer(data)
            return reader
        else:
            target_archive_path = self.filepath.parent / f'{self.filepath.stem[:-3]}{file_entry.archive_id:03d}.vpk'
            with open(target_archive_path, 'rb') as target_archive:
                target_archive.seek(file_entry.offset)
                reader = MemoryBuffer(file_entry.preload_data + target_archive.read(file_entry.size))
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

    def __init__(self, filepath: Union[str, Path]):
        super().__init__(filepath)
        self.entries: Dict[str, TitanfallEntry] = {}

    def read(self):
        buffer = self.buffer
        self.header.read(buffer)
        # entry = buffer.tell()
        self.read_entries()
        # self.buffer.seek(entry + self.header.tree_size)

    def read_entries(self):
        buffer = self.buffer
        while 1:
            type_name = buffer.read_ascii_string()
            if not type_name:
                break
            while 1:
                directory_name = buffer.read_ascii_string()
                if not directory_name:
                    break
                while 1:
                    file_name = buffer.read_ascii_string()
                    if not file_name:
                        break
                    full_path = f'{directory_name}/{file_name}.{type_name}'.lower()
                    entry = self.entries[full_path] = TitanfallEntry(full_path, buffer.tell())
                    entry.read(buffer)

    def read_file(self, file_entry: TitanfallEntry) -> IBuffer:
        if not file_entry.loaded:
            file_entry.read(self.buffer)
        if file_entry.archive_id == 0x7FFF:
            reader = MemoryBuffer(file_entry.preload_data)
            return reader
        else:
            archive_name_base = self.filepath.stem[:-3]
            archive_name_base = 'client_' + archive_name_base.split('_', 1)[-1]
            target_archive_path = self.filepath.parent / f'{archive_name_base}{file_entry.archive_id:03d}.vpk'
            with open(target_archive_path, 'rb') as target_archive:

                buffer = file_entry.preload_data
                for block in file_entry.blocks:
                    target_archive.seek(block.offset)
                    block_data = target_archive.read(block.compressed_size)
                    if block.compressed_size == block.uncompressed_size:
                        buffer += block_data
                    else:
                        buffer += LZHAM.decompress_memory(block_data, block.uncompressed_size, 20, 1 << 0)
                reader = MemoryBuffer(buffer)
                return reader
