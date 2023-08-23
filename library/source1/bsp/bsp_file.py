from pathlib import Path
from typing import Dict, Tuple

from ....logger import SLoggingManager
from ...shared.content_providers.content_manager import ContentManager
from ...utils.file_utils import FileBuffer
from .lump import *

log_manager = SLoggingManager()


def open_bsp(filepath):
    from struct import unpack
    assert Path(filepath).exists()
    with open(filepath, 'rb') as f:
        magic, version = unpack('4sI', f.read(8))

    if magic == b'VBSP':
        return BSPFile.from_filename(filepath)
    elif magic == b'rBSP':
        return RespawnBSPFile.from_filename(filepath)


CM = ContentManager()


class BSPFile:
    def __init__(self, filepath: Path, buffer: Buffer):
        self.filepath = Path(filepath)
        self.buffer = buffer
        self.logger = log_manager.get_logger(self.filepath.stem)
        self.version = 0
        self.is_l4d2 = False
        self.lumps_info: List[LumpInfo] = []
        self.lumps: Dict[str, Lump] = {}
        self.revision = 0
        self.content_manager = CM
        self.steam_app_id = CM.get_content_provider_from_path(filepath).steam_id

    @classmethod
    def from_filename(cls, filepath: Path):
        buffer = FileBuffer(filepath)
        self = cls(filepath, buffer)
        magic = buffer.read_fourcc()
        assert magic == "VBSP", "Invalid BSP header"
        version = buffer.read_int32()
        if version > 0xFFFF:
            self.version = version & 0xFFFF, version >> 16
        else:
            self.version = (version, 0)
        self.is_l4d2 = is_l4d2 = buffer.peek_uint32() <= 1036 and self.version == (21, 0)
        self.lumps_info = [None] * 64
        for lump_id in range(64):
            lump = LumpInfo.from_buffer(buffer, lump_id, is_l4d2)
            self.lumps_info[lump_id] = lump
        self.revision = buffer.read_int32()
        return self

    def get_lump(self, lump_name):
        if lump_name in self.lumps:
            return self.lumps[lump_name]
        else:
            matches: List[Tuple[Type[Lump], LumpTag]] = []
            for sub in Lump.all_subclasses():
                sub: Type[Lump]
                for dep in sub.tags:
                    if dep.lump_name == lump_name:
                        if dep.bsp_version is not None and dep.bsp_version > self.version:
                            continue
                        if dep.steam_id is not None and dep.steam_id != self.steam_app_id:
                            continue
                        if dep.lump_version is not None and dep.lump_version != self.lumps_info[dep.lump_id].version:
                            continue
                        matches.append((sub, dep))
            best_matches = []
            for match_sub, match_dep in matches:
                lump = self.lumps_info[match_dep.lump_id]
                rank = 0
                if match_dep.bsp_version is not None and match_dep.bsp_version == self.version:
                    rank += 2
                elif match_dep.bsp_version is not None and match_dep.bsp_version > self.version:
                    rank += 1
                if match_dep.steam_id is not None and match_dep.steam_id == self.steam_app_id:
                    rank += 1
                if match_dep.lump_version is not None and match_dep.lump_version == lump.version:
                    rank += 1
                best_matches.append((rank, match_sub, match_dep))
            if not best_matches:
                return
            best_matches = list(sorted(best_matches, key=lambda a: a[0]))
            _, sub, dep = best_matches[-1]

            parsed_lump = self.parse_lump(sub, dep.lump_id, dep.lump_name)
            self.lumps[lump_name] = parsed_lump
            return parsed_lump

    def _get_lump_buffer(self, lump_id: int, lump_info: LumpInfo) -> Buffer:
        base_path = self.filepath.parent
        lump_path = base_path / f'{self.filepath.name}.{lump_id:04x}.bsp_lump'

        if lump_path.exists():
            return FileBuffer(lump_path)

        if not lump_info.compressed:
            return self.buffer.slice(lump_info.offset, lump_info.size)
        else:
            buffer = Lump.decompress_lump(self.buffer.slice(lump_info.offset, lump_info.size))
            assert buffer.size() == lump_info.decompressed_size
            return buffer

    def parse_lump(self, lump_class: Type[Lump], lump_id, lump_name):
        base_path = self.filepath.parent
        lump_path = base_path / f'{self.filepath.stem}_l_{lump_id}.lmp'
        if lump_path.exists():
            buffer = FileBuffer(lump_path)
            lump_info = LumpInfo.from_buffer(buffer, lump_id, self.is_l4d2)
            self.lumps_info[lump_id] = lump_info
            buffer.seek(lump_info.offset)

            parsed_lump = lump_class(lump_info).parse(buffer, self)
            self.lumps[lump_id] = parsed_lump
            return parsed_lump

        if self.lumps_info[lump_id].size != 0:
            lump_info = self.lumps_info[lump_id]
            buffer = self._get_lump_buffer(lump_id, lump_info)

            parsed_lump = lump_class(lump_info).parse(buffer, self)
            self.lumps[lump_id] = parsed_lump
            return parsed_lump


class RespawnBSPFile(BSPFile):

    def __init__(self, filepath: Path, buffer: Buffer):
        super().__init__(filepath, buffer)

    @classmethod
    def from_filename(cls, filepath: Path):
        buffer = FileBuffer(filepath)
        self = cls(filepath, buffer)
        magic = buffer.read_fourcc()
        assert magic == "rBSP", "Invalid BSP header"
        self.version = buffer.read_uint32()
        self.revision = buffer.read_uint32()
        last_lump = buffer.read_uint32()
        self.lumps_info = [None] * last_lump
        for lump_id in range(last_lump + 1):
            lump = LumpInfo.from_buffer(buffer, lump_id)
            lump.id = lump_id
            self.lumps_info[lump_id] = lump
        return self
