from pathlib import Path

from typing import Dict, Type, Tuple

from .lump import *
from ...utils.byte_io_mdl import ByteIO
from ....logger import SLoggingManager
from ...shared.content_providers.content_manager import ContentManager

log_manager = SLoggingManager()


def open_bsp(filepath):
    from struct import unpack
    assert Path(filepath).exists()
    with open(filepath, 'rb') as f:
        magic, version = unpack('4sI', f.read(8))

    if magic == b'VBSP':
        return BSPFile(filepath)
    elif magic == b'rBSP':
        return RespawnBSPFile(filepath)


class BSPFile:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.logger = log_manager.get_logger(self.filepath.name)
        self.reader = ByteIO(self.filepath)
        self.version = 0
        self.lumps_info: List[LumpInfo] = []
        self.lumps: Dict[str, Lump] = {}
        self.revision = 0
        self.content_manager = ContentManager()
        content_provider = self.content_manager.get_content_provider_from_path(self.filepath)
        self.steam_app_id = content_provider.steam_id

    def parse(self):
        reader = self.reader
        magic = reader.read_fourcc()
        assert magic == "VBSP", "Invalid BSP header"
        self.version = reader.read_int32()
        is_l4d2 = reader.peek_uint32() <= 1036 and self.version == 21
        for lump_id in range(64):
            lump = LumpInfo(lump_id)
            lump.parse(reader, is_l4d2)
            self.lumps_info.append(lump)
        self.revision = reader.read_int32()

        # self.parse_lumps()

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
            best_matches = {}
            for match_sub, match_dep in matches:
                lump = self.lumps_info[match_dep.lump_id]
                rank = 0
                if match_dep.bsp_version is not None and match_dep.bsp_version > self.version:
                    rank += 1
                if match_dep.steam_id is not None and match_dep.steam_id == self.steam_app_id:
                    rank += 1
                if match_dep.lump_version is not None and match_dep.lump_version == lump.version:
                    rank += 1
                best_matches[rank] = (match_sub, match_dep)
            if not best_matches:
                return
            best_match_id = max(best_matches.keys())
            sub, dep = best_matches[best_match_id]

            parsed_lump = self.parse_lump(sub, dep.lump_id, dep.lump_name)
            self.lumps[lump_name] = parsed_lump
            return parsed_lump

    def parse_lump(self, lump_class: Type[Lump], lump_id, lump_name):
        if self.lumps_info[lump_id].size != 0:
            lump = self.lumps_info[lump_id]
            parsed_lump = lump_class(self, lump_id).parse()
            self.lumps[lump_id] = parsed_lump
            return parsed_lump


class RespawnBSPFile(BSPFile):

    def __init__(self, filepath: str):
        super().__init__(filepath)

    def parse(self):
        reader = self.reader
        magic = reader.read_fourcc()
        assert magic == 'rBSP'
        self.version = reader.read_uint32()
        self.revision = reader.read_uint32()
        last_lump = reader.read_uint32()

        for lump_id in range(last_lump + 1):
            lump = LumpInfo(lump_id)
            lump.parse(reader, False)
            self.lumps_info.append(lump)
