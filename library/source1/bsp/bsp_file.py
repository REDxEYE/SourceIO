from typing import Optional, Type

from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.source1.bsp.lump import RavenLumpInfo, LumpTag, LumpInfo, Lump
from SourceIO.library.utils import Buffer, FileBuffer
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()

logger = log_manager.get_logger("BSP")


def open_bsp(filepath: TinyPath, buffer: Buffer, content_manager: ContentManager,
             override_steamappid: Optional[SteamAppId] = None) -> Optional['BSPFile']:
    magic, version = buffer.read_fmt('4sI')
    buffer.seek(0)
    if magic == b'VBSP':
        return BSPFile.from_buffer(filepath, buffer, content_manager, override_steamappid)
    elif magic == b'rBSP':
        return RespawnBSPFile.from_buffer(filepath, buffer, content_manager, override_steamappid)
    elif magic == b'RBSP':
        return RavenBSPFile.from_buffer(filepath, buffer, content_manager, override_steamappid)
    logger.error("Unrecognized map magic number: {}".format(magic))
    return None


class BSPFile:
    def __init__(self, filepath: TinyPath, buffer: Buffer, content_manager: ContentManager):
        self.filepath = TinyPath(filepath)
        self.buffer = buffer
        self.version = 0
        self.is_l4d2 = False
        self.lumps_info: list[LumpInfo] = []
        self.lumps: dict[str, Lump] = {}
        self.revision = 0
        self.content_manager = content_manager
        self.steam_app_id = content_manager.get_steamid_from_asset(filepath) or SteamAppId.UNKNOWN

    @classmethod
    def from_buffer(cls, filepath: TinyPath, buffer: Buffer, content_manager: ContentManager,
                    override_steamappid: Optional[SteamAppId] = None):
        self = cls(filepath, buffer, content_manager)
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
        self.steam_app_id = override_steamappid or self.steam_app_id
        return self

    def get_lump(self, lump_name):
        if lump_name in self.lumps:
            return self.lumps[lump_name]
        else:
            matches: list[tuple[Type[Lump], LumpTag]] = []
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
                if match_dep.lump_id >= len(self.lumps_info):
                    continue
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

    def __init__(self, filepath: TinyPath, buffer: Buffer, content_manager: ContentManager):
        super().__init__(filepath, buffer, content_manager)

    @classmethod
    def from_buffer(cls, filepath: TinyPath, buffer: Buffer, content_manager: ContentManager,
                    override_steamappid: Optional[SteamAppId] = None):
        self = cls(filepath, buffer, content_manager)
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
        self.steam_app_id = override_steamappid or self.steam_app_id
        return self


class RavenBSPFile(BSPFile):
    def __init__(self, filepath: TinyPath, buffer: Buffer, content_manager: ContentManager):
        super().__init__(filepath, buffer, content_manager)

    @classmethod
    def from_buffer(cls, filepath: TinyPath, buffer: Buffer, content_manager: ContentManager,
                    override_steamappid: Optional[SteamAppId] = None):
        self = cls(filepath, buffer, content_manager)
        magic = buffer.read_fourcc()
        assert magic == "RBSP", "Invalid BSP header"
        self.version = (buffer.read_int32(), 0)
        self.lumps_info = []
        for lump_id in range(18):
            lump = RavenLumpInfo.from_buffer(buffer, lump_id, False)
            self.lumps_info.append(lump)
        self.steam_app_id = override_steamappid or SteamAppId.SOLDIERS_OF_FORTUNE2
        return self
