from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.source1.bsp.lump import Quake3LumpInfo, LumpTag, ValveLumpInfo, Lump, AbstractLump
from SourceIO.library.utils import Buffer, FileBuffer
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()

logger = log_manager.get_logger("BSP")
LumpType = TypeVar("LumpType", bound=Lump)


def open_bsp(filepath: TinyPath, buffer: Buffer, content_manager: ContentManager,
             override_steamappid: Optional[SteamAppId] = None) -> Optional['VBSPFile']:
    magic, version = buffer.read_fmt('4sI')
    buffer.seek(0)
    if magic == b'VBSP':
        return VBSPFile.from_buffer(filepath, buffer, content_manager, override_steamappid)
    elif magic == b'rBSP':
        return RespawnBSPFile.from_buffer(filepath, buffer, content_manager, override_steamappid)
    elif magic == b'RBSP':
        return RavenBSPFile.from_buffer(filepath, buffer, content_manager, override_steamappid)
    elif magic == b'IBSP':
        return IBSPFile.from_buffer(filepath, buffer, content_manager, override_steamappid)
    logger.error("Unrecognized map magic number: {}".format(magic))
    return None


@dataclass(slots=True)
class BSPInfo:
    ident: str
    version: tuple[int, int]
    lumps: list[AbstractLump]
    revision: int
    steam_app_id: SteamAppId


@dataclass
class BSPFile:
    info: BSPInfo
    filepath: TinyPath
    buffer: Buffer
    lump_cache: dict[str, Lump] = field(default_factory=dict, init=False)

    def get_lump(self, lump_name) -> LumpType | None:
        info = self.info
        if lump_name in self.lump_cache:
            return self.lump_cache[lump_name]
        else:
            matches: list[tuple[Type[Lump], LumpTag]] = []
            for sub in Lump.all_subclasses():
                sub: Type[Lump]
                for dep in sub.tags:
                    if dep.lump_id >= len(info.lumps):
                        continue
                    if dep.lump_name == lump_name:
                        if dep.bsp_ident is not None and dep.bsp_ident != info.ident:
                            continue
                        if dep.bsp_version is not None and dep.bsp_version > info.version:
                            continue
                        if dep.steam_id is not None and dep.steam_id != info.steam_app_id:
                            continue
                        if dep.lump_version is not None and dep.lump_version != info.lumps[dep.lump_id].version:
                            continue
                        matches.append((sub, dep))
            best_matches = []
            for match_sub, match_dep in matches:
                if match_dep.lump_id >= len(info.lumps):
                    continue
                lump = info.lumps[match_dep.lump_id]
                rank = 0
                if match_dep.bsp_version is not None and match_dep.bsp_version == info.version:
                    rank += 2
                elif match_dep.bsp_version is not None and match_dep.bsp_version > info.version:
                    rank += 1
                if match_dep.bsp_ident is not None and match_dep.bsp_ident == info.ident:
                    rank += 4
                if match_dep.steam_id is not None and match_dep.steam_id == info.steam_app_id:
                    rank += 1
                if match_dep.lump_version is not None and match_dep.lump_version == lump.version:
                    rank += 1
                best_matches.append((rank, match_sub, match_dep))
            if not best_matches:
                return None
            best_matches = list(sorted(best_matches, key=lambda a: a[0]))
            _, sub, dep = best_matches[-1]

            parsed_lump = self.parse_lump(sub, dep.lump_id, dep.lump_name)
            self.lump_cache[lump_name] = parsed_lump
            return parsed_lump

    def parse_lump(self, lump_class: Type[Lump], lump_id, lump_name):
        info = self.info
        base_path = self.filepath.parent
        lump_path = base_path / f'{self.filepath.stem}_l_{lump_id}.lmp'
        if lump_path.exists():
            buffer = FileBuffer(lump_path)
            lump_info = ValveLumpInfo.from_buffer(buffer, lump_id, self.is_l4d2)
            info.lumps[lump_id] = lump_info
            buffer.seek(lump_info.offset)

            parsed_lump = lump_class(lump_info).parse(buffer, self)
            self.lump_cache[lump_id] = parsed_lump
            return parsed_lump

        if info.lumps[lump_id].size != 0:
            lump_info = info.lumps[lump_id]
            buffer = self._get_lump_buffer(lump_id, lump_info)

            parsed_lump = lump_class(lump_info).parse(buffer, self)
            self.lump_cache[lump_id] = parsed_lump
            return parsed_lump
        return None

    def _get_lump_buffer(self, lump_id: int, lump_info: AbstractLump) -> Buffer:
        base_path = self.filepath.parent
        lump_path = base_path / f'{self.filepath.name}.{lump_id:04x}.bsp_lump'

        if lump_path.exists():
            return FileBuffer(lump_path)

        return self.buffer.slice(lump_info.offset, lump_info.size)


@dataclass(slots=True)
class VBSPFile(BSPFile):
    is_l4d2: bool

    @classmethod
    def from_buffer(cls, filepath: TinyPath, buffer: Buffer, content_manager: ContentManager,
                    override_steamappid: Optional[SteamAppId] = None):
        magic = buffer.read_fourcc()
        assert magic == "VBSP", "Invalid BSP header"
        version = buffer.read_int32()
        if version > 0xFFFF:
            version = version & 0xFFFF, version >> 16
        else:
            version = (version, 0)
        is_l4d2 = buffer.peek_uint32() <= 1036 and version == (21, 0)
        # noinspection PyTypeChecker
        lumps_info: list[AbstractLump] = [None] * 64  # Just pre-allocating array
        for lump_id in range(64):
            lump = ValveLumpInfo.from_buffer(buffer, lump_id, is_l4d2)
            lumps_info[lump_id] = lump
        revision = buffer.read_int32()
        steam_app_id = override_steamappid or content_manager.get_steamid_from_asset(filepath)
        return cls(BSPInfo(magic, version, lumps_info, revision, steam_app_id), filepath, buffer, is_l4d2)

    def _get_lump_buffer(self, lump_id: int, lump_info: ValveLumpInfo) -> Buffer:
        base_path = self.filepath.parent
        lump_path = base_path / f'{self.filepath.name}.{lump_id:04x}.bsp_lump'

        if lump_path.exists():
            return FileBuffer(lump_path)

        if not lump_info.compressed:
            return self.buffer.slice(lump_info.offset, lump_info.size)
        else:
            assert isinstance(lump_info, ValveLumpInfo)
            buffer = Lump.decompress_lump(self.buffer.slice(lump_info.offset, lump_info.size))
            assert buffer.size() == lump_info.decompressed_size
            return buffer


class RespawnBSPFile(VBSPFile):

    @classmethod
    def from_buffer(cls, filepath: TinyPath, buffer: Buffer, content_manager: ContentManager,
                    override_steamappid: Optional[SteamAppId] = None):
        magic = buffer.read_fourcc()
        assert magic == "rBSP", "Invalid BSP header"
        version = buffer.read_uint32()
        revision = buffer.read_uint32()
        last_lump = buffer.read_uint32()
        lumps_info = [None] * last_lump
        for lump_id in range(last_lump + 1):
            lump = ValveLumpInfo.from_buffer(buffer, lump_id)
            lump.id = lump_id
            lumps_info[lump_id] = lump
        steam_app_id = override_steamappid or content_manager.get_steamid_from_asset(filepath)
        return cls(BSPInfo(magic, (version, 0), lumps_info, revision, steam_app_id), filepath, buffer, False)


class IBSPFile(BSPFile):
    @classmethod
    def from_buffer(cls, filepath: TinyPath, buffer: Buffer, content_manager: ContentManager,
                    override_steamappid: Optional[SteamAppId] = None):
        magic = buffer.read_fourcc()
        assert magic == "IBSP", "Invalid BSP header"
        version = (buffer.read_int32(), 0)
        lumps_info = []
        for lump_id in range(17):
            lump = Quake3LumpInfo.from_buffer(buffer, lump_id)
            lumps_info.append(lump)

        detected_steam_app_id = content_manager.get_steamid_from_asset(filepath)
        if detected_steam_app_id == SteamAppId.UNKNOWN:  # Set to none if unknown to let the default work
            detected_steam_app_id = None
        steam_app_id: SteamAppId = override_steamappid or detected_steam_app_id or SteamAppId.QUAKE3
        return cls(BSPInfo(magic, version, lumps_info, 0, steam_app_id), filepath, buffer)


class RavenBSPFile(IBSPFile):
    def __init__(self, filepath: TinyPath, buffer: Buffer, content_manager: ContentManager):
        super().__init__(filepath, buffer, content_manager)

    @classmethod
    def from_buffer(cls, filepath: TinyPath, buffer: Buffer, content_manager: ContentManager,
                    override_steamappid: Optional[SteamAppId] = None):
        magic = buffer.read_fourcc()
        assert magic == "RBSP", "Invalid BSP header"
        version = (buffer.read_int32(), 0)
        lumps_info = []
        for lump_id in range(18):
            lump = Quake3LumpInfo.from_buffer(buffer, lump_id)
            lumps_info.append(lump)
        detected_steam_app_id = content_manager.get_steamid_from_asset(filepath)
        if detected_steam_app_id == SteamAppId.UNKNOWN: # Set to none if unknown to let the default work
            detected_steam_app_id = None
        steam_app_id: SteamAppId = override_steamappid or detected_steam_app_id or SteamAppId.RAVEN_Q3_ENGINE
        return cls(BSPInfo(magic, version, lumps_info, 0, steam_app_id), filepath, buffer)
