from typing import Iterator, Optional

from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.utils import Buffer, MemoryBuffer, TinyPath
from SourceIO.library.utils.rustlib import Vpk
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('VpkProvider')


class VPKContentProvider(ContentProvider):
    def __init__(self, filepath: TinyPath, override_steamid=SteamAppId.UNKNOWN):
        super().__init__(filepath)
        self._override_steamid = override_steamid
        self._initialized = False
        self.vpk_archive: Vpk | None = None

    def check(self, filepath: TinyPath) -> bool:
        self._init()
        return self.vpk_archive.find_file(filepath) is not None

    def get_relative_path(self, filepath: TinyPath) -> TinyPath | None:
        return None

    def get_provider_from_path(self, filepath) -> Optional['ContentProvider']:
        if self.check(filepath):
            return self
        return None

    def get_steamid_from_asset(self, asset_path: TinyPath) -> SteamAppId | None:
        if self.check(asset_path):
            return self.steam_id

    def _init(self):
        if self._initialized:
            return
        self.vpk_archive = Vpk.from_path(self.filepath)
        self._initialized = True

    def glob(self, pattern: str) -> Iterator[tuple[TinyPath, Buffer]]:
        self._init()
        for key, data in self.vpk_archive.glob(pattern):
            yield key, MemoryBuffer(data)

    def find_file(self, filepath: TinyPath) -> Optional[Buffer]:
        self._init()
        file = self.vpk_archive.find_file(filepath)
        if file:
            return MemoryBuffer(file)

    @property
    def root(self) -> TinyPath:
        return self.filepath.parent

    @property
    def name(self) -> str:
        return self.filepath.stem

    @property
    def steam_id(self) -> SteamAppId:
        return self._override_steamid
