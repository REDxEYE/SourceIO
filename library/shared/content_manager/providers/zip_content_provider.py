import fnmatch
from typing import Iterator, Optional
from zipfile import ZipFile

from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.utils import Buffer, MemoryBuffer, TinyPath


class ZIPContentProvider(ContentProvider):
    def __init__(self, filepath: TinyPath, steamapp_id: SteamAppId = SteamAppId.UNKNOWN):
        super().__init__(filepath)
        self._steamapp_id = steamapp_id
        self._zip_file = ZipFile(filepath)
        self._cache = {k.replace("\\", "/").lower(): k for k in self._zip_file.namelist()}

    def check(self, filepath: TinyPath) -> bool:
        return filepath.as_posix().lower() in self._cache

    def get_relative_path(self, filepath: TinyPath) -> TinyPath | None:
        return None

    def get_provider_from_path(self, filepath) -> Optional['ContentProvider']:
        if self.check(filepath):
            return self

    def get_steamid_from_asset(self, asset_path: TinyPath) -> SteamAppId | None:
        if self.check(asset_path):
            return self.steam_id

    def find_file(self, filepath: TinyPath) -> Optional[Buffer]:
        if filepath.as_posix().lower() in self._cache:
            return MemoryBuffer(self._zip_file.read(self._cache[filepath.as_posix().lower()]))

    def glob(self, pattern: str) -> Iterator[tuple[TinyPath, Buffer]]:
        matches = fnmatch.filter(self._cache.keys(), pattern)
        for match in matches:
            yield TinyPath(match), MemoryBuffer(self._zip_file.read(self._cache[match]))

    @property
    def root(self) -> TinyPath:
        return self.filepath.parent

    @property
    def name(self) -> str:
        return self.filepath.stem

    @property
    def steam_id(self) -> SteamAppId:
        return self._steamapp_id
