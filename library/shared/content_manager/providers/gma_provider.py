import fnmatch
from pathlib import Path
from typing import Iterator, Optional

from SourceIO.library.archives.gma import open_gma
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.utils import Buffer
from SourceIO.library.shared.app_id import SteamAppId


class GMAContentProvider(ContentProvider):
    def check(self, filepath: Path) -> bool:
        self._init()
        return self.gma_archive.has_file(filepath)

    def get_relative_path(self, filepath: Path) -> Path | None:
        return None

    def get_provider_from_path(self, filepath) -> Optional['ContentProvider']:
        if self.check(filepath):
            return self

    def get_steamid_from_asset(self, asset_path: Path) -> SteamAppId | None:
        if self.check(asset_path):
            return self.steam_id

    @property
    def root(self) -> Path:
        return self.filepath.parent

    @property
    def name(self) -> str:
        return self.filepath.stem

    def __init__(self, filepath: Path, override_steamid=0):
        super().__init__(filepath)
        self._override_steamid = override_steamid
        self._initialized = False
        self.gma_archive = None

    def _init(self):
        if self._initialized:
            return
        self.gma_archive = open_gma(self.filepath)

    def glob(self, pattern: str) -> Iterator[tuple[Path, Buffer]]:
        self._init()
        files: list[tuple[Path, Buffer]] = []
        for file_name, entry in self.gma_archive.file_entries.items():
            if fnmatch.fnmatch(file_name, pattern):
                files.append((file_name, self.gma_archive.find_file(file_name, )))
        return iter(files)

    def find_file(self, filepath: Path) -> Optional[Buffer]:
        self._init()
        entry = self.gma_archive.find_file(filepath)
        if entry:
            return entry

    @property
    def steam_id(self) -> SteamAppId:
        return self._override_steamid or SteamAppId.GARRYS_MOD
