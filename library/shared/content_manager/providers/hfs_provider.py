import fnmatch
from pathlib import Path
from typing import Iterator, Optional, Union

from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.archives.hfsv1 import HFS
from SourceIO.library.archives.hfsv2 import HFSv2
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.utils import Buffer


class HFS2ContentProvider(ContentProvider):

    def __init__(self, root_path: Path):
        if root_path.is_file():
            root_path = root_path.parent
        super().__init__(root_path)
        self.hfs_archive = HFSv2(root_path)

    def check(self, filepath: Path) -> bool:
        return self.hfs_archive.has_file(filepath)

    def get_relative_path(self, filepath: Path) -> Path | None:
        return None

    def get_provider_from_path(self, filepath) -> Optional['ContentProvider']:
        return self if self.hfs_archive.get_file(filepath) else None

    def get_steamid_from_asset(self, asset_path: Path) -> SteamAppId | None:
        return self.steam_id if self.hfs_archive.has_file(asset_path) else None

    @property
    def root(self) -> Path:
        return self.filepath.parent

    @property
    def name(self) -> str:
        return self.filepath.stem

    def find_file(self, filepath: Union[str, Path]) -> Optional[Buffer]:
        file = self.hfs_archive.get_file(filepath)
        if file:
            return file

    def glob(self, pattern: str) -> Iterator[tuple[Path, Buffer]]:
        for file_name in self.hfs_archive.files.keys():
            if fnmatch.fnmatch(file_name, pattern):
                yield file_name, self.hfs_archive.get_file(file_name)

    @property
    def steam_id(self) -> SteamAppId:
        return SteamAppId.VINDICTUS


class HFS1ContentProvider(ContentProvider):
    def __init__(self, root_path: Path):
        super().__init__(root_path)
        self.hfs_archive = HFS(root_path)

    def check(self, filepath: Path) -> bool:
        return self.hfs_archive.has_file(filepath)

    def get_relative_path(self, filepath: Path) -> Path | None:
        return None

    def get_provider_from_path(self, filepath) -> Optional['ContentProvider']:
        return self if self.hfs_archive.get_file(filepath) else None

    def get_steamid_from_asset(self, asset_path: Path) -> SteamAppId | None:
        return self.steam_id if self.hfs_archive.has_file(asset_path) else None

    @property
    def root(self) -> Path:
        return self.filepath.parent

    @property
    def name(self) -> str:
        return self.filepath.stem

    def find_file(self, filepath: Path):
        file = self.hfs_archive.get_file(filepath)
        if file:
            return file

    def glob(self, pattern: str):
        for file_name in self.hfs_archive.entries.keys():
            if fnmatch.fnmatch(file_name, pattern):
                yield file_name, self.hfs_archive.get_file(file_name)

    @property
    def steam_id(self):
        return SteamAppId.VINDICTUS
