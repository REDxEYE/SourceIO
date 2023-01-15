import fnmatch
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

from ...shared.app_id import SteamAppId
from ...source1.hfsv1 import HFS
from ...source1.hfsv2 import HFSv2
from ...utils import Buffer
from .content_provider_base import ContentProviderBase


class HFS2ContentProvider(ContentProviderBase):
    def __init__(self, root_path: Path):
        if root_path.is_file():
            root_path = root_path.parent
        super().__init__(root_path)
        self.hfs_archive = HFSv2(root_path)

    def find_file(self, filepath: Union[str, Path]) -> Optional[Buffer]:
        file = self.hfs_archive.get_file(filepath)
        if file:
            return file

    def find_path(self, filepath: Union[str, Path]) -> Optional[Path]:
        if self.hfs_archive.has_file(filepath):
            return filepath

    def glob(self, pattern: str) -> Iterator[Tuple[Path, Buffer]]:
        for file_name in self.hfs_archive.files.keys():
            if fnmatch.fnmatch(file_name, pattern):
                yield file_name, self.hfs_archive.get_file(file_name)

    @property
    def steam_id(self) -> SteamAppId:
        return SteamAppId.VINDICTUS


class HFS1ContentProvider(ContentProviderBase):
    def __init__(self, root_path: Path):
        super().__init__(root_path)
        self.hfs_archive = HFS(root_path)

    def find_file(self, filepath: Path):
        file = self.hfs_archive.get_file(filepath)
        if file:
            return file

    def find_path(self, filepath: Path):
        if self.hfs_archive.has_file(filepath):
            return filepath

    def glob(self, pattern: str):
        for file_name in self.hfs_archive.entries.keys():
            if fnmatch.fnmatch(file_name, pattern):
                yield file_name, self.hfs_archive.get_file(file_name)

    @property
    def steam_id(self):
        return SteamAppId.VINDICTUS
