from pathlib import Path
from typing import Union

from ..source1.hfsv1 import HFS
from ..source1.hfsv2 import HFSv2
from .content_provider_base import ContentProviderBase
from ..source_shared.app_id import SteamAppId


class HFS2ContentProvider(ContentProviderBase):
    def __init__(self, root_path: Path):
        if root_path.is_file():
            root_path = root_path.parent
        super().__init__(root_path)
        self.hfs_archive = HFSv2(root_path)

    def find_file(self, filepath: Union[str, Path]):
        cached_file = self.get_from_cache(filepath)
        if cached_file:
            return cached_file

        file = self.hfs_archive.get_file(filepath)
        if file:
            return self.cache_file(filepath, file)

    def find_path(self, filepath: Union[str, Path]):
        if self.hfs_archive.has_file(filepath):
            return filepath

    @property
    def steam_id(self):
        return SteamAppId.VINDICTUS


class HFS1ContentProvider(ContentProviderBase):
    def __init__(self, root_path: Path):
        super().__init__(root_path)
        self.hfs_archive = HFS(root_path)

    def find_file(self, filepath: Union[str, Path]):
        cached_file = self.get_from_cache(filepath)
        if cached_file:
            return cached_file

        file = self.hfs_archive.get_file(filepath)
        if file:
            return self.cache_file(filepath, file)

    def find_path(self, filepath: Union[str, Path]):
        if self.hfs_archive.has_file(filepath):
            return filepath

    @property
    def steam_id(self):
        return SteamAppId.VINDICTUS
