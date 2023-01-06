import fnmatch
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

from ...shared.vpk.vpk_file import open_vpk
from ...utils import Buffer
from ..app_id import SteamAppId
from .content_provider_base import ContentProviderBase


class VPKContentProvider(ContentProviderBase):
    def __init__(self, filepath: Path, override_steamid=0):
        super().__init__(filepath)
        self._override_steamid = override_steamid
        self.vpk_archive = open_vpk(filepath)
        self.vpk_archive.read()

    def glob(self, pattern: str) -> Iterator[Tuple[Path, Buffer]]:
        files = []
        for file_name, entry in self.vpk_archive.entries.items():
            if fnmatch.fnmatch(file_name, pattern):
                files.append((file_name, self.vpk_archive.read_file(entry)))
        return files

    def find_file(self, filepath: Union[str, Path]) -> Optional[Buffer]:
        cached_file = self.get_from_cache(filepath)
        if cached_file:
            return cached_file

        entry = self.vpk_archive.find_file(full_path=filepath)
        if entry:
            file = self.vpk_archive.read_file(entry)
            return self.cache_file(filepath, file)

    def find_path(self, filepath: Union[str, Path]) -> Optional[Path]:
        entry = self.vpk_archive.find_file(full_path=filepath)
        if entry:
            return Path(self.filepath.as_posix() + ":" + filepath.as_posix())

    @property
    def steam_id(self) -> SteamAppId:
        return self._override_steamid or super().steam_id
