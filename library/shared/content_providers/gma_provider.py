import fnmatch
from pathlib import Path
from typing import Union, Iterator, Tuple, List, Optional

from ..app_id import SteamAppId
from ...source1.gma import open_gma
from .content_provider_base import ContentProviderBase
from ...utils import Buffer


class GMAContentProvider(ContentProviderBase):
    def __init__(self, filepath: Path, override_steamid=0):
        super().__init__(filepath)
        self._override_steamid = override_steamid
        self.gma_archive = open_gma(filepath)

    def glob(self, pattern: str) -> Iterator[Tuple[Path, Buffer]]:
        files: List[Tuple[Path, Buffer]] = []
        for file_name, entry in self.gma_archive.file_entries.items():
            if fnmatch.fnmatch(file_name, pattern):
                files.append((file_name, self.gma_archive.find_file(file_name)))
        return iter(files)

    def find_file(self, filepath: Union[str, Path]) -> Optional[Buffer]:
        cached_file = self.get_from_cache(filepath)
        if cached_file:
            return cached_file

        entry = self.gma_archive.find_file(filename=filepath)
        if entry:
            return entry

    def find_path(self, filepath: Union[str, Path]):
        entry = self.gma_archive.find_file(filename=filepath)
        if entry:
            return Path(self.filepath.as_posix() + ":" + filepath.as_posix())

    @property
    def steam_id(self) -> SteamAppId:
        return self._override_steamid or SteamAppId.GARRYS_MOD
