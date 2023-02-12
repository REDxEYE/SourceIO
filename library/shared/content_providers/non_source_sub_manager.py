from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

from ...utils import Buffer, FileBuffer
from ...utils.path_utilities import backwalk_file_resolver
from ..app_id import SteamAppId
from .content_provider_base import ContentProviderBase


class NonSourceContentProvider(ContentProviderBase):
    def __init__(self, filepath: Path, override_steamid=0):
        self._override_steamid = override_steamid
        super().__init__(filepath)

    def find_file(self, filepath: Union[str, Path]) -> Optional[Buffer]:
        file = backwalk_file_resolver(self.filepath, filepath)
        if file and file.is_file():
            return FileBuffer(file)

    def find_path(self, filepath: Union[str, Path]) -> Optional[Path]:
        file = backwalk_file_resolver(self.filepath, filepath)
        if file:
            return file

    def glob(self, pattern: str) -> Iterator[Tuple[Path, Buffer]]:
        yield from self._glob_generic(pattern)

    @property
    def steam_id(self) -> SteamAppId:
        return self._override_steamid or SteamAppId.UNKNOWN
