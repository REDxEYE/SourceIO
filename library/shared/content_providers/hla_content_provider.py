from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

from ...shared.app_id import SteamAppId
from ...utils import Buffer
from .content_provider_base import ContentProviderBase


class HLAAddonProvider(ContentProviderBase):

    @property
    def steam_id(self) -> SteamAppId:
        return SteamAppId.HLA_STEAM_ID

    def find_file(self, filepath: Union[str, Path], additional_dir=None, extension=None) -> Optional[Buffer]:
        return self._find_file_generic(filepath, additional_dir, extension)

    def find_path(self, filepath: Union[str, Path], additional_dir=None, extension=None) -> Optional[Path]:
        return self._find_path_generic(filepath, additional_dir, extension)

    def glob(self, pattern: str) -> Iterator[Tuple[Path, Buffer]]:
        yield from self._glob_generic(pattern)
