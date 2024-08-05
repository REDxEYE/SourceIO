from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager.provider import ContentProvider, glob_generic, find_path_generic, \
    find_file_generic
from SourceIO.library.utils import Buffer


class HLAAddonProvider(ContentProvider):

    @property
    def steam_id(self) -> SteamAppId:
        return SteamAppId.HLA_STEAM_ID

    def find_file(self, filepath: Path) -> Optional[Buffer]:
        return find_file_generic(self.root, filepath)

    def find_path(self, filepath: Path) -> Optional[Path]:
        return find_path_generic(self.root, filepath)

    def glob(self, pattern: str) -> Iterator[tuple[Path, Buffer]]:
        yield from glob_generic(self.root, pattern)
