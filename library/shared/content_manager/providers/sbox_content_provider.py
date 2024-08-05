from pathlib import Path
from typing import Iterator, Optional, Union

from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager.provider import ContentProvider, find_file_generic, \
    find_path_generic, glob_generic
from SourceIO.library.utils import Buffer


class SBoxAddonProvider(ContentProvider):

    @property
    def steam_id(self):
        return SteamAppId.SBOX_STEAM_ID

    def find_file(self, filepath: Path) -> Optional[Buffer]:
        return find_file_generic(self.root, filepath)

    def find_path(self, filepath: Path) -> Optional[Path]:
        return find_path_generic(self.root, filepath)

    def glob(self, pattern: str) -> Iterator[tuple[Path, Buffer]]:
        yield from glob_generic(self.root, pattern)


class SBoxDownloadsProvider(ContentProvider):

    @property
    def steam_id(self):
        return SteamAppId.SBOX_STEAM_ID

    def find_file(self, filepath: Path) -> Optional[Buffer]:
        return find_file_generic(self.root, filepath)

    def find_path(self, filepath: Path) -> Optional[Path]:
        return find_path_generic(self.root, filepath)

    def glob(self, pattern: str) -> Iterator[tuple[Path, Buffer]]:
        yield from glob_generic(self.root, pattern)