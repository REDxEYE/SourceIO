from typing import Iterator, Optional

from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager.provider import ContentProvider, find_file_generic, \
    glob_generic, is_relative_to
from SourceIO.library.shared.content_manager.providers.loose_files import LooseFilesContentProvider
from SourceIO.library.utils import Buffer
from SourceIO.library.utils.tiny_path import TinyPath


class SBoxAddonProvider(LooseFilesContentProvider):

    @property
    def name(self) -> str:
        return f"S&Box {self.filepath.stem}"

    @property
    def steam_id(self):
        return SteamAppId.SBOX_STEAM_ID


class SBoxDownloadsProvider(LooseFilesContentProvider):

    @property
    def name(self) -> str:
        return f"S&Box {self.filepath.stem[:8]}"

    @property
    def steam_id(self):
        return SteamAppId.SBOX_STEAM_ID
