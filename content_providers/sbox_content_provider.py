from pathlib import Path
from typing import Dict, Union

from .content_provider_base import ContentProviderBase, ContentDetectorBase
from .source2_content_provider import GameinfoContentProvider
from ..source_shared.app_id import SteamAppId


class SBoxAddonProvider(ContentProviderBase):

    @property
    def steam_id(self):
        return SteamAppId.SBOX_STEAM_ID

    def find_file(self, filepath: Union[str, Path], additional_dir=None, extension=None):
        return self._find_file_generic(filepath, additional_dir, extension)

    def find_path(self, filepath: Union[str, Path], additional_dir=None, extension=None):
        return self._find_path_generic(filepath, additional_dir, extension)

    def glob(self, pattern: str):
        yield from self._glob_generic(pattern)


class SBoxDownloadsProvider(ContentProviderBase):

    @property
    def steam_id(self):
        return SteamAppId.SBOX_STEAM_ID

    def find_file(self, filepath: Union[str, Path], additional_dir=None, extension=None):
        return self._find_file_generic(filepath, additional_dir, extension)

    def find_path(self, filepath: Union[str, Path], additional_dir=None, extension=None):
        return self._find_path_generic(filepath, additional_dir, extension)

    def glob(self, pattern: str):
        yield from self._glob_generic(pattern)
