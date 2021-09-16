from pathlib import Path
from typing import Dict, Union

from .content_provider_base import ContentProviderBase
from ..source_shared.app_id import SteamAppId


class HLAAddonProvider(ContentProviderBase):

    @property
    def steam_id(self):
        return SteamAppId.HLA_STEAM_ID

    def find_file(self, filepath: Union[str, Path], additional_dir=None, extension=None):
        return self._find_file_generic(filepath, additional_dir, extension)

    def find_path(self, filepath: Union[str, Path], additional_dir=None, extension=None):
        return self._find_path_generic(filepath, additional_dir, extension)

    def glob(self, pattern: str):
        return self._glob_generic(pattern)
