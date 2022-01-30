from pathlib import Path
from typing import Union

from ...utils.path_utilities import backwalk_file_resolver
from .content_provider_base import ContentProviderBase


class NonSourceContentProvider(ContentProviderBase):
    def __init__(self, filepath: Path, override_steamid=0):
        self._override_steamid = override_steamid
        super().__init__(filepath)

    def find_file(self, filepath: Union[str, Path]):
        file = backwalk_file_resolver(self.filepath, filepath)
        if file:
            return file.open('rb')

    def find_path(self, filepath: Union[str, Path]):
        file = backwalk_file_resolver(self.filepath, filepath)
        if file:
            return file

    def glob(self, pattern: str):
        yield from self._glob_generic(pattern)

    @property
    def steam_id(self):
        return self._override_steamid or super().steam_id
