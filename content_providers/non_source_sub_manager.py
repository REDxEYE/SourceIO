from pathlib import Path
from typing import Union

from ..utilities.path_utilities import backwalk_file_resolver
from .content_provider_base import ContentProviderBase


class NonSourceContentProvider(ContentProviderBase):
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
