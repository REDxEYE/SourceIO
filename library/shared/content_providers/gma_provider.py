from io import BytesIO
from pathlib import Path
from typing import Union
import glob
from ...source1.gma import open_gma
from .content_provider_base import ContentProviderBase


class GMAContentProvider(ContentProviderBase):
    def __init__(self, filepath: Path, override_steamid=0):
        super().__init__(filepath)
        self._override_steamid = override_steamid
        self.gma_archive = open_gma(filepath)

    def glob(self, pattern: str):
        files = []
        for file_name, entry in self.gma_archive.file_entries.items():
            if glob.fnmatch.fnmatch(file_name, pattern):
                files.append((file_name, BytesIO(self.gma_archive.find_file(file_name))))
        return files

    def find_file(self, filepath: Union[str, Path]):
        cached_file = self.get_from_cache(filepath)
        if cached_file:
            return cached_file

        entry = self.gma_archive.find_file(filename=filepath)
        if entry:
            return entry

    def find_path(self, filepath: Union[str, Path]):
        entry = self.gma_archive.find_file(filename=filepath)
        if entry:
            return None
            # raise NotImplementedError('Cannot get path from VPK file')

    @property
    def steam_id(self):
        return self._override_steamid or super().steam_id
