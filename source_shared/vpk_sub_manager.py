from pathlib import Path

from .vpk.vpk_file import open_vpk
from ..source_shared.vpk import VPKFile
from ..source_shared.content_provider_base import ContentProviderBase


class VPKContentProvider(ContentProviderBase):
    def __init__(self, filepath: Path):
        super().__init__(filepath)
        self.vpk_archive = open_vpk(filepath)
        self.vpk_archive.read()

    def find_file(self, filepath: str):
        entry = self.vpk_archive.find_file(full_path=filepath)
        if entry:
            return self.vpk_archive.read_file(entry)

    def find_path(self, filepath: str):
        entry = self.vpk_archive.find_file(full_path=filepath)
        if entry:
            return None
            # raise NotImplementedError('Cannot get path from VPK file')
