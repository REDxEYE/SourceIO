from pathlib import Path

from ..source_shared.vpk import VPKFile
from ..source_shared.sub_manager import SubManager


class VPKSubManager(SubManager):
    def __init__(self, filepath: Path):
        super().__init__(filepath)
        self.vpk_archive = VPKFile(filepath)
        self.vpk_archive.read()

    def find_file(self, filepath: str):
        entry = self.vpk_archive.find_file(full_path=filepath)
        if entry:
            return self.vpk_archive.read_file(entry)
