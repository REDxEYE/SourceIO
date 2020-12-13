from pathlib import Path

from ..source1.sub_manager import SubManager
from ..source_shared.vpk import VPKFile


class VPKSubManager(SubManager):
    def __init__(self, filepath: Path):
        super().__init__(filepath)
        self.vpk_archive = VPKFile(filepath)
        self.vpk_archive.read()

    def find_file(self, filepath: str):
        entry = self.vpk_archive.find_file(full_path=str(filepath))
        if entry:
            return self.vpk_archive.read_file(entry)
