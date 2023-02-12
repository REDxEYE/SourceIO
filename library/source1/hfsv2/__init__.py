from pathlib import Path
from typing import Dict

from ...utils import FileBuffer
from ...utils.singleton import SingletonMeta
from .archive import Archive
from .file import File

# Based on yretenai code from https://github.com/yretenai/HFSExtract

class HFSv2(metaclass=SingletonMeta):

    def __init__(self, hfs_root: Path):
        self.files: Dict[str, Archive] = {}
        self._archives: Dict[str, Archive] = {}
        for hfs_file in hfs_root.iterdir():
            if hfs_file.stem in self._archives:
                continue
            buffer = FileBuffer(hfs_file)
            archive = Archive(hfs_file.name)
            archive.read(buffer)
            self._archives[hfs_file.stem] = archive
            self.files.update({k: archive for k in archive.files.keys()})

    def get_file(self, path):
        path = Path(path).as_posix().lower()
        if path in self.files:
            return self.files[path].get_file(path)
        return None

    def has_file(self, path):
        path = Path(path).as_posix()
        return path in self.files
