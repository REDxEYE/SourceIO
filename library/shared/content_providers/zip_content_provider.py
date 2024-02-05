import fnmatch
from pathlib import Path
from typing import Iterator, Optional
from zipfile import ZipFile

from ...utils import Buffer, MemoryBuffer
from .content_provider_base import ContentProviderBase


class ZIPContentProvider(ContentProviderBase):
    def __init__(self, filepath: Path):
        super().__init__(filepath)
        self._zip_file = ZipFile(filepath)
        self._cache = {k.replace("\\", "/").lower(): k for k in self._zip_file.namelist()}

    def find_file(self, filepath: Path) -> Optional[Buffer]:
        if filepath.as_posix().lower() in self._cache:
            return MemoryBuffer(self._zip_file.read(self._cache[filepath.as_posix()]))

    def find_path(self, filepath: Path) -> Optional[Path]:
        if filepath.as_posix().lower() in self._cache:
            return Path(self.filepath.as_posix() + ":" + filepath.as_posix())

    def glob(self, pattern: str) -> Iterator[tuple[Path, Buffer]]:
        matches = fnmatch.filter(self._cache.keys(), pattern)
        for match in matches:
            yield Path(match), MemoryBuffer(self._zip_file.read(self._cache[match]))
