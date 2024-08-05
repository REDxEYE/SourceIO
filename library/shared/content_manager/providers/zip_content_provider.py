import fnmatch
from pathlib import Path
from typing import Iterator, Optional
from zipfile import ZipFile

from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.utils import Buffer, MemoryBuffer


class ZIPContentProvider(ContentProvider):
    def __init__(self, filepath: Path):
        super().__init__(filepath)
        self._zip_file = ZipFile(filepath)
        self._cache = {k.replace("\\", "/").lower(): k for k in self._zip_file.namelist()}

    def find_file(self, filepath: Path) -> Optional[Buffer]:
        if filepath.as_posix().lower() in self._cache:
            return MemoryBuffer(self._zip_file.read(self._cache[filepath.as_posix()]))

    def find_path(self, filepath: Path) -> Optional[Path]:
        if filepath.as_posix().lower() in self._cache:
            return Path(self.root.as_posix() + ":" + filepath.as_posix())

    def glob(self, pattern: str) -> Iterator[tuple[Path, Buffer]]:
        matches = fnmatch.filter(self._cache.keys(), pattern)
        for match in matches:
            yield Path(match), MemoryBuffer(self._zip_file.read(self._cache[match]))
