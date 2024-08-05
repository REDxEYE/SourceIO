from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

from SourceIO.library.global_config import GoldSrcConfig
from SourceIO.library.goldsrc.wad import WadFile
from SourceIO.library.shared.content_manager.provider import ContentProvider, glob_generic, find_path_generic, \
    find_file_generic
from SourceIO.library.utils import Buffer


class GoldSrcContentProvider(ContentProvider):

    def __init__(self, filepath: Path):
        assert filepath.is_dir()
        super().__init__(filepath)

    def find_file(self, filepath: Path) -> Optional[Buffer]:
        if not GoldSrcConfig().use_hd and self.filepath.stem.endswith('_hd'):
            return None
        return find_file_generic(self.root, filepath)

    def find_path(self, filepath: Path) -> Optional[Path]:
        return find_path_generic(self.root, filepath)

    def glob(self, pattern: str) -> Iterator[tuple[Path, Buffer]]:
        yield from glob_generic(self.root, pattern)


class GoldSrcWADContentProvider(ContentProvider):

    def __init__(self, filepath: Path):
        assert filepath.suffix == '.wad'
        super().__init__(filepath)
        self.wad_file = WadFile(filepath)

    def find_file(self, filepath: Union[str, Path]) -> Optional[Buffer]:
        return self.wad_file.get_file(filepath.stem)

    def find_path(self, filepath: Union[str, Path]) -> Optional[Path]:
        entry = self.wad_file.get_file(filepath.stem)
        if entry:
            return Path(self.filepath.as_posix() + ":" + filepath.as_posix())

    def glob(self, pattern: str) -> Iterator[tuple[Path, Buffer]]:
        return iter([])
