from abc import abstractmethod
# from pathlib import Path
from typing import Iterator, Optional

from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.utils import Buffer, FileBuffer
from SourceIO.library.utils.path_utilities import corrected_path
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.logger import SourceLogMan

Path = TinyPath
log_manager = SourceLogMan()
logger = log_manager.get_logger('ContentManager')


def find_file_generic(root: Path, filepath: Path) -> Buffer | None:
    filepath = corrected_path(root / filepath)
    if filepath.exists():
        return FileBuffer(filepath)
    else:
        return None


def find_path_generic(root: Path, filepath: Path) -> Path | None:
    filepath = corrected_path(root / filepath)
    if filepath.exists():
        return filepath
    else:
        return None


def glob_generic(root: Path, pattern: str) -> Iterator[tuple[Path, Buffer]]:
    for filename in root.rglob(pattern):
        yield (filename.relative_to(root)).as_posix(), FileBuffer(filename)


# backport
def is_relative_to(path: Path, *other):
    """Return True if the path is relative to another path or False.
    """
    try:
        path.relative_to(*other)
        return True
    except ValueError:
        return False


class ContentProvider:
    @classmethod
    def class_name(cls):
        return cls.__name__

    def __init__(self, filepath: Path):
        self.filepath = filepath

    def __hash__(self):
        return hash(self.unique_name)

    @abstractmethod
    def glob(self, pattern: str):
        ...

    @abstractmethod
    def find_file(self, filepath: TinyPath) -> Buffer | None:
        ...

    @abstractmethod
    def check(self, filepath: Path) -> bool:
        ...

    @abstractmethod
    def get_relative_path(self, filepath: Path) -> Path | None:
        ...

    @abstractmethod
    def get_provider_from_path(self, filepath) -> Optional['ContentProvider']:
        ...

    @abstractmethod
    def get_steamid_from_asset(self, asset_path: Path) -> SteamAppId | None:
        ...

    @property
    @abstractmethod
    def root(self) -> Path:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def steam_id(self) -> SteamAppId:
        ...

    @property
    def unique_name(self) -> str:
        root_stem = self.root.stem
        if self.steam_id <= 0:
            return f"{root_stem} {self.name}" if root_stem != self.name else self.name
        else:
            return f"{self.steam_id.name} {root_stem} {self.name}" if root_stem != self.name else f"{self.steam_id.name} {self.name}"

    def __eq__(self, other: 'ContentProvider'):
        return self.filepath == other.filepath and self.name == other.name and self.steam_id == other.steam_id

    def __repr__(self):
        return f"{self.__class__.__name__}({self.filepath!r})"
