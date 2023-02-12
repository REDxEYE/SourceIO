from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterator, Optional, Tuple, Type, Union

from ...utils import Buffer, FileBuffer
from ...utils.path_utilities import corrected_path
from ..app_id import SteamAppId


class ContentProviderBase:

    @classmethod
    def class_name(cls):
        return cls.__name__

    def __init__(self, filepath: Path):
        self.filepath = filepath

    def find_file(self, filepath: Path) -> Optional[Buffer]:
        raise NotImplementedError('Implement me!')

    def find_path(self, filepath: Path) -> Optional[Path]:
        raise NotImplementedError('Implement me!')

    def glob(self, pattern: str) -> Iterator[Tuple[Path, Buffer]]:
        raise NotImplementedError('Implement me!')

    @property
    def steam_id(self) -> SteamAppId:
        return SteamAppId.UNKNOWN

    @property
    def root(self) -> Path:
        if self.filepath.is_file():
            return self.filepath.parent
        else:
            return self.filepath

    def _find_file_generic(self, filepath: Union[str, Path], additional_dir=None, extension=None) -> Optional[Buffer]:
        filepath = Path(str(filepath).strip("\\/"))

        new_filepath = filepath
        if additional_dir:
            new_filepath = Path(additional_dir, new_filepath)
        if extension:
            new_filepath = new_filepath.with_suffix(extension)
        new_filepath = corrected_path(self.root / new_filepath)
        if new_filepath.exists():
            return FileBuffer(new_filepath)
        else:
            return None

    def _find_path_generic(self, filepath: Union[str, Path], additional_dir=None,
                           extension=None) -> Optional[Path]:
        filepath = Path(str(filepath).strip("\\/"))

        new_filepath = filepath
        if additional_dir:
            new_filepath = Path(additional_dir, new_filepath)
        if extension:
            new_filepath = new_filepath.with_suffix(extension)
        new_filepath = corrected_path(self.root / new_filepath)
        if new_filepath.exists():
            return new_filepath
        else:
            return None

    def _glob_generic(self, pattern: str) -> Iterator[Tuple[Path, Buffer]]:
        for filename in self.root.rglob(pattern):
            yield (filename.relative_to(self.root)).as_posix(), FileBuffer(filename)


class ContentDetectorBase:

    @classmethod
    def scan(cls, path: Path) -> Dict[str, ContentProviderBase]:
        raise NotImplementedError("Implement me")

    @staticmethod
    def add_provider(name: str, provider: ContentProviderBase, content_providers):
        if name not in content_providers:
            content_providers[name] = provider

    @classmethod
    def add_if_exists(cls, path: Path,
                      content_provider_class: Type[ContentProviderBase],
                      content_providers: Dict[str, ContentProviderBase]):
        if path.exists():
            cls.add_provider(path.stem, content_provider_class(path), content_providers)
