from typing import Iterator, Optional, Union

from SourceIO.library.shared.content_manager.provider import ContentProvider, glob_generic
from SourceIO.library.utils import Buffer, FileBuffer
from SourceIO.library.utils.path_utilities import backwalk_file_resolver
from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.utils.tiny_path import TinyPath


class LooseFilesContentProvider(ContentProvider):
    def check(self, filepath: TinyPath) -> bool:
        return (self.root / filepath).exists()

    def get_relative_path(self, filepath: TinyPath):
        return filepath.relative_to(self.root)

    def get_provider_from_path(self, filepath: TinyPath) -> ContentProvider | None:
        full_path = self.root / filepath
        if full_path.exists() and full_path.is_file():
            return self
        return None

    def get_steamid_from_asset(self, asset_path: TinyPath) -> SteamAppId | None:
        return self.steam_id

    @property
    def name(self) -> str:
        return self.root.stem

    @property
    def root(self) -> TinyPath:
        return self.filepath

    def __init__(self, filepath: TinyPath, override_steamid=SteamAppId.UNKNOWN):
        self._override_steamid = override_steamid
        super().__init__(filepath)

    def find_file(self, filepath: Union[str, TinyPath]) -> Optional[Buffer]:
        file = backwalk_file_resolver(self.filepath, filepath)
        if file and file.is_file():
            return FileBuffer(file)

    def glob(self, pattern: str) -> Iterator[tuple[TinyPath, Buffer]]:
        yield from glob_generic(self.root, pattern)

    @property
    def steam_id(self) -> SteamAppId:
        return self._override_steamid
