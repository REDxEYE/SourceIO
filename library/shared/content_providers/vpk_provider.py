from pathlib import Path
from typing import Iterator, Optional

from ...utils import Buffer, MemoryBuffer
from ..app_id import SteamAppId
from .content_provider_base import ContentProviderBase
from ...utils.rustlib import Vpk


class VPKContentProvider(ContentProviderBase):
    def __init__(self, filepath: Path, override_steamid=0):
        super().__init__(filepath)
        self._override_steamid = override_steamid
        self.vpk_archive = Vpk.from_path(filepath)

    def glob(self, pattern: str) -> Iterator[tuple[Path, Buffer]]:
        for key, data in self.vpk_archive.glob(pattern):
            yield key, MemoryBuffer(data)

    def find_file(self, filepath: Path) -> Optional[Buffer]:
        file = self.vpk_archive.find_file(filepath)
        if file:
            return MemoryBuffer(file)

    def find_path(self, filepath: Path) -> Optional[Path]:
        entry = filepath in self.vpk_archive
        if entry:
            return Path(self.filepath.as_posix() + ":" + filepath.as_posix())

    @property
    def steam_id(self) -> SteamAppId:
        return self._override_steamid or super().steam_id
