from typing import Collection

from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager.detectors.source1 import Source1Detector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.vpk_provider import VPKContentProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath


class TitanfallDetector(Source1Detector):

    @classmethod
    def game(cls) -> str:
        return 'Titanfall'

    @classmethod
    def find_game_root(cls, path: TinyPath) -> TinyPath | None:
        game_exe = backwalk_file_resolver(path, 'Titanfall.exe')
        if game_exe is not None:
            return game_exe.parent
        return None

    @classmethod
    def scan(cls, path: TinyPath) -> tuple[Collection[ContentProvider] | None, TinyPath | None]:
        game_root = cls.find_game_root(path)
        if game_root is None:
            return None, None
        content_providers = set()
        for file in (game_root / 'vpk').glob('*_dir.vpk'):
            cls.add_provider(VPKContentProvider(file, SteamAppId.PORTAL_2), content_providers)
        return content_providers, game_root
