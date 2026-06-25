from typing import Collection

from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager.detectors.source2 import Source2Detector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.source2_gameinfo_provider import Source2GameInfoProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath


class Dota2Detector(Source2Detector):

    @classmethod
    def game(cls) -> str:
        return "Dota 2"

    @classmethod
    def find_game_root(cls, path: TinyPath) -> TinyPath | None:
        dota_pack01 = backwalk_file_resolver(path, r'dota/pak01_dir.vpk')
        if dota_pack01 is not None:
            return dota_pack01.parent.parent
        return None

    @classmethod
    def scan(cls, path: TinyPath) -> tuple[Collection[ContentProvider] | None, TinyPath | None]:
        game_root = cls.find_game_root(path)
        if game_root is None:
            return None, None
        providers = set()
        initial_mod_gi_path = backwalk_file_resolver(path, "gameinfo.gi")
        if initial_mod_gi_path is not None:
            cls.add_provider(Source2GameInfoProvider(initial_mod_gi_path, SteamAppId.COUNTER_STRIKE_GO), providers)
        user_mod_gi_path = game_root / "dota/gameinfo.gi"
        if initial_mod_gi_path != user_mod_gi_path and user_mod_gi_path.exists():
            cls.add_provider(Source2GameInfoProvider(user_mod_gi_path, SteamAppId.COUNTER_STRIKE_GO), providers)
        return providers, game_root
