from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager.detectors.source2 import Source2Detector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.source2_gameinfo_provider import Source2GameInfoProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath


class CS2Detector(Source2Detector):

    @classmethod
    def scan(cls, path: TinyPath) -> list[ContentProvider]:
        game_root = None
        cs2_client_dll = backwalk_file_resolver(path, r'game\bin\win64\cs2.exe')
        if cs2_client_dll is not None:
            game_root = cs2_client_dll.parent.parent.parent
        if game_root is None:
            return []
        providers = {}
        initial_mod_gi_path = backwalk_file_resolver(path, "gameinfo.gi")
        if initial_mod_gi_path is not None:
            cls.add_provider(Source2GameInfoProvider(initial_mod_gi_path, SteamAppId.COUNTER_STRIKE_GO), providers)
        user_mod_gi_path = game_root / "csgo/gameinfo.gi"
        if initial_mod_gi_path != user_mod_gi_path:
            cls.add_provider(Source2GameInfoProvider(user_mod_gi_path, SteamAppId.COUNTER_STRIKE_GO), providers)
        return list(providers.values())
