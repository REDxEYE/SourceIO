from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager.detectors.source1 import Source1Detector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.source1_gameinfo_provider import Source1GameInfoProvider
from SourceIO.library.shared.content_manager.providers.vpk_provider import VPKContentProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath


class CSGODetector(Source1Detector):

    @classmethod
    def scan(cls, path: TinyPath) -> list[ContentProvider]:
        game_root = None
        csgo_client_dll = backwalk_file_resolver(path, r'csgo.exe')
        if csgo_client_dll is not None:
            game_root = csgo_client_dll.parent
        if game_root is None:
            return []
        providers = {}
        initial_mod_gi_path = backwalk_file_resolver(path, "gameinfo.txt")
        if initial_mod_gi_path is not None:
            for vpk in initial_mod_gi_path.glob("*_dir.vpk"):
                cls.add_provider(VPKContentProvider(vpk, SteamAppId.COUNTER_STRIKE_GO), providers)
            cls.add_provider(Source1GameInfoProvider(initial_mod_gi_path), providers)
        user_mod_gi_path = game_root / "csgo/gameinfo.txt"
        if initial_mod_gi_path != user_mod_gi_path:
            for vpk in user_mod_gi_path.glob("*_dir.vpk"):
                cls.add_provider(VPKContentProvider(vpk, SteamAppId.COUNTER_STRIKE_GO), providers)

            cls.add_provider(Source1GameInfoProvider(user_mod_gi_path), providers)
        return list(providers.values())
