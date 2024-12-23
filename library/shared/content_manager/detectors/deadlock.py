from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager.detectors.source2 import Source2Detector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.source2_gameinfo_provider import Source2GameInfoProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath


class DeadlockDetector(Source2Detector):

    @classmethod
    def scan(cls, path: TinyPath) -> dict[str, ContentProvider]:
        game_root = None
        deadlock_folder = backwalk_file_resolver(path, 'citadel')
        if deadlock_folder is not None:
            game_root = deadlock_folder.parent
        if game_root is None:
            return []

        providers = {}

        initial_mod_gi_path = backwalk_file_resolver(path, "gameinfo.gi")
        if initial_mod_gi_path is not None:
            cls.add_provider(Source2GameInfoProvider(initial_mod_gi_path, SteamAppId.DEADLOCK), providers)
        user_mod_gi_path = game_root / "citadel/gameinfo.gi"
        if initial_mod_gi_path != user_mod_gi_path:
            cls.add_provider(Source2GameInfoProvider(user_mod_gi_path, SteamAppId.DEADLOCK), providers)
        return list(providers.values())
