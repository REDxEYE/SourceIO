from ..providers import register_provider
from ..providers.source2_gameinfo_provider import Source2GameInfoProvider
from ...app_id import SteamAppId
from ....utils.tiny_path import TinyPath
from .....library.utils.path_utilities import backwalk_file_resolver
from SourceIO.library.shared.content_manager.provider import ContentProvider
from .source2 import Source2Detector


class RobotRepairDetector(Source2Detector):

    @classmethod
    def scan(cls, path: TinyPath) -> list[ContentProvider]:
        game_root = None
        p2imp_folder = backwalk_file_resolver(path, 'portal2_imported')
        if p2imp_folder is not None:
            game_root = p2imp_folder.parent
        if game_root is None:
            return []
        providers = {}

        initial_mod_gi_path = backwalk_file_resolver(path, "gameinfo.gi")
        if initial_mod_gi_path is not None:
            cls.add_provider(Source2GameInfoProvider(initial_mod_gi_path, SteamAppId.ROBOT_REPAIR), providers)
        user_mod_gi_path = game_root / "csgo/gameinfo.gi"
        if initial_mod_gi_path != user_mod_gi_path:
            cls.add_provider(Source2GameInfoProvider(user_mod_gi_path, SteamAppId.ROBOT_REPAIR), providers)
        return list(providers.values())
