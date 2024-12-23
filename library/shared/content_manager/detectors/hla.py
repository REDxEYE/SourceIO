from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager.detectors.source2 import Source2Detector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.loose_files import LooseFilesContentProvider
from SourceIO.library.shared.content_manager.providers.source2_gameinfo_provider import Source2GameInfoProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath


class HLADetector(Source2Detector):

    @classmethod
    def scan(cls, path: TinyPath) -> list[ContentProvider]:
        hla_root = None
        hlvr_folder = backwalk_file_resolver(path, 'hlvr')
        if hlvr_folder is not None:
            hla_root = hlvr_folder.parent
        if hla_root is None:
            return []
        if not (hla_root / 'hlvr_addons').exists():
            return []

        providers = {}

        initial_mod_gi_path = backwalk_file_resolver(path, "gameinfo.gi")
        if initial_mod_gi_path is not None:
            cls.add_provider(Source2GameInfoProvider(initial_mod_gi_path, SteamAppId.HALF_LIFE_ALYX), providers)
        user_mod_gi_path = hla_root / "hlvr/gameinfo.gi"
        if initial_mod_gi_path != user_mod_gi_path:
            cls.add_provider(Source2GameInfoProvider(user_mod_gi_path, SteamAppId.HALF_LIFE_ALYX), providers)

        for folder in (hla_root / 'hlvr_addons').iterdir():
            if folder.stem.startswith('.') or folder.is_file():
                continue
            cls.add_provider(LooseFilesContentProvider(folder, SteamAppId.HALF_LIFE_ALYX), providers)
        return list(providers.values())
