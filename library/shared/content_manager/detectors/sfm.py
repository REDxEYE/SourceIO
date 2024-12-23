from SourceIO.library.shared.content_manager.detectors.source1 import Source1Detector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.loose_files import LooseFilesContentProvider
from SourceIO.library.shared.content_manager.providers.source1_gameinfo_provider import Source1GameInfoProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('SFMDetector')


class SFMDetector(Source1Detector):
    @classmethod
    def scan(cls, path: TinyPath) -> list[ContentProvider]:
        sfm_root = None
        sfm_exe = backwalk_file_resolver(path, "sfm.exe")
        if sfm_exe is not None:
            sfm_root = sfm_exe.parent
        if sfm_root is None:
            return []
        providers = {}

        initial_mod_gi_path = backwalk_file_resolver(path, "gameinfo.txt")
        if initial_mod_gi_path is not None:
            cls.add_provider(Source1GameInfoProvider(initial_mod_gi_path), providers)
        user_mod_gi_path = sfm_root / "usermod/gameinfo.txt"
        if initial_mod_gi_path != user_mod_gi_path:
            cls.add_provider(Source1GameInfoProvider(user_mod_gi_path), providers)
        cls.register_common(sfm_root, providers)
        for folder in sfm_root.iterdir():
            if (folder / 'gameinfo.txt').exists():
                try:
                    cls.add_provider(Source1GameInfoProvider(folder / 'gameinfo.txt'), providers)
                except ValueError as ex:
                    logger.exception(f"Failed to parse gameinfo for {folder}", ex)
        return list(providers.values())

    @classmethod
    def register_common(cls, root_path: TinyPath, content_providers: dict[str, ContentProvider]):
        cls.add_if_exists(root_path / 'workshop', LooseFilesContentProvider, content_providers)
        super().register_common(root_path, content_providers)
