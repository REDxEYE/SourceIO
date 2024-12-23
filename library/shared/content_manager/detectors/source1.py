import traceback
from abc import ABCMeta
from typing import Type

from SourceIO.library.shared.content_manager.detectors.content_detector import ContentDetector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.loose_files import LooseFilesContentProvider
from SourceIO.library.shared.content_manager.providers.source1_gameinfo_provider import Source1GameInfoProvider
from SourceIO.library.shared.content_manager.providers.vpk_provider import VPKContentProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('Source1DetectorBase')


class Source1Detector(ContentDetector, metaclass=ABCMeta):

    @classmethod
    def scan_for_vpk(cls, root_dir: TinyPath, content_providers: dict[str, ContentProvider]):
        for vpk in root_dir.glob('*_dir.vpk'):
            try:
                cls.add_provider(VPKContentProvider(vpk), content_providers)
            except IOError as ex:
                print(f'Failed to load "{vpk}" VPK due to {ex}.')
                traceback.print_exc()
                print(f'Skipping {vpk}.')

    @classmethod
    def add_if_exists(cls, path: TinyPath, content_provider_class: Type[ContentProvider],
                      content_providers: dict[str, ContentProvider]):
        super().add_if_exists(path, content_provider_class, content_providers)
        cls.scan_for_vpk(path, content_providers)

    @classmethod
    def scan(cls, path: TinyPath) -> list[ContentProvider]:
        game_root = None
        is_source = backwalk_file_resolver(path, 'platform') and backwalk_file_resolver(path, 'bin')
        if is_source:
            game_root = (backwalk_file_resolver(path, 'platform') or backwalk_file_resolver(path, 'bin')).parent
        if game_root is None:
            return []
        providers = {}
        initial_mod_gi_path = backwalk_file_resolver(path, "gameinfo.txt")
        if initial_mod_gi_path is not None:
            cls.add_provider(Source1GameInfoProvider(initial_mod_gi_path), providers)
        cls.register_common(game_root, providers)
        return list(providers.values())

    @classmethod
    def register_common(cls, root_path: TinyPath, content_providers: dict[str, ContentProvider]):
        cls.add_if_exists(root_path / 'synergy', LooseFilesContentProvider, content_providers)
