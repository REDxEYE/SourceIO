from pathlib import Path
from typing import Dict, Type

from .source1_base import Source1DetectorBase
from ..content_provider_base import ContentDetectorBase, ContentProviderBase
from .....library.utils.path_utilities import backwalk_file_resolver

from ..source1_content_provider import GameinfoContentProvider
from ..non_source_sub_manager import NonSourceContentProvider


class Source1Common(Source1DetectorBase):

    @classmethod
    def add_if_exists(cls, path: Path, content_provider_class: Type[ContentProviderBase],
                      content_providers: Dict[str, ContentProviderBase]):
        super().add_if_exists(path, content_provider_class, content_providers)
        cls.scan_for_vpk(path, content_providers)

    @classmethod
    def scan(cls, path: Path) -> Dict[str, ContentProviderBase]:
        game_root = None
        is_source = backwalk_file_resolver(path, 'platform') and backwalk_file_resolver(path, 'bin')
        if is_source:
            game_root = (backwalk_file_resolver(path, 'platform') or backwalk_file_resolver(path, 'bin')).parent
        if game_root is None:
            return {}
        content_providers = {}
        for folder in game_root.iterdir():
            if folder.stem in content_providers:
                continue
            elif (folder / 'gameinfo.txt').exists():
                cls.recursive_traversal(game_root, folder.stem, content_providers)
        cls.register_common(game_root, content_providers)
        return content_providers

    @classmethod
    def register_common(cls, root_path: Path, content_providers: Dict[str, ContentProviderBase]):
        cls.add_if_exists(root_path / 'platform', NonSourceContentProvider, content_providers)
        cls.add_if_exists(root_path / 'hl2', NonSourceContentProvider, content_providers)
        cls.add_if_exists(root_path / 'synergy', NonSourceContentProvider, content_providers)
