from pathlib import Path
from typing import Dict

from .source1_common import Source1Common
from ..content_provider_base import ContentProviderBase
from .....library.utils.path_utilities import backwalk_file_resolver

from ..source1_content_provider import GameinfoContentProvider
from ..non_source_sub_manager import NonSourceContentProvider


class SFMDetector(Source1Common):
    @classmethod
    def scan(cls, path: Path) -> Dict[str, ContentProviderBase]:
        sfm_root = None
        sfm_exe = backwalk_file_resolver(path, 'sfm.exe')
        if sfm_exe is not None:
            sfm_root = sfm_exe.parent
        if sfm_root is None:
            return {}
        content_providers = {}
        cls.recursive_traversal(sfm_root, 'usermod', content_providers)
        for folder in sfm_root.iterdir():
            if folder.stem in content_providers:
                continue
            elif (folder / 'gameinfo.txt').exists():
                content_providers[folder.stem] = GameinfoContentProvider(folder / 'gameinfo.txt')
        cls.register_common(sfm_root, content_providers)
        return content_providers

    @classmethod
    def register_common(cls, root_path: Path, content_providers: Dict[str, ContentProviderBase]):
        cls.add_if_exists(root_path / 'workshop', NonSourceContentProvider, content_providers)
        super().register_common(root_path, content_providers)