from pathlib import Path

from .....library.utils.path_utilities import backwalk_file_resolver
from ..content_provider_base import ContentProviderBase
from ..non_source_sub_manager import NonSourceContentProvider
from ..source1_content_provider import GameinfoContentProvider
from .source1_common import Source1Common
from .....logger import SLoggingManager

log_manager = SLoggingManager()
logger = log_manager.get_logger('SFMDetector')


class SFMDetector(Source1Common):
    @classmethod
    def scan(cls, path: Path) -> dict[str, ContentProviderBase]:
        sfm_root = None
        sfm_exe = backwalk_file_resolver(path, 'sfm.exe')
        if sfm_exe is not None:
            sfm_root = sfm_exe.parent
        if sfm_root is None:
            return {}
        content_providers = {}
        cls.recursive_traversal(sfm_root, 'usermod', content_providers)
        for folder in sfm_root.iterdir():
            if folder.stem in content_providers or folder.is_file():
                continue
            elif (folder / 'gameinfo.txt').exists():
                try:
                    content_providers[folder.stem] = GameinfoContentProvider(folder / 'gameinfo.txt')
                except ValueError as ex:
                    logger.exception(f"Failed to parse gameinfo for {folder}", ex)
        cls.register_common(sfm_root, content_providers)
        return content_providers

    @classmethod
    def register_common(cls, root_path: Path, content_providers: dict[str, ContentProviderBase]):
        cls.add_if_exists(root_path / 'workshop', NonSourceContentProvider, content_providers)
        super().register_common(root_path, content_providers)
