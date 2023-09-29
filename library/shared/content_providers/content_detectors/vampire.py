from pathlib import Path
from typing import Dict

from ..vpk_provider import VPKContentProvider
from .....library.utils.path_utilities import backwalk_file_resolver
from ..content_provider_base import ContentProviderBase
from ..hfs_provider import HFS1ContentProvider, HFS2ContentProvider
from .source1_common import Source1Common


class VampireDetector(Source1Common):
    @classmethod
    def scan(cls, path: Path) -> Dict[str, ContentProviderBase]:
        game_root = None
        game_exe = backwalk_file_resolver(path, r'\dlls\vampire.dll')
        if game_exe is not None:
            game_root = game_exe.parent.parent
        if game_root is None:
            return {}
        content_providers = {}
        for file in game_root.glob('*.vpk'):
            content_providers[file.stem] = VPKContentProvider(file)
        return content_providers
