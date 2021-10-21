from pathlib import Path
from typing import Dict

from .source1_common import Source1Common
from ..content_provider_base import ContentProviderBase
from ..hfs_provider import HFS2ContentProvider, HFS1ContentProvider
from .....library.utils.path_utilities import backwalk_file_resolver


class VindictusDetector(Source1Common):
    @classmethod
    def scan(cls, path: Path) -> Dict[str, ContentProviderBase]:
        game_root = None
        game_exe = backwalk_file_resolver(path, 'Vindictus.exe')
        if game_exe is not None:
            game_root = game_exe.parent
        if game_root is None:
            return {}
        hfs_provider = HFS2ContentProvider(game_root / 'hfs')
        content_providers = {'hfs': hfs_provider}
        for file in game_root.glob('*.hfs'):
            content_providers[file.stem] = HFS1ContentProvider(file)
        return content_providers
