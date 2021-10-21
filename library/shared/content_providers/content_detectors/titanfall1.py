from pathlib import Path
from typing import Dict

from .source1_common import Source1Common
from ..content_manager import ContentManager
from ..content_provider_base import ContentProviderBase
from ..vpk_provider import VPKContentProvider
from ...app_id import SteamAppId
from .....library.utils.path_utilities import backwalk_file_resolver


class TitanfallDetector(Source1Common):
    @classmethod
    def scan(cls, path: Path) -> Dict[str, ContentProviderBase]:
        game_root = None
        game_exe = backwalk_file_resolver(path, 'Titanfall.exe')
        if game_exe is not None:
            game_root = game_exe.parent
        if game_root is None:
            return {}
        ContentManager()._titanfall_mode = True
        content_providers = {}
        for file in (game_root / 'vpk').glob('*_dir.vpk'):
            content_providers[file.stem] = VPKContentProvider(file, SteamAppId.PORTAL_2)
        return content_providers
