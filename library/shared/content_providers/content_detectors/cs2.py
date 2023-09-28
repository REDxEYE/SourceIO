from pathlib import Path
from typing import Dict

from .....library.utils.path_utilities import backwalk_file_resolver
from ..content_provider_base import ContentProviderBase
from ..hla_content_provider import HLAAddonProvider
from .source2_base import Source2DetectorBase


class CS2Detector(Source2DetectorBase):

    @classmethod
    def scan(cls, path: Path) -> Dict[str, ContentProviderBase]:
        game_root = None
        cs2_client_dll = backwalk_file_resolver(path, r'csgo\bin\win64\client.dll')
        if cs2_client_dll is not None:
            game_root = cs2_client_dll.parent.parent.parent.parent
        if game_root is None:
            return {}
        content_providers = {}
        cls.recursive_traversal(game_root, 'csgo', content_providers)
        return content_providers
