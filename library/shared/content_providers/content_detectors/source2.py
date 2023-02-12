from pathlib import Path
from typing import Dict

from .....library.utils.path_utilities import backwalk_file_resolver
from ..content_provider_base import ContentProviderBase
from ..hla_content_provider import HLAAddonProvider
from .source2_base import Source2DetectorBase


class Source2Detector(Source2DetectorBase):

    @classmethod
    def scan(cls, path: Path) -> Dict[str, ContentProviderBase]:
        s2_root = None
        s2_gameinfo = backwalk_file_resolver(path, 'gameinfo.gi')
        if s2_gameinfo is not None:
            s2_root = s2_gameinfo.parent.parent
        if s2_root is None:
            return {}
        content_providers = {}
        cls.recursive_traversal(s2_root, s2_gameinfo.parent.stem, content_providers)
        return content_providers
