from pathlib import Path
from typing import Dict

from .....library.utils.path_utilities import backwalk_file_resolver
from ..content_provider_base import ContentProviderBase
from ..hla_content_provider import HLAAddonProvider
from .source2_base import Source2DetectorBase


class HLADetector(Source2DetectorBase):

    @classmethod
    def scan(cls, path: Path) -> Dict[str, ContentProviderBase]:
        hla_root = None
        hlvr_folder = backwalk_file_resolver(path, 'hlvr')
        if hlvr_folder is not None:
            hla_root = hlvr_folder.parent
        if hla_root is None:
            return {}
        if not (hla_root / 'hlvr_addons').exists():
            return {}
        content_providers = {}
        for folder in (hla_root / 'hlvr_addons').iterdir():
            if folder.stem.startswith('.'):
                continue
            content_providers[f'hla_addon_{folder.stem}'] = HLAAddonProvider(folder)
        cls.recursive_traversal(hla_root, 'hlvr', content_providers)
        return content_providers
