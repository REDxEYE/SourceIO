from pathlib import Path


from .....library.utils.path_utilities import backwalk_file_resolver
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.hla_content_provider import HLAAddonProvider
from .source2 import Source2Detector


class HLADetector(Source2Detector):

    @classmethod
    def scan(cls, path: Path) -> dict[str, ContentProvider]:
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
