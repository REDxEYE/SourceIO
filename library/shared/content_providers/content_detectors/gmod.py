from pathlib import Path
from typing import Dict

from .....library.utils.path_utilities import backwalk_file_resolver
from ..content_provider_base import ContentProviderBase
from ..gma_provider import GMAContentProvider
from ..non_source_sub_manager import NonSourceContentProvider
from ..source1_content_provider import GameinfoContentProvider
from .source1_common import Source1Common


class GModDetector(Source1Common):
    @classmethod
    def scan(cls, path: Path) -> Dict[str, ContentProviderBase]:
        gmod_root = None
        gmod_dir = backwalk_file_resolver(path, 'garrysmod')
        if gmod_dir is not None:
            gmod_root = gmod_dir.parent
        if gmod_root is None:
            return {}
        content_providers = {}
        cls.recursive_traversal(gmod_root, 'garrysmod', content_providers)
        cls.register_common(gmod_root, content_providers)
        if (gmod_dir / 'addons').exists():
            for addon in (gmod_dir / 'addons').iterdir():
                if addon.suffix == '.gma':
                    provider = GMAContentProvider(addon, 4000)
                    if provider.gma_archive is None:
                        continue
                    content_providers[addon.name] = provider
                elif addon.is_dir():
                    content_providers[addon.stem] = NonSourceContentProvider(addon, 4000)

        return content_providers

    @classmethod
    def register_common(cls, root_path: Path, content_providers: Dict[str, ContentProviderBase]):
        super().register_common(root_path, content_providers)
