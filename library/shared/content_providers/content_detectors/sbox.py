from pathlib import Path
from typing import Dict

from .....library.utils.path_utilities import backwalk_file_resolver
from ..content_provider_base import ContentProviderBase
from ..sbox_content_provider import SBoxAddonProvider, SBoxDownloadsProvider
from .source2_base import Source2DetectorBase


class SBoxDetector(Source2DetectorBase):

    @classmethod
    def scan(cls, path: Path) -> Dict[str, ContentProviderBase]:
        sbox_root = None
        sbox_exe = backwalk_file_resolver(path, 'sbox.exe')
        if sbox_exe is not None:
            sbox_root = sbox_exe.parent
        if sbox_root is None:
            return {}
        content_providers = {}
        for folder in (sbox_root / 'addons').iterdir():
            if folder.stem.startswith('.'):
                continue
            content_providers[f'sbox_addon_{folder.stem}'] = SBoxAddonProvider(folder)
        for folder in (sbox_root / 'download').iterdir():
            if folder.stem.startswith('.'):
                continue
            if folder.stem == 'http':
                for http_downloaded in folder.iterdir():
                    for addon in http_downloaded.iterdir():
                        content_providers[f'sbox_http_{addon.stem}'] = SBoxDownloadsProvider(addon)
            elif folder.stem == 'github':
                for addon in folder.iterdir():
                    for version in addon.iterdir():
                        content_providers[f'sbox_gh_{addon.stem}_{version.stem[:8]}'] = SBoxDownloadsProvider(version)
        cls.recursive_traversal(sbox_root, 'core',content_providers)
        return content_providers
