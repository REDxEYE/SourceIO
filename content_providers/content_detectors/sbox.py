from pathlib import Path
from typing import Dict

from ..content_provider_base import ContentProviderBase, ContentDetectorBase
from ..sbox_content_provider import SBoxDownloadsProvider, SBoxAddonProvider
from ..source2_content_provider import GameinfoContentProvider


class SBoxDetector(ContentDetectorBase):

    @classmethod
    def scan(cls, path: Path) -> Dict[str, ContentProviderBase]:
        sbox_root = None
        while len(path.parts) > 2:
            path = path.parent
            if (path / 'sbox.exe').exists():
                sbox_root = path
                break
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
            for download in folder.iterdir():
                content_providers[f'sbox_download_{download.stem}'] = SBoxDownloadsProvider(download)
        content_providers[f'sbox_core'] = GameinfoContentProvider(sbox_root / 'core' / 'gameinfo.gi')
        return content_providers
