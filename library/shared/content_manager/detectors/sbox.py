from typing import Collection

from SourceIO.library.shared.content_manager.detectors.source2 import Source2Detector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.sbox_content_provider import SBoxAddonProvider, \
    SBoxDownloadsProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath


class SBoxDetector(Source2Detector):

    @classmethod
    def game(cls) -> str:
        return 'S&Box'

    @classmethod
    def find_game_root(cls, path: TinyPath) -> TinyPath | None:
        sbox_exe = backwalk_file_resolver(path, 'sbox.exe')
        if sbox_exe is not None:
            return sbox_exe.parent
        return None

    @classmethod
    def scan(cls, path: TinyPath) -> tuple[Collection[ContentProvider] | None, TinyPath | None]:
        sbox_root = cls.find_game_root(path)
        if sbox_root is None:
            return None, None
        providers = set()
        for folder in (sbox_root / 'addons').iterdir():
            if folder.stem.startswith('.'):
                continue
            cls.add_provider(SBoxAddonProvider(folder), providers)
        for folder in (sbox_root / 'download').iterdir():
            if folder.stem.startswith('.'):
                continue
            if folder.stem == 'http':
                for http_downloaded in folder.iterdir():
                    for addon in http_downloaded.iterdir():
                        cls.add_provider(SBoxDownloadsProvider(addon), providers)
            elif folder.stem == 'github':
                for addon in folder.iterdir():
                    for version in addon.iterdir():
                        cls.add_provider(SBoxDownloadsProvider(version), providers)
        return providers, sbox_root
