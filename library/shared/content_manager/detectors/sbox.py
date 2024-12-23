from SourceIO.library.shared.content_manager.detectors.source2 import Source2Detector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.sbox_content_provider import SBoxAddonProvider, SBoxDownloadsProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath


class SBoxDetector(Source2Detector):

    @classmethod
    def scan(cls, path: TinyPath) -> list[ContentProvider]:
        sbox_root = None
        sbox_exe = backwalk_file_resolver(path, 'sbox.exe')
        if sbox_exe is not None:
            sbox_root = sbox_exe.parent
        if sbox_root is None:
            return []
        providers = {}
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
        return list(providers.values())
