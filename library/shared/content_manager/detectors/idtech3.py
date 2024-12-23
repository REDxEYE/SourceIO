from SourceIO.library.shared.content_manager.detectors.source1 import Source1Detector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.loose_files import LooseFilesContentProvider
from SourceIO.library.shared.content_manager.providers.zip_content_provider import ZIPContentProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath


class IDTech3Detector(Source1Detector):
    @classmethod
    def scan(cls, path: TinyPath) -> list[ContentProvider]:
        base_dir = backwalk_file_resolver(path, 'base')
        if base_dir is None:
            return []
        providers = {}
        cls.add_provider(LooseFilesContentProvider(base_dir), providers)
        for pk3_file in base_dir.glob('*.pk3'):
            cls.add_provider(ZIPContentProvider(pk3_file), providers)

        return list(providers.values())
