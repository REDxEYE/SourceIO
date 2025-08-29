from typing import Collection

from SourceIO.library.shared.content_manager.detectors.source1 import Source1Detector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.loose_files import LooseFilesContentProvider
from SourceIO.library.shared.content_manager.providers.zip_content_provider import ZIPContentProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath


class IDTech3Detector(Source1Detector):

    @classmethod
    def game(cls) -> str:
        return 'idtech3 game'

    @classmethod
    def find_game_root(cls, path: TinyPath) -> TinyPath | None:
        return backwalk_file_resolver(path, 'base')

    @classmethod
    def scan(cls, path: TinyPath) -> tuple[Collection[ContentProvider] | None, TinyPath | None]:
        base_dir = cls.find_game_root(path)
        if base_dir is None:
            return None, None
        providers = set()
        cls.add_provider(LooseFilesContentProvider(base_dir), providers)
        for pk3_file in base_dir.glob('*.pk3'):
            cls.add_provider(ZIPContentProvider(pk3_file), providers)

        return providers, base_dir
