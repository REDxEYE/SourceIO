from abc import ABCMeta
from typing import Collection

from SourceIO.library.shared.content_manager.detectors.content_detector import ContentDetector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.source2_gameinfo_provider import Source2GameInfoProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath


class Source2Detector(ContentDetector):

    @classmethod
    def game(cls) -> str:
        return 'Source 2 engine game'


    @classmethod
    def scan(cls, path: TinyPath) -> tuple[Collection[ContentProvider] | None, TinyPath | None]:
        s2_root = None
        s2_gameinfo = backwalk_file_resolver(path, 'gameinfo.gi')
        if s2_gameinfo is not None:
            s2_root = s2_gameinfo.parent.parent
        if s2_root is None:
            return None,None
        providers = set()
        if s2_gameinfo is not None:
            cls.add_provider(Source2GameInfoProvider(s2_gameinfo), providers)
        return providers, s2_root
