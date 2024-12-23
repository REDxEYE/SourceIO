from abc import ABCMeta

from SourceIO.library.shared.content_manager.detectors.content_detector import ContentDetector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.source2_gameinfo_provider import Source2GameInfoProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath


class Source2Detector(ContentDetector, metaclass=ABCMeta):

    @classmethod
    def scan(cls, path: TinyPath) -> list[ContentProvider]:
        s2_root = None
        s2_gameinfo = backwalk_file_resolver(path, 'gameinfo.gi')
        if s2_gameinfo is not None:
            s2_root = s2_gameinfo.parent.parent
        if s2_root is None:
            return []
        providers = {}
        if s2_gameinfo is not None:
            cls.add_provider(Source2GameInfoProvider(s2_gameinfo), providers)
        return list(providers.values())
