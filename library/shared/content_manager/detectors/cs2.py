from pathlib import Path


from .....library.utils.path_utilities import backwalk_file_resolver
from SourceIO.library.shared.content_manager.provider import ContentProvider
from .source2 import Source2Detector


class CS2Detector(Source2Detector):

    @classmethod
    def scan(cls, path: Path) -> list[ContentProvider]:
        game_root = None
        cs2_client_dll = backwalk_file_resolver(path, r'csgo\bin\win64\client.dll')
        if cs2_client_dll is not None:
            game_root = cs2_client_dll.parent.parent.parent.parent
        if game_root is None:
            return []
        content_providers = {}
        cls.recursive_traversal(game_root, 'csgo', content_providers)
        return list(content_providers.values())
