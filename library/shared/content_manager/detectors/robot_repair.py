from pathlib import Path


from .....library.utils.path_utilities import backwalk_file_resolver
from SourceIO.library.shared.content_manager.provider import ContentProvider
from .source2 import Source2Detector


class RobotRepairDetector(Source2Detector):

    @classmethod
    def scan(cls, path: Path) -> dict[str, ContentProvider]:
        game_root = None
        p2imp_folder = backwalk_file_resolver(path, 'portal2_imported')
        if p2imp_folder is not None:
            game_root = p2imp_folder.parent
        if game_root is None:
            return {}
        content_providers = {}
        cls.recursive_traversal(game_root, 'vr', content_providers)
        return content_providers
