from pathlib import Path


from SourceIO.library.shared.content_manager.providers.zip_content_provider import ZIPContentProvider
from .....library.utils.path_utilities import backwalk_file_resolver
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.loose_files import LooseFilesContentProvider
from .source1 import Source1Detector


class IDTech3Detector(Source1Detector):
    @classmethod
    def scan(cls, path: Path) -> dict[str, ContentProvider]:
        base_dir = backwalk_file_resolver(path, 'base')
        if base_dir is None:
            return {}
        game_name = base_dir.parent.stem
        content_providers = {game_name: LooseFilesContentProvider(base_dir)}
        for pk3_file in base_dir.glob('*.pk3'):
            content_providers[pk3_file.name] = ZIPContentProvider(pk3_file)

        return content_providers

    @classmethod
    def register_common(cls, root_path: Path, content_providers: dict[str, ContentProvider]):
        super().register_common(root_path, content_providers)
