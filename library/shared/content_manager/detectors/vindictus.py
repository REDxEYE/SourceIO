from SourceIO.library.shared.content_manager.detectors.source1 import Source1Detector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.hfs_provider import HFS1ContentProvider, HFS2ContentProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath


class VindictusDetector(Source1Detector):
    @classmethod
    def scan(cls, path: TinyPath) -> list[ContentProvider]:
        game_root = None
        game_exe = backwalk_file_resolver(path, 'Vindictus.exe')
        if game_exe is not None:
            game_root = game_exe.parent
        if game_root is None:
            return []
        hfs_provider = HFS2ContentProvider(game_root / 'hfs')
        content_providers = {'hfs': hfs_provider}
        for file in game_root.glob('*.hfs'):
            content_providers[file.stem] = HFS1ContentProvider(file)
        return list(content_providers.values())
