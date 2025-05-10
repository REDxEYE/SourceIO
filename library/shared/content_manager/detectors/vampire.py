from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager.detectors.source1 import Source1Detector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.vpk_provider import VPKContentProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath

class VampireDetector(Source1Detector):
    @classmethod
    def scan(cls, path: TinyPath) -> list[ContentProvider]:
        game_root = None
        game_exe = backwalk_file_resolver(path, r'dlls/vampire.dll')
        if game_exe is not None:
            game_root = game_exe.parent.parent
        if game_root is None:
            return {}
        content_providers = {}
        for file in game_root.glob('*.vpk'):
            content_providers[file.stem] = VPKContentProvider(file, SteamAppId.VAMPIRE_THE_MASQUERADE_BLOODLINES)
        return list(content_providers.values())
