from SourceIO.library.shared.content_manager.detectors.source1 import Source1Detector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.source1_gameinfo_provider import Source1GameInfoProvider
from SourceIO.library.shared.content_manager.providers.vpk_provider import VPKContentProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('Portal2Detector')


class Portal2Detector(Source1Detector):
    @classmethod
    def scan(cls, path: TinyPath) -> list[ContentProvider]:
        portal2_root = None
        portal2_exe = backwalk_file_resolver(path, 'portal2.exe')
        if portal2_exe is not None:
            portal2_root = portal2_exe.parent
        if portal2_root is None:
            return []
        providers = {}
        initial_mod_gi_path = backwalk_file_resolver(path, "gameinfo.txt")
        if initial_mod_gi_path is not None:
            cls.add_provider(Source1GameInfoProvider(initial_mod_gi_path), providers)
        for vpk_path in initial_mod_gi_path.parent.glob("*.vpk"):
            with vpk_path.open("rb") as f:
                if f.read(4) != b"\x34\x12\xAA\x55":
                    continue
            cls.add_provider(VPKContentProvider(vpk_path), providers)

        portal2_mod_gi_path = portal2_root / "portal2/gameinfo.txt"
        if initial_mod_gi_path != portal2_mod_gi_path:
            cls.add_provider(Source1GameInfoProvider(portal2_mod_gi_path), providers)
        for vpk_path in portal2_mod_gi_path.parent.glob("*.vpk"):
            with vpk_path.open("rb") as f:
                if f.read(4) != b"\x34\x12\xAA\x55":
                    continue
            cls.add_provider(VPKContentProvider(vpk_path), providers)

        cls.register_common(portal2_root, providers)
        return list(providers.values())
