from SourceIO.library.shared.content_manager.detectors.source1 import Source1Detector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.source1_gameinfo_provider import Source1GameInfoProvider
from SourceIO.library.shared.content_manager.providers.vpk_provider import VPKContentProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('InfraDetector')


class InfraDetector(Source1Detector):
    @classmethod
    def scan(cls, path: TinyPath) -> list[ContentProvider]:
        infra_root = None
        infra_exe = backwalk_file_resolver(path, 'infra.exe')
        if infra_exe is not None:
            infra_root = infra_exe.parent
        if infra_root is None:
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

        infra_mod_gi_path = infra_root / "infra/gameinfo.txt"
        if initial_mod_gi_path != infra_mod_gi_path:
            cls.add_provider(Source1GameInfoProvider(infra_mod_gi_path), providers)
        for vpk_path in infra_mod_gi_path.parent.glob("*.vpk"):
            with vpk_path.open("rb") as f:
                if f.read(4) != b"\x34\x12\xAA\x55":
                    continue
            cls.add_provider(VPKContentProvider(vpk_path), providers)

        cls.register_common(infra_root, providers)
        return list(providers.values())
