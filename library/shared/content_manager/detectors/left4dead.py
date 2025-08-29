from typing import Collection

from SourceIO.library.shared.content_manager.detectors.source1 import Source1Detector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.source1_gameinfo_provider import Source1GameInfoProvider
from SourceIO.library.shared.content_manager.providers.vpk_provider import VPKContentProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath


class Left4DeadDetector(Source1Detector):

    @classmethod
    def game(cls) -> str:
        return "Left 4 Dead"


    @classmethod
    def scan(cls, path: TinyPath) -> tuple[Collection[ContentProvider] | None, TinyPath | None]:
        game_root = None
        game_exe = backwalk_file_resolver(path, 'left4dead.exe')
        if game_exe is not None:
            game_root = game_exe.parent
        if game_root is None:
            return None, None
        providers = set()
        initial_mod_gi_path = backwalk_file_resolver(path, "gameinfo.txt")
        if initial_mod_gi_path is not None:
            cls.add_provider(Source1GameInfoProvider(initial_mod_gi_path), providers)
        for vpk_path in initial_mod_gi_path.parent.glob("*.vpk"):
            with vpk_path.open("rb") as f:
                if f.read(4) != b"\x34\x12\xAA\x55":
                    continue
            cls.add_provider(VPKContentProvider(vpk_path), providers)

        portal2_mod_gi_path = game_root / "left4dead/gameinfo.txt"
        if initial_mod_gi_path != portal2_mod_gi_path:
            cls.add_provider(Source1GameInfoProvider(portal2_mod_gi_path), providers)
        for vpk_path in portal2_mod_gi_path.parent.glob("*.vpk"):
            with vpk_path.open("rb") as f:
                if f.read(4) != b"\x34\x12\xAA\x55":
                    continue
            cls.add_provider(VPKContentProvider(vpk_path), providers)

        cls.register_common(game_root, providers)
        return providers, game_root
