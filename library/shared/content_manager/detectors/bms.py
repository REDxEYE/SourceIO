from typing import Collection
from SourceIO.library.shared.content_manager.detectors.source1 import Source1Detector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.source1_gameinfo_provider import Source1GameInfoProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath


class BlackMesaDetector(Source1Detector):

    @classmethod
    def game(cls) -> str:
        return "Black Mesa"

    GAME_MARKERS = ('bin/bms.fgd',)

    @classmethod
    def find_game_root(cls, path: TinyPath) -> TinyPath | None:
        for marker in cls.GAME_MARKERS:
            found = backwalk_file_resolver(path, marker)
            if found is not None:
                if found.name != TinyPath(marker).name:
                    continue
                return found.parent.parent
        return None

    @classmethod
    def scan(cls, path: TinyPath) -> tuple[Collection[ContentProvider] | None, TinyPath | None]:
        gmod_root = cls.find_game_root(path)
        if gmod_root is None:
            return None, None
        bms_dir = gmod_root / 'bms'

        providers = set()
        initial_mod_gi_path = backwalk_file_resolver(path, "gameinfo.txt")
        if initial_mod_gi_path is not None:
            cls.add_provider(Source1GameInfoProvider(initial_mod_gi_path), providers)

        # gameinfo.txt lives in the mod directory, not the game root.
        garrysmod_mod_gi_path = bms_dir / "gameinfo.txt"
        if initial_mod_gi_path != garrysmod_mod_gi_path and garrysmod_mod_gi_path.exists():
            cls.add_provider(Source1GameInfoProvider(garrysmod_mod_gi_path), providers)

        cls.register_common(gmod_root, providers)
        return providers, gmod_root
