from typing import Collection

from SourceIO.library.shared.content_manager.detectors.source1 import Source1Detector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.source1_gameinfo_provider import Source1GameInfoProvider
from SourceIO.library.utils import backwalk_file_resolver, TinyPath


class SourceMod(Source1Detector):

    @classmethod
    def game(cls) -> str:
        return "SourceMod mod"

    @classmethod
    def find_game_root(cls, path: TinyPath) -> TinyPath | None:
        smods_dir = backwalk_file_resolver(path, 'sourcemods')
        if smods_dir is not None and path.is_relative_to(smods_dir):
            mod_name = path.relative_to(smods_dir).parts[0]
            return smods_dir / mod_name
        return None

    @classmethod
    def scan(cls, path: TinyPath) -> tuple[Collection[ContentProvider] | None, TinyPath | None]:
        mod_root = cls.find_game_root(path)
        if mod_root is None:
            return None, None
        smods_dir = mod_root.parent

        content_providers = set()
        initial_mod_gi_path = backwalk_file_resolver(smods_dir, "gameinfo.txt")
        if initial_mod_gi_path is not None:
            cls.add_provider(Source1GameInfoProvider(initial_mod_gi_path), content_providers)

        other_mods_gi_path = backwalk_file_resolver(path, "gameinfo.txt")
        if initial_mod_gi_path is not None and other_mods_gi_path != initial_mod_gi_path:
            cls.add_provider(Source1GameInfoProvider(other_mods_gi_path), content_providers)
        return content_providers, mod_root
