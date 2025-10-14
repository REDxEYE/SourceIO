from typing import Collection

from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager.detectors import ContentDetector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.zip_content_provider import ZIPContentProvider
from SourceIO.library.utils import TinyPath, backwalk_file_resolver


class CallOfDutyModernWarfare2Detector(ContentDetector):
    @classmethod
    def find_game_root(cls, path: TinyPath) -> TinyPath | None:
        exe_path = backwalk_file_resolver(path, "iw4sp.exe")
        if exe_path is not None:
            return exe_path.parent
        return None

    @classmethod
    def game(cls) -> str:
        return "Call of Duty: Modern Warfare 2"

    @classmethod
    def scan(cls, path: TinyPath) -> tuple[Collection[ContentProvider] | None, TinyPath | None]:
        root = cls.find_game_root(path)
        if root is None:
            return None, None

        providers = set()

        main_archives_folder = root / "main"

        for arc in main_archives_folder.glob("*.iwd"):
            if arc.is_file():
                cls.add_provider(ZIPContentProvider(arc, SteamAppId.CALL_OF_DUTY_MV2), providers)

        zone_archives_folder = root / "zone"

        # for language in zone_archives_folder.iterdir():
        #     if not language.is_dir():
        #         continue
        #     for arc in language.glob("*.ff"):
        #         if arc.is_file():
        #             cls.add_provider(FastFileProvider(arc, SteamAppId.CALL_OF_DUTY_MV2), providers)

        return providers, main_archives_folder