from typing import Iterator, Optional, Any

from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager.provider import ContentProvider, is_relative_to
from SourceIO.library.shared.content_manager.providers import register_provider
from SourceIO.library.shared.content_manager.providers.loose_files import LooseFilesContentProvider
from SourceIO.library.shared.content_manager.providers.vpk_provider import VPKContentProvider
from SourceIO.library.utils import Buffer, TinyPath
from SourceIO.library.utils.s1_keyvalues import KVParser
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('GameInfoProvider')


class Source2GameInfoProvider(ContentProvider):
    def __init__(self, filepath: TinyPath, steamapp_id: SteamAppId = SteamAppId.UNKNOWN):
        super().__init__(filepath)
        with filepath.open('r', encoding="utf8") as f:
            header, gameinfo_data = KVParser(filepath, f.read()).parse()
            assert header == "gameinfo"
            self.data = gameinfo_data
        if header != "gameinfo":
            raise ValueError("Invalid gameinfo header")
        self.filesystem: dict[str, Any] = gameinfo_data["filesystem"]
        self._steamapp_id = steamapp_id
        self.mount: list[ContentProvider] = []

        mods_folder = self.root.parent
        for search_path_type, search_paths in self.filesystem.get("searchpaths", {}).items():
            if search_path_type.lower() not in ["game", "mod", "platform", "gamebin", "vpk", "addonroot"]:
                logger.debug(
                    f"Skipping mounting {search_paths!r} as is not one of supported mount types: {search_path_type}")
                continue
            for search_path in search_paths:
                if "all_source_engine_paths" in search_path.lower():
                    search_path = search_path.lower().replace("|all_source_engine_paths|", "")
                elif "gameinfo_path" in search_path.lower():
                    search_path = TinyPath(search_path.replace("|gameinfo_path|", self.root.stem + "/"))

                mod_folder = (mods_folder / search_path).resolve()
                if mod_folder.exists():
                    if mod_folder.is_file():
                        if mod_folder.suffix == ".vpk":
                            mod_provider = register_provider(VPKContentProvider(mod_folder, self._steamapp_id))
                        else:
                            logger.warn("Only VPK/HFS/GMA supported to be mounted as files")
                            continue
                    else:
                        mod_provider = register_provider(LooseFilesContentProvider(mod_folder, self._steamapp_id))
                    if mod_provider not in self.mount:
                        logger.info(f"Mounted: {mod_provider}")
                        self.mount.append(mod_provider)
                    if not mod_folder.is_file():
                        for file in mod_folder.iterdir():
                            if file.is_file() and file.suffix == ".vpk" and "_dir" in file.name:
                                mod_provider = register_provider(VPKContentProvider(file, self._steamapp_id))
                                if mod_provider not in self.mount:
                                    logger.info(f"Mounted: {mod_provider}")
                                    self.mount.append(mod_provider)

    def check(self, filepath: TinyPath) -> bool:
        for mount in self.mount:
            if mount.check(filepath):
                return True
        return False

    def get_relative_path(self, filepath: TinyPath):
        if is_relative_to(filepath, self.root):
            rel_path = filepath.relative_to(self.root)
            if self.check(rel_path):
                return rel_path

    def get_provider_from_path(self, filepath):
        if self.check(filepath):
            return self

    def get_steamid_from_asset(self, asset_path: TinyPath) -> SteamAppId | None:
        if self.check(asset_path):
            return self.steam_id

    @property
    def name(self) -> str:
        return self.filesystem.get("game", self.root.stem)

    def find_file(self, filepath: TinyPath) -> Optional[Buffer]:
        for mount in self.mount:
            file = mount.find_file(filepath)
            if file is not None:
                return file
        return None

    def glob(self, pattern: str) -> Iterator[tuple[TinyPath, Buffer]]:
        for mount in self.mount:
            yield from mount.glob(pattern)

    @property
    def steam_id(self) -> SteamAppId:
        return self._steamapp_id

    @property
    def root(self) -> TinyPath:
        return self.filepath.parent
