from typing import Iterator, Optional, Any

from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager.provider import ContentProvider, is_relative_to
from SourceIO.library.shared.content_manager.providers import register_provider
from SourceIO.library.shared.content_manager.providers.loose_files import LooseFilesContentProvider
from SourceIO.library.shared.content_manager.providers.vpk_provider import VPKContentProvider
from SourceIO.library.utils import Buffer, FileBuffer, TinyPath
from SourceIO.library.utils.s1_keyvalues import KVParser
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('GameInfoProvider')


class Source1GameInfoProvider(ContentProvider):
    def __init__(self, filepath: TinyPath):
        super().__init__(filepath)
        with FileBuffer(filepath, "r") as f:
            header, gameinfo_data = KVParser(filepath, f.read_ascii_string()).parse()
        if header != "gameinfo":
            raise ValueError("Invalid gameinfo header")
        self.filesystem: dict[str, Any] = gameinfo_data["filesystem"]
        self._steamapp_id = SteamAppId(int(self.filesystem.get("steamappid", 0)))
        self.mount: list[ContentProvider] = []

        mods_folder = self.root.parent
        for search_path_type, search_paths in self.filesystem.get("searchpaths", {}).items():
            if isinstance(search_paths, str):
                search_paths = [search_paths]
            if search_path_type.lower() not in ["game", "mod", "platform", "gamebin", "vpk"]:
                logger.debug(
                    f"Skipping mounting {search_paths!r} as is not one of supported mount types: {search_path_type}")
                continue
            for search_path in search_paths:
                if "all_source_engine_paths" in search_path.lower():
                    search_path = search_path.lower().replace("|all_source_engine_paths|", "")
                elif "gameinfo_path" in search_path.lower():
                    search_path = TinyPath(search_path.replace("|gameinfo_path|", self.root.stem + "/"))
                elif search_path.endswith("*"):
                    logger.warn(f"Wildcard search path is not supported: {search_path}")
                    continue
                if search_path.endswith(".vpk"):
                    tmp = TinyPath(search_path)
                    if (mods_folder / tmp.with_name(tmp.stem + "_dir")).resolve().exists():
                        search_path = tmp.with_name(tmp.stem + "_dir")
                    else:
                        search_path = TinyPath(search_path)
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

    @property
    def name(self) -> str:
        return self.filesystem.get("game", self.root.stem)

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
