from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager.providers.loose_files import LooseFilesContentProvider


class SBoxAddonProvider(LooseFilesContentProvider):

    @property
    def name(self) -> str:
        return f"S&Box {self.filepath.stem}"

    @property
    def steam_id(self):
        return SteamAppId.SBOX_STEAM_ID


class SBoxDownloadsProvider(LooseFilesContentProvider):

    @property
    def name(self) -> str:
        return f"S&Box {self.filepath.stem[:8]}"

    @property
    def steam_id(self):
        return SteamAppId.SBOX_STEAM_ID
