from typing import Collection

from SourceIO.library.global_config import GoldSrcConfig
from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager.detectors.content_detector import ContentDetector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.goldsrc_content_provider import (GoldSrcContentProvider,
                                                                                        GoldSrcWADContentProvider)
from SourceIO.library.utils import backwalk_file_resolver, TinyPath


class GoldSrcDetector(ContentDetector):
    @classmethod
    def game(cls) -> str:
        return "Half-Life"

    @classmethod
    def scan(cls, path: TinyPath) -> tuple[Collection[ContentProvider] | None, TinyPath | None]:
        hl_root = None
        hl_exe = backwalk_file_resolver(path, 'hl.exe')
        if hl_exe is not None:
            hl_root = hl_exe.parent
        if hl_root is None:
            return None, None
        mod_name = path.relative_to(hl_root).parts[0]
        providers = set()
        folder = hl_root / mod_name
        if (folder / 'liblist.gam').exists():
            cls.add_provider(GoldSrcContentProvider(folder, SteamAppId.HALF_LIFE), providers)
        for default_resource in ('decals.wad', 'halflife.wad', 'liquids.wad', 'xeno.wad'):
            if (folder / default_resource).exists():
                cls.add_provider(GoldSrcWADContentProvider(folder / default_resource, SteamAppId.HALF_LIFE), providers)
        if (hl_root / (mod_name + "_hd")).exists() and GoldSrcConfig().use_hd:
            cls.add_provider(GoldSrcContentProvider(hl_root / (mod_name + "_hd"), SteamAppId.HALF_LIFE), providers)
        return providers, hl_root
