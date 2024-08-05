from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.utils.path_utilities import backwalk_file_resolver
from SourceIO.library.global_config import GoldSrcConfig
from SourceIO.library.shared.content_manager.detectors.content_detector import ContentDetector
from SourceIO.library.shared.content_manager.providers.goldsrc_content_provider import (GoldSrcContentProvider,
                                                                                        GoldSrcWADContentProvider)
from SourceIO.library.utils.tiny_path import TinyPath


class GoldSrcDetector(ContentDetector):

    @classmethod
    def scan(cls, path: TinyPath) -> dict[str, ContentProvider]:
        hl_root = None
        hl_exe = backwalk_file_resolver(path, 'hl.exe')
        if hl_exe is not None:
            hl_root = hl_exe.parent
        if hl_root is None:
            return {}
        mod_name = path.relative_to(hl_root).parts[0]
        content_providers = {}
        folder = hl_root / mod_name
        if (folder / 'liblist.gam').exists():
            content_providers[folder.stem] = GoldSrcContentProvider(folder)
        for default_resource in ('decals.wad', 'halflife.wad', 'liquids.wad', 'xeno.wad'):
            if (folder / default_resource).exists():
                content_providers[f'{folder.stem}_{default_resource}'] = GoldSrcWADContentProvider(
                    folder / default_resource)
        if folder.stem.endswith('_hd') and GoldSrcConfig().use_hd:
            content_providers[f'{folder.stem}_{default_resource}'] = GoldSrcContentProvider(folder)
        return content_providers
