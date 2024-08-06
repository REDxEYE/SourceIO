from collections import Counter
from hashlib import md5
from typing import Optional, TypeVar, Union

from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.detectors import detect_game
from .providers import register_provider
from .providers.hfs_provider import HFS1ContentProvider, HFS2ContentProvider
from .providers.loose_files import LooseFilesContentProvider
from .providers.source1_gameinfo_provider import Source1GameInfoProvider
from .providers.source2_gameinfo_provider import Source2GameInfoProvider
from .providers.zip_content_provider import ZIPContentProvider
from ...utils.tiny_path import TinyPath
from ....library.utils.path_utilities import (backwalk_file_resolver,
                                              corrected_path, get_mod_path)
from ....logger import SourceLogMan
from ...utils import Buffer, FileBuffer
from ...utils.singleton import SingletonMeta
# from SourceIO.library.shared.content_manager.providers.zip_content_provider import ZIPContentProvider
# from SourceIO.library.shared.content_manager.providers.hfs_provider import HFS1ContentProvider, HFS2ContentProvider
# from SourceIO.library.shared.content_manager.providers.non_source_sub_manager import NonSourceContentProvider
# from SourceIO.library.shared.content_manager.providers.source1_content_provider import \
#     GameinfoContentProvider as Source1GameinfoContentProvider
# from SourceIO.library.shared.content_manager.providers.source2_content_provider import \
#     Gameinfo2ContentProvider as Source2GameinfoContentProvider
from SourceIO.library.shared.content_manager.providers.vpk_provider import VPKContentProvider

log_manager = SourceLogMan()
logger = log_manager.get_logger('ContentManager')

AnyContentDetector = TypeVar('AnyContentDetector', bound='ContentDetector')
AnyContentProvider = TypeVar('AnyContentProvider', bound='ContentProvider')


def get_loose_file_fs_root(path: TinyPath):
    return get_mod_path(path)


class ContentManager(metaclass=SingletonMeta):

    def check(self, filepath: TinyPath) -> bool:
        if filepath.is_absolute():
            return filepath.exists()
        for child in self.children:
            if child.check(filepath):
                return True
        return False

    def get_provider_from_path(self, filepath):
        if filepath.is_absolute():
            filepath = self.get_relative_path(filepath)
        for child in self.children:
            if provider := child.get_provider_from_path(filepath):
                return provider

    def get_steamid_from_asset(self, asset_path: TinyPath) -> ContentProvider | None:
        if asset_path.is_absolute():
            asset_path = self.get_relative_path(asset_path)
        for child in self.children:
            if provider := child.get_steamid_from_asset(asset_path):
                return provider

    def __init__(self):
        self.children: list[ContentProvider] = []
        self._steam_id = -1

    def _find_steam_appid(self, path: TinyPath):
        if self._steam_id != -1:
            return
        if path.is_file():
            path = path.parent
        file = backwalk_file_resolver(path, 'steam_appid.txt')
        if file is not None:
            with file.open('r') as f:
                try:
                    value = f.read().strip("\x00\n\t\r")
                    self._steam_id = int(value.strip())
                except Exception as e:
                    logger.exception(f'Failed to parse steam id due to: {e}')
                    self._steam_id = -1

    def get_relative_path(self, filepath: TinyPath):
        if not filepath.is_absolute():
            return filepath
        for provider in self.children:
            if (rel_path := provider.get_relative_path(filepath)) is not None:
                return rel_path
        return None

    def scan_for_content(self, scan_path: TinyPath):
        providers = detect_game(scan_path)
        if providers:
            for provider in providers:
                logger.info(f"Mounted: {provider}")
            self.children.extend(providers)
            return
        self._find_steam_appid(scan_path)
        if scan_path.suffix == '.vpk':
            if scan_path.exists():
                self.add_child(VPKContentProvider(scan_path))
                return

        root_path = get_loose_file_fs_root(scan_path)
        if root_path:
            self.add_child(LooseFilesContentProvider(root_path))
        raise NotImplementedError("TODO: Update other logic")
        if root_path:
            if root_path.stem in self.children:
                return
            else:
                gameinfos = list(root_path.glob('*gameinfo.txt'))
                if not gameinfos:
                    # for unknown gameinfo like gameinfo_srgb, they are confusing content manager steam id thingie
                    gameinfos = root_path.glob('gameinfo_*.txt')
                for gameinfo in gameinfos:
                    try:
                        sub_manager = Source1GameinfoContentProvider(gameinfo)
                    except ValueError as ex:
                        logger.exception(f"Failed to parse gameinfo for {gameinfo}", ex)
                        continue
                    if sub_manager.gameinfo.game == 'Titanfall':
                        self._titanfall_mode = True
                    self.content_providers[root_path.stem] = sub_manager
                    logger.info(f'Registered provider for {root_path.stem}')
                    for mod in sub_manager.get_search_paths():
                        if mod.parts[-1] == '*':
                            continue
                        self.scan_for_content(mod)

                gameinfos = root_path.glob('*gameinfo*.gi')
                for gameinfo in gameinfos:
                    sub_manager = Source2GameinfoContentProvider(gameinfo)
                    self.register_content_provider(root_path.stem, sub_manager)
                    for mod in sub_manager.get_search_paths():
                        self.scan_for_content(mod)
        elif 'download' in root_path.name:
            sub_manager = NonSourceContentProvider(root_path)
            self.content_providers[root_path.stem] = sub_manager
            logger.info(f'Registered provider for {root_path.stem}')
            self.scan_for_content(root_path.parent)
        else:
            if root_path.is_dir():
                self.register_content_provider(root_path.stem, NonSourceContentProvider(root_path))
            else:
                root_path = root_path.parent
                self.register_content_provider(root_path.stem, NonSourceContentProvider(root_path))

    def add_child(self, child: ContentProvider):
        if child not in self.children:
            self.children.append(child)

    def glob(self, pattern: str):
        for child in self.children:
            yield from child.glob(pattern)

    def find_file(self, filepath: TinyPath) -> Buffer | None:
        if filepath.is_absolute():
            return FileBuffer(filepath)
        logger.debug(f'Requesting {filepath} file')
        for child in self.children:
            if (file := child.find_file(filepath)) is not None:
                logger.debug(f'Found in {child}!')
                return file
        return None

    # TODO: MAYBE DEPRECATED
    def serialize(self):
        serialized = {}
        for provider in self.children:
            name = provider.unique_name.replace('\'', '').replace('\"', '').replace(' ', '_')
            info = {"name": name, "path": str(provider.filepath)}
            serialized[md5(name.encode("utf8")).hexdigest()] = info

        return serialized

    # TODO: MAYBE DEPRECATED
    def deserialize(self, data: dict[str, Union[str, dict]]):
        for name, item in data.items():
            name = item["name"]
            path = item["path"]
            t_path = TinyPath(path)
            if path.endswith('.vpk'):
                provider = VPKContentProvider(t_path)
                if provider not in self.children:
                    self.children.append(register_provider(provider))
            elif path.endswith('.pk3'):
                provider = ZIPContentProvider(t_path)
                if provider not in self.children:
                    self.children.append(register_provider(provider))
            elif path.endswith('.txt'):
                try:
                    provider = Source1GameInfoProvider(t_path)
                except ValueError as ex:
                    logger.exception(f"Failed to parse gameinfo for {t_path}", ex)
                    continue
                if provider not in self.children:
                    self.children.append(register_provider(provider))
            elif path.endswith('.gi'):
                provider = Source2GameInfoProvider(t_path)
                if provider not in self.children:
                    self.children.append(register_provider(provider))
            elif path.endswith('.bsp'):
                from ...source1.bsp.bsp_file import open_bsp
                with FileBuffer(path) as f:
                    bsp = open_bsp(path, f, self)
                    provider = bsp.get_lump('LUMP_PAK')
                if provider and provider not in self.children:
                    self.children.append(register_provider(provider))
            elif path.endswith('.hfs'):
                provider = HFS1ContentProvider(t_path)
                if provider not in self.children:
                    self.children.append(register_provider(provider))
            elif name == 'hfs':
                provider = HFS2ContentProvider(t_path)
                if provider not in self.children:
                    self.children.append(register_provider(provider))
            else:
                provider = LooseFilesContentProvider(t_path)
                if provider not in self.children:
                    self.children.append(register_provider(provider))

    def clean(self):
        self.children.clear()
        self._steam_id = -1

    @property
    def steam_id(self):
        if self._steam_id != -1:
            return self._steam_id
        used_appids = Counter([child.steam_id for child in self.children if child.steam_id > 0])
        if len(used_appids) == 0:
            return 0
        return used_appids.most_common(1)[0][0]

    def get_content_provider_from_asset_path(self, asset_path: TinyPath) -> Optional[ContentProvider]:
        for content_provider in self.children:
            if content_provider.find_file(asset_path):
                return content_provider
        return None
