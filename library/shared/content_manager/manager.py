from collections import Counter
from hashlib import md5
from typing import Optional, TypeVar, Union

from SourceIO.library.shared.content_manager.detectors import detect_game
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers import register_provider
from SourceIO.library.shared.content_manager.providers.hfs_provider import HFS1ContentProvider, HFS2ContentProvider
from SourceIO.library.shared.content_manager.providers.loose_files import LooseFilesContentProvider
from SourceIO.library.shared.content_manager.providers.source1_gameinfo_provider import Source1GameInfoProvider
from SourceIO.library.shared.content_manager.providers.source2_gameinfo_provider import Source2GameInfoProvider
from SourceIO.library.shared.content_manager.providers.vpk_provider import VPKContentProvider
from SourceIO.library.shared.content_manager.providers.zip_content_provider import ZIPContentProvider
from SourceIO.library.utils import Buffer, FileBuffer, TinyPath, backwalk_file_resolver
from SourceIO.library.utils.path_utilities import get_mod_path
from SourceIO.library.utils.singleton import SingletonMeta
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('ContentManager')

AnyContentDetector = TypeVar('AnyContentDetector', bound='ContentDetector')
AnyContentProvider = TypeVar('AnyContentProvider', bound='ContentProvider')

MAX_CACHE_SIZE = 16


def get_loose_file_fs_root(path: TinyPath):
    return get_mod_path(path)


class ContentManager(ContentProvider, metaclass=SingletonMeta):

    @property
    def root(self) -> TinyPath:
        return self.filepath

    @property
    def name(self) -> str:
        return "ContentManager"

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
        if asset_path is None:
            return None
        for child in self.children:
            if provider := child.get_steamid_from_asset(asset_path):
                return provider

    def __init__(self):
        super().__init__(TinyPath("."))
        self.children: list[ContentProvider] = []
        self._steam_id = -1
        self._cache: dict[TinyPath, Buffer] = {}

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
        if root_path:
            gameinfos = list(root_path.glob('*gameinfo.txt'))
            if not gameinfos:
                # for unknown gameinfo like gameinfo_srgb, they are confusing content manager steam id thingie
                gameinfos = root_path.glob('gameinfo_*.txt')
            for gameinfo in gameinfos:
                try:
                    sub_manager = Source1GameInfoProvider(gameinfo)
                except ValueError as ex:
                    logger.exception(f"Failed to parse gameinfo for {gameinfo}", ex)
                    continue
                self.add_child(sub_manager)
                logger.info(f'Registered provider for {root_path.stem}')

            gameinfos = root_path.glob('*gameinfo*.gi')
            for gameinfo in gameinfos:
                sub_manager = Source2GameInfoProvider(gameinfo)
                self.add_child(sub_manager)
        else:
            if root_path.is_dir():
                self.add_child(LooseFilesContentProvider(root_path))
            else:
                root_path = root_path.parent
                self.add_child(LooseFilesContentProvider(root_path))

    def add_child(self, child: ContentProvider):
        if child not in self.children:
            self.children.append(child)

    def glob(self, pattern: str):
        for child in self.children:
            yield from child.glob(pattern)

    def find_file(self, filepath: TinyPath) -> Buffer | None:
        if filepath.is_absolute():
            if filepath.exists():
                return FileBuffer(filepath)
            return None
        if (buffer := self._cache.get(filepath, None)) is not None:
            if not buffer.closed:
                buffer.seek(0)
                return buffer
        logger.debug(f'Requesting {filepath} file')
        for child in self.children:
            if (file := child.find_file(filepath)) is not None:
                logger.debug(f'Found in {child}!')
                self._cache[filepath] = file
                if len(self._cache) > MAX_CACHE_SIZE:
                    self._cache.pop(next(iter(self._cache.keys())))
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
                if t_path.is_absolute():
                    full_path = t_path
                else:
                    prov = self.get_content_provider_from_asset_path(t_path)
                    full_path = prov.root / t_path
                with FileBuffer(full_path) as f:
                    bsp = open_bsp(t_path, f, self)
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
