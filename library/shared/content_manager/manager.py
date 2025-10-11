from collections import Counter, OrderedDict
from hashlib import md5
from typing import Optional, TypeVar, Union, Iterator, Hashable

from SourceIO.library.shared.app_id import SteamAppId
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
META_CACHE_SIZE = 200_000

K = TypeVar('K', bound=Hashable)
T = TypeVar('T')


class _LRU(OrderedDict[K, T]):
    """Minimal LRU with O(1) move-to-end on get/set."""

    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize

    def get(self, key, default=None) -> T | None:
        v = super().get(key, default)
        if v is not default:
            self.move_to_end(key)
        return v

    def set(self, key:K, value:T):
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.maxsize:
            self.popitem(last=False)


def get_loose_file_fs_root(path: TinyPath):
    return get_mod_path(path)


class ContentManager(ContentProvider, metaclass=SingletonMeta):

    def __init__(self):
        super().__init__(TinyPath("."))
        self.children: list[ContentProvider] = []
        self._steam_id = -1
        self._cache: _LRU[TinyPath, Buffer] = _LRU(MAX_CACHE_SIZE)
        self._exists_cache: _LRU[TinyPath, bool] = _LRU(META_CACHE_SIZE)
        self._owner_cache: _LRU[TinyPath, ContentProvider] = _LRU(META_CACHE_SIZE)

    @property
    def root(self) -> TinyPath:
        return self.filepath

    @property
    def name(self) -> str:
        return "ContentManager"

    def check(self, filepath: TinyPath) -> bool:
        """Fast existence check with normalized keys and owner short-circuit."""
        if filepath.is_absolute():
            return filepath.exists()
        k = self._key(filepath)
        cached = self._exists_cache.get(k)
        if cached is not None:
            return cached
        owner = self._owner_cache.get(k)
        if owner is not None and owner.check(k):
            self._note_hit(k, owner)
            return True
        for child in self.children:
            if child.check(k):
                self._note_hit(k, child)
                return True
        self._note_miss(k)
        return False

    def get_provider_from_path(self, filepath):
        """Return the owning provider if the path resolves."""
        k = self._key(filepath)
        if not self.check(k):
            return None
        owner = self._owner_cache.get(k)
        if owner is not None:
            return owner
        for child in self.children:
            if child.check(k):
                self._note_hit(k, child)
                return child
        return None

    def get_steamid_from_asset(self, asset_path: TinyPath) -> SteamAppId | None:
        """Return provider that owns the asset, if any."""
        k = self._key(asset_path)
        if k is None:
            return None
        owner = self._owner_cache.get(k)
        if owner is not None and owner.check(k):
            return owner.steam_id
        for child in self.children:
            if child.check(k):
                self._note_hit(k, child)
                return child.steam_id
        self._note_miss(k)
        return None

    def _find_steam_appid(self, path: TinyPath):
        """Populate self._steam_id by scanning for steam_appid.txt."""
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
        """Return a relative path under any child root, if resolvable."""
        if not filepath.is_absolute():
            return filepath
        for provider in self.children:
            if (rel_path := provider.get_relative_path(filepath)) is not None:
                return rel_path
        return None

    def scan_for_content(self, scan_path: TinyPath):
        """Discover and mount providers for the given path."""
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
        """Mount a child provider and invalidate metadata caches."""
        if child not in self.children:
            self.children.append(child)
            self._clear_meta_caches()

    def glob(self, pattern: str) -> Iterator[tuple[TinyPath, Buffer]]:
        """Yield all matches across children."""
        for child in self.children:
            yield from child.glob(pattern)


    def find_file(self, filepath: TinyPath, do_not_cache=False) -> Buffer | None:
        """Find and optionally cache file buffers with owner/exists LRU metadata."""
        if filepath.is_absolute():
            if filepath.exists():
                return FileBuffer(filepath)
            return None
        k = self._key(filepath)
        buf = self._cache.get(k)
        if buf is not None and not buf.closed:
            buf.seek(0)
            return buf
        logger.debug(f'Requesting {k} file')
        owner = self._owner_cache.get(k)
        if owner is not None:
            file = owner.find_file(k)
            if file is not None:
                self._note_hit(k, owner)
                if do_not_cache:
                    return file
                self._cache.set(k, file)
                return file
            self._owner_cache.pop(k, None)
            self._exists_cache.pop(k, None)
        for child in self.children:
            file = child.find_file(k)
            if file is not None:
                logger.debug(f'Found in {child}!')
                self._note_hit(k, child)
                if do_not_cache:
                    return file
                self._cache.set(k, file)
                return file
        self._note_miss(k)
        return None

    # TODO: MAYBE DEPRECATED
    def serialize(self):
        """Serialize mounted providers."""
        serialized = {}
        for provider in self.children:
            name = provider.unique_name.replace('\'', '').replace('\"', '').replace(' ', '_')
            info = {"name": name, "path": str(provider.filepath)}
            serialized[md5(name.encode("utf8")).hexdigest()] = info

        return serialized

    # TODO: MAYBE DEPRECATED
    def deserialize(self, data: dict[str, Union[str, dict]]):
        """Recreate mounts from serialized data."""
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
                except (ValueError, FileNotFoundError) as ex:
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
        """Reset mounts and caches."""
        self.children.clear()
        self._steam_id = -1
        self._cache.clear()
        self._clear_meta_caches()

    @property
    def steam_id(self):
        if self._steam_id != -1:
            return self._steam_id
        used_appids = Counter([child.steam_id for child in self.children if child.steam_id > 0])
        if len(used_appids) == 0:
            return 0
        return used_appids.most_common(1)[0][0]

    def get_content_provider_from_asset_path(self, asset_path: TinyPath) -> Optional[ContentProvider]:
        """Return owning provider using existence checks to avoid opening files."""
        for content_provider in self.children:
            if content_provider.find_file(asset_path):
                return content_provider
        return None

    def _key(self, path: TinyPath) -> TinyPath:
        """Normalize to a relative key when possible."""
        if path.is_absolute():
            rel = self.get_relative_path(path)
            return rel if rel is not None else path
        return path

    def _note_hit(self, k: TinyPath, owner: ContentProvider) -> None:
        """Mark path as existing and owned by provider."""
        self._exists_cache.set(k, True)
        self._owner_cache.set(k, owner)

    def _note_miss(self, k: TinyPath) -> None:
        """Mark path as non-existent."""
        self._exists_cache.set(k, False)
        self._owner_cache.pop(k, None)

    def _clear_meta_caches(self) -> None:
        """Clear metadata caches."""
        self._exists_cache.clear()
        self._owner_cache.clear()
