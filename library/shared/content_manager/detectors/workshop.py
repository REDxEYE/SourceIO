"""Detector for content selected straight out of a Steam Workshop folder.

Workshop items live next to the games they belong to, in a layout that records
which game they are for::

    <library>/steamapps/workshop/content/<app id>/<item id>/<files>
    <library>/steamapps/appmanifest_<app id>.acf   -> "installdir"
    <library>/steamapps/common/<installdir>/       -> the game itself

Nothing else in the tree tells us which game an asset belongs to, so opening a
workshop file directly used to leave the content manager with no game mounted at
all (``steam_id`` 0). This walks that layout backwards: read the app id out of the
path, resolve the game through its manifest, hand the game off to whichever
detector normally handles it, and finally mount the workshop item itself so its
files take priority over the base game's.
"""
from typing import Collection

from SourceIO.library.archives.gma import check_gma
from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager.detectors.content_detector import ContentDetector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers.gma_provider import GMAContentProvider
from SourceIO.library.shared.content_manager.providers.loose_files import LooseFilesContentProvider
from SourceIO.library.shared.content_manager.providers.vpk_provider import VPKContentProvider
from SourceIO.library.utils.kv_parser import ValveKeyValueParser
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('WorkshopDetector')


class WorkshopDetector(ContentDetector):
    """Mounts the game a workshop item belongs to, plus the item itself."""

    @classmethod
    def game(cls) -> str:
        return 'Steam Workshop'

    @classmethod
    def find_game_root(cls, path: TinyPath) -> TinyPath | None:
        """Return the ``steamapps`` directory containing this workshop item."""
        info = cls._parse_workshop_path(path)
        if info is None:
            return None
        steamapps, _, _ = info
        return steamapps

    @classmethod
    def _parse_workshop_path(cls, path: TinyPath):
        """Split a path into ``(steamapps, app_id, item_dir)``, or None.

        Matches ``.../steamapps/workshop/content/<app id>/<item id>/...`` anywhere
        in the path, so it works whether the user picked the item folder or a file
        inside it.
        """
        parts = list(TinyPath(path).absolute().parts)
        for index in range(len(parts) - 4):
            if (parts[index + 1].lower(), parts[index + 2].lower()) != ('workshop', 'content'):
                continue
            if parts[index].lower() != 'steamapps':
                continue
            app_id_part = parts[index + 3]
            if not app_id_part.isdigit():
                continue
            steamapps = TinyPath('/'.join(parts[:index + 1]))
            # `.../content/<app id>/<item id>`; the user may have selected the item
            # folder itself or something inside it.
            item_dir = TinyPath('/'.join(parts[:index + 5]))
            return steamapps, int(app_id_part), item_dir
        return None

    @classmethod
    def _find_game_install(cls, steamapps: TinyPath, app_id: int) -> TinyPath | None:
        """Resolve an app id to its install directory via ``appmanifest_<id>.acf``.

        Workshop content normally sits in the same library as the game, but Steam
        allows them to diverge, so fall back to the other libraries listed in
        ``libraryfolders.vdf`` before giving up.
        """
        install_dir = cls._read_install_dir(steamapps, app_id)
        if install_dir is not None:
            return install_dir
        for library in cls._sibling_libraries(steamapps):
            install_dir = cls._read_install_dir(library, app_id)
            if install_dir is not None:
                logger.info(f'Game {app_id} lives in a different Steam library: {library}')
                return install_dir
        return None

    @classmethod
    def _read_install_dir(cls, steamapps: TinyPath, app_id: int) -> TinyPath | None:
        manifest = steamapps / f'appmanifest_{app_id}.acf'
        if not manifest.exists():
            return None
        try:
            parser = ValveKeyValueParser(manifest)
            parser.parse()
            _, app_state = parser.tree.top()
            install_name = app_state.get('installdir')
        except Exception as ex:
            logger.warn(f'Failed to parse {manifest}: {ex}')
            return None
        if not install_name:
            return None
        install_dir = steamapps / 'common' / str(install_name)
        return install_dir if install_dir.exists() else None

    @classmethod
    def _sibling_libraries(cls, steamapps: TinyPath) -> list[TinyPath]:
        """Other libraries listed in ``libraryfolders.vdf``, excluding this one.

        The registry lives in Steam's own install, which is where the *main*
        library's ``steamapps`` is; a secondary library only carries a per-folder
        ``libraryfolder.vdf`` (singular) that has no cross-references. So the
        registry is only reachable when the item is in the main library, and this
        returns nothing otherwise -- which is fine, since the game is
        almost always in the same library as its workshop content.
        """
        registry = steamapps / 'libraryfolders.vdf'
        if not registry.exists():
            return []
        try:
            parser = ValveKeyValueParser(registry)
            parser.parse()
            _, folders = parser.tree.top()
        except Exception as ex:
            logger.warn(f'Failed to parse {registry}: {ex}')
            return []
        libraries = []
        for _, entry in folders.items():
            # Modern Steam nests a block per library; very old versions stored a
            # bare path string.
            library_path = entry.get('path') if hasattr(entry, 'get') else entry
            if not library_path:
                continue
            library = TinyPath(str(library_path)) / 'steamapps'
            if library != steamapps and library.exists():
                libraries.append(library)
        return libraries

    @classmethod
    def scan(cls, path: TinyPath) -> tuple[Collection[ContentProvider] | None, TinyPath | None]:
        info = cls._parse_workshop_path(path)
        if info is None:
            return None, None
        steamapps, app_id, item_dir = info

        game_install = cls._find_game_install(steamapps, app_id)
        if game_install is None:
            logger.warn(f'Found workshop content for app {app_id} but the game is not installed')
            return None, None

        # Let the game's own detector mount it, so a workshop item ends up with
        # exactly the content set that opening the game directly would give.
        providers, game_root = cls._scan_game(game_install, app_id)
        if providers is None:
            return None, None

        providers = set(providers)
        cls._add_workshop_item(item_dir, app_id, providers)
        logger.info(f'Mounted workshop item {item_dir.name} for {cls._app_name(app_id)}')
        return providers, game_root or game_install

    @classmethod
    def _scan_game(cls, game_install: TinyPath, app_id: int):
        """Run the game's normal detector against its install directory.

        Imported lazily because the registry module imports this one.
        """
        from SourceIO.library.shared.content_manager.detectors import GAME_DETECTORS
        for probe in cls._probe_paths(game_install):
            for detector in GAME_DETECTORS:
                if isinstance(detector, WorkshopDetector):
                    continue  # would recurse: the install is under a Steam library
                providers, root = detector.scan(probe)
                if providers:
                    logger.info(f'Workshop game {app_id} detected as {detector.game()}')
                    return providers, root
        logger.warn(f'No detector recognized {game_install}')
        return None, None

    @classmethod
    def _probe_paths(cls, game_install: TinyPath) -> list[TinyPath]:
        """Paths to offer the detectors, deepest-useful first.

        Detectors locate a game by walking *up* from the given path looking for
        marker files, so handing them the install root cannot find markers that live
        in a subdirectory. Half-Life: Alyx is the case in point -- its detector wants
        ``hlvr``, which is under ``game/``. Offer each mod-like subdirectory as well
        as the root itself.
        """
        probes = []
        for child in sorted(game_install.iterdir()) if game_install.exists() else ():
            if not child.is_dir():
                continue
            # A mod directory is where gameinfo lives; `game/` is Source 2's wrapper
            # around them, so descend one more level there.
            if (child / 'gameinfo.txt').exists() or (child / 'gameinfo.gi').exists():
                probes.append(child / 'gameinfo.txt')
            for grandchild in sorted(child.iterdir()) if child.is_dir() else ():
                if grandchild.is_dir() and ((grandchild / 'gameinfo.gi').exists() or
                                            (grandchild / 'gameinfo.txt').exists()):
                    probes.append(grandchild)
        probes.append(game_install)
        return probes

    @classmethod
    def _add_workshop_item(cls, item_dir: TinyPath, app_id: int, providers: set[ContentProvider]):
        """Mount a workshop item's own files.

        Items ship either as loose files, as archives (``.gma`` for Garry's Mod,
        ``.vpk`` for Source 2 games), or as ``*_legacy.bin`` -- an older
        LZMA-wrapped container that :func:`check_gma` correctly rejects and that
        there is no reader for, so those are skipped with a warning.
        """
        if not item_dir.exists():
            return
        steam_id = SteamAppId(app_id)
        found_archive = False
        for entry in sorted(item_dir.iterdir()):
            if entry.suffix == '.gma':
                if check_gma(entry):
                    cls.add_provider(GMAContentProvider(entry, steam_id), providers)
                    found_archive = True
                else:
                    logger.warn(f'{entry.name} is not a valid GMA archive, skipping')
            elif entry.suffix == '.vpk':
                # A split archive is opened through its `_dir` index; only fall back
                # to a bare .vpk when there is no index alongside it.
                if entry.stem.endswith('_dir') or not any(item_dir.glob('*_dir.vpk')):
                    cls.add_provider(VPKContentProvider(entry, steam_id), providers)
                    found_archive = True
            elif entry.name.endswith('_legacy.bin'):
                logger.warn(f'{entry.name} uses the legacy compressed workshop format, which is not supported')
        if not found_archive:
            # Nothing recognized as an archive: treat the folder as loose content.
            cls.add_provider(LooseFilesContentProvider(item_dir, steam_id), providers)

    @classmethod
    def _app_name(cls, app_id: int) -> str:
        try:
            return SteamAppId(app_id).name
        except ValueError:
            return f'app {app_id}'
