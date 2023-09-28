from collections import Counter, OrderedDict
from hashlib import md5
from pathlib import Path
from typing import Dict, List, Optional, TypeVar, Union

from ....library.utils.path_utilities import (backwalk_file_resolver,
                                              corrected_path, get_mod_path)
from ....logger import SLoggingManager
from ...utils import Buffer
from ...utils.singleton import SingletonMeta
from .content_provider_base import ContentProviderBase
from .hfs_provider import HFS1ContentProvider, HFS2ContentProvider
from .non_source_sub_manager import NonSourceContentProvider
from .source1_content_provider import \
    GameinfoContentProvider as Source1GameinfoContentProvider
from .source2_content_provider import \
    Gameinfo2ContentProvider as Source2GameinfoContentProvider
from .vpk_provider import VPKContentProvider

log_manager = SLoggingManager()
logger = log_manager.get_logger('ContentManager')

AnyContentDetector = TypeVar('AnyContentDetector', bound='ContentDetectorBase')
AnyContentProvider = TypeVar('AnyContentProvider', bound='ContentProviderBase')


# backport
def is_relative_to(path: Path, *other):
    """Return True if the path is relative to another path or False.
    """
    try:
        path.relative_to(*other)
        return True
    except ValueError:
        return False


class ContentManager(metaclass=SingletonMeta):
    def __init__(self):
        self.detector_addons: List[AnyContentDetector] = []
        self.content_providers: OrderedDict[str, AnyContentProvider] = OrderedDict()
        self._titanfall_mode = False
        self._steam_id = -1
        self._register_supported_detectors()

    def _register_supported_detectors(self):
        from .content_detectors.gmod import GModDetector
        from .content_detectors.goldsrc import GoldSrcDetector
        from .content_detectors.hla import HLADetector
        from .content_detectors.robot_repair import RobotRepairDetector
        from .content_detectors.sbox import SBoxDetector
        from .content_detectors.sfm import SFMDetector
        from .content_detectors.source1_common import Source1Common
        from .content_detectors.sourcemod import SourceMod
        from .content_detectors.titanfall1 import TitanfallDetector
        from .content_detectors.vindictus import VindictusDetector
        from .content_detectors.source2 import Source2Detector
        from .content_detectors.cs2 import CS2Detector
        self.detector_addons.append(GoldSrcDetector())
        self.detector_addons.append(SBoxDetector())
        self.detector_addons.append(CS2Detector())
        self.detector_addons.append(HLADetector())
        self.detector_addons.append(RobotRepairDetector())
        self.detector_addons.append(Source2Detector())
        self.detector_addons.append(Source1Common())
        self.detector_addons.append(SFMDetector())
        self.detector_addons.append(VindictusDetector())
        self.detector_addons.append(TitanfallDetector())
        self.detector_addons.append(SourceMod())
        self.detector_addons.append(GModDetector())

    def _find_steam_appid(self, path: Path):
        if self._steam_id != -1:
            return
        if path.is_file():
            path = path.parent
        file = backwalk_file_resolver(path, 'steam_appid.txt')
        if file is not None:
            with file.open('r') as f:
                try:
                    value = f.read()
                    if value.find('\n') >= 0:
                        value = value[:value.find('\n')]
                    if value.find('\x00') >= 0:
                        value = value[:value.find('\x00')]
                    self._steam_id = int(value.strip())
                except Exception as e:
                    logger.exception(f'Failed to parse steam id due to: {e}')
                    self._steam_id = -1

    def register_content_provider(self, name: str, content_provider: AnyContentProvider):
        if name in self.content_providers:
            return
        for tmp in self.content_providers.values():
            if tmp.filepath == content_provider.filepath:
                return

        self.content_providers[name] = content_provider
        logger.info(f'Registered {content_provider.class_name()}({name!r}) provider for {content_provider.root.stem}')

    def get_relative_path(self, filepath: Path):
        # logger.debug(f'Trying to find root folder for {filepath.as_posix()!r}')
        for _, content_provider in self.content_providers.items():
            # logger.debug(f'\tChecking {content_provider.root.as_posix()!r}')
            content_provider: ContentProviderBase
            if filepath.is_absolute() and is_relative_to(filepath, content_provider.root):
                # logger.debug(f'\tMatch {content_provider.root.as_posix()}!')
                return filepath.relative_to(content_provider.root)
            elif not filepath.is_absolute() and content_provider.find_file(filepath):
                # logger.debug(f'\tMatch {content_provider.root.as_posix()}!')
                return filepath
        return None

    def scan_for_content(self, source_game_path: Union[str, Path]):
        source_game_path = Path(source_game_path)
        found_game = False
        for detector in self.detector_addons:
            for name, content_provider in detector.scan(source_game_path).items():
                self.register_content_provider(name, content_provider)
                found_game = True
        if source_game_path.is_file() and source_game_path.exists():
            if source_game_path.suffix == '.vpk':
                self.register_content_provider(f'{source_game_path.parent.stem}_{source_game_path.stem}',
                                               VPKContentProvider(source_game_path))
        if found_game:
            return
        if "*LANGUAGE*" in str(source_game_path):
            return
        self._find_steam_appid(source_game_path)
        if source_game_path.suffix == '.vpk':
            if self._titanfall_mode:
                if 'english' not in str(source_game_path):
                    return
            if f'{source_game_path.parent.stem}_{source_game_path.stem}' in self.content_providers:
                return
            vpk_path = source_game_path
            if vpk_path.exists():
                self.register_content_provider(f'{source_game_path.parent.stem}_{source_game_path.stem}',
                                               VPKContentProvider(vpk_path))
                return

        is_source, root_path = self.is_source_mod(source_game_path)
        if root_path.stem in self.content_providers:
            return
        if is_source:
            gameinfos = list(root_path.glob('*gameinfo.txt'))
            if not gameinfos:
                # for unknown gameinfo like gameinfo_srgb, they are confusing content manager steam id thingie
                gameinfos = root_path.glob('gameinfo_*.txt')
            for gameinfo in gameinfos:
                sub_manager = Source1GameinfoContentProvider(gameinfo)
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

    @staticmethod
    def is_source_mod(path: Path, second=False):
        if path.name == 'gameinfo.txt':
            path = path.parent
        if path.parts[-1] == '*':
            path = path.parent
        if "workshop" in path.parts:  # SFM detected
            path = Path(*path.parts[:path.parts.index('workshop') + 1])
        gameinfos = list(path.glob('*gameinfo*.*'))
        if gameinfos:
            return True, path
        elif not second:
            return ContentManager.is_source_mod(get_mod_path(path), True)
        return False, path

    def glob(self, pattern: str):
        for content_provider in self.content_providers.values():
            yield from content_provider.glob(pattern)

    def find_file(self, filepath: Union[str, Path], additional_dir: str = None, extension: str = None, *,
                  silent=False) -> Optional[Buffer]:

        new_filepath = Path(str(filepath).replace('\\', '/').replace('//', '/').strip('\\/'))
        if additional_dir is not None:
            new_filepath = Path(additional_dir, new_filepath)
        if extension is not None:
            new_filepath = new_filepath.with_suffix(extension)
        if not silent:
            logger.info(f'Requesting {new_filepath} file')
        for mod, submanager in self.content_providers.items():
            file = submanager.find_file(new_filepath)
            if file is not None:
                if not silent:
                    logger.debug(f'Found in {mod}!')
                return file
        return None

    def find_path(self, filepath: Union[str, Path], additional_dir=None, extension=None, *, silent=False):

        new_filepath = Path(str(filepath).replace('\\', '/').replace('//', '/'))
        if additional_dir:
            new_filepath = Path(additional_dir, new_filepath)
        if extension:
            new_filepath = new_filepath.with_suffix(extension)
        if not silent:
            logger.info(f'Requesting {new_filepath} file')
        for mod, submanager in self.content_providers.items():
            file = submanager.find_path(new_filepath)
            if file is not None:
                if not silent:
                    logger.debug(f'Found in {mod}!')
                return corrected_path(file)
        return None

    def find_texture(self, filepath, *, silent=False):
        return self.find_file(filepath, 'materials', extension='.vtf', silent=silent)

    def find_material(self, filepath, *, silent=False):
        return self.find_file(filepath, 'materials', extension='.vmt', silent=silent)

    def serialize(self):
        serialized = {}
        for name, sub_manager in self.content_providers.items():
            name = name.replace('\'', '').replace('\"', '').replace(' ', '_')
            info = {"name": name, "path": str(sub_manager.filepath)}
            serialized[md5(name.encode("utf8")).hexdigest()] = info

        return serialized

    def deserialize(self, data: Dict[str, Union[str, dict]]):
        for name, item in data.items():
            name = item["name"]
            path = item["path"]

            if path.endswith('.vpk'):
                sub_manager = VPKContentProvider(Path(path))
                self.register_content_provider(name, sub_manager)
            elif path.endswith('.txt'):
                sub_manager = Source1GameinfoContentProvider(Path(path))
                if sub_manager.gameinfo.game == 'Titanfall':
                    self._titanfall_mode = True
                self.register_content_provider(name, sub_manager)
            elif path.endswith('.gi'):
                sub_manager = Source2GameinfoContentProvider(Path(path))
                self.register_content_provider(name, sub_manager)
            elif path.endswith('.bsp'):
                from ...source1.bsp.bsp_file import open_bsp
                bsp = open_bsp(path)
                pak_lump = bsp.get_lump('LUMP_PAK')
                if pak_lump:
                    self.register_content_provider(name, pak_lump)
            elif path.endswith('.hfs'):
                sub_manager = HFS1ContentProvider(Path(path))
                self.register_content_provider(name, sub_manager)
            elif name == 'hfs':
                sub_manager = HFS2ContentProvider(Path(path))
                self.register_content_provider(name, sub_manager)
            else:
                sub_manager = NonSourceContentProvider(Path(path))
                self.register_content_provider(name, sub_manager)

    def get_content_provider_from_path(self, filepath):
        filepath = Path(filepath)
        for name, content_provider in self.content_providers.items():
            if content_provider.filepath.is_file():
                cp_root = content_provider.filepath.parent
            else:
                cp_root = content_provider.filepath
            is_sm, fp_root = self.is_source_mod(filepath)
            if fp_root == cp_root:
                return content_provider
        return NonSourceContentProvider(filepath.parent)

    def clean(self):
        self.content_providers.clear()
        self._steam_id = -1

    @property
    def steam_id(self):
        if self._steam_id != -1:
            return self._steam_id
        used_appid = Counter([cm.steam_id for cm in self.content_providers.values() if cm.steam_id > 0])
        if len(used_appid) == 0:
            return 0
        return used_appid.most_common(1)[0][0]
