from pathlib import Path
from typing import Union, Dict

from ..bpy_utilities.logging import BPYLoggingManager
from ..source_shared.non_source_sub_manager import NonSourceContentProvider
from ..source_shared.content_provider_base import ContentProviderBase
from ..source_shared.vpk_sub_manager import VPKContentProvider
from ..source1.source1_content_provider import GameinfoContentProvider as Source1GameinfoContentProvider
from ..source2.source2_content_provider import GameinfoContentProvider as Source2GameinfoContentProvider
from ..utilities.path_utilities import get_mod_path
from ..utilities.singleton import SingletonMeta

log_manager = BPYLoggingManager()
logger = log_manager.get_logger('content_manager')


class ContentManager(metaclass=SingletonMeta):
    def __init__(self):
        self.content_providers: Dict[str, ContentProviderBase] = {}
        self._titanfall_mode = False

    def scan_for_content(self, source_game_path: Union[str, Path]):
        if "*LANGUAGE*" in str(source_game_path):
            return
        source_game_path = Path(source_game_path)
        if source_game_path.suffix == '.vpk':
            if self._titanfall_mode:
                if 'english' not in str(source_game_path):
                    return
            if f'{source_game_path.parent.stem}_{source_game_path.stem}' in self.content_providers:
                return
            vpk_path = source_game_path
            if vpk_path.exists():
                sub_manager = VPKContentProvider(vpk_path)
                self.content_providers[f'{source_game_path.parent.stem}_{source_game_path.stem}'] = sub_manager
                logger.info(f'Registered sub manager for {source_game_path.parent.stem}_{source_game_path.stem}')
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
                logger.info(f'Registered sub manager for {root_path.stem}')
                for mod in sub_manager.get_search_paths():
                    self.scan_for_content(mod)

            gameinfos = root_path.glob('*gameinfo*.gi')
            for gameinfo in gameinfos:
                sub_manager = Source2GameinfoContentProvider(gameinfo)
                self.content_providers[root_path.stem] = sub_manager
                logger.info(f'Registered sub manager for {root_path.stem}')
                for mod in sub_manager.get_search_paths():
                    self.scan_for_content(mod)
        elif 'workshop' in root_path.name:
            sub_manager = NonSourceContentProvider(root_path)
            self.content_providers[root_path.stem] = sub_manager
            logger.info(f'Registered sub manager for {root_path.stem}')
            for mod in root_path.parent.iterdir():
                if mod.is_dir():
                    self.scan_for_content(mod)
        elif 'download' in root_path.name:
            sub_manager = NonSourceContentProvider(root_path)
            self.content_providers[root_path.stem] = sub_manager
            logger.info(f'Registered sub manager for {root_path.stem}')
            self.scan_for_content(root_path.parent)
        else:
            if root_path.is_dir():
                sub_manager = NonSourceContentProvider(root_path)
                self.content_providers[root_path.stem] = sub_manager
                logger.info(f'Registered sub manager for {source_game_path.stem}')
            else:
                root_path = root_path.parent
                sub_manager = NonSourceContentProvider(root_path)
                self.content_providers[root_path.stem] = sub_manager
                logger.info(f'Registered sub manager for {source_game_path.stem}')

    def deserialize(self, data: Dict[str, str]):
        for name, path in data.items():
            if path.endswith('.vpk'):
                sub_manager = VPKContentProvider(Path(path))
                self.content_providers[name] = sub_manager
            elif path.endswith('.txt'):
                sub_manager = Source1GameinfoContentProvider(Path(path))
                if sub_manager.gameinfo.game == 'Titanfall':
                    self._titanfall_mode = True
                self.content_providers[name] = sub_manager
            elif path.endswith('.gi'):
                sub_manager = Source2GameinfoContentProvider(Path(path))
                self.content_providers[name] = sub_manager
            elif path.endswith('.bsp'):
                from ..source1.bsp.bsp_file import open_bsp
                bsp = open_bsp(path)
                bsp.parse()
                pak_lump = bsp.get_lump('LUMP_PAK')
                if pak_lump:
                    self.content_providers[name] = pak_lump
            else:
                sub_manager = NonSourceContentProvider(Path(path))
                self.content_providers[name] = sub_manager

    @staticmethod
    def is_source_mod(path: Path, second=False):
        if path.name == 'gameinfo.txt':
            path = path.parent
        if path.parts[-1] == '*':
            path = path.parent
        gameinfos = list(path.glob('*gameinfo*.*'))
        if gameinfos:
            return True, path
        elif not second:
            return ContentManager.is_source_mod(get_mod_path(path), True)
        return False, path

    def find_file(self, filepath: str, additional_dir=None, extension=None, *, silent=False):

        new_filepath = Path(str(filepath).strip('/\\').rstrip('/\\'))
        if additional_dir:
            new_filepath = Path(additional_dir, new_filepath)
        if extension:
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

    def find_path(self, filepath: str, additional_dir=None, extension=None, *, silent=False):
        new_filepath = Path(str(filepath).strip('/\\').rstrip('/\\'))
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
                return file
        return None

    def find_texture(self, filepath, *, silent=False):
        return self.find_file(filepath, 'materials', extension='.vtf', silent=silent)

    def find_material(self, filepath, *, silent=False):
        return self.find_file(filepath, 'materials', extension='.vmt', silent=silent)

    def serialize(self):
        serialized = {}
        for name, sub_manager in self.content_providers.items():
            name = name.replace('\'', '').replace('\"', '').replace(' ', '_')
            serialized[name[:63]] = str(sub_manager.filepath)

        return serialized

    def get_content_provider_from_path(self, filepath):
        filepath = Path(filepath)
        for name, content_provider in self.content_providers.items():
            cp_root = content_provider.filepath.parent
            is_sm, fp_root = self.is_source_mod(filepath)
            if fp_root == cp_root:
                return content_provider
        return NonSourceContentProvider(filepath.parent)
