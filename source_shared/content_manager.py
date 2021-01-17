from pathlib import Path
from typing import Union, Dict

from ..bpy_utilities.logging import BPYLoggingManager
from ..source_shared.non_source_sub_manager import NonSourceSubManager
from ..source_shared.sub_manager import SubManager
from ..source_shared.vpk_sub_manager import VPKSubManager
from ..utilities.gameinfo import Gameinfo
from ..utilities.path_utilities import get_mod_path
from ..utilities.singleton import SingletonMeta

log_manager = BPYLoggingManager()
logger = log_manager.get_logger('content_manager')


class ContentManager(metaclass=SingletonMeta):
    def __init__(self):
        self.sub_managers: Dict[str, SubManager] = {}

    def scan_for_content(self, source_game_path: Union[str, Path]):
        source_game_path = Path(source_game_path)
        if source_game_path.suffix == '.vpk':
            if f'{source_game_path.parent.stem}_{source_game_path.stem}' in self.sub_managers:
                return
            if not source_game_path.stem.endswith('_dir'):
                vpk_path = source_game_path.parent / (source_game_path.stem + "_dir.vpk")
            else:
                vpk_path = source_game_path
            if vpk_path.exists():
                sub_manager = VPKSubManager(vpk_path)
                self.sub_managers[f'{source_game_path.parent.stem}_{source_game_path.stem}'] = sub_manager
                logger.info(f'Registered sub manager for {source_game_path.parent.stem}_{source_game_path.stem}')
                return

        is_source, root_path = self.is_source_mod(source_game_path)
        if root_path.stem in self.sub_managers:
            return
        if is_source:
            gameinfos = root_path.glob('*gameinfo*.txt')
            for gameinfo in gameinfos:
                sub_manager = Gameinfo(gameinfo)
                self.sub_managers[root_path.stem] = sub_manager
                logger.info(f'Registered sub manager for {root_path.stem}')
                for mod in sub_manager.get_search_paths():
                    self.scan_for_content(mod)
        elif 'workshop' in root_path.name:
            sub_manager = NonSourceSubManager(root_path)
            self.sub_managers[root_path.stem] = sub_manager
            logger.info(f'Registered sub manager for {root_path.stem}')
            for mod in root_path.parent.iterdir():
                if mod.is_dir():
                    self.scan_for_content(mod)
        elif 'download' in root_path.name:
            sub_manager = NonSourceSubManager(root_path)
            self.sub_managers[root_path.stem] = sub_manager
            logger.info(f'Registered sub manager for {root_path.stem}')
            self.scan_for_content(root_path.parent)
        else:
            if root_path.is_dir():
                sub_manager = NonSourceSubManager(root_path)
                self.sub_managers[root_path.stem] = sub_manager
                logger.info(f'Registered sub manager for {source_game_path.stem}')

    def deserialize(self, data: Dict[str, str]):
        for name, path in data.items():
            if path.endswith('.vpk'):
                sub_manager = VPKSubManager(Path(path))
                self.sub_managers[name] = sub_manager
            elif path.endswith('.txt'):
                sub_manager = Gameinfo(Path(path))
                self.sub_managers[name] = sub_manager
            elif path.endswith('.bsp'):
                from ..source1.bsp.bsp_file import BSPFile, LumpTypes
                bsp = BSPFile(path)
                bsp.parse()
                pak_lump = bsp.get_lump(LumpTypes.LUMP_PAK)
                if pak_lump:
                    self.sub_managers[name] = pak_lump
            else:
                sub_manager = NonSourceSubManager(Path(path))
                self.sub_managers[name] = sub_manager

    @staticmethod
    def is_source_mod(path: Path, second=False):
        if path.name == 'gameinfo.txt':
            path = path.parent
        if path.parts[-1] == '*':
            path = path.parent
        gameinfos = list(path.glob('*gameinfo*.txt'))
        if gameinfos:
            return True, path
        elif not second:
            return ContentManager.is_source_mod(get_mod_path(path), True)
        return False, path

    def find_file(self, filepath: str, additional_dir=None, extension=None):

        new_filepath = Path(str(filepath).strip('/\\').rstrip('/\\'))
        if additional_dir:
            new_filepath = Path(additional_dir, new_filepath)
        if extension:
            new_filepath = new_filepath.with_suffix(extension)
        logger.info(f'Requesting {new_filepath} file')
        for mod, submanager in self.sub_managers.items():
            # print(f'Searching in {mod}')
            file = submanager.find_file(new_filepath)
            if file is not None:
                logger.debug(f'Found in {mod}!')
                return file
        return None

    def find_texture(self, filepath):
        return self.find_file(filepath, 'materials', extension='.vtf')

    def find_material(self, filepath):
        return self.find_file(filepath, 'materials', extension='.vmt')

    def serialize(self):
        serialized = {}
        for name, sub_manager in self.sub_managers.items():
            name = name.replace('\'', '').replace('\"', '').replace(' ', '_')
            serialized[name] = str(sub_manager.filepath)

        return serialized
