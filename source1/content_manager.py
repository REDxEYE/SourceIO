from pathlib import Path

from typing import List, Union, Dict

from ..source_shared.vpk import VPKFile
from ..utilities.gameinfo import Gameinfo
from ..utilities.path_utilities import NonSourceInstall, get_mod_path
from ..utilities.singleton import Singleton, SingletonMeta


class ContentManager(metaclass=SingletonMeta):
    def __init__(self):
        self.sub_managers: Dict[str, Union[Gameinfo, NonSourceInstall, VPKFile]] = {}

    def scan_for_content(self, source_game_path: Union[str, Path]):
        source_game_path = Path(source_game_path)
        if source_game_path.suffix == '.vpk':
            if source_game_path.stem in self.sub_managers:
                return
            sub_manager = VPKFile(source_game_path.parent / (source_game_path.stem + "_dir.vpk"))
            sub_manager.read()
            self.sub_managers[source_game_path.stem] = sub_manager
            print(f'Registered sub manager for {source_game_path.stem}')
            return

        is_source, root_path = self.is_source_mod(source_game_path)
        if root_path.stem in self.sub_managers:
            return
        if is_source:
            sub_manager = Gameinfo(root_path / 'gameinfo.txt')
            self.sub_managers[root_path.stem] = sub_manager
            print(f'Registered sub manager for {root_path.stem}')
            for mod in sub_manager.get_search_paths():
                self.scan_for_content(mod)
        else:
            if root_path.is_dir():
                sub_manager = NonSourceInstall(root_path)
                self.sub_managers[root_path.stem] = sub_manager
                print(f'Registered sub manager for {source_game_path.stem}')

    @staticmethod
    def is_source_mod(path: Path, second=False):
        if (path / 'gameinfo.txt').exists():
            return True, path
        elif not second:
            return ContentManager.is_source_mod(get_mod_path(path), True)
        return False, path

    def find_file(self, filepath: str, additional_dir=None, extension=None):
        new_filepath = filepath
        if additional_dir:
            new_filepath = Path(additional_dir, new_filepath)
        if extension:
            new_filepath = new_filepath.with_suffix(extension)
        print(f'Requesting {new_filepath} file')
        for mod, submanager in self.sub_managers.items():
            file = submanager.find_file(filepath, additional_dir, extension)
            if file is not None:
                print(f'Found in {mod}!')
                return file
        return None

    def find_texture(self, filepath):
        return self.find_file(filepath, 'materials', extension='.vtf')

    def find_material(self, filepath):
        return self.find_file(filepath, 'materials', extension='.vmt')
