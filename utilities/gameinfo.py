from pathlib import Path
from typing import List

from .keyvalues import KVParser
from ..source_shared.sub_manager import SubManager


class Gameinfo(SubManager):
    path_cache: List[Path] = []

    @classmethod
    def add_new_path(cls, path):
        cls.path_cache.append(path)

    def __init__(self, filepath: Path):
        super().__init__(filepath)
        with filepath.open('r') as f:
            kv = KVParser('GAMEINFO', f.read())
            root_key, self.data = kv.parse()
            assert root_key == 'gameinfo', 'Not a gameinfo file'
        self.modname_dir: Path = filepath.parent
        self.project_dir: Path = filepath.parent.parent
        self.modname: str = self.modname_dir.stem

    def get_search_paths(self):
        def convert_path(path_to_convert):
            if '|all_source_engine_paths|' in path_to_convert.lower():
                return self.project_dir / path_to_convert.lower().replace('|all_source_engine_paths|', '')
            elif '|gameinfo_path|' in path_to_convert.lower():
                return self.modname_dir / path_to_convert.lower().replace('|gameinfo_path|', '')
            return self.project_dir / path_to_convert

        all_search_paths = []
        for paths in self.data['filesystem']['searchpaths'].values():
            if isinstance(paths, list):
                for path in paths:
                    all_search_paths.append(convert_path(path))
            else:
                all_search_paths.append(convert_path(paths))
        for file in self.modname_dir.glob('*_dir.vpk'):
            if file.suffix == '.vpk':
                all_search_paths.append(file)

        for file in self.project_dir.iterdir():
            if file.is_file():
                continue
            if (file/'gameinfo.txt').exists():
                all_search_paths.append(file)
            for vpk_file in file.glob('*_dir.vpk'):
                all_search_paths.append(vpk_file)

        return all_search_paths

    def find_file(self, filepath: str, additional_dir=None,
                  extension=None):
        filepath = Path(str(filepath).strip("\\/"))

        new_filepath = filepath
        if additional_dir:
            new_filepath = Path(additional_dir, new_filepath)
        if extension:
            new_filepath = new_filepath.with_suffix(extension)
        new_filepath = self.modname_dir / new_filepath
        if new_filepath.exists():
            return new_filepath.open('rb')
        else:
            return None
