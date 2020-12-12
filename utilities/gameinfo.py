from itertools import chain
from pathlib import Path
from typing import List

from .keyvalues import KVParser
from collections import OrderedDict

from ..source_shared.vpk.vpk_file import VPKFile


class Gameinfo:
    path_cache: List[Path] = []
    vpk_cache: List[VPKFile] = []

    @classmethod
    def add_new_path(cls, path):
        cls.path_cache.append(path)

    @classmethod
    def add_vpk_cache(cls, vpk_file):
        cls.vpk_cache.append(vpk_file)

    def __init__(self, path):
        path = Path(path)
        with open(path) as f:
            kv = KVParser('GAMEINFO', f)
            root_key, self.data = kv.parse()
            assert root_key == 'gameinfo', 'Not a gameinfo file'
        self.modname_dir: Path = path.parent
        self.project_dir: Path = path.parent.parent
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
        return all_search_paths

    def get_search_paths_recursive(self):
        paths = self.get_search_paths()
        for path in paths:
            if path not in self.path_cache:
                self.add_new_path(path)
            else:
                continue
            print(f"visiting {path}")
            if path.suffix == '.vpk' and path.with_name(f'{path.stem}_dir.vpk').is_file():
                vpk = VPKFile(path.with_name(f'{path.stem}_dir.vpk'))
                vpk.read()
                self.add_vpk_cache(vpk)
                continue
            elif (path / 'gameinfo.txt').exists():
                next_gi = Gameinfo(path / 'gameinfo.txt')
                next_gi.get_search_paths_recursive()

        return self.path_cache

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

    def find_texture(self, filepath):
        return self.find_file(filepath, 'materials', extension='.vtf')

    def find_material(self, filepath):
        return self.find_file(filepath, 'materials', extension='.vmt')
