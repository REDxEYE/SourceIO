from pathlib import Path
from typing import List

from .keyvalues import KVParser
from collections import OrderedDict

from ..source_shared.vpk.vpk_file import VPKFile


class DictX(OrderedDict):
    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict):
                return DictX(value)
            return value
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DictX ' + dict.__repr__(self) + '>'


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
            assert root_key == 'GameInfo', 'Not a gameinfo file'
            self.data = DictX(self.data)
        self.modname_dir: Path = path.parent
        self.project_dir: Path = path.parent.parent
        self.modname: str = self.modname_dir.stem

    def get_search_paths(self):
        gi = '|gameinfo_path|'
        sp = '|all_source_engine_paths|'
        all_search_paths = []

        def process_path(name, s_path):
            if gi in s_path:
                s_path = s_path.replace(gi, '')
            elif sp in s_path:
                s_path = s_path.replace(sp, '')
            return self.project_dir / s_path

        for mod_name, search_path in self.data.FileSystem.SearchPaths.items():

            if isinstance(search_path, list):
                for p in search_path:
                    all_search_paths.append(process_path(mod_name, p))
            else:
                all_search_paths.append(process_path(mod_name, search_path))

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
                  extention=None, use_recursive=False):
        filepath = Path(str(filepath).strip("\\/"))

        if use_recursive:
            if self.path_cache:
                paths = self.path_cache
            else:
                print("Collecting all possible search paths!")
                self.path_cache.clear()
                self.vpk_cache.clear()
                paths = self.get_search_paths_recursive()
        else:
            paths = self.get_search_paths()
        new_filepath = filepath
        if additional_dir:
            new_filepath = Path(additional_dir, new_filepath)
        if extention:
            new_filepath = new_filepath.with_suffix(extention)

        for vpk in self.vpk_cache:
            if new_filepath in vpk.path_cache:
                entry = vpk.find_file(full_path=new_filepath)
                if entry:
                    return vpk.read_file(entry)

        for mod_path in paths:
            if mod_path.stem == '*':
                mod_path = mod_path.parent
            search_path = mod_path / new_filepath
            if search_path.exists():
                return search_path
        else:
            return None

    def find_texture(self, filepath, use_recursive=False):
        return self.find_file(filepath, 'materials',
                              extention='.vtf', use_recursive=use_recursive)

    def find_material(self, filepath, use_recursive=False):
        return self.find_file(filepath, 'materials',
                              extention='.vmt', use_recursive=use_recursive)
