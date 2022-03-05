from pathlib import Path
from typing import List, Union

from ...utils.kv_parser import ValveKeyValueParser
from .content_provider_base import ContentProviderBase


class Gameinfo2ContentProvider(ContentProviderBase):
    path_cache: List[Path] = []

    @classmethod
    def add_new_path(cls, path):
        cls.path_cache.append(path)

    def __init__(self, filepath: Path):
        super().__init__(filepath)
        with filepath.open('r') as f:
            parser = ValveKeyValueParser(buffer_and_name=(f.read(), 'GAMEINFO'), self_recover=True)
            parser.parse()
            root_key, self.data = parser.tree.top()
            self.data = self.data.to_dict()
            assert root_key == 'gameinfo', 'Not a gameinfo file'
        self.modname_dir: Path = filepath.parent
        self.project_dir: Path = filepath.parent.parent
        self.modname: str = self.modname_dir.stem

    @property
    def steam_id(self):
        fs = self.data.get('filesystem', None)
        if not fs:
            return 0
        return int(fs.get('steamappid', 0))

    def get_paths(self):
        def convert_path(path_to_convert):
            if '|all_source_engine_paths|' in path_to_convert.lower():
                return path_to_convert.lower().replace('|all_source_engine_paths|', '')
            elif '|gameinfo_path|' in path_to_convert.lower():
                return path_to_convert.lower().replace('|gameinfo_path|', '')
            return path_to_convert

        all_search_paths = []
        fs = self.data.get('filesystem', None)
        if fs and fs.get('searchpaths', None):
            for paths in fs['searchpaths'].values():
                if isinstance(paths, list):
                    for path in paths:
                        all_search_paths.append(convert_path(path))
                else:
                    all_search_paths.append(convert_path(paths))

        return set(all_search_paths)

    def get_search_paths(self):
        def convert_path(path_to_convert):
            if '|all_source_engine_paths|' in path_to_convert.lower():
                return self.project_dir / path_to_convert.lower().replace('|all_source_engine_paths|', '')
            elif '|gameinfo_path|' in path_to_convert.lower():
                return self.modname_dir / path_to_convert.lower().replace('|gameinfo_path|', '')
            return self.project_dir / path_to_convert

        all_search_paths = []
        fs = self.data.get('filesystem', None)
        if fs and fs.get('searchpaths', None):
            for paths in fs['searchpaths'].values():
                if isinstance(paths, list):
                    for path in paths:
                        all_search_paths.append(convert_path(path))
                else:
                    all_search_paths.append(convert_path(paths))

        for file in self.modname_dir.glob('*_dir.vpk'):
            all_search_paths.append(file)
        for file in self.project_dir.iterdir():
            if file.is_file():
                continue
            if (file / 'gameinfo.gi').exists():
                all_search_paths.append(file)
            for vpk_file in file.glob('*_dir.vpk'):
                all_search_paths.append(vpk_file)
        return all_search_paths

    def glob(self, pattern: str):
        yield from self._glob_generic(pattern)

    def find_file(self, filepath: Union[str, Path]):
        filepath = Path(str(filepath).strip("\\/").replace('\\', '/'))
        new_filepath = self.modname_dir / filepath
        if new_filepath.exists():
            return new_filepath.open('rb')
        else:
            return None

    def find_path(self, filepath: Union[str, Path]):
        filepath = Path(str(filepath).strip("\\/").replace('\\', '/'))
        new_filepath = self.modname_dir / filepath.as_posix()
        if new_filepath.exists():
            return new_filepath
        else:
            return None
