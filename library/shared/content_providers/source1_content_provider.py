from pathlib import Path
from typing import List, Union

from ...utils.gameinfo_parser import GameInfoParser
from .content_provider_base import ContentProviderBase


class GameinfoContentProvider(ContentProviderBase):
    path_cache: List[Path] = []

    @classmethod
    def add_new_path(cls, path):
        cls.path_cache.append(path)

    def __init__(self, filepath: Path):
        super().__init__(filepath)
        with filepath.open('r') as f:
            self.gameinfo = GameInfoParser(f, filepath)
            assert self.gameinfo.header == 'gameinfo', 'Not a gameinfo file'
        if filepath.with_name(filepath.stem + '_srgb.txt').exists():
            with filepath.with_name(filepath.stem + '_srgb.txt').open('r') as f:
                gameinfo = GameInfoParser(f, filepath.with_name(filepath.stem + '_srgb.txt'))
                assert self.gameinfo.header == 'gameinfo', 'Not a gameinfo file'
                og_paths = self.gameinfo.file_system.search_paths
                srgb_paths = gameinfo.file_system.search_paths
                self.gameinfo.file_system.search_paths._raw_data['game'] = og_paths.game + srgb_paths.game

        self.modname_dir: Path = filepath.parent
        self.project_dir: Path = filepath.parent.parent
        self.modname: str = self.modname_dir.stem

    @property
    def steam_id(self):
        fs = self.gameinfo.file_system
        return fs.steam_app_id

    def get_search_paths(self):
        def convert_path(path_to_convert):
            if '|all_source_engine_paths|' in path_to_convert.lower():
                return self.project_dir / path_to_convert.lower().replace('|all_source_engine_paths|', '')
            elif '|gameinfo_path|' in path_to_convert.lower():
                return self.modname_dir / path_to_convert.lower().replace('|gameinfo_path|', '')
            return self.project_dir / path_to_convert

        all_search_paths = []
        for paths in self.gameinfo.file_system.search_paths.game:
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
            if (file / 'gameinfo.txt').exists():
                all_search_paths.append(file)
            for vpk_file in file.glob('*_dir.vpk'):
                all_search_paths.append(vpk_file)

        return all_search_paths

    def find_file(self, filepath: Union[str, Path]):
        path = self.find_path(filepath)
        if path:
            return path.open('rb')

    def find_path(self, filepath: Union[str, Path]):
        filepath = Path(str(filepath).strip("\\/").replace('\\', '/'))
        new_filepath = self.modname_dir / filepath.as_posix()
        if new_filepath.exists():
            return new_filepath
        else:
            return None

    def glob(self, pattern: str):
        yield from self._glob_generic(pattern)
