from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

from ....logger import SLoggingManager
from ...utils import Buffer, FileBuffer
from ...utils.gameinfo_parser import GameInfoParser
from ...utils.path_utilities import corrected_path
from ..app_id import SteamAppId
from .content_provider_base import ContentProviderBase

log_manager = SLoggingManager()
logger = log_manager.get_logger('Source1ContentProvider')


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
    def steam_id(self) -> SteamAppId:
        fs = self.gameinfo.file_system
        return SteamAppId(fs.steam_app_id)

    def get_search_paths(self):
        logger.info(f"Collecting search paths for {self.modname}. "
                    f"Mod path: {self.modname_dir}, project path: {self.project_dir}")

        def convert_path(path_to_convert):
            logger.info(f'Converting {path_to_convert!r} path.')
            if '|all_source_engine_paths|' in path_to_convert.lower():
                result = self.project_dir / path_to_convert.lower().replace('|all_source_engine_paths|', '')
            elif '|gameinfo_path|' in path_to_convert.lower():
                result = self.modname_dir / path_to_convert.lower().replace('|gameinfo_path|', '')
            else:
                result = self.project_dir / path_to_convert
            logger.info(f'Converted {path_to_convert!r} path to {result}.')
            return result

        all_search_paths = set()
        for paths in self.gameinfo.file_system.search_paths.game:
            if isinstance(paths, list):
                for path in paths:
                    all_search_paths.add(convert_path(path))
            else:
                all_search_paths.add(convert_path(paths))
        for file in self.modname_dir.glob('*_dir.vpk'):
            if file.suffix == '.vpk':
                all_search_paths.add(file)

        for file in self.project_dir.iterdir():
            if file.is_file():
                continue
            if (file / 'gameinfo.txt').exists():
                all_search_paths.add(file)
            for vpk_file in file.glob('*_dir.vpk'):
                all_search_paths.add(vpk_file)
        logger.info(f"Collected {len(all_search_paths)} paths: {all_search_paths}")
        return all_search_paths

    def find_file(self, filepath: Union[str, Path]) -> Optional[Buffer]:
        path = self.find_path(filepath)
        if path:
            return FileBuffer(path)

    def find_path(self, filepath: Union[str, Path]) -> Optional[Path]:
        filepath = Path(str(filepath).strip("\\/").replace('\\', '/'))
        new_filepath = corrected_path(self.modname_dir / filepath.as_posix())
        if new_filepath.exists():
            return new_filepath
        else:
            return None

    def glob(self, pattern: str) -> Iterator[Tuple[Path, Buffer]]:
        yield from self._glob_generic(pattern)
