from io import BufferedIOBase, BytesIO, StringIO, TextIOBase
from pathlib import Path
from typing import IO, List, Union

from ...logger import SLoggingManager
from ..utils.s1_keyvalues import KVParser

log_manager = SLoggingManager()
logger = log_manager.get_logger('GameInfoParser')


class GameInfoParser:
    class HiddenMaps:
        def __init__(self, raw_data):
            self._raw_data = raw_data

        @property
        def test_speakers(self):
            return bool(int(self._raw_data.get('test_speakers', '0')))

        @property
        def test_hardware(self):
            return bool(int(self._raw_data.get('test_hardware', '0')))

    class FileSystem:
        class SearchPaths:
            def __init__(self, raw_data):
                self._raw_data = raw_data

            @property
            def game(self):
                value = self._raw_data.get('game', [])
                if isinstance(value, str):
                    return [value]
                return value

            @property
            def all_paths(self) -> List[str]:
                paths = []
                for value in self._raw_data.values():
                    if isinstance(value, str):
                        value = [value]
                    for path in value:
                        if '|all_source_engine_paths|' in path:
                            path = path.replace('|all_source_engine_paths|', "")
                        paths.append(path)
                return list(set(paths))

        def __init__(self, raw_data):
            self._raw_data = raw_data

        @property
        def steam_app_id(self):
            return int(self._raw_data.get('steamappid', '0'))

        @property
        def tools_app_id(self):
            return int(self._raw_data.get('toolsappid', '0'))

        @property
        def search_paths(self):
            return self.SearchPaths(self._raw_data.get('searchpaths', {}))

    def __init__(self, file_or_string: Union[IO[str], IO[bytes], str], path: Path):
        self.path = path
        self.root = self.path.parent
        if isinstance(file_or_string, str):
            self._buffer = file_or_string
        elif isinstance(file_or_string, (TextIOBase, StringIO)):
            self._buffer = file_or_string.read()
        elif isinstance(file_or_string, (BufferedIOBase, BytesIO)):
            self._buffer = file_or_string.read().decode('latin')
        else:
            raise ValueError(f'Unknown input value type {type(file_or_string)}')

        self._parser = KVParser(str(path), self._buffer)
        self.header, self._raw_data = self._parser.parse()

    @property
    def all_paths(self) -> List[Path]:
        paths = []
        for path in self.file_system.search_paths.all_paths:
            try:
                if '|gameinfo_path|' in path:
                    path = f"{self.root.as_posix()}/{path.replace('|gameinfo_path|', '')}"
                    if path.endswith('*'):
                        path = path[:-1]
                        sub_mod = (self.root / path)
                        if not sub_mod.exists():
                            continue
                        for sub_path in sub_mod.iterdir():
                            paths.append(sub_path)
                    else:
                        paths.append(Path(path))
                elif path.endswith('*'):
                    path = path[:-1]
                    if (self.root.parent / path).exists():
                        for sub_path in (self.root.parent / path).iterdir():
                            paths.append(sub_path)
                else:
                    paths.append(Path(path))
            except Exception as e:
                logger.exception("failed to parse path", e)
        return paths

    @property
    def game(self):
        return self._raw_data.get('game')

    @property
    def title(self):
        return self._raw_data.get('title')

    @property
    def title2(self):
        return self._raw_data.get('title2')

    @property
    def type(self):
        return self._raw_data.get('type')

    @property
    def nomodels(self):
        return bool(int(self._raw_data.get('nomodels')))

    @property
    def nohimodel(self):
        return bool(int(self._raw_data.get('nohimodel')))

    @property
    def nocrosshair(self):
        return bool(int(self._raw_data.get('nocrosshair')))

    @property
    def hidden_maps(self):
        return self.HiddenMaps(self._raw_data.get('hidden_maps', {}))

    @property
    def nodegraph(self):
        return bool(int(self._raw_data.get('nodegraph')))

    @property
    def file_system(self):
        return self.FileSystem(self._raw_data.get('filesystem', {}))
