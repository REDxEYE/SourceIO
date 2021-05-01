from collections import deque
from io import BytesIO
from pathlib import Path


class ContentProviderBase:
    __cache = deque([], maxlen=16)

    def __init__(self, filepath: Path):
        self.filepath = filepath

    def find_file(self, filepath: str):
        raise NotImplementedError('Implement me!')

    def find_path(self, filepath: str):
        raise NotImplementedError('Implement me!')

    def cache_file(self, filename, file: BytesIO):
        self.__cache.append((filename, file))
        return file

    def get_from_cache(self, filename):
        for name, file in self.__cache:
            if name == filename:
                file.seek(0)
                return file

    def flush_cache(self):
        self.__cache.clear()

    @property
    def steam_id(self):
        return 0
