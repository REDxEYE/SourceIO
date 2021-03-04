from pathlib import Path


class ContentProviderBase:

    def __init__(self, filepath: Path):
        self.filepath = filepath

    def find_file(self, filepath: str):
        raise NotImplementedError('Implement me!')

    def find_path(self, filepath: str):
        raise NotImplementedError('Implement me!')

    @property
    def steam_id(self):
        return 0
