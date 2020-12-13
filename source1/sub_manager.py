from pathlib import Path


class SubManager:

    def __init__(self, filepath: Path):
        self.filepath = filepath

    def find_file(self, filepath: str):
        raise NotImplementedError('Implement me!')
