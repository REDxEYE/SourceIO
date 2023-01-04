from typing import List

from .....library.utils import Buffer
from .model import StudioModel


class StudioBodypart:
    def __init__(self):
        self.name = ''
        self.model_count = 0
        self.base = 0
        self.model_offset = 0
        self.models: List[StudioModel] = []

    def read(self, reader: Buffer):
        self.name = reader.read_ascii_string(64)
        (self.model_count, self.base, self.model_offset) = reader.read_fmt('3i')
        with reader.save_current_offset():
            reader.seek(self.model_offset)
            for _ in range(self.model_count):
                model = StudioModel()
                model.read(reader)
                self.models.append(model)
