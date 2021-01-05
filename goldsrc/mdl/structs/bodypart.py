from typing import List

from .model import StudioModel
from ....source_shared.base import Base
from ....utilities.byte_io_mdl import ByteIO


class StudioBodypart(Base):
    def __init__(self):
        self.name = ''
        self.model_count = 0
        self.base = 0
        self.model_offset = 0
        self.models: List[StudioModel] = []

    def read(self, reader: ByteIO):
        self.name = reader.read_ascii_string(64)
        (self.model_count, self.base, self.model_offset) = reader.read_fmt('3i')
        with reader.save_current_pos():
            reader.seek(self.model_offset)
            for _ in range(self.model_count):
                model = StudioModel()
                model.read(reader)
                self.models.append(model)
