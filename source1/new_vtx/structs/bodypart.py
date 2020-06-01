from typing import List

from ...new_shared.base import Base
from ....byte_io_mdl import ByteIO
from .model import Model


class BodyPart(Base):
    def __init__(self):
        self.models = []  # type: List[Model]

    def read(self, reader: ByteIO):
        entry = reader.tell()
        model_count, model_offset = reader.read_fmt('II')

        with reader.save_current_pos():
            reader.seek(entry + model_offset)
            for _ in range(model_count):
                model = Model()
                model.read(reader)
                self.models.append(model)
