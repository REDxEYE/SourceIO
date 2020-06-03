from typing import List

from ....byte_io_mdl import ByteIO
from ...new_shared.base import Base
from .model import Model


class BodyPart(Base):
    def __init__(self):
        self.base = 0
        self.name = ""
        self.models = []  # type: List[Model]

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.name = reader.read_source1_string(entry) or "no-name"
        model_count = reader.read_uint32()
        self.base = reader.read_uint32()
        model_offset = reader.read_uint32()
        if model_count > 0:
            with reader.save_current_pos():
                reader.seek(entry + model_offset)
                for _ in range(model_count):
                    model = Model()
                    model.read(reader)
                    self.models.append(model)
