from typing import List

from . import Base
from . import ByteIO
from .model import ModelV36, ModelV49, ModelV44


class BodyPartV36(Base):
    model_class = ModelV36

    def __init__(self):
        self.base = 0
        self.name = ""
        self.models = []  # type: List[ModelV36]

    def __repr__(self):
        return f'<BodyGroup "{self.name}" models:{len(self.models)}>'

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
                    model = self.model_class()
                    model.read(reader)
                    self.models.append(model)


class BodyPartV44(BodyPartV36):
    model_class = ModelV44

    def __init__(self):
        super().__init__()
        self.models = []  # type: List[ModelV49]


class BodyPartV49(BodyPartV44):
    model_class = ModelV49

    def __init__(self):
        super().__init__()
        self.models = []  # type: List[ModelV49]
