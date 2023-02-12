from dataclasses import dataclass
from typing import List, Type

from ....utils import Buffer
from .model import Model


@dataclass(slots=True)
class BodyPart:
    name: str
    models: List[Model]

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        start_offset = buffer.tell()
        name = buffer.read_source1_string(start_offset) or "no-name"
        model_count = buffer.read_uint32()
        base = buffer.read_uint32()
        model_offset = buffer.read_uint32()
        models = []
        if model_count > 0:
            with buffer.read_from_offset(start_offset + model_offset):
                for _ in range(model_count):
                    model = Model.from_buffer(buffer, version)
                    models.append(model)
        return cls(name, models)
