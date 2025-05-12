from dataclasses import dataclass

from SourceIO.library.utils import Buffer
from .model import Model, ModelV36Plus, ModelV2531


@dataclass(slots=True)
class BodyPart:
    name: str
    models: list[Model]

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        start_offset = buffer.tell()
        name = buffer.read_source1_string(start_offset) or "no-name"
        model_count = buffer.read_uint32()
        base = buffer.read_uint32()
        model_offset = buffer.read_uint32()
        models = []
        model_class = ModelV36Plus if version != 2531 else ModelV2531
        if model_count > 0:
            with buffer.read_from_offset(start_offset + model_offset):
                for _ in range(model_count):
                    model = model_class.from_buffer(buffer, version)
                    models.append(model)
        return cls(name, models)
