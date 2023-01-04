from dataclasses import dataclass
from typing import List

from ....utils import Buffer
from .model import StudioModel


@dataclass(slots=True)
class StudioBodypart:
    name: str
    base: int
    models: List[StudioModel]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        name = buffer.read_ascii_string(64)
        (model_count, base, model_offset) = buffer.read_fmt('3i')
        models = []
        with buffer.read_from_offset(model_offset):
            for _ in range(model_count):
                model = StudioModel.from_buffer(buffer)
                models.append(model)
        return cls(name, base, models)
