from dataclasses import dataclass
from typing import List

from .....utils import Buffer
from .lod import ModelLod


@dataclass(slots=True)
class Model:
    model_lods: List[ModelLod]

    @classmethod
    def from_buffer(cls, buffer: Buffer, extra8: bool = False):
        entry = buffer.tell()
        lod_count, lod_offset = buffer.read_fmt('ii')
        model_lods = []
        if lod_count > 0 and lod_offset != 0:
            with buffer.read_from_offset(entry + lod_offset):
                for lod_id in range(lod_count):
                    model_lod = ModelLod.from_buffer(buffer, lod_id, extra8)
                    model_lods.append(model_lod)
        return cls(model_lods)
