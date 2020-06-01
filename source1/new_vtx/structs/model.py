from typing import List

from ...new_shared.base import Base
from ....byte_io_mdl import ByteIO

from .lod import ModelLod

class Model(Base):
    def __init__(self):
        self.model_lods = []  # type: List[ModelLod]

    def read(self, reader: ByteIO):
        entry = reader.tell()
        lod_count, lod_offset = reader.read_fmt('ii')
        with reader.save_current_pos():
            if lod_count > 0 and lod_offset != 0:
                reader.seek(entry + lod_offset)
                for lod_id in range(lod_count):
                    model_lod = ModelLod(lod_id)
                    model_lod.read(reader)
                    self.model_lods.append(model_lod)
