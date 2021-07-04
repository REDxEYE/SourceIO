from typing import List

from .model import StudioModel
from ....source_shared.base import Base
from ....utilities.byte_io_mdl import ByteIO


class StudioBodypart(Base):
    def __init__(self):
        self.model_count = 0

    def read(self, reader: ByteIO):
        self.model_count = reader.read_int32()
