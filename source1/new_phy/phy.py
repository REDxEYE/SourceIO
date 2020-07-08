import struct
from typing import List

from .structs.header import Header
from .structs.surface import CompactSurface

from ..new_shared.base import Base
from ...byte_io_mdl import ByteIO


class Phy(Base):
    def __init__(self, filepath):
        self.reader = ByteIO(path=filepath)
        self.header = Header()
        self.solids = []  # type:List[CompactSurface]


    def read(self):
        self.header.read(self.reader)
        self.reader.seek(self.header.header_size)
        for _ in range(self.header.solid_count):
            solid = CompactSurface()
            solid.read(self.reader)
            self.solids.append(solid)

