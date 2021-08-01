from typing import List

from .structs.colliders import ColliderHeader, ColliderType, CompactSurfaceHeader, MoppHeader
from .structs.header import Header

from ...source_shared.base import Base
from ...utilities.byte_io_mdl import ByteIO


class Phy(Base):
    def __init__(self, filepath):
        self.reader = ByteIO(filepath)
        self.header = Header()
        self.solids = []  # type:List[ColliderHeader]
        self.kv = ''

    def read(self):
        reader = self.reader
        self.header.read(reader)
        reader.seek(self.header.size)
        for _ in range(self.header.solid_count):
            size = reader.read_uint32()
            solid = ColliderHeader.peek(reader)
            if solid.id == 'VPHY':
                if solid.model_type == ColliderType.COLLIDE_POLY:
                    solid = CompactSurfaceHeader()
                elif solid.model_type == ColliderType.COLLIDE_MOPP:
                    solid = MoppHeader()
            solid.read(reader)
            self.solids.append(solid)
