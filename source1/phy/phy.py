from typing import List

from .structs.header import Header
from .structs.surface import CompactSurface

from ...source_shared.base import Base
from ...utilities.byte_io_mdl  import ByteIO


class Phy(Base):
    def __init__(self, filepath):
        self.reader = ByteIO(filepath)
        self.header = Header()
        self.solids = []  # type:List[CompactSurface]
        self.kv = ''

    def read(self):
        self.header.read(self.reader)
        self.reader.seek(self.header.header_size)
        for _ in range(self.header.solid_count):
            solid = CompactSurface()
            solid.read(self.reader)
            self.solids.append(solid)
        # kv = self.reader.read_ascii_string().replace('}', '\n}\n').replace('{', '\n{\n')
        # kv_wrapped = 'phy_data \n{\n' + kv + '\n}'
        # self.kv = KeyValueFile(None, string_buffer=kv_wrapped.split('\n'))
