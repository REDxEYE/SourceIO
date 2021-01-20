from io import StringIO
from .. import Lump, lump_tag
from ....utilities.keyvalues import KVParser


@lump_tag(0, 'LUMP_ENTITIES')
class EntityLump(Lump):
    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.entities = []

    def parse(self):
        parser = KVParser('EntityLump', self.reader.read(-1).decode())
        entity = parser.parse_value()
        while entity is not None:
            self.entities.append(entity)
            entity = parser.parse_value()
        return self
