from io import StringIO
from .. import Lump, LumpTypes
from ....utilities.keyvalues import KVParser


class EntityLump(Lump):
    lump_id = LumpTypes.LUMP_ENTITIES

    def __init__(self, bsp):
        super().__init__(bsp)
        self.entities = []

    def parse(self):
        parser = KVParser('EntityLump', self.reader.read(-1).decode())
        entity = parser.parse_value()
        while entity is not None:
            self.entities.append(entity)
            entity = parser.parse_value()
        return self
