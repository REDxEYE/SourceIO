from SourceIO.library.goldsrc.bsp.bsp_file import BspFile
from SourceIO.library.goldsrc.bsp.lump import Lump, LumpInfo, LumpType
from SourceIO.library.utils import Buffer


class EntityLump(Lump):
    LUMP_TYPE = LumpType.LUMP_ENTITIES

    def __init__(self, info: LumpInfo):
        super().__init__(info)
        self.values: list[dict[str, str]] = []

    def parse(self, buffer: Buffer, bsp: BspFile):
        entities = buffer.read_ascii_string(self.info.length)
        entity = {}
        for line in entities.splitlines():
            if line == '{' or len(line) == 0:
                continue
            elif line == '}':
                self.values.append(entity)
                entity = {}
            else:
                entity_key_start = line.index('"') + 1
                entity_key_end = line.index('"', entity_key_start)
                entity_value_start = line.index('"', entity_key_end + 1) + 1
                entity_value_end = line.index('"', entity_value_start)
                entity[line[entity_key_start:entity_key_end]] = line[entity_value_start:entity_value_end]
