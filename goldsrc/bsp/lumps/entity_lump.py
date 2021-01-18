from typing import List, Dict

from ..lump import Lump, LumpType, LumpInfo


class EntityLump(Lump):
    LUMP_TYPE = LumpType.LUMP_ENTITIES

    def __init__(self, info: LumpInfo):
        super().__init__(info)
        self.values: List[Dict[str, str]] = []

    def parse(self):
        entities = self.buffer.read(self.info.length)
        entities = entities[:entities.index(b'\x00')].decode()
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
