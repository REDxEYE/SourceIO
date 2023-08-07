from pathlib import Path
import charset_normalizer

from .....logger import SLoggingManager
from ....utils import Buffer
from ....utils.kv_parser import ValveKeyValueParser
from ....utils.s1_keyvalues import KVParser
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile

log_manager = SLoggingManager()


@lump_tag(0, 'LUMP_ENTITIES')
class EntityLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.entities = []
        self._logger = log_manager.get_logger("Entity Lump")

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        buffer = buffer.read(-1).strip(b"\x00")
        chaset = charset_normalizer.from_bytes(buffer)
        self._logger.info(f"Detected {chaset.best().encoding!r} encoding in entity lump")
        buffer = buffer.decode(chaset.best().encoding, "ignore")
        parser = ValveKeyValueParser(buffer_and_name=(buffer, 'EntityLump'), self_recover=True, array_of_blocks=True)
        parser.parse()
        for ent in parser.tree:
            self.entities.append(ent.to_dict())
        return self


@lump_tag(24, 'LUMP_ENTITYPARTITIONS', bsp_version=29)
class EntityPartitionsLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.entities = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        data = buffer.read_ascii_string(-1)
        entity_files = data.split(' ')[1:]
        for ent_file in entity_files:
            ent_path: Path = bsp.filepath.parent / f'{bsp.filepath.stem}_{ent_file}.ent'
            if ent_path.exists():
                with ent_path.open('r') as f:
                    magic = f.read(11).strip()
                    assert magic == 'ENTITIES01', 'Invalid ent file'
                    parser = KVParser('EntityLump', f.read(-1))
                    entity = parser.parse_value()
                    while entity is not None:
                        self.entities.append(entity)
                        entity = parser.parse_value()

        return self
