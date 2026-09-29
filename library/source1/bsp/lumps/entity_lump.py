import charset_normalizer

from SourceIO.library.source1.bsp import Lump, ValveLumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import VBSPFile
from SourceIO.library.utils import Buffer
from SourceIO.library.utils import kv1
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()


@lump_tag(0, 'LUMP_ENTITIES')
class EntityLump(Lump):
    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.entities = []
        self._logger = log_manager.get_logger("Entity Lump")

    def parse(self, buffer: Buffer, bsp: VBSPFile):
        buffer = buffer.read(-1).strip(b"\x00")
        # chaset = charset_normalizer.from_bytes(buffer)
        # self._logger.info(f"Detected {chaset.best().encoding!r} encoding in entity lump")
        # buffer = buffer.decode(chaset.best().encoding, "replace")
        buffer = buffer.decode("utf8", "replace")
        buffer = buffer.translate(str.maketrans({chr(i): " " for i in range(0xA)}))
        buffer = buffer.translate(str.maketrans({chr(65533): " "}))
        for entity in kv1.loads_blocks(buffer, 'EntityLump'):
            self.entities.append(entity.to_dict())
        return self


@lump_tag(24, 'LUMP_ENTITYPARTITIONS', bsp_version=29)
class EntityPartitionsLump(Lump):
    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.entities = []

    def parse(self, buffer: Buffer, bsp: VBSPFile):
        data = buffer.read_ascii_string(-1)
        entity_files = data.split(' ')[1:]
        for ent_file in entity_files:
            ent_path: TinyPath = bsp.filepath.parent / f'{bsp.filepath.stem}_{ent_file}.ent'
            if ent_path.exists():
                with ent_path.open('r') as f:
                    magic = f.read(11).strip()
                    assert magic == 'ENTITIES01', 'Invalid ent file'
                    for entity in kv1.loads_blocks(f.read(-1), str(ent_path)):
                        self.entities.append(entity.to_dict())

        return self
