from typing import Optional

from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.utils import Buffer, FileBuffer, TinyPath
from .lump import Lump, LumpInfo, LumpType


class BspFile:
    def __init__(self, buffer: Buffer, content_manager: ContentManager):
        self.manager = content_manager
        self.buffer = buffer
        self.lumps = {}
        self.lumps_info: list[LumpInfo] = []
        self.version = 0

    @classmethod
    def from_filename(cls, filepath: TinyPath, content_manager: ContentManager):
        buffer = FileBuffer(filepath)
        self = cls(buffer, content_manager)
        self.version = buffer.read_uint32()
        self.lumps_info = []

        lump0 = LumpInfo.from_buffer(buffer, LumpType.LUMP_ENTITIES)
        lump1 = LumpInfo.from_buffer(buffer, LumpType.LUMP_PLANES)

        if lump0.offset <= lump1.offset:
            lump0.id = LumpType.LUMP_PLANES
            lump1.id = LumpType.LUMP_ENTITIES
            self.lumps_info.append(lump1)
            self.lumps_info.append(lump0)
            for lump_id in range(2, 15):
                lump_info = LumpInfo.from_buffer(buffer, LumpType(lump_id))
                self.lumps_info.append(lump_info)
        else:
            self.lumps_info.append(lump0)
            self.lumps_info.append(lump1)
            for lump_id in range(2, 15):
                lump_info = LumpInfo.from_buffer(buffer, LumpType(lump_id))
                self.lumps_info.append(lump_info)
        assert self.version in (29, 30), 'Not a GoldSRC map file (BSP29, BSP30)'
        return self

    def get_lump(self, lump_type: LumpType) -> Optional[Lump]:
        if lump_type not in self.lumps:
            for lump_handler in Lump.__subclasses__():
                if not hasattr(lump_handler, 'LUMP_TYPE'):
                    raise TypeError(f'Lump handler has no attribute \'LUMP_TYPE\': {lump_handler.__name__}')

                if getattr(lump_handler, 'LUMP_TYPE') == lump_type:
                    lump_info = self.lumps_info[lump_type.value]
                    lump_data = lump_handler(lump_info)
                    lump_data.parse(self.buffer.slice(lump_info.offset, lump_info.length), self)

                    self.lumps[lump_type] = lump_data

        return self.lumps[lump_type]
