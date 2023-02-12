from pathlib import Path
from typing import List, Optional

from ...shared.content_providers.content_manager import ContentManager
from ...utils import Buffer, FileBuffer
from .lump import Lump, LumpInfo, LumpType


class BspFile:
    def __init__(self, filepath: Path, buffer: Buffer):
        self.manager: ContentManager = ContentManager()
        self.manager.scan_for_content(filepath)
        self.buffer = buffer
        self.lumps = {}
        self.lumps_info: List[LumpInfo] = []
        self.version = 0

    @classmethod
    def from_filename(cls, filepath: Path):
        buffer = FileBuffer(filepath)
        self = cls(filepath, buffer)
        self.version = buffer.read_uint32()
        self.lumps_info = []
        for lump_id in LumpType:
            lump_info = LumpInfo.from_buffer(buffer, lump_id)
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
