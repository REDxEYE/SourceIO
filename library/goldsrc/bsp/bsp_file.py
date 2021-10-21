from pathlib import Path
from typing import Optional

from .lump import LumpType, LumpInfo, Lump
from ...shared.content_providers.content_manager import ContentManager
from ...utils.byte_io_mdl import ByteIO


class BspFile:
    def __init__(self, file: Path):
        self.manager: ContentManager = ContentManager()
        self.manager.scan_for_content(file)
        self.handle = ByteIO(file.open('rb'))
        self.version = self.handle.read_uint32()
        self.lumps_info = [LumpInfo(self, lump_type) for lump_type in LumpType]
        self.lumps = {}
        assert self.version in (29, 30), 'Not a GoldSRC map file (BSP29, BSP30)'

    def __del__(self):
        return self.handle.close()

    def get_lump(self, lump_type: LumpType) -> Optional[Lump]:
        if lump_type not in self.lumps:
            for lump_handler in Lump.__subclasses__():
                if not hasattr(lump_handler, 'LUMP_TYPE'):
                    raise TypeError(f'Lump handler has no attribute \'LUMP_TYPE\': {lump_handler.__name__}')

                if getattr(lump_handler, 'LUMP_TYPE') == lump_type:
                    lump_info = self.lumps_info[lump_type.value]
                    lump_data = lump_handler(lump_info)
                    lump_data.parse()

                    self.lumps[lump_type] = lump_data

        return self.lumps[lump_type]
