from typing import List

from ....utils import IBuffer
from .. import Lump, lump_tag, LumpInfo
from ..bsp_file import BSPFile
from ..datatypes.model import Model, RespawnModel


@lump_tag(14, 'LUMP_MODELS')
class ModelLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.models: List[Model] = []

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        while buffer:
            if bsp.version < 29:
                self.models.append(Model(self).parse(buffer, bsp))
            else:
                self.models.append(RespawnModel(self).parse(buffer, bsp))
        return self
