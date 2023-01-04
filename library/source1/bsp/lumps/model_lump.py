from typing import List

from ....utils import Buffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile
from ..datatypes.model import Model, RespawnModel


@lump_tag(14, 'LUMP_MODELS')
class ModelLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.models: List[Model] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        model_class = Model if bsp.version < 29 else RespawnModel
        while buffer:
            self.models.append(model_class.from_buffer(buffer, self.version, bsp))
        return self
