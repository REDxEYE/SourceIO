from typing import List

from ....utils import Buffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile
from ..datatypes.model import Model, RespawnModel, DMModel


@lump_tag(14, 'LUMP_MODELS')
class ModelLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.models: List[Model] = []

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        model_class = RespawnModel if bsp.version == (29, 0) else Model
        model_class = DMModel if bsp.version == (20, 4) else model_class
        while buffer:
            self.models.append(model_class.from_buffer(buffer, self.version, bsp))
        return self
