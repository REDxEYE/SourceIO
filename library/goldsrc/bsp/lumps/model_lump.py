from typing import TYPE_CHECKING, List

from ....utils import Buffer
from ..lump import Lump, LumpInfo, LumpType
from ..structs.model import Model

if TYPE_CHECKING:
    from ..bsp_file import BspFile


class ModelLump(Lump):
    LUMP_TYPE = LumpType.LUMP_MODELS

    def __init__(self, info: LumpInfo):
        super().__init__(info)
        self.values: List[Model] = []

    def parse(self, buffer: Buffer, bsp: 'BspFile'):
        while buffer:
            self.values.append(Model.from_buffer(buffer))
