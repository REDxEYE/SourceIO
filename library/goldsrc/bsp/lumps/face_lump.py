from typing import TYPE_CHECKING, List

from ....utils import Buffer
from ..lump import Lump, LumpInfo, LumpType
from ..structs.face import Face

if TYPE_CHECKING:
    from ..bsp_file import BspFile


class FaceLump(Lump):
    LUMP_TYPE = LumpType.LUMP_FACES

    def __init__(self, info: LumpInfo):
        super().__init__(info)
        self.values: List[Face] = []

    def parse(self, buffer: Buffer, bsp: 'BspFile'):
        while buffer:
            face = Face.from_buffer(buffer)
            self.values.append(face)
