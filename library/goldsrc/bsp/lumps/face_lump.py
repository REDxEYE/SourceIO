from SourceIO.library.goldsrc.bsp.bsp_file import BspFile
from SourceIO.library.goldsrc.bsp.lump import Lump, LumpInfo, LumpType
from SourceIO.library.goldsrc.bsp.structs.face import Face
from SourceIO.library.utils import Buffer


class FaceLump(Lump):
    LUMP_TYPE = LumpType.LUMP_FACES

    def __init__(self, info: LumpInfo):
        super().__init__(info)
        self.values: list[Face] = []

    def parse(self, buffer: Buffer, bsp: BspFile):
        while buffer:
            face = Face.from_buffer(buffer)
            self.values.append(face)
