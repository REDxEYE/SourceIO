import numpy as np

from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.source1.bsp import Lump, ValveLumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import VBSPFile
from SourceIO.library.source1.bsp.datatypes.occluderdata import OccluderData, OccluderPolyData
from SourceIO.library.utils import Buffer




@lump_tag(9, 'LUMP_OCCLUSION')
class OcclusionLump(Lump):

    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.occluders_data = []
        self.occluders_poly_data = []
        self.indices = []

    def parse(self, buffer: Buffer, bsp: VBSPFile):
        count = buffer.read_uint32()
        for _ in range(count):
            self.occluders_data.append(OccluderData.from_buffer(buffer, self.version, bsp))
        count = buffer.read_uint32()
        for _ in range(count):
            self.occluders_poly_data.append(OccluderPolyData.from_buffer(buffer, self.version, bsp))
        count = buffer.read_uint32()
        self.indices = np.frombuffer(buffer.read(count*4), dtype=np.uint32)
        return self
