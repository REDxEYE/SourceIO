from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.source1.bsp import Lump, ValveLumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import VBSPFile, IBSPFile
from SourceIO.library.source1.bsp.datatypes.model import Model, RespawnModel, DMModel, QuakeBspModel, QuakeBspModel
from SourceIO.library.utils import Buffer


@lump_tag(14, 'LUMP_MODELS')
class ModelLump(Lump):
    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.models: list[Model] = []

    def parse(self, buffer: Buffer, bsp: VBSPFile):
        model_class = RespawnModel if bsp.info.version == (29, 0) else Model
        model_class = DMModel if bsp.info.version == (20, 4) else model_class
        while buffer:
            self.models.append(model_class.from_buffer(buffer, self.version, bsp))
        return self

@lump_tag(7, 'LUMP_MODELS', bsp_ident="IBSP", bsp_version=(46, 0))
@lump_tag(7, 'LUMP_MODELS', bsp_ident="RBSP", bsp_version=(1, 0))
class Quake3ModelLump(Lump):
    def __init__(self, lump_info: ValveLumpInfo):
        super().__init__(lump_info)
        self.models: list[QuakeBspModel] = []

    def parse(self, buffer: Buffer, bsp: IBSPFile):
        while buffer:
            self.models.append(QuakeBspModel.from_buffer(buffer, self.version, bsp))
        return self
