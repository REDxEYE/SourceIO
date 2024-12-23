from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.source1.bsp import Lump, LumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.source1.bsp.datatypes.model import Model, RespawnModel, DMModel, RavenModel
from SourceIO.library.utils import Buffer


@lump_tag(14, 'LUMP_MODELS')
class ModelLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.models: list[Model] = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        model_class = RespawnModel if bsp.version == (29, 0) else Model
        model_class = DMModel if bsp.version == (20, 4) else model_class
        while buffer:
            self.models.append(model_class.from_buffer(buffer, self.version, bsp))
        return self


@lump_tag(7, 'LUMP_MODELS', bsp_version=(1, 0), steam_id=SteamAppId.SOLDIERS_OF_FORTUNE2)
class RavenModelLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.models: list[RavenModel] = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        while buffer:
            self.models.append(RavenModel.from_buffer(buffer, self.version, bsp))
        return self
