from typing import List, Optional
import bpy

from ..source1.mdl.mdl_file import Mdl as S1Mdl
from ..goldsrc.mdl.mdl_file import Mdl as GMdl
from ..source1.vtx.vtx import Vtx
from ..source1.vvd.vvd import Vvd


class ModelContainer:
    def __init__(self):
        self.armature: Optional[bpy.types.Object] = None
        self.objects: List[bpy.types.Object] = []


class GoldSrcModelContainer(ModelContainer):
    def __init__(self, mdl: GMdl):
        super().__init__()
        self.mdl: GMdl = mdl


class Source1ModelContainer(ModelContainer):
    def __init__(self, mdl: S1Mdl, vvd: Vvd, vtx: Vtx):
        super().__init__()
        self.mdl: S1Mdl = mdl
        self.vvd: Vvd = vvd
        self.vtx: Vtx = vtx
        self.collection = None
