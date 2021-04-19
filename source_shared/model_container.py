from collections import defaultdict
from typing import List, Optional, Dict
import bpy

from ..source1.mdl.mdl_file import Mdl as S1Mdl
from ..goldsrc.mdl.mdl_file import Mdl as GMdl
from ..source1.vtx.vtx import Vtx
from ..source1.vvd import Vvd


class ModelContainer:
    def __init__(self):
        self.armature: Optional[bpy.types.Object] = None
        self.objects: List[bpy.types.Object] = []
        self.bodygroups: Dict[str, List[bpy.types.Object]] = defaultdict(list)


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
        self.attachments = []
