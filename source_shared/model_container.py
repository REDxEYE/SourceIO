from collections import defaultdict
from typing import List, Optional, Dict
import bpy

from ..source1.mdl.mdl_file import Mdl as S1Mdl
from ..goldsrc.mdl.mdl_file import Mdl as GMdl
from ..goldsrc.mdl_v4.mdl_file import Mdl as GMdlV4
from ..source2.resouce_types.valve_model import ValveCompiledModel
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


class GoldSrcV4ModelContainer(ModelContainer):
    def __init__(self, mdl: GMdlV4):
        super().__init__()
        self.mdl: GMdlV4 = mdl


class Source1ModelContainer(ModelContainer):
    def __init__(self, mdl: S1Mdl, vvd: Vvd, vtx: Vtx):
        super().__init__()
        self.mdl: S1Mdl = mdl
        self.vvd: Vvd = vvd
        self.vtx: Vtx = vtx
        self.attachments = []


class Source2ModelContainer(ModelContainer):
    def __init__(self, vmdl: ValveCompiledModel):
        super().__init__()
        self.vmdl = vmdl
        self.physics_objects: List[bpy.types.Object] = []
        self.attachments = []
