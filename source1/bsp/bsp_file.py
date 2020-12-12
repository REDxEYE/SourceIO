from pathlib import Path

from typing import Dict, Type, List

from .lump import *
from .lumps.displacement_lump import DispInfoLump, DispVert
from .lumps.edge_lump import EdgeLump
from .lumps.entity_lump import EntityLump
from .lumps.face_lump import FaceLump, OriginalFaceLump
from .lumps.game_lump import GameLump
from .lumps.model_lump import ModelLump
from .lumps.node_lump import NodeLump
from .lumps.pak_lump import PakLump
from .lumps.plane_lump import PlaneLump
from .lumps.string_lump import StringsLump
from .lumps.surf_edge_lump import SurfEdgeLump
from .lumps.texture_lump import TextureDataLump, TextureInfoLump
from .lumps.vertex_lump import VertexLump
from .lumps.vertex_normal_lump import VertexNormalLump, VertexNormalIndicesLump
from .lumps.world_light_lump import WorldLightLump

from ...utilities.byte_io_mdl import ByteIO


class BSPFile:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.reader = ByteIO(self.filepath)
        self.version = 0
        self.lumps_info: List[LumpInfo] = []
        self.lumps: Dict[LumpTypes, Lump] = {}
        self.revision = 0

    def parse(self):
        reader = self.reader
        magic = reader.read_fourcc()
        assert magic == "VBSP", "Invalid BSP header"
        self.version = reader.read_int32()
        for lump_id in range(64):
            lump = LumpInfo(lump_id)
            lump.parse(reader)
            self.lumps_info.append(lump)
        self.revision = reader.read_int32()
        self.parse_lumps()

    def parse_lumps(self):
        self.parse_lump(EntityLump)
        self.parse_lump(VertexLump)
        self.parse_lump(PlaneLump)
        self.parse_lump(EdgeLump)
        self.parse_lump(SurfEdgeLump)
        self.parse_lump(VertexNormalLump)
        # self.parse_lump(VertexNormalIndicesLump)
        self.parse_lump(FaceLump)
        # self.parse_lump(OriginalFaceLump)
        self.parse_lump(TextureDataLump)
        self.parse_lump(TextureInfoLump)
        self.parse_lump(StringsLump)
        self.parse_lump(ModelLump)
        # self.parse_lump(WorldLightLump)
        # self.parse_lump(DispInfoLump)
        self.parse_lump(DispVert)
        # self.parse_lump(NodeLump)
        self.parse_lump(GameLump)
        self.parse_lump(PakLump)

    def parse_lump(self, lump_class: Type[Lump]):
        if self.lumps_info[lump_class.lump_id].size != 0:
            lump = self.lumps_info[lump_class.lump_id]
            print(f"Loading {lump_class.lump_id.name} lump.\n\tOffset: {lump.offset}\n\tSize:{lump.size}")
            self.lumps[lump_class.lump_id] = lump_class(self).parse()
