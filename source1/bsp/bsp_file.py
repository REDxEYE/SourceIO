from pathlib import Path

from typing import Dict, Type, List

from .lump import *
from .lumps.displacement_lump import DispVert
from .lumps.edge_lump import EdgeLump
from .lumps.entity_lump import EntityLump
from .lumps.face_lump import FaceLump
from .lumps.game_lump import GameLump
from .lumps.model_lump import ModelLump
from .lumps.pak_lump import PakLump
from .lumps.plane_lump import PlaneLump
from .lumps.string_lump import StringsLump
from .lumps.surf_edge_lump import SurfEdgeLump
from .lumps.texture_lump import TextureDataLump, TextureInfoLump
from .lumps.vertex_lump import VertexLump
from .lumps.vertex_normal_lump import VertexNormalLump
from ...bpy_utilities.logging import BPYLoggingManager
from ...source_shared.content_manager import ContentManager

from ...utilities.byte_io_mdl import ByteIO

log_manager = BPYLoggingManager()


class BSPFile:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.logger = log_manager.get_logger(self.filepath.name)
        self.reader = ByteIO(self.filepath)
        self.version = 0
        self.lumps_info: List[LumpInfo] = []
        self.lumps: Dict[str, Lump] = {}
        self.revision = 0
        self.content_manager = ContentManager()
        content_provider = self.content_manager.get_content_provider_from_path(self.filepath)
        self.steam_app_id = content_provider.steam_id

    def parse(self):
        reader = self.reader
        magic = reader.read_fourcc()
        assert magic == "VBSP", "Invalid BSP header"
        self.version = reader.read_int32()
        is_l4d2 = reader.peek_uint32() <= 1036 and self.version == 21
        for lump_id in range(64):
            lump = LumpInfo(lump_id)
            lump.parse(reader, is_l4d2)
            self.lumps_info.append(lump)
        self.revision = reader.read_int32()

        # self.parse_lumps()

    def get_lump(self, lump_name):

        if lump_name in self.lumps:
            return self.lumps[lump_name]
        else:
            for sub in Lump.__subclasses__():
                sub: Type[Lump]
                for dep in sub.tags:
                    if dep.lump_name == lump_name:
                        if dep.bsp_version is not None and dep.bsp_version > self.version:
                            continue
                        if dep.steam_id is not None and dep.steam_id != self.steam_app_id:
                            continue
                        parsed_lump = self.parse_lump(sub, dep.lump_id, dep.lump_name)
                        self.lumps[lump_name] = parsed_lump
                        return parsed_lump

    def parse_lump(self, lump_class: Type[Lump], lump_id, lump_name):
        if self.lumps_info[lump_id].size != 0:
            lump = self.lumps_info[lump_id]
            self.logger.debug(f'Loading {lump_name} lump', 'bps_parser')
            self.logger.debug(f'\tOffset: {lump.offset}', 'bps_parser')
            self.logger.debug(f'\tSize: {lump.size}', 'bps_parser')
            parsed_lump = lump_class(self, lump_id).parse()
            self.lumps[lump_id] = parsed_lump
            return parsed_lump
