import numpy as np

from ..structs.flex import VertexAminationType
from ....shared.base import Base
from ...vvd import Vvd
from .mdl_file import MdlV44
from ...mdl.structs.mesh import MeshV49
from .....logger import SLoggingManager

logger = SLoggingManager().get_logger("Source1::VertexAnimationCache")


class VertexAnimationCache(Base):

    def __init__(self, mdl: MdlV44, vvd: Vvd):
        self.vertex_cache = {}
        self.wrinkle_cache = []
        self.mdl = mdl
        self.vvd = vvd
        self.vertex_offset = 0

    def process_data(self):
        logger.info("[WIP ]Pre-computing vertex animation cache")
        for bodypart in self.mdl.body_parts:
            logger.info(f'Processing bodypart "{bodypart.name}"')
            for model in bodypart.models:
                if model.vertex_count == 0:
                    continue
                logger.info(f'\t+--model "{model.name}"')
                for mesh in model.meshes:
                    if mesh.flexes:
                        self.process_mesh(mesh)
                self.vertex_offset += model.vertex_count
        logger.info("[Done] Pre-computing vertex animation cache")

    def process_mesh(self, mesh: MeshV49, desired_lod=0):
        desired_lod_ = self.vvd.lod_data[desired_lod]
        self.wrinkle_cache = np.zeros((len(desired_lod_), 4), np.float32)
        for flex in mesh.flexes:
            if flex.name not in self.vertex_cache:
                vertex_cache = self.vertex_cache[flex.name] = np.zeros_like(desired_lod_['vertex'])
            else:
                vertex_cache = self.vertex_cache[flex.name]
            # Convert array to uint32 because uint16 could overflow on big models
            index_ = flex.vertex_animations['index'].astype(np.uint32).reshape(-1)
            vertex_indices = index_ + mesh.vertex_index_start + self.vertex_offset
            vertex_cache[vertex_indices] = flex.vertex_animations['vertex_delta']
            self.wrinkle_cache[vertex_indices, 0] += flex.vertex_animations['speed'].flatten() / 255
            if flex.vertex_anim_type == VertexAminationType.WRINKLE:
                self.wrinkle_cache[vertex_indices, 3] += flex.vertex_animations['wrinkle_delta'].flatten() / (65535 / 2)

        pass
