import numpy as np

from ....shared.base import Base
from ...vvd import Vvd
from .mdl_file import MdlV44
from ...mdl.structs.mesh import MeshV49
from .....logger import SLoggingManager

logger = SLoggingManager().get_logger("Source1::VertexAnimationCache")


class VertexAnimationCache(Base):

    def __init__(self, mdl: MdlV44, vvd: Vvd):
        self.vertex_cache = {}
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
                        self.process_mesh(mesh, self.vertex_offset)
                self.vertex_offset += model.vertex_count
        logger.info("[Done] Pre-computing vertex animation cache")

    def process_mesh(self, mesh: MeshV49, vertex_offset, desired_lod=0):
        for flex in mesh.flexes:
            if flex.name not in self.vertex_cache:
                vertex_cache = self.vertex_cache[flex.name] = np.zeros_like(self.vvd.lod_data[desired_lod]['vertex'])
            else:
                vertex_cache = self.vertex_cache[flex.name]
            vertex_indices = flex.vertex_animations['index'].reshape(-1) + mesh.vertex_index_start + vertex_offset
            vertex_cache[vertex_indices] = flex.vertex_animations['vertex_delta']

        pass
