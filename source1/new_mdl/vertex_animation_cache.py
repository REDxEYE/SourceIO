import numpy as np

from ..new_shared.base import Base
from ..new_vvd.vvd import Vvd
from .mdl import Mdl
from .structs.mesh import Mesh
from .structs.model import Model


class VertexAnimationCache(Base):

    def __init__(self, mdl: Mdl, vvd: Vvd):
        self.vertex_cache = {}
        self.mdl = mdl
        self.vvd = vvd

    def process_data(self):
        print("[WIP ]Pre-computing vertex animation vertex positions")
        for bodypart in self.mdl.body_parts:
            for model in bodypart.models:
                for mesh in model.meshes:
                    self.process_mesh(mesh)
        print("[Done] Pre-computing vertex animation vertex positions")

    def process_mesh(self, mesh: Mesh):
        for flex in mesh.flexes:
            if flex.name not in self.vertex_cache:
                vertex_cache = self.vertex_cache[flex.name] = np.copy(self.vvd.vertices)
            else:
                vertex_cache = self.vertex_cache[flex.name]
            for v_anim in flex.vertex_animations:
                vertex_index = v_anim['index'][0] + mesh.vertex_index_start
                vertex_cache[vertex_index] = np.add(vertex_cache[vertex_index], v_anim['vertex_delta'])

        pass
