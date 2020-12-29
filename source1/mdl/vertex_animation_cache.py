import numpy as np

from ..shared.base import Base
from ..vvd.vvd import Vvd
from .mdl import Mdl
from .structs.mesh import Mesh
from .structs.model import Model


class VertexAnimationCache(Base):

    def __init__(self, mdl: Mdl, vvd: Vvd):
        self.vertex_cache = {}
        self.mdl = mdl
        self.vvd = vvd

    def process_data(self):
        print("[WIP ]Pre-computing vertex animation cache")
        for bodypart in self.mdl.body_parts:
            print(f'Processing bodypart "{bodypart.name}"')
            for model in bodypart.models:
                if model.vertex_count == 0:
                    continue
                print(f'\t+--model "{model.name}"')
                for mesh in model.meshes:
                    if mesh.flexes:
                        self.process_mesh(mesh, model.vertex_offset)
        print("[Done] Pre-computing vertex animation cache")

    def process_mesh(self, mesh: Mesh, vertex_offset, desired_lod=0):
        for flex in mesh.flexes:
            if flex.name not in self.vertex_cache:
                vertex_cache = self.vertex_cache[flex.name] = np.copy(self.vvd.lod_data[desired_lod]['vertex'])
            else:
                vertex_cache = self.vertex_cache[flex.name]
            for v_anim in flex.vertex_animations:
                vertex_index = v_anim['index'][0] + mesh.vertex_index_start + vertex_offset
                vertex_cache[vertex_index] = np.add(vertex_cache[vertex_index], v_anim['vertex_delta'])

        pass
