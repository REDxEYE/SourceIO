import numpy as np

from ..structs.flex import VertexAminationType
from .....logger import SLoggingManager
from ...mdl.structs.mesh import Mesh
from ...vvd import Vvd
from .mdl_file import MdlV44

logger = SLoggingManager().get_logger("Source1::VertexAnimationCache")

DELTA_DTYPE = np.dtype([
    # ("index", np.int32, (1,)),
    ("pos", np.float32, (3,)),
    ("normal", np.float32, (3,)),
    ("wrinkle", np.float32, (1,)),
])


class VertexAnimationCache:

    def __init__(self, mdl: MdlV44, vvd: Vvd):
        self.vertex_cache = {}
        self.wrinkle_enabled = {}
        # self.wrinkle_cache = []
        self.mdl = mdl
        self.vvd = vvd
        self.vertex_offset = 0

    def process_data(self):
        for bodypart in self.mdl.body_parts:
            for model in bodypart.models:
                if model.vertex_count == 0:
                    continue
                for mesh in model.meshes:
                    if mesh.flexes:
                        self.process_mesh(mesh)
                self.vertex_offset += model.vertex_count

    def process_mesh(self, mesh: Mesh, desired_lod=0):
        desired_lod_ = self.vvd.lod_data[desired_lod]
        # self.wrinkle_cache = np.zeros((len(desired_lod_), 4), np.float32)
        for flex in mesh.flexes:
            flex_name = self.mdl.flex_names[flex.flex_desc_index]
            if flex_name not in self.vertex_cache:
                self.vertex_cache[flex_name] = np.zeros(desired_lod_['vertex'].shape[0], DELTA_DTYPE)
            # Convert array to uint32 because uint16 could overflow on big models
            index_ = flex.vertex_animations['index'].astype(np.uint32).reshape(-1)
            vertex_indices = index_ + mesh.vertex_index_start + self.vertex_offset
            # vertex_cache[vertex_indices]["index"][:] = (index_ + mesh.vertex_index_start)[:, None]
            vertex_data = self.vertex_cache[flex_name]
            vertex_data["pos"][vertex_indices] = flex.vertex_animations['vertex_delta']
            vertex_data["normal"][vertex_indices] = flex.vertex_animations['normal_delta']
            if flex.vertex_anim_type == VertexAminationType.WRINKLE:
                vertex_data["wrinkle"][vertex_indices] = flex.vertex_animations['wrinkle_delta']
                self.wrinkle_enabled[flex_name] = True
            # self.wrinkle_cache[vertex_indices, 0] += flex.vertex_animations['speed'].flatten() / 255
            # if flex.vertex_anim_type == VertexAminationType.WRINKLE:
            #     self.wrinkle_cache[vertex_indices, 3] += flex.vertex_animations['wrinkle_delta'].flatten() / (65535 / 2)

        pass
