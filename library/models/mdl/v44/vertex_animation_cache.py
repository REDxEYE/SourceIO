import numpy as np

from SourceIO.library.models.mdl.structs.header import StudioHDRFlags
from SourceIO.library.models.mdl.structs.mesh import Mesh
from SourceIO.library.models.vvd import Vvd
from SourceIO.logger import SourceLogMan
from .mdl_file import MdlV44

logger = SourceLogMan().get_logger("Source1::VertexAnimationCache")

DELTA_DTYPE = np.dtype([
    # ("index", np.int32, (1,)),
    ("pos", np.float32, (3,)),
    ("normal", np.float32, (3,)),
    ("wrinkle", np.float32, (1,)),
])


def preprocess_vertex_animation(mdl: MdlV44, vvd: Vvd) -> dict[str, DELTA_DTYPE]:
    if mdl.header.flags & StudioHDRFlags.STATIC_PROP != 0:
        return {}

    vertex_offset = 0
    vertex_cache = {}

    def process_mesh(mesh: Mesh, desired_lod=0):
        desired_lod_ = vvd.lod_data[desired_lod]
        for flex in mesh.flexes:
            flex_name = mdl.flex_names[flex.flex_desc_index]
            if flex_name not in vertex_cache:
                vertex_cache[flex_name] = np.zeros(desired_lod_['vertex'].shape[0], DELTA_DTYPE)
            # Convert array to uint32 because uint16 could overflow on big models
            index_ = flex.vertex_animations['index'].astype(np.uint32).reshape(-1)
            vertex_indices = index_ + mesh.vertex_index_start + vertex_offset
            vertex_data = vertex_cache[flex_name]
            vertex_data["pos"][vertex_indices] = flex.vertex_animations['vertex_delta']
            vertex_data["normal"][vertex_indices] = flex.vertex_animations['normal_delta']

    for body_part in mdl.body_parts:
        for model in body_part.models:
            if model.vertex_count == 0:
                continue
            for mesh in model.meshes:
                if mesh.flexes:
                    process_mesh(mesh)
            vertex_offset += model.vertex_count
    return vertex_cache
