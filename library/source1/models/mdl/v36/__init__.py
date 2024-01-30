from pathlib import Path
from typing import Optional

import numpy as np

from SourceIO.library.shared.content_providers.content_manager import ContentManager
from SourceIO.library.shared.intermediate_data import Model, Skeleton, Mesh, ShapeKey, Material
from SourceIO.library.shared.intermediate_data.material import MaterialMode
from SourceIO.library.utils.path_utilities import find_vtx_cm
from SourceIO.logger import SLoggingManager
from .mdl_file import MdlV36
from ..common import convert_mdl_skeleton, merge_meshes
from ..structs.header import StudioHDRFlags
from ...vtx.v6.vtx import Vtx

log_manager = SLoggingManager()
logger = log_manager.get_logger('Source1 model')


def convert_to_mdl_intermediate_format(mdl_path: Path, content_manager: ContentManager,
                                       load_lods: bool = False) -> Model:
    mdl_buffer = content_manager.find_file(mdl_path)
    if mdl_buffer is None:
        logger.error(f'Could not find model file: {mdl_path}')
        raise FileNotFoundError(mdl_path)
    mdl = MdlV36.from_buffer(mdl_buffer)
    vvd_buffer = content_manager.find_file(mdl_path.with_suffix('.vvd'))
    if vvd_buffer is None:
        logger.error(f"Could not find vertex buffer file: {mdl_path.with_suffix('.vvd')}")
        raise FileNotFoundError(mdl_path.with_suffix('.vvd'))

    vtx_buffer = find_vtx_cm(mdl_path, content_manager)
    if vtx_buffer is None:
        logger.error(f"Could not find optimized face indices file for: {mdl_path.with_suffix('')}")
        raise FileNotFoundError(mdl_path.with_suffix('.dx00.vtx'))
    vtx = Vtx.from_buffer(vtx_buffer)

    model_name = Path(mdl.header.name).stem

    static_prop = mdl.header.flags & StudioHDRFlags.STATIC_PROP != 0

    skeleton: Optional[Skeleton] = None
    if not static_prop and mdl.bones:
        skeleton = convert_mdl_skeleton(mdl)

    lods = []
    if load_lods:
        raise NotImplementedError()
        pass
    else:
        desired_lod = 0
        meshes = []
        for vtx_body_part, body_part in zip(vtx.body_parts, mdl.body_parts):
            for vtx_model, mdl_model in zip(vtx_body_part.models, body_part.models):

                if mdl_model.vertex_count == 0:
                    continue
                vertex_slice = slice(mdl_model.vertex_offset, mdl_model.vertex_offset + mdl_model.vertex_count)
                model_vertices = mdl_model.vertices
                vtx_vertices, indices_array, material_indices_array = merge_meshes(mdl_model,
                                                                                   vtx_model.model_lods[desired_lod])
                material_names = []
                material_remapper = np.zeros((material_indices_array.max() + 1,), dtype=np.uint32)
                for mat_id in np.unique(material_indices_array):
                    mat_name = mdl.materials[mat_id].name
                    material_names.append(mdl.materials[mat_id].name)
                    material_remapper[mat_id] = material_names.index(mat_name)

                indices_array = np.array(indices_array, dtype=np.uint32)
                vertices = model_vertices[vtx_vertices]
                uv0 = vertices['uv']
                uv0[:, 1] = 1 - uv0[:, 1]
                vertex_attributes = {
                    "position": vertices['vertex'],
                    "normal": vertices['normal'],
                    "uv0": uv0,
                }
                if not static_prop:
                    vertex_attributes["blend_indices"] = vertices['bone_id']
                    vertex_attributes["blend_weights"] = vertices['weight']
                mesh = Mesh(Path(mdl_model.name).with_suffix("").as_posix(),
                            body_part.name, vertex_attributes,
                            np.flip(indices_array).reshape((-1, 3)),
                            material_remapper[material_indices_array[::-1]],
                            material_names)
                meshes.append(mesh)
                if mdl.flex_names:
                    for mdl_mesh in mdl_model.meshes:
                        for flex in mdl_mesh.flexes:
                            flex_name = mdl.flex_names[flex.flex_desc_index]
                            flex_vertices = model_vertices['vertex'].copy()
                            flex_normal = model_vertices['normal'].copy()
                            vertex_indices = flex.vertex_animations['index'].reshape(-1) + mdl_mesh.vertex_index_start
                            flex_vertices[vertex_indices] = np.add(flex_vertices[vertex_indices],
                                                                   flex.vertex_animations['vertex_delta'])
                            flex_normal[vertex_indices] = np.add(flex_normal[vertex_indices],
                                                                 flex.vertex_animations['normal_delta'])
                            mesh.shape_keys[flex_name] = ShapeKey(flex_name,
                                                                  flex_vertices,
                                                                  flex_normal)
        lods.append((0, meshes))

    materials: list[Material] = []
    for material in mdl.materials:
        full_material_path = None
        for mat_path in mdl.materials_paths:
            full_material_path = Path(mat_path) / material.name
            material_path = content_manager.find_material(full_material_path)
            if material_path:
                break
        materials.append(Material(material.name, full_material_path.as_posix(), MaterialMode.source1))

    return Model(model_name, skeleton, lods, materials)
