import math
from pathlib import Path

import numpy as np

from SourceIO.library.goldsrc.mdl_v10.mdl_file import Mdl
from SourceIO.library.shared.content_providers.content_manager import ContentManager
from SourceIO.library.shared.intermediate_data import Model, Skeleton, Bone, Mesh
from SourceIO.library.shared.intermediate_data.bone import BoneFlags
from SourceIO.library.source1.models.mdl.common import create_transformation_matrix, create_rotation_matrix
from SourceIO.logger import SLoggingManager

log_manager = SLoggingManager()
logger = log_manager.get_logger('Goldsrc model')


def convert_to_mdl_intermediate_format(mdl_path: Path, content_manager: ContentManager,
                                       load_lods: bool = False) -> Model:
    mdl_buffer = content_manager.find_file(mdl_path)
    if mdl_buffer is None:
        logger.error(f'Could not find model file: {mdl_path}')
        raise FileNotFoundError(mdl_path)

    mdl = Mdl.from_buffer(mdl_buffer)
    model_name = Path(mdl.header.name).stem
    texture_file = mdl_path.with_name(Path(mdl_path.name).stem + 't.mdl')

    texture_mdl_buffer = content_manager.find_file(texture_file)
    mdl_file_textures = mdl.textures
    if not mdl_file_textures and texture_mdl_buffer is not None:
        texture_mdl = Mdl.from_buffer(texture_mdl_buffer)
        mdl_file_textures = texture_mdl.textures

    skeleton = None
    if mdl.bones:
        skeleton = Skeleton(model_name)

        rot_90_x = np.eye(4)
        rot_90_x[:3, :3] = create_rotation_matrix((-math.pi / 2, 0, 0))

        for n, mdl_bone in enumerate(mdl.bones):
            bone_name = mdl_bone.name or f'Bone_{n}'
            local_matrix = create_transformation_matrix(mdl_bone.pos, mdl_bone.rot)
            parent_name = ""
            if mdl_bone.parent >= 0:
                parent_matrix = skeleton.bones[mdl_bone.parent].world_matrix
                world_matrix = parent_matrix @ local_matrix
                parent_name = mdl.bones[mdl_bone.parent_bone_id].name
            else:
                world_matrix = rot_90_x @ local_matrix

            skeleton.bones.append(Bone(bone_name, parent_name,
                                       BoneFlags.NO_BONE_FLAGS,
                                       world_matrix, local_matrix))
    meshes = []
    for body_part in mdl.bodyparts:
        for body_model in body_part.models:
            if len(body_model.vertices) == 0:
                continue
            vertex_attributes = {
                "position": body_model.vertices,
                "normal": body_model.normals,
            }

            meshes.append(Mesh(body_model.name, body_part.name, ))
