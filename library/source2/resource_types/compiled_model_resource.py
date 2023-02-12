
from typing import Collection, List

import numpy as np

from ...shared.intermidiate_data.bone import Bone, BoneFlags
from ..exceptions import MissingBlock
from .resource import CompiledResource


class CompiledModelResource(CompiledResource):

    def get_name(self):
        data, = self.get_data_block(block_name='DATA')
        return data['m_name']

    def get_bones(self) -> List[Bone]:
        data, = self.get_data_block(block_name='DATA')
        if data is None:
            raise MissingBlock('Required block "DATA" is missing')
        bones: List[Bone] = []

        model_skeleton = data['m_modelSkeleton']
        names = model_skeleton['m_boneName']
        flags = model_skeleton['m_nFlag']
        parents = model_skeleton['m_nParent']
        positions = model_skeleton['m_bonePosParent']
        rotations = model_skeleton['m_boneRotParent']

        for bone_id, name in enumerate(names):
            parent_id = parents[bone_id]
            if parent_id >= 0:
                parent_name = names[parent_id]
            else:
                parent_name = None
            rotation = rotations[bone_id]
            x, y, z, w = rotation
            bones.append(
                Bone(name, parent_name, BoneFlags(int(flags[bone_id])), positions[bone_id], np.asarray((w, x, y, z))))

        return bones
