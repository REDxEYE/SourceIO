import numpy as np

from SourceIO.library.shared.intermidiate_data.bone import Bone, BoneFlags
from SourceIO.library.source2.exceptions import MissingBlock
from SourceIO.library.source2.compiled_resource import CompiledResource, DATA_BLOCK
from SourceIO.library.source2.blocks.kv3_block import KVBlock


class CompiledModelResource(CompiledResource):

    def get_name(self):
        data = self.get_block(KVBlock, block_id=DATA_BLOCK)
        return data['m_name']

    def get_bones(self) -> list[Bone]:
        data = self.get_block(KVBlock, block_id=DATA_BLOCK)
        if data is None:
            raise MissingBlock('Required block "DATA" is missing')
        bones: list[Bone] = []

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
