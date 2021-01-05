import math
from typing import List

import numpy as np

from .structs.bone import StudioBone
from .structs.studioheader import StudioHeader
from .structs.bodypart import StudioBodypart
from ...source_shared.base import Base
from ...utilities.byte_io_mdl import ByteIO
from ...utilities.math_utilities import angle_matrix, r_concat_transforms


class Mdl(Base):

    def __init__(self, filepath):
        self.store_value("MDL", self)
        self.reader = ByteIO(filepath)
        self.header = StudioHeader()
        self.bones: List[StudioBone] = []
        self.bone_transforms: List[np.ndarray] = []
        self.bodyparts: List[StudioBodypart] = []

    def read(self):
        self.header.read(self.reader)

        self.reader.seek(self.header.bone_offset)
        for _ in range(self.header.bone_count):
            bone = StudioBone()
            bone.read(self.reader)
            self.bones.append(bone)

        for bone in self.bones:
            bone_matrix = angle_matrix(bone.rot[0], bone.rot[1], bone.rot[2])
            bone_matrix[0][3] = bone.pos[2]
            bone_matrix[1][3] = bone.pos[0]
            bone_matrix[2][3] = bone.pos[1]
            if bone.parent != -1:
                parent = self.bone_transforms[bone.parent]
                bone_matrix = r_concat_transforms(parent, bone_matrix)
            self.bone_transforms.append(bone_matrix)

        self.reader.seek(self.header.body_part_offset)
        for _ in range(self.header.body_part_count):
            bodypart = StudioBodypart()
            bodypart.read(self.reader)
            self.bodyparts.append(bodypart)
