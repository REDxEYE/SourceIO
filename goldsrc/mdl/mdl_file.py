import math
from typing import List

import numpy as np

from .structs.bone import StudioBone
from .structs.studioheader import StudioHeader
from .structs.bodypart import StudioBodypart
from .structs.texture import StudioTexture
from ...source_shared.base import Base
from ...utilities.byte_io_mdl import ByteIO


class Mdl(Base):

    def __init__(self, filepath):
        self.store_value("MDL", self)
        self.reader = ByteIO(filepath)
        self.header = StudioHeader()
        self.bones: List[StudioBone] = []
        self.bodyparts: List[StudioBodypart] = []
        self.textures: List[StudioTexture] = []

    def read(self):
        self.header.read(self.reader)

        self.reader.seek(self.header.bone_offset)
        for _ in range(self.header.bone_count):
            bone = StudioBone()
            bone.read(self.reader)
            self.bones.append(bone)

        self.reader.seek(self.header.body_part_offset)
        for _ in range(self.header.body_part_count):
            bodypart = StudioBodypart()
            bodypart.read(self.reader)
            self.bodyparts.append(bodypart)

        self.reader.seek(self.header.texture_offset)
        for _ in range(self.header.texture_count):
            texture = StudioTexture()
            texture.read(self.reader)
            self.textures.append(texture)
