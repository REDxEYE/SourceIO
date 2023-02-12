from dataclasses import dataclass
from typing import List

from ...utils import Buffer
from .structs.bodypart import StudioBodypart
from .structs.bone import StudioBone
from .structs.studioheader import StudioHeader
from .structs.texture import StudioTexture


@dataclass(slots=True)
class Mdl:
    header: StudioHeader
    bones: List[StudioBone]
    bodyparts: List[StudioBodypart]
    textures: List[StudioTexture]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        header = StudioHeader.from_buffer(buffer)

        buffer.seek(header.bone_offset)
        bones = []
        for _ in range(header.bone_count):
            bone = StudioBone.from_buffer(buffer)
            bones.append(bone)

        buffer.seek(header.body_part_offset)
        bodyparts = []
        for _ in range(header.body_part_count):
            bodypart = StudioBodypart.from_buffer(buffer)
            bodyparts.append(bodypart)

        buffer.seek(header.texture_offset)
        textures = []
        for _ in range(header.texture_count):
            texture = StudioTexture.from_buffer(buffer)
            textures.append(texture)
        return cls(header, bones, bodyparts, textures)
