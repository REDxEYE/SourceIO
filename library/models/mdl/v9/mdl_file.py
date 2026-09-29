from dataclasses import dataclass

from SourceIO.library.utils import Buffer
from .structs.bodypart import StudioBodypart
from .structs.bone import StudioBone
from .structs.studioheader import StudioHeader
from .structs.texture import StudioTexture


@dataclass(slots=True)
class Mdl:
    header: StudioHeader
    bones: list[StudioBone]
    bodyparts: list[StudioBodypart]
    textures: list[StudioTexture]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        header = StudioHeader.from_buffer(buffer)

        bones = buffer.read_structure_array(header.bone_offset, header.bone_count, StudioBone)
        bodyparts = buffer.read_structure_array(header.body_part_offset, header.body_part_count, StudioBodypart)
        textures = buffer.read_structure_array(header.texture_offset, header.texture_count, StudioTexture)

        return cls(header, bones, bodyparts, textures)
