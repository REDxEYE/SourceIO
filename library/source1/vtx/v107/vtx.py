import struct
from dataclasses import dataclass
from typing import List

from ....utils import Buffer
from ..v6.structs.header import Header
from ..v6.vtx import Vtx as Vtx6
from .structs.bodypart import BodyPart
from .structs.material_replacement_list import MaterialReplacementList


@dataclass(slots=True)
class Vtx():
    header: Header
    body_parts: List[BodyPart]
    # TODO: Does this exist in 107?
    # material_replacement_lists: List[MaterialReplacementList]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        # Header is ok for 107
        header = Header.from_buffer(buffer)
        try:
            buffer.seek(header.body_part_offset)
            body_parts = []
            for _ in range(header.body_part_count):
                body_part = BodyPart.from_buffer(buffer)
                body_parts.append(body_part)
        except (struct.error, AssertionError):
            print('Got error when doing stuff')
            buffer.seek(header.body_part_offset)
            body_parts = []
            for _ in range(header.body_part_count):
                body_part = BodyPart.from_buffer(buffer, True)
                body_parts.append(body_part)

        # buffer.seek(header.material_replacement_list_offset)
        # material_replacement_lists = []
        # for _ in range(header.lod_count):
        #     material_replacement_list = MaterialReplacementList.from_buffer(buffer)
        #     material_replacement_lists.append(material_replacement_list)
        return cls(header, body_parts)
