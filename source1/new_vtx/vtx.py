import struct
from typing import List

from .structs.MaterialReplacementList import MaterialReplacementList
from ..new_shared.base import Base
from ...byte_io_mdl import ByteIO

from .structs.header import Header
from .structs.bodypart import BodyPart


class Vtx(Base):
    def __init__(self, filepath):
        self.reader = ByteIO(path=filepath)
        self.header = Header()
        self.body_parts = []  # type: List[BodyPart]
        self.material_replacement_lists = []  # type: List[MaterialReplacementList]

    def read(self):
        self.header.read(self.reader)

        try:
            self.reader.seek(self.header.body_part_offset)
            for _ in range(self.header.body_part_count):
                body_part = BodyPart()
                body_part.read(self.reader)
                self.body_parts.append(body_part)
        except (struct.error,AssertionError):
            self.reader.seek(self.header.body_part_offset)
            self.body_parts.clear()
            self.store_value('extra8', True)
            for _ in range(self.header.body_part_count):
                body_part = BodyPart()
                body_part.read(self.reader)
                self.body_parts.append(body_part)

        self.reader.seek(self.header.material_replacement_list_offset)
        for _ in range(self.header.lod_count):
            material_replacement_list = MaterialReplacementList()
            material_replacement_list.read(self.reader)
            self.material_replacement_lists.append(material_replacement_list)
