import struct

from .structs.material_replacement_list import MaterialReplacementList

from ..v6.vtx import Vtx as Vtx6
from .structs.bodypart import BodyPart


class Vtx(Vtx6):
    def read(self):
        self.header.read(self.reader)

        try:
            self.store_value('extra8', False)
            self.reader.seek(self.header.body_part_offset)
            for _ in range(self.header.body_part_count):
                body_part = BodyPart()
                body_part.read(self.reader)
                self.body_parts.append(body_part)
        except (struct.error, AssertionError):
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
