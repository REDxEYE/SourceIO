from typing import List

from ...new_shared.base import Base
from ....byte_io_mdl import ByteIO


class MaterialReplacementList(Base):

    def __init__(self):
        self.replacements = []  # type: List[MaterialReplacement]

    def read(self, reader: ByteIO):
        entry = reader.tell()
        replacements_count, replacement_offset = reader.read_fmt('2i')
        with reader.save_current_pos():
            reader.seek(entry + replacement_offset)
            for _ in range(replacements_count):
                mat = MaterialReplacement()
                mat.read(reader)
                self.replacements.append(mat)
        return self


class MaterialReplacement(Base):

    def __init__(self):
        self.material_id = 0
        self.replacement_material_name = ''

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.material_id = reader.read_int16()
        self.replacement_material_name = reader.read_source1_string(entry)
