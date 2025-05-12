from dataclasses import dataclass
from typing import List

from .....utils import Buffer


@dataclass(slots=True)
class MaterialReplacement:
    material_id: int
    replacement_material_name: str

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        entry = buffer.tell()
        material_id = buffer.read_int16()
        replacement_material_name = buffer.read_source1_string(entry)
        return cls(material_id, replacement_material_name)


@dataclass(slots=True)
class MaterialReplacementList:
    replacements: List[MaterialReplacement]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        entry = buffer.tell()
        replacements_count, replacement_offset = buffer.read_fmt('2i')
        replacements = []
        with buffer.read_from_offset(entry + replacement_offset):
            for _ in range(replacements_count):
                mat = MaterialReplacement.from_buffer(buffer)
                replacements.append(mat)
        return cls(replacements)
