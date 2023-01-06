from dataclasses import dataclass
from typing import Tuple

from ....utils import Buffer


@dataclass(slots=True)
class Material:
    name: str
    flags: int


@dataclass(slots=True)
class MaterialV36(Material):
    width: float
    height: float
    dp_du: float
    dp_dv: float
    unknown: Tuple[int, int]

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        entry = buffer.tell()
        name = buffer.read_source1_string(entry)
        flags = buffer.read_uint32()
        width = buffer.read_float()
        height = buffer.read_float()
        dp_du = buffer.read_float()
        dp_dv = buffer.read_float()
        unknown = buffer.read_fmt('2I')
        return cls(name, flags, width, height, dp_du, dp_dv, unknown)


@dataclass(slots=True)
class MaterialV49(Material):
    used: int
    unused1: int
    material_pointer: int
    client_material_pointer: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        entry = buffer.tell()
        name = buffer.read_source1_string(entry)
        flags = buffer.read_uint32()
        used = buffer.read_uint32()
        unused1 = buffer.read_uint32()
        material_pointer = buffer.read_uint32()
        client_material_pointer = buffer.read_uint32()
        buffer.skip((10 if version < 53 else 5) * 4)
        return cls(name, flags, used, unused1, material_pointer, client_material_pointer)
