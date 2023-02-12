from dataclasses import dataclass

from ...utils import Buffer


@dataclass(slots=True)
class Fixup:
    lod_index: int
    vertex_index: int
    vertex_count: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(*buffer.read_fmt("3I"))
