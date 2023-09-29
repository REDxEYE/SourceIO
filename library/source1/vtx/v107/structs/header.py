from dataclasses import dataclass

from .....utils import Buffer


@dataclass(slots=True)
class Header:
    version: int
    vertex_cache_size: int
    max_bones_per_strip: int
    max_bones_per_tri: int
    max_bones_per_vertex: int
    checksum: int
    lod_count: int
    material_replacement_list_offset: int
    body_part_count: int
    body_part_offset: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        version = buffer.read_uint32()
        vertex_cache_size = buffer.read_uint32()
        max_bones_per_strip = buffer.read_uint16()
        max_bones_per_tri = buffer.read_uint16()
        max_bones_per_vertex = buffer.read_uint32()
        checksum = buffer.read_uint32()
        lod_count = buffer.read_uint32()
        material_replacement_list_offset = buffer.read_uint32()
        body_part_count = buffer.read_uint32()
        body_part_offset = buffer.read_uint32()
        assert 3 == max_bones_per_vertex, 'Unsupported count of bones per vertex'
        return cls(version, vertex_cache_size, max_bones_per_strip, max_bones_per_tri, max_bones_per_vertex, checksum,
                   lod_count, material_replacement_list_offset, body_part_count, body_part_offset)
