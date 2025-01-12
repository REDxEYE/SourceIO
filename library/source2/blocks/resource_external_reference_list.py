from dataclasses import dataclass

from SourceIO.library.source2.utils.ntro_reader import NTROBuffer
from SourceIO.library.utils import Buffer
from SourceIO.library.source2.blocks.base import BaseBlock


@dataclass(slots=True)
class ResourceExternalReference:
    hash: int
    r_id: int
    name: str
    unk: int

    def __repr__(self):
        return '<External resource "{}">'.format(self.name)

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        h, r_id, name_offset, unk = buffer.read_fmt("4I")
        with buffer.read_from_offset(buffer.tell() - 8 + name_offset):
            name = buffer.read_nt_string()
        return cls(h, r_id, name, unk)


class ResourceExternalReferenceList(list[ResourceExternalReference], BaseBlock):

    def __init__(self):
        list.__init__(self)
        self._mapping: dict[int, ResourceExternalReference] = {}

    def __str__(self) -> str:
        str_data = list.__str__(self)
        return f"<ResourceExternalReferenceList  \"{str_data if len(str_data) < 50 else str_data[:50] + '...'}\">"

    @classmethod
    def from_buffer(cls, buffer: NTROBuffer) -> 'ResourceExternalReferenceList':
        offset = buffer.read_relative_offset32()
        count = buffer.read_uint32()
        self = cls()
        with buffer.read_from_offset(offset):
            for _ in range(count):
                ref = ResourceExternalReference.from_buffer(buffer)
                self._mapping[ref.hash] = ref
                self.append(ref)
        return self

    def find_resource(self, resource_id: int):
        if res := self._mapping.get(resource_id & 0xFFFF_FFFF, None):
            return res.name
