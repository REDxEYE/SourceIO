import typing
from dataclasses import dataclass
from typing import Dict, List

from ....utils import Buffer
from .base import BaseBlock
if typing.TYPE_CHECKING:
    from ...resource_types.resource import CompiledResource


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
        return cls(buffer.read_uint32(), buffer.read_uint32(), buffer.read_source2_string(), buffer.read_uint32())


class ResourceExternalReferenceList(List[ResourceExternalReference], BaseBlock):

    def __init__(self, buffer: Buffer, resource: 'CompiledResource'):
        list.__init__(self)
        BaseBlock.__init__(self, buffer, resource)
        self._mapping: Dict[int, ResourceExternalReference] = {}

    def __str__(self) -> str:
        str_data = list.__str__(self)
        return f"<ResourceExternalReferenceList  \"{str_data if len(str_data) < 50 else str_data[:50] + '...'}\">"

    @classmethod
    def from_buffer(cls, buffer: Buffer, resource: 'CompiledResource') -> 'ResourceExternalReferenceList':
        offset = buffer.read_relative_offset32()
        count = buffer.read_uint32()
        self = cls(buffer, resource)
        with buffer.read_from_offset(offset):
            for _ in range(count):
                ref = ResourceExternalReference.from_buffer(buffer)
                self._mapping[ref.hash] = ref
                self.append(ref)

        return self

    def find_resource(self, resource_id: int):
        if res := self._mapping.get(resource_id & 0xFFFF_FFFF, None):
            return res.name
