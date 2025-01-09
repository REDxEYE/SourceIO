from dataclasses import dataclass

from SourceIO.library.utils import Buffer
from SourceIO.library.source2.keyvalues3.types import String
from .dependency import Dependency, DependencyList


@dataclass(slots=True)
class ChildResource(Dependency):
    id: int
    name: str
    unk: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(buffer.read_uint64(), buffer.read_source2_string(), buffer.read_uint32())

    @classmethod
    def from_vkv3(cls, vkv: String) -> 'Dependency':
        return cls(-1, vkv, -1)


class ChildResources(DependencyList[ChildResource]):
    dependency_type = ChildResource
