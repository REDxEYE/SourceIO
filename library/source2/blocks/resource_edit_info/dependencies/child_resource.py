from dataclasses import dataclass

from SourceIO.library.utils import Buffer
from SourceIO.library.source2.keyvalues3.types import String, Object
from SourceIO.library.utils.file_utils import Label
from .dependency import Dependency, DependencyList


@dataclass(slots=True)
class ChildResource(Dependency):
    id: int
    name: str
    unk: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(buffer.read_uint64(), buffer.read_source2_string(), buffer.read_uint32())

    def to_buffer(self, buffer: Buffer) -> list[tuple[str, Label]]:
        sal = []
        buffer.write_uint64(self.id)
        sal.append((self.name, buffer.new_label(f"child_resource_name_{self.name}", 4, None)))
        buffer.write_uint32(self.unk)
        return sal

    @classmethod
    def from_vkv3(cls, vkv: String) -> 'Dependency':
        return cls(-1, vkv, -1)

    def to_vkv3(self) -> String:
        return String(self.name)

class ChildResources(DependencyList[ChildResource]):
    dependency_type = ChildResource
