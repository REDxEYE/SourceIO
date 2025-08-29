from dataclasses import dataclass

from SourceIO.library.utils import Buffer
from SourceIO.library.source2.keyvalues3.types import Object, String, UInt32
from SourceIO.library.utils.file_utils import Label
from .dependency import Dependency, DependencyList


@dataclass(slots=True)
class InputDependency(Dependency):
    relative_name: str
    search_path: str
    file_crc: int
    flags: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        rel_name = buffer.read_source2_string()
        search_path = buffer.read_source2_string()
        return cls(rel_name, search_path, *buffer.read_fmt('2I'))

    def to_buffer(self, buffer: Buffer) -> list[tuple[str, Label]]:
        sal = [
            (self.relative_name, buffer.new_label("relative_name", 4, None)),
            (self.search_path, buffer.new_label("search_path", 4, None))
        ]
        buffer.write_fmt('2I', self.file_crc, self.flags)
        return sal

    @classmethod
    def from_vkv3(cls, vkv: Object) -> 'Dependency':
        return cls(vkv['m_RelativeFilename'], vkv['m_SearchPath'], vkv['m_nFileCRC'], 0)

    def to_vkv3(self):
        return Object({
            "m_RelativeFilename": String(self.relative_name),
            "m_SearchPath": String(self.search_path),
            "m_nFileCRC": UInt32(self.file_crc),
        })


class InputDependencies(DependencyList[InputDependency]):
    dependency_type = InputDependency


class AdditionalInputDependencies(DependencyList[InputDependency]):
    dependency_type = InputDependency
