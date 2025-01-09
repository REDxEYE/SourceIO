from dataclasses import dataclass

from SourceIO.library.utils import Buffer
from SourceIO.library.source2.keyvalues3.types import Object
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

    @classmethod
    def from_vkv3(cls, vkv: Object) -> 'Dependency':
        return cls(vkv['m_RelativeFilename'], vkv['m_SearchPath'], vkv['m_nFileCRC'], 0)


class InputDependencies(DependencyList[InputDependency]):
    dependency_type = InputDependency


class AdditionalInputDependencies(DependencyList[InputDependency]):
    dependency_type = InputDependency
