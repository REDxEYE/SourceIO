from dataclasses import dataclass

from SourceIO.library.utils import Buffer
from SourceIO.library.source2.keyvalues3.types import Object
from .dependency import Dependency, DependencyList


@dataclass(slots=True)
class AdditionalRelatedFile(Dependency):
    relative_filename: str
    search_path: str

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        rel_name = buffer.read_source2_string()
        search_path = buffer.read_source2_string()
        return cls(rel_name, search_path)

    @classmethod
    def from_vkv3(cls, vkv: Object) -> 'Dependency':
        return cls(vkv["m_RelativeFilename"], vkv["m_SearchPath"])


class AdditionalRelatedFiles(DependencyList[AdditionalRelatedFile]):
    dependency_type = AdditionalRelatedFile
