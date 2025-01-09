from dataclasses import dataclass

from SourceIO.library.utils import Buffer
from SourceIO.library.source2.keyvalues3.types import Object
from .dependency import Dependency, DependencyList


@dataclass(slots=True)
class SpecialDependency(Dependency):
    string: str
    compiler_id: str
    fingerprint: int
    user_data: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        rel_name = buffer.read_source2_string()
        search_path = buffer.read_source2_string()
        return cls(rel_name, search_path, *buffer.read_fmt('2I'))

    @classmethod
    def from_vkv3(cls, vkv: Object) -> 'Dependency':
        return cls(vkv['m_String'], vkv['m_CompilerIdentifier'], vkv['m_nFingerprint'], vkv['m_nUserData'])


class SpecialDependencies(DependencyList[SpecialDependency]):
    dependency_type = SpecialDependency
