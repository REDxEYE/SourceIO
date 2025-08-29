from dataclasses import dataclass

from SourceIO.library.utils import Buffer
from SourceIO.library.source2.keyvalues3.types import Object, String, UInt32
from SourceIO.library.utils.file_utils import Label
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

    def to_buffer(self, buffer: Buffer) -> list[tuple[str, Label]]:
        sal = [
            (self.string, buffer.new_label("string", 8, None)),
            (self.compiler_id, buffer.new_label("compiler_id", 8, None))
        ]
        buffer.write_fmt('2I', self.fingerprint, self.user_data)
        return sal

    @classmethod
    def from_vkv3(cls, vkv: Object) -> 'Dependency':
        return cls(vkv['m_String'], vkv['m_CompilerIdentifier'], vkv['m_nFingerprint'], vkv['m_nUserData'])

    def to_vkv3(self) -> Object:
        return Object({
            'm_String': String(self.string),
            'm_CompilerIdentifier': String(self.compiler_id),
            'm_nFingerprint': UInt32(self.fingerprint),
            'm_nUserData': UInt32(self.user_data)
        })


class SpecialDependencies(DependencyList[SpecialDependency]):
    dependency_type = SpecialDependency
