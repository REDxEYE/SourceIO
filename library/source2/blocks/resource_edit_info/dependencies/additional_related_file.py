from dataclasses import dataclass

from SourceIO.library.utils import Buffer
from SourceIO.library.source2.keyvalues3.types import Object, String
from SourceIO.library.utils.file_utils import Label
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

    def to_buffer(self, buffer: Buffer) -> list[tuple[str, Label]]:
        sal = [
            (self.relative_filename, buffer.new_label("relative_filename", 4, None)),
            (self.search_path, buffer.new_label("search_path", 4, None))
        ]
        return sal

    @classmethod
    def from_vkv3(cls, vkv: Object) -> 'Dependency':
        return cls(vkv["m_RelativeFilename"], vkv["m_SearchPath"])

    def to_vkv3(self) -> Object:
        return Object({
            "m_RelativeFilename": String(self.relative_filename),
            "m_SearchPath": String(self.search_path)
        })

class AdditionalRelatedFiles(DependencyList[AdditionalRelatedFile]):
    dependency_type = AdditionalRelatedFile
