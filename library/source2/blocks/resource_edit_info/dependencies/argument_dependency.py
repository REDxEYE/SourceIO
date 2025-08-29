from dataclasses import dataclass
from typing import Union

from SourceIO.library.utils import Buffer
from SourceIO.library.source2.keyvalues3.types import Object, String, UInt32, Float
from SourceIO.library.utils.file_utils import Label
from .dependency import Dependency, DependencyList


@dataclass(slots=True)
class ArgumentDependency(Dependency):
    name: str
    type: str
    fingerprint: Union[int, float]
    fingerprint_default: Union[int, float]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        name = buffer.read_source2_string()
        data_type = buffer.read_source2_string()
        if data_type == 'FloatArg':
            data = buffer.read_fmt('2f')
        else:
            data = buffer.read_fmt('2I')
        return cls(name, data_type, *data)

    def to_buffer(self, buffer: Buffer) -> list[tuple[str, Label]]:
        sal = [
            (self.name, buffer.new_label("name", 4, None)),
            (self.type, buffer.new_label("type", 4, None))
        ]
        if self.type == 'FloatArg':
            buffer.write_fmt('2f', self.fingerprint, self.fingerprint_default)
        else:
            buffer.write_fmt('2I', self.fingerprint, self.fingerprint_default)
        return sal

    @classmethod
    def from_vkv3(cls, vkv: Object) -> 'Dependency':
        return cls(vkv['m_ParameterName'], vkv['m_ParameterType'], vkv['m_nFingerprint'], vkv['m_nFingerprintDefault'])

    def to_vkv3(self) -> Object:
        return Object({
            'm_ParameterName': String(self.name),
            'm_ParameterType': String(self.type),
            'm_nFingerprint': UInt32(self.fingerprint) if isinstance(self.fingerprint, int) else Float(self.fingerprint),
            'm_nFingerprintDefault': UInt32(self.fingerprint_default) if isinstance(self.fingerprint_default, int) else Float(self.fingerprint_default)
        })


class ArgumentDependencies(DependencyList[ArgumentDependency]):
    dependency_type = ArgumentDependency
