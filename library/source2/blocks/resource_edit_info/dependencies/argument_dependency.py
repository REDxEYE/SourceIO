from dataclasses import dataclass
from typing import Union

from SourceIO.library.utils import Buffer
from SourceIO.library.source2.keyvalues3.types import Object
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

    @classmethod
    def from_vkv3(cls, vkv: Object) -> 'Dependency':
        return cls(vkv['m_ParameterName'], vkv['m_ParameterType'], vkv['m_nFingerprint'], vkv['m_nFingerprintDefault'])


class ArgumentDependencies(DependencyList[ArgumentDependency]):
    dependency_type = ArgumentDependency
