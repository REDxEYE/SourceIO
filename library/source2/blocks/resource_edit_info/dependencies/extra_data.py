from dataclasses import dataclass

from SourceIO.library.utils import Buffer
from SourceIO.library.source2.keyvalues3.types import Object
from .dependency import Dependency, DependencyList


@dataclass(slots=True)
class ExtraData(Dependency):
    name: str

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'Dependency':
        raise NotImplementedError('Unsupported, if found please report to ValveResourceFormat repo and to SourceIO2')

    @classmethod
    def from_vkv3(cls, vkv: Object) -> 'Dependency':
        raise NotImplementedError('Unsupported, if found please report to ValveResourceFormat repo and to SourceIO2')


@dataclass(slots=True)
class ExtraIntData(ExtraData):
    value: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(buffer.read_source2_string(), buffer.read_int32())


@dataclass(slots=True)
class ExtraFloatData(ExtraData):
    value: float

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(buffer.read_source2_string(), buffer.read_float())


@dataclass(slots=True)
class ExtraStringData(ExtraData):
    value: str

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(buffer.read_source2_string(), buffer.read_source2_string())


class ExtraInts(DependencyList[ExtraIntData]):
    dependency_type = ExtraIntData


class ExtraFloats(DependencyList[ExtraFloatData]):
    dependency_type = ExtraFloatData


class ExtraStrings(DependencyList[ExtraStringData]):
    dependency_type = ExtraStringData
