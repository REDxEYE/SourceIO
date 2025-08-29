from dataclasses import dataclass

from SourceIO.library.utils import Buffer
from SourceIO.library.source2.keyvalues3.types import Object, AnyKVType, UInt32, Float, String
from SourceIO.library.utils.file_utils import Label
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

    def to_vkv3(self) -> AnyKVType:
        raise NotImplementedError('Unsupported, if found please report to ValveResourceFormat repo and to SourceIO2')

@dataclass(slots=True)
class ExtraIntData(ExtraData):
    value: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(buffer.read_source2_string(), buffer.read_int32())

    def to_buffer(self, buffer: Buffer) -> list[tuple[str, Label]]:
        sal = [
            (self.name, buffer.new_label("name", 4, None))
        ]
        buffer.write_fmt('i', self.value)
        return sal

    def to_vkv3(self) -> AnyKVType:
        return UInt32(self.value)

@dataclass(slots=True)
class ExtraFloatData(ExtraData):
    value: float

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(buffer.read_source2_string(), buffer.read_float())

    def to_buffer(self, buffer: Buffer) -> list[tuple[str, Label]]:
        sal = [
            (self.name, buffer.new_label("name", 4, None))
        ]
        buffer.write_fmt('f', self.value)
        return sal

    def to_vkv3(self) -> AnyKVType:
        return Float(self.value)

@dataclass(slots=True)
class ExtraStringData(ExtraData):
    value: str

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(buffer.read_source2_string(), buffer.read_source2_string())

    def to_buffer(self, buffer: Buffer) -> list[tuple[str, Label]]:
        sal = [
            (self.name, buffer.new_label("name", 4, None)),
            (self.value, buffer.new_label("value", 4, None))
        ]
        return sal

    def to_vkv3(self) -> AnyKVType:
        return String(self.value)

class ExtraInts(DependencyList[ExtraIntData]):
    dependency_type = ExtraIntData


class ExtraFloats(DependencyList[ExtraFloatData]):
    dependency_type = ExtraFloatData


class ExtraStrings(DependencyList[ExtraStringData]):
    dependency_type = ExtraStringData
