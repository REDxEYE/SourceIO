import abc
from abc import ABC
from typing import TypeVar

from SourceIO.library.source2.keyvalues3.enums import KV3Type, Specifier
from SourceIO.library.utils import Buffer, WritableMemoryBuffer
from SourceIO.library.source2.keyvalues3.types import Object, TypedArray, AnyKVType
from SourceIO.library.utils.file_utils import Label


class Dependency(ABC):
    @classmethod
    @abc.abstractmethod
    def from_buffer(cls, buffer: Buffer) -> 'Dependency':
        pass

    @abc.abstractmethod
    def to_buffer(self, buffer: Buffer) -> list[tuple[str, Label]]:
        pass

    @classmethod
    @abc.abstractmethod
    def from_vkv3(cls, vkv: Object) -> 'Dependency':
        pass

    @abc.abstractmethod
    def to_vkv3(self) -> AnyKVType:
        pass


DependencyT = TypeVar('DependencyT', bound=Dependency)


class DependencyList(list[DependencyT]):
    dependency_type: Dependency | DependencyT = Dependency

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'DependencyList':
        self = cls()
        offset = buffer.read_relative_offset32()
        count = buffer.read_uint32()
        with buffer.read_from_offset(offset):
            for _ in range(count):
                self.append(cls.dependency_type.from_buffer(buffer))
        return self

    def data_to_buffer(self) -> WritableMemoryBuffer:
        buffer = WritableMemoryBuffer()
        strings_and_labels = []
        for dependency in self:
            strings_and_labels.extend(dependency.to_buffer(buffer))
        for string, label in strings_and_labels:
            label.write('I', buffer.tell() - label.offset)
            buffer.write_ascii_string(string, True)
        return buffer

    @classmethod
    def from_vkv3(cls, vkv: TypedArray[Object]):
        self = cls()
        for dependency in vkv:
            self.append(cls.dependency_type.from_vkv3(dependency))
        return self

    def to_vkv3(self):
        lst = TypedArray(KV3Type.OBJECT, Specifier.UNSPECIFIED, [])
        for dependency in self:
            lst.append(dependency.to_vkv3())
        return lst
