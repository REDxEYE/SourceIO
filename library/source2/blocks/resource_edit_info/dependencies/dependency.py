import abc
from abc import ABC
from typing import TypeVar

from SourceIO.library.utils import Buffer
from SourceIO.library.source2.keyvalues3.types import Object, TypedArray


class Dependency(ABC):
    @classmethod
    @abc.abstractmethod
    def from_buffer(cls, buffer: Buffer) -> 'Dependency':
        pass

    @classmethod
    @abc.abstractmethod
    def from_vkv3(cls, vkv: Object) -> 'Dependency':
        pass


DependencyT = TypeVar('DependencyT', bound=Dependency)


class DependencyList(list[DependencyT]):
    dependency_type: DependencyT = Dependency

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'DependencyList':
        self = cls()
        offset = buffer.read_relative_offset32()
        size = buffer.read_uint32()
        with buffer.read_from_offset(offset):
            for _ in range(size):
                self.append(cls.dependency_type.from_buffer(buffer))

        return self

    @classmethod
    def from_vkv3(cls, vkv: TypedArray[Object]):
        self = cls()
        for dependency in vkv:
            self.append(cls.dependency_type.from_vkv3(dependency))
        return self
