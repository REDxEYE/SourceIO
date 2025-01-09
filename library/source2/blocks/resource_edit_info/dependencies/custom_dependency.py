from dataclasses import dataclass

from SourceIO.library.utils import Buffer
from SourceIO.library.source2.keyvalues3.types import Object
from .dependency import Dependency, DependencyList


@dataclass(slots=True)
class CustomDependency(Dependency):
    @classmethod
    def from_vkv3(cls, vkv: Object) -> 'Dependency':
        raise NotImplementedError('Unsupported, if found please report to ValveResourceFormat repo and to SourceIO2')

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        raise NotImplementedError('Unsupported, if found please report to ValveResourceFormat repo and to SourceIO2')


class CustomDependencies(DependencyList[CustomDependency]):
    dependency_type = CustomDependency
