import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ....utils import Buffer

if TYPE_CHECKING:
    from ...resource_types.resource import CompiledResource


class BaseBlock:
    custom_name: Optional[str] = None

    def __init__(self, buffer: Buffer, resource: 'CompiledResource'):
        self._buffer = buffer
        self._resource = resource
        pass

    @classmethod
    @abc.abstractmethod
    def from_buffer(cls, buffer: Buffer, resource: 'CompiledResource') -> 'BaseBlock':
        raise NotImplementedError()
