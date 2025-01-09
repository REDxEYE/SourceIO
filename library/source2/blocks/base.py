import abc
from typing import Optional

from SourceIO.library.utils import Buffer


class BaseBlock:
    custom_name: Optional[str] = None

    def __init__(self, buffer: Buffer):
        self._buffer = buffer

    @classmethod
    @abc.abstractmethod
    def from_buffer(cls, buffer: Buffer) -> 'BaseBlock':
        raise NotImplementedError()
