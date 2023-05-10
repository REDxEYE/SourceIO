import abc
from typing import Dict

from .enums import KV3Formats, KV3Encodings
from ....utils import Buffer


class KeyValues(abc.ABC):
    root: Dict
    encoding: KV3Encodings
    format: KV3Formats

    @classmethod
    @abc.abstractmethod
    def from_buffer(cls, buffer: Buffer):
        ...

    @abc.abstractmethod
    def to_file(self, buffer: Buffer, **kwargs):
        ...
