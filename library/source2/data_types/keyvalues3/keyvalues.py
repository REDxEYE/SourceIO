import abc
from typing import Dict

from SourceIO.library.source2.data_types.keyvalues3.enums import KV3Formats, KV3Encodings
from SourceIO.library.utils import Buffer


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
