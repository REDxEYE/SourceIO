from dataclasses import dataclass

from SourceIO.library.utils import Buffer
from .base import BaseBlock


@dataclass(slots=True)
class DummyBlock(BaseBlock):
    buffer: Buffer

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(buffer)
