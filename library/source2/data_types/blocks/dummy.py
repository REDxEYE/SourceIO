from dataclasses import dataclass

from ....utils import Buffer
from .base import BaseBlock


@dataclass(slots=True)
class DummyBlock(BaseBlock):
    buffer: Buffer

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(buffer)
