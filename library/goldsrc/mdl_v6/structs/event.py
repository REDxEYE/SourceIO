from dataclasses import dataclass

from ....utils import Buffer


@dataclass(slots=True)
class StudioPivot:
    frame_index: int
    event_type: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(*buffer.read_fmt('2H'))
