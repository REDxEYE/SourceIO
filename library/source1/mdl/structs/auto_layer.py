from dataclasses import dataclass

from ....utils import Buffer


@dataclass(slots=True)
class AutoLayer:
    sequence_id: int
    pose_id: int
    flags: int
    start: float
    peak: float
    tail: float
    end: float

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(buffer.read_int32(), buffer.read_int32(), buffer.read_int32(), *buffer.read_fmt('4f'))
