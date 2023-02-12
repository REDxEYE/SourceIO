from dataclasses import dataclass

from ....utils import Buffer


@dataclass(slots=True)
class StudioFrameAnim:
    constant_offset: int
    frame_offset: int
    frame_length: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        constant_offset, frame_offset, frame_length = buffer.read_fmt("3i")
        buffer.skip(12)
        return cls(constant_offset, frame_offset, frame_length)
