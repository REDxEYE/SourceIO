from dataclasses import dataclass

from SourceIO.library.utils import Buffer


@dataclass(slots=True)
class StudioEvent:
    frame_index: int
    event_index: int
    event_type: int
    options: str

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        frame_index = buffer.read_uint32()
        event_index = buffer.read_uint32()
        event_type = buffer.read_uint32()
        options = buffer.read_ascii_string(64)
        return cls(frame_index, event_index, event_type, options)
