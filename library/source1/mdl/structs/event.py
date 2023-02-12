from dataclasses import dataclass

from ....utils import Buffer


@dataclass(slots=True)
class Event:
    cycle: float
    event: int
    type: int
    options: str
    name: str

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        start_offset = buffer.tell()
        cycle = buffer.read_float()
        event = buffer.read_int32()
        event_type = buffer.read_int32()
        options = buffer.read_ascii_string(64)
        name = buffer.read_source1_string(start_offset)
        return cls(cycle, event, event_type, options, name)
