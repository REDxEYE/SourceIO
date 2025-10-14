from dataclasses import dataclass

from SourceIO.library.source1.bsp.bsp_file import VBSPFile
from SourceIO.library.utils.file_utils import Buffer


@dataclass(slots=True)
class LightmapHeader:
    count: int
    width: int
    height: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: VBSPFile):
        return cls(*buffer.read_fmt('I2H'))
