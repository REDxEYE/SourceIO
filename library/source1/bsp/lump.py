import lzma
from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING, Optional, Type

from ...shared.app_id import SteamAppId
from ...utils.file_utils import IBuffer, FileBuffer, MemoryBuffer
from ...utils.math_utilities import sizeof_fmt

if TYPE_CHECKING:
    from .bsp_file import BSPFile


@dataclass(slots=True, frozen=True)
class LumpTag:
    lump_id: int
    lump_name: str
    lump_version: Optional[int] = field(default=None)
    bsp_version: Optional[int] = field(default=None)
    steam_id: Optional[SteamAppId] = field(default=None)


def lump_tag(lump_id, lump_name,
             lump_version: Optional[int] = None,
             bsp_version: Optional[int] = None,
             steam_id: Optional[SteamAppId] = None):
    def loader(klass: Type[Lump]) -> Type[Lump]:
        if not klass.tags:
            klass.tags = []
        klass.tags.append(LumpTag(lump_id, lump_name, lump_version, bsp_version, steam_id))
        return klass

    return loader


@dataclass(slots=True)
class LumpInfo:
    id: int = field(init=False, default=-1)
    offset: int
    size: int
    version: int
    decompressed_size: int

    @property
    def compressed(self):
        return self.decompressed_size != 0

    @classmethod
    def from_buffer(cls, buffer: IBuffer, is_l4d2: bool = False):
        if is_l4d2:
            version = buffer.read_int32()
            offset = buffer.read_int32()
            size = buffer.read_int32()
            decompressed_size = buffer.read_uint32()
        else:
            offset = buffer.read_int32()
            size = buffer.read_int32()
            version = buffer.read_int32()
            decompressed_size = buffer.read_uint32()
        return cls(offset, size, version, decompressed_size)

    def __repr__(self):
        return f"<Lump{self.id}({self.id:04x}) o: {self.offset} s: {sizeof_fmt(self.size)}({self.size} bytes)>"


class Lump:
    tags: List[LumpTag] = []

    @classmethod
    def all_subclasses(cls):
        return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in c.all_subclasses()])

    def __init__(self, lump_info: LumpInfo):
        self._info = lump_info

    def parse(self, buffer: IBuffer, bsp: 'BSPFile'):
        return self

    # noinspection PyUnresolvedReferences,PyProtectedMember
    @staticmethod
    def decompress_lump(buffer: IBuffer) -> IBuffer:
        magic = buffer.read_fourcc()
        assert magic == 'LZMA', f'Invalid LZMA compressed header: {magic}'

        decompressed_size = buffer.read_uint32()
        compressed_size = buffer.read_uint32()
        filter_properties = lzma._decode_filter_properties(lzma.FILTER_LZMA1, buffer.read(5))

        compressed_buffer = buffer.read(compressed_size)
        chunks: List[bytes] = []

        while True:
            decompressor = lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=(filter_properties,))
            try:
                result = decompressor.decompress(compressed_buffer)
            except lzma.LZMAError:
                if not chunks:
                    raise  # Error on the first iteration; bail out.
                break  # Leftover data is not a valid LZMA/XZ stream; ignore it.
            chunks.append(result)
            compressed_buffer = decompressor.unused_data
            if not compressed_buffer:
                break
            assert decompressor.eof, 'Compressed data ended before the end-of-stream marker was reached'

        decompressed_buffer = b"".join(chunks)[:decompressed_size]
        assert decompressed_size == len(decompressed_buffer), 'Decompressed data does not match the expected size'
        return MemoryBuffer(decompressed_buffer)
