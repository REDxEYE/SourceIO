import lzma
from typing import List

from ...shared.content_providers.content_manager import ContentManager
from ...utils.byte_io_mdl import ByteIO
from ...utils.math_utilities import sizeof_fmt


class LumpTag:
    def __init__(self, lump_id, lump_name, lump_version=None, bsp_version=None, steam_id=None):
        self.lump_id = lump_id
        self.lump_name = lump_name
        self.lump_version = lump_version
        self.bsp_version = bsp_version
        self.steam_id = steam_id


def lump_tag(lump_id, lump_name, lump_version=None, bsp_version=None, steam_id=None):
    def loader(klass) -> object:
        if not klass.tags:
            klass.tags = []
        klass.tags.append(LumpTag(lump_id, lump_name, lump_version, bsp_version, steam_id))
        return klass

    return loader


class LumpInfo:
    def __init__(self, lump_id):
        self.id = lump_id
        self.offset = 0
        self.size = 0
        self.version = 0
        self.magic = 0

    @property
    def compressed(self):
        return self.magic != 0

    def parse(self, reader: ByteIO, is_l4d2):
        if is_l4d2:
            self.version = reader.read_int32()
            self.offset = reader.read_int32()
            self.size = reader.read_int32()
            self.magic = reader.read_uint32()
        else:
            self.offset = reader.read_int32()
            self.size = reader.read_int32()
            self.version = reader.read_int32()
            self.magic = reader.read_uint32()

    def __repr__(self):
        return f"<{self.id}({self.id:04x}) o: {self.offset} s: {sizeof_fmt(self.size)}({self.size} bytes)>"


class Lump:
    tags: List[LumpTag] = []

    @classmethod
    def all_subclasses(cls):
        return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in c.all_subclasses()])

    def __init__(self, bsp, lump_id):
        from .bsp_file import BSPFile
        self._bsp: BSPFile = bsp
        self._lump: LumpInfo = bsp.lumps_info[lump_id]
        self.reader = ByteIO()

        if ContentManager()._titanfall_mode:
            base_path = self._bsp.filepath.parent
            lump_path = base_path / f'{self._bsp.filepath.name}.{lump_id:04x}.bsp_lump'

            if lump_path.exists():
                self.reader = ByteIO(lump_path)
                return

        reader = self._bsp.reader
        reader.seek(self._lump.offset)

        if not self._lump.compressed:
            self.reader = ByteIO(reader.read(self._lump.size))
        else:
            self.reader = Lump.decompress_lump(reader)

    def parse(self):
        return self

    # noinspection PyUnresolvedReferences,PyProtectedMember
    @staticmethod
    def decompress_lump(reader: ByteIO) -> ByteIO:
        magic = reader.read_fourcc()
        assert magic == 'LZMA', f'Invalid LZMA compressed header: {magic}'

        decompressed_size = reader.read_uint32()
        compressed_size = reader.read_uint32()
        filter_properties = lzma._decode_filter_properties(lzma.FILTER_LZMA1, reader.read(5))

        compressed_buffer = reader.read(compressed_size)
        decompressed_buffer = bytearray()

        while True:
            decompressor = lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=(filter_properties,))
            try:
                result = decompressor.decompress(compressed_buffer)
            except lzma.LZMAError:
                if not decompressed_buffer:
                    raise  # Error on the first iteration; bail out.
                break  # Leftover data is not a valid LZMA/XZ stream; ignore it.
            decompressed_buffer.extend(result)
            compressed_buffer = decompressor.unused_data
            if not compressed_buffer:
                break
            assert decompressor.eof, 'Compressed data ended before the end-of-stream marker was reached'
        decompressed_buffer = decompressed_buffer[:decompressed_size]
        assert decompressed_size == len(decompressed_buffer), 'Decompressed data does not match the expected size'
        return ByteIO(decompressed_buffer)
