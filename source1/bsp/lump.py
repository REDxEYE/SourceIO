import lzma
from enum import IntEnum

from lzma import decompress as lzma_decompress

from typing import List

from ...utilities.byte_io_mdl import ByteIO
from ...utilities.math_utilities import sizeof_fmt


class LumpTag:
    def __init__(self, lump_id, lump_name, bsp_version=None, steam_id=None):
        self.lump_id = lump_id
        self.lump_name = lump_name
        self.bsp_version = bsp_version
        self.steam_id = steam_id


def lump_tag(lump_id, lump_name, bsp_version=None, steam_id=None):
    def loader(klass) -> object:
        if not klass.tags:
            klass.tags = []
        klass.tags.append(LumpTag(lump_id, lump_name, bsp_version, steam_id))
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

        base_path = self._bsp.filepath.parent
        lump_path = base_path / f'{self._bsp.filepath.name}.{lump_id:04x}.bsp_lump'
        if lump_path.exists():
            reader = ByteIO(lump_path)
        else:
            self._bsp.reader.seek(self._lump.offset)
            reader = self._bsp.reader
        if self._lump.compressed:

            lzma_id = reader.read_fourcc()
            assert lzma_id == "LZMA", f"Unknown compressed header({lzma_id})"
            decompressed_size = reader.read_uint32()
            compressed_size = reader.read_uint32()
            prob_byte = reader.read_int8()
            dict_size = reader.read_uint32()

            lc = prob_byte % 9
            props = int(prob_byte / 9)
            pb = int(props / 5)
            lp = props % 5
            my_filters = [{"id": lzma.FILTER_LZMA2, "dict_size": dict_size, "pb": pb, "lp": lp, "lc": lc}, ]
            self.reader = ByteIO(
                lzma_decompress(reader.read(compressed_size), lzma.FORMAT_RAW, filters=my_filters)
            )
            assert self.reader.size() == decompressed_size, 'Compressed lump size does not match expected'
        else:
            self.reader = ByteIO(reader.read(self._lump.size))

        with self.reader.save_current_pos():
            if not lump_path.exists():
                with lump_path.open('wb') as f:
                    f.write(self.reader.read(-1))

    def parse(self):
        return self
