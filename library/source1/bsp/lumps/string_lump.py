from dataclasses import dataclass

import numpy as np

from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.source1.bsp import Lump, LumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.utils import Buffer


@lump_tag(44, 'LUMP_TEXDATA_STRING_DATA')
class StringOffsetLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.string_ids = np.array([])

    def parse(self, buffer: Buffer, bsp: BSPFile):
        self.string_ids = np.frombuffer(buffer.read(), np.int32)
        return self


@lump_tag(43, 'LUMP_TEXDATA_STRING_TABLE')
class StringsLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.strings = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        data = buffer.read(-1)
        self.strings = list(map(lambda a: a.decode("latin1"), data.split(b'\x00')))
        return self


@dataclass(slots=True)
class Shader:
    name: str
    surface_flags: int
    content_flags: int


@lump_tag(1, 'LUMP_SHADERS', steam_id=SteamAppId.SOLDIERS_OF_FORTUNE2, bsp_version=(1, 0))
class ShadersLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.shaders: list[Shader] = []

    def parse(self, buffer: Buffer, bsp: BSPFile):
        while buffer:
            self.shaders.append(Shader(buffer.read_ascii_string(64), buffer.read_uint32(), buffer.read_uint32()))
        return self
