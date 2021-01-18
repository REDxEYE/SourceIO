from typing import List

from ..datatypes.gamelumps.detail_prop_lump import DetailPropLump
from ..datatypes.gamelumps.static_prop_lump import StaticPropLump
from ....utilities.byte_io_mdl import ByteIO
from .. import Lump, LumpTypes
from ..datatypes.game_lump_header import GameLumpHeader


class GameLump(Lump):
    lump_id = LumpTypes.LUMP_GAME_LUMP

    def __init__(self, bsp):
        super().__init__(bsp)
        self.lump_count = 0
        self.game_lumps_info: List[GameLumpHeader] = []
        self.game_lumps = {}

    def parse(self):
        reader = self.reader
        self.lump_count = reader.read_uint32()
        for _ in range(self.lump_count):
            self.game_lumps_info.append(GameLumpHeader(self, self._bsp).parse(reader))
        for lump in self.game_lumps_info:
            if lump.size == 0:
                continue
            relative_offset = lump.offset - self._lump.offset
            print(f'GLump "{lump.id}" offset: {relative_offset} size: {lump.size} ')
            with reader.save_current_pos():
                reader.seek(relative_offset)
                game_lump_reader = ByteIO(reader.read(lump.size))
            if lump.id == 'sprp':
                game_lump = StaticPropLump(lump)
                game_lump.parse(game_lump_reader)
                self.game_lumps[lump.id] = game_lump
            elif lump.id == 'dprp':
                detail_lump = DetailPropLump(lump)
                detail_lump.parse(game_lump_reader)
                self.game_lumps[lump.id] = detail_lump

        return self
