from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.source1.bsp import Lump, LumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.source1.bsp.datatypes.detail_prop_lump import DetailPropLump
from SourceIO.library.source1.bsp.datatypes.game_lump_header import (GameLumpHeader, VindictusGameLumpHeader,
                                                                     DMGameLumpHeader)
from SourceIO.library.source1.bsp.datatypes.static_prop_lump import StaticPropLump
from SourceIO.library.utils import Buffer


@lump_tag(35, 'LUMP_GAME_LUMP')
class GameLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.lump_count = 0
        self.game_lumps_info: list[GameLumpHeader] = []
        self.game_lumps = {}

    def parse(self, buffer: Buffer, bsp: BSPFile):
        self.lump_count = buffer.read_uint32()
        for _ in range(self.lump_count):
            lump = GameLumpHeader.from_buffer(buffer, bsp)
            if not lump.id:
                continue
            self.game_lumps_info.append(lump)
        for lump in self.game_lumps_info:
            relative_offset = lump.offset - self._info.offset
            print(f'GLump "{lump.id}" offset: {relative_offset} size: {lump.size} ')
            with buffer.save_current_offset():
                buffer.seek(relative_offset)
                if lump.flags == 1:
                    curr_index = self.game_lumps_info.index(lump)
                    if curr_index + 1 != len(self.game_lumps_info):
                        next_offset = self.game_lumps_info[curr_index + 1].offset - self._info.offset
                    else:
                        next_offset = self._info.size
                    compressed_size = next_offset - relative_offset
                    game_lump_buffer = Lump.decompress_lump(buffer.slice(size=compressed_size))
                else:
                    game_lump_buffer = buffer.slice(size=lump.size)

                pass  # TODO
            if lump.id == 'sprp':
                game_lump = StaticPropLump(lump)
                game_lump.parse(game_lump_buffer, bsp)
                self.game_lumps[lump.id] = game_lump
            elif lump.id == 'dprp':
                detail_lump = DetailPropLump(lump)
                detail_lump.parse(game_lump_buffer, bsp)
                self.game_lumps[lump.id] = detail_lump

        return self


@lump_tag(35, 'LUMP_GAME_LUMP', bsp_version=(21, 0))
class GameLump21(GameLump):
    pass


@lump_tag(35, 'LUMP_GAME_LUMP', bsp_version=(20, 4))
class GameLump204(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.lump_count = 0
        self.game_lumps_info: list[GameLumpHeader] = []
        self.game_lumps = {}

    def parse(self, buffer: Buffer, bsp: BSPFile):
        self.lump_count = buffer.read_uint32()
        for _ in range(self.lump_count):
            lump = DMGameLumpHeader.from_buffer(buffer, bsp)
            if not lump.id:
                continue
            self.game_lumps_info.append(lump)
        for lump in self.game_lumps_info:
            relative_offset = lump.offset - self._info.offset
            print(f'GLump "{lump.id}" offset: {relative_offset} size: {lump.size} ')
            with buffer.save_current_offset():
                buffer.seek(relative_offset)
                if lump.flags == 1:
                    curr_index = self.game_lumps_info.index(lump)
                    if curr_index + 1 != len(self.game_lumps_info):
                        next_offset = self.game_lumps_info[curr_index + 1].offset - self._info.offset
                    else:
                        next_offset = self._info.size
                    compressed_size = next_offset - relative_offset
                    game_lump_buffer = Lump.decompress_lump(buffer.slice(size=compressed_size))
                else:
                    game_lump_buffer = buffer.slice(size=lump.size)

                pass  # TODO
            if lump.id == 'sprp':
                game_lump = StaticPropLump(lump)
                game_lump.parse(game_lump_buffer, bsp)
                self.game_lumps[lump.id] = game_lump
            elif lump.id == 'dprp':
                detail_lump = DetailPropLump(lump)
                detail_lump.parse(game_lump_buffer, bsp)
                self.game_lumps[lump.id] = detail_lump

        return self


@lump_tag(35, 'LUMP_GAME_LUMP', steam_id=SteamAppId.VINDICTUS)
class VGameLump(Lump):
    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.lump_count = 0
        self.game_lumps_info: list[GameLumpHeader] = []
        self.game_lumps = {}

    def parse(self, buffer: Buffer, bsp: BSPFile):
        self.lump_count = buffer.read_uint32()
        for _ in range(self.lump_count):
            lump = VindictusGameLumpHeader.from_buffer(buffer, bsp)
            if not lump.id:
                continue
            self.game_lumps_info.append(lump)
        for lump in self.game_lumps_info:
            relative_offset = lump.offset - self._info.offset
            print(f'GLump "{lump.id}" offset: {relative_offset} size: {lump.size} ')
            with buffer.save_current_offset():
                buffer.seek(relative_offset)
                if lump.flags == 1:
                    curr_index = self.game_lumps_info.index(lump)
                    if curr_index + 1 != len(self.game_lumps_info):
                        next_offset = self.game_lumps_info[curr_index + 1].offset - self._info.offset
                    else:
                        next_offset = self._info.size
                    compressed_size = next_offset - relative_offset
                    game_lump_buffer = Lump.decompress_lump(buffer.slice(size=compressed_size))
                else:
                    game_lump_buffer = buffer.slice(size=lump.size)

                pass  # TODO
            if lump.id == 'sprp':
                game_lump = StaticPropLump(lump)
                game_lump.parse(game_lump_buffer, bsp)
                self.game_lumps[lump.id] = game_lump
            elif lump.id == 'dprp':
                detail_lump = DetailPropLump(lump)
                detail_lump.parse(game_lump_buffer, bsp)
                self.game_lumps[lump.id] = detail_lump

        return self
