from enum import IntFlag
from typing import TYPE_CHECKING, List, Tuple

from ......logger import SLoggingManager
from .....shared.app_id import SteamAppId
from .....shared.content_providers.content_manager import ContentManager
from .....utils.file_utils import Buffer

if TYPE_CHECKING:
    from ...bsp_file import BSPFile

log_manager = SLoggingManager()
logger = log_manager.get_logger('StaticPropLump')


class StaticPropFlag(IntFlag):
    NONE = 0x0
    FLAG_FADES = 0x1
    USE_LIGHTING_ORIGIN = 0x2
    NO_DRAW = 0x4
    IGNORE_NORMALS = 0x8
    NO_SHADOW = 0x10
    SCREEN_SPACE_FADE = 0x20
    NO_PER_VERTEX_LIGHTING = 0x40
    NO_SELF_SHADOWING = 0x80
    NO_PER_TEXEL_LIGHTING = 0x100


class StaticProp:

    def __init__(self):
        self.origin = []
        self.rotation = []
        self.prop_type = 0
        self.first_leaf = 0
        self.leaf_count = 0
        self.solid = 0
        self.flags = StaticPropFlag.NONE
        self.skin = 0
        self.fade_min_dist = 0.0
        self.fade_max_dist = 0.0
        self.lighting_origin = []

        self.forced_fade_scale = 0.0

        self.min_dx_level = 0
        self.max_dx_level = 0

        self.lightmap_resolution = []

        self.min_cpu_level = 0
        self.max_cpu_level = 0
        self.min_gpu_level = 0
        self.max_gpu_level = 0

        self.diffuse_modulation = []
        self.disable_x360 = 0
        self.flags_ex = 0
        self.uniform_scale = 0.0

        self.unk_vector = []

        # Vindictus specific
        self.scaling = [1.0, 1.0, 1.0]

    def parse(self, reader: Buffer, version: int, bsp_version: Tuple[int, int], size: int, app_id: int):
        if bsp_version == (20, 4):
            self._parse_v6(reader)
            reader.skip(72)
            return

        if app_id == SteamAppId.LEFT_4_DEAD and version == 7 and size == 68:
            # Old Left 4 Dead maps use v7 and incompatible with newer v7 from Source 2013
            self._parse_v7_l4d(reader)
            return

        if app_id == SteamAppId.TEAM_FORTRESS_2 and version == 7 and size == 72:
            # Old Team Fortress 2 maps use v7 which became v10 in Source 2013
            self._parse_v10(reader)
            return

        if app_id == SteamAppId.COUNTER_STRIKE_GO and version in (10, 11):
            # Some Counter-Strike: GO use v10 which is not compatible with Source 2013, now use v11
            if version == 10:
                self._parse_v10_csgo(reader)
            else:
                self._parse_v11_csgo(reader)
            return

        if app_id == SteamAppId.BLACK_MESA and version in (10, 11):
            # Black Mesa uses different structures
            if version == 10 and size == 72:
                self._parse_v10(reader)
                return
            elif version == 11:
                if size == 76:
                    self._parse_v11_lite(reader)
                    return
                elif size == 80:
                    self._parse_v11(reader)
                    return

        if version == 4:
            self._parse_v4(reader)
            return

        if version == 5:
            self._parse_v5(reader)
            return

        if version == 6:
            if app_id == SteamAppId.VINDICTUS:
                self._parse_v6_vin(reader)
                return
            else:
                self._parse_v6(reader)
                return

        if version == 7:
            if app_id == SteamAppId.VINDICTUS:
                self._parse_v7_vin(reader)
                return

            if app_id == SteamAppId.LEFT_4_DEAD and size == 68:
                # Old Left 4 Dead maps use v7 and incompatible with newer v7 from Source 2013
                self._parse_v7_l4d(reader)
                return

            if app_id == SteamAppId.TEAM_FORTRESS_2 and size == 72:
                # Old Team Fortress 2 maps use v7 which became v10 in Source 2013
                self._parse_v10(reader)
                return

        if version == 8:
            self._parse_v8(reader)
            return

        if version == 9:
            self._parse_v9(reader)
            return

        if version == 10:
            if app_id == SteamAppId.COUNTER_STRIKE_GO:
                self._parse_v10_csgo(reader)
                return

            self._parse_v10(reader)
            return

        if version == 11:
            if app_id == SteamAppId.COUNTER_STRIKE_GO:
                self._parse_v11_csgo(reader)
                return

            if app_id == SteamAppId.BLACK_MESA:
                if size == 76:
                    self._parse_v11_lite(reader)
                    return
                elif size == 80:
                    self._parse_v11(reader)
                    return

            self._parse_v11(reader)
            return

        if version == 12:
            self._parse_v12(reader)
            return

        logger.error(f'Cannot find handler for static prop of version {version} (size: {size}, app_id: {app_id})')
        reader.skip(size)

    def _parse_v4(self, reader: Buffer):
        self.origin = reader.read_fmt('3f')
        self.rotation = reader.read_fmt('3f')
        self.prop_type, self.first_leaf, self.leaf_count = reader.read_fmt('3H')
        self.solid, self.flags = reader.read_fmt('2B')
        self.skin = reader.read_int32()
        self.fade_min_dist, self.fade_max_dist = reader.read_fmt('2f')
        self.lighting_origin = reader.read_fmt('3f')

    def _parse_v5(self, reader: Buffer):
        self._parse_v4(reader)
        self.forced_fade_scale = reader.read_float()

    def _parse_v6_vin(self, reader: Buffer):
        self._parse_v5(reader)

    def _parse_v6(self, reader: Buffer):
        self._parse_v5(reader)
        self.min_dx_level, self.max_dx_level = reader.read_fmt('2H')

    def _parse_v7_l4d(self, reader: Buffer):
        self._parse_v6(reader)
        self.diffuse_modulation = reader.read_fmt('4B')

    def _parse_v7_vin(self, reader: Buffer):
        self._parse_v6(reader)

    def _parse_v8(self, reader: Buffer):
        self._parse_v5(reader)
        self.min_cpu_level, self.max_cpu_level, self.min_gpu_level, self.max_gpu_level = reader.read_fmt('4B')
        self.diffuse_modulation = reader.read_fmt('4B')

    def _parse_v9(self, reader: Buffer):
        self._parse_v8(reader)
        self.disable_x360 = reader.read_fmt('I')

    def _parse_v10(self, reader: Buffer):
        self._parse_v6(reader)
        self.flags = StaticPropFlag(reader.read_uint32())
        self.lightmap_resolution = reader.read_fmt('2H')

    def _parse_v10_csgo(self, reader: Buffer):
        self._parse_v9(reader)
        reader.skip(4)

    def _parse_v11_lite(self, reader: Buffer):
        self._parse_v10(reader)
        self.diffuse_modulation = reader.read_fmt('4B')

    def _parse_v11(self, reader: Buffer):
        self._parse_v11_lite(reader)
        self.flags_ex = reader.read_int32()

    def _parse_v11_csgo(self, reader: Buffer):
        self._parse_v10_csgo(reader)
        self.uniform_scale = reader.read_float()

    def _parse_v12(self, reader: Buffer):
        self.origin = reader.read_fmt('3f')
        self.rotation = reader.read_fmt('3f')
        self.prop_type = reader.read_int16()
        reader.skip(4 + 2)
        self.skin = reader.read_int32()
        reader.skip(12 * 4)


class StaticPropLump:
    def __init__(self, glump_info):
        from ..game_lump_header import GameLumpHeader
        self._glump_info: GameLumpHeader = glump_info
        self.model_names: List[str] = []
        self.leafs: List[int] = []
        self.static_props: List[StaticProp] = []

    def parse(self, reader: Buffer, bsp: 'BSPFile'):
        content_manager = ContentManager()
        for _ in range(reader.read_int32()):
            self.model_names.append(reader.read_ascii_string(128))
        for _ in range(reader.read_int32()):
            self.leafs.append(reader.read_uint16())
        if self._glump_info.version == 12:
            unk1 = reader.read_int32()
            unk2 = reader.read_int32()
        prop_scaling = {}
        if content_manager.steam_id == SteamAppId.VINDICTUS:
            for _ in range(reader.read_uint32()):
                prop_id = reader.read_int32()
                prop_scaling[prop_id] = reader.read_fmt('3f')

        prop_count = reader.read_int32()
        if prop_count == 0:
            return
        prop_size = reader.remaining() // prop_count
        for i in range(prop_count):
            prop = StaticProp()
            prop.parse(reader, self._glump_info.version, bsp.version, prop_size, content_manager.steam_id)
            self.static_props.append(prop)
        if prop_scaling:
            for prop_id, scale in prop_scaling.items():
                self.static_props[prop_id].scaling[:] = scale
