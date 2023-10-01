import numpy as np


def parse_source_value(value):
    if type(value) is str:
        value: str
        if value.replace('.', '', 1).replace('-', '', 1).isdecimal():
            return float(value) if '.' in value else int(value)
        return 0
    else:
        return value


def parse_int_vector(string):
    if isinstance(string, tuple):
        return list(string)
    elif isinstance(string, np.ndarray):
        return string
    return [parse_source_value(val) for val in string.replace('  ', ' ').split(' ')]


class PhysicsTypeOverride_Mesh:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data


class PhysicsTypeOverride_SingleConvex:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data


class PhysicsTypeOverride_MultiConvex:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data


class PosableSkeleton:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data


class DXLevelChoice:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data


class VScript:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def vscripts(self):
        if "vscripts" in self._entity_data:
            return self._entity_data.get('vscripts')
        return ""


class GameEntity(VScript):
    pass


class Targetname(GameEntity):
    pass

    @property
    def targetname(self):
        if "targetname" in self._entity_data:
            return self._entity_data.get('targetname')
        return None


class Parentname:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def parentname(self):
        if "parentname" in self._entity_data:
            return self._entity_data.get('parentname')
        return None

    @property
    def parentAttachmentName(self):
        if "parentAttachmentName" in self._entity_data:
            return self._entity_data.get('parentAttachmentName')
        return None

    @property
    def local_origin(self):
        if "local_origin" in self._entity_data:
            return parse_int_vector(self._entity_data.get('local_origin'))
        return parse_int_vector("None")

    @property
    def local_angles(self):
        if "local_angles" in self._entity_data:
            return parse_int_vector(self._entity_data.get('local_angles'))
        return parse_int_vector("None")

    @property
    def local_scales(self):
        if "local_scales" in self._entity_data:
            return parse_int_vector(self._entity_data.get('local_scales'))
        return parse_int_vector("None")

    @property
    def useLocalOffset(self):
        if "useLocalOffset" in self._entity_data:
            return bool(self._entity_data.get('useLocalOffset'))
        return bool(0)


class Studiomodel:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return None

    @property
    def skin(self):
        if "skin" in self._entity_data:
            return self._entity_data.get('skin')
        return "default"

    @property
    def bodygroups(self):
        if "bodygroups" in self._entity_data:
            return self._entity_data.get('bodygroups')
        return ""

    @property
    def disableshadows(self):
        if "disableshadows" in self._entity_data:
            return bool(self._entity_data.get('disableshadows'))
        return bool(0)


class BasePlat:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data


class BaseBrush:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data


class EnableDisable:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def StartDisabled(self):
        if "StartDisabled" in self._entity_data:
            return bool(self._entity_data.get('StartDisabled'))
        return bool(0)


class RenderFxChoices:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def renderfx(self):
        if "renderfx" in self._entity_data:
            return self._entity_data.get('renderfx')
        return "kRenderFxNone"


class RenderModeChoices:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def rendermode(self):
        if "rendermode" in self._entity_data:
            return self._entity_data.get('rendermode')
        return "kRenderNormal"


class Shadow:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def disableshadows(self):
        if "disableshadows" in self._entity_data:
            return bool(self._entity_data.get('disableshadows'))
        return bool(0)


class Glow:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def glowstate(self):
        if "glowstate" in self._entity_data:
            return self._entity_data.get('glowstate')
        return "0"

    @property
    def glowrange(self):
        if "glowrange" in self._entity_data:
            return int(self._entity_data.get('glowrange'))
        return int(0)

    @property
    def glowrangemin(self):
        if "glowrangemin" in self._entity_data:
            return int(self._entity_data.get('glowrangemin'))
        return int(0)

    @property
    def glowcolor(self):
        if "glowcolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('glowcolor'))
        return parse_int_vector("0 0 0")


class SystemLevelChoice:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def mincpulevel(self):
        if "mincpulevel" in self._entity_data:
            return self._entity_data.get('mincpulevel')
        return "0"

    @property
    def maxcpulevel(self):
        if "maxcpulevel" in self._entity_data:
            return self._entity_data.get('maxcpulevel')
        return "0"

    @property
    def mingpulevel(self):
        if "mingpulevel" in self._entity_data:
            return self._entity_data.get('mingpulevel')
        return "0"

    @property
    def maxgpulevel(self):
        if "maxgpulevel" in self._entity_data:
            return self._entity_data.get('maxgpulevel')
        return "0"

    @property
    def disableX360(self):
        if "disableX360" in self._entity_data:
            return self._entity_data.get('disableX360')
        return "0"


class RenderFields(RenderFxChoices, RenderModeChoices, SystemLevelChoice):
    pass

    @property
    def renderamt(self):
        if "renderamt" in self._entity_data:
            return int(self._entity_data.get('renderamt'))
        return int(255)

    @property
    def rendercolor(self):
        if "rendercolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('rendercolor'))
        return parse_int_vector("255 255 255")

    @property
    def disablereceiveshadows(self):
        if "disablereceiveshadows" in self._entity_data:
            return bool(self._entity_data.get('disablereceiveshadows'))
        return bool(0)

    @property
    def lightgroup(self):
        if "lightgroup" in self._entity_data:
            return self._entity_data.get('lightgroup')
        return ""

    @property
    def fademindist(self):
        if "fademindist" in self._entity_data:
            return float(self._entity_data.get('fademindist'))
        return float(-1)

    @property
    def fademaxdist(self):
        if "fademaxdist" in self._entity_data:
            return float(self._entity_data.get('fademaxdist'))
        return float(0)

    @property
    def fadescale(self):
        if "fadescale" in self._entity_data:
            return float(self._entity_data.get('fadescale'))
        return float(1)

    @property
    def rendertocubemaps(self):
        if "rendertocubemaps" in self._entity_data:
            return bool(self._entity_data.get('rendertocubemaps'))
        return bool(0)

    @property
    def lightmapstatic(self):
        if "lightmapstatic" in self._entity_data:
            return self._entity_data.get('lightmapstatic')
        return "0"


class Inputfilter:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def InputFilter(self):
        if "InputFilter" in self._entity_data:
            return self._entity_data.get('InputFilter')
        return "0"


class Global:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def globalname(self):
        if "globalname" in self._entity_data:
            return self._entity_data.get('globalname')
        return ""


class EnvGlobal(Targetname):
    pass

    @property
    def initialstate(self):
        if "initialstate" in self._entity_data:
            return self._entity_data.get('initialstate')
        return "0"

    @property
    def counter(self):
        if "counter" in self._entity_data:
            return int(self._entity_data.get('counter'))
        return int(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Set Initial State': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class DamageFilter:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def damagefilter(self):
        if "damagefilter" in self._entity_data:
            return self._entity_data.get('damagefilter')
        return ""


class ResponseContext:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def ResponseContext(self):
        if "ResponseContext" in self._entity_data:
            return self._entity_data.get('ResponseContext')
        return ""


class Breakable(Targetname, DamageFilter, Shadow):
    pass

    @property
    def ExplodeDamage(self):
        if "ExplodeDamage" in self._entity_data:
            return float(self._entity_data.get('ExplodeDamage'))
        return float(0)

    @property
    def ExplodeRadius(self):
        if "ExplodeRadius" in self._entity_data:
            return float(self._entity_data.get('ExplodeRadius'))
        return float(0)

    @property
    def PerformanceMode(self):
        if "PerformanceMode" in self._entity_data:
            return self._entity_data.get('PerformanceMode')
        return "PM_NORMAL"


class BreakableBrush(Breakable, Parentname, Global):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Only Break on Trigger': (1, 0), 'Break on Touch': (2, 0),
                                   'Break on Pressure': (4, 0), 'Break immediately on Physics': (512, 0),
                                   "Don't take physics damage": (1024, 0), "Don't allow bullet penetration": (2048, 0),
                                   "Don't allow hand physics damage": (4096, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def propdata(self):
        if "propdata" in self._entity_data:
            return self._entity_data.get('propdata')
        return "0"

    @property
    def health(self):
        if "health" in self._entity_data:
            return int(self._entity_data.get('health'))
        return int(1)

    @property
    def material(self):
        if "material" in self._entity_data:
            return self._entity_data.get('material')
        return "0"

    @property
    def explosion(self):
        if "explosion" in self._entity_data:
            return self._entity_data.get('explosion')
        return "0"

    @property
    def gibdir(self):
        if "gibdir" in self._entity_data:
            return parse_int_vector(self._entity_data.get('gibdir'))
        return parse_int_vector("0 0 0")

    @property
    def nodamageforces(self):
        if "nodamageforces" in self._entity_data:
            return bool(self._entity_data.get('nodamageforces'))
        return bool(0)

    @property
    def spawnobject(self):
        if "spawnobject" in self._entity_data:
            return self._entity_data.get('spawnobject')
        return "0"

    @property
    def explodemagnitude(self):
        if "explodemagnitude" in self._entity_data:
            return int(self._entity_data.get('explodemagnitude'))
        return int(0)

    @property
    def pressuredelay(self):
        if "pressuredelay" in self._entity_data:
            return float(self._entity_data.get('pressuredelay'))
        return float(0)


class CanBeClientOnly:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def clientSideEntity(self):
        if "clientSideEntity" in self._entity_data:
            return self._entity_data.get('clientSideEntity')
        return "0"


class BreakableProp(Breakable):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Break on Touch': (16, 0), 'Break on Pressure': (32, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def pressuredelay(self):
        if "pressuredelay" in self._entity_data:
            return float(self._entity_data.get('pressuredelay'))
        return float(0)


class BaseNPC(Targetname, RenderFields, DamageFilter, ResponseContext, Shadow, PosableSkeleton):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def squadname(self):
        if "squadname" in self._entity_data:
            return self._entity_data.get('squadname')
        return None

    @property
    def hintgroup(self):
        if "hintgroup" in self._entity_data:
            return self._entity_data.get('hintgroup')
        return ""

    @property
    def hintlimiting(self):
        if "hintlimiting" in self._entity_data:
            return bool(self._entity_data.get('hintlimiting'))
        return bool(0)

    @property
    def allowgenericnodes(self):
        if "allowgenericnodes" in self._entity_data:
            return bool(self._entity_data.get('allowgenericnodes'))
        return bool(None)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Wait Till Seen': (1, 0), 'Gag (No IDLE sounds until angry)': (2, 0),
                                   'Fall to ground (unchecked means *teleport* to ground)': (4, 1),
                                   'Drop Healthkit': (8, 0),
                                   "Efficient - Don't acquire enemies or avoid obstacles": (16, 0),
                                   'Wait For Script': (128, 0), 'Long Visibility/Shoot': (256, 0),
                                   'Fade Corpse': (512, 1), 'Think outside PVS': (1024, 0),
                                   'Template NPC (used by npc_maker, will not spawn)': (2048, 0),
                                   'Do Alternate collision for this NPC (player avoidance)': (4096, 0),
                                   "Don't drop weapons": (8192, 0),
                                   'Ignore player push (dont give way to player)': (16384, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def sleepstate(self):
        if "sleepstate" in self._entity_data:
            return self._entity_data.get('sleepstate')
        return "0"

    @property
    def wakeradius(self):
        if "wakeradius" in self._entity_data:
            return float(self._entity_data.get('wakeradius'))
        return float(0)

    @property
    def wakesquad(self):
        if "wakesquad" in self._entity_data:
            return bool(self._entity_data.get('wakesquad'))
        return bool(0)

    @property
    def enemyfilter(self):
        if "enemyfilter" in self._entity_data:
            return self._entity_data.get('enemyfilter')
        return ""

    @property
    def ignoreunseenenemies(self):
        if "ignoreunseenenemies" in self._entity_data:
            return bool(self._entity_data.get('ignoreunseenenemies'))
        return bool(0)

    @property
    def physdamagescale(self):
        if "physdamagescale" in self._entity_data:
            return float(self._entity_data.get('physdamagescale'))
        return float(1.0)

    @property
    def NavRestrictionVolume(self):
        if "NavRestrictionVolume" in self._entity_data:
            return self._entity_data.get('NavRestrictionVolume')
        return ""

    @property
    def spawnasragdoll(self):
        if "spawnasragdoll" in self._entity_data:
            return self._entity_data.get('spawnasragdoll')
        return "0"

    @property
    def UseAltNpcAvoid(self):
        if "UseAltNpcAvoid" in self._entity_data:
            return self._entity_data.get('UseAltNpcAvoid')
        return "0"

    @property
    def DefaultAnim(self):
        if "DefaultAnim" in self._entity_data:
            return self._entity_data.get('DefaultAnim')
        return ""


class info_npc_spawn_destination(Targetname, Parentname):
    pass

    icon_sprite = "editor/info_target.vmat"

    @property
    def ReuseDelay(self):
        if "ReuseDelay" in self._entity_data:
            return float(self._entity_data.get('ReuseDelay'))
        return float(1)

    @property
    def RenameNPC(self):
        if "RenameNPC" in self._entity_data:
            return self._entity_data.get('RenameNPC')
        return ""


class BaseNPCMaker(Targetname, Parentname):
    pass

    icon_sprite = "editor/npc_maker.vmat"

    @property
    def StartDisabled(self):
        if "StartDisabled" in self._entity_data:
            return bool(self._entity_data.get('StartDisabled'))
        return bool(1)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Fade Corpse': (16, 0), 'Infinite Children': (32, 0), 'Do Not Drop': (64, 0),
                                   "Don't Spawn While Visible": (128, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def Radius(self):
        if "Radius" in self._entity_data:
            return float(self._entity_data.get('Radius'))
        return float(256)

    @property
    def DestinationGroup(self):
        if "DestinationGroup" in self._entity_data:
            return self._entity_data.get('DestinationGroup')
        return None

    @property
    def CriterionVisibility(self):
        if "CriterionVisibility" in self._entity_data:
            return self._entity_data.get('CriterionVisibility')
        return "2"

    @property
    def CriterionDistance(self):
        if "CriterionDistance" in self._entity_data:
            return self._entity_data.get('CriterionDistance')
        return "2"

    @property
    def MinSpawnDistance(self):
        if "MinSpawnDistance" in self._entity_data:
            return int(self._entity_data.get('MinSpawnDistance'))
        return int(0)

    @property
    def MaxNPCCount(self):
        if "MaxNPCCount" in self._entity_data:
            return int(self._entity_data.get('MaxNPCCount'))
        return int(1)

    @property
    def SpawnFrequency(self):
        if "SpawnFrequency" in self._entity_data:
            return int(self._entity_data.get('SpawnFrequency'))
        return int(5)

    @property
    def RetryFrequency(self):
        if "RetryFrequency" in self._entity_data:
            return int(self._entity_data.get('RetryFrequency'))
        return int(-1)

    @property
    def MaxLiveChildren(self):
        if "MaxLiveChildren" in self._entity_data:
            return int(self._entity_data.get('MaxLiveChildren'))
        return int(5)

    @property
    def HullCheckMode(self):
        if "HullCheckMode" in self._entity_data:
            return self._entity_data.get('HullCheckMode')
        return "0"


class npc_template_maker(BaseNPCMaker):
    pass

    icon_sprite = "editor/npc_maker.vmat"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {"Don't preload template models": (512, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def TemplateName(self):
        if "TemplateName" in self._entity_data:
            return self._entity_data.get('TemplateName')
        return ""


class BaseHelicopter(BaseNPC):
    pass

    @property
    def InitialSpeed(self):
        if "InitialSpeed" in self._entity_data:
            return self._entity_data.get('InitialSpeed')
        return "0"

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'No Rotorwash': (32, 0), 'Await Input': (64, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class PlayerClass:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data


class Light:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def _light(self):
        if "_light" in self._entity_data:
            return parse_int_vector(self._entity_data.get('_light'))
        return parse_int_vector("255 255 255 1500")

    @property
    def _lightHDR(self):
        if "_lightHDR" in self._entity_data:
            return parse_int_vector(self._entity_data.get('_lightHDR'))
        return parse_int_vector("-1 -1 -1 1")

    @property
    def _lightscaleHDR(self):
        if "_lightscaleHDR" in self._entity_data:
            return float(self._entity_data.get('_lightscaleHDR'))
        return float(1)

    @property
    def style(self):
        if "style" in self._entity_data:
            return self._entity_data.get('style')
        return "0"

    @property
    def pattern(self):
        if "pattern" in self._entity_data:
            return self._entity_data.get('pattern')
        return ""

    @property
    def _constant_attn(self):
        if "_constant_attn" in self._entity_data:
            return self._entity_data.get('_constant_attn')
        return "0"

    @property
    def _linear_attn(self):
        if "_linear_attn" in self._entity_data:
            return self._entity_data.get('_linear_attn')
        return "0"

    @property
    def _quadratic_attn(self):
        if "_quadratic_attn" in self._entity_data:
            return self._entity_data.get('_quadratic_attn')
        return "1"

    @property
    def _fifty_percent_distance(self):
        if "_fifty_percent_distance" in self._entity_data:
            return self._entity_data.get('_fifty_percent_distance')
        return "0"

    @property
    def _zero_percent_distance(self):
        if "_zero_percent_distance" in self._entity_data:
            return self._entity_data.get('_zero_percent_distance')
        return "0"

    @property
    def _hardfalloff(self):
        if "_hardfalloff" in self._entity_data:
            return int(self._entity_data.get('_hardfalloff'))
        return int(0)


class Node:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def nodeid(self):
        if "nodeid" in self._entity_data:
            return int(self._entity_data.get('nodeid'))
        return int(0)


class HintNode(Node):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Allow jump up': (65536, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def hinttype(self):
        if "hinttype" in self._entity_data:
            return self._entity_data.get('hinttype')
        return "0"

    @property
    def generictype(self):
        if "generictype" in self._entity_data:
            return self._entity_data.get('generictype')
        return ""

    @property
    def hintactivity(self):
        if "hintactivity" in self._entity_data:
            return self._entity_data.get('hintactivity')
        return ""

    @property
    def nodeFOV(self):
        if "nodeFOV" in self._entity_data:
            return self._entity_data.get('nodeFOV')
        return "180"

    @property
    def StartHintDisabled(self):
        if "StartHintDisabled" in self._entity_data:
            return bool(self._entity_data.get('StartHintDisabled'))
        return bool(0)

    @property
    def Group(self):
        if "Group" in self._entity_data:
            return self._entity_data.get('Group')
        return ""

    @property
    def TargetNode(self):
        if "TargetNode" in self._entity_data:
            return int(self._entity_data.get('TargetNode'))
        return int(-1)

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return int(self._entity_data.get('radius'))
        return int(0)

    @property
    def IgnoreFacing(self):
        if "IgnoreFacing" in self._entity_data:
            return self._entity_data.get('IgnoreFacing')
        return "2"

    @property
    def MinimumState(self):
        if "MinimumState" in self._entity_data:
            return self._entity_data.get('MinimumState')
        return "1"

    @property
    def MaximumState(self):
        if "MaximumState" in self._entity_data:
            return self._entity_data.get('MaximumState')
        return "3"


class TriggerOnce(Targetname, Parentname, EnableDisable, Global):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Clients': (1, 1), 'NPCs': (2, 0), 'Pushables': (4, 0), 'Physics Objects': (8, 0),
                                   'Only player ally NPCs': (16, 0), 'Only clients in vehicles': (32, 0),
                                   'Everything (not including physics debris)': (64, 0),
                                   'Only clients *not* in vehicles': (512, 0), 'Physics debris': (1024, 0),
                                   'Only NPCs in vehicles (respects player ally flag)': (2048, 0),
                                   'Correctly account for object mass (trigger_push used to assume 100Kg) and multiple component physobjs (car, blob...)': (
                                   4096, 1), "Ignore client's hands": (8192, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def filtername(self):
        if "filtername" in self._entity_data:
            return self._entity_data.get('filtername')
        return None


class Trigger(TriggerOnce):
    pass


class worldbase:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def targetname(self):
        if "targetname" in self._entity_data:
            return self._entity_data.get('targetname')
        return None

    @property
    def skyname(self):
        if "skyname" in self._entity_data:
            return self._entity_data.get('skyname')
        return "sky_day01_01"

    @property
    def startdark(self):
        if "startdark" in self._entity_data:
            return bool(self._entity_data.get('startdark'))
        return bool(0)

    @property
    def startcolor(self):
        if "startcolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('startcolor'))
        return parse_int_vector("0 0 0")

    @property
    def pvstype(self):
        if "pvstype" in self._entity_data:
            return self._entity_data.get('pvstype')
        return "10"

    @property
    def newunit(self):
        if "newunit" in self._entity_data:
            return self._entity_data.get('newunit')
        return "0"

    @property
    def maxpropscreenwidth(self):
        if "maxpropscreenwidth" in self._entity_data:
            return float(self._entity_data.get('maxpropscreenwidth'))
        return float(-1)

    @property
    def minpropscreenwidth(self):
        if "minpropscreenwidth" in self._entity_data:
            return float(self._entity_data.get('minpropscreenwidth'))
        return float(0)

    @property
    def vrchaperone(self):
        if "vrchaperone" in self._entity_data:
            return self._entity_data.get('vrchaperone')
        return "0"

    @property
    def vrmovement(self):
        if "vrmovement" in self._entity_data:
            return self._entity_data.get('vrmovement')
        return "0"


class ambient_generic(Targetname, Parentname):
    pass

    icon_sprite = "editor/ambient_generic.vmat"

    @property
    def message(self):
        if "message" in self._entity_data:
            return self._entity_data.get('message')
        return ""

    @property
    def health(self):
        if "health" in self._entity_data:
            return int(self._entity_data.get('health'))
        return int(10)

    @property
    def preset(self):
        if "preset" in self._entity_data:
            return self._entity_data.get('preset')
        return "0"

    @property
    def volstart(self):
        if "volstart" in self._entity_data:
            return int(self._entity_data.get('volstart'))
        return int(0)

    @property
    def fadeinsecs(self):
        if "fadeinsecs" in self._entity_data:
            return int(self._entity_data.get('fadeinsecs'))
        return int(0)

    @property
    def fadeoutsecs(self):
        if "fadeoutsecs" in self._entity_data:
            return int(self._entity_data.get('fadeoutsecs'))
        return int(0)

    @property
    def pitch(self):
        if "pitch" in self._entity_data:
            return int(self._entity_data.get('pitch'))
        return int(100)

    @property
    def pitchstart(self):
        if "pitchstart" in self._entity_data:
            return int(self._entity_data.get('pitchstart'))
        return int(100)

    @property
    def spinup(self):
        if "spinup" in self._entity_data:
            return int(self._entity_data.get('spinup'))
        return int(0)

    @property
    def spindown(self):
        if "spindown" in self._entity_data:
            return int(self._entity_data.get('spindown'))
        return int(0)

    @property
    def lfotype(self):
        if "lfotype" in self._entity_data:
            return int(self._entity_data.get('lfotype'))
        return int(0)

    @property
    def lforate(self):
        if "lforate" in self._entity_data:
            return int(self._entity_data.get('lforate'))
        return int(0)

    @property
    def lfomodpitch(self):
        if "lfomodpitch" in self._entity_data:
            return int(self._entity_data.get('lfomodpitch'))
        return int(0)

    @property
    def lfomodvol(self):
        if "lfomodvol" in self._entity_data:
            return int(self._entity_data.get('lfomodvol'))
        return int(0)

    @property
    def cspinup(self):
        if "cspinup" in self._entity_data:
            return int(self._entity_data.get('cspinup'))
        return int(0)

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return self._entity_data.get('radius')
        return "1250"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Play everywhere': (1, 0), 'Start Silent': (16, 1),
                                   'Is NOT Looped': (32, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def SourceEntityName(self):
        if "SourceEntityName" in self._entity_data:
            return self._entity_data.get('SourceEntityName')
        return None


class point_soundevent(Targetname, Parentname):
    pass

    icon_sprite = "editor/snd_event.vmat"

    @property
    def soundName(self):
        if "soundName" in self._entity_data:
            return self._entity_data.get('soundName')
        return ""

    @property
    def sourceEntityName(self):
        if "sourceEntityName" in self._entity_data:
            return self._entity_data.get('sourceEntityName')
        return ""

    @property
    def startOnSpawn(self):
        if "startOnSpawn" in self._entity_data:
            return bool(self._entity_data.get('startOnSpawn'))
        return bool()

    @property
    def toLocalPlayer(self):
        if "toLocalPlayer" in self._entity_data:
            return bool(self._entity_data.get('toLocalPlayer'))
        return bool()

    @property
    def stopOnNew(self):
        if "stopOnNew" in self._entity_data:
            return bool(self._entity_data.get('stopOnNew'))
        return bool(1)

    @property
    def saveAndRestore(self):
        if "saveAndRestore" in self._entity_data:
            return bool(self._entity_data.get('saveAndRestore'))
        return bool(0)

    @property
    def sourceEntityAttachment(self):
        if "sourceEntityAttachment" in self._entity_data:
            return self._entity_data.get('sourceEntityAttachment')
        return None


class snd_event_point(point_soundevent):
    pass

    icon_sprite = "editor/snd_event.vmat"


class snd_event_alignedbox(point_soundevent):
    pass

    icon_sprite = "editor/snd_event.vmat"

    @property
    def box_mins(self):
        if "box_mins" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_mins'))
        return parse_int_vector("-64 -64 -64")

    @property
    def box_maxs(self):
        if "box_maxs" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_maxs'))
        return parse_int_vector("64 64 64")


class snd_stack_save(Targetname):
    pass

    icon_sprite = "editor/snd_event.vmat"

    @property
    def stackToSave(self):
        if "stackToSave" in self._entity_data:
            return self._entity_data.get('stackToSave')
        return ""


class snd_event_param(Targetname, Parentname):
    pass

    icon_sprite = "editor/snd_opvar_set.vmat"

    @property
    def parameterName(self):
        if "parameterName" in self._entity_data:
            return self._entity_data.get('parameterName')
        return ""

    @property
    def floatValue(self):
        if "floatValue" in self._entity_data:
            return float(self._entity_data.get('floatValue'))
        return float()


class snd_opvar_set(Targetname):
    pass

    icon_sprite = "editor/snd_opvar_set.vmat"

    @property
    def stackName(self):
        if "stackName" in self._entity_data:
            return self._entity_data.get('stackName')
        return ""

    @property
    def operatorName(self):
        if "operatorName" in self._entity_data:
            return self._entity_data.get('operatorName')
        return ""

    @property
    def opvarName(self):
        if "opvarName" in self._entity_data:
            return self._entity_data.get('opvarName')
        return ""

    @property
    def opvarValueType(self):
        if "opvarValueType" in self._entity_data:
            return self._entity_data.get('opvarValueType')
        return "0"

    @property
    def opvarValue(self):
        if "opvarValue" in self._entity_data:
            return float(self._entity_data.get('opvarValue'))
        return float(1.0)

    @property
    def opvarValueString(self):
        if "opvarValueString" in self._entity_data:
            return self._entity_data.get('opvarValueString')
        return "null"

    @property
    def opvarIndex(self):
        if "opvarIndex" in self._entity_data:
            return int(self._entity_data.get('opvarIndex'))
        return int(0)

    @property
    def setOnSpawn(self):
        if "setOnSpawn" in self._entity_data:
            return bool(self._entity_data.get('setOnSpawn'))
        return bool(0)


class SndOpvarSetPointBase:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def stackName(self):
        if "stackName" in self._entity_data:
            return self._entity_data.get('stackName')
        return ""

    @property
    def operatorName(self):
        if "operatorName" in self._entity_data:
            return self._entity_data.get('operatorName')
        return ""

    @property
    def opvarName(self):
        if "opvarName" in self._entity_data:
            return self._entity_data.get('opvarName')
        return ""

    @property
    def opvarArrayIndex(self):
        if "opvarArrayIndex" in self._entity_data:
            return int(self._entity_data.get('opvarArrayIndex'))
        return int()

    @property
    def distanceMapMin(self):
        if "distanceMapMin" in self._entity_data:
            return float(self._entity_data.get('distanceMapMin'))
        return float()

    @property
    def distanceMapMax(self):
        if "distanceMapMax" in self._entity_data:
            return float(self._entity_data.get('distanceMapMax'))
        return float()

    @property
    def occlusionRadius(self):
        if "occlusionRadius" in self._entity_data:
            return float(self._entity_data.get('occlusionRadius'))
        return float()

    @property
    def occlusionMin(self):
        if "occlusionMin" in self._entity_data:
            return float(self._entity_data.get('occlusionMin'))
        return float()

    @property
    def occlusionMax(self):
        if "occlusionMax" in self._entity_data:
            return float(self._entity_data.get('occlusionMax'))
        return float()

    @property
    def simulationMode(self):
        if "simulationMode" in self._entity_data:
            return self._entity_data.get('simulationMode')
        return "0"

    @property
    def visibilitySamples(self):
        if "visibilitySamples" in self._entity_data:
            return int(self._entity_data.get('visibilitySamples'))
        return int(8)

    @property
    def setToValueOnDisable(self):
        if "setToValueOnDisable" in self._entity_data:
            return bool(self._entity_data.get('setToValueOnDisable'))
        return bool(0)

    @property
    def disabledValue(self):
        if "disabledValue" in self._entity_data:
            return float(self._entity_data.get('disabledValue'))
        return float(0.0)

    @property
    def startDisabled(self):
        if "startDisabled" in self._entity_data:
            return bool(self._entity_data.get('startDisabled'))
        return bool(0)

    @property
    def autoDisable(self):
        if "autoDisable" in self._entity_data:
            return bool(self._entity_data.get('autoDisable'))
        return bool(0)


class SndOpvarSetPointBaseAddition:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def distanceMin(self):
        if "distanceMin" in self._entity_data:
            return float(self._entity_data.get('distanceMin'))
        return float(5.0)

    @property
    def distanceMax(self):
        if "distanceMax" in self._entity_data:
            return float(self._entity_data.get('distanceMax'))
        return float(25.0)

    @property
    def sourceEntityName(self):
        if "sourceEntityName" in self._entity_data:
            return self._entity_data.get('sourceEntityName')
        return ""


class snd_opvar_set_point(Targetname, SndOpvarSetPointBaseAddition, SndOpvarSetPointBase):
    pass

    icon_sprite = "editor/snd_opvar_set.vmat"

    @property
    def dynamicEntityName(self):
        if "dynamicEntityName" in self._entity_data:
            return self._entity_data.get('dynamicEntityName')
        return ""

    @property
    def dynamicProxyPoint(self):
        if "dynamicProxyPoint" in self._entity_data:
            return parse_int_vector(self._entity_data.get('dynamicProxyPoint'))
        return parse_int_vector("")

    @property
    def dynamicMaximumOcclusion(self):
        if "dynamicMaximumOcclusion" in self._entity_data:
            return float(self._entity_data.get('dynamicMaximumOcclusion'))
        return float(1.0)


class snd_opvar_set_aabb(Targetname, SndOpvarSetPointBase):
    pass

    icon_sprite = "editor/snd_opvar_set.vmat"

    @property
    def AABBDirection(self):
        if "AABBDirection" in self._entity_data:
            return self._entity_data.get('AABBDirection')
        return "0"

    @property
    def box_inner_mins(self):
        if "box_inner_mins" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_inner_mins'))
        return parse_int_vector("-32 -32 -32")

    @property
    def box_inner_maxs(self):
        if "box_inner_maxs" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_inner_maxs'))
        return parse_int_vector("32 32 32")

    @property
    def box_outer_mins(self):
        if "box_outer_mins" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_outer_mins'))
        return parse_int_vector("-64 -64 -64")

    @property
    def box_outer_maxs(self):
        if "box_outer_maxs" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_outer_maxs'))
        return parse_int_vector("64 64 64")


class snd_opvar_set_obb(snd_opvar_set_aabb):
    pass

    icon_sprite = "editor/snd_opvar_set.vmat"


class func_lod(Targetname):
    pass

    @property
    def DisappearMinDist(self):
        if "DisappearMinDist" in self._entity_data:
            return int(self._entity_data.get('DisappearMinDist'))
        return int(2000)

    @property
    def DisappearMaxDist(self):
        if "DisappearMaxDist" in self._entity_data:
            return int(self._entity_data.get('DisappearMaxDist'))
        return int(2200)

    @property
    def Solid(self):
        if "Solid" in self._entity_data:
            return self._entity_data.get('Solid')
        return "0"


class env_zoom(Targetname):
    pass

    @property
    def Rate(self):
        if "Rate" in self._entity_data:
            return float(self._entity_data.get('Rate'))
        return float(1.0)

    @property
    def FOV(self):
        if "FOV" in self._entity_data:
            return int(self._entity_data.get('FOV'))
        return int(75)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Allow Suit Zoom': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class env_screenoverlay(Targetname):
    pass

    @property
    def OverlayName1(self):
        if "OverlayName1" in self._entity_data:
            return self._entity_data.get('OverlayName1')
        return ""

    @property
    def OverlayTime1(self):
        if "OverlayTime1" in self._entity_data:
            return float(self._entity_data.get('OverlayTime1'))
        return float(1.0)

    @property
    def OverlayName2(self):
        if "OverlayName2" in self._entity_data:
            return self._entity_data.get('OverlayName2')
        return ""

    @property
    def OverlayTime2(self):
        if "OverlayTime2" in self._entity_data:
            return float(self._entity_data.get('OverlayTime2'))
        return float(1.0)

    @property
    def OverlayName3(self):
        if "OverlayName3" in self._entity_data:
            return self._entity_data.get('OverlayName3')
        return ""

    @property
    def OverlayTime3(self):
        if "OverlayTime3" in self._entity_data:
            return float(self._entity_data.get('OverlayTime3'))
        return float(1.0)

    @property
    def OverlayName4(self):
        if "OverlayName4" in self._entity_data:
            return self._entity_data.get('OverlayName4')
        return ""

    @property
    def OverlayTime4(self):
        if "OverlayTime4" in self._entity_data:
            return float(self._entity_data.get('OverlayTime4'))
        return float(1.0)

    @property
    def OverlayName5(self):
        if "OverlayName5" in self._entity_data:
            return self._entity_data.get('OverlayName5')
        return ""

    @property
    def OverlayTime5(self):
        if "OverlayTime5" in self._entity_data:
            return float(self._entity_data.get('OverlayTime5'))
        return float(1.0)

    @property
    def OverlayName6(self):
        if "OverlayName6" in self._entity_data:
            return self._entity_data.get('OverlayName6')
        return ""

    @property
    def OverlayTime6(self):
        if "OverlayTime6" in self._entity_data:
            return float(self._entity_data.get('OverlayTime6'))
        return float(1.0)

    @property
    def OverlayName7(self):
        if "OverlayName7" in self._entity_data:
            return self._entity_data.get('OverlayName7')
        return ""

    @property
    def OverlayTime7(self):
        if "OverlayTime7" in self._entity_data:
            return float(self._entity_data.get('OverlayTime7'))
        return float(1.0)

    @property
    def OverlayName8(self):
        if "OverlayName8" in self._entity_data:
            return self._entity_data.get('OverlayName8')
        return ""

    @property
    def OverlayTime8(self):
        if "OverlayTime8" in self._entity_data:
            return float(self._entity_data.get('OverlayTime8'))
        return float(1.0)

    @property
    def OverlayName9(self):
        if "OverlayName9" in self._entity_data:
            return self._entity_data.get('OverlayName9')
        return ""

    @property
    def OverlayTime9(self):
        if "OverlayTime9" in self._entity_data:
            return float(self._entity_data.get('OverlayTime9'))
        return float(1.0)

    @property
    def OverlayName10(self):
        if "OverlayName10" in self._entity_data:
            return self._entity_data.get('OverlayName10')
        return ""

    @property
    def OverlayTime10(self):
        if "OverlayTime10" in self._entity_data:
            return float(self._entity_data.get('OverlayTime10'))
        return float(1.0)


class env_screeneffect(Targetname):
    pass

    @property
    def type(self):
        if "type" in self._entity_data:
            return self._entity_data.get('type')
        return "0"


class env_texturetoggle(Targetname):
    pass

    icon_sprite = "editor/env_texturetoggle.vmat"

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None


class env_splash(Targetname):
    pass

    @property
    def scale(self):
        if "scale" in self._entity_data:
            return float(self._entity_data.get('scale'))
        return float(8.0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Automatically find water surface (place entity above water)': (1, 0),
                                   'Diminish with depth (diminished completely in 10 feet of water)': (2, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class env_particlelight(Parentname):
    pass

    @property
    def Color(self):
        if "Color" in self._entity_data:
            return parse_int_vector(self._entity_data.get('Color'))
        return parse_int_vector("255 0 0")

    @property
    def Intensity(self):
        if "Intensity" in self._entity_data:
            return int(self._entity_data.get('Intensity'))
        return int(5000)

    @property
    def directional(self):
        if "directional" in self._entity_data:
            return bool(self._entity_data.get('directional'))
        return bool(0)

    @property
    def PSName(self):
        if "PSName" in self._entity_data:
            return self._entity_data.get('PSName')
        return ""


class env_sun(Targetname):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def use_angles(self):
        if "use_angles" in self._entity_data:
            return bool(self._entity_data.get('use_angles'))
        return bool(0)

    @property
    def rendercolor(self):
        if "rendercolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('rendercolor'))
        return parse_int_vector("100 80 80")

    @property
    def overlaycolor(self):
        if "overlaycolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('overlaycolor'))
        return parse_int_vector("0 0 0")

    @property
    def size(self):
        if "size" in self._entity_data:
            return int(self._entity_data.get('size'))
        return int(16)

    @property
    def overlaysize(self):
        if "overlaysize" in self._entity_data:
            return int(self._entity_data.get('overlaysize'))
        return int(-1)

    @property
    def material(self):
        if "material" in self._entity_data:
            return self._entity_data.get('material')
        return "materials/sprites/light_glow02_add_noz"

    @property
    def overlaymaterial(self):
        if "overlaymaterial" in self._entity_data:
            return self._entity_data.get('overlaymaterial')
        return "materials/sprites/light_glow02_add_noz"

    @property
    def HDRColorScale(self):
        if "HDRColorScale" in self._entity_data:
            return float(self._entity_data.get('HDRColorScale'))
        return float(1.0)


class env_tonemap_controller(Targetname):
    pass

    icon_sprite = "materials/editor/env_tonemap_controller.vmat"

    @property
    def MinExposure(self):
        if "MinExposure" in self._entity_data:
            return float(self._entity_data.get('MinExposure'))
        return float(0.25)

    @property
    def MaxExposure(self):
        if "MaxExposure" in self._entity_data:
            return float(self._entity_data.get('MaxExposure'))
        return float(8.0)

    @property
    def percent_bright_pixels(self):
        if "percent_bright_pixels" in self._entity_data:
            return float(self._entity_data.get('percent_bright_pixels'))
        return float(-1.0)

    @property
    def percent_target(self):
        if "percent_target" in self._entity_data:
            return float(self._entity_data.get('percent_target'))
        return float(-1.0)

    @property
    def rate(self):
        if "rate" in self._entity_data:
            return float(self._entity_data.get('rate'))
        return float(2.0)

    @property
    def master(self):
        if "master" in self._entity_data:
            return bool(self._entity_data.get('master'))
        return bool(0)


class game_ragdoll_manager(Targetname):
    pass

    @property
    def MaxRagdollCount(self):
        if "MaxRagdollCount" in self._entity_data:
            return int(self._entity_data.get('MaxRagdollCount'))
        return int(-1)

    @property
    def MaxRagdollCountDX8(self):
        if "MaxRagdollCountDX8" in self._entity_data:
            return int(self._entity_data.get('MaxRagdollCountDX8'))
        return int(-1)

    @property
    def SaveImportant(self):
        if "SaveImportant" in self._entity_data:
            return bool(self._entity_data.get('SaveImportant'))
        return bool(0)


class game_gib_manager(Targetname):
    pass

    @property
    def maxpieces(self):
        if "maxpieces" in self._entity_data:
            return int(self._entity_data.get('maxpieces'))
        return int(-1)

    @property
    def maxpiecesdx8(self):
        if "maxpiecesdx8" in self._entity_data:
            return int(self._entity_data.get('maxpiecesdx8'))
        return int(-1)

    @property
    def allownewgibs(self):
        if "allownewgibs" in self._entity_data:
            return bool(self._entity_data.get('allownewgibs'))
        return bool(0)


class env_dof_controller(Targetname):
    pass

    icon_sprite = "editor/env_dof_controller.vmat"


class env_lightglow(Parentname, Targetname):
    pass

    @property
    def rendercolor(self):
        if "rendercolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('rendercolor'))
        return parse_int_vector("255 255 255")

    @property
    def VerticalGlowSize(self):
        if "VerticalGlowSize" in self._entity_data:
            return int(self._entity_data.get('VerticalGlowSize'))
        return int(30)

    @property
    def HorizontalGlowSize(self):
        if "HorizontalGlowSize" in self._entity_data:
            return int(self._entity_data.get('HorizontalGlowSize'))
        return int(30)

    @property
    def MinDist(self):
        if "MinDist" in self._entity_data:
            return int(self._entity_data.get('MinDist'))
        return int(500)

    @property
    def MaxDist(self):
        if "MaxDist" in self._entity_data:
            return int(self._entity_data.get('MaxDist'))
        return int(2000)

    @property
    def OuterMaxDist(self):
        if "OuterMaxDist" in self._entity_data:
            return int(self._entity_data.get('OuterMaxDist'))
        return int(0)

    @property
    def GlowProxySize(self):
        if "GlowProxySize" in self._entity_data:
            return float(self._entity_data.get('GlowProxySize'))
        return float(2.0)

    @property
    def HDRColorScale(self):
        if "HDRColorScale" in self._entity_data:
            return float(self._entity_data.get('HDRColorScale'))
        return float(1.0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Visible only from front': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class env_smokestack(Parentname):
    pass

    @property
    def targetname(self):
        if "targetname" in self._entity_data:
            return self._entity_data.get('targetname')
        return None

    @property
    def InitialState(self):
        if "InitialState" in self._entity_data:
            return self._entity_data.get('InitialState')
        return "0"

    @property
    def BaseSpread(self):
        if "BaseSpread" in self._entity_data:
            return int(self._entity_data.get('BaseSpread'))
        return int(20)

    @property
    def SpreadSpeed(self):
        if "SpreadSpeed" in self._entity_data:
            return int(self._entity_data.get('SpreadSpeed'))
        return int(15)

    @property
    def Speed(self):
        if "Speed" in self._entity_data:
            return int(self._entity_data.get('Speed'))
        return int(30)

    @property
    def StartSize(self):
        if "StartSize" in self._entity_data:
            return int(self._entity_data.get('StartSize'))
        return int(20)

    @property
    def EndSize(self):
        if "EndSize" in self._entity_data:
            return int(self._entity_data.get('EndSize'))
        return int(30)

    @property
    def Rate(self):
        if "Rate" in self._entity_data:
            return int(self._entity_data.get('Rate'))
        return int(20)

    @property
    def JetLength(self):
        if "JetLength" in self._entity_data:
            return int(self._entity_data.get('JetLength'))
        return int(180)

    @property
    def WindAngle(self):
        if "WindAngle" in self._entity_data:
            return int(self._entity_data.get('WindAngle'))
        return int(0)

    @property
    def WindSpeed(self):
        if "WindSpeed" in self._entity_data:
            return int(self._entity_data.get('WindSpeed'))
        return int(0)

    @property
    def SmokeMaterial(self):
        if "SmokeMaterial" in self._entity_data:
            return self._entity_data.get('SmokeMaterial')
        return "particle/SmokeStack.vmat"

    @property
    def twist(self):
        if "twist" in self._entity_data:
            return int(self._entity_data.get('twist'))
        return int(0)

    @property
    def roll(self):
        if "roll" in self._entity_data:
            return float(self._entity_data.get('roll'))
        return float(0)

    @property
    def rendercolor(self):
        if "rendercolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('rendercolor'))
        return parse_int_vector("255 255 255")

    @property
    def renderamt(self):
        if "renderamt" in self._entity_data:
            return int(self._entity_data.get('renderamt'))
        return int(255)


class env_fade(Targetname):
    pass

    icon_sprite = "editor/env_fade"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Fade From': (1, 0), 'Modulate': (2, 0), 'Triggering player only': (4, 0),
                                   'Stay Out': (8, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def duration(self):
        if "duration" in self._entity_data:
            return self._entity_data.get('duration')
        return "2"

    @property
    def holdtime(self):
        if "holdtime" in self._entity_data:
            return self._entity_data.get('holdtime')
        return "0"

    @property
    def renderamt(self):
        if "renderamt" in self._entity_data:
            return int(self._entity_data.get('renderamt'))
        return int(255)

    @property
    def rendercolor(self):
        if "rendercolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('rendercolor'))
        return parse_int_vector("0 0 0")


class env_player_surface_trigger(Targetname):
    pass

    @property
    def gamematerial(self):
        if "gamematerial" in self._entity_data:
            return self._entity_data.get('gamematerial')
        return "0"


class trigger_tonemap(Targetname):
    pass

    @property
    def TonemapName(self):
        if "TonemapName" in self._entity_data:
            return self._entity_data.get('TonemapName')
        return None


class func_useableladder(Targetname, Parentname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Fake Ladder': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def point1(self):
        if "point1" in self._entity_data:
            return parse_int_vector(self._entity_data.get('point1'))
        return parse_int_vector("None")

    @property
    def StartDisabled(self):
        if "StartDisabled" in self._entity_data:
            return bool(self._entity_data.get('StartDisabled'))
        return bool(0)

    @property
    def AutoRideSpeed(self):
        if "AutoRideSpeed" in self._entity_data:
            return float(self._entity_data.get('AutoRideSpeed'))
        return float(0)

    @property
    def ladderSurfaceProperties(self):
        if "ladderSurfaceProperties" in self._entity_data:
            return self._entity_data.get('ladderSurfaceProperties')
        return None


class func_ladderendpoint(Targetname, Parentname):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None


class info_ladder_dismount(Parentname):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None


class func_areaportalwindow(Targetname):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def FadeStartDist(self):
        if "FadeStartDist" in self._entity_data:
            return int(self._entity_data.get('FadeStartDist'))
        return int(128)

    @property
    def FadeDist(self):
        if "FadeDist" in self._entity_data:
            return int(self._entity_data.get('FadeDist'))
        return int(512)

    @property
    def TranslucencyLimit(self):
        if "TranslucencyLimit" in self._entity_data:
            return self._entity_data.get('TranslucencyLimit')
        return "0"

    @property
    def BackgroundBModel(self):
        if "BackgroundBModel" in self._entity_data:
            return self._entity_data.get('BackgroundBModel')
        return ""

    @property
    def PortalVersion(self):
        if "PortalVersion" in self._entity_data:
            return int(self._entity_data.get('PortalVersion'))
        return int(1)


class func_wall(Targetname, RenderFields, Global, Shadow):
    pass

    @property
    def _minlight(self):
        if "_minlight" in self._entity_data:
            return self._entity_data.get('_minlight')
        return None


class func_clip_interaction_layer(Targetname, EnableDisable):
    pass

    @property
    def InteractsAs(self):
        if "InteractsAs" in self._entity_data:
            return self._entity_data.get('InteractsAs')
        return "Default"

    @property
    def InteractsWith(self):
        if "InteractsWith" in self._entity_data:
            return self._entity_data.get('InteractsWith')
        return "Default"


class func_brush(Targetname, Parentname, RenderFields, Global, Inputfilter, EnableDisable, Shadow):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Ignore player +USE': (2, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def _minlight(self):
        if "_minlight" in self._entity_data:
            return self._entity_data.get('_minlight')
        return None

    @property
    def Solidity(self):
        if "Solidity" in self._entity_data:
            return self._entity_data.get('Solidity')
        return "1"

    @property
    def excludednpc(self):
        if "excludednpc" in self._entity_data:
            return self._entity_data.get('excludednpc')
        return ""

    @property
    def invert_exclusion(self):
        if "invert_exclusion" in self._entity_data:
            return self._entity_data.get('invert_exclusion')
        return "0"

    @property
    def interactAs(self):
        if "interactAs" in self._entity_data:
            return self._entity_data.get('interactAs')
        return ""

    @property
    def solidbsp(self):
        if "solidbsp" in self._entity_data:
            return bool(self._entity_data.get('solidbsp'))
        return bool(0)

    @property
    def ScriptedMovement(self):
        if "ScriptedMovement" in self._entity_data:
            return bool(self._entity_data.get('ScriptedMovement'))
        return bool(0)

    @property
    def vrad_brush_cast_shadows(self):
        if "vrad_brush_cast_shadows" in self._entity_data:
            return bool(self._entity_data.get('vrad_brush_cast_shadows'))
        return bool(0)


class VGUIScreenBase(Targetname, Parentname):
    pass

    @property
    def panelname(self):
        if "panelname" in self._entity_data:
            return self._entity_data.get('panelname')
        return None

    @property
    def overlaymaterial(self):
        if "overlaymaterial" in self._entity_data:
            return self._entity_data.get('overlaymaterial')
        return ""

    @property
    def width(self):
        if "width" in self._entity_data:
            return int(self._entity_data.get('width'))
        return int(32)

    @property
    def height(self):
        if "height" in self._entity_data:
            return int(self._entity_data.get('height'))
        return int(32)


class vgui_screen(VGUIScreenBase):
    pass


class vgui_slideshow_display(Targetname, Parentname):
    pass

    @property
    def displaytext(self):
        if "displaytext" in self._entity_data:
            return self._entity_data.get('displaytext')
        return ""

    @property
    def directory(self):
        if "directory" in self._entity_data:
            return self._entity_data.get('directory')
        return "slideshow"

    @property
    def minslidetime(self):
        if "minslidetime" in self._entity_data:
            return float(self._entity_data.get('minslidetime'))
        return float(0.5)

    @property
    def maxslidetime(self):
        if "maxslidetime" in self._entity_data:
            return float(self._entity_data.get('maxslidetime'))
        return float(0.5)

    @property
    def cycletype(self):
        if "cycletype" in self._entity_data:
            return self._entity_data.get('cycletype')
        return "0"

    @property
    def nolistrepeat(self):
        if "nolistrepeat" in self._entity_data:
            return self._entity_data.get('nolistrepeat')
        return "0"

    @property
    def width(self):
        if "width" in self._entity_data:
            return int(self._entity_data.get('width'))
        return int(256)

    @property
    def height(self):
        if "height" in self._entity_data:
            return int(self._entity_data.get('height'))
        return int(128)


class vgui_movie_display(Targetname, Parentname):
    pass

    @property
    def displaytext(self):
        if "displaytext" in self._entity_data:
            return self._entity_data.get('displaytext')
        return ""

    @property
    def moviefilename(self):
        if "moviefilename" in self._entity_data:
            return self._entity_data.get('moviefilename')
        return "media/"

    @property
    def groupname(self):
        if "groupname" in self._entity_data:
            return self._entity_data.get('groupname')
        return ""

    @property
    def looping(self):
        if "looping" in self._entity_data:
            return bool(self._entity_data.get('looping'))
        return bool(0)

    @property
    def width(self):
        if "width" in self._entity_data:
            return int(self._entity_data.get('width'))
        return int(256)

    @property
    def height(self):
        if "height" in self._entity_data:
            return int(self._entity_data.get('height'))
        return int(128)


class cycler(Targetname, Parentname, RenderFields, CanBeClientOnly):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Not Solid': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return None

    @property
    def skin(self):
        if "skin" in self._entity_data:
            return self._entity_data.get('skin')
        return "default"

    @property
    def testMode(self):
        if "testMode" in self._entity_data:
            return self._entity_data.get('testMode')
        return "0"

    @property
    def doClientSideAnimation(self):
        if "doClientSideAnimation" in self._entity_data:
            return bool(self._entity_data.get('doClientSideAnimation'))
        return bool(0)

    @property
    def sequenceName(self):
        if "sequenceName" in self._entity_data:
            return self._entity_data.get('sequenceName')
        return "idle"

    @property
    def sequenceName2(self):
        if "sequenceName2" in self._entity_data:
            return self._entity_data.get('sequenceName2')
        return ""

    @property
    def poseParameterName(self):
        if "poseParameterName" in self._entity_data:
            return self._entity_data.get('poseParameterName')
        return ""

    @property
    def layerSequence1(self):
        if "layerSequence1" in self._entity_data:
            return self._entity_data.get('layerSequence1')
        return ""

    @property
    def layerSequence2(self):
        if "layerSequence2" in self._entity_data:
            return self._entity_data.get('layerSequence2')
        return ""


class func_orator(Targetname, Parentname, RenderFields, Studiomodel):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Not Solid': (1, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def maxThenAnyDispatchDist(self):
        if "maxThenAnyDispatchDist" in self._entity_data:
            return float(self._entity_data.get('maxThenAnyDispatchDist'))
        return float(0)


class gibshooterbase(Targetname, Parentname):
    pass

    @property
    def angles(self):
        if "angles" in self._entity_data:
            return self._entity_data.get('angles')
        return "0 0 0"

    @property
    def m_iGibs(self):
        if "m_iGibs" in self._entity_data:
            return int(self._entity_data.get('m_iGibs'))
        return int(3)

    @property
    def delay(self):
        if "delay" in self._entity_data:
            return self._entity_data.get('delay')
        return "0"

    @property
    def gibangles(self):
        if "gibangles" in self._entity_data:
            return self._entity_data.get('gibangles')
        return "0 0 0"

    @property
    def gibanglevelocity(self):
        if "gibanglevelocity" in self._entity_data:
            return self._entity_data.get('gibanglevelocity')
        return "0"

    @property
    def m_flVelocity(self):
        if "m_flVelocity" in self._entity_data:
            return int(self._entity_data.get('m_flVelocity'))
        return int(200)

    @property
    def m_flVariance(self):
        if "m_flVariance" in self._entity_data:
            return self._entity_data.get('m_flVariance')
        return "0.15"

    @property
    def m_flGibLife(self):
        if "m_flGibLife" in self._entity_data:
            return self._entity_data.get('m_flGibLife')
        return "4"

    @property
    def lightingorigin(self):
        if "lightingorigin" in self._entity_data:
            return self._entity_data.get('lightingorigin')
        return ""

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Repeatable': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class env_beam(Targetname, Parentname, RenderFxChoices):
    pass

    @property
    def renderamt(self):
        if "renderamt" in self._entity_data:
            return int(self._entity_data.get('renderamt'))
        return int(100)

    @property
    def rendercolor(self):
        if "rendercolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('rendercolor'))
        return parse_int_vector("255 255 255")

    @property
    def Radius(self):
        if "Radius" in self._entity_data:
            return int(self._entity_data.get('Radius'))
        return int(256)

    @property
    def life(self):
        if "life" in self._entity_data:
            return self._entity_data.get('life')
        return "1"

    @property
    def BoltWidth(self):
        if "BoltWidth" in self._entity_data:
            return float(self._entity_data.get('BoltWidth'))
        return float(2)

    @property
    def NoiseAmplitude(self):
        if "NoiseAmplitude" in self._entity_data:
            return float(self._entity_data.get('NoiseAmplitude'))
        return float(0)

    @property
    def texture(self):
        if "texture" in self._entity_data:
            return self._entity_data.get('texture')
        return "sprites/laserbeam.spr"

    @property
    def TextureScroll(self):
        if "TextureScroll" in self._entity_data:
            return int(self._entity_data.get('TextureScroll'))
        return int(35)

    @property
    def framerate(self):
        if "framerate" in self._entity_data:
            return int(self._entity_data.get('framerate'))
        return int(0)

    @property
    def framestart(self):
        if "framestart" in self._entity_data:
            return int(self._entity_data.get('framestart'))
        return int(0)

    @property
    def StrikeTime(self):
        if "StrikeTime" in self._entity_data:
            return self._entity_data.get('StrikeTime')
        return "1"

    @property
    def damage(self):
        if "damage" in self._entity_data:
            return self._entity_data.get('damage')
        return "0"

    @property
    def LightningStart(self):
        if "LightningStart" in self._entity_data:
            return self._entity_data.get('LightningStart')
        return ""

    @property
    def LightningEnd(self):
        if "LightningEnd" in self._entity_data:
            return self._entity_data.get('LightningEnd')
        return ""

    @property
    def decalname(self):
        if "decalname" in self._entity_data:
            return self._entity_data.get('decalname')
        return "Bigshot"

    @property
    def HDRColorScale(self):
        if "HDRColorScale" in self._entity_data:
            return float(self._entity_data.get('HDRColorScale'))
        return float(1.0)

    @property
    def targetpoint(self):
        if "targetpoint" in self._entity_data:
            return parse_int_vector(self._entity_data.get('targetpoint'))
        return parse_int_vector("0 0 0")

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start On': (1, 0), 'Toggle': (2, 0), 'Random Strike': (4, 0), 'Ring': (8, 0),
                                   'StartSparks': (16, 0), 'EndSparks': (32, 0), 'Decal End': (64, 0),
                                   'Shade Start': (128, 0), 'Shade End': (256, 0), 'Taper Out': (512, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def TouchType(self):
        if "TouchType" in self._entity_data:
            return self._entity_data.get('TouchType')
        return "0"

    @property
    def ClipStyle(self):
        if "ClipStyle" in self._entity_data:
            return self._entity_data.get('ClipStyle')
        return "0"

    @property
    def filtername(self):
        if "filtername" in self._entity_data:
            return self._entity_data.get('filtername')
        return None


class env_beverage(Targetname, Parentname):
    pass

    @property
    def health(self):
        if "health" in self._entity_data:
            return int(self._entity_data.get('health'))
        return int(10)

    @property
    def beveragetype(self):
        if "beveragetype" in self._entity_data:
            return self._entity_data.get('beveragetype')
        return "0"


class env_funnel(Targetname, Parentname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Reverse': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class env_blood(Targetname, Parentname):
    pass

    @property
    def spraydir(self):
        if "spraydir" in self._entity_data:
            return parse_int_vector(self._entity_data.get('spraydir'))
        return parse_int_vector("0 0 0")

    @property
    def color(self):
        if "color" in self._entity_data:
            return self._entity_data.get('color')
        return "0"

    @property
    def amount(self):
        if "amount" in self._entity_data:
            return self._entity_data.get('amount')
        return "100"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Random Direction': (1, 0), 'Blood Stream': (2, 0), 'On Player': (4, 0),
                                   'Spray decals': (8, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class env_bubbles(Targetname, Parentname):
    pass

    @property
    def density(self):
        if "density" in self._entity_data:
            return int(self._entity_data.get('density'))
        return int(2)

    @property
    def frequency(self):
        if "frequency" in self._entity_data:
            return int(self._entity_data.get('frequency'))
        return int(2)

    @property
    def current(self):
        if "current" in self._entity_data:
            return int(self._entity_data.get('current'))
        return int(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start Off': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class env_explosion(Targetname, Parentname, RenderModeChoices):
    pass

    icon_sprite = "editor/env_explosion.vmat"

    @property
    def iMagnitude(self):
        if "iMagnitude" in self._entity_data:
            return int(self._entity_data.get('iMagnitude'))
        return int(100)

    @property
    def iRadiusOverride(self):
        if "iRadiusOverride" in self._entity_data:
            return int(self._entity_data.get('iRadiusOverride'))
        return int(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'No Damage': (1, 0), 'Repeatable': (2, 0), 'No Decal': (16, 0), 'No Sound': (64, 0),
                                   'Damage above water surface only': (8192, 0), 'Generic damage': (16384, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def explosion_type(self):
        if "explosion_type" in self._entity_data:
            return self._entity_data.get('explosion_type')
        return ""

    @property
    def explosion_custom_effect(self):
        if "explosion_custom_effect" in self._entity_data:
            return self._entity_data.get('explosion_custom_effect')
        return None

    @property
    def explosion_custom_sound(self):
        if "explosion_custom_sound" in self._entity_data:
            return self._entity_data.get('explosion_custom_sound')
        return ""

    @property
    def ignoredEntity(self):
        if "ignoredEntity" in self._entity_data:
            return self._entity_data.get('ignoredEntity')
        return None

    @property
    def ignoredClass(self):
        if "ignoredClass" in self._entity_data:
            return int(self._entity_data.get('ignoredClass'))
        return int(0)


class env_smoketrail(Targetname, Parentname):
    pass

    @property
    def opacity(self):
        if "opacity" in self._entity_data:
            return float(self._entity_data.get('opacity'))
        return float(0.75)

    @property
    def spawnrate(self):
        if "spawnrate" in self._entity_data:
            return float(self._entity_data.get('spawnrate'))
        return float(20)

    @property
    def lifetime(self):
        if "lifetime" in self._entity_data:
            return float(self._entity_data.get('lifetime'))
        return float(5.0)

    @property
    def startcolor(self):
        if "startcolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('startcolor'))
        return parse_int_vector("192 192 192")

    @property
    def endcolor(self):
        if "endcolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('endcolor'))
        return parse_int_vector("160 160 160")

    @property
    def emittime(self):
        if "emittime" in self._entity_data:
            return float(self._entity_data.get('emittime'))
        return float(0)

    @property
    def minspeed(self):
        if "minspeed" in self._entity_data:
            return float(self._entity_data.get('minspeed'))
        return float(10)

    @property
    def maxspeed(self):
        if "maxspeed" in self._entity_data:
            return float(self._entity_data.get('maxspeed'))
        return float(20)

    @property
    def mindirectedspeed(self):
        if "mindirectedspeed" in self._entity_data:
            return float(self._entity_data.get('mindirectedspeed'))
        return float(0)

    @property
    def maxdirectedspeed(self):
        if "maxdirectedspeed" in self._entity_data:
            return float(self._entity_data.get('maxdirectedspeed'))
        return float(0)

    @property
    def startsize(self):
        if "startsize" in self._entity_data:
            return float(self._entity_data.get('startsize'))
        return float(15)

    @property
    def endsize(self):
        if "endsize" in self._entity_data:
            return float(self._entity_data.get('endsize'))
        return float(50)

    @property
    def spawnradius(self):
        if "spawnradius" in self._entity_data:
            return float(self._entity_data.get('spawnradius'))
        return float(15)

    @property
    def firesprite(self):
        if "firesprite" in self._entity_data:
            return self._entity_data.get('firesprite')
        return "sprites/firetrail.spr"

    @property
    def smokesprite(self):
        if "smokesprite" in self._entity_data:
            return self._entity_data.get('smokesprite')
        return "sprites/whitepuff.spr"


class env_physexplosion(Targetname, Parentname):
    pass

    icon_sprite = "editor/env_physexplosion.vmat"

    @property
    def magnitude(self):
        if "magnitude" in self._entity_data:
            return self._entity_data.get('magnitude')
        return "100"

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return self._entity_data.get('radius')
        return "0"

    @property
    def targetentityname(self):
        if "targetentityname" in self._entity_data:
            return self._entity_data.get('targetentityname')
        return ""

    @property
    def explodeonspawn(self):
        if "explodeonspawn" in self._entity_data:
            return bool(self._entity_data.get('explodeonspawn'))
        return bool(None)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'No Damage - Only Force': (1, 1), 'Push players': (2, 0),
                                   'Push radially - not as a sphere': (4, 0), 'Test LOS before pushing': (8, 0),
                                   'Disorient player if pushed': (16, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def inner_radius(self):
        if "inner_radius" in self._entity_data:
            return float(self._entity_data.get('inner_radius'))
        return float(0)

    @property
    def pushscale(self):
        if "pushscale" in self._entity_data:
            return float(self._entity_data.get('pushscale'))
        return float(1)

    @property
    def ConvertToDebrisWhenPossible(self):
        if "ConvertToDebrisWhenPossible" in self._entity_data:
            return self._entity_data.get('ConvertToDebrisWhenPossible')
        return "0"


class env_physimpact(Targetname, Parentname):
    pass

    icon_sprite = "editor/env_physexplosion.vmat"

    @property
    def angles(self):
        if "angles" in self._entity_data:
            return self._entity_data.get('angles')
        return "0 0 0"

    @property
    def magnitude(self):
        if "magnitude" in self._entity_data:
            return int(self._entity_data.get('magnitude'))
        return int(100)

    @property
    def distance(self):
        if "distance" in self._entity_data:
            return int(self._entity_data.get('distance'))
        return int(0)

    @property
    def directionentityname(self):
        if "directionentityname" in self._entity_data:
            return self._entity_data.get('directionentityname')
        return ""

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'No fall-off': (1, 0), 'Infinite Length': (2, 0), 'Ignore Mass': (4, 0),
                                   'Ignore Surface Normal When Applying Force': (8, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class env_fire(Targetname, Parentname, EnableDisable):
    pass

    icon_sprite = "editor/env_fire"

    @property
    def health(self):
        if "health" in self._entity_data:
            return int(self._entity_data.get('health'))
        return int(30)

    @property
    def firesize(self):
        if "firesize" in self._entity_data:
            return int(self._entity_data.get('firesize'))
        return int(64)

    @property
    def fireattack(self):
        if "fireattack" in self._entity_data:
            return int(self._entity_data.get('fireattack'))
        return int(4)

    @property
    def firetype(self):
        if "firetype" in self._entity_data:
            return self._entity_data.get('firetype')
        return "0"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Infinite Duration': (1, 0), 'Smokeless': (2, 0), 'Start On': (4, 0),
                                   'Start Full': (8, 0), "Don't drop": (16, 0), 'No glow': (32, 0),
                                   'Delete when out': (128, 0), 'Visible from above': (256, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def ignitionpoint(self):
        if "ignitionpoint" in self._entity_data:
            return float(self._entity_data.get('ignitionpoint'))
        return float(32)

    @property
    def damagescale(self):
        if "damagescale" in self._entity_data:
            return float(self._entity_data.get('damagescale'))
        return float(1.0)


class env_firesource(Targetname, Parentname):
    pass

    icon_sprite = "editor/env_firesource"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start On': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def fireradius(self):
        if "fireradius" in self._entity_data:
            return float(self._entity_data.get('fireradius'))
        return float(128)

    @property
    def firedamage(self):
        if "firedamage" in self._entity_data:
            return float(self._entity_data.get('firedamage'))
        return float(10)


class env_firesensor(Targetname, Parentname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start On': (1, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def fireradius(self):
        if "fireradius" in self._entity_data:
            return float(self._entity_data.get('fireradius'))
        return float(128)

    @property
    def heatlevel(self):
        if "heatlevel" in self._entity_data:
            return float(self._entity_data.get('heatlevel'))
        return float(32)

    @property
    def heattime(self):
        if "heattime" in self._entity_data:
            return float(self._entity_data.get('heattime'))
        return float(0)


class env_entity_igniter(Targetname):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def lifetime(self):
        if "lifetime" in self._entity_data:
            return float(self._entity_data.get('lifetime'))
        return float(10)


class env_fog_controller(Targetname, SystemLevelChoice):
    pass

    icon_sprite = "materials/editor/env_fog_controller.vmat"

    @property
    def fogenable(self):
        if "fogenable" in self._entity_data:
            return bool(self._entity_data.get('fogenable'))
        return bool(1)

    @property
    def fogblend(self):
        if "fogblend" in self._entity_data:
            return bool(self._entity_data.get('fogblend'))
        return bool(0)

    @property
    def use_angles(self):
        if "use_angles" in self._entity_data:
            return bool(self._entity_data.get('use_angles'))
        return bool(0)

    @property
    def fogcolor(self):
        if "fogcolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('fogcolor'))
        return parse_int_vector("255 255 255")

    @property
    def fogcolor2(self):
        if "fogcolor2" in self._entity_data:
            return parse_int_vector(self._entity_data.get('fogcolor2'))
        return parse_int_vector("255 255 255")

    @property
    def fogdir(self):
        if "fogdir" in self._entity_data:
            return self._entity_data.get('fogdir')
        return "1 0 0"

    @property
    def fogstart(self):
        if "fogstart" in self._entity_data:
            return self._entity_data.get('fogstart')
        return "500.0"

    @property
    def fogend(self):
        if "fogend" in self._entity_data:
            return self._entity_data.get('fogend')
        return "2000.0"

    @property
    def fogmaxdensity(self):
        if "fogmaxdensity" in self._entity_data:
            return float(self._entity_data.get('fogmaxdensity'))
        return float(1)

    @property
    def fogexponent(self):
        if "fogexponent" in self._entity_data:
            return float(self._entity_data.get('fogexponent'))
        return float(2)

    @property
    def foglerptime(self):
        if "foglerptime" in self._entity_data:
            return float(self._entity_data.get('foglerptime'))
        return float(0)

    @property
    def farz(self):
        if "farz" in self._entity_data:
            return self._entity_data.get('farz')
        return "-1"

    @property
    def HDRColorScale(self):
        if "HDRColorScale" in self._entity_data:
            return float(self._entity_data.get('HDRColorScale'))
        return float(1.0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Master (Has priority if multiple env_fog_controllers exist)': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class postprocess_controller(Targetname):
    pass

    icon_sprite = "editor/postprocess_controller.vmat"

    @property
    def localcontraststrength(self):
        if "localcontraststrength" in self._entity_data:
            return float(self._entity_data.get('localcontraststrength'))
        return float(0)

    @property
    def localcontrastedgestrength(self):
        if "localcontrastedgestrength" in self._entity_data:
            return float(self._entity_data.get('localcontrastedgestrength'))
        return float(0)

    @property
    def vignettestart(self):
        if "vignettestart" in self._entity_data:
            return float(self._entity_data.get('vignettestart'))
        return float(1)

    @property
    def vignetteend(self):
        if "vignetteend" in self._entity_data:
            return float(self._entity_data.get('vignetteend'))
        return float(2)

    @property
    def vignetteblurstrength(self):
        if "vignetteblurstrength" in self._entity_data:
            return float(self._entity_data.get('vignetteblurstrength'))
        return float(0)

    @property
    def fadetoblackstrength(self):
        if "fadetoblackstrength" in self._entity_data:
            return float(self._entity_data.get('fadetoblackstrength'))
        return float(0)

    @property
    def grainstrength(self):
        if "grainstrength" in self._entity_data:
            return float(self._entity_data.get('grainstrength'))
        return float(1)

    @property
    def topvignettestrength(self):
        if "topvignettestrength" in self._entity_data:
            return float(self._entity_data.get('topvignettestrength'))
        return float(1)

    @property
    def fadetime(self):
        if "fadetime" in self._entity_data:
            return float(self._entity_data.get('fadetime'))
        return float(2)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Master (Has priority if multiple postprocess_controllers exist)': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class env_laser(Targetname, Parentname, RenderFxChoices):
    pass

    @property
    def LaserTarget(self):
        if "LaserTarget" in self._entity_data:
            return self._entity_data.get('LaserTarget')
        return None

    @property
    def renderamt(self):
        if "renderamt" in self._entity_data:
            return int(self._entity_data.get('renderamt'))
        return int(100)

    @property
    def rendercolor(self):
        if "rendercolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('rendercolor'))
        return parse_int_vector("255 255 255")

    @property
    def width(self):
        if "width" in self._entity_data:
            return float(self._entity_data.get('width'))
        return float(2)

    @property
    def NoiseAmplitude(self):
        if "NoiseAmplitude" in self._entity_data:
            return int(self._entity_data.get('NoiseAmplitude'))
        return int(0)

    @property
    def texture(self):
        if "texture" in self._entity_data:
            return self._entity_data.get('texture')
        return "sprites/laserbeam.spr"

    @property
    def EndSprite(self):
        if "EndSprite" in self._entity_data:
            return self._entity_data.get('EndSprite')
        return ""

    @property
    def TextureScroll(self):
        if "TextureScroll" in self._entity_data:
            return int(self._entity_data.get('TextureScroll'))
        return int(35)

    @property
    def framestart(self):
        if "framestart" in self._entity_data:
            return int(self._entity_data.get('framestart'))
        return int(0)

    @property
    def damage(self):
        if "damage" in self._entity_data:
            return self._entity_data.get('damage')
        return "100"

    @property
    def dissolvetype(self):
        if "dissolvetype" in self._entity_data:
            return self._entity_data.get('dissolvetype')
        return "None"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start On': (1, 0), 'StartSparks': (16, 0), 'EndSparks': (32, 0),
                                   'Decal End': (64, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class env_message(Targetname):
    pass

    @property
    def message(self):
        if "message" in self._entity_data:
            return self._entity_data.get('message')
        return None

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Play Once': (1, 0), 'All Clients': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def messagesound(self):
        if "messagesound" in self._entity_data:
            return self._entity_data.get('messagesound')
        return ""

    @property
    def messagevolume(self):
        if "messagevolume" in self._entity_data:
            return self._entity_data.get('messagevolume')
        return "10"

    @property
    def messageattenuation(self):
        if "messageattenuation" in self._entity_data:
            return self._entity_data.get('messageattenuation')
        return "0"


class env_hudhint(Targetname):
    pass

    @property
    def message(self):
        if "message" in self._entity_data:
            return self._entity_data.get('message')
        return ""


class env_shake(Targetname, Parentname):
    pass

    icon_sprite = "editor/env_shake.vmat"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'GlobalShake': (1, 0), 'In Air': (4, 0), 'Physics': (8, 0), 'Ropes': (16, 0),
                                   "DON'T shake view (for shaking ropes or physics only)": (32, 0),
                                   "DON'T Rumble Controller": (64, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def limittoentity(self):
        if "limittoentity" in self._entity_data:
            return self._entity_data.get('limittoentity')
        return ""

    @property
    def amplitude(self):
        if "amplitude" in self._entity_data:
            return float(self._entity_data.get('amplitude'))
        return float(4)

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return float(self._entity_data.get('radius'))
        return float(500)

    @property
    def duration(self):
        if "duration" in self._entity_data:
            return float(self._entity_data.get('duration'))
        return float(1)

    @property
    def frequency(self):
        if "frequency" in self._entity_data:
            return float(self._entity_data.get('frequency'))
        return float(2.5)


class env_tilt(Targetname, Parentname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'GlobalTilt': (1, 0), 'Ease in/out': (128, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return float(self._entity_data.get('radius'))
        return float(500)

    @property
    def duration(self):
        if "duration" in self._entity_data:
            return float(self._entity_data.get('duration'))
        return float(1)

    @property
    def tilttime(self):
        if "tilttime" in self._entity_data:
            return float(self._entity_data.get('tilttime'))
        return float(2.5)


class env_viewpunch(Targetname, Parentname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Punch all players (ignore radius)': (1, 0),
                                   'Punch players in the air': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def punchangle(self):
        if "punchangle" in self._entity_data:
            return parse_int_vector(self._entity_data.get('punchangle'))
        return parse_int_vector("0 0 90")

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return float(self._entity_data.get('radius'))
        return float(500)


class env_rotorwash_emitter(Targetname, Parentname):
    pass

    @property
    def altitude(self):
        if "altitude" in self._entity_data:
            return float(self._entity_data.get('altitude'))
        return float(1024)


class gibshooter(gibshooterbase):
    pass

    icon_sprite = "editor/gibshooter.vmat"


class env_shooter(gibshooterbase, RenderFields):
    pass

    icon_sprite = "editor/env_shooter.vmat"

    @property
    def shootmodel(self):
        if "shootmodel" in self._entity_data:
            return self._entity_data.get('shootmodel')
        return ""

    @property
    def shootsounds(self):
        if "shootsounds" in self._entity_data:
            return self._entity_data.get('shootsounds')
        return "-1"

    @property
    def simulation(self):
        if "simulation" in self._entity_data:
            return self._entity_data.get('simulation')
        return "0"

    @property
    def skin(self):
        if "skin" in self._entity_data:
            return int(self._entity_data.get('skin'))
        return int(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'On fire': (2, 0), 'strict remove after lifetime': (4, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def nogibshadows(self):
        if "nogibshadows" in self._entity_data:
            return bool(self._entity_data.get('nogibshadows'))
        return bool(0)

    @property
    def gibgravityscale(self):
        if "gibgravityscale" in self._entity_data:
            return float(self._entity_data.get('gibgravityscale'))
        return float(1)

    @property
    def massoverride(self):
        if "massoverride" in self._entity_data:
            return float(self._entity_data.get('massoverride'))
        return float(0)


class env_rotorshooter(gibshooterbase, RenderFields):
    pass

    icon_sprite = "editor/env_shooter.vmat"

    @property
    def shootmodel(self):
        if "shootmodel" in self._entity_data:
            return self._entity_data.get('shootmodel')
        return ""

    @property
    def shootsounds(self):
        if "shootsounds" in self._entity_data:
            return self._entity_data.get('shootsounds')
        return "-1"

    @property
    def simulation(self):
        if "simulation" in self._entity_data:
            return self._entity_data.get('simulation')
        return "0"

    @property
    def skin(self):
        if "skin" in self._entity_data:
            return int(self._entity_data.get('skin'))
        return int(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'On fire': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def rotortime(self):
        if "rotortime" in self._entity_data:
            return float(self._entity_data.get('rotortime'))
        return float(1)

    @property
    def rotortimevariance(self):
        if "rotortimevariance" in self._entity_data:
            return float(self._entity_data.get('rotortimevariance'))
        return float(0.3)


class env_soundscape_proxy(Targetname, Parentname):
    pass

    icon_sprite = "editor/env_soundscape.vmat"

    @property
    def MainSoundscapeName(self):
        if "MainSoundscapeName" in self._entity_data:
            return self._entity_data.get('MainSoundscapeName')
        return ""

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return int(self._entity_data.get('radius'))
        return int(128)


class snd_soundscape_proxy(env_soundscape_proxy):
    pass

    icon_sprite = "editor/env_soundscape.vmat"


class env_soundscape(Targetname, Parentname, EnableDisable):
    pass

    icon_sprite = "editor/env_soundscape.vmat"

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return int(self._entity_data.get('radius'))
        return int(128)

    @property
    def soundscape(self):
        if "soundscape" in self._entity_data:
            return self._entity_data.get('soundscape')
        return "Nothing"

    @property
    def position0(self):
        if "position0" in self._entity_data:
            return self._entity_data.get('position0')
        return ""

    @property
    def position1(self):
        if "position1" in self._entity_data:
            return self._entity_data.get('position1')
        return ""

    @property
    def position2(self):
        if "position2" in self._entity_data:
            return self._entity_data.get('position2')
        return ""

    @property
    def position3(self):
        if "position3" in self._entity_data:
            return self._entity_data.get('position3')
        return ""

    @property
    def position4(self):
        if "position4" in self._entity_data:
            return self._entity_data.get('position4')
        return ""

    @property
    def position5(self):
        if "position5" in self._entity_data:
            return self._entity_data.get('position5')
        return ""

    @property
    def position6(self):
        if "position6" in self._entity_data:
            return self._entity_data.get('position6')
        return ""

    @property
    def position7(self):
        if "position7" in self._entity_data:
            return self._entity_data.get('position7')
        return ""


class snd_soundscape(env_soundscape):
    pass

    icon_sprite = "editor/env_soundscape.vmat"


class env_soundscape_triggerable(env_soundscape):
    pass

    icon_sprite = "editor/env_soundscape.vmat"


class snd_soundscape_triggerable(env_soundscape_triggerable):
    pass

    icon_sprite = "editor/env_soundscape.vmat"


class env_spark(Targetname, Parentname):
    pass

    icon_sprite = "editor/env_spark.vmat"

    @property
    def MaxDelay(self):
        if "MaxDelay" in self._entity_data:
            return self._entity_data.get('MaxDelay')
        return "0"

    @property
    def SparkType(self):
        if "SparkType" in self._entity_data:
            return self._entity_data.get('SparkType')
        return "1"

    @property
    def Magnitude(self):
        if "Magnitude" in self._entity_data:
            return self._entity_data.get('Magnitude')
        return "1"

    @property
    def TrailLength(self):
        if "TrailLength" in self._entity_data:
            return self._entity_data.get('TrailLength')
        return "1"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start ON': (64, 0), 'Glow': (128, 0), 'Silent': (256, 0),
                                   'Directional': (512, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class env_sprite(Targetname, Parentname, RenderFields):
    pass

    @property
    def framerate(self):
        if "framerate" in self._entity_data:
            return self._entity_data.get('framerate')
        return "10.0"

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "sprites/glow01.spr"

    @property
    def scale(self):
        if "scale" in self._entity_data:
            return self._entity_data.get('scale')
        return ""

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start on': (1, 0), 'Play Once': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def GlowProxySize(self):
        if "GlowProxySize" in self._entity_data:
            return float(self._entity_data.get('GlowProxySize'))
        return float(2.0)

    @property
    def HDRColorScale(self):
        if "HDRColorScale" in self._entity_data:
            return float(self._entity_data.get('HDRColorScale'))
        return float(0.7)


class env_sprite_oriented(env_sprite):
    pass

    @property
    def framerate(self):
        if "framerate" in self._entity_data:
            return self._entity_data.get('framerate')
        return "10.0"

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "sprites/glow01.spr"

    @property
    def scale(self):
        if "scale" in self._entity_data:
            return self._entity_data.get('scale')
        return ""

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start on': (1, 0), 'Play Once': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def GlowProxySize(self):
        if "GlowProxySize" in self._entity_data:
            return float(self._entity_data.get('GlowProxySize'))
        return float(2.0)

    @property
    def HDRColorScale(self):
        if "HDRColorScale" in self._entity_data:
            return float(self._entity_data.get('HDRColorScale'))
        return float(1.0)


class BaseEnvWind(Targetname):
    pass

    @property
    def minwind(self):
        if "minwind" in self._entity_data:
            return int(self._entity_data.get('minwind'))
        return int(20)

    @property
    def maxwind(self):
        if "maxwind" in self._entity_data:
            return int(self._entity_data.get('maxwind'))
        return int(50)

    @property
    def windradius(self):
        if "windradius" in self._entity_data:
            return float(self._entity_data.get('windradius'))
        return float(-1)

    @property
    def mingust(self):
        if "mingust" in self._entity_data:
            return int(self._entity_data.get('mingust'))
        return int(100)

    @property
    def maxgust(self):
        if "maxgust" in self._entity_data:
            return int(self._entity_data.get('maxgust'))
        return int(250)

    @property
    def mingustdelay(self):
        if "mingustdelay" in self._entity_data:
            return int(self._entity_data.get('mingustdelay'))
        return int(10)

    @property
    def maxgustdelay(self):
        if "maxgustdelay" in self._entity_data:
            return int(self._entity_data.get('maxgustdelay'))
        return int(20)

    @property
    def gustduration(self):
        if "gustduration" in self._entity_data:
            return int(self._entity_data.get('gustduration'))
        return int(5)

    @property
    def gustdirchange(self):
        if "gustdirchange" in self._entity_data:
            return int(self._entity_data.get('gustdirchange'))
        return int(20)


class env_wind(BaseEnvWind):
    pass

    icon_sprite = "editor/env_wind.vmat"


class env_wind_clientside(BaseEnvWind):
    pass

    icon_sprite = "editor/env_wind.vmat"


class sky_camera:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def scale(self):
        if "scale" in self._entity_data:
            return int(self._entity_data.get('scale'))
        return int(16)

    @property
    def fogenable(self):
        if "fogenable" in self._entity_data:
            return bool(self._entity_data.get('fogenable'))
        return bool(0)

    @property
    def fogblend(self):
        if "fogblend" in self._entity_data:
            return bool(self._entity_data.get('fogblend'))
        return bool(0)

    @property
    def use_angles(self):
        if "use_angles" in self._entity_data:
            return bool(self._entity_data.get('use_angles'))
        return bool(0)

    @property
    def clip_3D_skybox_near_to_world_far(self):
        if "clip_3D_skybox_near_to_world_far" in self._entity_data:
            return bool(self._entity_data.get('clip_3D_skybox_near_to_world_far'))
        return bool(0)

    @property
    def clip_3D_skybox_near_to_world_far_offset(self):
        if "clip_3D_skybox_near_to_world_far_offset" in self._entity_data:
            return self._entity_data.get('clip_3D_skybox_near_to_world_far_offset')
        return "0.0"

    @property
    def fogcolor(self):
        if "fogcolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('fogcolor'))
        return parse_int_vector("255 255 255")

    @property
    def fogcolor2(self):
        if "fogcolor2" in self._entity_data:
            return parse_int_vector(self._entity_data.get('fogcolor2'))
        return parse_int_vector("255 255 255")

    @property
    def fogdir(self):
        if "fogdir" in self._entity_data:
            return self._entity_data.get('fogdir')
        return "1 0 0"

    @property
    def fogstart(self):
        if "fogstart" in self._entity_data:
            return self._entity_data.get('fogstart')
        return "500.0"

    @property
    def fogend(self):
        if "fogend" in self._entity_data:
            return self._entity_data.get('fogend')
        return "2000.0"

    @property
    def fogmaxdensity(self):
        if "fogmaxdensity" in self._entity_data:
            return float(self._entity_data.get('fogmaxdensity'))
        return float(1)

    @property
    def HDRColorScale(self):
        if "HDRColorScale" in self._entity_data:
            return float(self._entity_data.get('HDRColorScale'))
        return float(1.0)

    @property
    def SkyboxSlot(self):
        if "SkyboxSlot" in self._entity_data:
            return self._entity_data.get('SkyboxSlot')
        return ""


class BaseSpeaker(Targetname, ResponseContext):
    pass

    @property
    def delaymin(self):
        if "delaymin" in self._entity_data:
            return self._entity_data.get('delaymin')
        return "15"

    @property
    def delaymax(self):
        if "delaymax" in self._entity_data:
            return self._entity_data.get('delaymax')
        return "135"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start Silent': (1, 0), 'Play Everywhere': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def rulescript(self):
        if "rulescript" in self._entity_data:
            return self._entity_data.get('rulescript')
        return ""

    @property
    def concept(self):
        if "concept" in self._entity_data:
            return self._entity_data.get('concept')
        return ""


class game_weapon_manager(Targetname):
    pass

    @property
    def weaponname(self):
        if "weaponname" in self._entity_data:
            return self._entity_data.get('weaponname')
        return ""

    @property
    def maxpieces(self):
        if "maxpieces" in self._entity_data:
            return int(self._entity_data.get('maxpieces'))
        return int(0)

    @property
    def ammomod(self):
        if "ammomod" in self._entity_data:
            return float(self._entity_data.get('ammomod'))
        return float(1)


class game_end(Targetname):
    pass

    icon_sprite = "editor/game_end.vmat"

    @property
    def master(self):
        if "master" in self._entity_data:
            return self._entity_data.get('master')
        return None


class game_player_equip(Targetname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Use Only': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def master(self):
        if "master" in self._entity_data:
            return self._entity_data.get('master')
        return None


class game_player_team(Targetname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Remove On fire': (1, 0), 'Kill Player': (2, 0), 'Gib Player': (4, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def master(self):
        if "master" in self._entity_data:
            return self._entity_data.get('master')
        return None


class game_score(Targetname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Allow Negative': (1, 0), 'Team Points': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def points(self):
        if "points" in self._entity_data:
            return int(self._entity_data.get('points'))
        return int(1)

    @property
    def master(self):
        if "master" in self._entity_data:
            return self._entity_data.get('master')
        return None


class game_text(Targetname):
    pass

    icon_sprite = "editor/game_text.vmat"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'All Players': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def message(self):
        if "message" in self._entity_data:
            return self._entity_data.get('message')
        return ""

    @property
    def x(self):
        if "x" in self._entity_data:
            return self._entity_data.get('x')
        return "-1"

    @property
    def y(self):
        if "y" in self._entity_data:
            return self._entity_data.get('y')
        return "-1"

    @property
    def effect(self):
        if "effect" in self._entity_data:
            return self._entity_data.get('effect')
        return "0"

    @property
    def color(self):
        if "color" in self._entity_data:
            return parse_int_vector(self._entity_data.get('color'))
        return parse_int_vector("100 100 100")

    @property
    def color2(self):
        if "color2" in self._entity_data:
            return parse_int_vector(self._entity_data.get('color2'))
        return parse_int_vector("240 110 0")

    @property
    def fadein(self):
        if "fadein" in self._entity_data:
            return self._entity_data.get('fadein')
        return "1.5"

    @property
    def fadeout(self):
        if "fadeout" in self._entity_data:
            return self._entity_data.get('fadeout')
        return "0.5"

    @property
    def holdtime(self):
        if "holdtime" in self._entity_data:
            return self._entity_data.get('holdtime')
        return "1.2"

    @property
    def fxtime(self):
        if "fxtime" in self._entity_data:
            return self._entity_data.get('fxtime')
        return "0.25"

    @property
    def channel(self):
        if "channel" in self._entity_data:
            return self._entity_data.get('channel')
        return "1"

    @property
    def master(self):
        if "master" in self._entity_data:
            return self._entity_data.get('master')
        return None


class point_enable_motion_fixup(Parentname):
    pass


class point_message(Targetname, Parentname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start Disabled': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def message(self):
        if "message" in self._entity_data:
            return self._entity_data.get('message')
        return None

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return int(self._entity_data.get('radius'))
        return int(128)

    @property
    def developeronly(self):
        if "developeronly" in self._entity_data:
            return bool(self._entity_data.get('developeronly'))
        return bool(0)


class point_spotlight(Targetname, Parentname, RenderFields):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start On': (1, 1), 'No Dynamic Light': (2, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def spotlightlength(self):
        if "spotlightlength" in self._entity_data:
            return int(self._entity_data.get('spotlightlength'))
        return int(500)

    @property
    def spotlightwidth(self):
        if "spotlightwidth" in self._entity_data:
            return int(self._entity_data.get('spotlightwidth'))
        return int(50)

    @property
    def rendercolor(self):
        if "rendercolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('rendercolor'))
        return parse_int_vector("255 255 255")

    @property
    def HaloScale(self):
        if "HaloScale" in self._entity_data:
            return float(self._entity_data.get('HaloScale'))
        return float(60)

    @property
    def HDRColorScale(self):
        if "HDRColorScale" in self._entity_data:
            return float(self._entity_data.get('HDRColorScale'))
        return float(1.0)


class point_tesla(Targetname, Parentname):
    pass

    @property
    def m_SourceEntityName(self):
        if "m_SourceEntityName" in self._entity_data:
            return self._entity_data.get('m_SourceEntityName')
        return ""

    @property
    def m_SoundName(self):
        if "m_SoundName" in self._entity_data:
            return self._entity_data.get('m_SoundName')
        return "DoSpark"

    @property
    def texture(self):
        if "texture" in self._entity_data:
            return self._entity_data.get('texture')
        return "sprites/physbeam.vmat"

    @property
    def m_Color(self):
        if "m_Color" in self._entity_data:
            return parse_int_vector(self._entity_data.get('m_Color'))
        return parse_int_vector("255 255 255")

    @property
    def m_flRadius(self):
        if "m_flRadius" in self._entity_data:
            return int(self._entity_data.get('m_flRadius'))
        return int(200)

    @property
    def beamcount_min(self):
        if "beamcount_min" in self._entity_data:
            return int(self._entity_data.get('beamcount_min'))
        return int(6)

    @property
    def beamcount_max(self):
        if "beamcount_max" in self._entity_data:
            return int(self._entity_data.get('beamcount_max'))
        return int(8)

    @property
    def thick_min(self):
        if "thick_min" in self._entity_data:
            return self._entity_data.get('thick_min')
        return "4"

    @property
    def thick_max(self):
        if "thick_max" in self._entity_data:
            return self._entity_data.get('thick_max')
        return "5"

    @property
    def lifetime_min(self):
        if "lifetime_min" in self._entity_data:
            return self._entity_data.get('lifetime_min')
        return "0.3"

    @property
    def lifetime_max(self):
        if "lifetime_max" in self._entity_data:
            return self._entity_data.get('lifetime_max')
        return "0.3"

    @property
    def interval_min(self):
        if "interval_min" in self._entity_data:
            return self._entity_data.get('interval_min')
        return "0.5"

    @property
    def interval_max(self):
        if "interval_max" in self._entity_data:
            return self._entity_data.get('interval_max')
        return "2"


class point_clientcommand(Targetname):
    pass


class point_servercommand(Targetname):
    pass


class point_broadcastclientcommand(Targetname):
    pass


class point_bonusmaps_accessor(Targetname):
    pass

    @property
    def filename(self):
        if "filename" in self._entity_data:
            return self._entity_data.get('filename')
        return ""

    @property
    def mapname(self):
        if "mapname" in self._entity_data:
            return self._entity_data.get('mapname')
        return ""


class game_ui(Targetname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Freeze Player': (32, 1), 'Hide Weapon': (64, 1), '+Use Deactivates': (128, 1),
                                   'Jump Deactivates': (256, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def FieldOfView(self):
        if "FieldOfView" in self._entity_data:
            return float(self._entity_data.get('FieldOfView'))
        return float(-1.0)


class point_entity_finder(Targetname):
    pass

    @property
    def filtername(self):
        if "filtername" in self._entity_data:
            return self._entity_data.get('filtername')
        return None

    @property
    def referencename(self):
        if "referencename" in self._entity_data:
            return self._entity_data.get('referencename')
        return ""

    @property
    def Method(self):
        if "Method" in self._entity_data:
            return self._entity_data.get('Method')
        return "0"


class game_zone_player(Targetname, Parentname):
    pass


class info_projecteddecal(Targetname):
    pass

    @property
    def texture(self):
        if "texture" in self._entity_data:
            return self._entity_data.get('texture')
        return None

    @property
    def Distance(self):
        if "Distance" in self._entity_data:
            return float(self._entity_data.get('Distance'))
        return float(64)


class info_no_dynamic_shadow:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def sides(self):
        if "sides" in self._entity_data:
            return parse_int_vector(self._entity_data.get('sides'))
        return parse_int_vector("None")


class info_player_start(Targetname, EnableDisable, PlayerClass):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Master (Has priority if multiple info_player_starts exist)': (1, 0),
                                   'VR Anchor location (vs player location)': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class info_overlay:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def targetname(self):
        if "targetname" in self._entity_data:
            return self._entity_data.get('targetname')
        return None

    @property
    def material(self):
        if "material" in self._entity_data:
            return self._entity_data.get('material')
        return "materials/decaltests/red_paint_decal.vmat"

    @property
    def RenderOrder(self):
        if "RenderOrder" in self._entity_data:
            return int(self._entity_data.get('RenderOrder'))
        return int(0)

    @property
    def sequence(self):
        if "sequence" in self._entity_data:
            return int(self._entity_data.get('sequence'))
        return int(-1)

    @property
    def width(self):
        if "width" in self._entity_data:
            return float(self._entity_data.get('width'))
        return float(-1.0)

    @property
    def height(self):
        if "height" in self._entity_data:
            return float(self._entity_data.get('height'))
        return float(-1.0)

    @property
    def depth(self):
        if "depth" in self._entity_data:
            return float(self._entity_data.get('depth'))
        return float(1.0)

    @property
    def startu(self):
        if "startu" in self._entity_data:
            return float(self._entity_data.get('startu'))
        return float(0.0)

    @property
    def endu(self):
        if "endu" in self._entity_data:
            return float(self._entity_data.get('endu'))
        return float(1.0)

    @property
    def startv(self):
        if "startv" in self._entity_data:
            return float(self._entity_data.get('startv'))
        return float(0.0)

    @property
    def endv(self):
        if "endv" in self._entity_data:
            return float(self._entity_data.get('endv'))
        return float(1.0)

    @property
    def centeru(self):
        if "centeru" in self._entity_data:
            return float(self._entity_data.get('centeru'))
        return float(0.5)

    @property
    def centerv(self):
        if "centerv" in self._entity_data:
            return float(self._entity_data.get('centerv'))
        return float(0.5)

    @property
    def fademindist(self):
        if "fademindist" in self._entity_data:
            return float(self._entity_data.get('fademindist'))
        return float(-1)

    @property
    def fademaxdist(self):
        if "fademaxdist" in self._entity_data:
            return float(self._entity_data.get('fademaxdist'))
        return float(0)

    @property
    def rendercolor(self):
        if "rendercolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('rendercolor'))
        return parse_int_vector("255 255 255")

    @property
    def disablelowviolence(self):
        if "disablelowviolence" in self._entity_data:
            return self._entity_data.get('disablelowviolence')
        return "0"


class info_overlay_transition:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def material(self):
        if "material" in self._entity_data:
            return self._entity_data.get('material')
        return None

    @property
    def sides(self):
        if "sides" in self._entity_data:
            return parse_int_vector(self._entity_data.get('sides'))
        return parse_int_vector("None")

    @property
    def sides2(self):
        if "sides2" in self._entity_data:
            return parse_int_vector(self._entity_data.get('sides2'))
        return parse_int_vector("None")

    @property
    def LengthTexcoordStart(self):
        if "LengthTexcoordStart" in self._entity_data:
            return float(self._entity_data.get('LengthTexcoordStart'))
        return float(0.0)

    @property
    def LengthTexcoordEnd(self):
        if "LengthTexcoordEnd" in self._entity_data:
            return float(self._entity_data.get('LengthTexcoordEnd'))
        return float(1.0)

    @property
    def WidthTexcoordStart(self):
        if "WidthTexcoordStart" in self._entity_data:
            return float(self._entity_data.get('WidthTexcoordStart'))
        return float(0.0)

    @property
    def WidthTexcoordEnd(self):
        if "WidthTexcoordEnd" in self._entity_data:
            return float(self._entity_data.get('WidthTexcoordEnd'))
        return float(1.0)

    @property
    def Width1(self):
        if "Width1" in self._entity_data:
            return float(self._entity_data.get('Width1'))
        return float(25.0)

    @property
    def Width2(self):
        if "Width2" in self._entity_data:
            return float(self._entity_data.get('Width2'))
        return float(25.0)

    @property
    def DebugDraw(self):
        if "DebugDraw" in self._entity_data:
            return int(self._entity_data.get('DebugDraw'))
        return int(0)


class info_intermission:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None


class info_landmark(Targetname):
    pass

    icon_sprite = "editor/info_landmark"


class info_spawngroup_load_unload(Targetname):
    pass

    icon_sprite = "editor/info_target.vmat"

    @property
    def targetname(self):
        if "targetname" in self._entity_data:
            return self._entity_data.get('targetname')
        return None

    @property
    def mapname(self):
        if "mapname" in self._entity_data:
            return self._entity_data.get('mapname')
        return None

    @property
    def entityfiltername(self):
        if "entityfiltername" in self._entity_data:
            return self._entity_data.get('entityfiltername')
        return None

    @property
    def landmark(self):
        if "landmark" in self._entity_data:
            return self._entity_data.get('landmark')
        return None

    @property
    def timeoutInterval(self):
        if "timeoutInterval" in self._entity_data:
            return float(self._entity_data.get('timeoutInterval'))
        return float(0)


class info_null(Targetname):
    pass


class info_target(Targetname, Parentname):
    pass

    icon_sprite = "editor/info_target.vmat"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Transmit to client (respect PVS)': (1, 0),
                                   'Always transmit to client (ignore PVS)': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class info_particle_target(Targetname, Parentname):
    pass


class info_particle_system(Targetname, Parentname, CanBeClientOnly):
    pass

    @property
    def effect_name(self):
        if "effect_name" in self._entity_data:
            return self._entity_data.get('effect_name')
        return None

    @property
    def start_active(self):
        if "start_active" in self._entity_data:
            return bool(self._entity_data.get('start_active'))
        return bool(1)

    @property
    def snapshot_file(self):
        if "snapshot_file" in self._entity_data:
            return self._entity_data.get('snapshot_file')
        return ""

    @property
    def snapshot_mesh(self):
        if "snapshot_mesh" in self._entity_data:
            return int(self._entity_data.get('snapshot_mesh'))
        return int()

    @property
    def cpoint0(self):
        if "cpoint0" in self._entity_data:
            return self._entity_data.get('cpoint0')
        return None

    @property
    def cpoint1(self):
        if "cpoint1" in self._entity_data:
            return self._entity_data.get('cpoint1')
        return None

    @property
    def cpoint2(self):
        if "cpoint2" in self._entity_data:
            return self._entity_data.get('cpoint2')
        return None

    @property
    def cpoint3(self):
        if "cpoint3" in self._entity_data:
            return self._entity_data.get('cpoint3')
        return None

    @property
    def cpoint4(self):
        if "cpoint4" in self._entity_data:
            return self._entity_data.get('cpoint4')
        return None

    @property
    def cpoint5(self):
        if "cpoint5" in self._entity_data:
            return self._entity_data.get('cpoint5')
        return None

    @property
    def cpoint6(self):
        if "cpoint6" in self._entity_data:
            return self._entity_data.get('cpoint6')
        return None

    @property
    def cpoint7(self):
        if "cpoint7" in self._entity_data:
            return self._entity_data.get('cpoint7')
        return None

    @property
    def cpoint8(self):
        if "cpoint8" in self._entity_data:
            return self._entity_data.get('cpoint8')
        return None

    @property
    def cpoint9(self):
        if "cpoint9" in self._entity_data:
            return self._entity_data.get('cpoint9')
        return None

    @property
    def cpoint10(self):
        if "cpoint10" in self._entity_data:
            return self._entity_data.get('cpoint10')
        return None

    @property
    def cpoint11(self):
        if "cpoint11" in self._entity_data:
            return self._entity_data.get('cpoint11')
        return None

    @property
    def cpoint12(self):
        if "cpoint12" in self._entity_data:
            return self._entity_data.get('cpoint12')
        return None

    @property
    def cpoint13(self):
        if "cpoint13" in self._entity_data:
            return self._entity_data.get('cpoint13')
        return None

    @property
    def cpoint14(self):
        if "cpoint14" in self._entity_data:
            return self._entity_data.get('cpoint14')
        return None

    @property
    def cpoint15(self):
        if "cpoint15" in self._entity_data:
            return self._entity_data.get('cpoint15')
        return None

    @property
    def cpoint16(self):
        if "cpoint16" in self._entity_data:
            return self._entity_data.get('cpoint16')
        return None

    @property
    def cpoint17(self):
        if "cpoint17" in self._entity_data:
            return self._entity_data.get('cpoint17')
        return None

    @property
    def cpoint18(self):
        if "cpoint18" in self._entity_data:
            return self._entity_data.get('cpoint18')
        return None

    @property
    def cpoint19(self):
        if "cpoint19" in self._entity_data:
            return self._entity_data.get('cpoint19')
        return None

    @property
    def cpoint20(self):
        if "cpoint20" in self._entity_data:
            return self._entity_data.get('cpoint20')
        return None

    @property
    def cpoint21(self):
        if "cpoint21" in self._entity_data:
            return self._entity_data.get('cpoint21')
        return None

    @property
    def cpoint22(self):
        if "cpoint22" in self._entity_data:
            return self._entity_data.get('cpoint22')
        return None

    @property
    def cpoint23(self):
        if "cpoint23" in self._entity_data:
            return self._entity_data.get('cpoint23')
        return None

    @property
    def cpoint24(self):
        if "cpoint24" in self._entity_data:
            return self._entity_data.get('cpoint24')
        return None

    @property
    def cpoint25(self):
        if "cpoint25" in self._entity_data:
            return self._entity_data.get('cpoint25')
        return None

    @property
    def cpoint26(self):
        if "cpoint26" in self._entity_data:
            return self._entity_data.get('cpoint26')
        return None

    @property
    def cpoint27(self):
        if "cpoint27" in self._entity_data:
            return self._entity_data.get('cpoint27')
        return None

    @property
    def cpoint28(self):
        if "cpoint28" in self._entity_data:
            return self._entity_data.get('cpoint28')
        return None

    @property
    def cpoint29(self):
        if "cpoint29" in self._entity_data:
            return self._entity_data.get('cpoint29')
        return None

    @property
    def cpoint30(self):
        if "cpoint30" in self._entity_data:
            return self._entity_data.get('cpoint30')
        return None

    @property
    def cpoint31(self):
        if "cpoint31" in self._entity_data:
            return self._entity_data.get('cpoint31')
        return None

    @property
    def cpoint32(self):
        if "cpoint32" in self._entity_data:
            return self._entity_data.get('cpoint32')
        return None

    @property
    def cpoint33(self):
        if "cpoint33" in self._entity_data:
            return self._entity_data.get('cpoint33')
        return None

    @property
    def cpoint34(self):
        if "cpoint34" in self._entity_data:
            return self._entity_data.get('cpoint34')
        return None

    @property
    def cpoint35(self):
        if "cpoint35" in self._entity_data:
            return self._entity_data.get('cpoint35')
        return None

    @property
    def cpoint36(self):
        if "cpoint36" in self._entity_data:
            return self._entity_data.get('cpoint36')
        return None

    @property
    def cpoint37(self):
        if "cpoint37" in self._entity_data:
            return self._entity_data.get('cpoint37')
        return None

    @property
    def cpoint38(self):
        if "cpoint38" in self._entity_data:
            return self._entity_data.get('cpoint38')
        return None

    @property
    def cpoint39(self):
        if "cpoint39" in self._entity_data:
            return self._entity_data.get('cpoint39')
        return None

    @property
    def cpoint40(self):
        if "cpoint40" in self._entity_data:
            return self._entity_data.get('cpoint40')
        return None

    @property
    def cpoint41(self):
        if "cpoint41" in self._entity_data:
            return self._entity_data.get('cpoint41')
        return None

    @property
    def cpoint42(self):
        if "cpoint42" in self._entity_data:
            return self._entity_data.get('cpoint42')
        return None

    @property
    def cpoint43(self):
        if "cpoint43" in self._entity_data:
            return self._entity_data.get('cpoint43')
        return None

    @property
    def cpoint44(self):
        if "cpoint44" in self._entity_data:
            return self._entity_data.get('cpoint44')
        return None

    @property
    def cpoint45(self):
        if "cpoint45" in self._entity_data:
            return self._entity_data.get('cpoint45')
        return None

    @property
    def cpoint46(self):
        if "cpoint46" in self._entity_data:
            return self._entity_data.get('cpoint46')
        return None

    @property
    def cpoint47(self):
        if "cpoint47" in self._entity_data:
            return self._entity_data.get('cpoint47')
        return None

    @property
    def cpoint48(self):
        if "cpoint48" in self._entity_data:
            return self._entity_data.get('cpoint48')
        return None

    @property
    def cpoint49(self):
        if "cpoint49" in self._entity_data:
            return self._entity_data.get('cpoint49')
        return None

    @property
    def cpoint50(self):
        if "cpoint50" in self._entity_data:
            return self._entity_data.get('cpoint50')
        return None

    @property
    def cpoint51(self):
        if "cpoint51" in self._entity_data:
            return self._entity_data.get('cpoint51')
        return None

    @property
    def cpoint52(self):
        if "cpoint52" in self._entity_data:
            return self._entity_data.get('cpoint52')
        return None

    @property
    def cpoint53(self):
        if "cpoint53" in self._entity_data:
            return self._entity_data.get('cpoint53')
        return None

    @property
    def cpoint54(self):
        if "cpoint54" in self._entity_data:
            return self._entity_data.get('cpoint54')
        return None

    @property
    def cpoint55(self):
        if "cpoint55" in self._entity_data:
            return self._entity_data.get('cpoint55')
        return None

    @property
    def cpoint56(self):
        if "cpoint56" in self._entity_data:
            return self._entity_data.get('cpoint56')
        return None

    @property
    def cpoint57(self):
        if "cpoint57" in self._entity_data:
            return self._entity_data.get('cpoint57')
        return None

    @property
    def cpoint58(self):
        if "cpoint58" in self._entity_data:
            return self._entity_data.get('cpoint58')
        return None

    @property
    def cpoint59(self):
        if "cpoint59" in self._entity_data:
            return self._entity_data.get('cpoint59')
        return None

    @property
    def cpoint60(self):
        if "cpoint60" in self._entity_data:
            return self._entity_data.get('cpoint60')
        return None

    @property
    def cpoint61(self):
        if "cpoint61" in self._entity_data:
            return self._entity_data.get('cpoint61')
        return None

    @property
    def cpoint62(self):
        if "cpoint62" in self._entity_data:
            return self._entity_data.get('cpoint62')
        return None

    @property
    def cpoint63(self):
        if "cpoint63" in self._entity_data:
            return self._entity_data.get('cpoint63')
        return None

    @property
    def cpoint1_parent(self):
        if "cpoint1_parent" in self._entity_data:
            return int(self._entity_data.get('cpoint1_parent'))
        return int(0)

    @property
    def cpoint2_parent(self):
        if "cpoint2_parent" in self._entity_data:
            return int(self._entity_data.get('cpoint2_parent'))
        return int(0)

    @property
    def cpoint3_parent(self):
        if "cpoint3_parent" in self._entity_data:
            return int(self._entity_data.get('cpoint3_parent'))
        return int(0)

    @property
    def cpoint4_parent(self):
        if "cpoint4_parent" in self._entity_data:
            return int(self._entity_data.get('cpoint4_parent'))
        return int(0)

    @property
    def cpoint5_parent(self):
        if "cpoint5_parent" in self._entity_data:
            return int(self._entity_data.get('cpoint5_parent'))
        return int(0)

    @property
    def cpoint6_parent(self):
        if "cpoint6_parent" in self._entity_data:
            return int(self._entity_data.get('cpoint6_parent'))
        return int(0)

    @property
    def cpoint7_parent(self):
        if "cpoint7_parent" in self._entity_data:
            return int(self._entity_data.get('cpoint7_parent'))
        return int(0)


class phys_ragdollmagnet(Targetname, EnableDisable, Parentname):
    pass

    icon_sprite = "editor/info_target.vmat"

    @property
    def axis(self):
        if "axis" in self._entity_data:
            return parse_int_vector(self._entity_data.get('axis'))
        return parse_int_vector("None")

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return float(self._entity_data.get('radius'))
        return float(512)

    @property
    def force(self):
        if "force" in self._entity_data:
            return float(self._entity_data.get('force'))
        return float(5000)

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return ""

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Bar Magnet (use axis helper)': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class info_lighting(Targetname):
    pass

    icon_sprite = "editor/info_lighting.vmat"


class info_teleport_destination(Targetname, Parentname, PlayerClass):
    pass


class AiHullFlags:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def aihull_human(self):
        if "aihull_human" in self._entity_data:
            return bool(self._entity_data.get('aihull_human'))
        return bool(0)

    @property
    def aihull_small_centered(self):
        if "aihull_small_centered" in self._entity_data:
            return bool(self._entity_data.get('aihull_small_centered'))
        return bool(0)

    @property
    def aihull_wide_human(self):
        if "aihull_wide_human" in self._entity_data:
            return bool(self._entity_data.get('aihull_wide_human'))
        return bool(0)

    @property
    def aihull_tiny(self):
        if "aihull_tiny" in self._entity_data:
            return bool(self._entity_data.get('aihull_tiny'))
        return bool(0)

    @property
    def aihull_medium(self):
        if "aihull_medium" in self._entity_data:
            return bool(self._entity_data.get('aihull_medium'))
        return bool(0)

    @property
    def aihull_tiny_centered(self):
        if "aihull_tiny_centered" in self._entity_data:
            return bool(self._entity_data.get('aihull_tiny_centered'))
        return bool(0)

    @property
    def aihull_large(self):
        if "aihull_large" in self._entity_data:
            return bool(self._entity_data.get('aihull_large'))
        return bool(0)

    @property
    def aihull_large_centered(self):
        if "aihull_large_centered" in self._entity_data:
            return bool(self._entity_data.get('aihull_large_centered'))
        return bool(0)

    @property
    def aihull_medium_tall(self):
        if "aihull_medium_tall" in self._entity_data:
            return bool(self._entity_data.get('aihull_medium_tall'))
        return bool(0)


class info_node(Node, AiHullFlags):
    pass

    @property
    def ai_node_dont_drop(self):
        if "ai_node_dont_drop" in self._entity_data:
            return bool(self._entity_data.get('ai_node_dont_drop'))
        return bool(0)


class info_node_hint(Targetname, HintNode):
    pass

    @property
    def ai_node_dont_drop(self):
        if "ai_node_dont_drop" in self._entity_data:
            return bool(self._entity_data.get('ai_node_dont_drop'))
        return bool(0)


class info_node_air(Node):
    pass

    @property
    def nodeheight(self):
        if "nodeheight" in self._entity_data:
            return int(self._entity_data.get('nodeheight'))
        return int(0)


class info_node_air_hint(Targetname, HintNode):
    pass

    @property
    def nodeheight(self):
        if "nodeheight" in self._entity_data:
            return int(self._entity_data.get('nodeheight'))
        return int(0)


class info_hint(Targetname, HintNode):
    pass


class BaseNodeLink:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def initialstate(self):
        if "initialstate" in self._entity_data:
            return self._entity_data.get('initialstate')
        return "1"

    @property
    def AllowUse(self):
        if "AllowUse" in self._entity_data:
            return self._entity_data.get('AllowUse')
        return None

    @property
    def InvertAllow(self):
        if "InvertAllow" in self._entity_data:
            return bool(self._entity_data.get('InvertAllow'))
        return bool(0)

    @property
    def priority(self):
        if "priority" in self._entity_data:
            return self._entity_data.get('priority')
        return "0"


class info_node_link(Targetname, BaseNodeLink, AiHullFlags):
    pass

    @property
    def StartNode(self):
        if "StartNode" in self._entity_data:
            return int(self._entity_data.get('StartNode'))
        return int(0)

    @property
    def EndNode(self):
        if "EndNode" in self._entity_data:
            return int(self._entity_data.get('EndNode'))
        return int(0)

    @property
    def linktype(self):
        if "linktype" in self._entity_data:
            return self._entity_data.get('linktype')
        return "1"

    @property
    def preciseMovement(self):
        if "preciseMovement" in self._entity_data:
            return bool(self._entity_data.get('preciseMovement'))
        return bool(0)


class info_node_link_controller(Targetname, BaseNodeLink):
    pass

    @property
    def mins(self):
        if "mins" in self._entity_data:
            return parse_int_vector(self._entity_data.get('mins'))
        return parse_int_vector("-8 -32 -36")

    @property
    def maxs(self):
        if "maxs" in self._entity_data:
            return parse_int_vector(self._entity_data.get('maxs'))
        return parse_int_vector("8 32 36")

    @property
    def useairlinkradius(self):
        if "useairlinkradius" in self._entity_data:
            return bool(self._entity_data.get('useairlinkradius'))
        return bool(0)


class info_radial_link_controller(Targetname, Parentname):
    pass

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return float(self._entity_data.get('radius'))
        return float(120)


class info_node_climb(Targetname, HintNode):
    pass


class light_dynamic(Targetname, Parentname):
    pass

    icon_sprite = "editor/light.vmat"

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def _light(self):
        if "_light" in self._entity_data:
            return parse_int_vector(self._entity_data.get('_light'))
        return parse_int_vector("255 255 255 200")

    @property
    def brightness(self):
        if "brightness" in self._entity_data:
            return int(self._entity_data.get('brightness'))
        return int(0)

    @property
    def _inner_cone(self):
        if "_inner_cone" in self._entity_data:
            return int(self._entity_data.get('_inner_cone'))
        return int(30)

    @property
    def _cone(self):
        if "_cone" in self._entity_data:
            return int(self._entity_data.get('_cone'))
        return int(45)

    @property
    def distance(self):
        if "distance" in self._entity_data:
            return float(self._entity_data.get('distance'))
        return float(120)

    @property
    def spotlight_radius(self):
        if "spotlight_radius" in self._entity_data:
            return float(self._entity_data.get('spotlight_radius'))
        return float(80)

    @property
    def style(self):
        if "style" in self._entity_data:
            return self._entity_data.get('style')
        return "0"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Do not light world (better perf)': (1, 0), 'Do not light models': (2, 0),
                                   'Add Displacement Alpha': (4, 0), 'Subtract Displacement Alpha': (8, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class shadow_control(Targetname):
    pass

    icon_sprite = "editor/shadow_control.vmat"

    @property
    def angles(self):
        if "angles" in self._entity_data:
            return self._entity_data.get('angles')
        return "80 30 0"

    @property
    def color(self):
        if "color" in self._entity_data:
            return parse_int_vector(self._entity_data.get('color'))
        return parse_int_vector("128 128 128")

    @property
    def distance(self):
        if "distance" in self._entity_data:
            return float(self._entity_data.get('distance'))
        return float(75)

    @property
    def disableallshadows(self):
        if "disableallshadows" in self._entity_data:
            return bool(self._entity_data.get('disableallshadows'))
        return bool(0)

    @property
    def enableshadowsfromlocallights(self):
        if "enableshadowsfromlocallights" in self._entity_data:
            return bool(self._entity_data.get('enableshadowsfromlocallights'))
        return bool(0)


class color_correction(Targetname, EnableDisable):
    pass

    icon_sprite = "editor/color_correction.vmat"

    @property
    def minfalloff(self):
        if "minfalloff" in self._entity_data:
            return float(self._entity_data.get('minfalloff'))
        return float(0.0)

    @property
    def maxfalloff(self):
        if "maxfalloff" in self._entity_data:
            return float(self._entity_data.get('maxfalloff'))
        return float(200.0)

    @property
    def maxweight(self):
        if "maxweight" in self._entity_data:
            return float(self._entity_data.get('maxweight'))
        return float(1.0)

    @property
    def filename(self):
        if "filename" in self._entity_data:
            return self._entity_data.get('filename')
        return ""

    @property
    def fadeInDuration(self):
        if "fadeInDuration" in self._entity_data:
            return float(self._entity_data.get('fadeInDuration'))
        return float(0.0)

    @property
    def fadeOutDuration(self):
        if "fadeOutDuration" in self._entity_data:
            return float(self._entity_data.get('fadeOutDuration'))
        return float(0.0)

    @property
    def exclusive(self):
        if "exclusive" in self._entity_data:
            return bool(self._entity_data.get('exclusive'))
        return bool(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Master (Default color correction when used with L4D fog_volume)': (1, 0),
                                   'Simulate client-side (Must be set when used with L4D fog_volume)': (2, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class color_correction_volume(Targetname, EnableDisable):
    pass

    @property
    def fadeDuration(self):
        if "fadeDuration" in self._entity_data:
            return float(self._entity_data.get('fadeDuration'))
        return float(10.0)

    @property
    def maxweight(self):
        if "maxweight" in self._entity_data:
            return float(self._entity_data.get('maxweight'))
        return float(1.0)

    @property
    def filename(self):
        if "filename" in self._entity_data:
            return self._entity_data.get('filename')
        return ""


class KeyFrame:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def NextKey(self):
        if "NextKey" in self._entity_data:
            return self._entity_data.get('NextKey')
        return None

    @property
    def MoveSpeed(self):
        if "MoveSpeed" in self._entity_data:
            return int(self._entity_data.get('MoveSpeed'))
        return int(64)


class Mover:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def PositionInterpolator(self):
        if "PositionInterpolator" in self._entity_data:
            return self._entity_data.get('PositionInterpolator')
        return "0"


class func_movelinear(Targetname, Parentname, RenderFields):
    pass

    @property
    def movedir(self):
        if "movedir" in self._entity_data:
            return parse_int_vector(self._entity_data.get('movedir'))
        return parse_int_vector("0 0 0")

    @property
    def movedir_islocal(self):
        if "movedir_islocal" in self._entity_data:
            return bool(self._entity_data.get('movedir_islocal'))
        return bool(1)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Not Solid': (8, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def startposition(self):
        if "startposition" in self._entity_data:
            return float(self._entity_data.get('startposition'))
        return float(0)

    @property
    def speed(self):
        if "speed" in self._entity_data:
            return int(self._entity_data.get('speed'))
        return int(100)

    @property
    def movedistance(self):
        if "movedistance" in self._entity_data:
            return float(self._entity_data.get('movedistance'))
        return float(100)

    @property
    def blockdamage(self):
        if "blockdamage" in self._entity_data:
            return float(self._entity_data.get('blockdamage'))
        return float(0)

    @property
    def startsound(self):
        if "startsound" in self._entity_data:
            return self._entity_data.get('startsound')
        return None

    @property
    def stopsound(self):
        if "stopsound" in self._entity_data:
            return self._entity_data.get('stopsound')
        return None

    @property
    def CreateNavObstacle(self):
        if "CreateNavObstacle" in self._entity_data:
            return bool(self._entity_data.get('CreateNavObstacle'))
        return bool(0)


class func_water_analog(Targetname, Parentname):
    pass

    @property
    def movedir(self):
        if "movedir" in self._entity_data:
            return parse_int_vector(self._entity_data.get('movedir'))
        return parse_int_vector("0 0 0")

    @property
    def startposition(self):
        if "startposition" in self._entity_data:
            return float(self._entity_data.get('startposition'))
        return float(0)

    @property
    def speed(self):
        if "speed" in self._entity_data:
            return int(self._entity_data.get('speed'))
        return int(100)

    @property
    def movedistance(self):
        if "movedistance" in self._entity_data:
            return float(self._entity_data.get('movedistance'))
        return float(100)

    @property
    def startsound(self):
        if "startsound" in self._entity_data:
            return self._entity_data.get('startsound')
        return None

    @property
    def stopsound(self):
        if "stopsound" in self._entity_data:
            return self._entity_data.get('stopsound')
        return None

    @property
    def WaveHeight(self):
        if "WaveHeight" in self._entity_data:
            return self._entity_data.get('WaveHeight')
        return "3.0"


class func_rotating(Targetname, Parentname, RenderFields, Shadow):
    pass

    @property
    def maxspeed(self):
        if "maxspeed" in self._entity_data:
            return int(self._entity_data.get('maxspeed'))
        return int(100)

    @property
    def fanfriction(self):
        if "fanfriction" in self._entity_data:
            return int(self._entity_data.get('fanfriction'))
        return int(20)

    @property
    def message(self):
        if "message" in self._entity_data:
            return self._entity_data.get('message')
        return None

    @property
    def volume(self):
        if "volume" in self._entity_data:
            return int(self._entity_data.get('volume'))
        return int(10)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start ON': (1, 0), 'Reverse Direction': (2, 0), 'X Axis': (4, 0), 'Y Axis': (8, 0),
                                   'Acc/Dcc': (16, 0), 'Fan Pain': (32, 0), 'Not Solid': (64, 0),
                                   'Small Sound Radius': (128, 0), 'Medium Sound Radius': (256, 0),
                                   'Large Sound Radius': (512, 1), 'Client-side Animation': (1024, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def _minlight(self):
        if "_minlight" in self._entity_data:
            return self._entity_data.get('_minlight')
        return None

    @property
    def dmg(self):
        if "dmg" in self._entity_data:
            return int(self._entity_data.get('dmg'))
        return int(0)

    @property
    def solidbsp(self):
        if "solidbsp" in self._entity_data:
            return self._entity_data.get('solidbsp')
        return "0"


class func_platrot(Targetname, Parentname, RenderFields, BasePlat, Shadow):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Toggle': (1, 1), 'X Axis': (64, 0), 'Y Axis': (128, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def noise1(self):
        if "noise1" in self._entity_data:
            return self._entity_data.get('noise1')
        return None

    @property
    def noise2(self):
        if "noise2" in self._entity_data:
            return self._entity_data.get('noise2')
        return None

    @property
    def speed(self):
        if "speed" in self._entity_data:
            return int(self._entity_data.get('speed'))
        return int(50)

    @property
    def height(self):
        if "height" in self._entity_data:
            return int(self._entity_data.get('height'))
        return int(0)

    @property
    def rotation(self):
        if "rotation" in self._entity_data:
            return int(self._entity_data.get('rotation'))
        return int(0)

    @property
    def _minlight(self):
        if "_minlight" in self._entity_data:
            return self._entity_data.get('_minlight')
        return None


class keyframe_track(Targetname, Parentname, KeyFrame):
    pass


class move_keyframed(Targetname, Parentname, KeyFrame, Mover):
    pass


class move_track(Targetname, Parentname, Mover, KeyFrame):
    pass

    @property
    def WheelBaseLength(self):
        if "WheelBaseLength" in self._entity_data:
            return int(self._entity_data.get('WheelBaseLength'))
        return int(50)

    @property
    def Damage(self):
        if "Damage" in self._entity_data:
            return int(self._entity_data.get('Damage'))
        return int(0)

    @property
    def NoRotate(self):
        if "NoRotate" in self._entity_data:
            return bool(self._entity_data.get('NoRotate'))
        return bool(0)


class RopeKeyFrame(SystemLevelChoice):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Auto Resize': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def Slack(self):
        if "Slack" in self._entity_data:
            return int(self._entity_data.get('Slack'))
        return int(25)

    @property
    def Type(self):
        if "Type" in self._entity_data:
            return self._entity_data.get('Type')
        return "0"

    @property
    def Subdiv(self):
        if "Subdiv" in self._entity_data:
            return int(self._entity_data.get('Subdiv'))
        return int(2)

    @property
    def Barbed(self):
        if "Barbed" in self._entity_data:
            return bool(self._entity_data.get('Barbed'))
        return bool(0)

    @property
    def Width(self):
        if "Width" in self._entity_data:
            return self._entity_data.get('Width')
        return "2"

    @property
    def TextureScale(self):
        if "TextureScale" in self._entity_data:
            return self._entity_data.get('TextureScale')
        return "1"

    @property
    def Collide(self):
        if "Collide" in self._entity_data:
            return bool(self._entity_data.get('Collide'))
        return bool(0)

    @property
    def Dangling(self):
        if "Dangling" in self._entity_data:
            return bool(self._entity_data.get('Dangling'))
        return bool(0)

    @property
    def Breakable(self):
        if "Breakable" in self._entity_data:
            return bool(self._entity_data.get('Breakable'))
        return bool(0)

    @property
    def UseWind(self):
        if "UseWind" in self._entity_data:
            return bool(self._entity_data.get('UseWind'))
        return bool(0)

    @property
    def RopeMaterial(self):
        if "RopeMaterial" in self._entity_data:
            return self._entity_data.get('RopeMaterial')
        return "cable/cable.vmat"


class keyframe_rope(Targetname, Parentname, KeyFrame, RopeKeyFrame):
    pass


class move_rope(Targetname, Parentname, KeyFrame, RopeKeyFrame):
    pass

    @property
    def PositionInterpolator(self):
        if "PositionInterpolator" in self._entity_data:
            return self._entity_data.get('PositionInterpolator')
        return "2"


class Button:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def displaytext(self):
        if "displaytext" in self._entity_data:
            return self._entity_data.get('displaytext')
        return None


class BaseFuncButton(Targetname, Parentname, RenderFields, DamageFilter, Button):
    pass

    @property
    def movedir(self):
        if "movedir" in self._entity_data:
            return parse_int_vector(self._entity_data.get('movedir'))
        return parse_int_vector("0 0 0")

    @property
    def movedir_islocal(self):
        if "movedir_islocal" in self._entity_data:
            return bool(self._entity_data.get('movedir_islocal'))
        return bool(1)

    @property
    def speed(self):
        if "speed" in self._entity_data:
            return int(self._entity_data.get('speed'))
        return int(5)

    @property
    def glow(self):
        if "glow" in self._entity_data:
            return self._entity_data.get('glow')
        return None

    @property
    def wait(self):
        if "wait" in self._entity_data:
            return int(self._entity_data.get('wait'))
        return int(3)

    @property
    def use_sound(self):
        if "use_sound" in self._entity_data:
            return self._entity_data.get('use_sound')
        return ""

    @property
    def locked_sound(self):
        if "locked_sound" in self._entity_data:
            return self._entity_data.get('locked_sound')
        return ""

    @property
    def unlocked_sound(self):
        if "unlocked_sound" in self._entity_data:
            return self._entity_data.get('unlocked_sound')
        return ""

    @property
    def _minlight(self):
        if "_minlight" in self._entity_data:
            return self._entity_data.get('_minlight')
        return None


class func_button(BaseFuncButton):
    pass

    @property
    def lip(self):
        if "lip" in self._entity_data:
            return int(self._entity_data.get('lip'))
        return int(0)

    @property
    def health(self):
        if "health" in self._entity_data:
            return int(self._entity_data.get('health'))
        return int(0)

    @property
    def master(self):
        if "master" in self._entity_data:
            return self._entity_data.get('master')
        return None

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {"Don't move": (1, 0), 'Toggle': (32, 0), 'Touch Activates': (256, 0),
                                   'Damage Activates': (512, 0), 'Use Activates': (1024, 1), 'Starts locked': (2048, 0),
                                   'Sparks': (4096, 0), 'Non-solid': (16384, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class func_physical_button(BaseFuncButton):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Toggle': (32, 0), 'Damage Activates': (512, 0), 'Starts locked': (2048, 0),
                                   'Sparks': (4096, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class func_rot_button(Targetname, Parentname, Global, Button, EnableDisable):
    pass

    @property
    def master(self):
        if "master" in self._entity_data:
            return self._entity_data.get('master')
        return None

    @property
    def speed(self):
        if "speed" in self._entity_data:
            return int(self._entity_data.get('speed'))
        return int(50)

    @property
    def health(self):
        if "health" in self._entity_data:
            return int(self._entity_data.get('health'))
        return int(0)

    @property
    def sounds(self):
        if "sounds" in self._entity_data:
            return self._entity_data.get('sounds')
        return "21"

    @property
    def wait(self):
        if "wait" in self._entity_data:
            return int(self._entity_data.get('wait'))
        return int(3)

    @property
    def distance(self):
        if "distance" in self._entity_data:
            return int(self._entity_data.get('distance'))
        return int(90)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Not solid': (1, 0), 'Reverse Dir': (2, 0), 'Toggle': (32, 0), 'X Axis': (64, 0),
                                   'Y Axis': (128, 0), 'Touch Activates': (256, 0), 'Damage Activates': (512, 0),
                                   'Use Activates': (1024, 0), 'Starts locked': (2048, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def _minlight(self):
        if "_minlight" in self._entity_data:
            return self._entity_data.get('_minlight')
        return None


class momentary_rot_button(Targetname, Parentname, RenderFields):
    pass

    @property
    def speed(self):
        if "speed" in self._entity_data:
            return int(self._entity_data.get('speed'))
        return int(50)

    @property
    def master(self):
        if "master" in self._entity_data:
            return self._entity_data.get('master')
        return None

    @property
    def glow(self):
        if "glow" in self._entity_data:
            return self._entity_data.get('glow')
        return None

    @property
    def sounds(self):
        if "sounds" in self._entity_data:
            return self._entity_data.get('sounds')
        return "0"

    @property
    def distance(self):
        if "distance" in self._entity_data:
            return int(self._entity_data.get('distance'))
        return int(90)

    @property
    def returnspeed(self):
        if "returnspeed" in self._entity_data:
            return int(self._entity_data.get('returnspeed'))
        return int(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Not Solid': (1, 1), 'Toggle (Disable Auto Return)': (32, 1), 'X Axis': (64, 0),
                                   'Y Axis': (128, 0), 'Use Activates': (1024, 1), 'Starts locked': (2048, 0),
                                   'Jiggle when used while locked': (8192, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def _minlight(self):
        if "_minlight" in self._entity_data:
            return self._entity_data.get('_minlight')
        return None

    @property
    def startposition(self):
        if "startposition" in self._entity_data:
            return float(self._entity_data.get('startposition'))
        return float(0)

    @property
    def startdirection(self):
        if "startdirection" in self._entity_data:
            return self._entity_data.get('startdirection')
        return "Forward"

    @property
    def solidbsp(self):
        if "solidbsp" in self._entity_data:
            return bool(self._entity_data.get('solidbsp'))
        return bool(0)


class Door(Targetname, Parentname, RenderFields, Global, Shadow):
    pass

    @property
    def speed(self):
        if "speed" in self._entity_data:
            return int(self._entity_data.get('speed'))
        return int(100)

    @property
    def master(self):
        if "master" in self._entity_data:
            return self._entity_data.get('master')
        return None

    @property
    def noise1(self):
        if "noise1" in self._entity_data:
            return self._entity_data.get('noise1')
        return None

    @property
    def noise2(self):
        if "noise2" in self._entity_data:
            return self._entity_data.get('noise2')
        return None

    @property
    def startclosesound(self):
        if "startclosesound" in self._entity_data:
            return self._entity_data.get('startclosesound')
        return None

    @property
    def closesound(self):
        if "closesound" in self._entity_data:
            return self._entity_data.get('closesound')
        return None

    @property
    def wait(self):
        if "wait" in self._entity_data:
            return int(self._entity_data.get('wait'))
        return int(4)

    @property
    def lip(self):
        if "lip" in self._entity_data:
            return int(self._entity_data.get('lip'))
        return int(0)

    @property
    def dmg(self):
        if "dmg" in self._entity_data:
            return int(self._entity_data.get('dmg'))
        return int(0)

    @property
    def forceclosed(self):
        if "forceclosed" in self._entity_data:
            return bool(self._entity_data.get('forceclosed'))
        return bool(0)

    @property
    def ignoredebris(self):
        if "ignoredebris" in self._entity_data:
            return bool(self._entity_data.get('ignoredebris'))
        return bool(0)

    @property
    def message(self):
        if "message" in self._entity_data:
            return self._entity_data.get('message')
        return None

    @property
    def health(self):
        if "health" in self._entity_data:
            return int(self._entity_data.get('health'))
        return int(0)

    @property
    def locked_sound(self):
        if "locked_sound" in self._entity_data:
            return self._entity_data.get('locked_sound')
        return ""

    @property
    def unlocked_sound(self):
        if "unlocked_sound" in self._entity_data:
            return self._entity_data.get('unlocked_sound')
        return ""

    @property
    def spawnpos(self):
        if "spawnpos" in self._entity_data:
            return self._entity_data.get('spawnpos')
        return "0"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {"Starts Open - OBSOLETE, use 'Spawn Position' key instead": (1, 0),
                                   'Non-solid to Player': (4, 0), 'Passable': (8, 0), 'Toggle': (32, 0),
                                   'Use Opens': (256, 0), "NPCs Can't": (512, 0), 'Touch Opens': (1024, 1),
                                   'Starts locked': (2048, 0), 'Door Silent': (4096, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def locked_sentence(self):
        if "locked_sentence" in self._entity_data:
            return self._entity_data.get('locked_sentence')
        return "0"

    @property
    def unlocked_sentence(self):
        if "unlocked_sentence" in self._entity_data:
            return self._entity_data.get('unlocked_sentence')
        return "0"

    @property
    def _minlight(self):
        if "_minlight" in self._entity_data:
            return self._entity_data.get('_minlight')
        return None

    @property
    def loopmovesound(self):
        if "loopmovesound" in self._entity_data:
            return bool(self._entity_data.get('loopmovesound'))
        return bool(0)


class func_door(Door):
    pass

    @property
    def movedir(self):
        if "movedir" in self._entity_data:
            return parse_int_vector(self._entity_data.get('movedir'))
        return parse_int_vector("0 0 0")

    @property
    def movedir_islocal(self):
        if "movedir_islocal" in self._entity_data:
            return bool(self._entity_data.get('movedir_islocal'))
        return bool(1)

    @property
    def filtername(self):
        if "filtername" in self._entity_data:
            return self._entity_data.get('filtername')
        return None


class func_door_rotating(Door):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Reverse Dir': (2, 0), 'One-way': (16, 0), 'X Axis': (64, 0), 'Y Axis': (128, 0),
                                   'New func_door +USE rules (NOT for prop_doors!!)': (65536, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def distance(self):
        if "distance" in self._entity_data:
            return int(self._entity_data.get('distance'))
        return int(90)

    @property
    def solidbsp(self):
        if "solidbsp" in self._entity_data:
            return self._entity_data.get('solidbsp')
        return "0"


class BaseFadeProp:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def fademindist(self):
        if "fademindist" in self._entity_data:
            return float(self._entity_data.get('fademindist'))
        return float(-1)

    @property
    def fademaxdist(self):
        if "fademaxdist" in self._entity_data:
            return float(self._entity_data.get('fademaxdist'))
        return float(0)

    @property
    def fadescale(self):
        if "fadescale" in self._entity_data:
            return float(self._entity_data.get('fadescale'))
        return float(1)

    @property
    def rendercolor(self):
        if "rendercolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('rendercolor'))
        return parse_int_vector("255 255 255")


class BasePropDoorRotating(Targetname, Parentname, Global, Studiomodel, BaseFadeProp, Glow):
    pass

    @property
    def slavename(self):
        if "slavename" in self._entity_data:
            return self._entity_data.get('slavename')
        return None

    @property
    def hardware(self):
        if "hardware" in self._entity_data:
            return self._entity_data.get('hardware')
        return "1"

    @property
    def ajarangle(self):
        if "ajarangle" in self._entity_data:
            return float(self._entity_data.get('ajarangle'))
        return float(0)

    @property
    def spawnpos(self):
        if "spawnpos" in self._entity_data:
            return self._entity_data.get('spawnpos')
        return "0"

    @property
    def axis(self):
        if "axis" in self._entity_data:
            return parse_int_vector(self._entity_data.get('axis'))
        return parse_int_vector("None")

    @property
    def distance(self):
        if "distance" in self._entity_data:
            return float(self._entity_data.get('distance'))
        return float(90)

    @property
    def speed(self):
        if "speed" in self._entity_data:
            return int(self._entity_data.get('speed'))
        return int(100)

    @property
    def soundopenoverride(self):
        if "soundopenoverride" in self._entity_data:
            return self._entity_data.get('soundopenoverride')
        return None

    @property
    def soundcloseoverride(self):
        if "soundcloseoverride" in self._entity_data:
            return self._entity_data.get('soundcloseoverride')
        return None

    @property
    def soundmoveoverride(self):
        if "soundmoveoverride" in self._entity_data:
            return self._entity_data.get('soundmoveoverride')
        return None

    @property
    def soundjiggleoverride(self):
        if "soundjiggleoverride" in self._entity_data:
            return self._entity_data.get('soundjiggleoverride')
        return None

    @property
    def returndelay(self):
        if "returndelay" in self._entity_data:
            return int(self._entity_data.get('returndelay'))
        return int(-1)

    @property
    def dmg(self):
        if "dmg" in self._entity_data:
            return int(self._entity_data.get('dmg'))
        return int(0)

    @property
    def health(self):
        if "health" in self._entity_data:
            return int(self._entity_data.get('health'))
        return int(0)

    @property
    def soundlockedoverride(self):
        if "soundlockedoverride" in self._entity_data:
            return self._entity_data.get('soundlockedoverride')
        return None

    @property
    def soundunlockedoverride(self):
        if "soundunlockedoverride" in self._entity_data:
            return self._entity_data.get('soundunlockedoverride')
        return None

    @property
    def soundlatchoverride(self):
        if "soundlatchoverride" in self._entity_data:
            return self._entity_data.get('soundlatchoverride')
        return None

    @property
    def forceclosed(self):
        if "forceclosed" in self._entity_data:
            return bool(self._entity_data.get('forceclosed'))
        return bool(0)

    @property
    def rendertocubemaps(self):
        if "rendertocubemaps" in self._entity_data:
            return bool(self._entity_data.get('rendertocubemaps'))
        return bool(1)

    @property
    def lightmapstatic(self):
        if "lightmapstatic" in self._entity_data:
            return self._entity_data.get('lightmapstatic')
        return "0"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Starts Open': (1, 0), 'Starts locked': (2048, 0),
                                   'Door silent (No sound, and does not alert NPCs)': (4096, 0),
                                   'Use closes': (8192, 1), 'Door silent to NPCS (Does not alert NPCs)': (16384, 0),
                                   'Ignore player +USE': (32768, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def opendir(self):
        if "opendir" in self._entity_data:
            return self._entity_data.get('opendir')
        return "0"


class BModelParticleSpawner:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def StartDisabled(self):
        if "StartDisabled" in self._entity_data:
            return bool(self._entity_data.get('StartDisabled'))
        return bool(0)

    @property
    def Color(self):
        if "Color" in self._entity_data:
            return parse_int_vector(self._entity_data.get('Color'))
        return parse_int_vector("255 255 255")

    @property
    def SpawnRate(self):
        if "SpawnRate" in self._entity_data:
            return int(self._entity_data.get('SpawnRate'))
        return int(40)

    @property
    def SpeedMax(self):
        if "SpeedMax" in self._entity_data:
            return self._entity_data.get('SpeedMax')
        return "13"

    @property
    def LifetimeMin(self):
        if "LifetimeMin" in self._entity_data:
            return self._entity_data.get('LifetimeMin')
        return "3"

    @property
    def LifetimeMax(self):
        if "LifetimeMax" in self._entity_data:
            return self._entity_data.get('LifetimeMax')
        return "5"

    @property
    def DistMax(self):
        if "DistMax" in self._entity_data:
            return int(self._entity_data.get('DistMax'))
        return int(1024)

    @property
    def Frozen(self):
        if "Frozen" in self._entity_data:
            return bool(self._entity_data.get('Frozen'))
        return bool(0)


class func_dustmotes(Targetname, BModelParticleSpawner):
    pass

    @property
    def SizeMin(self):
        if "SizeMin" in self._entity_data:
            return self._entity_data.get('SizeMin')
        return "10"

    @property
    def SizeMax(self):
        if "SizeMax" in self._entity_data:
            return self._entity_data.get('SizeMax')
        return "20"

    @property
    def Alpha(self):
        if "Alpha" in self._entity_data:
            return int(self._entity_data.get('Alpha'))
        return int(255)


class func_dustcloud(Targetname, BModelParticleSpawner):
    pass

    @property
    def Alpha(self):
        if "Alpha" in self._entity_data:
            return int(self._entity_data.get('Alpha'))
        return int(30)

    @property
    def SizeMin(self):
        if "SizeMin" in self._entity_data:
            return self._entity_data.get('SizeMin')
        return "100"

    @property
    def SizeMax(self):
        if "SizeMax" in self._entity_data:
            return self._entity_data.get('SizeMax')
        return "200"


class env_dustpuff(Targetname, Parentname):
    pass

    @property
    def scale(self):
        if "scale" in self._entity_data:
            return float(self._entity_data.get('scale'))
        return float(8)

    @property
    def speed(self):
        if "speed" in self._entity_data:
            return float(self._entity_data.get('speed'))
        return float(16)

    @property
    def color(self):
        if "color" in self._entity_data:
            return parse_int_vector(self._entity_data.get('color'))
        return parse_int_vector("128 128 128")


class env_particlescript(Targetname, Parentname):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/Ambient_citadel_paths.vmdl"


class env_effectscript(Targetname, Parentname):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/Effects/teleporttrail.vmdl"

    @property
    def scriptfile(self):
        if "scriptfile" in self._entity_data:
            return self._entity_data.get('scriptfile')
        return "scripts/effects/testeffect.txt"


class logic_auto(Targetname):
    pass

    icon_sprite = "editor/logic_auto.vmat"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Remove on fire': (1, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def globalstate(self):
        if "globalstate" in self._entity_data:
            return self._entity_data.get('globalstate')
        return None


class point_viewcontrol(Targetname, Parentname):
    pass

    @property
    def fov(self):
        if "fov" in self._entity_data:
            return float(self._entity_data.get('fov'))
        return float(90)

    @property
    def fov_rate(self):
        if "fov_rate" in self._entity_data:
            return float(self._entity_data.get('fov_rate'))
        return float(1.0)

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def targetattachment(self):
        if "targetattachment" in self._entity_data:
            return self._entity_data.get('targetattachment')
        return None

    @property
    def wait(self):
        if "wait" in self._entity_data:
            return int(self._entity_data.get('wait'))
        return int(10)

    @property
    def moveto(self):
        if "moveto" in self._entity_data:
            return self._entity_data.get('moveto')
        return None

    @property
    def interpolatepositiontoplayer(self):
        if "interpolatepositiontoplayer" in self._entity_data:
            return bool(self._entity_data.get('interpolatepositiontoplayer'))
        return bool(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start At Player': (1, 1), 'Follow Player': (2, 1), 'Freeze Player': (4, 0),
                                   'Infinite Hold Time': (8, 0), 'Snap to goal angles': (16, 0),
                                   'Make Player non-solid': (32, 0), 'Interruptable by Player': (64, 0),
                                   'Set FOV': (128, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def speed(self):
        if "speed" in self._entity_data:
            return self._entity_data.get('speed')
        return "0"

    @property
    def acceleration(self):
        if "acceleration" in self._entity_data:
            return self._entity_data.get('acceleration')
        return "500"

    @property
    def deceleration(self):
        if "deceleration" in self._entity_data:
            return self._entity_data.get('deceleration')
        return "500"


class point_posecontroller(Targetname):
    pass

    @property
    def PropName(self):
        if "PropName" in self._entity_data:
            return self._entity_data.get('PropName')
        return None

    @property
    def PoseParameterName(self):
        if "PoseParameterName" in self._entity_data:
            return self._entity_data.get('PoseParameterName')
        return None

    @property
    def PoseValue(self):
        if "PoseValue" in self._entity_data:
            return float(self._entity_data.get('PoseValue'))
        return float(0.0)

    @property
    def InterpolationTime(self):
        if "InterpolationTime" in self._entity_data:
            return float(self._entity_data.get('InterpolationTime'))
        return float(0.0)

    @property
    def InterpolationWrap(self):
        if "InterpolationWrap" in self._entity_data:
            return bool(self._entity_data.get('InterpolationWrap'))
        return bool(0)

    @property
    def CycleFrequency(self):
        if "CycleFrequency" in self._entity_data:
            return float(self._entity_data.get('CycleFrequency'))
        return float(0.0)

    @property
    def FModulationType(self):
        if "FModulationType" in self._entity_data:
            return self._entity_data.get('FModulationType')
        return "0"

    @property
    def FModTimeOffset(self):
        if "FModTimeOffset" in self._entity_data:
            return float(self._entity_data.get('FModTimeOffset'))
        return float(0.0)

    @property
    def FModRate(self):
        if "FModRate" in self._entity_data:
            return float(self._entity_data.get('FModRate'))
        return float(0.0)

    @property
    def FModAmplitude(self):
        if "FModAmplitude" in self._entity_data:
            return float(self._entity_data.get('FModAmplitude'))
        return float(0.0)


class logic_compare(Targetname):
    pass

    icon_sprite = "editor/logic_compare.vmat"

    @property
    def InitialValue(self):
        if "InitialValue" in self._entity_data:
            return int(self._entity_data.get('InitialValue'))
        return int(0)

    @property
    def CompareValue(self):
        if "CompareValue" in self._entity_data:
            return int(self._entity_data.get('CompareValue'))
        return int(0)


class logic_branch(Targetname):
    pass

    icon_sprite = "editor/logic_branch.vmat"

    @property
    def InitialValue(self):
        if "InitialValue" in self._entity_data:
            return int(self._entity_data.get('InitialValue'))
        return int(0)


class logic_branch_listener(Targetname):
    pass

    @property
    def Branch01(self):
        if "Branch01" in self._entity_data:
            return self._entity_data.get('Branch01')
        return None

    @property
    def Branch02(self):
        if "Branch02" in self._entity_data:
            return self._entity_data.get('Branch02')
        return None

    @property
    def Branch03(self):
        if "Branch03" in self._entity_data:
            return self._entity_data.get('Branch03')
        return None

    @property
    def Branch04(self):
        if "Branch04" in self._entity_data:
            return self._entity_data.get('Branch04')
        return None

    @property
    def Branch05(self):
        if "Branch05" in self._entity_data:
            return self._entity_data.get('Branch05')
        return None

    @property
    def Branch06(self):
        if "Branch06" in self._entity_data:
            return self._entity_data.get('Branch06')
        return None

    @property
    def Branch07(self):
        if "Branch07" in self._entity_data:
            return self._entity_data.get('Branch07')
        return None

    @property
    def Branch08(self):
        if "Branch08" in self._entity_data:
            return self._entity_data.get('Branch08')
        return None

    @property
    def Branch09(self):
        if "Branch09" in self._entity_data:
            return self._entity_data.get('Branch09')
        return None

    @property
    def Branch10(self):
        if "Branch10" in self._entity_data:
            return self._entity_data.get('Branch10')
        return None

    @property
    def Branch11(self):
        if "Branch11" in self._entity_data:
            return self._entity_data.get('Branch11')
        return None

    @property
    def Branch12(self):
        if "Branch12" in self._entity_data:
            return self._entity_data.get('Branch12')
        return None

    @property
    def Branch13(self):
        if "Branch13" in self._entity_data:
            return self._entity_data.get('Branch13')
        return None

    @property
    def Branch14(self):
        if "Branch14" in self._entity_data:
            return self._entity_data.get('Branch14')
        return None

    @property
    def Branch15(self):
        if "Branch15" in self._entity_data:
            return self._entity_data.get('Branch15')
        return None

    @property
    def Branch16(self):
        if "Branch16" in self._entity_data:
            return self._entity_data.get('Branch16')
        return None


class logic_case(Targetname):
    pass

    icon_sprite = "editor/logic_case.vmat"

    @property
    def Case01(self):
        if "Case01" in self._entity_data:
            return self._entity_data.get('Case01')
        return None

    @property
    def Case02(self):
        if "Case02" in self._entity_data:
            return self._entity_data.get('Case02')
        return None

    @property
    def Case03(self):
        if "Case03" in self._entity_data:
            return self._entity_data.get('Case03')
        return None

    @property
    def Case04(self):
        if "Case04" in self._entity_data:
            return self._entity_data.get('Case04')
        return None

    @property
    def Case05(self):
        if "Case05" in self._entity_data:
            return self._entity_data.get('Case05')
        return None

    @property
    def Case06(self):
        if "Case06" in self._entity_data:
            return self._entity_data.get('Case06')
        return None

    @property
    def Case07(self):
        if "Case07" in self._entity_data:
            return self._entity_data.get('Case07')
        return None

    @property
    def Case08(self):
        if "Case08" in self._entity_data:
            return self._entity_data.get('Case08')
        return None

    @property
    def Case09(self):
        if "Case09" in self._entity_data:
            return self._entity_data.get('Case09')
        return None

    @property
    def Case10(self):
        if "Case10" in self._entity_data:
            return self._entity_data.get('Case10')
        return None

    @property
    def Case11(self):
        if "Case11" in self._entity_data:
            return self._entity_data.get('Case11')
        return None

    @property
    def Case12(self):
        if "Case12" in self._entity_data:
            return self._entity_data.get('Case12')
        return None

    @property
    def Case13(self):
        if "Case13" in self._entity_data:
            return self._entity_data.get('Case13')
        return None

    @property
    def Case14(self):
        if "Case14" in self._entity_data:
            return self._entity_data.get('Case14')
        return None

    @property
    def Case15(self):
        if "Case15" in self._entity_data:
            return self._entity_data.get('Case15')
        return None

    @property
    def Case16(self):
        if "Case16" in self._entity_data:
            return self._entity_data.get('Case16')
        return None


class logic_multicompare(Targetname):
    pass

    icon_sprite = "editor/logic_multicompare.vmat"

    @property
    def IntegerValue(self):
        if "IntegerValue" in self._entity_data:
            return int(self._entity_data.get('IntegerValue'))
        return int(0)

    @property
    def ShouldComparetoValue(self):
        if "ShouldComparetoValue" in self._entity_data:
            return bool(self._entity_data.get('ShouldComparetoValue'))
        return bool(0)


class LogicNPCCounterPointBase:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def startDisabled(self):
        if "startDisabled" in self._entity_data:
            return bool(self._entity_data.get('startDisabled'))
        return bool(0)

    @property
    def sourceEntityName(self):
        if "sourceEntityName" in self._entity_data:
            return self._entity_data.get('sourceEntityName')
        return ""

    @property
    def minCount(self):
        if "minCount" in self._entity_data:
            return int(self._entity_data.get('minCount'))
        return int(-1)

    @property
    def maxCount(self):
        if "maxCount" in self._entity_data:
            return int(self._entity_data.get('maxCount'))
        return int(-1)

    @property
    def minFactor(self):
        if "minFactor" in self._entity_data:
            return int(self._entity_data.get('minFactor'))
        return int(0)

    @property
    def maxFactor(self):
        if "maxFactor" in self._entity_data:
            return int(self._entity_data.get('maxFactor'))
        return int(0)

    @property
    def NPCType1(self):
        if "NPCType1" in self._entity_data:
            return self._entity_data.get('NPCType1')
        return None

    @property
    def NPCState1(self):
        if "NPCState1" in self._entity_data:
            return self._entity_data.get('NPCState1')
        return "-1"

    @property
    def invertState1(self):
        if "invertState1" in self._entity_data:
            return bool(self._entity_data.get('invertState1'))
        return bool(0)

    @property
    def minCount1(self):
        if "minCount1" in self._entity_data:
            return int(self._entity_data.get('minCount1'))
        return int(-1)

    @property
    def maxCount1(self):
        if "maxCount1" in self._entity_data:
            return int(self._entity_data.get('maxCount1'))
        return int(-1)

    @property
    def minFactor1(self):
        if "minFactor1" in self._entity_data:
            return int(self._entity_data.get('minFactor1'))
        return int(0)

    @property
    def maxFactor1(self):
        if "maxFactor1" in self._entity_data:
            return int(self._entity_data.get('maxFactor1'))
        return int(0)

    @property
    def defaultDist1(self):
        if "defaultDist1" in self._entity_data:
            return float(self._entity_data.get('defaultDist1'))
        return float(0)

    @property
    def NPCType2(self):
        if "NPCType2" in self._entity_data:
            return self._entity_data.get('NPCType2')
        return None

    @property
    def NPCState2(self):
        if "NPCState2" in self._entity_data:
            return self._entity_data.get('NPCState2')
        return "-1"

    @property
    def invertState2(self):
        if "invertState2" in self._entity_data:
            return bool(self._entity_data.get('invertState2'))
        return bool(0)

    @property
    def minCount2(self):
        if "minCount2" in self._entity_data:
            return int(self._entity_data.get('minCount2'))
        return int(-1)

    @property
    def maxCount2(self):
        if "maxCount2" in self._entity_data:
            return int(self._entity_data.get('maxCount2'))
        return int(-1)

    @property
    def minFactor2(self):
        if "minFactor2" in self._entity_data:
            return int(self._entity_data.get('minFactor2'))
        return int(0)

    @property
    def maxFactor2(self):
        if "maxFactor2" in self._entity_data:
            return int(self._entity_data.get('maxFactor2'))
        return int(0)

    @property
    def defaultDist2(self):
        if "defaultDist2" in self._entity_data:
            return float(self._entity_data.get('defaultDist2'))
        return float(0)

    @property
    def NPCType3(self):
        if "NPCType3" in self._entity_data:
            return self._entity_data.get('NPCType3')
        return None

    @property
    def NPCState3(self):
        if "NPCState3" in self._entity_data:
            return self._entity_data.get('NPCState3')
        return "-1"

    @property
    def invertState3(self):
        if "invertState3" in self._entity_data:
            return bool(self._entity_data.get('invertState3'))
        return bool(0)

    @property
    def minCount3(self):
        if "minCount3" in self._entity_data:
            return int(self._entity_data.get('minCount3'))
        return int(-1)

    @property
    def maxCount3(self):
        if "maxCount3" in self._entity_data:
            return int(self._entity_data.get('maxCount3'))
        return int(-1)

    @property
    def minFactor3(self):
        if "minFactor3" in self._entity_data:
            return int(self._entity_data.get('minFactor3'))
        return int(0)

    @property
    def maxFactor3(self):
        if "maxFactor3" in self._entity_data:
            return int(self._entity_data.get('maxFactor3'))
        return int(0)

    @property
    def defaultDist3(self):
        if "defaultDist3" in self._entity_data:
            return float(self._entity_data.get('defaultDist3'))
        return float(0)


class logic_npc_counter_radius(Targetname, LogicNPCCounterPointBase):
    pass

    icon_sprite = "editor/math_counter.vmat"

    @property
    def distanceMax(self):
        if "distanceMax" in self._entity_data:
            return float(self._entity_data.get('distanceMax'))
        return float(25.0)


class logic_npc_counter_aabb(Targetname, LogicNPCCounterPointBase):
    pass

    icon_sprite = "editor/math_counter.vmat"

    @property
    def box_outer_mins(self):
        if "box_outer_mins" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_outer_mins'))
        return parse_int_vector("-64 -64 -64")

    @property
    def box_outer_maxs(self):
        if "box_outer_maxs" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_outer_maxs'))
        return parse_int_vector("64 64 64")


class logic_npc_counter_obb(logic_npc_counter_aabb):
    pass

    icon_sprite = "editor/math_counter.vmat"


class logic_random_outputs(Targetname, EnableDisable):
    pass

    icon_sprite = "editor/logic_random_outputs.vmat"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Only trigger once': (1, 0), 'Allow fast retrigger': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def OnTriggerChance1(self):
        if "OnTriggerChance1" in self._entity_data:
            return float(self._entity_data.get('OnTriggerChance1'))
        return float(1.0)

    @property
    def OnTriggerChance2(self):
        if "OnTriggerChance2" in self._entity_data:
            return float(self._entity_data.get('OnTriggerChance2'))
        return float(1.0)

    @property
    def OnTriggerChance3(self):
        if "OnTriggerChance3" in self._entity_data:
            return float(self._entity_data.get('OnTriggerChance3'))
        return float(1.0)

    @property
    def OnTriggerChance4(self):
        if "OnTriggerChance4" in self._entity_data:
            return float(self._entity_data.get('OnTriggerChance4'))
        return float(1.0)

    @property
    def OnTriggerChance5(self):
        if "OnTriggerChance5" in self._entity_data:
            return float(self._entity_data.get('OnTriggerChance5'))
        return float(1.0)

    @property
    def OnTriggerChance6(self):
        if "OnTriggerChance6" in self._entity_data:
            return float(self._entity_data.get('OnTriggerChance6'))
        return float(1.0)

    @property
    def OnTriggerChance7(self):
        if "OnTriggerChance7" in self._entity_data:
            return float(self._entity_data.get('OnTriggerChance7'))
        return float(1.0)

    @property
    def OnTriggerChance8(self):
        if "OnTriggerChance8" in self._entity_data:
            return float(self._entity_data.get('OnTriggerChance8'))
        return float(1.0)


class logic_script(Targetname):
    pass

    icon_sprite = "editor/logic_script.vmat"

    @property
    def Group00(self):
        if "Group00" in self._entity_data:
            return self._entity_data.get('Group00')
        return None

    @property
    def Group01(self):
        if "Group01" in self._entity_data:
            return self._entity_data.get('Group01')
        return None

    @property
    def Group02(self):
        if "Group02" in self._entity_data:
            return self._entity_data.get('Group02')
        return None

    @property
    def Group03(self):
        if "Group03" in self._entity_data:
            return self._entity_data.get('Group03')
        return None

    @property
    def Group04(self):
        if "Group04" in self._entity_data:
            return self._entity_data.get('Group04')
        return None

    @property
    def Group05(self):
        if "Group05" in self._entity_data:
            return self._entity_data.get('Group05')
        return None

    @property
    def Group06(self):
        if "Group06" in self._entity_data:
            return self._entity_data.get('Group06')
        return None

    @property
    def Group07(self):
        if "Group07" in self._entity_data:
            return self._entity_data.get('Group07')
        return None

    @property
    def Group08(self):
        if "Group08" in self._entity_data:
            return self._entity_data.get('Group08')
        return None

    @property
    def Group09(self):
        if "Group09" in self._entity_data:
            return self._entity_data.get('Group09')
        return None

    @property
    def Group10(self):
        if "Group10" in self._entity_data:
            return self._entity_data.get('Group10')
        return None

    @property
    def Group11(self):
        if "Group11" in self._entity_data:
            return self._entity_data.get('Group11')
        return None

    @property
    def Group12(self):
        if "Group12" in self._entity_data:
            return self._entity_data.get('Group12')
        return None

    @property
    def Group13(self):
        if "Group13" in self._entity_data:
            return self._entity_data.get('Group13')
        return None

    @property
    def Group14(self):
        if "Group14" in self._entity_data:
            return self._entity_data.get('Group14')
        return None

    @property
    def Group15(self):
        if "Group15" in self._entity_data:
            return self._entity_data.get('Group15')
        return None


class logic_relay(Targetname, EnableDisable):
    pass

    icon_sprite = "editor/logic_relay.vmat"

    @property
    def TriggerOnce(self):
        if "TriggerOnce" in self._entity_data:
            return bool(self._entity_data.get('TriggerOnce'))
        return bool(0)

    @property
    def FastRetrigger(self):
        if "FastRetrigger" in self._entity_data:
            return bool(self._entity_data.get('FastRetrigger'))
        return bool(0)

    @property
    def PassthroughCaller(self):
        if "PassthroughCaller" in self._entity_data:
            return bool(self._entity_data.get('PassthroughCaller'))
        return bool(0)


class logic_timer(Targetname, EnableDisable):
    pass

    icon_sprite = "editor/logic_timer.vmat"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {
                'Oscillator (alternates between OnTimerHigh and OnTimerLow outputs)': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def UseRandomTime(self):
        if "UseRandomTime" in self._entity_data:
            return bool(self._entity_data.get('UseRandomTime'))
        return bool(0)

    @property
    def LowerRandomBound(self):
        if "LowerRandomBound" in self._entity_data:
            return float(self._entity_data.get('LowerRandomBound'))
        return float(0)

    @property
    def UpperRandomBound(self):
        if "UpperRandomBound" in self._entity_data:
            return float(self._entity_data.get('UpperRandomBound'))
        return float(0)

    @property
    def RefireTime(self):
        if "RefireTime" in self._entity_data:
            return float(self._entity_data.get('RefireTime'))
        return float(0)

    @property
    def InitialDelay(self):
        if "InitialDelay" in self._entity_data:
            return float(self._entity_data.get('InitialDelay'))
        return float(0)


class hammer_updateignorelist(Targetname):
    pass

    @property
    def IgnoredName01(self):
        if "IgnoredName01" in self._entity_data:
            return self._entity_data.get('IgnoredName01')
        return ""

    @property
    def IgnoredName02(self):
        if "IgnoredName02" in self._entity_data:
            return self._entity_data.get('IgnoredName02')
        return ""

    @property
    def IgnoredName03(self):
        if "IgnoredName03" in self._entity_data:
            return self._entity_data.get('IgnoredName03')
        return ""

    @property
    def IgnoredName04(self):
        if "IgnoredName04" in self._entity_data:
            return self._entity_data.get('IgnoredName04')
        return ""

    @property
    def IgnoredName05(self):
        if "IgnoredName05" in self._entity_data:
            return self._entity_data.get('IgnoredName05')
        return ""

    @property
    def IgnoredName06(self):
        if "IgnoredName06" in self._entity_data:
            return self._entity_data.get('IgnoredName06')
        return ""

    @property
    def IgnoredName07(self):
        if "IgnoredName07" in self._entity_data:
            return self._entity_data.get('IgnoredName07')
        return ""

    @property
    def IgnoredName08(self):
        if "IgnoredName08" in self._entity_data:
            return self._entity_data.get('IgnoredName08')
        return ""

    @property
    def IgnoredName09(self):
        if "IgnoredName09" in self._entity_data:
            return self._entity_data.get('IgnoredName09')
        return ""

    @property
    def IgnoredName10(self):
        if "IgnoredName10" in self._entity_data:
            return self._entity_data.get('IgnoredName10')
        return ""

    @property
    def IgnoredName11(self):
        if "IgnoredName11" in self._entity_data:
            return self._entity_data.get('IgnoredName11')
        return ""

    @property
    def IgnoredName12(self):
        if "IgnoredName12" in self._entity_data:
            return self._entity_data.get('IgnoredName12')
        return ""

    @property
    def IgnoredName13(self):
        if "IgnoredName13" in self._entity_data:
            return self._entity_data.get('IgnoredName13')
        return ""

    @property
    def IgnoredName14(self):
        if "IgnoredName14" in self._entity_data:
            return self._entity_data.get('IgnoredName14')
        return ""

    @property
    def IgnoredName15(self):
        if "IgnoredName15" in self._entity_data:
            return self._entity_data.get('IgnoredName15')
        return ""

    @property
    def IgnoredName16(self):
        if "IgnoredName16" in self._entity_data:
            return self._entity_data.get('IgnoredName16')
        return ""


class logic_collision_pair(Targetname):
    pass

    @property
    def attach1(self):
        if "attach1" in self._entity_data:
            return self._entity_data.get('attach1')
        return ""

    @property
    def attach2(self):
        if "attach2" in self._entity_data:
            return self._entity_data.get('attach2')
        return ""

    @property
    def startdisabled(self):
        if "startdisabled" in self._entity_data:
            return bool(self._entity_data.get('startdisabled'))
        return bool(1)


class env_microphone(Targetname, Parentname, EnableDisable):
    pass

    icon_sprite = "editor/env_microphone.vmat"

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def SpeakerName(self):
        if "SpeakerName" in self._entity_data:
            return self._entity_data.get('SpeakerName')
        return ""

    @property
    def ListenFilter(self):
        if "ListenFilter" in self._entity_data:
            return self._entity_data.get('ListenFilter')
        return ""

    @property
    def speaker_dsp_preset(self):
        if "speaker_dsp_preset" in self._entity_data:
            return self._entity_data.get('speaker_dsp_preset')
        return "0"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Hears combat sounds': (1, 1), 'Hears world sounds': (2, 1),
                                   'Hears player sounds': (4, 1), 'Hears bullet impacts': (8, 1),
                                   'Swallows sounds routed through speakers': (16, 0), 'Hears explosions': (32, 0),
                                   'Ignores non-attenuated sounds': (64, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def Sensitivity(self):
        if "Sensitivity" in self._entity_data:
            return float(self._entity_data.get('Sensitivity'))
        return float(1)

    @property
    def SmoothFactor(self):
        if "SmoothFactor" in self._entity_data:
            return float(self._entity_data.get('SmoothFactor'))
        return float(0)

    @property
    def MaxRange(self):
        if "MaxRange" in self._entity_data:
            return float(self._entity_data.get('MaxRange'))
        return float(240)


class math_remap(Targetname, EnableDisable):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Ignore out of range input values': (1, 1),
                                   'Clamp out of range output values': (2, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def in1(self):
        if "in1" in self._entity_data:
            return int(self._entity_data.get('in1'))
        return int(0)

    @property
    def in2(self):
        if "in2" in self._entity_data:
            return int(self._entity_data.get('in2'))
        return int(1)

    @property
    def out1(self):
        if "out1" in self._entity_data:
            return int(self._entity_data.get('out1'))
        return int(0)

    @property
    def out2(self):
        if "out2" in self._entity_data:
            return int(self._entity_data.get('out2'))
        return int(0)


class math_colorblend(Targetname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Ignore out of range input values': (1, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def inmin(self):
        if "inmin" in self._entity_data:
            return int(self._entity_data.get('inmin'))
        return int(0)

    @property
    def inmax(self):
        if "inmax" in self._entity_data:
            return int(self._entity_data.get('inmax'))
        return int(1)

    @property
    def colormin(self):
        if "colormin" in self._entity_data:
            return parse_int_vector(self._entity_data.get('colormin'))
        return parse_int_vector("0 0 0")

    @property
    def colormax(self):
        if "colormax" in self._entity_data:
            return parse_int_vector(self._entity_data.get('colormax'))
        return parse_int_vector("255 255 255")


class math_counter(Targetname, EnableDisable):
    pass

    icon_sprite = "editor/math_counter.vmat"

    @property
    def startvalue(self):
        if "startvalue" in self._entity_data:
            return int(self._entity_data.get('startvalue'))
        return int(0)

    @property
    def min(self):
        if "min" in self._entity_data:
            return int(self._entity_data.get('min'))
        return int(0)

    @property
    def max(self):
        if "max" in self._entity_data:
            return int(self._entity_data.get('max'))
        return int(0)


class logic_lineto(Targetname):
    pass

    @property
    def source(self):
        if "source" in self._entity_data:
            return self._entity_data.get('source')
        return None

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None


class logic_navigation(Targetname):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return "Name of the entity to set navigation properties on."

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start On': (1, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def navprop(self):
        if "navprop" in self._entity_data:
            return self._entity_data.get('navprop')
        return "Ignore"


class logic_autosave(Targetname):
    pass

    icon_sprite = "editor/logic_autosave.vmat"

    @property
    def NewLevelUnit(self):
        if "NewLevelUnit" in self._entity_data:
            return bool(self._entity_data.get('NewLevelUnit'))
        return bool(0)

    @property
    def MinimumHitPoints(self):
        if "MinimumHitPoints" in self._entity_data:
            return int(self._entity_data.get('MinimumHitPoints'))
        return int(0)

    @property
    def MinHitPointsToCommit(self):
        if "MinHitPointsToCommit" in self._entity_data:
            return int(self._entity_data.get('MinHitPointsToCommit'))
        return int(0)


class logic_active_autosave(Targetname):
    pass

    @property
    def MinimumHitPoints(self):
        if "MinimumHitPoints" in self._entity_data:
            return int(self._entity_data.get('MinimumHitPoints'))
        return int(30)

    @property
    def TriggerHitPoints(self):
        if "TriggerHitPoints" in self._entity_data:
            return int(self._entity_data.get('TriggerHitPoints'))
        return int(75)

    @property
    def TimeToTrigget(self):
        if "TimeToTrigget" in self._entity_data:
            return float(self._entity_data.get('TimeToTrigget'))
        return float(0)

    @property
    def DangerousTime(self):
        if "DangerousTime" in self._entity_data:
            return float(self._entity_data.get('DangerousTime'))
        return float(10)


class logic_playmovie(Targetname):
    pass

    @property
    def MovieFilename(self):
        if "MovieFilename" in self._entity_data:
            return self._entity_data.get('MovieFilename')
        return ""

    @property
    def allowskip(self):
        if "allowskip" in self._entity_data:
            return bool(self._entity_data.get('allowskip'))
        return bool(0)


class info_world_layer(Targetname):
    pass

    icon_sprite = "editor/info_world_layer.vmat"

    @property
    def layerName(self):
        if "layerName" in self._entity_data:
            return self._entity_data.get('layerName')
        return ""

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Visible on spawn': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def childSpawnGroup(self):
        if "childSpawnGroup" in self._entity_data:
            return self._entity_data.get('childSpawnGroup')
        return "0"


class point_template(Targetname):
    pass

    icon_sprite = "editor/point_template.vmat"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {"Don't remove template entities": (1, 0),
                                   "Preserve entity names (Don't do name fixup)": (2, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def asynchronous(self):
        if "asynchronous" in self._entity_data:
            return bool(self._entity_data.get('asynchronous'))
        return bool(0)

    @property
    def timeoutInterval(self):
        if "timeoutInterval" in self._entity_data:
            return float(self._entity_data.get('timeoutInterval'))
        return float(0)

    @property
    def entityFilterName(self):
        if "entityFilterName" in self._entity_data:
            return self._entity_data.get('entityFilterName')
        return ""

    @property
    def spawnGroupType(self):
        if "spawnGroupType" in self._entity_data:
            return self._entity_data.get('spawnGroupType')
        return "INSERT_INTO_POINT_TEMPLATE_SPAWN_GROUP"

    @property
    def clientOnlyEntityBehavior(self):
        if "clientOnlyEntityBehavior" in self._entity_data:
            return self._entity_data.get('clientOnlyEntityBehavior')
        return "CREATE_FOR_CURRENTLY_CONNECTED_CLIENTS_ONLY"

    @property
    def Template01(self):
        if "Template01" in self._entity_data:
            return self._entity_data.get('Template01')
        return None

    @property
    def Template02(self):
        if "Template02" in self._entity_data:
            return self._entity_data.get('Template02')
        return None

    @property
    def Template03(self):
        if "Template03" in self._entity_data:
            return self._entity_data.get('Template03')
        return None

    @property
    def Template04(self):
        if "Template04" in self._entity_data:
            return self._entity_data.get('Template04')
        return None

    @property
    def Template05(self):
        if "Template05" in self._entity_data:
            return self._entity_data.get('Template05')
        return None

    @property
    def Template06(self):
        if "Template06" in self._entity_data:
            return self._entity_data.get('Template06')
        return None

    @property
    def Template07(self):
        if "Template07" in self._entity_data:
            return self._entity_data.get('Template07')
        return None

    @property
    def Template08(self):
        if "Template08" in self._entity_data:
            return self._entity_data.get('Template08')
        return None

    @property
    def Template09(self):
        if "Template09" in self._entity_data:
            return self._entity_data.get('Template09')
        return None

    @property
    def Template10(self):
        if "Template10" in self._entity_data:
            return self._entity_data.get('Template10')
        return None

    @property
    def Template11(self):
        if "Template11" in self._entity_data:
            return self._entity_data.get('Template11')
        return None

    @property
    def Template12(self):
        if "Template12" in self._entity_data:
            return self._entity_data.get('Template12')
        return None

    @property
    def Template13(self):
        if "Template13" in self._entity_data:
            return self._entity_data.get('Template13')
        return None

    @property
    def Template14(self):
        if "Template14" in self._entity_data:
            return self._entity_data.get('Template14')
        return None

    @property
    def Template15(self):
        if "Template15" in self._entity_data:
            return self._entity_data.get('Template15')
        return None

    @property
    def Template16(self):
        if "Template16" in self._entity_data:
            return self._entity_data.get('Template16')
        return None

    @property
    def Template17(self):
        if "Template17" in self._entity_data:
            return self._entity_data.get('Template17')
        return None

    @property
    def Template18(self):
        if "Template18" in self._entity_data:
            return self._entity_data.get('Template18')
        return None

    @property
    def Template19(self):
        if "Template19" in self._entity_data:
            return self._entity_data.get('Template19')
        return None

    @property
    def Template20(self):
        if "Template20" in self._entity_data:
            return self._entity_data.get('Template20')
        return None

    @property
    def Template21(self):
        if "Template21" in self._entity_data:
            return self._entity_data.get('Template21')
        return None

    @property
    def Template22(self):
        if "Template22" in self._entity_data:
            return self._entity_data.get('Template22')
        return None

    @property
    def Template23(self):
        if "Template23" in self._entity_data:
            return self._entity_data.get('Template23')
        return None

    @property
    def Template24(self):
        if "Template24" in self._entity_data:
            return self._entity_data.get('Template24')
        return None

    @property
    def Template25(self):
        if "Template25" in self._entity_data:
            return self._entity_data.get('Template25')
        return None

    @property
    def Template26(self):
        if "Template26" in self._entity_data:
            return self._entity_data.get('Template26')
        return None

    @property
    def Template27(self):
        if "Template27" in self._entity_data:
            return self._entity_data.get('Template27')
        return None

    @property
    def Template28(self):
        if "Template28" in self._entity_data:
            return self._entity_data.get('Template28')
        return None

    @property
    def Template29(self):
        if "Template29" in self._entity_data:
            return self._entity_data.get('Template29')
        return None

    @property
    def Template30(self):
        if "Template30" in self._entity_data:
            return self._entity_data.get('Template30')
        return None

    @property
    def Template31(self):
        if "Template31" in self._entity_data:
            return self._entity_data.get('Template31')
        return None

    @property
    def Template32(self):
        if "Template32" in self._entity_data:
            return self._entity_data.get('Template32')
        return None

    @property
    def Template33(self):
        if "Template33" in self._entity_data:
            return self._entity_data.get('Template33')
        return None

    @property
    def Template34(self):
        if "Template34" in self._entity_data:
            return self._entity_data.get('Template34')
        return None

    @property
    def Template35(self):
        if "Template35" in self._entity_data:
            return self._entity_data.get('Template35')
        return None

    @property
    def Template36(self):
        if "Template36" in self._entity_data:
            return self._entity_data.get('Template36')
        return None

    @property
    def Template37(self):
        if "Template37" in self._entity_data:
            return self._entity_data.get('Template37')
        return None

    @property
    def Template38(self):
        if "Template38" in self._entity_data:
            return self._entity_data.get('Template38')
        return None

    @property
    def Template39(self):
        if "Template39" in self._entity_data:
            return self._entity_data.get('Template39')
        return None

    @property
    def Template40(self):
        if "Template40" in self._entity_data:
            return self._entity_data.get('Template40')
        return None

    @property
    def Template41(self):
        if "Template41" in self._entity_data:
            return self._entity_data.get('Template41')
        return None

    @property
    def Template42(self):
        if "Template42" in self._entity_data:
            return self._entity_data.get('Template42')
        return None

    @property
    def Template43(self):
        if "Template43" in self._entity_data:
            return self._entity_data.get('Template43')
        return None

    @property
    def Template44(self):
        if "Template44" in self._entity_data:
            return self._entity_data.get('Template44')
        return None

    @property
    def Template45(self):
        if "Template45" in self._entity_data:
            return self._entity_data.get('Template45')
        return None

    @property
    def Template46(self):
        if "Template46" in self._entity_data:
            return self._entity_data.get('Template46')
        return None

    @property
    def Template47(self):
        if "Template47" in self._entity_data:
            return self._entity_data.get('Template47')
        return None

    @property
    def Template48(self):
        if "Template48" in self._entity_data:
            return self._entity_data.get('Template48')
        return None

    @property
    def Template49(self):
        if "Template49" in self._entity_data:
            return self._entity_data.get('Template49')
        return None

    @property
    def Template50(self):
        if "Template50" in self._entity_data:
            return self._entity_data.get('Template50')
        return None

    @property
    def Template51(self):
        if "Template51" in self._entity_data:
            return self._entity_data.get('Template51')
        return None

    @property
    def Template52(self):
        if "Template52" in self._entity_data:
            return self._entity_data.get('Template52')
        return None

    @property
    def Template53(self):
        if "Template53" in self._entity_data:
            return self._entity_data.get('Template53')
        return None

    @property
    def Template54(self):
        if "Template54" in self._entity_data:
            return self._entity_data.get('Template54')
        return None

    @property
    def Template55(self):
        if "Template55" in self._entity_data:
            return self._entity_data.get('Template55')
        return None

    @property
    def Template56(self):
        if "Template56" in self._entity_data:
            return self._entity_data.get('Template56')
        return None

    @property
    def Template57(self):
        if "Template57" in self._entity_data:
            return self._entity_data.get('Template57')
        return None

    @property
    def Template58(self):
        if "Template58" in self._entity_data:
            return self._entity_data.get('Template58')
        return None

    @property
    def Template59(self):
        if "Template59" in self._entity_data:
            return self._entity_data.get('Template59')
        return None

    @property
    def Template60(self):
        if "Template60" in self._entity_data:
            return self._entity_data.get('Template60')
        return None

    @property
    def Template61(self):
        if "Template61" in self._entity_data:
            return self._entity_data.get('Template61')
        return None

    @property
    def Template62(self):
        if "Template62" in self._entity_data:
            return self._entity_data.get('Template62')
        return None

    @property
    def Template63(self):
        if "Template63" in self._entity_data:
            return self._entity_data.get('Template63')
        return None

    @property
    def Template64(self):
        if "Template64" in self._entity_data:
            return self._entity_data.get('Template64')
        return None


class env_entity_maker(Targetname, Parentname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {"Enable AutoSpawn (will spawn whenever there's room)": (1, 0),
                                   'AutoSpawn: Wait for entity destruction': (2, 0),
                                   'AutoSpawn: Even if the player is looking': (4, 0),
                                   "ForceSpawn: Only if there's room": (8, 0),
                                   "ForceSpawn: Only if the player isn't looking": (16, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def EntityTemplate(self):
        if "EntityTemplate" in self._entity_data:
            return self._entity_data.get('EntityTemplate')
        return ""

    @property
    def PostSpawnSpeed(self):
        if "PostSpawnSpeed" in self._entity_data:
            return float(self._entity_data.get('PostSpawnSpeed'))
        return float(0)

    @property
    def PostSpawnDirection(self):
        if "PostSpawnDirection" in self._entity_data:
            return parse_int_vector(self._entity_data.get('PostSpawnDirection'))
        return parse_int_vector("0 0 0")

    @property
    def PostSpawnDirectionVariance(self):
        if "PostSpawnDirectionVariance" in self._entity_data:
            return float(self._entity_data.get('PostSpawnDirectionVariance'))
        return float(0.15)

    @property
    def PostSpawnInheritAngles(self):
        if "PostSpawnInheritAngles" in self._entity_data:
            return bool(self._entity_data.get('PostSpawnInheritAngles'))
        return bool(0)


class BaseFilter(Targetname):
    pass

    @property
    def Negated(self):
        if "Negated" in self._entity_data:
            return self._entity_data.get('Negated')
        return "0"


class filter_multi(BaseFilter):
    pass

    icon_sprite = "editor/filter_multiple.vmat"

    @property
    def filtertype(self):
        if "filtertype" in self._entity_data:
            return self._entity_data.get('filtertype')
        return "0"

    @property
    def Negated(self):
        if "Negated" in self._entity_data:
            return self._entity_data.get('Negated')
        return "0"

    @property
    def Filter01(self):
        if "Filter01" in self._entity_data:
            return self._entity_data.get('Filter01')
        return None

    @property
    def Filter02(self):
        if "Filter02" in self._entity_data:
            return self._entity_data.get('Filter02')
        return None

    @property
    def Filter03(self):
        if "Filter03" in self._entity_data:
            return self._entity_data.get('Filter03')
        return None

    @property
    def Filter04(self):
        if "Filter04" in self._entity_data:
            return self._entity_data.get('Filter04')
        return None

    @property
    def Filter05(self):
        if "Filter05" in self._entity_data:
            return self._entity_data.get('Filter05')
        return None

    @property
    def Filter06(self):
        if "Filter06" in self._entity_data:
            return self._entity_data.get('Filter06')
        return None

    @property
    def Filter07(self):
        if "Filter07" in self._entity_data:
            return self._entity_data.get('Filter07')
        return None

    @property
    def Filter08(self):
        if "Filter08" in self._entity_data:
            return self._entity_data.get('Filter08')
        return None

    @property
    def Filter09(self):
        if "Filter09" in self._entity_data:
            return self._entity_data.get('Filter09')
        return None

    @property
    def Filter10(self):
        if "Filter10" in self._entity_data:
            return self._entity_data.get('Filter10')
        return None


class filter_activator_name(BaseFilter):
    pass

    icon_sprite = "editor/filter_name.vmat"

    @property
    def filtername(self):
        if "filtername" in self._entity_data:
            return self._entity_data.get('filtername')
        return None


class filter_activator_model(BaseFilter):
    pass

    icon_sprite = "editor/filter_name.vmat"

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return None


class filter_activator_context(BaseFilter):
    pass

    icon_sprite = "editor/filter_name.vmat"

    @property
    def ResponseContext(self):
        if "ResponseContext" in self._entity_data:
            return self._entity_data.get('ResponseContext')
        return None


class filter_activator_class(BaseFilter):
    pass

    icon_sprite = "editor/filter_class.vmat"

    @property
    def filterclass(self):
        if "filterclass" in self._entity_data:
            return self._entity_data.get('filterclass')
        return None


class filter_activator_mass_greater(BaseFilter):
    pass

    icon_sprite = "editor/filter_class.vmat"

    @property
    def filtermass(self):
        if "filtermass" in self._entity_data:
            return float(self._entity_data.get('filtermass'))
        return float(0)


class filter_damage_type(BaseFilter):
    pass

    icon_sprite = "editor/filter_type.vmat"

    @property
    def damagetype(self):
        if "damagetype" in self._entity_data:
            return self._entity_data.get('damagetype')
        return "64"


class filter_enemy(BaseFilter):
    pass

    icon_sprite = "editor/filter_class.vmat"

    @property
    def filtername(self):
        if "filtername" in self._entity_data:
            return self._entity_data.get('filtername')
        return None

    @property
    def filter_radius(self):
        if "filter_radius" in self._entity_data:
            return float(self._entity_data.get('filter_radius'))
        return float(0)

    @property
    def filter_outer_radius(self):
        if "filter_outer_radius" in self._entity_data:
            return float(self._entity_data.get('filter_outer_radius'))
        return float(0)

    @property
    def filter_max_per_enemy(self):
        if "filter_max_per_enemy" in self._entity_data:
            return int(self._entity_data.get('filter_max_per_enemy'))
        return int(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Do not lose target if already aquired but filter failed.': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class filter_proximity(BaseFilter):
    pass

    icon_sprite = "editor/filter_class.vmat"

    @property
    def filter_radius(self):
        if "filter_radius" in self._entity_data:
            return float(self._entity_data.get('filter_radius'))
        return float(0)


class filter_los(BaseFilter):
    pass

    icon_sprite = "editor/filter_class.vmat"


class point_anglesensor(Targetname, Parentname, EnableDisable):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def lookatname(self):
        if "lookatname" in self._entity_data:
            return self._entity_data.get('lookatname')
        return None

    @property
    def duration(self):
        if "duration" in self._entity_data:
            return float(self._entity_data.get('duration'))
        return float(0)

    @property
    def tolerance(self):
        if "tolerance" in self._entity_data:
            return int(self._entity_data.get('tolerance'))
        return int(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {"Use target entity's angles (NOT position)": (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class point_angularvelocitysensor(Targetname):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def threshold(self):
        if "threshold" in self._entity_data:
            return float(self._entity_data.get('threshold'))
        return float(0)

    @property
    def fireinterval(self):
        if "fireinterval" in self._entity_data:
            return float(self._entity_data.get('fireinterval'))
        return float(0.2)

    @property
    def axis(self):
        if "axis" in self._entity_data:
            return parse_int_vector(self._entity_data.get('axis'))
        return parse_int_vector("None")

    @property
    def usehelper(self):
        if "usehelper" in self._entity_data:
            return bool(self._entity_data.get('usehelper'))
        return bool(0)


class point_velocitysensor(Targetname):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def axis(self):
        if "axis" in self._entity_data:
            return parse_int_vector(self._entity_data.get('axis'))
        return parse_int_vector("None")

    @property
    def enabled(self):
        if "enabled" in self._entity_data:
            return bool(self._entity_data.get('enabled'))
        return bool(1)

    @property
    def avginterval(self):
        if "avginterval" in self._entity_data:
            return float(self._entity_data.get('avginterval'))
        return float(0)


class point_proximity_sensor(Targetname, Parentname, EnableDisable):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {
                'Test the distance as measured along the axis specified by our direction.': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class point_teleport(Targetname, Parentname):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Teleport Home': (1, 0), 'Into Duck (episodic)': (2, 0),
                                   'Change View Direction (VR)': (4, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def teleport_parented_entities(self):
        if "teleport_parented_entities" in self._entity_data:
            return bool(self._entity_data.get('teleport_parented_entities'))
        return bool(0)


class point_hurt(Targetname):
    pass

    @property
    def DamageTarget(self):
        if "DamageTarget" in self._entity_data:
            return self._entity_data.get('DamageTarget')
        return ""

    @property
    def DamageRadius(self):
        if "DamageRadius" in self._entity_data:
            return float(self._entity_data.get('DamageRadius'))
        return float(256)

    @property
    def Damage(self):
        if "Damage" in self._entity_data:
            return int(self._entity_data.get('Damage'))
        return int(5)

    @property
    def DamageDelay(self):
        if "DamageDelay" in self._entity_data:
            return float(self._entity_data.get('DamageDelay'))
        return float(1)

    @property
    def DamageType(self):
        if "DamageType" in self._entity_data:
            return self._entity_data.get('DamageType')
        return "0"


class point_playermoveconstraint(Targetname):
    pass

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return float(self._entity_data.get('radius'))
        return float(256)

    @property
    def width(self):
        if "width" in self._entity_data:
            return float(self._entity_data.get('width'))
        return float(75.0)

    @property
    def speedfactor(self):
        if "speedfactor" in self._entity_data:
            return float(self._entity_data.get('speedfactor'))
        return float(0.15)


class BasePhysicsSimulated:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def skipPreSettle(self):
        if "skipPreSettle" in self._entity_data:
            return bool(self._entity_data.get('skipPreSettle'))
        return bool(0)


class BasePhysicsNoSettleAttached:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data


class func_physbox(BreakableBrush, RenderFields, BasePhysicsSimulated):
    pass

    @property
    def _minlight(self):
        if "_minlight" in self._entity_data:
            return self._entity_data.get('_minlight')
        return None

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {"Debris - Don't collide with the player or other debris": (16384, 0),
                                   'Motion Disabled': (32768, 0), 'Use Preferred Carry Angles': (65536, 0),
                                   'Enable motion on Physcannon grab': (131072, 0),
                                   'Ignore +USE for Pickup': (262144, 0), 'Generate output on +USE ': (524288, 1),
                                   'Start Asleep': (1048576, 0),
                                   'Physgun is NOT allowed to pick this up.': (2097152, 0),
                                   'Physgun is NOT allowed to punt this object.': (4194304, 0),
                                   'Prevent motion enable on player bump': (8388608, 0),
                                   'Force nav-ignore': (16777216, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def Damagetype(self):
        if "Damagetype" in self._entity_data:
            return self._entity_data.get('Damagetype')
        return "0"

    @property
    def massScale(self):
        if "massScale" in self._entity_data:
            return float(self._entity_data.get('massScale'))
        return float(0)

    @property
    def damagetoenablemotion(self):
        if "damagetoenablemotion" in self._entity_data:
            return int(self._entity_data.get('damagetoenablemotion'))
        return int(0)

    @property
    def forcetoenablemotion(self):
        if "forcetoenablemotion" in self._entity_data:
            return float(self._entity_data.get('forcetoenablemotion'))
        return float(0)

    @property
    def health(self):
        if "health" in self._entity_data:
            return int(self._entity_data.get('health'))
        return int(0)

    @property
    def preferredcarryangles(self):
        if "preferredcarryangles" in self._entity_data:
            return parse_int_vector(self._entity_data.get('preferredcarryangles'))
        return parse_int_vector("0 0 0")

    @property
    def notsolid(self):
        if "notsolid" in self._entity_data:
            return self._entity_data.get('notsolid')
        return "0"

    @property
    def ExploitableByPlayer(self):
        if "ExploitableByPlayer" in self._entity_data:
            return self._entity_data.get('ExploitableByPlayer')
        return "0"

    @property
    def touchoutputperentitydelay(self):
        if "touchoutputperentitydelay" in self._entity_data:
            return float(self._entity_data.get('touchoutputperentitydelay'))
        return float(0)


class TwoObjectPhysics(Targetname, BasePhysicsNoSettleAttached):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'No Collision until break': (1, 0), 'Start inactive': (4, 0),
                                   'Change mass to keep stable attachment to world': (8, 0),
                                   'Do not connect entities until turned on': (16, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def attach1(self):
        if "attach1" in self._entity_data:
            return self._entity_data.get('attach1')
        return ""

    @property
    def attach2(self):
        if "attach2" in self._entity_data:
            return self._entity_data.get('attach2')
        return ""

    @property
    def forcelimit(self):
        if "forcelimit" in self._entity_data:
            return float(self._entity_data.get('forcelimit'))
        return float(0)

    @property
    def torquelimit(self):
        if "torquelimit" in self._entity_data:
            return float(self._entity_data.get('torquelimit'))
        return float(0)

    @property
    def breaksound(self):
        if "breaksound" in self._entity_data:
            return self._entity_data.get('breaksound')
        return ""

    @property
    def teleportfollowdistance(self):
        if "teleportfollowdistance" in self._entity_data:
            return float(self._entity_data.get('teleportfollowdistance'))
        return float(0)


class phys_keepupright(Targetname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start inactive': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def attach1(self):
        if "attach1" in self._entity_data:
            return self._entity_data.get('attach1')
        return ""

    @property
    def angularlimit(self):
        if "angularlimit" in self._entity_data:
            return float(self._entity_data.get('angularlimit'))
        return float(15)


class physics_cannister(Targetname):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return ""

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start Asleep': (1, 0), 'Explodes': (2, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def expdamage(self):
        if "expdamage" in self._entity_data:
            return self._entity_data.get('expdamage')
        return "200.0"

    @property
    def expradius(self):
        if "expradius" in self._entity_data:
            return self._entity_data.get('expradius')
        return "250.0"

    @property
    def health(self):
        if "health" in self._entity_data:
            return int(self._entity_data.get('health'))
        return int(25)

    @property
    def thrust(self):
        if "thrust" in self._entity_data:
            return self._entity_data.get('thrust')
        return "3000.0"

    @property
    def fuel(self):
        if "fuel" in self._entity_data:
            return self._entity_data.get('fuel')
        return "12.0"

    @property
    def rendercolor(self):
        if "rendercolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('rendercolor'))
        return parse_int_vector("255 255 255")

    @property
    def renderamt(self):
        if "renderamt" in self._entity_data:
            return int(self._entity_data.get('renderamt'))
        return int(128)

    @property
    def gassound(self):
        if "gassound" in self._entity_data:
            return self._entity_data.get('gassound')
        return "ambient/objects/cannister_loop.wav"


class info_constraint_anchor(Targetname, Parentname):
    pass

    @property
    def massScale(self):
        if "massScale" in self._entity_data:
            return float(self._entity_data.get('massScale'))
        return float(1)


class info_mass_center:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return ""


class phys_spring(Targetname, BasePhysicsNoSettleAttached):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Force only on stretch': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def attach1(self):
        if "attach1" in self._entity_data:
            return self._entity_data.get('attach1')
        return ""

    @property
    def attach2(self):
        if "attach2" in self._entity_data:
            return self._entity_data.get('attach2')
        return ""

    @property
    def springaxis(self):
        if "springaxis" in self._entity_data:
            return parse_int_vector(self._entity_data.get('springaxis'))
        return parse_int_vector("")

    @property
    def length(self):
        if "length" in self._entity_data:
            return self._entity_data.get('length')
        return "0"

    @property
    def frequency(self):
        if "frequency" in self._entity_data:
            return self._entity_data.get('frequency')
        return "5"

    @property
    def damping(self):
        if "damping" in self._entity_data:
            return self._entity_data.get('damping')
        return "0.7"


class ConstraintSoundInfo:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def minSoundThreshold(self):
        if "minSoundThreshold" in self._entity_data:
            return float(self._entity_data.get('minSoundThreshold'))
        return float(6)

    @property
    def maxSoundThreshold(self):
        if "maxSoundThreshold" in self._entity_data:
            return float(self._entity_data.get('maxSoundThreshold'))
        return float(80)

    @property
    def slidesoundfwd(self):
        if "slidesoundfwd" in self._entity_data:
            return self._entity_data.get('slidesoundfwd')
        return ""

    @property
    def slidesoundback(self):
        if "slidesoundback" in self._entity_data:
            return self._entity_data.get('slidesoundback')
        return ""

    @property
    def reversalsoundthresholdSmall(self):
        if "reversalsoundthresholdSmall" in self._entity_data:
            return float(self._entity_data.get('reversalsoundthresholdSmall'))
        return float(0)

    @property
    def reversalsoundthresholdMedium(self):
        if "reversalsoundthresholdMedium" in self._entity_data:
            return float(self._entity_data.get('reversalsoundthresholdMedium'))
        return float(0)

    @property
    def reversalsoundthresholdLarge(self):
        if "reversalsoundthresholdLarge" in self._entity_data:
            return float(self._entity_data.get('reversalsoundthresholdLarge'))
        return float(0)

    @property
    def reversalsoundSmall(self):
        if "reversalsoundSmall" in self._entity_data:
            return self._entity_data.get('reversalsoundSmall')
        return ""

    @property
    def reversalsoundMedium(self):
        if "reversalsoundMedium" in self._entity_data:
            return self._entity_data.get('reversalsoundMedium')
        return ""

    @property
    def reversalsoundLarge(self):
        if "reversalsoundLarge" in self._entity_data:
            return self._entity_data.get('reversalsoundLarge')
        return ""


class phys_hinge(TwoObjectPhysics, ConstraintSoundInfo):
    pass

    @property
    def hingefriction(self):
        if "hingefriction" in self._entity_data:
            return float(self._entity_data.get('hingefriction'))
        return float(0)

    @property
    def min_rotation(self):
        if "min_rotation" in self._entity_data:
            return float(self._entity_data.get('min_rotation'))
        return float(0)

    @property
    def max_rotation(self):
        if "max_rotation" in self._entity_data:
            return float(self._entity_data.get('max_rotation'))
        return float(0)

    @property
    def initial_rotation(self):
        if "initial_rotation" in self._entity_data:
            return float(self._entity_data.get('initial_rotation'))
        return float(0)

    @property
    def hingeaxis(self):
        if "hingeaxis" in self._entity_data:
            return parse_int_vector(self._entity_data.get('hingeaxis'))
        return parse_int_vector("None")

    @property
    def motorfrequency(self):
        if "motorfrequency" in self._entity_data:
            return float(self._entity_data.get('motorfrequency'))
        return float(10)

    @property
    def motordampingratio(self):
        if "motordampingratio" in self._entity_data:
            return float(self._entity_data.get('motordampingratio'))
        return float(1)

    @property
    def SystemLoadScale(self):
        if "SystemLoadScale" in self._entity_data:
            return float(self._entity_data.get('SystemLoadScale'))
        return float(1)

    @property
    def AngleSpeedThreshold(self):
        if "AngleSpeedThreshold" in self._entity_data:
            return float(self._entity_data.get('AngleSpeedThreshold'))
        return float(0)


class phys_hinge_local(TwoObjectPhysics, ConstraintSoundInfo):
    pass

    @property
    def hingefriction(self):
        if "hingefriction" in self._entity_data:
            return float(self._entity_data.get('hingefriction'))
        return float(0)

    @property
    def min_rotation(self):
        if "min_rotation" in self._entity_data:
            return float(self._entity_data.get('min_rotation'))
        return float(0)

    @property
    def max_rotation(self):
        if "max_rotation" in self._entity_data:
            return float(self._entity_data.get('max_rotation'))
        return float(0)

    @property
    def initial_rotation(self):
        if "initial_rotation" in self._entity_data:
            return float(self._entity_data.get('initial_rotation'))
        return float(0)

    @property
    def hingeaxis(self):
        if "hingeaxis" in self._entity_data:
            return self._entity_data.get('hingeaxis')
        return None

    @property
    def motorfrequency(self):
        if "motorfrequency" in self._entity_data:
            return float(self._entity_data.get('motorfrequency'))
        return float(10)

    @property
    def motordampingratio(self):
        if "motordampingratio" in self._entity_data:
            return float(self._entity_data.get('motordampingratio'))
        return float(1)

    @property
    def SystemLoadScale(self):
        if "SystemLoadScale" in self._entity_data:
            return float(self._entity_data.get('SystemLoadScale'))
        return float(1)

    @property
    def AngleSpeedThreshold(self):
        if "AngleSpeedThreshold" in self._entity_data:
            return float(self._entity_data.get('AngleSpeedThreshold'))
        return float(0)


class phys_ballsocket(TwoObjectPhysics):
    pass

    icon_sprite = "editor/phys_ballsocket.vmat"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def friction(self):
        if "friction" in self._entity_data:
            return float(self._entity_data.get('friction'))
        return float(0)


class phys_constraint(TwoObjectPhysics):
    pass

    @property
    def linearfrequency(self):
        if "linearfrequency" in self._entity_data:
            return float(self._entity_data.get('linearfrequency'))
        return float(0)

    @property
    def lineardampingratio(self):
        if "lineardampingratio" in self._entity_data:
            return float(self._entity_data.get('lineardampingratio'))
        return float(0)

    @property
    def angularfrequency(self):
        if "angularfrequency" in self._entity_data:
            return float(self._entity_data.get('angularfrequency'))
        return float(0)

    @property
    def angulardampingratio(self):
        if "angulardampingratio" in self._entity_data:
            return float(self._entity_data.get('angulardampingratio'))
        return float(0)

    @property
    def enablelinearconstraint(self):
        if "enablelinearconstraint" in self._entity_data:
            return bool(self._entity_data.get('enablelinearconstraint'))
        return bool(1)

    @property
    def enableangularconstraint(self):
        if "enableangularconstraint" in self._entity_data:
            return bool(self._entity_data.get('enableangularconstraint'))
        return bool(1)


class phys_pulleyconstraint(TwoObjectPhysics):
    pass

    @property
    def addlength(self):
        if "addlength" in self._entity_data:
            return float(self._entity_data.get('addlength'))
        return float(0)

    @property
    def gearratio(self):
        if "gearratio" in self._entity_data:
            return float(self._entity_data.get('gearratio'))
        return float(1)

    @property
    def position2(self):
        if "position2" in self._entity_data:
            return parse_int_vector(self._entity_data.get('position2'))
        return parse_int_vector("None")

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'No Collision until break': (1, 1), 'Keep Rigid': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class phys_slideconstraint(TwoObjectPhysics, ConstraintSoundInfo):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'No Collision until break': (1, 1), 'Limit Endpoints': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def slideaxis(self):
        if "slideaxis" in self._entity_data:
            return parse_int_vector(self._entity_data.get('slideaxis'))
        return parse_int_vector("None")

    @property
    def slidefriction(self):
        if "slidefriction" in self._entity_data:
            return float(self._entity_data.get('slidefriction'))
        return float(0)

    @property
    def SystemLoadScale(self):
        if "SystemLoadScale" in self._entity_data:
            return float(self._entity_data.get('SystemLoadScale'))
        return float(1)

    @property
    def initialoffset(self):
        if "initialoffset" in self._entity_data:
            return float(self._entity_data.get('initialoffset'))
        return float(0)

    @property
    def enablelinearconstraint(self):
        if "enablelinearconstraint" in self._entity_data:
            return bool(self._entity_data.get('enablelinearconstraint'))
        return bool(1)

    @property
    def enableangularconstraint(self):
        if "enableangularconstraint" in self._entity_data:
            return bool(self._entity_data.get('enableangularconstraint'))
        return bool(1)

    @property
    def motorfrequency(self):
        if "motorfrequency" in self._entity_data:
            return float(self._entity_data.get('motorfrequency'))
        return float(10)

    @property
    def motordampingratio(self):
        if "motordampingratio" in self._entity_data:
            return float(self._entity_data.get('motordampingratio'))
        return float(1)

    @property
    def motormaxforcemultiplier(self):
        if "motormaxforcemultiplier" in self._entity_data:
            return float(self._entity_data.get('motormaxforcemultiplier'))
        return float(0)

    @property
    def useEntityPivot(self):
        if "useEntityPivot" in self._entity_data:
            return bool(self._entity_data.get('useEntityPivot'))
        return bool(0)


class phys_lengthconstraint(TwoObjectPhysics):
    pass

    @property
    def addlength(self):
        if "addlength" in self._entity_data:
            return float(self._entity_data.get('addlength'))
        return float(0)

    @property
    def minlength(self):
        if "minlength" in self._entity_data:
            return float(self._entity_data.get('minlength'))
        return float(0)

    @property
    def attachpoint(self):
        if "attachpoint" in self._entity_data:
            return parse_int_vector(self._entity_data.get('attachpoint'))
        return parse_int_vector("The position the rope attaches to object 2")

    @property
    def enablecollision(self):
        if "enablecollision" in self._entity_data:
            return bool(self._entity_data.get('enablecollision'))
        return bool(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'No Collision until break': (1, 1), 'Keep Rigid': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class phys_ragdollconstraint(TwoObjectPhysics):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'No Collision until break': (1, 1),
                                   'Only limit rotation (free movement)': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def xmin(self):
        if "xmin" in self._entity_data:
            return float(self._entity_data.get('xmin'))
        return float(-90)

    @property
    def xmax(self):
        if "xmax" in self._entity_data:
            return float(self._entity_data.get('xmax'))
        return float(90)

    @property
    def ymin(self):
        if "ymin" in self._entity_data:
            return float(self._entity_data.get('ymin'))
        return float(0)

    @property
    def ymax(self):
        if "ymax" in self._entity_data:
            return float(self._entity_data.get('ymax'))
        return float(0)

    @property
    def zmin(self):
        if "zmin" in self._entity_data:
            return float(self._entity_data.get('zmin'))
        return float(0)

    @property
    def zmax(self):
        if "zmax" in self._entity_data:
            return float(self._entity_data.get('zmax'))
        return float(0)

    @property
    def xfriction(self):
        if "xfriction" in self._entity_data:
            return float(self._entity_data.get('xfriction'))
        return float(0)

    @property
    def yfriction(self):
        if "yfriction" in self._entity_data:
            return float(self._entity_data.get('yfriction'))
        return float(0)

    @property
    def zfriction(self):
        if "zfriction" in self._entity_data:
            return float(self._entity_data.get('zfriction'))
        return float(0)


class phys_genericconstraint(TwoObjectPhysics):
    pass

    @property
    def linear_motion_x(self):
        if "linear_motion_x" in self._entity_data:
            return self._entity_data.get('linear_motion_x')
        return "JOINT_MOTION_FREE"

    @property
    def linear_frequency_x(self):
        if "linear_frequency_x" in self._entity_data:
            return float(self._entity_data.get('linear_frequency_x'))
        return float(0)

    @property
    def linear_damping_ratio_x(self):
        if "linear_damping_ratio_x" in self._entity_data:
            return float(self._entity_data.get('linear_damping_ratio_x'))
        return float(0)

    @property
    def forcelimit_x(self):
        if "forcelimit_x" in self._entity_data:
            return float(self._entity_data.get('forcelimit_x'))
        return float(0)

    @property
    def notifyforce_x(self):
        if "notifyforce_x" in self._entity_data:
            return float(self._entity_data.get('notifyforce_x'))
        return float(0)

    @property
    def notifyforcemintime_x(self):
        if "notifyforcemintime_x" in self._entity_data:
            return float(self._entity_data.get('notifyforcemintime_x'))
        return float(0)

    @property
    def breakaftertime_x(self):
        if "breakaftertime_x" in self._entity_data:
            return float(self._entity_data.get('breakaftertime_x'))
        return float(0)

    @property
    def breakaftertimethreshold_x(self):
        if "breakaftertimethreshold_x" in self._entity_data:
            return float(self._entity_data.get('breakaftertimethreshold_x'))
        return float(0)

    @property
    def linear_motion_y(self):
        if "linear_motion_y" in self._entity_data:
            return self._entity_data.get('linear_motion_y')
        return "JOINT_MOTION_FREE"

    @property
    def linear_frequency_y(self):
        if "linear_frequency_y" in self._entity_data:
            return float(self._entity_data.get('linear_frequency_y'))
        return float(0)

    @property
    def linear_damping_ratio_y(self):
        if "linear_damping_ratio_y" in self._entity_data:
            return float(self._entity_data.get('linear_damping_ratio_y'))
        return float(0)

    @property
    def forcelimit_y(self):
        if "forcelimit_y" in self._entity_data:
            return float(self._entity_data.get('forcelimit_y'))
        return float(0)

    @property
    def notifyforce_y(self):
        if "notifyforce_y" in self._entity_data:
            return float(self._entity_data.get('notifyforce_y'))
        return float(0)

    @property
    def notifyforcemintime_y(self):
        if "notifyforcemintime_y" in self._entity_data:
            return float(self._entity_data.get('notifyforcemintime_y'))
        return float(0)

    @property
    def breakaftertime_y(self):
        if "breakaftertime_y" in self._entity_data:
            return float(self._entity_data.get('breakaftertime_y'))
        return float(0)

    @property
    def breakaftertimethreshold_y(self):
        if "breakaftertimethreshold_y" in self._entity_data:
            return float(self._entity_data.get('breakaftertimethreshold_y'))
        return float(0)

    @property
    def linear_motion_z(self):
        if "linear_motion_z" in self._entity_data:
            return self._entity_data.get('linear_motion_z')
        return "JOINT_MOTION_FREE"

    @property
    def linear_frequency_z(self):
        if "linear_frequency_z" in self._entity_data:
            return float(self._entity_data.get('linear_frequency_z'))
        return float(0)

    @property
    def linear_damping_ratio_z(self):
        if "linear_damping_ratio_z" in self._entity_data:
            return float(self._entity_data.get('linear_damping_ratio_z'))
        return float(0)

    @property
    def forcelimit_z(self):
        if "forcelimit_z" in self._entity_data:
            return float(self._entity_data.get('forcelimit_z'))
        return float(0)

    @property
    def notifyforce_z(self):
        if "notifyforce_z" in self._entity_data:
            return float(self._entity_data.get('notifyforce_z'))
        return float(0)

    @property
    def notifyforcemintime_z(self):
        if "notifyforcemintime_z" in self._entity_data:
            return float(self._entity_data.get('notifyforcemintime_z'))
        return float(0)

    @property
    def breakaftertime_z(self):
        if "breakaftertime_z" in self._entity_data:
            return float(self._entity_data.get('breakaftertime_z'))
        return float(0)

    @property
    def breakaftertimethreshold_z(self):
        if "breakaftertimethreshold_z" in self._entity_data:
            return float(self._entity_data.get('breakaftertimethreshold_z'))
        return float(0)

    @property
    def angular_motion_x(self):
        if "angular_motion_x" in self._entity_data:
            return self._entity_data.get('angular_motion_x')
        return "JOINT_MOTION_FREE"

    @property
    def angular_frequency_x(self):
        if "angular_frequency_x" in self._entity_data:
            return float(self._entity_data.get('angular_frequency_x'))
        return float(0)

    @property
    def angular_damping_ratio_x(self):
        if "angular_damping_ratio_x" in self._entity_data:
            return float(self._entity_data.get('angular_damping_ratio_x'))
        return float(0)

    @property
    def torquelimit_x(self):
        if "torquelimit_x" in self._entity_data:
            return float(self._entity_data.get('torquelimit_x'))
        return float(0)

    @property
    def angular_motion_y(self):
        if "angular_motion_y" in self._entity_data:
            return self._entity_data.get('angular_motion_y')
        return "JOINT_MOTION_FREE"

    @property
    def angular_frequency_y(self):
        if "angular_frequency_y" in self._entity_data:
            return float(self._entity_data.get('angular_frequency_y'))
        return float(0)

    @property
    def angular_damping_ratio_y(self):
        if "angular_damping_ratio_y" in self._entity_data:
            return float(self._entity_data.get('angular_damping_ratio_y'))
        return float(0)

    @property
    def torquelimit_y(self):
        if "torquelimit_y" in self._entity_data:
            return float(self._entity_data.get('torquelimit_y'))
        return float(0)

    @property
    def angular_motion_z(self):
        if "angular_motion_z" in self._entity_data:
            return self._entity_data.get('angular_motion_z')
        return "JOINT_MOTION_FREE"

    @property
    def angular_frequency_z(self):
        if "angular_frequency_z" in self._entity_data:
            return float(self._entity_data.get('angular_frequency_z'))
        return float(0)

    @property
    def angular_damping_ratio_z(self):
        if "angular_damping_ratio_z" in self._entity_data:
            return float(self._entity_data.get('angular_damping_ratio_z'))
        return float(0)

    @property
    def torquelimit_z(self):
        if "torquelimit_z" in self._entity_data:
            return float(self._entity_data.get('torquelimit_z'))
        return float(0)


class phys_convert(Targetname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Convert Asleep': (1, 0), 'Convert As Debris': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def swapmodel(self):
        if "swapmodel" in self._entity_data:
            return self._entity_data.get('swapmodel')
        return None

    @property
    def massoverride(self):
        if "massoverride" in self._entity_data:
            return float(self._entity_data.get('massoverride'))
        return float(0)


class ForceController(Targetname, BasePhysicsNoSettleAttached):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start On': (1, 0), 'Apply Force': (2, 1), 'Apply Torque': (4, 1),
                                   'Orient Locally': (8, 1), 'Ignore Mass': (16, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def attach1(self):
        if "attach1" in self._entity_data:
            return self._entity_data.get('attach1')
        return ""

    @property
    def forcetime(self):
        if "forcetime" in self._entity_data:
            return self._entity_data.get('forcetime')
        return "0"


class phys_thruster(ForceController):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Ignore Pos': (32, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def force(self):
        if "force" in self._entity_data:
            return self._entity_data.get('force')
        return "0"


class phys_torque(ForceController):
    pass

    @property
    def force(self):
        if "force" in self._entity_data:
            return self._entity_data.get('force')
        return "0"

    @property
    def axis(self):
        if "axis" in self._entity_data:
            return parse_int_vector(self._entity_data.get('axis'))
        return parse_int_vector("")


class phys_motor(Targetname):
    pass

    @property
    def speed(self):
        if "speed" in self._entity_data:
            return self._entity_data.get('speed')
        return "0"

    @property
    def spinup(self):
        if "spinup" in self._entity_data:
            return self._entity_data.get('spinup')
        return "1"

    @property
    def inertiafactor(self):
        if "inertiafactor" in self._entity_data:
            return float(self._entity_data.get('inertiafactor'))
        return float(1.0)

    @property
    def axis(self):
        if "axis" in self._entity_data:
            return parse_int_vector(self._entity_data.get('axis'))
        return parse_int_vector("")

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start On': (1, 1), 'No world collision': (2, 0), 'Hinge Object': (4, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def attach1(self):
        if "attach1" in self._entity_data:
            return self._entity_data.get('attach1')
        return ""


class phys_magnet(Targetname, Parentname, Studiomodel):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start Asleep': (1, 0), 'Motion Disabled': (2, 0), 'Suck On Touch': (4, 0),
                                   'Allow Attached Rotation': (8, 0), 'Coast jeep pickup hack': (16, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def forcelimit(self):
        if "forcelimit" in self._entity_data:
            return float(self._entity_data.get('forcelimit'))
        return float(0)

    @property
    def torquelimit(self):
        if "torquelimit" in self._entity_data:
            return float(self._entity_data.get('torquelimit'))
        return float(0)

    @property
    def massScale(self):
        if "massScale" in self._entity_data:
            return float(self._entity_data.get('massScale'))
        return float(0)

    @property
    def maxobjects(self):
        if "maxobjects" in self._entity_data:
            return int(self._entity_data.get('maxobjects'))
        return int(0)


class prop_detail_base:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return None


class prop_static_base(SystemLevelChoice):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return None

    @property
    def skin(self):
        if "skin" in self._entity_data:
            return self._entity_data.get('skin')
        return "default"

    @property
    def solid(self):
        if "solid" in self._entity_data:
            return self._entity_data.get('solid')
        return "6"

    @property
    def disableshadows(self):
        if "disableshadows" in self._entity_data:
            return bool(self._entity_data.get('disableshadows'))
        return bool(0)

    @property
    def fademindist(self):
        if "fademindist" in self._entity_data:
            return float(self._entity_data.get('fademindist'))
        return float(-1)

    @property
    def fademaxdist(self):
        if "fademaxdist" in self._entity_data:
            return float(self._entity_data.get('fademaxdist'))
        return float(0)

    @property
    def fadescale(self):
        if "fadescale" in self._entity_data:
            return float(self._entity_data.get('fadescale'))
        return float(1)

    @property
    def detailgeometry(self):
        if "detailgeometry" in self._entity_data:
            return bool(self._entity_data.get('detailgeometry'))
        return bool(0)

    @property
    def visoccluder(self):
        if "visoccluder" in self._entity_data:
            return bool(self._entity_data.get('visoccluder'))
        return bool(0)

    @property
    def renderamt(self):
        if "renderamt" in self._entity_data:
            return int(self._entity_data.get('renderamt'))
        return int(255)

    @property
    def rendercolor(self):
        if "rendercolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('rendercolor'))
        return parse_int_vector("255 255 255")

    @property
    def disablelowviolence(self):
        if "disablelowviolence" in self._entity_data:
            return self._entity_data.get('disablelowviolence')
        return "0"


class prop_dynamic_base(Parentname, Global, Studiomodel, BreakableProp, RenderFields, Glow):
    pass

    @property
    def solid(self):
        if "solid" in self._entity_data:
            return self._entity_data.get('solid')
        return "6"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Use Hitboxes for Renderbox': (64, 0), 'Start with collision disabled': (256, 0),
                                   'Set to NAVIgnore': (512, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def disablelowviolence(self):
        if "disablelowviolence" in self._entity_data:
            return self._entity_data.get('disablelowviolence')
        return "0"

    @property
    def DefaultAnim(self):
        if "DefaultAnim" in self._entity_data:
            return self._entity_data.get('DefaultAnim')
        return ""

    @property
    def RandomAnimation(self):
        if "RandomAnimation" in self._entity_data:
            return bool(self._entity_data.get('RandomAnimation'))
        return bool(0)

    @property
    def HoldAnimation(self):
        if "HoldAnimation" in self._entity_data:
            return bool(self._entity_data.get('HoldAnimation'))
        return bool(0)

    @property
    def randomizecycle(self):
        if "randomizecycle" in self._entity_data:
            return bool(self._entity_data.get('randomizecycle'))
        return bool(0)

    @property
    def MinAnimTime(self):
        if "MinAnimTime" in self._entity_data:
            return float(self._entity_data.get('MinAnimTime'))
        return float(5)

    @property
    def MaxAnimTime(self):
        if "MaxAnimTime" in self._entity_data:
            return float(self._entity_data.get('MaxAnimTime'))
        return float(10)

    @property
    def LagCompensate(self):
        if "LagCompensate" in self._entity_data:
            return bool(self._entity_data.get('LagCompensate'))
        return bool(0)

    @property
    def AnimateOnServer(self):
        if "AnimateOnServer" in self._entity_data:
            return bool(self._entity_data.get('AnimateOnServer'))
        return bool(0)

    @property
    def ScriptedMovement(self):
        if "ScriptedMovement" in self._entity_data:
            return bool(self._entity_data.get('ScriptedMovement'))
        return bool(0)

    @property
    def velocity(self):
        if "velocity" in self._entity_data:
            return parse_int_vector(self._entity_data.get('velocity'))
        return parse_int_vector("None")

    @property
    def avelocity(self):
        if "avelocity" in self._entity_data:
            return parse_int_vector(self._entity_data.get('avelocity'))
        return parse_int_vector("None")

    @property
    def lightingorigin(self):
        if "lightingorigin" in self._entity_data:
            return self._entity_data.get('lightingorigin')
        return ""

    @property
    def renderamt(self):
        if "renderamt" in self._entity_data:
            return int(self._entity_data.get('renderamt'))
        return int(255)

    @property
    def rendercolor(self):
        if "rendercolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('rendercolor'))
        return parse_int_vector("255 255 255")

    @property
    def updatechildren(self):
        if "updatechildren" in self._entity_data:
            return bool(self._entity_data.get('updatechildren'))
        return bool(0)

    @property
    def use_animgraph(self):
        if "use_animgraph" in self._entity_data:
            return bool(self._entity_data.get('use_animgraph'))
        return bool(0)

    @property
    def CreateNavObstacle(self):
        if "CreateNavObstacle" in self._entity_data:
            return bool(self._entity_data.get('CreateNavObstacle'))
        return bool(0)

    @property
    def forcenpcexclude(self):
        if "forcenpcexclude" in self._entity_data:
            return bool(self._entity_data.get('forcenpcexclude'))
        return bool(0)


class prop_detail(prop_detail_base):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return None


class prop_static(PosableSkeleton):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return None

    @property
    def solid(self):
        if "solid" in self._entity_data:
            return self._entity_data.get('solid')
        return "6"

    @property
    def disableshadows(self):
        if "disableshadows" in self._entity_data:
            return bool(self._entity_data.get('disableshadows'))
        return bool(0)

    @property
    def screenspacefade(self):
        if "screenspacefade" in self._entity_data:
            return bool(self._entity_data.get('screenspacefade'))
        return bool(0)

    @property
    def skin(self):
        if "skin" in self._entity_data:
            return self._entity_data.get('skin')
        return "default"

    @property
    def lodlevel(self):
        if "lodlevel" in self._entity_data:
            return int(self._entity_data.get('lodlevel'))
        return int(-1)

    @property
    def fademindist(self):
        if "fademindist" in self._entity_data:
            return float(self._entity_data.get('fademindist'))
        return float(-1)

    @property
    def fademaxdist(self):
        if "fademaxdist" in self._entity_data:
            return float(self._entity_data.get('fademaxdist'))
        return float(0)

    @property
    def fadescale(self):
        if "fadescale" in self._entity_data:
            return float(self._entity_data.get('fadescale'))
        return float(1)

    @property
    def detailgeometry(self):
        if "detailgeometry" in self._entity_data:
            return bool(self._entity_data.get('detailgeometry'))
        return bool(0)

    @property
    def visoccluder(self):
        if "visoccluder" in self._entity_data:
            return bool(self._entity_data.get('visoccluder'))
        return bool(0)

    @property
    def baketoworld(self):
        if "baketoworld" in self._entity_data:
            return bool(self._entity_data.get('baketoworld'))
        return bool(0)

    @property
    def renderamt(self):
        if "renderamt" in self._entity_data:
            return int(self._entity_data.get('renderamt'))
        return int(255)

    @property
    def rendercolor(self):
        if "rendercolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('rendercolor'))
        return parse_int_vector("255 255 255")

    @property
    def lightingorigin(self):
        if "lightingorigin" in self._entity_data:
            return self._entity_data.get('lightingorigin')
        return ""

    @property
    def lightgroup(self):
        if "lightgroup" in self._entity_data:
            return self._entity_data.get('lightgroup')
        return ""

    @property
    def rendertocubemaps(self):
        if "rendertocubemaps" in self._entity_data:
            return bool(self._entity_data.get('rendertocubemaps'))
        return bool(1)

    @property
    def precomputelightprobes(self):
        if "precomputelightprobes" in self._entity_data:
            return bool(self._entity_data.get('precomputelightprobes'))
        return bool(1)

    @property
    def materialoverride(self):
        if "materialoverride" in self._entity_data:
            return self._entity_data.get('materialoverride')
        return ""

    @property
    def disableinlowquality(self):
        if "disableinlowquality" in self._entity_data:
            return bool(self._entity_data.get('disableinlowquality'))
        return bool(0)

    @property
    def lightmapscalebias(self):
        if "lightmapscalebias" in self._entity_data:
            return self._entity_data.get('lightmapscalebias')
        return "0"

    @property
    def bakelighting(self):
        if "bakelighting" in self._entity_data:
            return self._entity_data.get('bakelighting')
        return "-1"

    @property
    def renderwithdynamic(self):
        if "renderwithdynamic" in self._entity_data:
            return bool(self._entity_data.get('renderwithdynamic'))
        return bool(0)


class prop_dynamic(prop_dynamic_base, EnableDisable):
    pass

    @property
    def clothScale(self):
        if "clothScale" in self._entity_data:
            return float(self._entity_data.get('clothScale'))
        return float(1)


class prop_dynamic_override(prop_dynamic_base):
    pass

    @property
    def health(self):
        if "health" in self._entity_data:
            return int(self._entity_data.get('health'))
        return int(0)


class BasePropPhysics(Parentname, Global, CanBeClientOnly, Studiomodel, BreakableProp, SystemLevelChoice, Glow):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start Asleep': (1, 0), "Don't take physics damage": (2, 0),
                                   "Debris - Don't collide with the player or other debris": (4, 0),
                                   'Motion Disabled': (8, 0),
                                   'Enable motion on Physcannon grab (This also applies to the HLVR hand interaction)': (
                                   64, 0), 'Not affected by rotor wash': (128, 0), 'Generate output on +USE ': (256, 1),
                                   'Prevent pickup': (512, 0), 'Prevent motion enable on player bump': (1024, 0),
                                   'Debris with trigger interaction': (4096, 0),
                                   'Force non-solid to players': (8192, 0), 'Enable +use glow effect': (32768, 0),
                                   'Physgun can ALWAYS pick up. No matter what.': (1048576, 0),
                                   'Important Grabbity Glove target.': (16777216, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def minhealthdmg(self):
        if "minhealthdmg" in self._entity_data:
            return int(self._entity_data.get('minhealthdmg'))
        return int(0)

    @property
    def shadowcastdist(self):
        if "shadowcastdist" in self._entity_data:
            return int(self._entity_data.get('shadowcastdist'))
        return int(0)

    @property
    def physdamagescale(self):
        if "physdamagescale" in self._entity_data:
            return float(self._entity_data.get('physdamagescale'))
        return float(0.1)

    @property
    def Damagetype(self):
        if "Damagetype" in self._entity_data:
            return self._entity_data.get('Damagetype')
        return "0"

    @property
    def nodamageforces(self):
        if "nodamageforces" in self._entity_data:
            return self._entity_data.get('nodamageforces')
        return "0"

    @property
    def acceptdamagefromheldobjects(self):
        if "acceptdamagefromheldobjects" in self._entity_data:
            return bool(self._entity_data.get('acceptdamagefromheldobjects'))
        return bool(0)

    @property
    def inertiaScale(self):
        if "inertiaScale" in self._entity_data:
            return float(self._entity_data.get('inertiaScale'))
        return float(1.0)

    @property
    def massScale(self):
        if "massScale" in self._entity_data:
            return float(self._entity_data.get('massScale'))
        return float(0)

    @property
    def damagetoenablemotion(self):
        if "damagetoenablemotion" in self._entity_data:
            return int(self._entity_data.get('damagetoenablemotion'))
        return int(0)

    @property
    def forcetoenablemotion(self):
        if "forcetoenablemotion" in self._entity_data:
            return float(self._entity_data.get('forcetoenablemotion'))
        return float(0)

    @property
    def puntsound(self):
        if "puntsound" in self._entity_data:
            return self._entity_data.get('puntsound')
        return None

    @property
    def addon(self):
        if "addon" in self._entity_data:
            return self._entity_data.get('addon')
        return ""

    @property
    def interactAs(self):
        if "interactAs" in self._entity_data:
            return self._entity_data.get('interactAs')
        return ""

    @property
    def forcenavignore(self):
        if "forcenavignore" in self._entity_data:
            return bool(self._entity_data.get('forcenavignore'))
        return bool(0)

    @property
    def forcenpcexclude(self):
        if "forcenpcexclude" in self._entity_data:
            return bool(self._entity_data.get('forcenpcexclude'))
        return bool(0)

    @property
    def auto_convert_back_from_debris(self):
        if "auto_convert_back_from_debris" in self._entity_data:
            return bool(self._entity_data.get('auto_convert_back_from_debris'))
        return bool(1)


class prop_physics_override(BasePropPhysics, BaseFadeProp, BasePhysicsSimulated):
    pass

    @property
    def health(self):
        if "health" in self._entity_data:
            return int(self._entity_data.get('health'))
        return int(0)

    @property
    def lightgroup(self):
        if "lightgroup" in self._entity_data:
            return self._entity_data.get('lightgroup')
        return ""


class prop_physics(BasePropPhysics, RenderFields, BasePhysicsSimulated):
    pass

    @property
    def ExploitableByPlayer(self):
        if "ExploitableByPlayer" in self._entity_data:
            return self._entity_data.get('ExploitableByPlayer')
        return "0"

    @property
    def CollisionGroupOverride(self):
        if "CollisionGroupOverride" in self._entity_data:
            return self._entity_data.get('CollisionGroupOverride')
        return "-1"

    @property
    def SilentToZombies(self):
        if "SilentToZombies" in self._entity_data:
            return bool(self._entity_data.get('SilentToZombies'))
        return bool(0)


class prop_physics_multiplayer(prop_physics):
    pass

    @property
    def physicsmode(self):
        if "physicsmode" in self._entity_data:
            return self._entity_data.get('physicsmode')
        return "0"


class prop_ragdoll(Targetname, Studiomodel, SystemLevelChoice, BaseFadeProp, EnableDisable, PosableSkeleton,
                   BasePhysicsSimulated):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {"Debris - Don't collide with the player or other debris": (4, 1),
                                   'Allow Dissolve': (8192, 0), 'Motion Disabled': (16384, 0),
                                   'Allow stretch': (32768, 0), 'Start asleep': (65536, 0),
                                   "Don't force sleep": (131072, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def angleOverride(self):
        if "angleOverride" in self._entity_data:
            return self._entity_data.get('angleOverride')
        return ""

    @property
    def lightingorigin(self):
        if "lightingorigin" in self._entity_data:
            return self._entity_data.get('lightingorigin')
        return ""


class prop_dynamic_ornament(prop_dynamic_base):
    pass

    @property
    def solid(self):
        if "solid" in self._entity_data:
            return self._entity_data.get('solid')
        return "6"

    @property
    def InitialOwner(self):
        if "InitialOwner" in self._entity_data:
            return self._entity_data.get('InitialOwner')
        return None


class func_areaportal(Targetname):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def StartOpen(self):
        if "StartOpen" in self._entity_data:
            return self._entity_data.get('StartOpen')
        return "1"

    @property
    def PortalVersion(self):
        if "PortalVersion" in self._entity_data:
            return int(self._entity_data.get('PortalVersion'))
        return int(1)


class func_breakable(BreakableBrush, RenderFields):
    pass

    @property
    def minhealthdmg(self):
        if "minhealthdmg" in self._entity_data:
            return int(self._entity_data.get('minhealthdmg'))
        return int(0)

    @property
    def gamemass(self):
        if "gamemass" in self._entity_data:
            return int(self._entity_data.get('gamemass'))
        return int(0)

    @property
    def _minlight(self):
        if "_minlight" in self._entity_data:
            return self._entity_data.get('_minlight')
        return None

    @property
    def physdamagescale(self):
        if "physdamagescale" in self._entity_data:
            return float(self._entity_data.get('physdamagescale'))
        return float(1.0)


class func_conveyor(Targetname, Parentname, RenderFields, Shadow):
    pass

    @property
    def movedir(self):
        if "movedir" in self._entity_data:
            return parse_int_vector(self._entity_data.get('movedir'))
        return parse_int_vector("0 0 0")

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'No Push': (1, 0), 'Not Solid': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def speed(self):
        if "speed" in self._entity_data:
            return self._entity_data.get('speed')
        return "100"

    @property
    def _minlight(self):
        if "_minlight" in self._entity_data:
            return self._entity_data.get('_minlight')
        return None


class func_viscluster:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data


class func_illusionary(Targetname, Parentname, RenderFields, Shadow):
    pass

    @property
    def _minlight(self):
        if "_minlight" in self._entity_data:
            return self._entity_data.get('_minlight')
        return None


class func_precipitation(Targetname, Parentname):
    pass

    @property
    def renderamt(self):
        if "renderamt" in self._entity_data:
            return int(self._entity_data.get('renderamt'))
        return int(100)

    @property
    def preciptype(self):
        if "preciptype" in self._entity_data:
            return self._entity_data.get('preciptype')
        return "4"


class func_precipitation_blocker(Targetname, Parentname):
    pass


class func_detail_blocker(Targetname, Parentname):
    pass


class func_wall_toggle(func_wall):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Starts Invisible': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class func_guntarget(Targetname, Parentname, RenderFields, Global):
    pass

    @property
    def speed(self):
        if "speed" in self._entity_data:
            return int(self._entity_data.get('speed'))
        return int(100)

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def health(self):
        if "health" in self._entity_data:
            return int(self._entity_data.get('health'))
        return int(0)

    @property
    def _minlight(self):
        if "_minlight" in self._entity_data:
            return self._entity_data.get('_minlight')
        return None


class func_fish_pool:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/Junkola.vmdl"

    @property
    def fish_count(self):
        if "fish_count" in self._entity_data:
            return int(self._entity_data.get('fish_count'))
        return int(10)

    @property
    def max_range(self):
        if "max_range" in self._entity_data:
            return float(self._entity_data.get('max_range'))
        return float(150)


class PlatSounds:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def movesnd(self):
        if "movesnd" in self._entity_data:
            return self._entity_data.get('movesnd')
        return "0"

    @property
    def stopsnd(self):
        if "stopsnd" in self._entity_data:
            return self._entity_data.get('stopsnd')
        return "0"

    @property
    def volume(self):
        if "volume" in self._entity_data:
            return self._entity_data.get('volume')
        return "0.85"


class Trackchange(Targetname, Parentname, RenderFields, Global, PlatSounds):
    pass

    @property
    def height(self):
        if "height" in self._entity_data:
            return int(self._entity_data.get('height'))
        return int(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Auto Activate train': (1, 0), 'Relink track': (2, 0), 'Start at Bottom': (8, 0),
                                   'Rotate Only': (16, 0), 'X Axis': (64, 0), 'Y Axis': (128, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def rotation(self):
        if "rotation" in self._entity_data:
            return int(self._entity_data.get('rotation'))
        return int(0)

    @property
    def train(self):
        if "train" in self._entity_data:
            return self._entity_data.get('train')
        return None

    @property
    def toptrack(self):
        if "toptrack" in self._entity_data:
            return self._entity_data.get('toptrack')
        return None

    @property
    def bottomtrack(self):
        if "bottomtrack" in self._entity_data:
            return self._entity_data.get('bottomtrack')
        return None

    @property
    def speed(self):
        if "speed" in self._entity_data:
            return int(self._entity_data.get('speed'))
        return int(0)


class BaseTrain(Targetname, Parentname, RenderFields, Global, Shadow):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'No Pitch (X-rot)': (1, 0), 'No User Control': (2, 0), 'Passable': (8, 0),
                                   'Fixed Orientation': (16, 0), 'HL1 Train': (128, 0),
                                   'Use max speed for pitch shifting move sound': (256, 0),
                                   'Is unblockable by player': (512, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return ""

    @property
    def startspeed(self):
        if "startspeed" in self._entity_data:
            return int(self._entity_data.get('startspeed'))
        return int(100)

    @property
    def speed(self):
        if "speed" in self._entity_data:
            return int(self._entity_data.get('speed'))
        return int(0)

    @property
    def velocitytype(self):
        if "velocitytype" in self._entity_data:
            return self._entity_data.get('velocitytype')
        return "0"

    @property
    def orientationtype(self):
        if "orientationtype" in self._entity_data:
            return self._entity_data.get('orientationtype')
        return "1"

    @property
    def wheels(self):
        if "wheels" in self._entity_data:
            return int(self._entity_data.get('wheels'))
        return int(50)

    @property
    def height(self):
        if "height" in self._entity_data:
            return int(self._entity_data.get('height'))
        return int(4)

    @property
    def bank(self):
        if "bank" in self._entity_data:
            return self._entity_data.get('bank')
        return "0"

    @property
    def dmg(self):
        if "dmg" in self._entity_data:
            return int(self._entity_data.get('dmg'))
        return int(0)

    @property
    def _minlight(self):
        if "_minlight" in self._entity_data:
            return self._entity_data.get('_minlight')
        return None

    @property
    def MoveSound(self):
        if "MoveSound" in self._entity_data:
            return self._entity_data.get('MoveSound')
        return ""

    @property
    def MovePingSound(self):
        if "MovePingSound" in self._entity_data:
            return self._entity_data.get('MovePingSound')
        return ""

    @property
    def StartSound(self):
        if "StartSound" in self._entity_data:
            return self._entity_data.get('StartSound')
        return ""

    @property
    def StopSound(self):
        if "StopSound" in self._entity_data:
            return self._entity_data.get('StopSound')
        return ""

    @property
    def volume(self):
        if "volume" in self._entity_data:
            return int(self._entity_data.get('volume'))
        return int(10)

    @property
    def MoveSoundMinPitch(self):
        if "MoveSoundMinPitch" in self._entity_data:
            return int(self._entity_data.get('MoveSoundMinPitch'))
        return int(60)

    @property
    def MoveSoundMaxPitch(self):
        if "MoveSoundMaxPitch" in self._entity_data:
            return int(self._entity_data.get('MoveSoundMaxPitch'))
        return int(200)

    @property
    def MoveSoundMinTime(self):
        if "MoveSoundMinTime" in self._entity_data:
            return float(self._entity_data.get('MoveSoundMinTime'))
        return float(0)

    @property
    def MoveSoundMaxTime(self):
        if "MoveSoundMaxTime" in self._entity_data:
            return float(self._entity_data.get('MoveSoundMaxTime'))
        return float(0)


class func_trackautochange(Trackchange):
    pass

    @property
    def _minlight(self):
        if "_minlight" in self._entity_data:
            return self._entity_data.get('_minlight')
        return None


class func_trackchange(Trackchange):
    pass

    @property
    def _minlight(self):
        if "_minlight" in self._entity_data:
            return self._entity_data.get('_minlight')
        return None


class func_tracktrain(BaseTrain):
    pass


class func_tanktrain(BaseTrain):
    pass

    @property
    def health(self):
        if "health" in self._entity_data:
            return int(self._entity_data.get('health'))
        return int(100)


class func_traincontrols(Parentname, Global):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None


class tanktrain_aitarget(Targetname):
    pass

    icon_sprite = "editor/tanktrain_aitarget.vmat"

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def newtarget(self):
        if "newtarget" in self._entity_data:
            return self._entity_data.get('newtarget')
        return None


class tanktrain_ai(Targetname):
    pass

    icon_sprite = "editor/tanktrain_ai.vmat"

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def startsound(self):
        if "startsound" in self._entity_data:
            return self._entity_data.get('startsound')
        return "vehicles/diesel_start1.wav"

    @property
    def enginesound(self):
        if "enginesound" in self._entity_data:
            return self._entity_data.get('enginesound')
        return "vehicles/diesel_turbo_loop1.wav"

    @property
    def movementsound(self):
        if "movementsound" in self._entity_data:
            return self._entity_data.get('movementsound')
        return "vehicles/tank_treads_loop1.wav"


class path_track(Targetname, Parentname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Disabled': (1, 0), 'Fire once': (2, 0), 'Branch Reverse': (4, 0),
                                   'Disable train': (8, 0), 'Teleport to THIS path track': (16, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def altpath(self):
        if "altpath" in self._entity_data:
            return self._entity_data.get('altpath')
        return None

    @property
    def speed(self):
        if "speed" in self._entity_data:
            return float(self._entity_data.get('speed'))
        return float(0)

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return float(self._entity_data.get('radius'))
        return float(0)

    @property
    def orientationtype(self):
        if "orientationtype" in self._entity_data:
            return self._entity_data.get('orientationtype')
        return "1"


class test_traceline:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data


class trigger_autosave(Targetname):
    pass

    @property
    def master(self):
        if "master" in self._entity_data:
            return self._entity_data.get('master')
        return None

    @property
    def NewLevelUnit(self):
        if "NewLevelUnit" in self._entity_data:
            return bool(self._entity_data.get('NewLevelUnit'))
        return bool(0)

    @property
    def DangerousTimer(self):
        if "DangerousTimer" in self._entity_data:
            return float(self._entity_data.get('DangerousTimer'))
        return float(0)

    @property
    def MinimumHitPoints(self):
        if "MinimumHitPoints" in self._entity_data:
            return int(self._entity_data.get('MinimumHitPoints'))
        return int(0)


class trigger_changelevel(EnableDisable):
    pass

    @property
    def targetname(self):
        if "targetname" in self._entity_data:
            return self._entity_data.get('targetname')
        return None

    @property
    def map(self):
        if "map" in self._entity_data:
            return self._entity_data.get('map')
        return None

    @property
    def landmark(self):
        if "landmark" in self._entity_data:
            return self._entity_data.get('landmark')
        return None

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Disable Touch': (2, 0), 'To Previous Chapter': (4, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class trigger_gravity(Trigger):
    pass

    @property
    def gravity(self):
        if "gravity" in self._entity_data:
            return int(self._entity_data.get('gravity'))
        return int(1)


class trigger_playermovement(Trigger):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'(OBSOLETE, Uncheck me)': (16, 0), 'Disable auto player movement': (128, 1),
                                   'Auto-duck while in trigger': (2048, 0), 'Auto-walk while in trigger': (4096, 0),
                                   'Disable jump while in trigger': (8192, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class trigger_soundscape(Trigger):
    pass

    @property
    def soundscape(self):
        if "soundscape" in self._entity_data:
            return self._entity_data.get('soundscape')
        return None


class trigger_hurt(Trigger):
    pass

    @property
    def master(self):
        if "master" in self._entity_data:
            return self._entity_data.get('master')
        return None

    @property
    def damage(self):
        if "damage" in self._entity_data:
            return int(self._entity_data.get('damage'))
        return int(10)

    @property
    def damagecap(self):
        if "damagecap" in self._entity_data:
            return int(self._entity_data.get('damagecap'))
        return int(20)

    @property
    def damagetype(self):
        if "damagetype" in self._entity_data:
            return self._entity_data.get('damagetype')
        return "0"

    @property
    def damagemodel(self):
        if "damagemodel" in self._entity_data:
            return self._entity_data.get('damagemodel')
        return "0"

    @property
    def forgivedelay(self):
        if "forgivedelay" in self._entity_data:
            return float(self._entity_data.get('forgivedelay'))
        return float(3)

    @property
    def nodmgforce(self):
        if "nodmgforce" in self._entity_data:
            return bool(self._entity_data.get('nodmgforce'))
        return bool(0)

    @property
    def damageforce(self):
        if "damageforce" in self._entity_data:
            return parse_int_vector(self._entity_data.get('damageforce'))
        return parse_int_vector("None")

    @property
    def thinkalways(self):
        if "thinkalways" in self._entity_data:
            return bool(self._entity_data.get('thinkalways'))
        return bool(0)


class trigger_remove(Trigger):
    pass


class trigger_multiple(Trigger):
    pass

    @property
    def wait(self):
        if "wait" in self._entity_data:
            return int(self._entity_data.get('wait'))
        return int(1)


class trigger_once(TriggerOnce):
    pass


class trigger_snd_sos_opvar(Trigger):
    pass

    @property
    def wait(self):
        if "wait" in self._entity_data:
            return float(self._entity_data.get('wait'))
        return float(0.2)

    @property
    def minimum_value(self):
        if "minimum_value" in self._entity_data:
            return float(self._entity_data.get('minimum_value'))
        return float(0.0)

    @property
    def maximum_value(self):
        if "maximum_value" in self._entity_data:
            return float(self._entity_data.get('maximum_value'))
        return float(1.0)

    @property
    def stackname(self):
        if "stackname" in self._entity_data:
            return self._entity_data.get('stackname')
        return "system_globals"

    @property
    def operatorname(self):
        if "operatorname" in self._entity_data:
            return self._entity_data.get('operatorname')
        return "test_opvars"

    @property
    def opvarname(self):
        if "opvarname" in self._entity_data:
            return self._entity_data.get('opvarname')
        return "none"

    @property
    def centersize(self):
        if "centersize" in self._entity_data:
            return float(self._entity_data.get('centersize'))
        return float(0.0)

    @property
    def is2d(self):
        if "is2d" in self._entity_data:
            return bool(self._entity_data.get('is2d'))
        return bool(1)


class trigger_look(Trigger):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Fire Once': (128, 1), 'Use Velocity instead of facing': (256, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def LookTime(self):
        if "LookTime" in self._entity_data:
            return self._entity_data.get('LookTime')
        return "0.5"

    @property
    def FieldOfView(self):
        if "FieldOfView" in self._entity_data:
            return self._entity_data.get('FieldOfView')
        return "0.9"

    @property
    def FOV2D(self):
        if "FOV2D" in self._entity_data:
            return bool(self._entity_data.get('FOV2D'))
        return bool(0)

    @property
    def Timeout(self):
        if "Timeout" in self._entity_data:
            return float(self._entity_data.get('Timeout'))
        return float(0)

    @property
    def test_occlusion(self):
        if "test_occlusion" in self._entity_data:
            return bool(self._entity_data.get('test_occlusion'))
        return bool(0)


class trigger_push(Trigger):
    pass

    @property
    def pushdir(self):
        if "pushdir" in self._entity_data:
            return parse_int_vector(self._entity_data.get('pushdir'))
        return parse_int_vector("0 0 0")

    @property
    def pushdir_islocal(self):
        if "pushdir_islocal" in self._entity_data:
            return bool(self._entity_data.get('pushdir_islocal'))
        return bool(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Once Only': (128, 0), 'Affects Ladders (Half-Life 2)': (256, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def speed(self):
        if "speed" in self._entity_data:
            return int(self._entity_data.get('speed'))
        return int(40)

    @property
    def alternateticksfix(self):
        if "alternateticksfix" in self._entity_data:
            return float(self._entity_data.get('alternateticksfix'))
        return float(0)

    @property
    def triggeronstarttouch(self):
        if "triggeronstarttouch" in self._entity_data:
            return bool(self._entity_data.get('triggeronstarttouch'))
        return bool(0)


class trigger_wind(Trigger):
    pass

    @property
    def Speed(self):
        if "Speed" in self._entity_data:
            return int(self._entity_data.get('Speed'))
        return int(200)

    @property
    def SpeedNoise(self):
        if "SpeedNoise" in self._entity_data:
            return int(self._entity_data.get('SpeedNoise'))
        return int(0)

    @property
    def DirectionNoise(self):
        if "DirectionNoise" in self._entity_data:
            return int(self._entity_data.get('DirectionNoise'))
        return int(10)

    @property
    def HoldTime(self):
        if "HoldTime" in self._entity_data:
            return int(self._entity_data.get('HoldTime'))
        return int(0)

    @property
    def HoldNoise(self):
        if "HoldNoise" in self._entity_data:
            return int(self._entity_data.get('HoldNoise'))
        return int(0)


class trigger_impact(Targetname):
    pass

    @property
    def Magnitude(self):
        if "Magnitude" in self._entity_data:
            return float(self._entity_data.get('Magnitude'))
        return float(200)

    @property
    def noise(self):
        if "noise" in self._entity_data:
            return float(self._entity_data.get('noise'))
        return float(0.1)

    @property
    def viewkick(self):
        if "viewkick" in self._entity_data:
            return float(self._entity_data.get('viewkick'))
        return float(0.05)


class trigger_proximity(Trigger):
    pass

    @property
    def measuretarget(self):
        if "measuretarget" in self._entity_data:
            return self._entity_data.get('measuretarget')
        return None

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return self._entity_data.get('radius')
        return "256"


class trigger_teleport(Trigger):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def landmark(self):
        if "landmark" in self._entity_data:
            return self._entity_data.get('landmark')
        return None

    @property
    def use_landmark_angles(self):
        if "use_landmark_angles" in self._entity_data:
            return bool(self._entity_data.get('use_landmark_angles'))
        return bool(0)

    @property
    def mirror_player(self):
        if "mirror_player" in self._entity_data:
            return bool(self._entity_data.get('mirror_player'))
        return bool(0)


class trigger_transition(Targetname):
    pass

    @property
    def filtername(self):
        if "filtername" in self._entity_data:
            return self._entity_data.get('filtername')
        return None


class trigger_serverragdoll(Targetname):
    pass


class ai_speechfilter(Targetname, ResponseContext, EnableDisable):
    pass

    @property
    def subject(self):
        if "subject" in self._entity_data:
            return self._entity_data.get('subject')
        return ""

    @property
    def IdleModifier(self):
        if "IdleModifier" in self._entity_data:
            return float(self._entity_data.get('IdleModifier'))
        return float(1.0)

    @property
    def NeverSayHello(self):
        if "NeverSayHello" in self._entity_data:
            return self._entity_data.get('NeverSayHello')
        return "0"


class water_lod_control(Targetname):
    pass

    icon_sprite = "editor/waterlodcontrol.vmat"

    @property
    def cheapwaterstartdistance(self):
        if "cheapwaterstartdistance" in self._entity_data:
            return float(self._entity_data.get('cheapwaterstartdistance'))
        return float(1000)

    @property
    def cheapwaterenddistance(self):
        if "cheapwaterenddistance" in self._entity_data:
            return float(self._entity_data.get('cheapwaterenddistance'))
        return float(2000)


class point_camera(Parentname, Targetname, CanBeClientOnly):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start Off': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def FOV(self):
        if "FOV" in self._entity_data:
            return float(self._entity_data.get('FOV'))
        return float(90)

    @property
    def ZNear(self):
        if "ZNear" in self._entity_data:
            return float(self._entity_data.get('ZNear'))
        return float(4)

    @property
    def ZFar(self):
        if "ZFar" in self._entity_data:
            return float(self._entity_data.get('ZFar'))
        return float(10000)

    @property
    def UseScreenAspectRatio(self):
        if "UseScreenAspectRatio" in self._entity_data:
            return bool(self._entity_data.get('UseScreenAspectRatio'))
        return bool(0)

    @property
    def aspectRatio(self):
        if "aspectRatio" in self._entity_data:
            return float(self._entity_data.get('aspectRatio'))
        return float(1)

    @property
    def fogEnable(self):
        if "fogEnable" in self._entity_data:
            return bool(self._entity_data.get('fogEnable'))
        return bool(0)

    @property
    def fogColor(self):
        if "fogColor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('fogColor'))
        return parse_int_vector("0 0 0")

    @property
    def fogStart(self):
        if "fogStart" in self._entity_data:
            return float(self._entity_data.get('fogStart'))
        return float(2048)

    @property
    def fogEnd(self):
        if "fogEnd" in self._entity_data:
            return float(self._entity_data.get('fogEnd'))
        return float(4096)

    @property
    def fogMaxDensity(self):
        if "fogMaxDensity" in self._entity_data:
            return float(self._entity_data.get('fogMaxDensity'))
        return float(1)

    @property
    def rendercolor(self):
        if "rendercolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('rendercolor'))
        return parse_int_vector("128 128 128")


class info_camera_link(Targetname):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def PointCamera(self):
        if "PointCamera" in self._entity_data:
            return self._entity_data.get('PointCamera')
        return None


class logic_measure_movement(Targetname):
    pass

    @property
    def MeasureTarget(self):
        if "MeasureTarget" in self._entity_data:
            return self._entity_data.get('MeasureTarget')
        return ""

    @property
    def MeasureReference(self):
        if "MeasureReference" in self._entity_data:
            return self._entity_data.get('MeasureReference')
        return ""

    @property
    def Target(self):
        if "Target" in self._entity_data:
            return self._entity_data.get('Target')
        return ""

    @property
    def TargetReference(self):
        if "TargetReference" in self._entity_data:
            return self._entity_data.get('TargetReference')
        return ""

    @property
    def TargetScale(self):
        if "TargetScale" in self._entity_data:
            return float(self._entity_data.get('TargetScale'))
        return float(1)

    @property
    def MeasureType(self):
        if "MeasureType" in self._entity_data:
            return self._entity_data.get('MeasureType')
        return "0"


class npc_furniture(BaseNPC, Parentname):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return None

    @property
    def has_animated_face(self):
        if "has_animated_face" in self._entity_data:
            return self._entity_data.get('has_animated_face')
        return "0"

    @property
    def furniture_physics(self):
        if "furniture_physics" in self._entity_data:
            return bool(self._entity_data.get('furniture_physics'))
        return bool(0)

    @property
    def lightingorigin(self):
        if "lightingorigin" in self._entity_data:
            return self._entity_data.get('lightingorigin')
        return ""


class env_credits(Targetname):
    pass


class material_modify_control(Parentname, Targetname):
    pass

    @property
    def materialName(self):
        if "materialName" in self._entity_data:
            return self._entity_data.get('materialName')
        return None

    @property
    def materialVar(self):
        if "materialVar" in self._entity_data:
            return self._entity_data.get('materialVar')
        return None


class point_devshot_camera:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def cameraname(self):
        if "cameraname" in self._entity_data:
            return self._entity_data.get('cameraname')
        return ""

    @property
    def FOV(self):
        if "FOV" in self._entity_data:
            return int(self._entity_data.get('FOV'))
        return int(75)


class logic_playerproxy(Targetname, DamageFilter):
    pass


class env_projectedtexture(Targetname, Parentname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Enabled': (1, 1), 'Always Update (moving light)': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def lightfov(self):
        if "lightfov" in self._entity_data:
            return float(self._entity_data.get('lightfov'))
        return float(90.0)

    @property
    def nearz(self):
        if "nearz" in self._entity_data:
            return float(self._entity_data.get('nearz'))
        return float(4.0)

    @property
    def farz(self):
        if "farz" in self._entity_data:
            return float(self._entity_data.get('farz'))
        return float(750.0)

    @property
    def enableshadows(self):
        if "enableshadows" in self._entity_data:
            return bool(self._entity_data.get('enableshadows'))
        return bool(0)

    @property
    def shadowquality(self):
        if "shadowquality" in self._entity_data:
            return self._entity_data.get('shadowquality')
        return "1"

    @property
    def lightonlytarget(self):
        if "lightonlytarget" in self._entity_data:
            return bool(self._entity_data.get('lightonlytarget'))
        return bool(0)

    @property
    def lightworld(self):
        if "lightworld" in self._entity_data:
            return bool(self._entity_data.get('lightworld'))
        return bool(1)

    @property
    def simpleprojection(self):
        if "simpleprojection" in self._entity_data:
            return bool(self._entity_data.get('simpleprojection'))
        return bool(0)

    @property
    def brightnessscale(self):
        if "brightnessscale" in self._entity_data:
            return float(self._entity_data.get('brightnessscale'))
        return float(1.0)

    @property
    def lightcolor(self):
        if "lightcolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('lightcolor'))
        return parse_int_vector("255 255 255 200")

    @property
    def cameraspace(self):
        if "cameraspace" in self._entity_data:
            return int(self._entity_data.get('cameraspace'))
        return int(0)

    @property
    def texturename(self):
        if "texturename" in self._entity_data:
            return self._entity_data.get('texturename')
        return "effects/flashlight001"

    @property
    def flip_horizontal(self):
        if "flip_horizontal" in self._entity_data:
            return self._entity_data.get('flip_horizontal')
        return "0"


class func_reflective_glass(func_brush):
    pass


class env_particle_performance_monitor(Targetname):
    pass


class npc_puppet(BaseNPC, Parentname, Studiomodel):
    pass

    @property
    def animationtarget(self):
        if "animationtarget" in self._entity_data:
            return self._entity_data.get('animationtarget')
        return ""

    @property
    def attachmentname(self):
        if "attachmentname" in self._entity_data:
            return self._entity_data.get('attachmentname')
        return ""


class point_gamestats_counter(Targetname, EnableDisable):
    pass

    @property
    def Name(self):
        if "Name" in self._entity_data:
            return self._entity_data.get('Name')
        return None


class beam_spotlight(Targetname, Parentname, RenderFields):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start On': (1, 1), 'No Dynamic Light': (2, 0), 'Start rotation on': (4, 0),
                                   'Reverse Direction': (8, 0), 'X Axis': (16, 0), 'Y Axis': (32, 0),
                                   'No Fog': (64, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def maxspeed(self):
        if "maxspeed" in self._entity_data:
            return int(self._entity_data.get('maxspeed'))
        return int(100)

    @property
    def spotlightlength(self):
        if "spotlightlength" in self._entity_data:
            return int(self._entity_data.get('spotlightlength'))
        return int(500)

    @property
    def spotlightwidth(self):
        if "spotlightwidth" in self._entity_data:
            return int(self._entity_data.get('spotlightwidth'))
        return int(50)

    @property
    def rendercolor(self):
        if "rendercolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('rendercolor'))
        return parse_int_vector("255 255 255")

    @property
    def HDRColorScale(self):
        if "HDRColorScale" in self._entity_data:
            return float(self._entity_data.get('HDRColorScale'))
        return float(0.7)


class func_instance:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def targetname(self):
        if "targetname" in self._entity_data:
            return self._entity_data.get('targetname')
        return None

    @property
    def file(self):
        if "file" in self._entity_data:
            return self._entity_data.get('file')
        return None

    @property
    def fixup_style(self):
        if "fixup_style" in self._entity_data:
            return self._entity_data.get('fixup_style')
        return "0"

    @property
    def replace01(self):
        if "replace01" in self._entity_data:
            return self._entity_data.get('replace01')
        return None

    @property
    def replace02(self):
        if "replace02" in self._entity_data:
            return self._entity_data.get('replace02')
        return None

    @property
    def replace03(self):
        if "replace03" in self._entity_data:
            return self._entity_data.get('replace03')
        return None

    @property
    def replace04(self):
        if "replace04" in self._entity_data:
            return self._entity_data.get('replace04')
        return None

    @property
    def replace05(self):
        if "replace05" in self._entity_data:
            return self._entity_data.get('replace05')
        return None

    @property
    def replace06(self):
        if "replace06" in self._entity_data:
            return self._entity_data.get('replace06')
        return None

    @property
    def replace07(self):
        if "replace07" in self._entity_data:
            return self._entity_data.get('replace07')
        return None

    @property
    def replace08(self):
        if "replace08" in self._entity_data:
            return self._entity_data.get('replace08')
        return None

    @property
    def replace09(self):
        if "replace09" in self._entity_data:
            return self._entity_data.get('replace09')
        return None

    @property
    def replace10(self):
        if "replace10" in self._entity_data:
            return self._entity_data.get('replace10')
        return None


class point_event_proxy(Targetname):
    pass

    @property
    def EventName(self):
        if "EventName" in self._entity_data:
            return self._entity_data.get('EventName')
        return None

    @property
    def ActivatorAsUserID(self):
        if "ActivatorAsUserID" in self._entity_data:
            return self._entity_data.get('ActivatorAsUserID')
        return "1"


class env_instructor_hint(Targetname):
    pass

    icon_sprite = "editor/env_instructor_hint.vmat"

    @property
    def hint_replace_key(self):
        if "hint_replace_key" in self._entity_data:
            return self._entity_data.get('hint_replace_key')
        return None

    @property
    def hint_target(self):
        if "hint_target" in self._entity_data:
            return self._entity_data.get('hint_target')
        return None

    @property
    def hint_static(self):
        if "hint_static" in self._entity_data:
            return self._entity_data.get('hint_static')
        return "0"

    @property
    def hint_allow_nodraw_target(self):
        if "hint_allow_nodraw_target" in self._entity_data:
            return self._entity_data.get('hint_allow_nodraw_target')
        return "1"

    @property
    def hint_caption(self):
        if "hint_caption" in self._entity_data:
            return self._entity_data.get('hint_caption')
        return None

    @property
    def hint_activator_caption(self):
        if "hint_activator_caption" in self._entity_data:
            return self._entity_data.get('hint_activator_caption')
        return None

    @property
    def hint_color(self):
        if "hint_color" in self._entity_data:
            return parse_int_vector(self._entity_data.get('hint_color'))
        return parse_int_vector("255 255 255")

    @property
    def hint_forcecaption(self):
        if "hint_forcecaption" in self._entity_data:
            return self._entity_data.get('hint_forcecaption')
        return "0"

    @property
    def hint_icon_onscreen(self):
        if "hint_icon_onscreen" in self._entity_data:
            return self._entity_data.get('hint_icon_onscreen')
        return "icon_tip"

    @property
    def hint_icon_offscreen(self):
        if "hint_icon_offscreen" in self._entity_data:
            return self._entity_data.get('hint_icon_offscreen')
        return "icon_tip"

    @property
    def hint_nooffscreen(self):
        if "hint_nooffscreen" in self._entity_data:
            return self._entity_data.get('hint_nooffscreen')
        return "0"

    @property
    def hint_binding(self):
        if "hint_binding" in self._entity_data:
            return self._entity_data.get('hint_binding')
        return None

    @property
    def hint_icon_offset(self):
        if "hint_icon_offset" in self._entity_data:
            return float(self._entity_data.get('hint_icon_offset'))
        return float(0)

    @property
    def hint_pulseoption(self):
        if "hint_pulseoption" in self._entity_data:
            return self._entity_data.get('hint_pulseoption')
        return "0"

    @property
    def hint_alphaoption(self):
        if "hint_alphaoption" in self._entity_data:
            return self._entity_data.get('hint_alphaoption')
        return "0"

    @property
    def hint_shakeoption(self):
        if "hint_shakeoption" in self._entity_data:
            return self._entity_data.get('hint_shakeoption')
        return "0"

    @property
    def hint_local_player_only(self):
        if "hint_local_player_only" in self._entity_data:
            return bool(self._entity_data.get('hint_local_player_only'))
        return bool(False)

    @property
    def hint_timeout(self):
        if "hint_timeout" in self._entity_data:
            return int(self._entity_data.get('hint_timeout'))
        return int(0)

    @property
    def hint_range(self):
        if "hint_range" in self._entity_data:
            return float(self._entity_data.get('hint_range'))
        return float(0)

    @property
    def hint_auto_start(self):
        if "hint_auto_start" in self._entity_data:
            return bool(self._entity_data.get('hint_auto_start'))
        return bool(1)


class info_target_instructor_hint(Targetname, Parentname):
    pass


class env_instructor_vr_hint(Targetname):
    pass

    icon_sprite = "editor/env_instructor_hint.vmat"

    @property
    def hint_caption(self):
        if "hint_caption" in self._entity_data:
            return self._entity_data.get('hint_caption')
        return None

    @property
    def hint_start_sound(self):
        if "hint_start_sound" in self._entity_data:
            return self._entity_data.get('hint_start_sound')
        return "Instructor.StartLesson"

    @property
    def hint_timeout(self):
        if "hint_timeout" in self._entity_data:
            return int(self._entity_data.get('hint_timeout'))
        return int(0)

    @property
    def hint_layoutfiletype(self):
        if "hint_layoutfiletype" in self._entity_data:
            return self._entity_data.get('hint_layoutfiletype')
        return "0"

    @property
    def hint_custom_layoutfile(self):
        if "hint_custom_layoutfile" in self._entity_data:
            return self._entity_data.get('hint_custom_layoutfile')
        return None

    @property
    def hint_vr_panel_type(self):
        if "hint_vr_panel_type" in self._entity_data:
            return self._entity_data.get('hint_vr_panel_type')
        return "0"

    @property
    def hint_target(self):
        if "hint_target" in self._entity_data:
            return self._entity_data.get('hint_target')
        return None

    @property
    def hint_vr_height_offset(self):
        if "hint_vr_height_offset" in self._entity_data:
            return float(self._entity_data.get('hint_vr_height_offset'))
        return float(0)


class point_instructor_event(Targetname):
    pass

    icon_sprite = "editor/env_instructor_hint.vmat"

    @property
    def hint_name(self):
        if "hint_name" in self._entity_data:
            return self._entity_data.get('hint_name')
        return None

    @property
    def hint_target(self):
        if "hint_target" in self._entity_data:
            return self._entity_data.get('hint_target')
        return None


class func_timescale(Targetname):
    pass

    @property
    def desiredTimescale(self):
        if "desiredTimescale" in self._entity_data:
            return float(self._entity_data.get('desiredTimescale'))
        return float(1.0)

    @property
    def acceleration(self):
        if "acceleration" in self._entity_data:
            return float(self._entity_data.get('acceleration'))
        return float(0.05)

    @property
    def minBlendRate(self):
        if "minBlendRate" in self._entity_data:
            return float(self._entity_data.get('minBlendRate'))
        return float(0.1)

    @property
    def blendDeltaMultiplier(self):
        if "blendDeltaMultiplier" in self._entity_data:
            return float(self._entity_data.get('blendDeltaMultiplier'))
        return float(3.0)


class prop_hallucination(Targetname, Parentname, Studiomodel):
    pass

    @property
    def EnabledChance(self):
        if "EnabledChance" in self._entity_data:
            return float(self._entity_data.get('EnabledChance'))
        return float(100.0)

    @property
    def VisibleTime(self):
        if "VisibleTime" in self._entity_data:
            return float(self._entity_data.get('VisibleTime'))
        return float(0.215)

    @property
    def RechargeTime(self):
        if "RechargeTime" in self._entity_data:
            return float(self._entity_data.get('RechargeTime'))
        return float(0.0)


class point_worldtext(Targetname, Parentname, RenderFields):
    pass

    @property
    def message(self):
        if "message" in self._entity_data:
            return self._entity_data.get('message')
        return None

    @property
    def enabled(self):
        if "enabled" in self._entity_data:
            return bool(self._entity_data.get('enabled'))
        return bool(1)

    @property
    def fullbright(self):
        if "fullbright" in self._entity_data:
            return bool(self._entity_data.get('fullbright'))
        return bool(0)

    @property
    def color(self):
        if "color" in self._entity_data:
            return parse_int_vector(self._entity_data.get('color'))
        return parse_int_vector("0 0 0 255")

    @property
    def world_units_per_pixel(self):
        if "world_units_per_pixel" in self._entity_data:
            return float(self._entity_data.get('world_units_per_pixel'))
        return float(0.25)

    @property
    def font_size(self):
        if "font_size" in self._entity_data:
            return float(self._entity_data.get('font_size'))
        return float(20)

    @property
    def font_name(self):
        if "font_name" in self._entity_data:
            return self._entity_data.get('font_name')
        return "Arial Black"

    @property
    def justify_horizontal(self):
        if "justify_horizontal" in self._entity_data:
            return self._entity_data.get('justify_horizontal')
        return "0"

    @property
    def justify_vertical(self):
        if "justify_vertical" in self._entity_data:
            return self._entity_data.get('justify_vertical')
        return "0"

    @property
    def reorient_mode(self):
        if "reorient_mode" in self._entity_data:
            return self._entity_data.get('reorient_mode')
        return "0"

    @property
    def depth_render_offset(self):
        if "depth_render_offset" in self._entity_data:
            return float(self._entity_data.get('depth_render_offset'))
        return float(0.125)


class fog_volume(Targetname, EnableDisable):
    pass

    @property
    def FogName(self):
        if "FogName" in self._entity_data:
            return self._entity_data.get('FogName')
        return None

    @property
    def PostProcessName(self):
        if "PostProcessName" in self._entity_data:
            return self._entity_data.get('PostProcessName')
        return None

    @property
    def ColorCorrectionName(self):
        if "ColorCorrectionName" in self._entity_data:
            return self._entity_data.get('ColorCorrectionName')
        return None


class func_occluder(Targetname, Parentname, EnableDisable):
    pass


class func_distance_occluder(Targetname, Parentname, EnableDisable):
    pass

    @property
    def FadeStartDist(self):
        if "FadeStartDist" in self._entity_data:
            return float(self._entity_data.get('FadeStartDist'))
        return float(128)

    @property
    def FadeEndDist(self):
        if "FadeEndDist" in self._entity_data:
            return float(self._entity_data.get('FadeEndDist'))
        return float(512)

    @property
    def TranslucencyLimit(self):
        if "TranslucencyLimit" in self._entity_data:
            return float(self._entity_data.get('TranslucencyLimit'))
        return float(0)


class point_workplane(Targetname):
    pass

    @property
    def editor_only(self):
        if "editor_only" in self._entity_data:
            return bool(self._entity_data.get('editor_only'))
        return bool(1)


class path_corner(Targetname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Wait for retrigger': (1, 0), 'Teleport to THIS path_corner': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def wait(self):
        if "wait" in self._entity_data:
            return int(self._entity_data.get('wait'))
        return int(0)

    @property
    def speed(self):
        if "speed" in self._entity_data:
            return int(self._entity_data.get('speed'))
        return int(0)

    @property
    def yaw_speed(self):
        if "yaw_speed" in self._entity_data:
            return int(self._entity_data.get('yaw_speed'))
        return int(0)

    @property
    def moveactivity(self):
        if "moveactivity" in self._entity_data:
            return self._entity_data.get('moveactivity')
        return ""

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return float(self._entity_data.get('radius'))
        return float(0)


class point_value_remapper(Targetname, EnableDisable):
    pass

    @property
    def updateOnClient(self):
        if "updateOnClient" in self._entity_data:
            return bool(self._entity_data.get('updateOnClient'))
        return bool(1)

    @property
    def inputType(self):
        if "inputType" in self._entity_data:
            return self._entity_data.get('inputType')
        return "0"

    @property
    def remapLineStart(self):
        if "remapLineStart" in self._entity_data:
            return self._entity_data.get('remapLineStart')
        return None

    @property
    def remapLineEnd(self):
        if "remapLineEnd" in self._entity_data:
            return self._entity_data.get('remapLineEnd')
        return None

    @property
    def customOutputValue(self):
        if "customOutputValue" in self._entity_data:
            return float(self._entity_data.get('customOutputValue'))
        return float(-1.0)

    @property
    def maximumChangePerSecond(self):
        if "maximumChangePerSecond" in self._entity_data:
            return float(self._entity_data.get('maximumChangePerSecond'))
        return float(1000.0)

    @property
    def maximumDistanceFromLine(self):
        if "maximumDistanceFromLine" in self._entity_data:
            return float(self._entity_data.get('maximumDistanceFromLine'))
        return float(1000.0)

    @property
    def engageDistance(self):
        if "engageDistance" in self._entity_data:
            return float(self._entity_data.get('engageDistance'))
        return float(1000.0)

    @property
    def requiresUseKey(self):
        if "requiresUseKey" in self._entity_data:
            return self._entity_data.get('requiresUseKey')
        return "0"

    @property
    def outputType(self):
        if "outputType" in self._entity_data:
            return self._entity_data.get('outputType')
        return "0"

    @property
    def outputEntity(self):
        if "outputEntity" in self._entity_data:
            return self._entity_data.get('outputEntity')
        return None

    @property
    def outputEntity2(self):
        if "outputEntity2" in self._entity_data:
            return self._entity_data.get('outputEntity2')
        return None

    @property
    def outputEntity3(self):
        if "outputEntity3" in self._entity_data:
            return self._entity_data.get('outputEntity3')
        return None

    @property
    def outputEntity4(self):
        if "outputEntity4" in self._entity_data:
            return self._entity_data.get('outputEntity4')
        return None

    @property
    def hapticsType(self):
        if "hapticsType" in self._entity_data:
            return self._entity_data.get('hapticsType')
        return "0"

    @property
    def momentumType(self):
        if "momentumType" in self._entity_data:
            return self._entity_data.get('momentumType')
        return "0"

    @property
    def momentumModifier(self):
        if "momentumModifier" in self._entity_data:
            return float(self._entity_data.get('momentumModifier'))
        return float(0.0)

    @property
    def snapValue(self):
        if "snapValue" in self._entity_data:
            return float(self._entity_data.get('snapValue'))
        return float(0.0)

    @property
    def ratchetType(self):
        if "ratchetType" in self._entity_data:
            return self._entity_data.get('ratchetType')
        return "0"

    @property
    def inputOffset(self):
        if "inputOffset" in self._entity_data:
            return float(self._entity_data.get('inputOffset'))
        return float(0.0)

    @property
    def soundEngage(self):
        if "soundEngage" in self._entity_data:
            return self._entity_data.get('soundEngage')
        return ""

    @property
    def soundDisengage(self):
        if "soundDisengage" in self._entity_data:
            return self._entity_data.get('soundDisengage')
        return ""

    @property
    def soundReachedValueZero(self):
        if "soundReachedValueZero" in self._entity_data:
            return self._entity_data.get('soundReachedValueZero')
        return ""

    @property
    def soundReachedValueOne(self):
        if "soundReachedValueOne" in self._entity_data:
            return self._entity_data.get('soundReachedValueOne')
        return ""

    @property
    def soundMovingLoop(self):
        if "soundMovingLoop" in self._entity_data:
            return self._entity_data.get('soundMovingLoop')
        return ""


class prop_magic_carpet:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/editor/sky_helper.vmdl"

    @property
    def max_ride_speed(self):
        if "max_ride_speed" in self._entity_data:
            return float(self._entity_data.get('max_ride_speed'))
        return float(50)


class env_clock(Targetname):
    pass

    icon_sprite = "editor/logic_timer.vmat"

    @property
    def hourhand(self):
        if "hourhand" in self._entity_data:
            return self._entity_data.get('hourhand')
        return None

    @property
    def minutehand(self):
        if "minutehand" in self._entity_data:
            return self._entity_data.get('minutehand')
        return None

    @property
    def secondhand(self):
        if "secondhand" in self._entity_data:
            return self._entity_data.get('secondhand')
        return None

    @property
    def uselocaltime(self):
        if "uselocaltime" in self._entity_data:
            return bool(self._entity_data.get('uselocaltime'))
        return bool(1)

    @property
    def timezone(self):
        if "timezone" in self._entity_data:
            return int(self._entity_data.get('timezone'))
        return int(-8)


class base_clientui_ent(Targetname):
    pass

    @property
    def dialog_layout_name(self):
        if "dialog_layout_name" in self._entity_data:
            return self._entity_data.get('dialog_layout_name')
        return None

    @property
    def panel_class_name(self):
        if "panel_class_name" in self._entity_data:
            return self._entity_data.get('panel_class_name')
        return None

    @property
    def panel_id(self):
        if "panel_id" in self._entity_data:
            return self._entity_data.get('panel_id')
        return None


class point_clientui_dialog(base_clientui_ent):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return None

    @property
    def StartEnabled(self):
        if "StartEnabled" in self._entity_data:
            return bool(self._entity_data.get('StartEnabled'))
        return bool(1)


class point_clientui_world_panel(base_clientui_ent, Parentname):
    pass

    @property
    def width(self):
        if "width" in self._entity_data:
            return float(self._entity_data.get('width'))
        return float(32)

    @property
    def height(self):
        if "height" in self._entity_data:
            return float(self._entity_data.get('height'))
        return float(32)

    @property
    def panel_dpi(self):
        if "panel_dpi" in self._entity_data:
            return float(self._entity_data.get('panel_dpi'))
        return float(32)

    @property
    def ignore_input(self):
        if "ignore_input" in self._entity_data:
            return bool(self._entity_data.get('ignore_input'))
        return bool(0)

    @property
    def interact_distance(self):
        if "interact_distance" in self._entity_data:
            return float(self._entity_data.get('interact_distance'))
        return float(8)

    @property
    def lit(self):
        if "lit" in self._entity_data:
            return bool(self._entity_data.get('lit'))
        return bool(0)

    @property
    def horizontal_align(self):
        if "horizontal_align" in self._entity_data:
            return self._entity_data.get('horizontal_align')
        return None

    @property
    def vertical_align(self):
        if "vertical_align" in self._entity_data:
            return self._entity_data.get('vertical_align')
        return None

    @property
    def orientation(self):
        if "orientation" in self._entity_data:
            return self._entity_data.get('orientation')
        return None


class point_clientui_world_text_panel(point_clientui_world_panel):
    pass

    @property
    def message(self):
        if "message" in self._entity_data:
            return self._entity_data.get('message')
        return None

    @property
    def enabled(self):
        if "enabled" in self._entity_data:
            return bool(self._entity_data.get('enabled'))
        return bool(1)

    @property
    def dialog_layout_name(self):
        if "dialog_layout_name" in self._entity_data:
            return self._entity_data.get('dialog_layout_name')
        return "file://{resources}/layout/worldui_text.xml"

    @property
    def width(self):
        if "width" in self._entity_data:
            return float(self._entity_data.get('width'))
        return float(128)

    @property
    def height(self):
        if "height" in self._entity_data:
            return float(self._entity_data.get('height'))
        return float(24)

    @property
    def enable_offscreen_indicator(self):
        if "enable_offscreen_indicator" in self._entity_data:
            return bool(self._entity_data.get('enable_offscreen_indicator'))
        return bool(0)

    @property
    def horizontal_align(self):
        if "horizontal_align" in self._entity_data:
            return self._entity_data.get('horizontal_align')
        return "1"

    @property
    def vertical_align(self):
        if "vertical_align" in self._entity_data:
            return self._entity_data.get('vertical_align')
        return "0"

    @property
    def orientation(self):
        if "orientation" in self._entity_data:
            return self._entity_data.get('orientation')
        return "2"


class info_spawngroup_landmark(Targetname):
    pass

    icon_sprite = "editor/info_target.vmat"


class env_sky(Targetname, Parentname, EnableDisable):
    pass

    icon_sprite = "editor/env_sky.vmat"

    @property
    def skyname(self):
        if "skyname" in self._entity_data:
            return self._entity_data.get('skyname')
        return "materials/dev/default_sky.vmat"

    @property
    def tint_color(self):
        if "tint_color" in self._entity_data:
            return parse_int_vector(self._entity_data.get('tint_color'))
        return parse_int_vector("255 255 255")

    @property
    def fog_type(self):
        if "fog_type" in self._entity_data:
            return self._entity_data.get('fog_type')
        return "1"

    @property
    def angular_fog_max_end(self):
        if "angular_fog_max_end" in self._entity_data:
            return float(self._entity_data.get('angular_fog_max_end'))
        return float(35.0)

    @property
    def angular_fog_max_start(self):
        if "angular_fog_max_start" in self._entity_data:
            return float(self._entity_data.get('angular_fog_max_start'))
        return float(25.0)

    @property
    def angular_fog_min_start(self):
        if "angular_fog_min_start" in self._entity_data:
            return float(self._entity_data.get('angular_fog_min_start'))
        return float(-25.0)

    @property
    def angular_fog_min_end(self):
        if "angular_fog_min_end" in self._entity_data:
            return float(self._entity_data.get('angular_fog_min_end'))
        return float(-35.0)


class func_shatterglass(PhysicsTypeOverride_Mesh, Targetname, Parentname, Global, EnableDisable, Shadow):
    pass

    @property
    def GlassNavIgnore(self):
        if "GlassNavIgnore" in self._entity_data:
            return bool(self._entity_data.get('GlassNavIgnore'))
        return bool(1)

    @property
    def GlassThickness(self):
        if "GlassThickness" in self._entity_data:
            return float(self._entity_data.get('GlassThickness'))
        return float(0.6)

    @property
    def GlassInFrame(self):
        if "GlassInFrame" in self._entity_data:
            return bool(self._entity_data.get('GlassInFrame'))
        return bool(1)

    @property
    def SpawnInvulnerability(self):
        if "SpawnInvulnerability" in self._entity_data:
            return float(self._entity_data.get('SpawnInvulnerability'))
        return float(3)

    @property
    def StartBroken(self):
        if "StartBroken" in self._entity_data:
            return bool(self._entity_data.get('StartBroken'))
        return bool(0)

    @property
    def BreakShardless(self):
        if "BreakShardless" in self._entity_data:
            return bool(self._entity_data.get('BreakShardless'))
        return bool(0)

    @property
    def DamageType(self):
        if "DamageType" in self._entity_data:
            return self._entity_data.get('DamageType')
        return "1"

    @property
    def DamagePositioningEntity(self):
        if "DamagePositioningEntity" in self._entity_data:
            return self._entity_data.get('DamagePositioningEntity')
        return None

    @property
    def DamagePositioningEntity02(self):
        if "DamagePositioningEntity02" in self._entity_data:
            return self._entity_data.get('DamagePositioningEntity02')
        return None

    @property
    def DamagePositioningEntity03(self):
        if "DamagePositioningEntity03" in self._entity_data:
            return self._entity_data.get('DamagePositioningEntity03')
        return None

    @property
    def DamagePositioningEntity04(self):
        if "DamagePositioningEntity04" in self._entity_data:
            return self._entity_data.get('DamagePositioningEntity04')
        return None

    @property
    def surface_type(self):
        if "surface_type" in self._entity_data:
            return self._entity_data.get('surface_type')
        return "0"

    @property
    def lightgroup(self):
        if "lightgroup" in self._entity_data:
            return self._entity_data.get('lightgroup')
        return ""


class env_volumetric_fog_controller(Targetname, Parentname, EnableDisable):
    pass

    @property
    def IsMaster(self):
        if "IsMaster" in self._entity_data:
            return bool(self._entity_data.get('IsMaster'))
        return bool(0)

    @property
    def FogStrength(self):
        if "FogStrength" in self._entity_data:
            return float(self._entity_data.get('FogStrength'))
        return float(1.0)

    @property
    def DrawDistance(self):
        if "DrawDistance" in self._entity_data:
            return float(self._entity_data.get('DrawDistance'))
        return float(600.0)

    @property
    def FadeInStart(self):
        if "FadeInStart" in self._entity_data:
            return float(self._entity_data.get('FadeInStart'))
        return float(20.0)

    @property
    def FadeInEnd(self):
        if "FadeInEnd" in self._entity_data:
            return float(self._entity_data.get('FadeInEnd'))
        return float(100.0)

    @property
    def IndirectEnabled(self):
        if "IndirectEnabled" in self._entity_data:
            return bool(self._entity_data.get('IndirectEnabled'))
        return bool(0)

    @property
    def IndirectStrength(self):
        if "IndirectStrength" in self._entity_data:
            return float(self._entity_data.get('IndirectStrength'))
        return float(1.0)

    @property
    def IndirectVoxelDim(self):
        if "IndirectVoxelDim" in self._entity_data:
            return self._entity_data.get('IndirectVoxelDim')
        return "256.0"

    @property
    def FadeSpeed(self):
        if "FadeSpeed" in self._entity_data:
            return float(self._entity_data.get('FadeSpeed'))
        return float(2.0)

    @property
    def Anisotropy(self):
        if "Anisotropy" in self._entity_data:
            return float(self._entity_data.get('Anisotropy'))
        return float(1.0)

    @property
    def fogirradiancevolume(self):
        if "fogirradiancevolume" in self._entity_data:
            return self._entity_data.get('fogirradiancevolume')
        return ""


class env_volumetric_fog_volume(Targetname, Parentname, EnableDisable):
    pass

    icon_sprite = "materials/editor/fog_volume.vmat"

    @property
    def box_mins(self):
        if "box_mins" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_mins'))
        return parse_int_vector("-64 -64 -64")

    @property
    def box_maxs(self):
        if "box_maxs" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_maxs'))
        return parse_int_vector("64 64 64")

    @property
    def FogStrength(self):
        if "FogStrength" in self._entity_data:
            return float(self._entity_data.get('FogStrength'))
        return float(1.0)

    @property
    def FalloffExponent(self):
        if "FalloffExponent" in self._entity_data:
            return float(self._entity_data.get('FalloffExponent'))
        return float(1.0)

    @property
    def Shape(self):
        if "Shape" in self._entity_data:
            return self._entity_data.get('Shape')
        return "0"


class visibility_hint:
    pass

    icon_sprite = "materials/editor/visibility_hint.vmat"

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def box_mins(self):
        if "box_mins" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_mins'))
        return parse_int_vector("-64 -64 -64")

    @property
    def box_maxs(self):
        if "box_maxs" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_maxs'))
        return parse_int_vector("64 64 64")

    @property
    def hintType(self):
        if "hintType" in self._entity_data:
            return self._entity_data.get('hintType')
        return "3"


class info_visibility_box(Targetname, EnableDisable):
    pass

    icon_sprite = "materials/editor/info_visibility_box.vmat"

    @property
    def box_size(self):
        if "box_size" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_size'))
        return parse_int_vector("128 128 128")

    @property
    def cull_mode(self):
        if "cull_mode" in self._entity_data:
            return self._entity_data.get('cull_mode')
        return "0"


class info_cull_triangles:
    pass

    icon_sprite = "materials/editor/info_cull_triangles.vmat"

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def box_size(self):
        if "box_size" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_size'))
        return parse_int_vector("128 128 128")

    @property
    def limit_to_world(self):
        if "limit_to_world" in self._entity_data:
            return bool(self._entity_data.get('limit_to_world'))
        return bool(0)

    @property
    def targets(self):
        if "targets" in self._entity_data:
            return parse_int_vector(self._entity_data.get('targets'))
        return parse_int_vector("")

    @property
    def geometry_type(self):
        if "geometry_type" in self._entity_data:
            return self._entity_data.get('geometry_type')
        return "0"


class path_particle_rope(Targetname):
    pass

    @property
    def effect_name(self):
        if "effect_name" in self._entity_data:
            return self._entity_data.get('effect_name')
        return "particles/entity/path_particle_cable_default.vpcf"

    @property
    def start_active(self):
        if "start_active" in self._entity_data:
            return bool(self._entity_data.get('start_active'))
        return bool(1)

    @property
    def max_simulation_time(self):
        if "max_simulation_time" in self._entity_data:
            return float(self._entity_data.get('max_simulation_time'))
        return float(0)

    @property
    def particle_spacing(self):
        if "particle_spacing" in self._entity_data:
            return float(self._entity_data.get('particle_spacing'))
        return float(32)

    @property
    def slack(self):
        if "slack" in self._entity_data:
            return float(self._entity_data.get('slack'))
        return float(0.5)

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return float(self._entity_data.get('radius'))
        return float(4.0)

    @property
    def static_collision(self):
        if "static_collision" in self._entity_data:
            return bool(self._entity_data.get('static_collision'))
        return bool(0)

    @property
    def surface_properties(self):
        if "surface_properties" in self._entity_data:
            return self._entity_data.get('surface_properties')
        return ""

    @property
    def color_tint(self):
        if "color_tint" in self._entity_data:
            return parse_int_vector(self._entity_data.get('color_tint'))
        return parse_int_vector("255 255 255")


class cable_static:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data


class cable_dynamic(Targetname, Parentname, Global, RenderFields, Glow, EnableDisable):
    pass

    @property
    def secondary_material(self):
        if "secondary_material" in self._entity_data:
            return self._entity_data.get('secondary_material')
        return ""

    @property
    def lightingorigin(self):
        if "lightingorigin" in self._entity_data:
            return self._entity_data.get('lightingorigin')
        return ""

    @property
    def disableshadows(self):
        if "disableshadows" in self._entity_data:
            return bool(self._entity_data.get('disableshadows'))
        return bool(0)


class haptic_relay(Targetname):
    pass

    icon_sprite = "editor/haptic_relay.vmat"

    @property
    def Frequency(self):
        if "Frequency" in self._entity_data:
            return float(self._entity_data.get('Frequency'))
        return float(50)

    @property
    def Amplitude(self):
        if "Amplitude" in self._entity_data:
            return float(self._entity_data.get('Amplitude'))
        return float(0.5)

    @property
    def Duration(self):
        if "Duration" in self._entity_data:
            return float(self._entity_data.get('Duration'))
        return float(0.1)


class commentary_auto(Targetname):
    pass

    icon_sprite = "editor/commentary_auto.vmat"


class point_commentary_node(Targetname, Parentname):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/props/commentary_node.vmdl"

    @property
    def commentaryfile(self):
        if "commentaryfile" in self._entity_data:
            return self._entity_data.get('commentaryfile')
        return ""

    @property
    def title(self):
        if "title" in self._entity_data:
            return self._entity_data.get('title')
        return ""

    @property
    def speakers(self):
        if "speakers" in self._entity_data:
            return self._entity_data.get('speakers')
        return ""

    @property
    def node_id(self):
        if "node_id" in self._entity_data:
            return int(self._entity_data.get('node_id'))
        return int(0)

    @property
    def precommands(self):
        if "precommands" in self._entity_data:
            return self._entity_data.get('precommands')
        return ""

    @property
    def postcommands(self):
        if "postcommands" in self._entity_data:
            return self._entity_data.get('postcommands')
        return ""

    @property
    def viewtarget(self):
        if "viewtarget" in self._entity_data:
            return self._entity_data.get('viewtarget')
        return ""

    @property
    def viewposition(self):
        if "viewposition" in self._entity_data:
            return self._entity_data.get('viewposition')
        return ""

    @property
    def teleport_origin(self):
        if "teleport_origin" in self._entity_data:
            return parse_int_vector(self._entity_data.get('teleport_origin'))
        return parse_int_vector("")

    @property
    def prevent_movement(self):
        if "prevent_movement" in self._entity_data:
            return self._entity_data.get('prevent_movement')
        return "0"

    @property
    def start_disabled(self):
        if "start_disabled" in self._entity_data:
            return self._entity_data.get('start_disabled')
        return "0"


entity_class_handle = {
    "PhysicsTypeOverride_Mesh": PhysicsTypeOverride_Mesh,
    "PhysicsTypeOverride_SingleConvex": PhysicsTypeOverride_SingleConvex,
    "PhysicsTypeOverride_MultiConvex": PhysicsTypeOverride_MultiConvex,
    "PosableSkeleton": PosableSkeleton,
    "DXLevelChoice": DXLevelChoice,
    "VScript": VScript,
    "GameEntity": GameEntity,
    "Targetname": Targetname,
    "Parentname": Parentname,
    "Studiomodel": Studiomodel,
    "BasePlat": BasePlat,
    "BaseBrush": BaseBrush,
    "EnableDisable": EnableDisable,
    "RenderFxChoices": RenderFxChoices,
    "RenderModeChoices": RenderModeChoices,
    "Shadow": Shadow,
    "Glow": Glow,
    "SystemLevelChoice": SystemLevelChoice,
    "RenderFields": RenderFields,
    "Inputfilter": Inputfilter,
    "Global": Global,
    "EnvGlobal": EnvGlobal,
    "DamageFilter": DamageFilter,
    "ResponseContext": ResponseContext,
    "Breakable": Breakable,
    "BreakableBrush": BreakableBrush,
    "CanBeClientOnly": CanBeClientOnly,
    "BreakableProp": BreakableProp,
    "BaseNPC": BaseNPC,
    "info_npc_spawn_destination": info_npc_spawn_destination,
    "BaseNPCMaker": BaseNPCMaker,
    "npc_template_maker": npc_template_maker,
    "BaseHelicopter": BaseHelicopter,
    "PlayerClass": PlayerClass,
    "Light": Light,
    "Node": Node,
    "HintNode": HintNode,
    "TriggerOnce": TriggerOnce,
    "Trigger": Trigger,
    "worldbase": worldbase,
    "ambient_generic": ambient_generic,
    "point_soundevent": point_soundevent,
    "snd_event_point": snd_event_point,
    "snd_event_alignedbox": snd_event_alignedbox,
    "snd_stack_save": snd_stack_save,
    "snd_event_param": snd_event_param,
    "snd_opvar_set": snd_opvar_set,
    "SndOpvarSetPointBase": SndOpvarSetPointBase,
    "SndOpvarSetPointBaseAddition": SndOpvarSetPointBaseAddition,
    "snd_opvar_set_point": snd_opvar_set_point,
    "snd_opvar_set_aabb": snd_opvar_set_aabb,
    "snd_opvar_set_obb": snd_opvar_set_obb,
    "func_lod": func_lod,
    "env_zoom": env_zoom,
    "env_screenoverlay": env_screenoverlay,
    "env_screeneffect": env_screeneffect,
    "env_texturetoggle": env_texturetoggle,
    "env_splash": env_splash,
    "env_particlelight": env_particlelight,
    "env_sun": env_sun,
    "env_tonemap_controller": env_tonemap_controller,
    "game_ragdoll_manager": game_ragdoll_manager,
    "game_gib_manager": game_gib_manager,
    "env_dof_controller": env_dof_controller,
    "env_lightglow": env_lightglow,
    "env_smokestack": env_smokestack,
    "env_fade": env_fade,
    "env_player_surface_trigger": env_player_surface_trigger,
    "trigger_tonemap": trigger_tonemap,
    "func_useableladder": func_useableladder,
    "func_ladderendpoint": func_ladderendpoint,
    "info_ladder_dismount": info_ladder_dismount,
    "func_areaportalwindow": func_areaportalwindow,
    "func_wall": func_wall,
    "func_clip_interaction_layer": func_clip_interaction_layer,
    "func_brush": func_brush,
    "VGUIScreenBase": VGUIScreenBase,
    "vgui_screen": vgui_screen,
    "vgui_slideshow_display": vgui_slideshow_display,
    "vgui_movie_display": vgui_movie_display,
    "cycler": cycler,
    "func_orator": func_orator,
    "gibshooterbase": gibshooterbase,
    "env_beam": env_beam,
    "env_beverage": env_beverage,
    "env_funnel": env_funnel,
    "env_blood": env_blood,
    "env_bubbles": env_bubbles,
    "env_explosion": env_explosion,
    "env_smoketrail": env_smoketrail,
    "env_physexplosion": env_physexplosion,
    "env_physimpact": env_physimpact,
    "env_fire": env_fire,
    "env_firesource": env_firesource,
    "env_firesensor": env_firesensor,
    "env_entity_igniter": env_entity_igniter,
    "env_fog_controller": env_fog_controller,
    "postprocess_controller": postprocess_controller,
    "env_laser": env_laser,
    "env_message": env_message,
    "env_hudhint": env_hudhint,
    "env_shake": env_shake,
    "env_tilt": env_tilt,
    "env_viewpunch": env_viewpunch,
    "env_rotorwash_emitter": env_rotorwash_emitter,
    "gibshooter": gibshooter,
    "env_shooter": env_shooter,
    "env_rotorshooter": env_rotorshooter,
    "env_soundscape_proxy": env_soundscape_proxy,
    "snd_soundscape_proxy": snd_soundscape_proxy,
    "env_soundscape": env_soundscape,
    "snd_soundscape": snd_soundscape,
    "env_soundscape_triggerable": env_soundscape_triggerable,
    "snd_soundscape_triggerable": snd_soundscape_triggerable,
    "env_spark": env_spark,
    "env_sprite": env_sprite,
    "env_sprite_oriented": env_sprite_oriented,
    "BaseEnvWind": BaseEnvWind,
    "env_wind": env_wind,
    "env_wind_clientside": env_wind_clientside,
    "sky_camera": sky_camera,
    "BaseSpeaker": BaseSpeaker,
    "game_weapon_manager": game_weapon_manager,
    "game_end": game_end,
    "game_player_equip": game_player_equip,
    "game_player_team": game_player_team,
    "game_score": game_score,
    "game_text": game_text,
    "point_enable_motion_fixup": point_enable_motion_fixup,
    "point_message": point_message,
    "point_spotlight": point_spotlight,
    "point_tesla": point_tesla,
    "point_clientcommand": point_clientcommand,
    "point_servercommand": point_servercommand,
    "point_broadcastclientcommand": point_broadcastclientcommand,
    "point_bonusmaps_accessor": point_bonusmaps_accessor,
    "game_ui": game_ui,
    "point_entity_finder": point_entity_finder,
    "game_zone_player": game_zone_player,
    "info_projecteddecal": info_projecteddecal,
    "info_no_dynamic_shadow": info_no_dynamic_shadow,
    "info_player_start": info_player_start,
    "info_overlay": info_overlay,
    "info_overlay_transition": info_overlay_transition,
    "info_intermission": info_intermission,
    "info_landmark": info_landmark,
    "info_spawngroup_load_unload": info_spawngroup_load_unload,
    "info_null": info_null,
    "info_target": info_target,
    "info_particle_target": info_particle_target,
    "info_particle_system": info_particle_system,
    "phys_ragdollmagnet": phys_ragdollmagnet,
    "info_lighting": info_lighting,
    "info_teleport_destination": info_teleport_destination,
    "AiHullFlags": AiHullFlags,
    "info_node": info_node,
    "info_node_hint": info_node_hint,
    "info_node_air": info_node_air,
    "info_node_air_hint": info_node_air_hint,
    "info_hint": info_hint,
    "BaseNodeLink": BaseNodeLink,
    "info_node_link": info_node_link,
    "info_node_link_controller": info_node_link_controller,
    "info_radial_link_controller": info_radial_link_controller,
    "info_node_climb": info_node_climb,
    "light_dynamic": light_dynamic,
    "shadow_control": shadow_control,
    "color_correction": color_correction,
    "color_correction_volume": color_correction_volume,
    "KeyFrame": KeyFrame,
    "Mover": Mover,
    "func_movelinear": func_movelinear,
    "func_water_analog": func_water_analog,
    "func_rotating": func_rotating,
    "func_platrot": func_platrot,
    "keyframe_track": keyframe_track,
    "move_keyframed": move_keyframed,
    "move_track": move_track,
    "RopeKeyFrame": RopeKeyFrame,
    "keyframe_rope": keyframe_rope,
    "move_rope": move_rope,
    "Button": Button,
    "BaseFuncButton": BaseFuncButton,
    "func_button": func_button,
    "func_physical_button": func_physical_button,
    "func_rot_button": func_rot_button,
    "momentary_rot_button": momentary_rot_button,
    "Door": Door,
    "func_door": func_door,
    "func_door_rotating": func_door_rotating,
    "BaseFadeProp": BaseFadeProp,
    "BasePropDoorRotating": BasePropDoorRotating,
    "BModelParticleSpawner": BModelParticleSpawner,
    "func_dustmotes": func_dustmotes,
    "func_dustcloud": func_dustcloud,
    "env_dustpuff": env_dustpuff,
    "env_particlescript": env_particlescript,
    "env_effectscript": env_effectscript,
    "logic_auto": logic_auto,
    "point_viewcontrol": point_viewcontrol,
    "point_posecontroller": point_posecontroller,
    "logic_compare": logic_compare,
    "logic_branch": logic_branch,
    "logic_branch_listener": logic_branch_listener,
    "logic_case": logic_case,
    "logic_multicompare": logic_multicompare,
    "LogicNPCCounterPointBase": LogicNPCCounterPointBase,
    "logic_npc_counter_radius": logic_npc_counter_radius,
    "logic_npc_counter_aabb": logic_npc_counter_aabb,
    "logic_npc_counter_obb": logic_npc_counter_obb,
    "logic_random_outputs": logic_random_outputs,
    "logic_script": logic_script,
    "logic_relay": logic_relay,
    "logic_timer": logic_timer,
    "hammer_updateignorelist": hammer_updateignorelist,
    "logic_collision_pair": logic_collision_pair,
    "env_microphone": env_microphone,
    "math_remap": math_remap,
    "math_colorblend": math_colorblend,
    "math_counter": math_counter,
    "logic_lineto": logic_lineto,
    "logic_navigation": logic_navigation,
    "logic_autosave": logic_autosave,
    "logic_active_autosave": logic_active_autosave,
    "logic_playmovie": logic_playmovie,
    "info_world_layer": info_world_layer,
    "point_template": point_template,
    "env_entity_maker": env_entity_maker,
    "BaseFilter": BaseFilter,
    "filter_multi": filter_multi,
    "filter_activator_name": filter_activator_name,
    "filter_activator_model": filter_activator_model,
    "filter_activator_context": filter_activator_context,
    "filter_activator_class": filter_activator_class,
    "filter_activator_mass_greater": filter_activator_mass_greater,
    "filter_damage_type": filter_damage_type,
    "filter_enemy": filter_enemy,
    "filter_proximity": filter_proximity,
    "filter_los": filter_los,
    "point_anglesensor": point_anglesensor,
    "point_angularvelocitysensor": point_angularvelocitysensor,
    "point_velocitysensor": point_velocitysensor,
    "point_proximity_sensor": point_proximity_sensor,
    "point_teleport": point_teleport,
    "point_hurt": point_hurt,
    "point_playermoveconstraint": point_playermoveconstraint,
    "BasePhysicsSimulated": BasePhysicsSimulated,
    "BasePhysicsNoSettleAttached": BasePhysicsNoSettleAttached,
    "func_physbox": func_physbox,
    "TwoObjectPhysics": TwoObjectPhysics,
    "phys_keepupright": phys_keepupright,
    "physics_cannister": physics_cannister,
    "info_constraint_anchor": info_constraint_anchor,
    "info_mass_center": info_mass_center,
    "phys_spring": phys_spring,
    "ConstraintSoundInfo": ConstraintSoundInfo,
    "phys_hinge": phys_hinge,
    "phys_hinge_local": phys_hinge_local,
    "phys_ballsocket": phys_ballsocket,
    "phys_constraint": phys_constraint,
    "phys_pulleyconstraint": phys_pulleyconstraint,
    "phys_slideconstraint": phys_slideconstraint,
    "phys_lengthconstraint": phys_lengthconstraint,
    "phys_ragdollconstraint": phys_ragdollconstraint,
    "phys_genericconstraint": phys_genericconstraint,
    "phys_convert": phys_convert,
    "ForceController": ForceController,
    "phys_thruster": phys_thruster,
    "phys_torque": phys_torque,
    "phys_motor": phys_motor,
    "phys_magnet": phys_magnet,
    "prop_detail_base": prop_detail_base,
    "prop_static_base": prop_static_base,
    "prop_dynamic_base": prop_dynamic_base,
    "prop_detail": prop_detail,
    "prop_static": prop_static,
    "prop_dynamic": prop_dynamic,
    "prop_dynamic_override": prop_dynamic_override,
    "BasePropPhysics": BasePropPhysics,
    "prop_physics_override": prop_physics_override,
    "prop_physics": prop_physics,
    "prop_physics_multiplayer": prop_physics_multiplayer,
    "prop_ragdoll": prop_ragdoll,
    "prop_dynamic_ornament": prop_dynamic_ornament,
    "func_areaportal": func_areaportal,
    "func_breakable": func_breakable,
    "func_conveyor": func_conveyor,
    "func_viscluster": func_viscluster,
    "func_illusionary": func_illusionary,
    "func_precipitation": func_precipitation,
    "func_precipitation_blocker": func_precipitation_blocker,
    "func_detail_blocker": func_detail_blocker,
    "func_wall_toggle": func_wall_toggle,
    "func_guntarget": func_guntarget,
    "func_fish_pool": func_fish_pool,
    "PlatSounds": PlatSounds,
    "Trackchange": Trackchange,
    "BaseTrain": BaseTrain,
    "func_trackautochange": func_trackautochange,
    "func_trackchange": func_trackchange,
    "func_tracktrain": func_tracktrain,
    "func_tanktrain": func_tanktrain,
    "func_traincontrols": func_traincontrols,
    "tanktrain_aitarget": tanktrain_aitarget,
    "tanktrain_ai": tanktrain_ai,
    "path_track": path_track,
    "test_traceline": test_traceline,
    "trigger_autosave": trigger_autosave,
    "trigger_changelevel": trigger_changelevel,
    "trigger_gravity": trigger_gravity,
    "trigger_playermovement": trigger_playermovement,
    "trigger_soundscape": trigger_soundscape,
    "trigger_hurt": trigger_hurt,
    "trigger_remove": trigger_remove,
    "trigger_multiple": trigger_multiple,
    "trigger_once": trigger_once,
    "trigger_snd_sos_opvar": trigger_snd_sos_opvar,
    "trigger_look": trigger_look,
    "trigger_push": trigger_push,
    "trigger_wind": trigger_wind,
    "trigger_impact": trigger_impact,
    "trigger_proximity": trigger_proximity,
    "trigger_teleport": trigger_teleport,
    "trigger_transition": trigger_transition,
    "trigger_serverragdoll": trigger_serverragdoll,
    "ai_speechfilter": ai_speechfilter,
    "water_lod_control": water_lod_control,
    "point_camera": point_camera,
    "info_camera_link": info_camera_link,
    "logic_measure_movement": logic_measure_movement,
    "npc_furniture": npc_furniture,
    "env_credits": env_credits,
    "material_modify_control": material_modify_control,
    "point_devshot_camera": point_devshot_camera,
    "logic_playerproxy": logic_playerproxy,
    "env_projectedtexture": env_projectedtexture,
    "func_reflective_glass": func_reflective_glass,
    "env_particle_performance_monitor": env_particle_performance_monitor,
    "npc_puppet": npc_puppet,
    "point_gamestats_counter": point_gamestats_counter,
    "beam_spotlight": beam_spotlight,
    "func_instance": func_instance,
    "point_event_proxy": point_event_proxy,
    "env_instructor_hint": env_instructor_hint,
    "info_target_instructor_hint": info_target_instructor_hint,
    "env_instructor_vr_hint": env_instructor_vr_hint,
    "point_instructor_event": point_instructor_event,
    "func_timescale": func_timescale,
    "prop_hallucination": prop_hallucination,
    "point_worldtext": point_worldtext,
    "fog_volume": fog_volume,
    "func_occluder": func_occluder,
    "func_distance_occluder": func_distance_occluder,
    "point_workplane": point_workplane,
    "path_corner": path_corner,
    "point_value_remapper": point_value_remapper,
    "prop_magic_carpet": prop_magic_carpet,
    "env_clock": env_clock,
    "base_clientui_ent": base_clientui_ent,
    "point_clientui_dialog": point_clientui_dialog,
    "point_clientui_world_panel": point_clientui_world_panel,
    "point_clientui_world_text_panel": point_clientui_world_text_panel,
    "info_spawngroup_landmark": info_spawngroup_landmark,
    "env_sky": env_sky,
    "func_shatterglass": func_shatterglass,
    "env_volumetric_fog_controller": env_volumetric_fog_controller,
    "env_volumetric_fog_volume": env_volumetric_fog_volume,
    "visibility_hint": visibility_hint,
    "info_visibility_box": info_visibility_box,
    "info_cull_triangles": info_cull_triangles,
    "path_particle_rope": path_particle_rope,
    "cable_static": cable_static,
    "cable_dynamic": cable_dynamic,
    "haptic_relay": haptic_relay,
    "commentary_auto": commentary_auto,
    "point_commentary_node": point_commentary_node,
}
