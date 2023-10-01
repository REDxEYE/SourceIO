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

    icon_sprite =  "editor/info_target.vmat"

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

    icon_sprite =  "editor/npc_maker.vmat"

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

    icon_sprite =  "editor/npc_maker.vmat"

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

    icon_sprite =  "editor/ambient_generic.vmat"

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

    icon_sprite =  "editor/snd_event.vmat"

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

    icon_sprite =  "editor/snd_event.vmat"


class snd_event_alignedbox(point_soundevent):
    pass

    icon_sprite =  "editor/snd_event.vmat"

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

    icon_sprite =  "editor/snd_event.vmat"

    @property
    def stackToSave(self):
        if "stackToSave" in self._entity_data:
            return self._entity_data.get('stackToSave')
        return ""


class snd_event_param(Targetname, Parentname):
    pass

    icon_sprite =  "editor/snd_opvar_set.vmat"

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

    icon_sprite =  "editor/snd_opvar_set.vmat"

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

    icon_sprite =  "editor/snd_opvar_set.vmat"

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

    icon_sprite =  "editor/snd_opvar_set.vmat"

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

    icon_sprite =  "editor/snd_opvar_set.vmat"


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

    icon_sprite =  "editor/env_texturetoggle.vmat"

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

    icon_sprite =  "materials/editor/env_tonemap_controller.vmat"

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

    icon_sprite =  "editor/env_dof_controller.vmat"


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

    icon_sprite =  "editor/env_fade"

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

    icon_sprite =  "editor/env_explosion.vmat"

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

    icon_sprite =  "editor/env_physexplosion.vmat"

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

    icon_sprite =  "editor/env_physexplosion.vmat"

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

    icon_sprite =  "editor/env_fire"

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

    icon_sprite =  "editor/env_firesource"

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

    icon_sprite =  "materials/editor/env_fog_controller.vmat"

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

    icon_sprite =  "editor/postprocess_controller.vmat"

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

    icon_sprite =  "editor/env_shake.vmat"

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

    icon_sprite =  "editor/gibshooter.vmat"


class env_shooter(gibshooterbase, RenderFields):
    pass

    icon_sprite =  "editor/env_shooter.vmat"

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

    icon_sprite =  "editor/env_shooter.vmat"

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

    icon_sprite =  "editor/env_soundscape.vmat"

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

    icon_sprite =  "editor/env_soundscape.vmat"


class env_soundscape(Targetname, Parentname, EnableDisable):
    pass

    icon_sprite =  "editor/env_soundscape.vmat"

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

    icon_sprite =  "editor/env_soundscape.vmat"


class env_soundscape_triggerable(env_soundscape):
    pass

    icon_sprite =  "editor/env_soundscape.vmat"


class snd_soundscape_triggerable(env_soundscape_triggerable):
    pass

    icon_sprite =  "editor/env_soundscape.vmat"


class env_spark(Targetname, Parentname):
    pass

    icon_sprite =  "editor/env_spark.vmat"

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

    icon_sprite =  "editor/env_wind.vmat"


class env_wind_clientside(BaseEnvWind):
    pass

    icon_sprite =  "editor/env_wind.vmat"


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

    icon_sprite =  "editor/game_end.vmat"

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

    icon_sprite =  "editor/game_text.vmat"

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

    icon_sprite =  "editor/info_landmark"


class info_spawngroup_load_unload(Targetname):
    pass

    icon_sprite =  "editor/info_target.vmat"

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

    icon_sprite =  "editor/info_target.vmat"

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

    icon_sprite =  "editor/info_target.vmat"

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

    icon_sprite =  "editor/info_lighting.vmat"


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

    icon_sprite =  "editor/light.vmat"

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

    icon_sprite =  "editor/shadow_control.vmat"

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

    icon_sprite =  "editor/color_correction.vmat"

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

    icon_sprite =  "editor/logic_auto.vmat"

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

    icon_sprite =  "editor/logic_compare.vmat"

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

    icon_sprite =  "editor/logic_branch.vmat"

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

    icon_sprite =  "editor/logic_case.vmat"

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

    icon_sprite =  "editor/logic_multicompare.vmat"

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

    icon_sprite =  "editor/math_counter.vmat"

    @property
    def distanceMax(self):
        if "distanceMax" in self._entity_data:
            return float(self._entity_data.get('distanceMax'))
        return float(25.0)


class logic_npc_counter_aabb(Targetname, LogicNPCCounterPointBase):
    pass

    icon_sprite =  "editor/math_counter.vmat"

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

    icon_sprite =  "editor/math_counter.vmat"


class logic_random_outputs(Targetname, EnableDisable):
    pass

    icon_sprite =  "editor/logic_random_outputs.vmat"

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

    icon_sprite =  "editor/logic_script.vmat"

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

    icon_sprite =  "editor/logic_relay.vmat"

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

    icon_sprite =  "editor/logic_timer.vmat"

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

    icon_sprite =  "editor/env_microphone.vmat"

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

    icon_sprite =  "editor/math_counter.vmat"

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

    icon_sprite =  "editor/logic_autosave.vmat"

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

    icon_sprite =  "editor/info_world_layer.vmat"

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

    icon_sprite =  "editor/point_template.vmat"

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

    icon_sprite =  "editor/filter_multiple.vmat"

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

    icon_sprite =  "editor/filter_name.vmat"

    @property
    def filtername(self):
        if "filtername" in self._entity_data:
            return self._entity_data.get('filtername')
        return None


class filter_activator_model(BaseFilter):
    pass

    icon_sprite =  "editor/filter_name.vmat"

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return None


class filter_activator_context(BaseFilter):
    pass

    icon_sprite =  "editor/filter_name.vmat"

    @property
    def ResponseContext(self):
        if "ResponseContext" in self._entity_data:
            return self._entity_data.get('ResponseContext')
        return None


class filter_activator_class(BaseFilter):
    pass

    icon_sprite =  "editor/filter_class.vmat"

    @property
    def filterclass(self):
        if "filterclass" in self._entity_data:
            return self._entity_data.get('filterclass')
        return None


class filter_activator_mass_greater(BaseFilter):
    pass

    icon_sprite =  "editor/filter_class.vmat"

    @property
    def filtermass(self):
        if "filtermass" in self._entity_data:
            return float(self._entity_data.get('filtermass'))
        return float(0)


class filter_damage_type(BaseFilter):
    pass

    icon_sprite =  "editor/filter_type.vmat"

    @property
    def damagetype(self):
        if "damagetype" in self._entity_data:
            return self._entity_data.get('damagetype')
        return "64"


class filter_enemy(BaseFilter):
    pass

    icon_sprite =  "editor/filter_class.vmat"

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

    icon_sprite =  "editor/filter_class.vmat"

    @property
    def filter_radius(self):
        if "filter_radius" in self._entity_data:
            return float(self._entity_data.get('filter_radius'))
        return float(0)


class filter_los(BaseFilter):
    pass

    icon_sprite =  "editor/filter_class.vmat"


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

    icon_sprite =  "editor/phys_ballsocket.vmat"

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

    icon_sprite =  "editor/tanktrain_aitarget.vmat"

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

    icon_sprite =  "editor/tanktrain_ai.vmat"

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

    icon_sprite =  "editor/waterlodcontrol.vmat"

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

    icon_sprite =  "editor/env_instructor_hint.vmat"

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

    icon_sprite =  "editor/env_instructor_hint.vmat"

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

    icon_sprite =  "editor/env_instructor_hint.vmat"

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

    icon_sprite =  "editor/logic_timer.vmat"

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

    icon_sprite =  "editor/info_target.vmat"


class env_sky(Targetname, Parentname, EnableDisable):
    pass

    icon_sprite =  "editor/env_sky.vmat"

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

    icon_sprite =  "materials/editor/fog_volume.vmat"

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

    icon_sprite =  "materials/editor/visibility_hint.vmat"

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

    icon_sprite =  "materials/editor/info_visibility_box.vmat"

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

    icon_sprite =  "materials/editor/info_cull_triangles.vmat"

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

    icon_sprite =  "editor/haptic_relay.vmat"

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

    icon_sprite =  "editor/commentary_auto.vmat"


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


class light_base(Targetname, Parentname):
    pass

    @property
    def enabled(self):
        if "enabled" in self._entity_data:
            return bool(self._entity_data.get('enabled'))
        return bool(1)

    @property
    def color(self):
        if "color" in self._entity_data:
            return parse_int_vector(self._entity_data.get('color'))
        return parse_int_vector("255 255 255")

    @property
    def brightness(self):
        if "brightness" in self._entity_data:
            return float(self._entity_data.get('brightness'))
        return float(1.0)

    @property
    def range(self):
        if "range" in self._entity_data:
            return float(self._entity_data.get('range'))
        return float(512)

    @property
    def castshadows(self):
        if "castshadows" in self._entity_data:
            return self._entity_data.get('castshadows')
        return "1"

    @property
    def nearclipplane(self):
        if "nearclipplane" in self._entity_data:
            return float(self._entity_data.get('nearclipplane'))
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
    def fademindist(self):
        if "fademindist" in self._entity_data:
            return float(self._entity_data.get('fademindist'))
        return float(-250)

    @property
    def fademaxdist(self):
        if "fademaxdist" in self._entity_data:
            return float(self._entity_data.get('fademaxdist'))
        return float(1250)

    @property
    def rendertocubemaps(self):
        if "rendertocubemaps" in self._entity_data:
            return bool(self._entity_data.get('rendertocubemaps'))
        return bool(1)

    @property
    def priority(self):
        if "priority" in self._entity_data:
            return int(self._entity_data.get('priority'))
        return int(0)

    @property
    def lightgroup(self):
        if "lightgroup" in self._entity_data:
            return self._entity_data.get('lightgroup')
        return ""


class light_base_legacy_params:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def bouncescale(self):
        if "bouncescale" in self._entity_data:
            return float(self._entity_data.get('bouncescale'))
        return float(1.0)

    @property
    def renderdiffuse(self):
        if "renderdiffuse" in self._entity_data:
            return bool(self._entity_data.get('renderdiffuse'))
        return bool(1)

    @property
    def renderspecular(self):
        if "renderspecular" in self._entity_data:
            return self._entity_data.get('renderspecular')
        return "1"

    @property
    def rendertransmissive(self):
        if "rendertransmissive" in self._entity_data:
            return self._entity_data.get('rendertransmissive')
        return "1"

    @property
    def directlight(self):
        if "directlight" in self._entity_data:
            return self._entity_data.get('directlight')
        return "2"

    @property
    def indirectlight(self):
        if "indirectlight" in self._entity_data:
            return self._entity_data.get('indirectlight')
        return "1"


class light_base_attenuation_params:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def attenuation1(self):
        if "attenuation1" in self._entity_data:
            return float(self._entity_data.get('attenuation1'))
        return float(0.0)

    @property
    def attenuation2(self):
        if "attenuation2" in self._entity_data:
            return float(self._entity_data.get('attenuation2'))
        return float(1.0)

    @property
    def lightsourceradius(self):
        if "lightsourceradius" in self._entity_data:
            return float(self._entity_data.get('lightsourceradius'))
        return float(2.0)


class light_environment(light_base, light_base_legacy_params):
    pass

    @property
    def angulardiameter(self):
        if "angulardiameter" in self._entity_data:
            return float(self._entity_data.get('angulardiameter'))
        return float(1.0)

    @property
    def numcascades(self):
        if "numcascades" in self._entity_data:
            return int(self._entity_data.get('numcascades'))
        return int(3)

    @property
    def shadowcascadedistance0(self):
        if "shadowcascadedistance0" in self._entity_data:
            return float(self._entity_data.get('shadowcascadedistance0'))
        return float(0.0)

    @property
    def shadowcascadedistance1(self):
        if "shadowcascadedistance1" in self._entity_data:
            return float(self._entity_data.get('shadowcascadedistance1'))
        return float(0.0)

    @property
    def shadowcascadedistance2(self):
        if "shadowcascadedistance2" in self._entity_data:
            return float(self._entity_data.get('shadowcascadedistance2'))
        return float(0.0)

    @property
    def shadowcascaderesolution0(self):
        if "shadowcascaderesolution0" in self._entity_data:
            return int(self._entity_data.get('shadowcascaderesolution0'))
        return int(0)

    @property
    def shadowcascaderesolution1(self):
        if "shadowcascaderesolution1" in self._entity_data:
            return int(self._entity_data.get('shadowcascaderesolution1'))
        return int(0)

    @property
    def shadowcascaderesolution2(self):
        if "shadowcascaderesolution2" in self._entity_data:
            return int(self._entity_data.get('shadowcascaderesolution2'))
        return int(0)

    @property
    def skycolor(self):
        if "skycolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('skycolor'))
        return parse_int_vector("255 255 255")

    @property
    def skyintensity(self):
        if "skyintensity" in self._entity_data:
            return float(self._entity_data.get('skyintensity'))
        return float(1.0)

    @property
    def lower_hemisphere_is_black(self):
        if "lower_hemisphere_is_black" in self._entity_data:
            return bool(self._entity_data.get('lower_hemisphere_is_black'))
        return bool(1)

    @property
    def skytexture(self):
        if "skytexture" in self._entity_data:
            return self._entity_data.get('skytexture')
        return ""

    @property
    def skytexturescale(self):
        if "skytexturescale" in self._entity_data:
            return float(self._entity_data.get('skytexturescale'))
        return float(1.0)

    @property
    def skyambientbounce(self):
        if "skyambientbounce" in self._entity_data:
            return parse_int_vector(self._entity_data.get('skyambientbounce'))
        return parse_int_vector("147 147 147")

    @property
    def sunlightminbrightness(self):
        if "sunlightminbrightness" in self._entity_data:
            return float(self._entity_data.get('sunlightminbrightness'))
        return float(32)

    @property
    def ambient_occlusion(self):
        if "ambient_occlusion" in self._entity_data:
            return bool(self._entity_data.get('ambient_occlusion'))
        return bool(0)

    @property
    def max_occlusion_distance(self):
        if "max_occlusion_distance" in self._entity_data:
            return float(self._entity_data.get('max_occlusion_distance'))
        return float(16.0)

    @property
    def fully_occluded_fraction(self):
        if "fully_occluded_fraction" in self._entity_data:
            return float(self._entity_data.get('fully_occluded_fraction'))
        return float(1.0)

    @property
    def occlusion_exponent(self):
        if "occlusion_exponent" in self._entity_data:
            return float(self._entity_data.get('occlusion_exponent'))
        return float(1.0)

    @property
    def ambient_color(self):
        if "ambient_color" in self._entity_data:
            return parse_int_vector(self._entity_data.get('ambient_color'))
        return parse_int_vector("0 0 0")


class light_irradvolume:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def targetname(self):
        if "targetname" in self._entity_data:
            return self._entity_data.get('targetname')
        return None

    @property
    def sortkey(self):
        if "sortkey" in self._entity_data:
            return int(self._entity_data.get('sortkey'))
        return int(1)

    @property
    def voxelsize(self):
        if "voxelsize" in self._entity_data:
            return float(self._entity_data.get('voxelsize'))
        return float(64.0)

    @property
    def lightmaxdist(self):
        if "lightmaxdist" in self._entity_data:
            return float(self._entity_data.get('lightmaxdist'))
        return float(512.0)

    @property
    def fademindist(self):
        if "fademindist" in self._entity_data:
            return float(self._entity_data.get('fademindist'))
        return float(2048)

    @property
    def fademaxdist(self):
        if "fademaxdist" in self._entity_data:
            return float(self._entity_data.get('fademaxdist'))
        return float(2560)

    @property
    def boundary_maxradius(self):
        if "boundary_maxradius" in self._entity_data:
            return float(self._entity_data.get('boundary_maxradius'))
        return float(2000)

    @property
    def boundary_tracebias(self):
        if "boundary_tracebias" in self._entity_data:
            return float(self._entity_data.get('boundary_tracebias'))
        return float(8.0)

    @property
    def boundary_tracebackfaces(self):
        if "boundary_tracebackfaces" in self._entity_data:
            return bool(self._entity_data.get('boundary_tracebackfaces'))
        return bool(1)

    @property
    def boundary_traceoccluders(self):
        if "boundary_traceoccluders" in self._entity_data:
            return bool(self._entity_data.get('boundary_traceoccluders'))
        return bool(1)

    @property
    def editcommand_traceshape(self):
        if "editcommand_traceshape" in self._entity_data:
            return bool(self._entity_data.get('editcommand_traceshape'))
        return bool(0)

    @property
    def convex_volume(self):
        if "convex_volume" in self._entity_data:
            return self._entity_data.get('convex_volume')
        return "6 ; -64 0 0 32 0 0 ; 64 0 0 -32 0 0 ; 0 -64 0 0 32 0 ; 0 64 0 0 -32 0 ; 0 0 -64 0 0 32 ; 0 0 64 0 0 -32"

    @property
    def convex_max_planes(self):
        if "convex_max_planes" in self._entity_data:
            return int(self._entity_data.get('convex_max_planes'))
        return int(16)

    @property
    def convex_default_plane_softness(self):
        if "convex_default_plane_softness" in self._entity_data:
            return float(self._entity_data.get('convex_default_plane_softness'))
        return float(4.0)


class TimeOfDay:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def classnameoverride(self):
        if "classnameoverride" in self._entity_data:
            return self._entity_data.get('classnameoverride')
        return "env_time_of_day"


class RealisticDayNightCycle:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def time(self):
        if "time" in self._entity_data:
            return self._entity_data.get('time')
        return "15.0"

    @property
    def date(self):
        if "date" in self._entity_data:
            return self._entity_data.get('date')
        return "172"

    @property
    def daylength(self):
        if "daylength" in self._entity_data:
            return float(self._entity_data.get('daylength'))
        return float(300.0)

    @property
    def location(self):
        if "location" in self._entity_data:
            return self._entity_data.get('location')
        return "Seattle"

    @property
    def custom_timezone(self):
        if "custom_timezone" in self._entity_data:
            return self._entity_data.get('custom_timezone')
        return "UTC-08:00"

    @property
    def custom_latitude(self):
        if "custom_latitude" in self._entity_data:
            return float(self._entity_data.get('custom_latitude'))
        return float(47.6106)

    @property
    def custom_longitude(self):
        if "custom_longitude" in self._entity_data:
            return float(self._entity_data.get('custom_longitude'))
        return float(-122.1994)

    @property
    def synodic_month(self):
        if "synodic_month" in self._entity_data:
            return float(self._entity_data.get('synodic_month'))
        return float(29.5)

    @property
    def lunar_phase(self):
        if "lunar_phase" in self._entity_data:
            return self._entity_data.get('lunar_phase')
        return "3.1416"


class env_time_of_day(Targetname, TimeOfDay, RealisticDayNightCycle):
    pass

    @property
    def script(self):
        if "script" in self._entity_data:
            return self._entity_data.get('script')
        return "env_time_of_day"


class light_omni(light_base, light_base_legacy_params, light_base_attenuation_params):
    pass

    @property
    def castshadows(self):
        if "castshadows" in self._entity_data:
            return self._entity_data.get('castshadows')
        return "0"


class light_spot(light_base, light_base_legacy_params, light_base_attenuation_params, CanBeClientOnly):
    pass

    @property
    def lightcookie(self):
        if "lightcookie" in self._entity_data:
            return self._entity_data.get('lightcookie')
        return ""

    @property
    def falloff(self):
        if "falloff" in self._entity_data:
            return float(self._entity_data.get('falloff'))
        return float(1)

    @property
    def innerconeangle(self):
        if "innerconeangle" in self._entity_data:
            return float(self._entity_data.get('innerconeangle'))
        return float(45)

    @property
    def outerconeangle(self):
        if "outerconeangle" in self._entity_data:
            return float(self._entity_data.get('outerconeangle'))
        return float(60)

    @property
    def shadowfademindist(self):
        if "shadowfademindist" in self._entity_data:
            return float(self._entity_data.get('shadowfademindist'))
        return float(-250)

    @property
    def shadowfademaxdist(self):
        if "shadowfademaxdist" in self._entity_data:
            return float(self._entity_data.get('shadowfademaxdist'))
        return float(1000)

    @property
    def shadowtexturewidth(self):
        if "shadowtexturewidth" in self._entity_data:
            return int(self._entity_data.get('shadowtexturewidth'))
        return int(0)

    @property
    def shadowtextureheight(self):
        if "shadowtextureheight" in self._entity_data:
            return int(self._entity_data.get('shadowtextureheight'))
        return int(0)


class light_ortho(light_base, light_base_legacy_params, CanBeClientOnly):
    pass

    @property
    def lightcookie(self):
        if "lightcookie" in self._entity_data:
            return self._entity_data.get('lightcookie')
        return ""

    @property
    def ortholightwidth(self):
        if "ortholightwidth" in self._entity_data:
            return float(self._entity_data.get('ortholightwidth'))
        return float(512.0)

    @property
    def ortholightheight(self):
        if "ortholightheight" in self._entity_data:
            return float(self._entity_data.get('ortholightheight'))
        return float(512.0)

    @property
    def range(self):
        if "range" in self._entity_data:
            return float(self._entity_data.get('range'))
        return float(2048.0)

    @property
    def angulardiameter(self):
        if "angulardiameter" in self._entity_data:
            return float(self._entity_data.get('angulardiameter'))
        return float(1.0)

    @property
    def shadowtexturewidth(self):
        if "shadowtexturewidth" in self._entity_data:
            return int(self._entity_data.get('shadowtexturewidth'))
        return int(0)


class light_importance_volume:
    pass

    icon_sprite =  "materials/editor/light_importance_volume.vmat"

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def box_mins(self):
        if "box_mins" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_mins'))
        return parse_int_vector("-512 -512 -512")

    @property
    def box_maxs(self):
        if "box_maxs" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_maxs'))
        return parse_int_vector("512 512 512")


class IndoorOutdoorLevel:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def indoor_outdoor_level(self):
        if "indoor_outdoor_level" in self._entity_data:
            return self._entity_data.get('indoor_outdoor_level')
        return "0"


class SetBrightnessColor:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data


class BaseLightProbeVolume:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def lightprobetexture(self):
        if "lightprobetexture" in self._entity_data:
            return self._entity_data.get('lightprobetexture')
        return ""

    @property
    def box_mins(self):
        if "box_mins" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_mins'))
        return parse_int_vector("-72 -72 -72")

    @property
    def box_maxs(self):
        if "box_maxs" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_maxs'))
        return parse_int_vector("72 72 72")

    @property
    def voxel_size(self):
        if "voxel_size" in self._entity_data:
            return self._entity_data.get('voxel_size')
        return "48.0"

    @property
    def flood_fill(self):
        if "flood_fill" in self._entity_data:
            return bool(self._entity_data.get('flood_fill'))
        return bool(1)

    @property
    def voxelize(self):
        if "voxelize" in self._entity_data:
            return bool(self._entity_data.get('voxelize'))
        return bool(1)

    @property
    def light_probe_volume_from_cubemap(self):
        if "light_probe_volume_from_cubemap" in self._entity_data:
            return bool(self._entity_data.get('light_probe_volume_from_cubemap'))
        return bool(0)

    @property
    def lightgroup(self):
        if "lightgroup" in self._entity_data:
            return self._entity_data.get('lightgroup')
        return ""

    @property
    def moveable(self):
        if "moveable" in self._entity_data:
            return bool(self._entity_data.get('moveable'))
        return bool(0)

    @property
    def storage(self):
        if "storage" in self._entity_data:
            return self._entity_data.get('storage')
        return "-1"


class BaseCubemap:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def cubemaptexture(self):
        if "cubemaptexture" in self._entity_data:
            return self._entity_data.get('cubemaptexture')
        return ""

    @property
    def bakenearz(self):
        if "bakenearz" in self._entity_data:
            return float(self._entity_data.get('bakenearz'))
        return float(2.0)

    @property
    def bakefarz(self):
        if "bakefarz" in self._entity_data:
            return float(self._entity_data.get('bakefarz'))
        return float(4096.0)

    @property
    def lightgroup(self):
        if "lightgroup" in self._entity_data:
            return self._entity_data.get('lightgroup')
        return ""

    @property
    def moveable(self):
        if "moveable" in self._entity_data:
            return bool(self._entity_data.get('moveable'))
        return bool(0)


class env_light_probe_volume(Targetname, Parentname, EnableDisable, BaseLightProbeVolume, IndoorOutdoorLevel):
    pass


class env_cubemap(Targetname, Parentname, EnableDisable, BaseCubemap, IndoorOutdoorLevel):
    pass

    @property
    def influenceradius(self):
        if "influenceradius" in self._entity_data:
            return float(self._entity_data.get('influenceradius'))
        return float(512.0)


class env_cubemap_box(Targetname, Parentname, EnableDisable, BaseCubemap, IndoorOutdoorLevel):
    pass

    @property
    def box_mins(self):
        if "box_mins" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_mins'))
        return parse_int_vector("-72 -72 -72")

    @property
    def box_maxs(self):
        if "box_maxs" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_maxs'))
        return parse_int_vector("72 72 72")


class env_combined_light_probe_volume(Targetname, Parentname, EnableDisable, BaseCubemap, BaseLightProbeVolume,
                                      SetBrightnessColor, IndoorOutdoorLevel):
    pass


class TalkNPC(BaseNPC):
    pass

    @property
    def UseSentence(self):
        if "UseSentence" in self._entity_data:
            return self._entity_data.get('UseSentence')
        return None

    @property
    def UnUseSentence(self):
        if "UnUseSentence" in self._entity_data:
            return self._entity_data.get('UnUseSentence')
        return None

    @property
    def DontUseSpeechSemaphore(self):
        if "DontUseSpeechSemaphore" in self._entity_data:
            return self._entity_data.get('DontUseSpeechSemaphore')
        return "0"


class PlayerCompanion(BaseNPC):
    pass

    @property
    def AlwaysTransition(self):
        if "AlwaysTransition" in self._entity_data:
            return bool(self._entity_data.get('AlwaysTransition'))
        return bool(False)

    @property
    def DontPickupWeapons(self):
        if "DontPickupWeapons" in self._entity_data:
            return bool(self._entity_data.get('DontPickupWeapons'))
        return bool(False)

    @property
    def GameEndAlly(self):
        if "GameEndAlly" in self._entity_data:
            return bool(self._entity_data.get('GameEndAlly'))
        return bool(False)


class RappelNPC(BaseNPC):
    pass

    @property
    def waitingtorappel(self):
        if "waitingtorappel" in self._entity_data:
            return bool(self._entity_data.get('waitingtorappel'))
        return bool(False)


class VehicleDriverNPC(BaseNPC):
    pass

    @property
    def vehicle(self):
        if "vehicle" in self._entity_data:
            return self._entity_data.get('vehicle')
        return None

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start Inactive': (65536, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class npc_vehicledriver(VehicleDriverNPC):
    pass

    @property
    def drivermaxspeed(self):
        if "drivermaxspeed" in self._entity_data:
            return float(self._entity_data.get('drivermaxspeed'))
        return float(1)

    @property
    def driverminspeed(self):
        if "driverminspeed" in self._entity_data:
            return float(self._entity_data.get('driverminspeed'))
        return float(0)


class npc_bullseye(Parentname, BaseNPC):
    pass

    icon_sprite =  "editor/bullseye.vmat"

    @property
    def health(self):
        if "health" in self._entity_data:
            return int(self._entity_data.get('health'))
        return int(35)

    @property
    def minangle(self):
        if "minangle" in self._entity_data:
            return self._entity_data.get('minangle')
        return "360"

    @property
    def mindist(self):
        if "mindist" in self._entity_data:
            return self._entity_data.get('mindist')
        return "0"

    @property
    def autoaimradius(self):
        if "autoaimradius" in self._entity_data:
            return float(self._entity_data.get('autoaimradius'))
        return float(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Not Solid': (65536, 0), 'Take No Damage': (131072, 0),
                                   'Enemy Damage Only': (262144, 0), 'Bleed': (524288, 0),
                                   'Perfect Accuracy': (1048576, 0),
                                   'Collide against physics objects (Creates VPhysics Shadow)': (2097152, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class npc_enemyfinder(Parentname, BaseNPC):
    pass

    icon_sprite =  "editor/enemyfinder.vmat"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Check Visibility': (65536, 1), 'APC Visibility checks': (131072, 0),
                                   'Short memory': (262144, 0), 'Can be an enemy': (524288, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def FieldOfView(self):
        if "FieldOfView" in self._entity_data:
            return self._entity_data.get('FieldOfView')
        return "0.2"

    @property
    def MinSearchDist(self):
        if "MinSearchDist" in self._entity_data:
            return int(self._entity_data.get('MinSearchDist'))
        return int(0)

    @property
    def MaxSearchDist(self):
        if "MaxSearchDist" in self._entity_data:
            return int(self._entity_data.get('MaxSearchDist'))
        return int(2048)

    @property
    def freepass_timetotrigger(self):
        if "freepass_timetotrigger" in self._entity_data:
            return float(self._entity_data.get('freepass_timetotrigger'))
        return float(0)

    @property
    def freepass_duration(self):
        if "freepass_duration" in self._entity_data:
            return float(self._entity_data.get('freepass_duration'))
        return float(0)

    @property
    def freepass_movetolerance(self):
        if "freepass_movetolerance" in self._entity_data:
            return float(self._entity_data.get('freepass_movetolerance'))
        return float(120)

    @property
    def freepass_refillrate(self):
        if "freepass_refillrate" in self._entity_data:
            return float(self._entity_data.get('freepass_refillrate'))
        return float(0.5)

    @property
    def freepass_peektime(self):
        if "freepass_peektime" in self._entity_data:
            return float(self._entity_data.get('freepass_peektime'))
        return float(0)

    @property
    def StartOn(self):
        if "StartOn" in self._entity_data:
            return bool(self._entity_data.get('StartOn'))
        return bool(1)


class ai_goal_operator(Targetname, EnableDisable):
    pass

    @property
    def actor(self):
        if "actor" in self._entity_data:
            return self._entity_data.get('actor')
        return ""

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return ""

    @property
    def contexttarget(self):
        if "contexttarget" in self._entity_data:
            return self._entity_data.get('contexttarget')
        return ""

    @property
    def state(self):
        if "state" in self._entity_data:
            return self._entity_data.get('state')
        return "0"

    @property
    def moveto(self):
        if "moveto" in self._entity_data:
            return self._entity_data.get('moveto')
        return "1"


class monster_generic(BaseNPC):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Not solid': (65536, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return None

    @property
    def body(self):
        if "body" in self._entity_data:
            return int(self._entity_data.get('body'))
        return int(0)


class generic_actor(BaseNPC, Parentname, Studiomodel):
    pass

    @property
    def hull_name(self):
        if "hull_name" in self._entity_data:
            return self._entity_data.get('hull_name')
        return "Human"

    @property
    def footstep_script(self):
        if "footstep_script" in self._entity_data:
            return self._entity_data.get('footstep_script')
        return ""

    @property
    def act_as_flyer(self):
        if "act_as_flyer" in self._entity_data:
            return self._entity_data.get('act_as_flyer')
        return "0"

    @property
    def is_friendly_npc(self):
        if "is_friendly_npc" in self._entity_data:
            return bool(self._entity_data.get('is_friendly_npc'))
        return bool(0)


class cycler_actor(BaseNPC):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return None

    @property
    def Sentence(self):
        if "Sentence" in self._entity_data:
            return self._entity_data.get('Sentence')
        return ""


class npc_maker(BaseNPCMaker):
    pass

    icon_sprite =  "editor/npc_maker.vmat"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Fade Corpse': (16, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def NPCType(self):
        if "NPCType" in self._entity_data:
            return self._entity_data.get('NPCType')
        return None

    @property
    def NPCTargetname(self):
        if "NPCTargetname" in self._entity_data:
            return self._entity_data.get('NPCTargetname')
        return None

    @property
    def NPCSquadname(self):
        if "NPCSquadname" in self._entity_data:
            return self._entity_data.get('NPCSquadname')
        return None

    @property
    def NPCHintGroup(self):
        if "NPCHintGroup" in self._entity_data:
            return self._entity_data.get('NPCHintGroup')
        return None

    @property
    def additionalequipment(self):
        if "additionalequipment" in self._entity_data:
            return self._entity_data.get('additionalequipment')
        return "0"


class BaseScripted(Targetname, Parentname):
    pass

    @property
    def m_iszEntity(self):
        if "m_iszEntity" in self._entity_data:
            return self._entity_data.get('m_iszEntity')
        return None

    @property
    def m_iszIdle(self):
        if "m_iszIdle" in self._entity_data:
            return self._entity_data.get('m_iszIdle')
        return ""

    @property
    def m_iszEntry(self):
        if "m_iszEntry" in self._entity_data:
            return self._entity_data.get('m_iszEntry')
        return ""

    @property
    def m_iszPlay(self):
        if "m_iszPlay" in self._entity_data:
            return self._entity_data.get('m_iszPlay')
        return ""

    @property
    def m_iszPostIdle(self):
        if "m_iszPostIdle" in self._entity_data:
            return self._entity_data.get('m_iszPostIdle')
        return ""

    @property
    def m_iszCustomMove(self):
        if "m_iszCustomMove" in self._entity_data:
            return self._entity_data.get('m_iszCustomMove')
        return ""

    @property
    def sync_group(self):
        if "sync_group" in self._entity_data:
            return self._entity_data.get('sync_group')
        return None

    @property
    def m_bLoopActionSequence(self):
        if "m_bLoopActionSequence" in self._entity_data:
            return bool(self._entity_data.get('m_bLoopActionSequence'))
        return bool(0)

    @property
    def m_bSynchPostIdles(self):
        if "m_bSynchPostIdles" in self._entity_data:
            return bool(self._entity_data.get('m_bSynchPostIdles'))
        return bool(0)

    @property
    def m_bAllowCustomInterruptConditions(self):
        if "m_bAllowCustomInterruptConditions" in self._entity_data:
            return bool(self._entity_data.get('m_bAllowCustomInterruptConditions'))
        return bool(0)

    @property
    def conflict_response(self):
        if "conflict_response" in self._entity_data:
            return self._entity_data.get('conflict_response')
        return "0"

    @property
    def m_nGroundIKPreference(self):
        if "m_nGroundIKPreference" in self._entity_data:
            return self._entity_data.get('m_nGroundIKPreference')
        return "0"

    @property
    def m_flRadius(self):
        if "m_flRadius" in self._entity_data:
            return int(self._entity_data.get('m_flRadius'))
        return int(0)

    @property
    def m_flRepeat(self):
        if "m_flRepeat" in self._entity_data:
            return int(self._entity_data.get('m_flRepeat'))
        return int(0)

    @property
    def m_fMoveTo(self):
        if "m_fMoveTo" in self._entity_data:
            return self._entity_data.get('m_fMoveTo')
        return "1"

    @property
    def m_iszNextScript(self):
        if "m_iszNextScript" in self._entity_data:
            return self._entity_data.get('m_iszNextScript')
        return None

    @property
    def m_bIgnoreGravity(self):
        if "m_bIgnoreGravity" in self._entity_data:
            return bool(self._entity_data.get('m_bIgnoreGravity'))
        return bool(0)

    @property
    def m_bDisableNPCCollisions(self):
        if "m_bDisableNPCCollisions" in self._entity_data:
            return bool(self._entity_data.get('m_bDisableNPCCollisions'))
        return bool(0)

    @property
    def m_bKeepAnimgraphLockedPost(self):
        if "m_bKeepAnimgraphLockedPost" in self._entity_data:
            return bool(self._entity_data.get('m_bKeepAnimgraphLockedPost'))
        return bool(0)


class scripted_sentence(Targetname):
    pass

    icon_sprite =  "editor/scripted_sentence.vmat"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Fire Once': (1, 1), 'Followers Only': (2, 0), 'Interrupt Speech': (4, 1),
                                   'Concurrent': (8, 0), 'Speak to Activator': (16, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def sentence(self):
        if "sentence" in self._entity_data:
            return self._entity_data.get('sentence')
        return ""

    @property
    def entity(self):
        if "entity" in self._entity_data:
            return self._entity_data.get('entity')
        return None

    @property
    def delay(self):
        if "delay" in self._entity_data:
            return self._entity_data.get('delay')
        return "0"

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return int(self._entity_data.get('radius'))
        return int(512)

    @property
    def refire(self):
        if "refire" in self._entity_data:
            return self._entity_data.get('refire')
        return "3"

    @property
    def listener(self):
        if "listener" in self._entity_data:
            return self._entity_data.get('listener')
        return None

    @property
    def volume(self):
        if "volume" in self._entity_data:
            return self._entity_data.get('volume')
        return "10"

    @property
    def attenuation(self):
        if "attenuation" in self._entity_data:
            return self._entity_data.get('attenuation')
        return "0"


class scripted_target(Targetname, Parentname):
    pass

    icon_sprite =  "editor/info_target.vmat"

    @property
    def StartDisabled(self):
        if "StartDisabled" in self._entity_data:
            return bool(self._entity_data.get('StartDisabled'))
        return bool(1)

    @property
    def m_iszEntity(self):
        if "m_iszEntity" in self._entity_data:
            return self._entity_data.get('m_iszEntity')
        return None

    @property
    def m_flRadius(self):
        if "m_flRadius" in self._entity_data:
            return int(self._entity_data.get('m_flRadius'))
        return int(0)

    @property
    def MoveSpeed(self):
        if "MoveSpeed" in self._entity_data:
            return int(self._entity_data.get('MoveSpeed'))
        return int(5)

    @property
    def PauseDuration(self):
        if "PauseDuration" in self._entity_data:
            return int(self._entity_data.get('PauseDuration'))
        return int(0)

    @property
    def EffectDuration(self):
        if "EffectDuration" in self._entity_data:
            return int(self._entity_data.get('EffectDuration'))
        return int(2)

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None


class base_ai_relationship(Targetname):
    pass

    icon_sprite =  "editor/ai_relationship.vmat"

    @property
    def disposition(self):
        if "disposition" in self._entity_data:
            return self._entity_data.get('disposition')
        return "3"

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return float(self._entity_data.get('radius'))
        return float(0)

    @property
    def rank(self):
        if "rank" in self._entity_data:
            return int(self._entity_data.get('rank'))
        return int(0)

    @property
    def StartActive(self):
        if "StartActive" in self._entity_data:
            return bool(self._entity_data.get('StartActive'))
        return bool(0)

    @property
    def Reciprocal(self):
        if "Reciprocal" in self._entity_data:
            return bool(self._entity_data.get('Reciprocal'))
        return bool(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {"Notify subject of target's location": (1, 0),
                                   "Notify target of subject's location": (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class ai_relationship(base_ai_relationship):
    pass

    icon_sprite =  "editor/ai_relationship.vmat"

    @property
    def subject(self):
        if "subject" in self._entity_data:
            return self._entity_data.get('subject')
        return ""

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return ""


class LeadGoalBase(Targetname):
    pass

    @property
    def actor(self):
        if "actor" in self._entity_data:
            return self._entity_data.get('actor')
        return None

    @property
    def goal(self):
        if "goal" in self._entity_data:
            return self._entity_data.get('goal')
        return None

    @property
    def WaitPointName(self):
        if "WaitPointName" in self._entity_data:
            return self._entity_data.get('WaitPointName')
        return None

    @property
    def WaitDistance(self):
        if "WaitDistance" in self._entity_data:
            return float(self._entity_data.get('WaitDistance'))
        return float(0)

    @property
    def LeadDistance(self):
        if "LeadDistance" in self._entity_data:
            return float(self._entity_data.get('LeadDistance'))
        return float(64)

    @property
    def RetrieveDistance(self):
        if "RetrieveDistance" in self._entity_data:
            return float(self._entity_data.get('RetrieveDistance'))
        return float(96)

    @property
    def SuccessDistance(self):
        if "SuccessDistance" in self._entity_data:
            return float(self._entity_data.get('SuccessDistance'))
        return float(0)

    @property
    def Run(self):
        if "Run" in self._entity_data:
            return bool(self._entity_data.get('Run'))
        return bool(0)

    @property
    def Retrieve(self):
        if "Retrieve" in self._entity_data:
            return self._entity_data.get('Retrieve')
        return "1"

    @property
    def ComingBackWaitForSpeak(self):
        if "ComingBackWaitForSpeak" in self._entity_data:
            return self._entity_data.get('ComingBackWaitForSpeak')
        return "1"

    @property
    def RetrieveWaitForSpeak(self):
        if "RetrieveWaitForSpeak" in self._entity_data:
            return self._entity_data.get('RetrieveWaitForSpeak')
        return "1"

    @property
    def DontSpeakStart(self):
        if "DontSpeakStart" in self._entity_data:
            return self._entity_data.get('DontSpeakStart')
        return "0"

    @property
    def LeadDuringCombat(self):
        if "LeadDuringCombat" in self._entity_data:
            return self._entity_data.get('LeadDuringCombat')
        return "0"

    @property
    def GagLeader(self):
        if "GagLeader" in self._entity_data:
            return self._entity_data.get('GagLeader')
        return "0"

    @property
    def AttractPlayerConceptModifier(self):
        if "AttractPlayerConceptModifier" in self._entity_data:
            return self._entity_data.get('AttractPlayerConceptModifier')
        return ""

    @property
    def WaitOverConceptModifier(self):
        if "WaitOverConceptModifier" in self._entity_data:
            return self._entity_data.get('WaitOverConceptModifier')
        return ""

    @property
    def ArrivalConceptModifier(self):
        if "ArrivalConceptModifier" in self._entity_data:
            return self._entity_data.get('ArrivalConceptModifier')
        return ""

    @property
    def PostArrivalConceptModifier(self):
        if "PostArrivalConceptModifier" in self._entity_data:
            return self._entity_data.get('PostArrivalConceptModifier')
        return None

    @property
    def SuccessConceptModifier(self):
        if "SuccessConceptModifier" in self._entity_data:
            return self._entity_data.get('SuccessConceptModifier')
        return ""

    @property
    def FailureConceptModifier(self):
        if "FailureConceptModifier" in self._entity_data:
            return self._entity_data.get('FailureConceptModifier')
        return ""

    @property
    def ComingBackConceptModifier(self):
        if "ComingBackConceptModifier" in self._entity_data:
            return self._entity_data.get('ComingBackConceptModifier')
        return ""

    @property
    def RetrieveConceptModifier(self):
        if "RetrieveConceptModifier" in self._entity_data:
            return self._entity_data.get('RetrieveConceptModifier')
        return ""

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'No def success': (1, 0), 'No def failure': (2, 0),
                                   'Use goal facing': (4, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class ai_goal_lead(LeadGoalBase):
    pass

    icon_sprite =  "editor/ai_goal_lead.vmat"

    @property
    def SearchType(self):
        if "SearchType" in self._entity_data:
            return self._entity_data.get('SearchType')
        return "0"


class FollowGoal(Targetname):
    pass

    @property
    def actor(self):
        if "actor" in self._entity_data:
            return self._entity_data.get('actor')
        return None

    @property
    def goal(self):
        if "goal" in self._entity_data:
            return self._entity_data.get('goal')
        return None

    @property
    def SearchType(self):
        if "SearchType" in self._entity_data:
            return self._entity_data.get('SearchType')
        return "0"

    @property
    def StartActive(self):
        if "StartActive" in self._entity_data:
            return bool(self._entity_data.get('StartActive'))
        return bool(0)

    @property
    def MaximumState(self):
        if "MaximumState" in self._entity_data:
            return self._entity_data.get('MaximumState')
        return "1"

    @property
    def Formation(self):
        if "Formation" in self._entity_data:
            return self._entity_data.get('Formation')
        return "0"


class ai_goal_follow(FollowGoal):
    pass

    icon_sprite =  "editor/ai_goal_follow.vmat"


class ai_goal_injured_follow(FollowGoal):
    pass

    icon_sprite =  "editor/ai_goal_follow.vmat"


class ai_battle_line(Targetname, Parentname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {"Use parent's orientation": (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def actor(self):
        if "actor" in self._entity_data:
            return self._entity_data.get('actor')
        return None

    @property
    def MatchByNameOnly(self):
        if "MatchByNameOnly" in self._entity_data:
            return bool(self._entity_data.get('MatchByNameOnly'))
        return bool(0)

    @property
    def Active(self):
        if "Active" in self._entity_data:
            return bool(self._entity_data.get('Active'))
        return bool(0)

    @property
    def Strict(self):
        if "Strict" in self._entity_data:
            return bool(self._entity_data.get('Strict'))
        return bool(1)


class ai_goal_fightfromcover(Targetname):
    pass

    icon_sprite =  "editor/ai_goal_follow.vmat"

    @property
    def actor(self):
        if "actor" in self._entity_data:
            return self._entity_data.get('actor')
        return None

    @property
    def goal(self):
        if "goal" in self._entity_data:
            return self._entity_data.get('goal')
        return None

    @property
    def DirectionalMarker(self):
        if "DirectionalMarker" in self._entity_data:
            return self._entity_data.get('DirectionalMarker')
        return ""

    @property
    def GenericHintType(self):
        if "GenericHintType" in self._entity_data:
            return self._entity_data.get('GenericHintType')
        return ""

    @property
    def width(self):
        if "width" in self._entity_data:
            return float(self._entity_data.get('width'))
        return float(600)

    @property
    def length(self):
        if "length" in self._entity_data:
            return float(self._entity_data.get('length'))
        return float(480)

    @property
    def height(self):
        if "height" in self._entity_data:
            return float(self._entity_data.get('height'))
        return float(2400)

    @property
    def bias(self):
        if "bias" in self._entity_data:
            return float(self._entity_data.get('bias'))
        return float(60)

    @property
    def StartActive(self):
        if "StartActive" in self._entity_data:
            return bool(self._entity_data.get('StartActive'))
        return bool(0)


class ai_goal_standoff(Targetname):
    pass

    icon_sprite =  "editor/ai_goal_standoff.vmat"

    @property
    def actor(self):
        if "actor" in self._entity_data:
            return self._entity_data.get('actor')
        return None

    @property
    def SearchType(self):
        if "SearchType" in self._entity_data:
            return self._entity_data.get('SearchType')
        return "0"

    @property
    def StartActive(self):
        if "StartActive" in self._entity_data:
            return bool(self._entity_data.get('StartActive'))
        return bool(0)

    @property
    def HintGroupChangeReaction(self):
        if "HintGroupChangeReaction" in self._entity_data:
            return self._entity_data.get('HintGroupChangeReaction')
        return "1"

    @property
    def Aggressiveness(self):
        if "Aggressiveness" in self._entity_data:
            return self._entity_data.get('Aggressiveness')
        return "2"

    @property
    def PlayerBattleline(self):
        if "PlayerBattleline" in self._entity_data:
            return bool(self._entity_data.get('PlayerBattleline'))
        return bool(1)

    @property
    def StayAtCover(self):
        if "StayAtCover" in self._entity_data:
            return bool(self._entity_data.get('StayAtCover'))
        return bool(0)

    @property
    def AbandonIfEnemyHides(self):
        if "AbandonIfEnemyHides" in self._entity_data:
            return bool(self._entity_data.get('AbandonIfEnemyHides'))
        return bool(0)


class ai_goal_police(Targetname, Parentname):
    pass

    icon_sprite =  "editor/ai_goal_police.vmat"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Knock-out target past crossing plane': (2, 0), 'Do not leave post': (4, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def policeradius(self):
        if "policeradius" in self._entity_data:
            return float(self._entity_data.get('policeradius'))
        return float(512)

    @property
    def policetarget(self):
        if "policetarget" in self._entity_data:
            return self._entity_data.get('policetarget')
        return ""


class assault_rallypoint(Targetname, Parentname):
    pass

    icon_sprite =  "editor/assault_rally.vmat"

    @property
    def assaultpoint(self):
        if "assaultpoint" in self._entity_data:
            return self._entity_data.get('assaultpoint')
        return ""

    @property
    def assaultdelay(self):
        if "assaultdelay" in self._entity_data:
            return float(self._entity_data.get('assaultdelay'))
        return float(0)

    @property
    def rallysequence(self):
        if "rallysequence" in self._entity_data:
            return self._entity_data.get('rallysequence')
        return ""

    @property
    def priority(self):
        if "priority" in self._entity_data:
            return int(self._entity_data.get('priority'))
        return int(1)

    @property
    def forcecrouch(self):
        if "forcecrouch" in self._entity_data:
            return bool(self._entity_data.get('forcecrouch'))
        return bool(0)

    @property
    def urgent(self):
        if "urgent" in self._entity_data:
            return bool(self._entity_data.get('urgent'))
        return bool(0)

    @property
    def lockpoint(self):
        if "lockpoint" in self._entity_data:
            return bool(self._entity_data.get('lockpoint'))
        return bool(1)


class assault_assaultpoint(Targetname, Parentname):
    pass

    icon_sprite =  "editor/assault_point.vmat"

    @property
    def assaultgroup(self):
        if "assaultgroup" in self._entity_data:
            return self._entity_data.get('assaultgroup')
        return ""

    @property
    def nextassaultpoint(self):
        if "nextassaultpoint" in self._entity_data:
            return self._entity_data.get('nextassaultpoint')
        return None

    @property
    def assaulttimeout(self):
        if "assaulttimeout" in self._entity_data:
            return float(self._entity_data.get('assaulttimeout'))
        return float(3.0)

    @property
    def clearoncontact(self):
        if "clearoncontact" in self._entity_data:
            return bool(self._entity_data.get('clearoncontact'))
        return bool(0)

    @property
    def allowdiversion(self):
        if "allowdiversion" in self._entity_data:
            return bool(self._entity_data.get('allowdiversion'))
        return bool(0)

    @property
    def allowdiversionradius(self):
        if "allowdiversionradius" in self._entity_data:
            return float(self._entity_data.get('allowdiversionradius'))
        return float(0)

    @property
    def nevertimeout(self):
        if "nevertimeout" in self._entity_data:
            return bool(self._entity_data.get('nevertimeout'))
        return bool(0)

    @property
    def strict(self):
        if "strict" in self._entity_data:
            return self._entity_data.get('strict')
        return "0"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Clear this point upon arrival, UNCONDITIONALLY': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def forcecrouch(self):
        if "forcecrouch" in self._entity_data:
            return bool(self._entity_data.get('forcecrouch'))
        return bool(0)

    @property
    def urgent(self):
        if "urgent" in self._entity_data:
            return bool(self._entity_data.get('urgent'))
        return bool(0)

    @property
    def assaulttolerance(self):
        if "assaulttolerance" in self._entity_data:
            return self._entity_data.get('assaulttolerance')
        return "36"


class ai_goal_assault(Targetname):
    pass

    @property
    def actor(self):
        if "actor" in self._entity_data:
            return self._entity_data.get('actor')
        return ""

    @property
    def rallypoint(self):
        if "rallypoint" in self._entity_data:
            return self._entity_data.get('rallypoint')
        return ""

    @property
    def SearchType(self):
        if "SearchType" in self._entity_data:
            return self._entity_data.get('SearchType')
        return "0"

    @property
    def StartActive(self):
        if "StartActive" in self._entity_data:
            return bool(self._entity_data.get('StartActive'))
        return bool(0)

    @property
    def AssaultCue(self):
        if "AssaultCue" in self._entity_data:
            return self._entity_data.get('AssaultCue')
        return "1"

    @property
    def RallySelectMethod(self):
        if "RallySelectMethod" in self._entity_data:
            return self._entity_data.get('RallySelectMethod')
        return "0"

    @property
    def BranchingMethod(self):
        if "BranchingMethod" in self._entity_data:
            return self._entity_data.get('BranchingMethod')
        return "0"


class BaseActBusy(Targetname):
    pass

    @property
    def actor(self):
        if "actor" in self._entity_data:
            return self._entity_data.get('actor')
        return ""

    @property
    def StartActive(self):
        if "StartActive" in self._entity_data:
            return bool(self._entity_data.get('StartActive'))
        return bool(0)

    @property
    def SearchType(self):
        if "SearchType" in self._entity_data:
            return self._entity_data.get('SearchType')
        return "0"

    @property
    def busysearchrange(self):
        if "busysearchrange" in self._entity_data:
            return float(self._entity_data.get('busysearchrange'))
        return float(2048)

    @property
    def visibleonly(self):
        if "visibleonly" in self._entity_data:
            return bool(self._entity_data.get('visibleonly'))
        return bool(0)


class ai_goal_actbusy(BaseActBusy):
    pass

    @property
    def seeentity(self):
        if "seeentity" in self._entity_data:
            return self._entity_data.get('seeentity')
        return ""

    @property
    def seeentitytimeout(self):
        if "seeentitytimeout" in self._entity_data:
            return self._entity_data.get('seeentitytimeout')
        return "1"

    @property
    def disablesearch(self):
        if "disablesearch" in self._entity_data:
            return self._entity_data.get('disablesearch')
        return "0"

    @property
    def sightmethod(self):
        if "sightmethod" in self._entity_data:
            return self._entity_data.get('sightmethod')
        return "0"

    @property
    def forcetype(self):
        if "forcetype" in self._entity_data:
            return self._entity_data.get('forcetype')
        return "0"

    @property
    def type(self):
        if "type" in self._entity_data:
            return self._entity_data.get('type')
        return "0"

    @property
    def safezone(self):
        if "safezone" in self._entity_data:
            return self._entity_data.get('safezone')
        return ""

    @property
    def allowteleport(self):
        if "allowteleport" in self._entity_data:
            return self._entity_data.get('allowteleport')
        return "0"


class ai_goal_actbusy_queue(BaseActBusy):
    pass

    @property
    def node_exit(self):
        if "node_exit" in self._entity_data:
            return self._entity_data.get('node_exit')
        return ""

    @property
    def node01(self):
        if "node01" in self._entity_data:
            return self._entity_data.get('node01')
        return ""

    @property
    def node02(self):
        if "node02" in self._entity_data:
            return self._entity_data.get('node02')
        return ""

    @property
    def node03(self):
        if "node03" in self._entity_data:
            return self._entity_data.get('node03')
        return ""

    @property
    def node04(self):
        if "node04" in self._entity_data:
            return self._entity_data.get('node04')
        return ""

    @property
    def node05(self):
        if "node05" in self._entity_data:
            return self._entity_data.get('node05')
        return ""

    @property
    def node06(self):
        if "node06" in self._entity_data:
            return self._entity_data.get('node06')
        return ""

    @property
    def node07(self):
        if "node07" in self._entity_data:
            return self._entity_data.get('node07')
        return ""

    @property
    def node08(self):
        if "node08" in self._entity_data:
            return self._entity_data.get('node08')
        return ""

    @property
    def node09(self):
        if "node09" in self._entity_data:
            return self._entity_data.get('node09')
        return ""

    @property
    def node10(self):
        if "node10" in self._entity_data:
            return self._entity_data.get('node10')
        return ""

    @property
    def node11(self):
        if "node11" in self._entity_data:
            return self._entity_data.get('node11')
        return ""

    @property
    def node12(self):
        if "node12" in self._entity_data:
            return self._entity_data.get('node12')
        return ""

    @property
    def node13(self):
        if "node13" in self._entity_data:
            return self._entity_data.get('node13')
        return ""

    @property
    def node14(self):
        if "node14" in self._entity_data:
            return self._entity_data.get('node14')
        return ""

    @property
    def node15(self):
        if "node15" in self._entity_data:
            return self._entity_data.get('node15')
        return ""

    @property
    def node16(self):
        if "node16" in self._entity_data:
            return self._entity_data.get('node16')
        return ""

    @property
    def node17(self):
        if "node17" in self._entity_data:
            return self._entity_data.get('node17')
        return ""

    @property
    def node18(self):
        if "node18" in self._entity_data:
            return self._entity_data.get('node18')
        return ""

    @property
    def node19(self):
        if "node19" in self._entity_data:
            return self._entity_data.get('node19')
        return ""

    @property
    def node20(self):
        if "node20" in self._entity_data:
            return self._entity_data.get('node20')
        return ""

    @property
    def mustreachfront(self):
        if "mustreachfront" in self._entity_data:
            return bool(self._entity_data.get('mustreachfront'))
        return bool(0)


class ai_changetarget(Targetname):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def m_iszNewTarget(self):
        if "m_iszNewTarget" in self._entity_data:
            return self._entity_data.get('m_iszNewTarget')
        return None


class ai_npc_eventresponsesystem(Targetname):
    pass


class ai_changehintgroup(Targetname):
    pass

    @property
    def SearchType(self):
        if "SearchType" in self._entity_data:
            return self._entity_data.get('SearchType')
        return "0"

    @property
    def SearchName(self):
        if "SearchName" in self._entity_data:
            return self._entity_data.get('SearchName')
        return None

    @property
    def NewHintGroup(self):
        if "NewHintGroup" in self._entity_data:
            return self._entity_data.get('NewHintGroup')
        return None

    @property
    def Radius(self):
        if "Radius" in self._entity_data:
            return self._entity_data.get('Radius')
        return "0.0"

    @property
    def hintlimiting(self):
        if "hintlimiting" in self._entity_data:
            return bool(self._entity_data.get('hintlimiting'))
        return bool(0)


class ai_script_conditions(Targetname):
    pass

    @property
    def Actor(self):
        if "Actor" in self._entity_data:
            return self._entity_data.get('Actor')
        return None

    @property
    def StartDisabled(self):
        if "StartDisabled" in self._entity_data:
            return bool(self._entity_data.get('StartDisabled'))
        return bool(1)

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

    @property
    def ScriptStatus(self):
        if "ScriptStatus" in self._entity_data:
            return self._entity_data.get('ScriptStatus')
        return "2"

    @property
    def RequiredTime(self):
        if "RequiredTime" in self._entity_data:
            return float(self._entity_data.get('RequiredTime'))
        return float(0)

    @property
    def MinTimeout(self):
        if "MinTimeout" in self._entity_data:
            return float(self._entity_data.get('MinTimeout'))
        return float(0)

    @property
    def MaxTimeout(self):
        if "MaxTimeout" in self._entity_data:
            return float(self._entity_data.get('MaxTimeout'))
        return float(0)

    @property
    def ActorSeePlayer(self):
        if "ActorSeePlayer" in self._entity_data:
            return self._entity_data.get('ActorSeePlayer')
        return "2"

    @property
    def PlayerActorProximity(self):
        if "PlayerActorProximity" in self._entity_data:
            return float(self._entity_data.get('PlayerActorProximity'))
        return float(0)

    @property
    def PlayerActorFOV(self):
        if "PlayerActorFOV" in self._entity_data:
            return float(self._entity_data.get('PlayerActorFOV'))
        return float(360)

    @property
    def PlayerActorFOVTrueCone(self):
        if "PlayerActorFOVTrueCone" in self._entity_data:
            return self._entity_data.get('PlayerActorFOVTrueCone')
        return "0"

    @property
    def PlayerActorLOS(self):
        if "PlayerActorLOS" in self._entity_data:
            return self._entity_data.get('PlayerActorLOS')
        return "2"

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def ActorSeeTarget(self):
        if "ActorSeeTarget" in self._entity_data:
            return self._entity_data.get('ActorSeeTarget')
        return "2"

    @property
    def ActorTargetProximity(self):
        if "ActorTargetProximity" in self._entity_data:
            return float(self._entity_data.get('ActorTargetProximity'))
        return float(0)

    @property
    def PlayerTargetProximity(self):
        if "PlayerTargetProximity" in self._entity_data:
            return float(self._entity_data.get('PlayerTargetProximity'))
        return float(0)

    @property
    def PlayerTargetFOV(self):
        if "PlayerTargetFOV" in self._entity_data:
            return float(self._entity_data.get('PlayerTargetFOV'))
        return float(360)

    @property
    def PlayerTargetFOVTrueCone(self):
        if "PlayerTargetFOVTrueCone" in self._entity_data:
            return self._entity_data.get('PlayerTargetFOVTrueCone')
        return "0"

    @property
    def PlayerTargetLOS(self):
        if "PlayerTargetLOS" in self._entity_data:
            return self._entity_data.get('PlayerTargetLOS')
        return "2"

    @property
    def PlayerBlockingActor(self):
        if "PlayerBlockingActor" in self._entity_data:
            return self._entity_data.get('PlayerBlockingActor')
        return "2"

    @property
    def ActorInPVS(self):
        if "ActorInPVS" in self._entity_data:
            return self._entity_data.get('ActorInPVS')
        return "2"

    @property
    def ActorInVehicle(self):
        if "ActorInVehicle" in self._entity_data:
            return self._entity_data.get('ActorInVehicle')
        return "2"

    @property
    def PlayerInVehicle(self):
        if "PlayerInVehicle" in self._entity_data:
            return self._entity_data.get('PlayerInVehicle')
        return "2"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Fire outputs with the Actor as Activator': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class scripted_sequence(BaseScripted, DXLevelChoice):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Repeatable': (4, 0), 'Leave Corpse (if not, fade)': (8, 0),
                                   'Start on Spawn': (16, 0), 'No Interruptions': (32, 0), 'Override AI': (64, 0),
                                   "Don't Teleport NPC On End": (128, 0), 'Loop in Post Idle': (256, 0),
                                   'Priority Script': (512, 0), 'Hide Debug Complaints': (2048, 0),
                                   "Allow other NPC actors to continue after this NPC actor's death": (
                                   4096, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def onplayerdeath(self):
        if "onplayerdeath" in self._entity_data:
            return self._entity_data.get('onplayerdeath')
        return "0"

    @property
    def onnpcdeath(self):
        if "onnpcdeath" in self._entity_data:
            return self._entity_data.get('onnpcdeath')
        return "0"

    @property
    def prevent_update_yaw_on_finish(self):
        if "prevent_update_yaw_on_finish" in self._entity_data:
            return self._entity_data.get('prevent_update_yaw_on_finish')
        return "0"

    @property
    def ensure_on_navmesh_on_finish(self):
        if "ensure_on_navmesh_on_finish" in self._entity_data:
            return self._entity_data.get('ensure_on_navmesh_on_finish')
        return "1"


class aiscripted_schedule(Targetname):
    pass

    icon_sprite =  "editor/aiscripted_schedule"

    @property
    def m_iszEntity(self):
        if "m_iszEntity" in self._entity_data:
            return self._entity_data.get('m_iszEntity')
        return None

    @property
    def m_flRadius(self):
        if "m_flRadius" in self._entity_data:
            return int(self._entity_data.get('m_flRadius'))
        return int(0)

    @property
    def graball(self):
        if "graball" in self._entity_data:
            return bool(self._entity_data.get('graball'))
        return bool(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Repeatable': (4, 1), 'Search Cyclically': (1024, 0),
                                   "Don't Complain": (2048, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def forcestate(self):
        if "forcestate" in self._entity_data:
            return self._entity_data.get('forcestate')
        return "0"

    @property
    def schedule(self):
        if "schedule" in self._entity_data:
            return self._entity_data.get('schedule')
        return "1"

    @property
    def interruptability(self):
        if "interruptability" in self._entity_data:
            return self._entity_data.get('interruptability')
        return "0"

    @property
    def resilient(self):
        if "resilient" in self._entity_data:
            return bool(self._entity_data.get('resilient'))
        return bool(0)

    @property
    def goalent(self):
        if "goalent" in self._entity_data:
            return self._entity_data.get('goalent')
        return None


class logic_choreographed_scene(Targetname):
    pass

    icon_sprite =  "editor/choreo_scene.vmat"

    @property
    def SceneFile(self):
        if "SceneFile" in self._entity_data:
            return self._entity_data.get('SceneFile')
        return None

    @property
    def target1(self):
        if "target1" in self._entity_data:
            return self._entity_data.get('target1')
        return None

    @property
    def target2(self):
        if "target2" in self._entity_data:
            return self._entity_data.get('target2')
        return None

    @property
    def target3(self):
        if "target3" in self._entity_data:
            return self._entity_data.get('target3')
        return None

    @property
    def target4(self):
        if "target4" in self._entity_data:
            return self._entity_data.get('target4')
        return None

    @property
    def target5(self):
        if "target5" in self._entity_data:
            return self._entity_data.get('target5')
        return None

    @property
    def target6(self):
        if "target6" in self._entity_data:
            return self._entity_data.get('target6')
        return None

    @property
    def target7(self):
        if "target7" in self._entity_data:
            return self._entity_data.get('target7')
        return None

    @property
    def target8(self):
        if "target8" in self._entity_data:
            return self._entity_data.get('target8')
        return None

    @property
    def busyactor(self):
        if "busyactor" in self._entity_data:
            return self._entity_data.get('busyactor')
        return "1"

    @property
    def onplayerdeath(self):
        if "onplayerdeath" in self._entity_data:
            return self._entity_data.get('onplayerdeath')
        return "0"


class logic_scene_list_manager(Targetname):
    pass

    icon_sprite =  "editor/choreo_manager.vmat"

    @property
    def scene0(self):
        if "scene0" in self._entity_data:
            return self._entity_data.get('scene0')
        return ""

    @property
    def scene1(self):
        if "scene1" in self._entity_data:
            return self._entity_data.get('scene1')
        return ""

    @property
    def scene2(self):
        if "scene2" in self._entity_data:
            return self._entity_data.get('scene2')
        return ""

    @property
    def scene3(self):
        if "scene3" in self._entity_data:
            return self._entity_data.get('scene3')
        return ""

    @property
    def scene4(self):
        if "scene4" in self._entity_data:
            return self._entity_data.get('scene4')
        return ""

    @property
    def scene5(self):
        if "scene5" in self._entity_data:
            return self._entity_data.get('scene5')
        return ""

    @property
    def scene6(self):
        if "scene6" in self._entity_data:
            return self._entity_data.get('scene6')
        return ""

    @property
    def scene7(self):
        if "scene7" in self._entity_data:
            return self._entity_data.get('scene7')
        return ""

    @property
    def scene8(self):
        if "scene8" in self._entity_data:
            return self._entity_data.get('scene8')
        return ""

    @property
    def scene9(self):
        if "scene9" in self._entity_data:
            return self._entity_data.get('scene9')
        return ""

    @property
    def scene10(self):
        if "scene10" in self._entity_data:
            return self._entity_data.get('scene10')
        return ""

    @property
    def scene11(self):
        if "scene11" in self._entity_data:
            return self._entity_data.get('scene11')
        return ""

    @property
    def scene12(self):
        if "scene12" in self._entity_data:
            return self._entity_data.get('scene12')
        return ""

    @property
    def scene13(self):
        if "scene13" in self._entity_data:
            return self._entity_data.get('scene13')
        return ""

    @property
    def scene14(self):
        if "scene14" in self._entity_data:
            return self._entity_data.get('scene14')
        return ""

    @property
    def scene15(self):
        if "scene15" in self._entity_data:
            return self._entity_data.get('scene15')
        return ""


class ai_sound(Targetname, Parentname):
    pass

    icon_sprite =  "editor/ai_sound.vmat"

    @property
    def volume(self):
        if "volume" in self._entity_data:
            return int(self._entity_data.get('volume'))
        return int(120)

    @property
    def duration(self):
        if "duration" in self._entity_data:
            return float(self._entity_data.get('duration'))
        return float(0.5)

    @property
    def soundtype(self):
        if "soundtype" in self._entity_data:
            return self._entity_data.get('soundtype')
        return "0"

    @property
    def soundcontext(self):
        if "soundcontext" in self._entity_data:
            return self._entity_data.get('soundcontext')
        return "0"

    @property
    def locationproxy(self):
        if "locationproxy" in self._entity_data:
            return self._entity_data.get('locationproxy')
        return ""


class ai_attached_item_manager(Targetname):
    pass

    icon_sprite =  "editor/ai_attached_item_manager.vmat"

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return ""

    @property
    def num_attached_items(self):
        if "num_attached_items" in self._entity_data:
            return int(self._entity_data.get('num_attached_items'))
        return int(0)

    @property
    def item_1(self):
        if "item_1" in self._entity_data:
            return self._entity_data.get('item_1')
        return ""

    @property
    def item_2(self):
        if "item_2" in self._entity_data:
            return self._entity_data.get('item_2')
        return ""

    @property
    def item_3(self):
        if "item_3" in self._entity_data:
            return self._entity_data.get('item_3')
        return ""

    @property
    def item_4(self):
        if "item_4" in self._entity_data:
            return self._entity_data.get('item_4')
        return ""

    @property
    def listen_entityspawns(self):
        if "listen_entityspawns" in self._entity_data:
            return bool(self._entity_data.get('listen_entityspawns'))
        return bool(0)

    @property
    def mark_as_removable(self):
        if "mark_as_removable" in self._entity_data:
            return self._entity_data.get('mark_as_removable')
        return "0"


class ai_addon(Targetname):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None


class ai_addon_builder(Targetname, EnableDisable):
    pass

    @property
    def NPCName(self):
        if "NPCName" in self._entity_data:
            return self._entity_data.get('NPCName')
        return ""

    @property
    def AddOnName(self):
        if "AddOnName" in self._entity_data:
            return self._entity_data.get('AddOnName')
        return ""

    @property
    def NpcPoints(self):
        if "NpcPoints" in self._entity_data:
            return int(self._entity_data.get('NpcPoints'))
        return int(10)

    @property
    def AddonPoints(self):
        if "AddonPoints" in self._entity_data:
            return int(self._entity_data.get('AddonPoints'))
        return int(10)


class AlyxInteractable:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data


class CombineBallSpawners(Targetname, Global):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start inactive': (4096, 1), 'Combine power supply': (8192, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def ballcount(self):
        if "ballcount" in self._entity_data:
            return int(self._entity_data.get('ballcount'))
        return int(3)

    @property
    def minspeed(self):
        if "minspeed" in self._entity_data:
            return float(self._entity_data.get('minspeed'))
        return float(300.0)

    @property
    def maxspeed(self):
        if "maxspeed" in self._entity_data:
            return float(self._entity_data.get('maxspeed'))
        return float(600.0)

    @property
    def ballradius(self):
        if "ballradius" in self._entity_data:
            return float(self._entity_data.get('ballradius'))
        return float(20.0)

    @property
    def balltype(self):
        if "balltype" in self._entity_data:
            return self._entity_data.get('balltype')
        return "Combine Energy Ball 1"

    @property
    def ballrespawntime(self):
        if "ballrespawntime" in self._entity_data:
            return float(self._entity_data.get('ballrespawntime'))
        return float(4.0)


class prop_combine_ball(BasePropPhysics):
    pass


class trigger_physics_trap(Trigger):
    pass

    @property
    def dissolvetype(self):
        if "dissolvetype" in self._entity_data:
            return self._entity_data.get('dissolvetype')
        return "Energy"


class trigger_weapon_dissolve(Trigger):
    pass

    @property
    def emittername(self):
        if "emittername" in self._entity_data:
            return self._entity_data.get('emittername')
        return ""


class trigger_weapon_strip(Trigger):
    pass

    @property
    def KillWeapons(self):
        if "KillWeapons" in self._entity_data:
            return bool(self._entity_data.get('KillWeapons'))
        return bool(False)


class func_combine_ball_spawner(CombineBallSpawners):
    pass


class point_combine_ball_launcher(CombineBallSpawners):
    pass

    @property
    def launchconenoise(self):
        if "launchconenoise" in self._entity_data:
            return float(self._entity_data.get('launchconenoise'))
        return float(0.0)

    @property
    def bullseyename(self):
        if "bullseyename" in self._entity_data:
            return self._entity_data.get('bullseyename')
        return ""

    @property
    def maxballbounces(self):
        if "maxballbounces" in self._entity_data:
            return int(self._entity_data.get('maxballbounces'))
        return int(8)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Attach Bullseye': (1, 0), 'Balls should collide against player': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class npc_grenade_frag(BaseNPC):
    pass


class npc_combine_cannon(BaseNPC):
    pass

    @property
    def sightdist(self):
        if "sightdist" in self._entity_data:
            return float(self._entity_data.get('sightdist'))
        return float(1024)


class npc_combine_camera(BaseNPC):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Always Become Angry On New Enemy': (32, 1),
                                   'Ignore Enemies (Scripted Targets Only)': (64, 0),
                                   'Start Inactive': (128, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def innerradius(self):
        if "innerradius" in self._entity_data:
            return int(self._entity_data.get('innerradius'))
        return int(300)

    @property
    def outerradius(self):
        if "outerradius" in self._entity_data:
            return int(self._entity_data.get('outerradius'))
        return int(450)

    @property
    def minhealthdmg(self):
        if "minhealthdmg" in self._entity_data:
            return int(self._entity_data.get('minhealthdmg'))
        return int(0)

    @property
    def defaulttarget(self):
        if "defaulttarget" in self._entity_data:
            return self._entity_data.get('defaulttarget')
        return ""


class npc_turret_ground(BaseNPC, Parentname, AlyxInteractable):
    pass


class npc_turret_ceiling(Targetname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Autostart': (32, 1), 'Start Inactive': (64, 0), 'Never Retire': (128, 0),
                                   'Out of Ammo': (256, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def health(self):
        if "health" in self._entity_data:
            return int(self._entity_data.get('health'))
        return int(1000)

    @property
    def minhealthdmg(self):
        if "minhealthdmg" in self._entity_data:
            return int(self._entity_data.get('minhealthdmg'))
        return int(0)

    @property
    def scanwhenidle(self):
        if "scanwhenidle" in self._entity_data:
            return bool(self._entity_data.get('scanwhenidle'))
        return bool(False)

    @property
    def scanspeed(self):
        if "scanspeed" in self._entity_data:
            return float(self._entity_data.get('scanspeed'))
        return float(2.0)

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/combine_turrets/ceiling_turret.vmdl"

    @property
    def gun_model(self):
        if "gun_model" in self._entity_data:
            return self._entity_data.get('gun_model')
        return ""

    @property
    def targeting_entity_name(self):
        if "targeting_entity_name" in self._entity_data:
            return self._entity_data.get('targeting_entity_name')
        return ""


class npc_turret_floor(Targetname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Autostart': (32, 0), 'Start Inactive': (64, 0), 'Fast Retire': (128, 0),
                                   'Out of Ammo': (256, 0), 'Citizen modified (Friendly)': (512, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def KeepUpright(self):
        if "KeepUpright" in self._entity_data:
            return bool(self._entity_data.get('KeepUpright'))
        return bool(1)

    @property
    def SkinNumber(self):
        if "SkinNumber" in self._entity_data:
            return int(self._entity_data.get('SkinNumber'))
        return int(0)


class npc_cranedriver(VehicleDriverNPC):
    pass

    @property
    def releasepause(self):
        if "releasepause" in self._entity_data:
            return float(self._entity_data.get('releasepause'))
        return float(0)


class npc_apcdriver(VehicleDriverNPC):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'No Rocket Attacks': (65536, 0), 'No Gun Attacks': (131072, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def drivermaxspeed(self):
        if "drivermaxspeed" in self._entity_data:
            return float(self._entity_data.get('drivermaxspeed'))
        return float(1)

    @property
    def driverminspeed(self):
        if "driverminspeed" in self._entity_data:
            return float(self._entity_data.get('driverminspeed'))
        return float(0)


class npc_rollermine(BaseNPC, AlyxInteractable):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Friendly': (65536, 0), 'Use prop_physics collision rules': (131072, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def startburied(self):
        if "startburied" in self._entity_data:
            return bool(self._entity_data.get('startburied'))
        return bool(False)

    @property
    def uniformsightdist(self):
        if "uniformsightdist" in self._entity_data:
            return self._entity_data.get('uniformsightdist')
        return "0"


class npc_missiledefense(BaseNPC):
    pass


class npc_sniper(BaseNPC):
    pass

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return int(self._entity_data.get('radius'))
        return int(0)

    @property
    def misses(self):
        if "misses" in self._entity_data:
            return int(self._entity_data.get('misses'))
        return int(0)

    @property
    def beambrightness(self):
        if "beambrightness" in self._entity_data:
            return int(self._entity_data.get('beambrightness'))
        return int(100)

    @property
    def shootZombiesInChest(self):
        if "shootZombiesInChest" in self._entity_data:
            return bool(self._entity_data.get('shootZombiesInChest'))
        return bool(0)

    @property
    def shielddistance(self):
        if "shielddistance" in self._entity_data:
            return float(self._entity_data.get('shielddistance'))
        return float(64)

    @property
    def shieldradius(self):
        if "shieldradius" in self._entity_data:
            return float(self._entity_data.get('shieldradius'))
        return float(48)

    @property
    def PaintInterval(self):
        if "PaintInterval" in self._entity_data:
            return float(self._entity_data.get('PaintInterval'))
        return float(1)

    @property
    def PaintIntervalVariance(self):
        if "PaintIntervalVariance" in self._entity_data:
            return float(self._entity_data.get('PaintIntervalVariance'))
        return float(0.75)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Hidden': (65536, 0), 'Laser Viewcone': (131072, 0), 'No Corpse': (262144, 0),
                                   'Start Disabled': (524288, 0), 'Faster shooting (Episodic)': (1048576, 0),
                                   'No sweep away from target (Episodic)': (2097152, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class info_radar_target(Targetname, Parentname, EnableDisable):
    pass

    icon_sprite =  "editor/info_target.vmat"

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return float(self._entity_data.get('radius'))
        return float(6000)

    @property
    def type(self):
        if "type" in self._entity_data:
            return self._entity_data.get('type')
        return "0"

    @property
    def mode(self):
        if "mode" in self._entity_data:
            return self._entity_data.get('mode')
        return "0"


class info_target_vehicle_transition(Targetname, EnableDisable):
    pass

    icon_sprite =  "editor/info_target.vmat"


class info_snipertarget(Targetname, Parentname):
    pass

    icon_sprite =  "editor/info_target.vmat"

    @property
    def speed(self):
        if "speed" in self._entity_data:
            return int(self._entity_data.get('speed'))
        return int(2)

    @property
    def groupname(self):
        if "groupname" in self._entity_data:
            return self._entity_data.get('groupname')
        return None

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Shoot Me': (1, 0), 'No Interruptions': (2, 0), 'Resume if Interrupted': (8, 0),
                                   'Snap to me': (16, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class prop_thumper(Targetname):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/props_combine/CombineThumper002.vmdl"

    @property
    def dustscale(self):
        if "dustscale" in self._entity_data:
            return self._entity_data.get('dustscale')
        return "Small Thumper"

    @property
    def EffectRadius(self):
        if "EffectRadius" in self._entity_data:
            return int(self._entity_data.get('EffectRadius'))
        return int(1000)


class npc_antlion(BaseNPC):
    pass

    @property
    def startburrowed(self):
        if "startburrowed" in self._entity_data:
            return bool(self._entity_data.get('startburrowed'))
        return bool(False)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Burrow when eluded': (65536, 0), 'Use Ground Checks': (131072, 0),
                                   'Worker Type': (262144, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return int(self._entity_data.get('radius'))
        return int(256)

    @property
    def eludedist(self):
        if "eludedist" in self._entity_data:
            return int(self._entity_data.get('eludedist'))
        return int(1024)

    @property
    def ignorebugbait(self):
        if "ignorebugbait" in self._entity_data:
            return bool(self._entity_data.get('ignorebugbait'))
        return bool(False)

    @property
    def unburroweffects(self):
        if "unburroweffects" in self._entity_data:
            return bool(self._entity_data.get('unburroweffects'))
        return bool(False)


class npc_antlionguard(BaseNPC):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Create server-side ragdoll on death': (65536, 0),
                                   'Use inside footsteps': (131072, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def startburrowed(self):
        if "startburrowed" in self._entity_data:
            return bool(self._entity_data.get('startburrowed'))
        return bool(False)

    @property
    def allowbark(self):
        if "allowbark" in self._entity_data:
            return bool(self._entity_data.get('allowbark'))
        return bool(False)

    @property
    def cavernbreed(self):
        if "cavernbreed" in self._entity_data:
            return bool(self._entity_data.get('cavernbreed'))
        return bool(False)

    @property
    def incavern(self):
        if "incavern" in self._entity_data:
            return bool(self._entity_data.get('incavern'))
        return bool(False)

    @property
    def shovetargets(self):
        if "shovetargets" in self._entity_data:
            return self._entity_data.get('shovetargets')
        return ""


class BaseBird(BaseNPC):
    pass

    @property
    def deaf(self):
        if "deaf" in self._entity_data:
            return self._entity_data.get('deaf')
        return "0"


class npc_crow(BaseBird):
    pass


class npc_seagull(BaseBird):
    pass


class npc_pigeon(BaseBird):
    pass


class npc_ichthyosaur(BaseNPC):
    pass


class BaseHeadcrab(BaseNPC):
    pass

    @property
    def startburrowed(self):
        if "startburrowed" in self._entity_data:
            return bool(self._entity_data.get('startburrowed'))
        return bool(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start hidden': (65536, 0), 'Start hanging from ceiling': (131072, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class npc_headcrab(BaseHeadcrab, Parentname):
    pass


class npc_headcrab_fast(BaseHeadcrab):
    pass


class npc_headcrab_black(BaseHeadcrab):
    pass


class npc_stalker(BaseNPC):
    pass

    @property
    def BeamPower(self):
        if "BeamPower" in self._entity_data:
            return self._entity_data.get('BeamPower')
        return "Low"


class npc_enemyfinder_combinecannon(Parentname, BaseNPC):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Check Visibility': (65536, 1), 'APC Visibility checks': (131072, 0),
                                   'Short memory': (262144, 0), 'Can be an enemy': (524288, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def FieldOfView(self):
        if "FieldOfView" in self._entity_data:
            return self._entity_data.get('FieldOfView')
        return "0.2"

    @property
    def MinSearchDist(self):
        if "MinSearchDist" in self._entity_data:
            return int(self._entity_data.get('MinSearchDist'))
        return int(0)

    @property
    def MaxSearchDist(self):
        if "MaxSearchDist" in self._entity_data:
            return int(self._entity_data.get('MaxSearchDist'))
        return int(2048)

    @property
    def SnapToEnt(self):
        if "SnapToEnt" in self._entity_data:
            return self._entity_data.get('SnapToEnt')
        return ""

    @property
    def freepass_timetotrigger(self):
        if "freepass_timetotrigger" in self._entity_data:
            return float(self._entity_data.get('freepass_timetotrigger'))
        return float(0)

    @property
    def freepass_duration(self):
        if "freepass_duration" in self._entity_data:
            return float(self._entity_data.get('freepass_duration'))
        return float(0)

    @property
    def freepass_movetolerance(self):
        if "freepass_movetolerance" in self._entity_data:
            return float(self._entity_data.get('freepass_movetolerance'))
        return float(120)

    @property
    def freepass_refillrate(self):
        if "freepass_refillrate" in self._entity_data:
            return float(self._entity_data.get('freepass_refillrate'))
        return float(0.5)

    @property
    def freepass_peektime(self):
        if "freepass_peektime" in self._entity_data:
            return float(self._entity_data.get('freepass_peektime'))
        return float(0)

    @property
    def StartOn(self):
        if "StartOn" in self._entity_data:
            return bool(self._entity_data.get('StartOn'))
        return bool(1)


class npc_citizen(Parentname, TalkNPC, PlayerCompanion):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Follow player on spawn': (65536, 0), 'Medic': (131072, 0),
                                   'Random Head': (262144, 1), 'Ammo Resupplier': (524288, 0),
                                   'Not Commandable': (1048576, 0),
                                   "Don't use Speech Semaphore - OBSOLETE": (2097152, 0),
                                   'Random male head': (4194304, 0), 'Random female head': (8388608, 0),
                                   'Use RenderBox in ActBusies': (16777216, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def additionalequipment(self):
        if "additionalequipment" in self._entity_data:
            return self._entity_data.get('additionalequipment')
        return "0"

    @property
    def ammosupply(self):
        if "ammosupply" in self._entity_data:
            return self._entity_data.get('ammosupply')
        return "SMG1"

    @property
    def ammoamount(self):
        if "ammoamount" in self._entity_data:
            return int(self._entity_data.get('ammoamount'))
        return int(1)

    @property
    def citizentype(self):
        if "citizentype" in self._entity_data:
            return self._entity_data.get('citizentype')
        return "0"

    @property
    def expressiontype(self):
        if "expressiontype" in self._entity_data:
            return self._entity_data.get('expressiontype')
        return "0"

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/humans/group01/male_01.vmdl"

    @property
    def ExpressionOverride(self):
        if "ExpressionOverride" in self._entity_data:
            return self._entity_data.get('ExpressionOverride')
        return None

    @property
    def notifynavfailblocked(self):
        if "notifynavfailblocked" in self._entity_data:
            return bool(self._entity_data.get('notifynavfailblocked'))
        return bool(0)

    @property
    def neverleaveplayersquad(self):
        if "neverleaveplayersquad" in self._entity_data:
            return self._entity_data.get('neverleaveplayersquad')
        return "0"

    @property
    def denycommandconcept(self):
        if "denycommandconcept" in self._entity_data:
            return self._entity_data.get('denycommandconcept')
        return ""

    @property
    def is_quest_member(self):
        if "is_quest_member" in self._entity_data:
            return bool(self._entity_data.get('is_quest_member'))
        return bool(0)


class npc_fisherman(BaseNPC):
    pass

    @property
    def ExpressionOverride(self):
        if "ExpressionOverride" in self._entity_data:
            return self._entity_data.get('ExpressionOverride')
        return None


class npc_barney(TalkNPC, PlayerCompanion):
    pass

    @property
    def additionalequipment(self):
        if "additionalequipment" in self._entity_data:
            return self._entity_data.get('additionalequipment')
        return "weapon_pistol"

    @property
    def ExpressionOverride(self):
        if "ExpressionOverride" in self._entity_data:
            return self._entity_data.get('ExpressionOverride')
        return None


class BaseCombine(TalkNPC, RappelNPC):
    pass

    @property
    def additionalequipment(self):
        if "additionalequipment" in self._entity_data:
            return self._entity_data.get('additionalequipment')
        return "default"

    @property
    def min_advance_range_override(self):
        if "min_advance_range_override" in self._entity_data:
            return float(self._entity_data.get('min_advance_range_override'))
        return float(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start LookOff': (65536, 0), "Don't drop grenades": (131072, 0),
                                   "Don't drop ar2 alt fire (elite only) ": (262144, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def TeleportGrenades(self):
        if "TeleportGrenades" in self._entity_data:
            return self._entity_data.get('TeleportGrenades')
        return "0"


class npc_combine_s(BaseCombine):
    pass

    @property
    def model_state(self):
        if "model_state" in self._entity_data:
            return self._entity_data.get('model_state')
        return ""

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/characters/combine_grunt/combine_grunt.vmdl"

    @property
    def tacticalvariant(self):
        if "tacticalvariant" in self._entity_data:
            return self._entity_data.get('tacticalvariant')
        return "0"

    @property
    def usemarch(self):
        if "usemarch" in self._entity_data:
            return self._entity_data.get('usemarch')
        return "0"

    @property
    def is_medic(self):
        if "is_medic" in self._entity_data:
            return self._entity_data.get('is_medic')
        return "0"

    @property
    def prevent_grenade_explosive_removal(self):
        if "prevent_grenade_explosive_removal" in self._entity_data:
            return self._entity_data.get('prevent_grenade_explosive_removal')
        return "0"

    @property
    def officer_reinforcements(self):
        if "officer_reinforcements" in self._entity_data:
            return int(self._entity_data.get('officer_reinforcements'))
        return int(0)

    @property
    def use_VRstealth_outside_combat(self):
        if "use_VRstealth_outside_combat" in self._entity_data:
            return bool(self._entity_data.get('use_VRstealth_outside_combat'))
        return bool(0)

    @property
    def grenade_proclivity(self):
        if "grenade_proclivity" in self._entity_data:
            return float(self._entity_data.get('grenade_proclivity'))
        return float(1)

    @property
    def manhack_proclivity(self):
        if "manhack_proclivity" in self._entity_data:
            return float(self._entity_data.get('manhack_proclivity'))
        return float(1)

    @property
    def initial_manhack_delay(self):
        if "initial_manhack_delay" in self._entity_data:
            return float(self._entity_data.get('initial_manhack_delay'))
        return float(20)

    @property
    def sentry_position_name(self):
        if "sentry_position_name" in self._entity_data:
            return self._entity_data.get('sentry_position_name')
        return ""


class npc_launcher(Parentname, BaseNPC):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Check LOS': (65536, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def StartOn(self):
        if "StartOn" in self._entity_data:
            return self._entity_data.get('StartOn')
        return "0"

    @property
    def MissileModel(self):
        if "MissileModel" in self._entity_data:
            return self._entity_data.get('MissileModel')
        return "models/Weapons/wscanner_grenade.vmdl"

    @property
    def LaunchSound(self):
        if "LaunchSound" in self._entity_data:
            return self._entity_data.get('LaunchSound')
        return "npc/waste_scanner/grenade_fire.wav"

    @property
    def FlySound(self):
        if "FlySound" in self._entity_data:
            return self._entity_data.get('FlySound')
        return "ambient/objects/machine2.wav"

    @property
    def SmokeTrail(self):
        if "SmokeTrail" in self._entity_data:
            return self._entity_data.get('SmokeTrail')
        return "1"

    @property
    def LaunchSmoke(self):
        if "LaunchSmoke" in self._entity_data:
            return self._entity_data.get('LaunchSmoke')
        return "1"

    @property
    def LaunchDelay(self):
        if "LaunchDelay" in self._entity_data:
            return int(self._entity_data.get('LaunchDelay'))
        return int(8)

    @property
    def LaunchSpeed(self):
        if "LaunchSpeed" in self._entity_data:
            return self._entity_data.get('LaunchSpeed')
        return "200"

    @property
    def PathCornerName(self):
        if "PathCornerName" in self._entity_data:
            return self._entity_data.get('PathCornerName')
        return ""

    @property
    def HomingSpeed(self):
        if "HomingSpeed" in self._entity_data:
            return self._entity_data.get('HomingSpeed')
        return "0"

    @property
    def HomingStrength(self):
        if "HomingStrength" in self._entity_data:
            return int(self._entity_data.get('HomingStrength'))
        return int(10)

    @property
    def HomingDelay(self):
        if "HomingDelay" in self._entity_data:
            return self._entity_data.get('HomingDelay')
        return "0"

    @property
    def HomingRampUp(self):
        if "HomingRampUp" in self._entity_data:
            return self._entity_data.get('HomingRampUp')
        return "0.5"

    @property
    def HomingDuration(self):
        if "HomingDuration" in self._entity_data:
            return self._entity_data.get('HomingDuration')
        return "5"

    @property
    def HomingRampDown(self):
        if "HomingRampDown" in self._entity_data:
            return self._entity_data.get('HomingRampDown')
        return "1.0"

    @property
    def Gravity(self):
        if "Gravity" in self._entity_data:
            return self._entity_data.get('Gravity')
        return "1.0"

    @property
    def MinRange(self):
        if "MinRange" in self._entity_data:
            return int(self._entity_data.get('MinRange'))
        return int(100)

    @property
    def MaxRange(self):
        if "MaxRange" in self._entity_data:
            return int(self._entity_data.get('MaxRange'))
        return int(2048)

    @property
    def SpinMagnitude(self):
        if "SpinMagnitude" in self._entity_data:
            return self._entity_data.get('SpinMagnitude')
        return "0"

    @property
    def SpinSpeed(self):
        if "SpinSpeed" in self._entity_data:
            return self._entity_data.get('SpinSpeed')
        return "0"

    @property
    def Damage(self):
        if "Damage" in self._entity_data:
            return self._entity_data.get('Damage')
        return "50"

    @property
    def DamageRadius(self):
        if "DamageRadius" in self._entity_data:
            return self._entity_data.get('DamageRadius')
        return "200"


class npc_hunter(BaseNPC):
    pass

    @property
    def FollowTarget(self):
        if "FollowTarget" in self._entity_data:
            return self._entity_data.get('FollowTarget')
        return ""


class npc_hunter_maker(npc_template_maker):
    pass

    icon_sprite =  "editor/npc_maker.vmat"


class npc_advisor(BaseNPC):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/advisor.vmdl"

    @property
    def levitationarea(self):
        if "levitationarea" in self._entity_data:
            return self._entity_data.get('levitationarea')
        return ""

    @property
    def levitategoal_bottom(self):
        if "levitategoal_bottom" in self._entity_data:
            return self._entity_data.get('levitategoal_bottom')
        return ""

    @property
    def levitategoal_top(self):
        if "levitategoal_top" in self._entity_data:
            return self._entity_data.get('levitategoal_top')
        return ""

    @property
    def staging_ent_names(self):
        if "staging_ent_names" in self._entity_data:
            return self._entity_data.get('staging_ent_names')
        return ""

    @property
    def priority_grab_name(self):
        if "priority_grab_name" in self._entity_data:
            return self._entity_data.get('priority_grab_name')
        return ""


class env_sporeexplosion(Targetname, Parentname, EnableDisable):
    pass

    @property
    def spawnrate(self):
        if "spawnrate" in self._entity_data:
            return float(self._entity_data.get('spawnrate'))
        return float(25)


class env_gunfire(Targetname, Parentname, EnableDisable):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return ""

    @property
    def minburstsize(self):
        if "minburstsize" in self._entity_data:
            return int(self._entity_data.get('minburstsize'))
        return int(2)

    @property
    def maxburstsize(self):
        if "maxburstsize" in self._entity_data:
            return int(self._entity_data.get('maxburstsize'))
        return int(7)

    @property
    def minburstdelay(self):
        if "minburstdelay" in self._entity_data:
            return float(self._entity_data.get('minburstdelay'))
        return float(2)

    @property
    def maxburstdelay(self):
        if "maxburstdelay" in self._entity_data:
            return float(self._entity_data.get('maxburstdelay'))
        return float(5)

    @property
    def rateoffire(self):
        if "rateoffire" in self._entity_data:
            return float(self._entity_data.get('rateoffire'))
        return float(10)

    @property
    def spread(self):
        if "spread" in self._entity_data:
            return self._entity_data.get('spread')
        return "5"

    @property
    def bias(self):
        if "bias" in self._entity_data:
            return self._entity_data.get('bias')
        return "1"

    @property
    def collisions(self):
        if "collisions" in self._entity_data:
            return self._entity_data.get('collisions')
        return "0"

    @property
    def shootsound(self):
        if "shootsound" in self._entity_data:
            return self._entity_data.get('shootsound')
        return "Weapon_AR2.NPC_Single"

    @property
    def tracertype(self):
        if "tracertype" in self._entity_data:
            return self._entity_data.get('tracertype')
        return "AR2TRACER"


class env_headcrabcanister(Parentname, Targetname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'No Impact Sound': (1, 0), 'No Launch Sound': (2, 0), 'Start Impacted': (4096, 0),
                                   'Land at initial position': (8192, 0), 'Wait for input to open': (16384, 0),
                                   'Wait for input to spawn headcrabs': (32768, 0), 'No smoke': (65536, 0),
                                   'No shake': (131072, 0), 'Remove on impact': (262144, 0),
                                   'No impact effects': (524288, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def HeadcrabType(self):
        if "HeadcrabType" in self._entity_data:
            return self._entity_data.get('HeadcrabType')
        return "0"

    @property
    def HeadcrabCount(self):
        if "HeadcrabCount" in self._entity_data:
            return int(self._entity_data.get('HeadcrabCount'))
        return int(6)

    @property
    def FlightSpeed(self):
        if "FlightSpeed" in self._entity_data:
            return float(self._entity_data.get('FlightSpeed'))
        return float(3000)

    @property
    def FlightTime(self):
        if "FlightTime" in self._entity_data:
            return float(self._entity_data.get('FlightTime'))
        return float(5)

    @property
    def StartingHeight(self):
        if "StartingHeight" in self._entity_data:
            return float(self._entity_data.get('StartingHeight'))
        return float(0)

    @property
    def MinSkyboxRefireTime(self):
        if "MinSkyboxRefireTime" in self._entity_data:
            return float(self._entity_data.get('MinSkyboxRefireTime'))
        return float(0)

    @property
    def MaxSkyboxRefireTime(self):
        if "MaxSkyboxRefireTime" in self._entity_data:
            return float(self._entity_data.get('MaxSkyboxRefireTime'))
        return float(0)

    @property
    def SkyboxCannisterCount(self):
        if "SkyboxCannisterCount" in self._entity_data:
            return int(self._entity_data.get('SkyboxCannisterCount'))
        return int(1)

    @property
    def Damage(self):
        if "Damage" in self._entity_data:
            return float(self._entity_data.get('Damage'))
        return float(150)

    @property
    def DamageRadius(self):
        if "DamageRadius" in self._entity_data:
            return float(self._entity_data.get('DamageRadius'))
        return float(750)

    @property
    def SmokeLifetime(self):
        if "SmokeLifetime" in self._entity_data:
            return float(self._entity_data.get('SmokeLifetime'))
        return float(30)

    @property
    def LaunchPositionName(self):
        if "LaunchPositionName" in self._entity_data:
            return self._entity_data.get('LaunchPositionName')
        return ""


class npc_vortigaunt(TalkNPC, PlayerCompanion):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/vortigaunt.vmdl"

    @property
    def ArmorRechargeEnabled(self):
        if "ArmorRechargeEnabled" in self._entity_data:
            return bool(self._entity_data.get('ArmorRechargeEnabled'))
        return bool(1)

    @property
    def HealthRegenerateEnabled(self):
        if "HealthRegenerateEnabled" in self._entity_data:
            return bool(self._entity_data.get('HealthRegenerateEnabled'))
        return bool(0)


class npc_spotlight(BaseNPC):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start Track On': (65536, 1), 'Start Light On': (131072, 1),
                                   'No Dynamic Light': (262144, 0), 'Never Move': (524288, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def health(self):
        if "health" in self._entity_data:
            return int(self._entity_data.get('health'))
        return int(100)

    @property
    def YawRange(self):
        if "YawRange" in self._entity_data:
            return int(self._entity_data.get('YawRange'))
        return int(90)

    @property
    def PitchMin(self):
        if "PitchMin" in self._entity_data:
            return int(self._entity_data.get('PitchMin'))
        return int(35)

    @property
    def PitchMax(self):
        if "PitchMax" in self._entity_data:
            return int(self._entity_data.get('PitchMax'))
        return int(50)

    @property
    def IdleSpeed(self):
        if "IdleSpeed" in self._entity_data:
            return int(self._entity_data.get('IdleSpeed'))
        return int(2)

    @property
    def AlertSpeed(self):
        if "AlertSpeed" in self._entity_data:
            return int(self._entity_data.get('AlertSpeed'))
        return int(5)

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


class npc_strider(BaseNPC):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Can Stomp Player': (65536, 0),
                                   'Minimal damage taken from NPCs (1 point per missile)': (131072, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def disablephysics(self):
        if "disablephysics" in self._entity_data:
            return bool(self._entity_data.get('disablephysics'))
        return bool(0)

    @property
    def strider_feet_not_dangerous(self):
        if "strider_feet_not_dangerous" in self._entity_data:
            return bool(self._entity_data.get('strider_feet_not_dangerous'))
        return bool(0)


class npc_barnacle(BaseNPC, BaseFadeProp):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Cheap death': (65536, 0), 'Ambush Mode': (131072, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def RestDist(self):
        if "RestDist" in self._entity_data:
            return float(self._entity_data.get('RestDist'))
        return float(16)

    @property
    def basepullspeed(self):
        if "basepullspeed" in self._entity_data:
            return float(self._entity_data.get('basepullspeed'))
        return float(0)

    @property
    def holdforever(self):
        if "holdforever" in self._entity_data:
            return self._entity_data.get('holdforever')
        return "0"

    @property
    def lightmapstatic(self):
        if "lightmapstatic" in self._entity_data:
            return self._entity_data.get('lightmapstatic')
        return "2"


class npc_combinegunship(BaseHelicopter):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'No ground attack': (4096, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def MaxAngAccel(self):
        if "MaxAngAccel" in self._entity_data:
            return float(self._entity_data.get('MaxAngAccel'))
        return float(1000)

    @property
    def MaxAngVelocity(self):
        if "MaxAngVelocity" in self._entity_data:
            return parse_int_vector(self._entity_data.get('MaxAngVelocity'))
        return parse_int_vector("300 120 300")


class info_target_helicopter_crash(Targetname, Parentname):
    pass

    icon_sprite =  "editor/info_target.vmat"


class info_target_gunshipcrash(Targetname, Parentname):
    pass

    icon_sprite =  "editor/info_target.vmat"


class npc_combinedropship(BaseHelicopter):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Wait for input before dropoff': (32768, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def LandTarget(self):
        if "LandTarget" in self._entity_data:
            return self._entity_data.get('LandTarget')
        return None

    @property
    def GunRange(self):
        if "GunRange" in self._entity_data:
            return float(self._entity_data.get('GunRange'))
        return float(2048)

    @property
    def RollermineTemplate(self):
        if "RollermineTemplate" in self._entity_data:
            return self._entity_data.get('RollermineTemplate')
        return ""

    @property
    def NPCTemplate1(self):
        if "NPCTemplate1" in self._entity_data:
            return self._entity_data.get('NPCTemplate1')
        return None

    @property
    def NPCTemplate2(self):
        if "NPCTemplate2" in self._entity_data:
            return self._entity_data.get('NPCTemplate2')
        return None

    @property
    def NPCTemplate3(self):
        if "NPCTemplate3" in self._entity_data:
            return self._entity_data.get('NPCTemplate3')
        return None

    @property
    def NPCTemplate4(self):
        if "NPCTemplate4" in self._entity_data:
            return self._entity_data.get('NPCTemplate4')
        return None

    @property
    def NPCTemplate5(self):
        if "NPCTemplate5" in self._entity_data:
            return self._entity_data.get('NPCTemplate5')
        return None

    @property
    def NPCTemplate6(self):
        if "NPCTemplate6" in self._entity_data:
            return self._entity_data.get('NPCTemplate6')
        return None

    @property
    def Dustoff1(self):
        if "Dustoff1" in self._entity_data:
            return self._entity_data.get('Dustoff1')
        return None

    @property
    def Dustoff2(self):
        if "Dustoff2" in self._entity_data:
            return self._entity_data.get('Dustoff2')
        return None

    @property
    def Dustoff3(self):
        if "Dustoff3" in self._entity_data:
            return self._entity_data.get('Dustoff3')
        return None

    @property
    def Dustoff4(self):
        if "Dustoff4" in self._entity_data:
            return self._entity_data.get('Dustoff4')
        return None

    @property
    def Dustoff5(self):
        if "Dustoff5" in self._entity_data:
            return self._entity_data.get('Dustoff5')
        return None

    @property
    def Dustoff6(self):
        if "Dustoff6" in self._entity_data:
            return self._entity_data.get('Dustoff6')
        return None

    @property
    def APCVehicleName(self):
        if "APCVehicleName" in self._entity_data:
            return self._entity_data.get('APCVehicleName')
        return None

    @property
    def Invulnerable(self):
        if "Invulnerable" in self._entity_data:
            return bool(self._entity_data.get('Invulnerable'))
        return bool(0)

    @property
    def CrateType(self):
        if "CrateType" in self._entity_data:
            return self._entity_data.get('CrateType')
        return "2"


class npc_helicopter(BaseHelicopter):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Loud rotor wash sound': (65536, 0), 'Electrical drone': (131072, 0),
                                   'Helicopter lights': (262144, 0), 'Ignore avoid spheres+boxes': (524288, 0),
                                   'More aggressive attacks': (1048576, 0), 'Cast long shadow': (2097152, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def InitialSpeed(self):
        if "InitialSpeed" in self._entity_data:
            return self._entity_data.get('InitialSpeed')
        return "0"

    @property
    def GracePeriod(self):
        if "GracePeriod" in self._entity_data:
            return float(self._entity_data.get('GracePeriod'))
        return float(2.0)

    @property
    def PatrolSpeed(self):
        if "PatrolSpeed" in self._entity_data:
            return float(self._entity_data.get('PatrolSpeed'))
        return float(0)

    @property
    def noncombat(self):
        if "noncombat" in self._entity_data:
            return bool(self._entity_data.get('noncombat'))
        return bool(False)


class grenade_helicopter(Targetname, Parentname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Is a dud': (65536, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class npc_heli_avoidsphere(Targetname, Parentname):
    pass

    icon_sprite =  "editor/env_firesource"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Avoid the sphere above and below': (65536, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return float(self._entity_data.get('radius'))
        return float(128)


class npc_heli_avoidbox(Parentname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Avoid the box above and below': (65536, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class npc_heli_nobomb(Parentname):
    pass


class info_target_advisor_roaming_crash(Targetname, Parentname):
    pass

    icon_sprite =  "editor/info_target.vmat"


class npc_combine_advisor_roaming(BaseHelicopter):
    pass


class BaseZombie:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Enable Teleport Blocker': (65536, 0),
                                   "Don't release headcrab upon death": (131072, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def revivable(self):
        if "revivable" in self._entity_data:
            return bool(self._entity_data.get('revivable'))
        return bool(0)

    @property
    def waitforrevival(self):
        if "waitforrevival" in self._entity_data:
            return bool(self._entity_data.get('waitforrevival'))
        return bool(0)


class npc_fastzombie(BaseNPC, BaseZombie):
    pass


class npc_fastzombie_torso(BaseNPC, BaseZombie):
    pass


class npc_zombie(BaseNPC, BaseZombie):
    pass

    @property
    def reviver_group(self):
        if "reviver_group" in self._entity_data:
            return self._entity_data.get('reviver_group')
        return ""


class npc_zombie_torso(BaseNPC, BaseZombie):
    pass


class point_zombie_noise_generator(Targetname, Parentname):
    pass

    @property
    def accumulate(self):
        if "accumulate" in self._entity_data:
            return bool(self._entity_data.get('accumulate'))
        return bool(0)

    @property
    def accumulatelevel(self):
        if "accumulatelevel" in self._entity_data:
            return int(self._entity_data.get('accumulatelevel'))
        return int(12)

    @property
    def suppresstime(self):
        if "suppresstime" in self._entity_data:
            return float(self._entity_data.get('suppresstime'))
        return float(5)

    @property
    def delay(self):
        if "delay" in self._entity_data:
            return float(self._entity_data.get('delay'))
        return float(2)


class npc_zombine(BaseNPC):
    pass


class npc_poisonzombie(BaseNPC):
    pass

    @property
    def crabcount(self):
        if "crabcount" in self._entity_data:
            return self._entity_data.get('crabcount')
        return "3"


class npc_cscanner(BaseNPC):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'No Dynamic Light': (65536, 0), 'Strider Scout Scanner': (131072, 0)}.items():
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
    def spotlightdisabled(self):
        if "spotlightdisabled" in self._entity_data:
            return bool(self._entity_data.get('spotlightdisabled'))
        return bool(0)

    @property
    def ShouldInspect(self):
        if "ShouldInspect" in self._entity_data:
            return bool(self._entity_data.get('ShouldInspect'))
        return bool(1)

    @property
    def OnlyInspectPlayers(self):
        if "OnlyInspectPlayers" in self._entity_data:
            return bool(self._entity_data.get('OnlyInspectPlayers'))
        return bool(0)

    @property
    def NeverInspectPlayers(self):
        if "NeverInspectPlayers" in self._entity_data:
            return bool(self._entity_data.get('NeverInspectPlayers'))
        return bool(0)

    @property
    def upclosemode(self):
        if "upclosemode" in self._entity_data:
            return bool(self._entity_data.get('upclosemode'))
        return bool(0)


class npc_clawscanner(BaseNPC):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'No Dynamic Light': (65536, 0), 'Strider Scout Scanner': (131072, 0)}.items():
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
    def spotlightdisabled(self):
        if "spotlightdisabled" in self._entity_data:
            return bool(self._entity_data.get('spotlightdisabled'))
        return bool(0)

    @property
    def ShouldInspect(self):
        if "ShouldInspect" in self._entity_data:
            return bool(self._entity_data.get('ShouldInspect'))
        return bool(1)

    @property
    def OnlyInspectPlayers(self):
        if "OnlyInspectPlayers" in self._entity_data:
            return bool(self._entity_data.get('OnlyInspectPlayers'))
        return bool(0)

    @property
    def NeverInspectPlayers(self):
        if "NeverInspectPlayers" in self._entity_data:
            return bool(self._entity_data.get('NeverInspectPlayers'))
        return bool(0)


class npc_manhack(BaseNPC, AlyxInteractable):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start packed up (folded and engine off)': (65536, 0),
                                   "Don't use any damage effects": (131072, 0),
                                   'No Danger Sounds': (1048576, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def ignoreclipbrushes(self):
        if "ignoreclipbrushes" in self._entity_data:
            return bool(self._entity_data.get('ignoreclipbrushes'))
        return bool(0)

    @property
    def player_usable(self):
        if "player_usable" in self._entity_data:
            return bool(self._entity_data.get('player_usable'))
        return bool(0)


class npc_mortarsynth(BaseNPC):
    pass


class npc_metropolice(RappelNPC):
    pass

    @property
    def additionalequipment(self):
        if "additionalequipment" in self._entity_data:
            return self._entity_data.get('additionalequipment')
        return "weapon_pistol"

    @property
    def manhacks(self):
        if "manhacks" in self._entity_data:
            return self._entity_data.get('manhacks')
        return "0"

    @property
    def weapondrawn(self):
        if "weapondrawn" in self._entity_data:
            return bool(self._entity_data.get('weapondrawn'))
        return bool(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Simple cops': (131072, 0), 'Rappel (UNCHECK THIS IF IT IS CHECKED!)': (262144, 0),
                                   'Always stitch': (524288, 0), 'No chatter': (1048576, 0),
                                   'Arrest enemies': (2097152, 0), 'No far stitching': (4194304, 0),
                                   'Prevent manhack toss': (8388608, 0),
                                   'Allowed to respond to thrown objects': (16777216, 0),
                                   'Mid-range attacks (halfway between normal + long-range)': (33554432, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class npc_crabsynth(BaseNPC):
    pass


class npc_monk(TalkNPC):
    pass

    @property
    def additionalequipment(self):
        if "additionalequipment" in self._entity_data:
            return self._entity_data.get('additionalequipment')
        return "weapon_annabelle"

    @property
    def HasGun(self):
        if "HasGun" in self._entity_data:
            return bool(self._entity_data.get('HasGun'))
        return bool(1)


class npc_alyx(TalkNPC, Parentname, PlayerCompanion):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/alyx.vmdl"

    @property
    def additionalequipment(self):
        if "additionalequipment" in self._entity_data:
            return self._entity_data.get('additionalequipment')
        return "weapon_alyxgun"

    @property
    def DontPickupWeapons(self):
        if "DontPickupWeapons" in self._entity_data:
            return bool(self._entity_data.get('DontPickupWeapons'))
        return bool(1)

    @property
    def ShouldHaveEMP(self):
        if "ShouldHaveEMP" in self._entity_data:
            return bool(self._entity_data.get('ShouldHaveEMP'))
        return bool(1)


class info_darknessmode_lightsource(Targetname, EnableDisable):
    pass

    @property
    def LightRadius(self):
        if "LightRadius" in self._entity_data:
            return float(self._entity_data.get('LightRadius'))
        return float(256.0)


class npc_kleiner(TalkNPC):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/kleiner.vmdl"


class npc_eli(TalkNPC, Parentname):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/eli.vmdl"


class npc_magnusson(TalkNPC):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/magnusson.vmdl"


class npc_breen(TalkNPC):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/breen.vmdl"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Ignore speech semaphore': (65536, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class npc_mossman(TalkNPC):
    pass


class npc_gman(TalkNPC):
    pass


class npc_dog(BaseNPC):
    pass


class npc_antlion_template_maker(BaseNPCMaker):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Random spawn node': (1024, 0),
                                   'Try to spawn close to the current target': (2048, 0),
                                   'Pick a random fight target': (4096, 0),
                                   'Try to play blocked effects near the player': (8192, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def TemplateName(self):
        if "TemplateName" in self._entity_data:
            return self._entity_data.get('TemplateName')
        return None

    @property
    def spawngroup(self):
        if "spawngroup" in self._entity_data:
            return self._entity_data.get('spawngroup')
        return ""

    @property
    def spawnradius(self):
        if "spawnradius" in self._entity_data:
            return float(self._entity_data.get('spawnradius'))
        return float(512)

    @property
    def spawntarget(self):
        if "spawntarget" in self._entity_data:
            return self._entity_data.get('spawntarget')
        return ""

    @property
    def fighttarget(self):
        if "fighttarget" in self._entity_data:
            return self._entity_data.get('fighttarget')
        return ""

    @property
    def followtarget(self):
        if "followtarget" in self._entity_data:
            return self._entity_data.get('followtarget')
        return ""

    @property
    def vehicledistance(self):
        if "vehicledistance" in self._entity_data:
            return float(self._entity_data.get('vehicledistance'))
        return float(1)

    @property
    def workerspawnrate(self):
        if "workerspawnrate" in self._entity_data:
            return float(self._entity_data.get('workerspawnrate'))
        return float(0)

    @property
    def ignorebugbait(self):
        if "ignorebugbait" in self._entity_data:
            return bool(self._entity_data.get('ignorebugbait'))
        return bool(0)

    @property
    def pool_start(self):
        if "pool_start" in self._entity_data:
            return int(self._entity_data.get('pool_start'))
        return int(0)

    @property
    def pool_max(self):
        if "pool_max" in self._entity_data:
            return int(self._entity_data.get('pool_max'))
        return int(0)

    @property
    def pool_regen_amount(self):
        if "pool_regen_amount" in self._entity_data:
            return int(self._entity_data.get('pool_regen_amount'))
        return int(0)

    @property
    def pool_regen_time(self):
        if "pool_regen_time" in self._entity_data:
            return float(self._entity_data.get('pool_regen_time'))
        return float(0)

    @property
    def createspores(self):
        if "createspores" in self._entity_data:
            return bool(self._entity_data.get('createspores'))
        return bool(0)


class point_antlion_repellant(Targetname):
    pass

    @property
    def repelradius(self):
        if "repelradius" in self._entity_data:
            return float(self._entity_data.get('repelradius'))
        return float(512)


class player_control(Targetname):
    pass


class ai_ally_manager(Targetname):
    pass

    @property
    def maxallies(self):
        if "maxallies" in self._entity_data:
            return int(self._entity_data.get('maxallies'))
        return int(5)

    @property
    def maxmedics(self):
        if "maxmedics" in self._entity_data:
            return int(self._entity_data.get('maxmedics'))
        return int(1)


class ai_goal_lead_weapon(LeadGoalBase):
    pass

    icon_sprite =  "editor/ai_goal_lead.vmat"

    @property
    def WeaponName(self):
        if "WeaponName" in self._entity_data:
            return self._entity_data.get('WeaponName')
        return "weapon_bugbait"

    @property
    def MissingWeaponConceptModifier(self):
        if "MissingWeaponConceptModifier" in self._entity_data:
            return self._entity_data.get('MissingWeaponConceptModifier')
        return None

    @property
    def SearchType(self):
        if "SearchType" in self._entity_data:
            return self._entity_data.get('SearchType')
        return "0"


class ai_citizen_response_system(Targetname):
    pass


class func_healthcharger(EnableDisable, Parentname, Global):
    pass

    @property
    def _minlight(self):
        if "_minlight" in self._entity_data:
            return self._entity_data.get('_minlight')
        return None


class func_recharge(Targetname, Parentname):
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
            for name, (key, _) in {'Citadel recharger': (8192, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class func_vehicleclip(Parentname, Targetname, Global):
    pass


class func_lookdoor(func_movelinear):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'LookDoor Threshold': (8192, 0), 'LookDoor Invert': (16384, 0),
                                   'LookDoor From Open': (32768, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def ProximityDistance(self):
        if "ProximityDistance" in self._entity_data:
            return self._entity_data.get('ProximityDistance')
        return "0.0"

    @property
    def ProximityOffset(self):
        if "ProximityOffset" in self._entity_data:
            return self._entity_data.get('ProximityOffset')
        return "0.0"

    @property
    def FieldOfView(self):
        if "FieldOfView" in self._entity_data:
            return self._entity_data.get('FieldOfView')
        return "0.0"


class trigger_waterydeath(Trigger):
    pass


class env_global(EnvGlobal):
    pass

    @property
    def globalstate(self):
        if "globalstate" in self._entity_data:
            return self._entity_data.get('globalstate')
        return None


class BaseTank(Targetname, Parentname, RenderFields, Global, Shadow):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Active': (1, 0), 'Only Direct': (16, 0), 'Controllable': (32, 0),
                                   'Damage Kick': (64, 0), 'NPC Controllable': (1024, 0),
                                   'NPC Set Controller': (2048, 0), 'Allow friendlies to hit player': (4096, 0),
                                   'Non-solid.': (32768, 0),
                                   'Perfect accuracy every 3rd shot at player': (131072, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def control_volume(self):
        if "control_volume" in self._entity_data:
            return self._entity_data.get('control_volume')
        return ""

    @property
    def master(self):
        if "master" in self._entity_data:
            return self._entity_data.get('master')
        return None

    @property
    def yawrate(self):
        if "yawrate" in self._entity_data:
            return self._entity_data.get('yawrate')
        return "30"

    @property
    def yawrange(self):
        if "yawrange" in self._entity_data:
            return self._entity_data.get('yawrange')
        return "180"

    @property
    def yawtolerance(self):
        if "yawtolerance" in self._entity_data:
            return self._entity_data.get('yawtolerance')
        return "15"

    @property
    def pitchrate(self):
        if "pitchrate" in self._entity_data:
            return self._entity_data.get('pitchrate')
        return "0"

    @property
    def pitchrange(self):
        if "pitchrange" in self._entity_data:
            return self._entity_data.get('pitchrange')
        return "0"

    @property
    def pitchtolerance(self):
        if "pitchtolerance" in self._entity_data:
            return self._entity_data.get('pitchtolerance')
        return "5"

    @property
    def barrel(self):
        if "barrel" in self._entity_data:
            return self._entity_data.get('barrel')
        return "0"

    @property
    def barrely(self):
        if "barrely" in self._entity_data:
            return self._entity_data.get('barrely')
        return "0"

    @property
    def barrelz(self):
        if "barrelz" in self._entity_data:
            return self._entity_data.get('barrelz')
        return "0"

    @property
    def spritesmoke(self):
        if "spritesmoke" in self._entity_data:
            return self._entity_data.get('spritesmoke')
        return ""

    @property
    def spriteflash(self):
        if "spriteflash" in self._entity_data:
            return self._entity_data.get('spriteflash')
        return ""

    @property
    def spritescale(self):
        if "spritescale" in self._entity_data:
            return self._entity_data.get('spritescale')
        return "1"

    @property
    def rotatestartsound(self):
        if "rotatestartsound" in self._entity_data:
            return self._entity_data.get('rotatestartsound')
        return ""

    @property
    def rotatesound(self):
        if "rotatesound" in self._entity_data:
            return self._entity_data.get('rotatesound')
        return ""

    @property
    def rotatestopsound(self):
        if "rotatestopsound" in self._entity_data:
            return self._entity_data.get('rotatestopsound')
        return ""

    @property
    def firerate(self):
        if "firerate" in self._entity_data:
            return self._entity_data.get('firerate')
        return "1"

    @property
    def bullet_damage(self):
        if "bullet_damage" in self._entity_data:
            return self._entity_data.get('bullet_damage')
        return "0"

    @property
    def bullet_damage_vs_player(self):
        if "bullet_damage_vs_player" in self._entity_data:
            return self._entity_data.get('bullet_damage_vs_player')
        return "0"

    @property
    def persistence(self):
        if "persistence" in self._entity_data:
            return self._entity_data.get('persistence')
        return "1"

    @property
    def persistence2(self):
        if "persistence2" in self._entity_data:
            return self._entity_data.get('persistence2')
        return "0"

    @property
    def firespread(self):
        if "firespread" in self._entity_data:
            return self._entity_data.get('firespread')
        return "0"

    @property
    def minRange(self):
        if "minRange" in self._entity_data:
            return self._entity_data.get('minRange')
        return "0"

    @property
    def maxRange(self):
        if "maxRange" in self._entity_data:
            return self._entity_data.get('maxRange')
        return "0"

    @property
    def _minlight(self):
        if "_minlight" in self._entity_data:
            return self._entity_data.get('_minlight')
        return None

    @property
    def gun_base_attach(self):
        if "gun_base_attach" in self._entity_data:
            return self._entity_data.get('gun_base_attach')
        return ""

    @property
    def gun_barrel_attach(self):
        if "gun_barrel_attach" in self._entity_data:
            return self._entity_data.get('gun_barrel_attach')
        return ""

    @property
    def gun_yaw_pose_param(self):
        if "gun_yaw_pose_param" in self._entity_data:
            return self._entity_data.get('gun_yaw_pose_param')
        return ""

    @property
    def gun_yaw_pose_center(self):
        if "gun_yaw_pose_center" in self._entity_data:
            return float(self._entity_data.get('gun_yaw_pose_center'))
        return float(0)

    @property
    def gun_pitch_pose_param(self):
        if "gun_pitch_pose_param" in self._entity_data:
            return self._entity_data.get('gun_pitch_pose_param')
        return ""

    @property
    def gun_pitch_pose_center(self):
        if "gun_pitch_pose_center" in self._entity_data:
            return float(self._entity_data.get('gun_pitch_pose_center'))
        return float(0)

    @property
    def ammo_count(self):
        if "ammo_count" in self._entity_data:
            return int(self._entity_data.get('ammo_count'))
        return int(-1)

    @property
    def LeadTarget(self):
        if "LeadTarget" in self._entity_data:
            return bool(self._entity_data.get('LeadTarget'))
        return bool(False)

    @property
    def npc_man_point(self):
        if "npc_man_point" in self._entity_data:
            return self._entity_data.get('npc_man_point')
        return ""

    @property
    def playergraceperiod(self):
        if "playergraceperiod" in self._entity_data:
            return float(self._entity_data.get('playergraceperiod'))
        return float(0)

    @property
    def ignoregraceupto(self):
        if "ignoregraceupto" in self._entity_data:
            return float(self._entity_data.get('ignoregraceupto'))
        return float(768)

    @property
    def playerlocktimebeforefire(self):
        if "playerlocktimebeforefire" in self._entity_data:
            return float(self._entity_data.get('playerlocktimebeforefire'))
        return float(0)

    @property
    def effecthandling(self):
        if "effecthandling" in self._entity_data:
            return self._entity_data.get('effecthandling')
        return "0"


class func_tank(BaseTank):
    pass

    @property
    def ammotype(self):
        if "ammotype" in self._entity_data:
            return self._entity_data.get('ammotype')
        return ""

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Ignore range when making viewcone checks': (8192, 0),
                                   'Aiming Assistance (Player Only)': (256, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class func_tankpulselaser(BaseTank):
    pass

    @property
    def PulseSpeed(self):
        if "PulseSpeed" in self._entity_data:
            return float(self._entity_data.get('PulseSpeed'))
        return float(1000)

    @property
    def PulseColor(self):
        if "PulseColor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('PulseColor'))
        return parse_int_vector("255 0 0")

    @property
    def PulseWidth(self):
        if "PulseWidth" in self._entity_data:
            return float(self._entity_data.get('PulseWidth'))
        return float(20)

    @property
    def PulseLife(self):
        if "PulseLife" in self._entity_data:
            return float(self._entity_data.get('PulseLife'))
        return float(2)

    @property
    def PulseLag(self):
        if "PulseLag" in self._entity_data:
            return float(self._entity_data.get('PulseLag'))
        return float(0.05)

    @property
    def PulseFireSound(self):
        if "PulseFireSound" in self._entity_data:
            return self._entity_data.get('PulseFireSound')
        return ""


class func_tank_gatling(BaseTank):
    pass


class func_tanklaser(BaseTank):
    pass

    @property
    def laserentity(self):
        if "laserentity" in self._entity_data:
            return self._entity_data.get('laserentity')
        return None


class func_tankrocket(BaseTank):
    pass

    @property
    def rocketspeed(self):
        if "rocketspeed" in self._entity_data:
            return float(self._entity_data.get('rocketspeed'))
        return float(800)


class func_tankairboatgun(BaseTank):
    pass

    @property
    def airboat_gun_model(self):
        if "airboat_gun_model" in self._entity_data:
            return self._entity_data.get('airboat_gun_model')
        return None


class func_tankapcrocket(BaseTank):
    pass

    @property
    def rocketspeed(self):
        if "rocketspeed" in self._entity_data:
            return float(self._entity_data.get('rocketspeed'))
        return float(800)

    @property
    def burstcount(self):
        if "burstcount" in self._entity_data:
            return int(self._entity_data.get('burstcount'))
        return int(10)


class func_tankmortar(BaseTank):
    pass

    @property
    def iMagnitude(self):
        if "iMagnitude" in self._entity_data:
            return int(self._entity_data.get('iMagnitude'))
        return int(100)

    @property
    def firedelay(self):
        if "firedelay" in self._entity_data:
            return self._entity_data.get('firedelay')
        return "2"

    @property
    def firestartsound(self):
        if "firestartsound" in self._entity_data:
            return self._entity_data.get('firestartsound')
        return ""

    @property
    def fireendsound(self):
        if "fireendsound" in self._entity_data:
            return self._entity_data.get('fireendsound')
        return ""

    @property
    def incomingsound(self):
        if "incomingsound" in self._entity_data:
            return self._entity_data.get('incomingsound')
        return ""

    @property
    def warningtime(self):
        if "warningtime" in self._entity_data:
            return float(self._entity_data.get('warningtime'))
        return float(1)

    @property
    def firevariance(self):
        if "firevariance" in self._entity_data:
            return float(self._entity_data.get('firevariance'))
        return float(0)


class func_tankphyscannister(BaseTank):
    pass

    @property
    def barrel_volume(self):
        if "barrel_volume" in self._entity_data:
            return self._entity_data.get('barrel_volume')
        return ""


class func_tank_combine_cannon(BaseTank):
    pass

    @property
    def ammotype(self):
        if "ammotype" in self._entity_data:
            return self._entity_data.get('ammotype')
        return ""


class Item(Targetname, Shadow):
    pass

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
    def phys_start_asleep(self):
        if "phys_start_asleep" in self._entity_data:
            return bool(self._entity_data.get('phys_start_asleep'))
        return bool(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start Constrained': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class item_dynamic_resupply(Item):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {"Use Master's values": (1, 1), 'Is Master': (2, 0),
                                   'Fallback to Health Vial': (8, 0), 'Alternate master': (16, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def DesiredHealth(self):
        if "DesiredHealth" in self._entity_data:
            return float(self._entity_data.get('DesiredHealth'))
        return float(1)

    @property
    def DesiredArmor(self):
        if "DesiredArmor" in self._entity_data:
            return float(self._entity_data.get('DesiredArmor'))
        return float(0.3)

    @property
    def DesiredAmmoPistol(self):
        if "DesiredAmmoPistol" in self._entity_data:
            return float(self._entity_data.get('DesiredAmmoPistol'))
        return float(0.5)

    @property
    def DesiredAmmoSMG1(self):
        if "DesiredAmmoSMG1" in self._entity_data:
            return float(self._entity_data.get('DesiredAmmoSMG1'))
        return float(0.5)

    @property
    def DesiredAmmoSMG1_Grenade(self):
        if "DesiredAmmoSMG1_Grenade" in self._entity_data:
            return float(self._entity_data.get('DesiredAmmoSMG1_Grenade'))
        return float(0.1)

    @property
    def DesiredAmmoAR2(self):
        if "DesiredAmmoAR2" in self._entity_data:
            return float(self._entity_data.get('DesiredAmmoAR2'))
        return float(0.4)

    @property
    def DesiredAmmoBuckshot(self):
        if "DesiredAmmoBuckshot" in self._entity_data:
            return float(self._entity_data.get('DesiredAmmoBuckshot'))
        return float(0.5)

    @property
    def DesiredAmmoRPG_Round(self):
        if "DesiredAmmoRPG_Round" in self._entity_data:
            return float(self._entity_data.get('DesiredAmmoRPG_Round'))
        return float(0)

    @property
    def DesiredAmmoGrenade(self):
        if "DesiredAmmoGrenade" in self._entity_data:
            return float(self._entity_data.get('DesiredAmmoGrenade'))
        return float(0.1)

    @property
    def DesiredAmmo357(self):
        if "DesiredAmmo357" in self._entity_data:
            return float(self._entity_data.get('DesiredAmmo357'))
        return float(0)

    @property
    def DesiredAmmoCrossbow(self):
        if "DesiredAmmoCrossbow" in self._entity_data:
            return float(self._entity_data.get('DesiredAmmoCrossbow'))
        return float(0)

    @property
    def DesiredAmmoAR2_AltFire(self):
        if "DesiredAmmoAR2_AltFire" in self._entity_data:
            return float(self._entity_data.get('DesiredAmmoAR2_AltFire'))
        return float(0)


class item_ammo_pistol(Item):
    pass


class item_ammo_pistol_large(Item):
    pass


class item_ammo_smg1(Item):
    pass


class item_ammo_smg1_large(Item):
    pass


class item_ammo_ar2(Item):
    pass


class item_ammo_ar2_large(Item):
    pass


class item_ammo_357(Item):
    pass


class item_ammo_357_large(Item):
    pass


class item_ammo_crossbow(Item):
    pass


class item_box_buckshot(Item):
    pass


class item_rpg_round(Item):
    pass


class item_ammo_smg1_grenade(Item):
    pass


class item_battery(Item):
    pass


class item_healthkit(Item):
    pass


class item_healthvial_DEPRECATED(Item):
    pass


class item_ammo_ar2_altfire(Item):
    pass


class item_suit(Item):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Short Logon': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class item_ammo_crate(Targetname, BaseFadeProp):
    pass

    @property
    def AmmoType(self):
        if "AmmoType" in self._entity_data:
            return self._entity_data.get('AmmoType')
        return "0"


class item_healthcharger_DEPRECATED(Targetname, BaseFadeProp):
    pass

    @property
    def _minlight(self):
        if "_minlight" in self._entity_data:
            return self._entity_data.get('_minlight')
        return None


class item_suitcharger(Targetname, BaseFadeProp):
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
            for name, (key, _) in {'Citadel recharger': (8192, 0), "Kleiner's recharger": (16384, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class Weapon(Targetname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start constrained': (1, 0), 'Deny player pickup (reserve for NPC)': (2, 0),
                                   'Not puntable by Gravity Gun': (4, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

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


class weapon_crowbar(Weapon):
    pass


class weapon_stunstick(Weapon):
    pass


class weapon_pistol(Weapon):
    pass


class weapon_ar2(Weapon):
    pass


class weapon_rpg(Weapon):
    pass


class weapon_smg1(Weapon):
    pass


class weapon_357(Weapon):
    pass


class weapon_crossbow(Weapon):
    pass


class weapon_zipline(Weapon):
    pass


class weapon_shotgun(Weapon):
    pass


class weapon_frag(Weapon):
    pass


class weapon_physcannon(Weapon):
    pass


class weapon_bugbait(Weapon):
    pass


class weapon_alyxgun(Weapon):
    pass


class weapon_annabelle(Weapon):
    pass


class trigger_rpgfire(Trigger):
    pass


class trigger_vphysics_motion(Trigger):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Can move (through hierarchical attachment)': (4096, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def StartDisabled(self):
        if "StartDisabled" in self._entity_data:
            return bool(self._entity_data.get('StartDisabled'))
        return bool(0)

    @property
    def filtername(self):
        if "filtername" in self._entity_data:
            return self._entity_data.get('filtername')
        return None

    @property
    def SetGravityScale(self):
        if "SetGravityScale" in self._entity_data:
            return float(self._entity_data.get('SetGravityScale'))
        return float(1.0)

    @property
    def SetAdditionalAirDensity(self):
        if "SetAdditionalAirDensity" in self._entity_data:
            return float(self._entity_data.get('SetAdditionalAirDensity'))
        return float(0)

    @property
    def SetVelocityLimit(self):
        if "SetVelocityLimit" in self._entity_data:
            return float(self._entity_data.get('SetVelocityLimit'))
        return float(0.0)

    @property
    def SetVelocityLimitDelta(self):
        if "SetVelocityLimitDelta" in self._entity_data:
            return float(self._entity_data.get('SetVelocityLimitDelta'))
        return float(0.0)

    @property
    def SetVelocityScale(self):
        if "SetVelocityScale" in self._entity_data:
            return float(self._entity_data.get('SetVelocityScale'))
        return float(1.0)

    @property
    def SetAngVelocityLimit(self):
        if "SetAngVelocityLimit" in self._entity_data:
            return float(self._entity_data.get('SetAngVelocityLimit'))
        return float(0.0)

    @property
    def SetAngVelocityScale(self):
        if "SetAngVelocityScale" in self._entity_data:
            return float(self._entity_data.get('SetAngVelocityScale'))
        return float(1.0)

    @property
    def SetLinearForce(self):
        if "SetLinearForce" in self._entity_data:
            return float(self._entity_data.get('SetLinearForce'))
        return float(0.0)

    @property
    def SetLinearForceAngles(self):
        if "SetLinearForceAngles" in self._entity_data:
            return parse_int_vector(self._entity_data.get('SetLinearForceAngles'))
        return parse_int_vector("0 0 0")

    @property
    def ParticleTrailMaterial(self):
        if "ParticleTrailMaterial" in self._entity_data:
            return self._entity_data.get('ParticleTrailMaterial')
        return None

    @property
    def ParticleTrailLifetime(self):
        if "ParticleTrailLifetime" in self._entity_data:
            return float(self._entity_data.get('ParticleTrailLifetime'))
        return float(4)

    @property
    def ParticleTrailStartSize(self):
        if "ParticleTrailStartSize" in self._entity_data:
            return float(self._entity_data.get('ParticleTrailStartSize'))
        return float(2)

    @property
    def ParticleTrailEndSize(self):
        if "ParticleTrailEndSize" in self._entity_data:
            return float(self._entity_data.get('ParticleTrailEndSize'))
        return float(3)


class point_bugbait(Targetname):
    pass

    @property
    def Enabled(self):
        if "Enabled" in self._entity_data:
            return bool(self._entity_data.get('Enabled'))
        return bool(1)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Do not call antlions to position': (1, 0),
                                   "Don't activate on thrown bugbait splashes": (2, 0),
                                   "Don't activate on squeezed bugbait": (4, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return int(self._entity_data.get('radius'))
        return int(512)


class weapon_brickbat(Weapon):
    pass

    @property
    def BrickbatType(self):
        if "BrickbatType" in self._entity_data:
            return self._entity_data.get('BrickbatType')
        return "Rock"


class path_corner_crash(Targetname):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None


class player_loadsaved(Targetname):
    pass

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

    @property
    def loadtime(self):
        if "loadtime" in self._entity_data:
            return self._entity_data.get('loadtime')
        return "0"


class player_weaponstrip(Targetname):
    pass


class player_speedmod(Targetname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Suppress weapons': (1, 0), 'Suppress HUD': (2, 0), 'Suppress jump': (4, 0),
                                   'Suppress duck': (8, 0), 'Suppress use': (16, 0), 'Suppress sprint': (32, 0),
                                   'Suppress attack': (64, 0), 'Suppress zoom': (128, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class env_rotorwash(Targetname, Parentname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Ignore solid': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class combine_mine(Targetname, Parentname):
    pass

    @property
    def bounce(self):
        if "bounce" in self._entity_data:
            return bool(self._entity_data.get('bounce'))
        return bool(1)

    @property
    def LockSilently(self):
        if "LockSilently" in self._entity_data:
            return bool(self._entity_data.get('LockSilently'))
        return bool(1)

    @property
    def StartDisarmed(self):
        if "StartDisarmed" in self._entity_data:
            return bool(self._entity_data.get('StartDisarmed'))
        return bool(0)

    @property
    def Modification(self):
        if "Modification" in self._entity_data:
            return self._entity_data.get('Modification')
        return "0"


class env_ar2explosion(Targetname, Parentname):
    pass

    @property
    def material(self):
        if "material" in self._entity_data:
            return self._entity_data.get('material')
        return "particle/particle_noisesphere"


class env_starfield(Targetname):
    pass


class env_flare(Targetname, Parentname):
    pass

    @property
    def scale(self):
        if "scale" in self._entity_data:
            return float(self._entity_data.get('scale'))
        return float(1)

    @property
    def duration(self):
        if "duration" in self._entity_data:
            return float(self._entity_data.get('duration'))
        return float(30)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'No DLight': (1, 0), 'No Smoke': (2, 0), 'Infinite': (4, 0),
                                   'Start off': (8, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class env_muzzleflash(Targetname, Parentname):
    pass

    @property
    def parentattachment(self):
        if "parentattachment" in self._entity_data:
            return self._entity_data.get('parentattachment')
        return ""

    @property
    def scale(self):
        if "scale" in self._entity_data:
            return float(self._entity_data.get('scale'))
        return float(1)


class logic_achievement(Targetname, EnableDisable):
    pass

    @property
    def AchievementEvent(self):
        if "AchievementEvent" in self._entity_data:
            return self._entity_data.get('AchievementEvent')
        return "0"


class func_monitor(func_brush):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def resolution(self):
        if "resolution" in self._entity_data:
            return self._entity_data.get('resolution')
        return "0"

    @property
    def render_shadows(self):
        if "render_shadows" in self._entity_data:
            return bool(self._entity_data.get('render_shadows'))
        return bool(1)

    @property
    def unique_target(self):
        if "unique_target" in self._entity_data:
            return bool(self._entity_data.get('unique_target'))
        return bool(1)

    @property
    def start_enabled(self):
        if "start_enabled" in self._entity_data:
            return bool(self._entity_data.get('start_enabled'))
        return bool(1)

    @property
    def draw_3dskybox(self):
        if "draw_3dskybox" in self._entity_data:
            return bool(self._entity_data.get('draw_3dskybox'))
        return bool(0)


class func_bulletshield(func_brush):
    pass


class BaseVehicle(Targetname, Global, prop_static_base):
    pass

    @property
    def vehiclescript(self):
        if "vehiclescript" in self._entity_data:
            return self._entity_data.get('vehiclescript')
        return "scripts/vehicles/jeep_test.txt"

    @property
    def actionScale(self):
        if "actionScale" in self._entity_data:
            return float(self._entity_data.get('actionScale'))
        return float(1)


class BaseDriveableVehicle(BaseVehicle):
    pass

    @property
    def VehicleLocked(self):
        if "VehicleLocked" in self._entity_data:
            return bool(self._entity_data.get('VehicleLocked'))
        return bool(0)


class prop_vehicle(BaseVehicle):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Always Think (Run physics every frame)': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class prop_vehicle_driveable(BaseDriveableVehicle):
    pass


class point_apc_controller(Targetname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Active': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def yawrate(self):
        if "yawrate" in self._entity_data:
            return self._entity_data.get('yawrate')
        return "30"

    @property
    def yawtolerance(self):
        if "yawtolerance" in self._entity_data:
            return self._entity_data.get('yawtolerance')
        return "15"

    @property
    def pitchrate(self):
        if "pitchrate" in self._entity_data:
            return self._entity_data.get('pitchrate')
        return "0"

    @property
    def pitchtolerance(self):
        if "pitchtolerance" in self._entity_data:
            return self._entity_data.get('pitchtolerance')
        return "20"

    @property
    def rotatestartsound(self):
        if "rotatestartsound" in self._entity_data:
            return self._entity_data.get('rotatestartsound')
        return ""

    @property
    def rotatesound(self):
        if "rotatesound" in self._entity_data:
            return self._entity_data.get('rotatesound')
        return ""

    @property
    def rotatestopsound(self):
        if "rotatestopsound" in self._entity_data:
            return self._entity_data.get('rotatestopsound')
        return ""

    @property
    def minRange(self):
        if "minRange" in self._entity_data:
            return self._entity_data.get('minRange')
        return "0"

    @property
    def maxRange(self):
        if "maxRange" in self._entity_data:
            return self._entity_data.get('maxRange')
        return "0"

    @property
    def targetentityname(self):
        if "targetentityname" in self._entity_data:
            return self._entity_data.get('targetentityname')
        return ""


class prop_vehicle_apc(BaseDriveableVehicle):
    pass

    @property
    def missilehint(self):
        if "missilehint" in self._entity_data:
            return self._entity_data.get('missilehint')
        return ""


class info_apc_missile_hint(Targetname, EnableDisable):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return ""


class prop_vehicle_jeep(BaseDriveableVehicle):
    pass

    @property
    def CargoVisible(self):
        if "CargoVisible" in self._entity_data:
            return bool(self._entity_data.get('CargoVisible'))
        return bool(0)

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'HUD Locator Precache': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class vehicle_viewcontroller(BaseDriveableVehicle):
    pass


class prop_vehicle_airboat(BaseDriveableVehicle):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/airboat.vmdl"

    @property
    def vehiclescript(self):
        if "vehiclescript" in self._entity_data:
            return self._entity_data.get('vehiclescript')
        return "scripts/vehicles/airboat.txt"

    @property
    def EnableGun(self):
        if "EnableGun" in self._entity_data:
            return bool(self._entity_data.get('EnableGun'))
        return bool(0)


class prop_vehicle_cannon(BaseDriveableVehicle):
    pass


class prop_vehicle_crane(BaseDriveableVehicle):
    pass

    @property
    def magnetname(self):
        if "magnetname" in self._entity_data:
            return self._entity_data.get('magnetname')
        return ""


class prop_vehicle_prisoner_pod(BaseDriveableVehicle, Parentname):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/vehicles/prisoner_pod.vmdl"

    @property
    def vehiclescript(self):
        if "vehiclescript" in self._entity_data:
            return self._entity_data.get('vehiclescript')
        return "scripts/vehicles/prisoner_pod.txt"


class env_speaker(BaseSpeaker):
    pass

    icon_sprite =  "editor/ambient_generic.vmat"


class script_tauremoval(Targetname, Parentname):
    pass

    @property
    def vortigaunt(self):
        if "vortigaunt" in self._entity_data:
            return self._entity_data.get('vortigaunt')
        return None


class script_intro(Targetname):
    pass

    @property
    def alternatefovchange(self):
        if "alternatefovchange" in self._entity_data:
            return bool(self._entity_data.get('alternatefovchange'))
        return bool(0)


class env_citadel_energy_core(Targetname, Parentname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'No small particles': (1, 0), 'Start on': (2, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def scale(self):
        if "scale" in self._entity_data:
            return float(self._entity_data.get('scale'))
        return float(1)


class env_alyxemp(Targetname, Parentname):
    pass

    @property
    def Type(self):
        if "Type" in self._entity_data:
            return self._entity_data.get('Type')
        return "0"

    @property
    def EndTargetName(self):
        if "EndTargetName" in self._entity_data:
            return self._entity_data.get('EndTargetName')
        return ""


class test_sidelist:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def sides(self):
        if "sides" in self._entity_data:
            return parse_int_vector(self._entity_data.get('sides'))
        return parse_int_vector("None")


class info_teleporter_countdown(Targetname):
    pass

    icon_sprite =  "editor/info_target.vmat"


class prop_vehicle_choreo_generic(BaseDriveableVehicle, Parentname):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/vehicles/prisoner_pod.vmdl"

    @property
    def vehiclescript(self):
        if "vehiclescript" in self._entity_data:
            return self._entity_data.get('vehiclescript')
        return "scripts/vehicles/choreo_vehicle.txt"

    @property
    def ignoremoveparent(self):
        if "ignoremoveparent" in self._entity_data:
            return bool(self._entity_data.get('ignoremoveparent'))
        return bool(0)

    @property
    def ignoreplayer(self):
        if "ignoreplayer" in self._entity_data:
            return bool(self._entity_data.get('ignoreplayer'))
        return bool(0)


class filter_combineball_type(BaseFilter):
    pass

    icon_sprite =  "editor/filter_class.vmat"

    @property
    def balltype(self):
        if "balltype" in self._entity_data:
            return self._entity_data.get('balltype')
        return "1"


class env_entity_dissolver(Targetname):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return ""

    @property
    def magnitude(self):
        if "magnitude" in self._entity_data:
            return int(self._entity_data.get('magnitude'))
        return int(250)

    @property
    def dissolvetype(self):
        if "dissolvetype" in self._entity_data:
            return self._entity_data.get('dissolvetype')
        return "Energy"


class prop_coreball(Targetname):
    pass


class prop_scalable(Targetname, Studiomodel, RenderFields):
    pass


class point_push(Targetname):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Test LOS before pushing': (1, 0), 'Use angles for push direction': (2, 0),
                                   'No falloff (constant push at any distance)': (4, 0), 'Push players': (8, 1),
                                   'Push physics': (16, 1)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def enabled(self):
        if "enabled" in self._entity_data:
            return bool(self._entity_data.get('enabled'))
        return bool(1)

    @property
    def magnitude(self):
        if "magnitude" in self._entity_data:
            return float(self._entity_data.get('magnitude'))
        return float(100)

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return float(self._entity_data.get('radius'))
        return float(128)

    @property
    def inner_radius(self):
        if "inner_radius" in self._entity_data:
            return float(self._entity_data.get('inner_radius'))
        return float(0)

    @property
    def influence_cone(self):
        if "influence_cone" in self._entity_data:
            return float(self._entity_data.get('influence_cone'))
        return float(0)

    @property
    def filtername(self):
        if "filtername" in self._entity_data:
            return self._entity_data.get('filtername')
        return ""


class npc_antlion_grub(Targetname, BaseFadeProp, Global):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Do not automatically attach to surface': (1, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class weapon_striderbuster(BasePropPhysics):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {"Don't use game_weapon_manager": (8388608, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def dud(self):
        if "dud" in self._entity_data:
            return bool(self._entity_data.get('dud'))
        return bool(0)


class point_flesh_effect_target(Targetname, Parentname):
    pass

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return float(self._entity_data.get('radius'))
        return float(8)


class prop_door_rotating(BasePropDoorRotating):
    pass


class prop_door_rotating_physics(BasePropDoorRotating, ConstraintSoundInfo):
    pass

    @property
    def LatchIsBreakable(self):
        if "LatchIsBreakable" in self._entity_data:
            return bool(self._entity_data.get('LatchIsBreakable'))
        return bool(0)

    @property
    def HingeIsBreakable(self):
        if "HingeIsBreakable" in self._entity_data:
            return bool(self._entity_data.get('HingeIsBreakable'))
        return bool(0)

    @property
    def ForceFullyOpen(self):
        if "ForceFullyOpen" in self._entity_data:
            return bool(self._entity_data.get('ForceFullyOpen'))
        return bool(0)

    @property
    def friction(self):
        if "friction" in self._entity_data:
            return float(self._entity_data.get('friction'))
        return float(0.001)

    @property
    def GrabAttachmentName(self):
        if "GrabAttachmentName" in self._entity_data:
            return self._entity_data.get('GrabAttachmentName')
        return "grab"


class markup_volume(Targetname, Parentname, Global, EnableDisable):
    pass


class markup_volume_tagged(markup_volume):
    pass

    @property
    def groupnames(self):
        flags = []
        if "groupnames" in self._entity_data:
            value = self._entity_data.get("groupnames", None)
            for name, (key, _) in {}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def tagFieldNames(self):
        if "tagFieldNames" in self._entity_data:
            return self._entity_data.get('tagFieldNames')
        return "groupnames"


class markup_group(markup_volume_tagged):
    pass

    @property
    def groupbyvolume(self):
        if "groupbyvolume" in self._entity_data:
            return bool(self._entity_data.get('groupbyvolume'))
        return bool(0)

    @property
    def groupothergroups(self):
        if "groupothergroups" in self._entity_data:
            return bool(self._entity_data.get('groupothergroups'))
        return bool(0)


class func_nav_markup(markup_volume_tagged):
    pass

    @property
    def navProperty_NavGen(self):
        flags = []
        if "navProperty_NavGen" in self._entity_data:
            value = self._entity_data.get("navProperty_NavGen", None)
            for name, (key, _) in {'Walkable Seed': ('WALKABLESEED', 0), 'No Nav': ('NONAV', 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def navProperty_NavAttributes(self):
        flags = []
        if "navProperty_NavAttributes" in self._entity_data:
            value = self._entity_data.get("navProperty_NavAttributes", None)
            for name, (key, _) in {'Avoid': ('AVOID', 0), 'Split': ('SPLIT', 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def tagFieldNames(self):
        if "tagFieldNames" in self._entity_data:
            return self._entity_data.get('tagFieldNames')
        return "navProperty_NavAttributes"


class markup_volume_with_ref(markup_volume_tagged):
    pass

    @property
    def ref_position(self):
        if "ref_position" in self._entity_data:
            return parse_int_vector(self._entity_data.get('ref_position'))
        return parse_int_vector("0 0 0")

    @property
    def use_ref_position(self):
        if "use_ref_position" in self._entity_data:
            return bool(self._entity_data.get('use_ref_position'))
        return bool(1)


class post_processing_volume(Trigger):
    pass

    @property
    def postprocessing(self):
        if "postprocessing" in self._entity_data:
            return self._entity_data.get('postprocessing')
        return None

    @property
    def fadetime(self):
        if "fadetime" in self._entity_data:
            return float(self._entity_data.get('fadetime'))
        return float(1.0)

    @property
    def enableexposure(self):
        if "enableexposure" in self._entity_data:
            return bool(self._entity_data.get('enableexposure'))
        return bool(1)

    @property
    def minexposure(self):
        if "minexposure" in self._entity_data:
            return float(self._entity_data.get('minexposure'))
        return float(0.25)

    @property
    def maxexposure(self):
        if "maxexposure" in self._entity_data:
            return float(self._entity_data.get('maxexposure'))
        return float(8)

    @property
    def exposurecompensation(self):
        if "exposurecompensation" in self._entity_data:
            return float(self._entity_data.get('exposurecompensation'))
        return float(0)

    @property
    def exposurespeedup(self):
        if "exposurespeedup" in self._entity_data:
            return float(self._entity_data.get('exposurespeedup'))
        return float(1)

    @property
    def exposurespeeddown(self):
        if "exposurespeeddown" in self._entity_data:
            return float(self._entity_data.get('exposurespeeddown'))
        return float(1)

    @property
    def master(self):
        if "master" in self._entity_data:
            return bool(self._entity_data.get('master'))
        return bool(0)


class worldspawn(worldbase):
    pass

    @property
    def baked_light_index_min(self):
        if "baked_light_index_min" in self._entity_data:
            return int(self._entity_data.get('baked_light_index_min'))
        return int(0)

    @property
    def baked_light_index_max(self):
        if "baked_light_index_max" in self._entity_data:
            return int(self._entity_data.get('baked_light_index_max'))
        return int(256)

    @property
    def max_lightmap_resolution(self):
        if "max_lightmap_resolution" in self._entity_data:
            return self._entity_data.get('max_lightmap_resolution')
        return "0"

    @property
    def lightmap_queries(self):
        if "lightmap_queries" in self._entity_data:
            return bool(self._entity_data.get('lightmap_queries'))
        return bool(1)


class shared_enable_disable:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def start_enabled(self):
        if "start_enabled" in self._entity_data:
            return bool(self._entity_data.get('start_enabled'))
        return bool(1)


class trigger_traversal_modifier(Trigger):
    pass

    @property
    def target_point(self):
        if "target_point" in self._entity_data:
            return parse_int_vector(self._entity_data.get('target_point'))
        return parse_int_vector("0 0 0")

    @property
    def top_point(self):
        if "top_point" in self._entity_data:
            return parse_int_vector(self._entity_data.get('top_point'))
        return parse_int_vector("0 64 0")

    @property
    def bottom_point(self):
        if "bottom_point" in self._entity_data:
            return parse_int_vector(self._entity_data.get('bottom_point'))
        return parse_int_vector("64 0 0")

    @property
    def instant_traversal(self):
        if "instant_traversal" in self._entity_data:
            return bool(self._entity_data.get('instant_traversal'))
        return bool(0)

    @property
    def wooden(self):
        if "wooden" in self._entity_data:
            return bool(self._entity_data.get('wooden'))
        return bool(0)

    @property
    def object_type(self):
        if "object_type" in self._entity_data:
            return self._entity_data.get('object_type')
        return "0"

    @property
    def window_shatter(self):
        if "window_shatter" in self._entity_data:
            return bool(self._entity_data.get('window_shatter'))
        return bool(0)


class trigger_traversal_modifier_to_line(Trigger):
    pass

    @property
    def point_A(self):
        if "point_A" in self._entity_data:
            return parse_int_vector(self._entity_data.get('point_A'))
        return parse_int_vector("0 0 0")

    @property
    def point_B(self):
        if "point_B" in self._entity_data:
            return parse_int_vector(self._entity_data.get('point_B'))
        return parse_int_vector("0 64 0")


class trigger_traversal_no_teleport(Trigger):
    pass


class trigger_traversal_invalid_spot(Trigger):
    pass

    @property
    def allow_walk_move(self):
        if "allow_walk_move" in self._entity_data:
            return bool(self._entity_data.get('allow_walk_move'))
        return bool(0)


class trigger_traversal_tp_interrupt(Trigger):
    pass

    @property
    def landing_entity_name(self):
        if "landing_entity_name" in self._entity_data:
            return self._entity_data.get('landing_entity_name')
        return None

    @property
    def landing_relative_offset(self):
        if "landing_relative_offset" in self._entity_data:
            return parse_int_vector(self._entity_data.get('landing_relative_offset'))
        return parse_int_vector("0 0 0")

    @property
    def tp_suppress_remind_interval(self):
        if "tp_suppress_remind_interval" in self._entity_data:
            return float(self._entity_data.get('tp_suppress_remind_interval'))
        return float(1)

    @property
    def capture_on_interrupt(self):
        if "capture_on_interrupt" in self._entity_data:
            return bool(self._entity_data.get('capture_on_interrupt'))
        return bool(1)

    @property
    def capture_on_touch(self):
        if "capture_on_touch" in self._entity_data:
            return bool(self._entity_data.get('capture_on_touch'))
        return bool(0)

    @property
    def capture_ignore_continuous(self):
        if "capture_ignore_continuous" in self._entity_data:
            return bool(self._entity_data.get('capture_ignore_continuous'))
        return bool(0)

    @property
    def tp_suppress_sound(self):
        if "tp_suppress_sound" in self._entity_data:
            return self._entity_data.get('tp_suppress_sound')
        return None

    @property
    def interrupt_sound(self):
        if "interrupt_sound" in self._entity_data:
            return self._entity_data.get('interrupt_sound')
        return None


class VRHandAttachment:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data


class BaseItemPhysics(Targetname, Shadow):
    pass

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
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start Asleep': (1, 0), 'Motion Disabled': (8, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def interactAs(self):
        if "interactAs" in self._entity_data:
            return self._entity_data.get('interactAs')
        return ""


class item_hlvr_prop_flashlight(Targetname, Parentname, VRHandAttachment):
    pass

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Start Constrained': (1048576, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/weapons/vr_flashlight/vr_flashlight.vmdl"

    @property
    def StartDisabled(self):
        if "StartDisabled" in self._entity_data:
            return bool(self._entity_data.get('StartDisabled'))
        return bool(0)

    @property
    def BounceLightEnabled(self):
        if "BounceLightEnabled" in self._entity_data:
            return bool(self._entity_data.get('BounceLightEnabled'))
        return bool(0)


class item_hlvr_weapon_energygun(Item, VRHandAttachment):
    pass

    @property
    def set_spawn_ammo(self):
        if "set_spawn_ammo" in self._entity_data:
            return int(self._entity_data.get('set_spawn_ammo'))
        return int(-1)


class hlvr_weapon_energygun(Item, VRHandAttachment, Parentname):
    pass

    @property
    def set_spawn_ammo(self):
        if "set_spawn_ammo" in self._entity_data:
            return int(self._entity_data.get('set_spawn_ammo'))
        return int(-1)


class item_hlvr_weapon_shotgun(Item, VRHandAttachment, Parentname):
    pass

    @property
    def set_spawn_ammo(self):
        if "set_spawn_ammo" in self._entity_data:
            return int(self._entity_data.get('set_spawn_ammo'))
        return int(-1)


class item_hlvr_weapon_rapidfire(Item, VRHandAttachment):
    pass

    @property
    def set_spawn_ammo(self):
        if "set_spawn_ammo" in self._entity_data:
            return int(self._entity_data.get('set_spawn_ammo'))
        return int(-1)


class item_hlvr_weapon_generic_pistol(Item, VRHandAttachment):
    pass

    @property
    def inventory_name(self):
        if "inventory_name" in self._entity_data:
            return self._entity_data.get('inventory_name')
        return "generic_pistol"

    @property
    def inventory_model(self):
        if "inventory_model" in self._entity_data:
            return self._entity_data.get('inventory_model')
        return "models/interface/inventory/pistol_interface_ui.vmdl"

    @property
    def inventory_position(self):
        if "inventory_position" in self._entity_data:
            return self._entity_data.get('inventory_position')
        return "8"

    @property
    def set_spawn_ammo(self):
        if "set_spawn_ammo" in self._entity_data:
            return int(self._entity_data.get('set_spawn_ammo'))
        return int(-1)

    @property
    def ammo_per_clip(self):
        if "ammo_per_clip" in self._entity_data:
            return int(self._entity_data.get('ammo_per_clip'))
        return int(10)

    @property
    def damage(self):
        if "damage" in self._entity_data:
            return int(self._entity_data.get('damage'))
        return int(8)

    @property
    def attack_interval(self):
        if "attack_interval" in self._entity_data:
            return float(self._entity_data.get('attack_interval'))
        return float(0.175)

    @property
    def clip_grab_dist(self):
        if "clip_grab_dist" in self._entity_data:
            return float(self._entity_data.get('clip_grab_dist'))
        return float(8.0)

    @property
    def bullet_count_anim_rate(self):
        if "bullet_count_anim_rate" in self._entity_data:
            return float(self._entity_data.get('bullet_count_anim_rate'))
        return float(1.0)

    @property
    def slide_interact_min_dist(self):
        if "slide_interact_min_dist" in self._entity_data:
            return float(self._entity_data.get('slide_interact_min_dist'))
        return float(6.0)

    @property
    def slide_interact_max_dist(self):
        if "slide_interact_max_dist" in self._entity_data:
            return float(self._entity_data.get('slide_interact_max_dist'))
        return float(6.0)

    @property
    def bottom_grip_min_dist(self):
        if "bottom_grip_min_dist" in self._entity_data:
            return float(self._entity_data.get('bottom_grip_min_dist'))
        return float(4.0)

    @property
    def bottom_grip_max_dist(self):
        if "bottom_grip_max_dist" in self._entity_data:
            return float(self._entity_data.get('bottom_grip_max_dist'))
        return float(4.5)

    @property
    def bottom_grip_disengage_dist(self):
        if "bottom_grip_disengage_dist" in self._entity_data:
            return float(self._entity_data.get('bottom_grip_disengage_dist'))
        return float(5.0)

    @property
    def model_right_handed(self):
        if "model_right_handed" in self._entity_data:
            return self._entity_data.get('model_right_handed')
        return "models/weapons/vr_alyxgun/vr_alyxgun.vmdl"

    @property
    def model_left_handed(self):
        if "model_left_handed" in self._entity_data:
            return self._entity_data.get('model_left_handed')
        return "models/weapons/vr_alyxgun/vr_alyxgun_lhand.vmdl"

    @property
    def slide_model_right_handed(self):
        if "slide_model_right_handed" in self._entity_data:
            return self._entity_data.get('slide_model_right_handed')
        return "models/weapons/vr_alyxgun/vr_alyxgun_slide_anim_interact.vmdl"

    @property
    def slide_model_left_handed(self):
        if "slide_model_left_handed" in self._entity_data:
            return self._entity_data.get('slide_model_left_handed')
        return "models/weapons/vr_alyxgun/vr_alyxgun_slide_anim_interact_lhand.vmdl"

    @property
    def clip_model_right_handed(self):
        if "clip_model_right_handed" in self._entity_data:
            return self._entity_data.get('clip_model_right_handed')
        return "models/weapons/vr_alyxgun/vr_alyxgun_clip.vmdl"

    @property
    def clip_model_left_handed(self):
        if "clip_model_left_handed" in self._entity_data:
            return self._entity_data.get('clip_model_left_handed')
        return "models/weapons/vr_alyxgun/vr_alyxgun_clip_lhand.vmdl"

    @property
    def single_bullet_model(self):
        if "single_bullet_model" in self._entity_data:
            return self._entity_data.get('single_bullet_model')
        return "models/weapons/vr_alyxgun/vr_alyxgun_bullet.vmdl"

    @property
    def eject_shell_model(self):
        if "eject_shell_model" in self._entity_data:
            return self._entity_data.get('eject_shell_model')
        return "models/weapons/vr_shellcases/pistol_shellcase01.vmdl"

    @property
    def shoot_sound(self):
        if "shoot_sound" in self._entity_data:
            return self._entity_data.get('shoot_sound')
        return "AlyxPistol.Fire"

    @property
    def no_ammo_sound(self):
        if "no_ammo_sound" in self._entity_data:
            return self._entity_data.get('no_ammo_sound')
        return "AlyxPistol.CarryingNoAmmo"

    @property
    def last_shot_chambered(self):
        if "last_shot_chambered" in self._entity_data:
            return self._entity_data.get('last_shot_chambered')
        return "Pistol.LastShotChambered"

    @property
    def slide_lock_sound(self):
        if "slide_lock_sound" in self._entity_data:
            return self._entity_data.get('slide_lock_sound')
        return "Pistol.SlideLock"

    @property
    def slide_back_sound(self):
        if "slide_back_sound" in self._entity_data:
            return self._entity_data.get('slide_back_sound')
        return "AlyxPistol.Slideback"

    @property
    def slide_close_sound(self):
        if "slide_close_sound" in self._entity_data:
            return self._entity_data.get('slide_close_sound')
        return "Pistol.CloseSlide"

    @property
    def clip_insert_sound(self):
        if "clip_insert_sound" in self._entity_data:
            return self._entity_data.get('clip_insert_sound')
        return "Pistol.ClipInsert"

    @property
    def clip_release_sound(self):
        if "clip_release_sound" in self._entity_data:
            return self._entity_data.get('clip_release_sound')
        return "Pistol.ClipRelease"

    @property
    def muzzle_flash_effect(self):
        if "muzzle_flash_effect" in self._entity_data:
            return self._entity_data.get('muzzle_flash_effect')
        return "particles/weapon_fx/muzzleflash_pistol_small.vpcf"

    @property
    def tracer_effect(self):
        if "tracer_effect" in self._entity_data:
            return self._entity_data.get('tracer_effect')
        return "particles/tracer_fx/pistol_tracer.vpcf"

    @property
    def eject_shell_smoke_effect(self):
        if "eject_shell_smoke_effect" in self._entity_data:
            return self._entity_data.get('eject_shell_smoke_effect')
        return "particles/weapon_fx/weapon_shell_smoke.vpcf"

    @property
    def glow_effect(self):
        if "glow_effect" in self._entity_data:
            return self._entity_data.get('glow_effect')
        return "models/weapons/vr_alyxgun/vr_alyxgun_worlditem_glow.vpcf"

    @property
    def barrel_smoke_effect(self):
        if "barrel_smoke_effect" in self._entity_data:
            return self._entity_data.get('barrel_smoke_effect')
        return "particles/weapon_fx/weapon_pistol_barrel_smoke.vpcf"

    @property
    def clip_glow_effect(self):
        if "clip_glow_effect" in self._entity_data:
            return self._entity_data.get('clip_glow_effect')
        return "models/weapons/vr_alyxgun/vr_alyxgun_clip_glow.vpcf"


class hlvr_weapon_crowbar(Item):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/weapons/w_crowbar.vmdl"


class item_hlvr_grenade_frag(Item, DamageFilter):
    pass

    @property
    def ammobalancing_removable(self):
        if "ammobalancing_removable" in self._entity_data:
            return self._entity_data.get('ammobalancing_removable')
        return "0"

    @property
    def interactAs(self):
        if "interactAs" in self._entity_data:
            return self._entity_data.get('interactAs')
        return ""


class item_hlvr_grenade_remote_sticky(Item):
    pass


class item_hlvr_grenade_bomb(Item):
    pass


class item_hlvr_grenade_xen(Item):
    pass


class HLVRAmmo:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def ammobalancing_removable(self):
        if "ammobalancing_removable" in self._entity_data:
            return self._entity_data.get('ammobalancing_removable')
        return "0"


class item_hlvr_clip_energygun(BaseItemPhysics, HLVRAmmo):
    pass


class item_hlvr_clip_energygun_multiple(BaseItemPhysics, HLVRAmmo):
    pass


class item_hlvr_clip_rapidfire(BaseItemPhysics, HLVRAmmo):
    pass


class item_hlvr_clip_shotgun_single(BaseItemPhysics, HLVRAmmo):
    pass


class item_hlvr_clip_shotgun_multiple(BaseItemPhysics, HLVRAmmo):
    pass


class item_hlvr_clip_generic_pistol(BaseItemPhysics, HLVRAmmo):
    pass

    @property
    def ammo_per_clip(self):
        if "ammo_per_clip" in self._entity_data:
            return int(self._entity_data.get('ammo_per_clip'))
        return int(8)

    @property
    def model_right_handed(self):
        if "model_right_handed" in self._entity_data:
            return self._entity_data.get('model_right_handed')
        return "models/weapons/vr_alyxgun/vr_alyxgun_clip.vmdl"

    @property
    def model_left_handed(self):
        if "model_left_handed" in self._entity_data:
            return self._entity_data.get('model_left_handed')
        return "models/weapons/vr_alyxgun/vr_alyxgun_clip_lhand.vmdl"

    @property
    def glow_effect(self):
        if "glow_effect" in self._entity_data:
            return self._entity_data.get('glow_effect')
        return "models/weapons/vr_alyxgun/vr_alyxgun_clip_glow.vpcf"


class item_hlvr_clip_generic_pistol_multiple(BaseItemPhysics, HLVRAmmo):
    pass

    @property
    def ammo_per_clip(self):
        if "ammo_per_clip" in self._entity_data:
            return int(self._entity_data.get('ammo_per_clip'))
        return int(32)

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/weapons/vr_alyxgun/pistol_clip_holder.vmdl"

    @property
    def glow_effect(self):
        if "glow_effect" in self._entity_data:
            return self._entity_data.get('glow_effect')
        return "models/weapons/vr_alyxgun/vr_alyxgun_clip_holder_glow.vpcf"


class item_healthvial(BaseItemPhysics, Parentname):
    pass

    @property
    def ammobalancing_removable(self):
        if "ammobalancing_removable" in self._entity_data:
            return self._entity_data.get('ammobalancing_removable')
        return "0"

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Physgun can ALWAYS pick up. No matter what.': (1048576, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags


class item_hlvr_weapon_grabbity_glove(Item):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/hands/grabbity_glove_worldmodel.vmdl"


class item_hlvr_weapon_grabbity_slingshot(Item):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/hands/grabbity_glove_worldmodel.vmdl"


class item_hlvr_weapon_tripmine(BaseItemPhysics):
    pass

    @property
    def StartActivated(self):
        if "StartActivated" in self._entity_data:
            return bool(self._entity_data.get('StartActivated'))
        return bool(0)

    @property
    def StartAttached(self):
        if "StartAttached" in self._entity_data:
            return bool(self._entity_data.get('StartAttached'))
        return bool(0)

    @property
    def PreventTripping(self):
        if "PreventTripping" in self._entity_data:
            return bool(self._entity_data.get('PreventTripping'))
        return bool(0)

    @property
    def HackDifficultyName(self):
        if "HackDifficultyName" in self._entity_data:
            return self._entity_data.get('HackDifficultyName')
        return "Medium"


class item_hlvr_weapon_radio(Item, VRHandAttachment):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/props/max/walkietalkie/walkietalkie.vmdl"


class item_hlvr_multitool(Item, VRHandAttachment):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/weapons/vr_alyxtool/alyx_tool.vmdl"


class npc_headcrab_runner(BaseHeadcrab):
    pass

    @property
    def reviver_group(self):
        if "reviver_group" in self._entity_data:
            return self._entity_data.get('reviver_group')
        return ""


class item_hlvr_weaponmodule_rapidfire(Item):
    pass


class item_hlvr_weaponmodule_ricochet(Item):
    pass


class item_hlvr_weaponmodule_snark(Item):
    pass


class item_hlvr_weaponmodule_zapper(Item):
    pass


class item_hlvr_weaponmodule_guidedmissle(Item):
    pass


class item_hlvr_weaponmodule_guidedmissle_cluster(Item):
    pass


class item_hlvr_weaponmodule_physcannon(Item):
    pass


class hlvr_grenadepin_proxy(Item):
    pass


class func_hlvr_nav_markup(func_nav_markup):
    pass

    @property
    def tagFieldNames(self):
        if "tagFieldNames" in self._entity_data:
            return self._entity_data.get('tagFieldNames')
        return "navProperty_NavAttributes,navProperty_NavGen"


class func_nav_blocker(Targetname):
    pass

    @property
    def StartDisabled(self):
        if "StartDisabled" in self._entity_data:
            return bool(self._entity_data.get('StartDisabled'))
        return bool(0)


class prop_animinteractable(Targetname, Parentname, BaseFadeProp, EnableDisable):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/interaction/anim_interact/valve/valve.vmdl"

    @property
    def InitialCompletionAmount(self):
        if "InitialCompletionAmount" in self._entity_data:
            return float(self._entity_data.get('InitialCompletionAmount'))
        return float(0)

    @property
    def TargetCompletionValueA(self):
        if "TargetCompletionValueA" in self._entity_data:
            return float(self._entity_data.get('TargetCompletionValueA'))
        return float(1)

    @property
    def TargetCompletionValueB(self):
        if "TargetCompletionValueB" in self._entity_data:
            return float(self._entity_data.get('TargetCompletionValueB'))
        return float(-1)

    @property
    def TargetCompletionValueC(self):
        if "TargetCompletionValueC" in self._entity_data:
            return float(self._entity_data.get('TargetCompletionValueC'))
        return float(-1)

    @property
    def TargetCompletionValueD(self):
        if "TargetCompletionValueD" in self._entity_data:
            return float(self._entity_data.get('TargetCompletionValueD'))
        return float(-1)

    @property
    def TargetCompletionValueE(self):
        if "TargetCompletionValueE" in self._entity_data:
            return float(self._entity_data.get('TargetCompletionValueE'))
        return float(-1)

    @property
    def TargetCompletionValueF(self):
        if "TargetCompletionValueF" in self._entity_data:
            return float(self._entity_data.get('TargetCompletionValueF'))
        return float(-1)

    @property
    def TargetCompletionThreshold(self):
        if "TargetCompletionThreshold" in self._entity_data:
            return float(self._entity_data.get('TargetCompletionThreshold'))
        return float(0.1)

    @property
    def ObjectRequirement(self):
        if "ObjectRequirement" in self._entity_data:
            return self._entity_data.get('ObjectRequirement')
        return ""

    @property
    def OnlyRunForward(self):
        if "OnlyRunForward" in self._entity_data:
            return bool(self._entity_data.get('OnlyRunForward'))
        return bool(0)

    @property
    def OnlyRunBackward(self):
        if "OnlyRunBackward" in self._entity_data:
            return bool(self._entity_data.get('OnlyRunBackward'))
        return bool(0)

    @property
    def LimitForward(self):
        if "LimitForward" in self._entity_data:
            return float(self._entity_data.get('LimitForward'))
        return float(1)

    @property
    def LimitBackward(self):
        if "LimitBackward" in self._entity_data:
            return float(self._entity_data.get('LimitBackward'))
        return float(0)

    @property
    def LimitStop(self):
        if "LimitStop" in self._entity_data:
            return float(self._entity_data.get('LimitStop'))
        return float(-1)

    @property
    def StartLocked(self):
        if "StartLocked" in self._entity_data:
            return bool(self._entity_data.get('StartLocked'))
        return bool(0)

    @property
    def LimitLocked(self):
        if "LimitLocked" in self._entity_data:
            return float(self._entity_data.get('LimitLocked'))
        return float(0)

    @property
    def ReturnToCompletion(self):
        if "ReturnToCompletion" in self._entity_data:
            return bool(self._entity_data.get('ReturnToCompletion'))
        return bool(0)

    @property
    def ReturnToCompletionAmount(self):
        if "ReturnToCompletionAmount" in self._entity_data:
            return float(self._entity_data.get('ReturnToCompletionAmount'))
        return float(0)

    @property
    def ReturnToCompletionThreshold(self):
        if "ReturnToCompletionThreshold" in self._entity_data:
            return float(self._entity_data.get('ReturnToCompletionThreshold'))
        return float(-1)

    @property
    def ReturnToCompletionDelay(self):
        if "ReturnToCompletionDelay" in self._entity_data:
            return float(self._entity_data.get('ReturnToCompletionDelay'))
        return float(0)

    @property
    def AnimationDuration(self):
        if "AnimationDuration" in self._entity_data:
            return float(self._entity_data.get('AnimationDuration'))
        return float(5)

    @property
    def StartSound(self):
        if "StartSound" in self._entity_data:
            return self._entity_data.get('StartSound')
        return None

    @property
    def MoveSound(self):
        if "MoveSound" in self._entity_data:
            return self._entity_data.get('MoveSound')
        return None

    @property
    def StopSound(self):
        if "StopSound" in self._entity_data:
            return self._entity_data.get('StopSound')
        return None

    @property
    def OpenCompleteSound(self):
        if "OpenCompleteSound" in self._entity_data:
            return self._entity_data.get('OpenCompleteSound')
        return None

    @property
    def CloseCompleteSound(self):
        if "CloseCompleteSound" in self._entity_data:
            return self._entity_data.get('CloseCompleteSound')
        return None

    @property
    def BounceSound(self):
        if "BounceSound" in self._entity_data:
            return self._entity_data.get('BounceSound')
        return None

    @property
    def LockedSound(self):
        if "LockedSound" in self._entity_data:
            return self._entity_data.get('LockedSound')
        return None

    @property
    def ReturnForwardMoveSound(self):
        if "ReturnForwardMoveSound" in self._entity_data:
            return self._entity_data.get('ReturnForwardMoveSound')
        return None

    @property
    def ReturnBackwardMoveSound(self):
        if "ReturnBackwardMoveSound" in self._entity_data:
            return self._entity_data.get('ReturnBackwardMoveSound')
        return None

    @property
    def InteractionBoneName(self):
        if "InteractionBoneName" in self._entity_data:
            return self._entity_data.get('InteractionBoneName')
        return "interact"

    @property
    def ReturnToCompletionStyle(self):
        if "ReturnToCompletionStyle" in self._entity_data:
            return self._entity_data.get('ReturnToCompletionStyle')
        return "0"

    @property
    def AllowGravityGunPull(self):
        if "AllowGravityGunPull" in self._entity_data:
            return bool(self._entity_data.get('AllowGravityGunPull'))
        return bool(0)

    @property
    def RetainVelocity(self):
        if "RetainVelocity" in self._entity_data:
            return bool(self._entity_data.get('RetainVelocity'))
        return bool(0)

    @property
    def ReactToDynamicPhysics(self):
        if "ReactToDynamicPhysics" in self._entity_data:
            return bool(self._entity_data.get('ReactToDynamicPhysics'))
        return bool(0)

    @property
    def IgnoreHandRotation(self):
        if "IgnoreHandRotation" in self._entity_data:
            return bool(self._entity_data.get('IgnoreHandRotation'))
        return bool(1)

    @property
    def IgnoreHandPosition(self):
        if "IgnoreHandPosition" in self._entity_data:
            return bool(self._entity_data.get('IgnoreHandPosition'))
        return bool(0)

    @property
    def DoHapticsOnBothHands(self):
        if "DoHapticsOnBothHands" in self._entity_data:
            return bool(self._entity_data.get('DoHapticsOnBothHands'))
        return bool(0)

    @property
    def PositiveResistance(self):
        if "PositiveResistance" in self._entity_data:
            return float(self._entity_data.get('PositiveResistance'))
        return float(1)

    @property
    def UpdateChildModels(self):
        if "UpdateChildModels" in self._entity_data:
            return bool(self._entity_data.get('UpdateChildModels'))
        return bool(0)

    @property
    def NormalizeChildModelUpdates(self):
        if "NormalizeChildModelUpdates" in self._entity_data:
            return bool(self._entity_data.get('NormalizeChildModelUpdates'))
        return bool(0)

    @property
    def ChildModelAnimgraphParameter(self):
        if "ChildModelAnimgraphParameter" in self._entity_data:
            return self._entity_data.get('ChildModelAnimgraphParameter')
        return ""

    @property
    def SetNavIgnore(self):
        if "SetNavIgnore" in self._entity_data:
            return bool(self._entity_data.get('SetNavIgnore'))
        return bool(0)

    @property
    def CreateNavObstacle(self):
        if "CreateNavObstacle" in self._entity_data:
            return bool(self._entity_data.get('CreateNavObstacle'))
        return bool(0)

    @property
    def ReleaseOnPlayerDamage(self):
        if "ReleaseOnPlayerDamage" in self._entity_data:
            return bool(self._entity_data.get('ReleaseOnPlayerDamage'))
        return bool(0)

    @property
    def BehaveAsPropPhysics(self):
        if "BehaveAsPropPhysics" in self._entity_data:
            return bool(self._entity_data.get('BehaveAsPropPhysics'))
        return bool(0)

    @property
    def AddToSpatialPartition(self):
        if "AddToSpatialPartition" in self._entity_data:
            return bool(self._entity_data.get('AddToSpatialPartition'))
        return bool(1)

    @property
    def interactAs(self):
        if "interactAs" in self._entity_data:
            return self._entity_data.get('interactAs')
        return ""


class info_hlvr_equip_player(Targetname):
    pass

    icon_sprite =  "editor/info_hlvr_equip_player.vmat"

    @property
    def equip_on_mapstart(self):
        if "equip_on_mapstart" in self._entity_data:
            return bool(self._entity_data.get('equip_on_mapstart'))
        return bool(1)

    @property
    def energygun(self):
        if "energygun" in self._entity_data:
            return bool(self._entity_data.get('energygun'))
        return bool(0)

    @property
    def shotgun(self):
        if "shotgun" in self._entity_data:
            return bool(self._entity_data.get('shotgun'))
        return bool(0)

    @property
    def rapidfire(self):
        if "rapidfire" in self._entity_data:
            return bool(self._entity_data.get('rapidfire'))
        return bool(0)

    @property
    def multitool(self):
        if "multitool" in self._entity_data:
            return bool(self._entity_data.get('multitool'))
        return bool(0)

    @property
    def flashlight(self):
        if "flashlight" in self._entity_data:
            return bool(self._entity_data.get('flashlight'))
        return bool(0)

    @property
    def flashlight_enabled(self):
        if "flashlight_enabled" in self._entity_data:
            return bool(self._entity_data.get('flashlight_enabled'))
        return bool(0)

    @property
    def grabbitygloves(self):
        if "grabbitygloves" in self._entity_data:
            return bool(self._entity_data.get('grabbitygloves'))
        return bool(0)

    @property
    def itemholder(self):
        if "itemholder" in self._entity_data:
            return bool(self._entity_data.get('itemholder'))
        return bool(0)

    @property
    def set_ammo(self):
        if "set_ammo" in self._entity_data:
            return int(self._entity_data.get('set_ammo'))
        return int(-1)

    @property
    def set_ammo_rapidfire(self):
        if "set_ammo_rapidfire" in self._entity_data:
            return int(self._entity_data.get('set_ammo_rapidfire'))
        return int(-1)

    @property
    def set_ammo_shotgun(self):
        if "set_ammo_shotgun" in self._entity_data:
            return int(self._entity_data.get('set_ammo_shotgun'))
        return int(-1)

    @property
    def set_resin(self):
        if "set_resin" in self._entity_data:
            return int(self._entity_data.get('set_resin'))
        return int(-1)

    @property
    def start_weapons_empty(self):
        if "start_weapons_empty" in self._entity_data:
            return bool(self._entity_data.get('start_weapons_empty'))
        return bool(0)

    @property
    def inventory_enabled(self):
        if "inventory_enabled" in self._entity_data:
            return bool(self._entity_data.get('inventory_enabled'))
        return bool(1)

    @property
    def backpack_enabled(self):
        if "backpack_enabled" in self._entity_data:
            return bool(self._entity_data.get('backpack_enabled'))
        return bool(1)

    @property
    def allow_removal(self):
        if "allow_removal" in self._entity_data:
            return bool(self._entity_data.get('allow_removal'))
        return bool(0)

    @property
    def pistol_upgrade_lasersight(self):
        if "pistol_upgrade_lasersight" in self._entity_data:
            return bool(self._entity_data.get('pistol_upgrade_lasersight'))
        return bool(0)

    @property
    def pistol_upgrade_reflexsight(self):
        if "pistol_upgrade_reflexsight" in self._entity_data:
            return bool(self._entity_data.get('pistol_upgrade_reflexsight'))
        return bool(0)

    @property
    def pistol_upgrade_bullethopper(self):
        if "pistol_upgrade_bullethopper" in self._entity_data:
            return bool(self._entity_data.get('pistol_upgrade_bullethopper'))
        return bool(0)

    @property
    def pistol_upgrade_burstfire(self):
        if "pistol_upgrade_burstfire" in self._entity_data:
            return bool(self._entity_data.get('pistol_upgrade_burstfire'))
        return bool(0)

    @property
    def rapidfire_upgrade_reflexsight(self):
        if "rapidfire_upgrade_reflexsight" in self._entity_data:
            return bool(self._entity_data.get('rapidfire_upgrade_reflexsight'))
        return bool(0)

    @property
    def rapidfire_upgrade_lasersight(self):
        if "rapidfire_upgrade_lasersight" in self._entity_data:
            return bool(self._entity_data.get('rapidfire_upgrade_lasersight'))
        return bool(0)

    @property
    def rapidfire_upgrade_extended_magazine(self):
        if "rapidfire_upgrade_extended_magazine" in self._entity_data:
            return bool(self._entity_data.get('rapidfire_upgrade_extended_magazine'))
        return bool(0)

    @property
    def shotgun_upgrade_autoloader(self):
        if "shotgun_upgrade_autoloader" in self._entity_data:
            return bool(self._entity_data.get('shotgun_upgrade_autoloader'))
        return bool(0)

    @property
    def shotgun_upgrade_grenade(self):
        if "shotgun_upgrade_grenade" in self._entity_data:
            return bool(self._entity_data.get('shotgun_upgrade_grenade'))
        return bool(0)

    @property
    def shotgun_upgrade_lasersight(self):
        if "shotgun_upgrade_lasersight" in self._entity_data:
            return bool(self._entity_data.get('shotgun_upgrade_lasersight'))
        return bool(0)

    @property
    def shotgun_upgrade_quickfire(self):
        if "shotgun_upgrade_quickfire" in self._entity_data:
            return bool(self._entity_data.get('shotgun_upgrade_quickfire'))
        return bool(0)


class point_hlvr_strip_player(Targetname):
    pass

    icon_sprite =  "editor/point_hlvr_strip_player.vmat"

    @property
    def EnablePhysicsDelay(self):
        if "EnablePhysicsDelay" in self._entity_data:
            return float(self._entity_data.get('EnablePhysicsDelay'))
        return float(1)

    @property
    def DissolveItemsDelay(self):
        if "DissolveItemsDelay" in self._entity_data:
            return float(self._entity_data.get('DissolveItemsDelay'))
        return float(3)

    @property
    def ItemVelocity(self):
        if "ItemVelocity" in self._entity_data:
            return float(self._entity_data.get('ItemVelocity'))
        return float(20)


class item_item_crate(BasePropPhysics):
    pass

    @property
    def ItemClass(self):
        if "ItemClass" in self._entity_data:
            return self._entity_data.get('ItemClass')
        return "item_hlvr_clip_energygun"

    @property
    def CrateAppearance(self):
        if "CrateAppearance" in self._entity_data:
            return self._entity_data.get('CrateAppearance')
        return "2"

    @property
    def ItemCount(self):
        if "ItemCount" in self._entity_data:
            return int(self._entity_data.get('ItemCount'))
        return int(1)

    @property
    def SpecificResupply(self):
        if "SpecificResupply" in self._entity_data:
            return self._entity_data.get('SpecificResupply')
        return ""

    @property
    def ammobalancing_removable(self):
        if "ammobalancing_removable" in self._entity_data:
            return self._entity_data.get('ammobalancing_removable')
        return "0"


class item_hlvr_crafting_currency_large(BasePropPhysics):
    pass

    @property
    def remove_over_amount(self):
        if "remove_over_amount" in self._entity_data:
            return int(self._entity_data.get('remove_over_amount'))
        return int(0)


class item_hlvr_crafting_currency_small(BasePropPhysics):
    pass

    @property
    def remove_over_amount(self):
        if "remove_over_amount" in self._entity_data:
            return int(self._entity_data.get('remove_over_amount'))
        return int(0)


class item_hlvr_prop_discovery(BasePropPhysics):
    pass


class item_hlvr_prop_ammobag(Item):
    pass


class prop_hlvr_crafting_station(Targetname):
    pass

    @property
    def hacking_plug(self):
        if "hacking_plug" in self._entity_data:
            return self._entity_data.get('hacking_plug')
        return ""

    @property
    def is_powered(self):
        if "is_powered" in self._entity_data:
            return bool(self._entity_data.get('is_powered'))
        return bool(None)

    @property
    def lightmapstatic(self):
        if "lightmapstatic" in self._entity_data:
            return self._entity_data.get('lightmapstatic')
        return "0"


class trigger_crafting_station_object_placement(Trigger):
    pass


class item_healthcharger(Targetname):
    pass

    @property
    def start_with_vial(self):
        if "start_with_vial" in self._entity_data:
            return bool(self._entity_data.get('start_with_vial'))
        return bool(1)

    @property
    def vial_level(self):
        if "vial_level" in self._entity_data:
            return float(self._entity_data.get('vial_level'))
        return float(1)

    @property
    def lightmapstatic(self):
        if "lightmapstatic" in self._entity_data:
            return self._entity_data.get('lightmapstatic')
        return "0"


class item_combine_console(Targetname):
    pass

    @property
    def hacking_plug(self):
        if "hacking_plug" in self._entity_data:
            return self._entity_data.get('hacking_plug')
        return ""

    @property
    def rack0_active(self):
        if "rack0_active" in self._entity_data:
            return bool(self._entity_data.get('rack0_active'))
        return bool(None)

    @property
    def rack1_active(self):
        if "rack1_active" in self._entity_data:
            return bool(self._entity_data.get('rack1_active'))
        return bool(None)

    @property
    def rack2_active(self):
        if "rack2_active" in self._entity_data:
            return bool(self._entity_data.get('rack2_active'))
        return bool(None)

    @property
    def rack3_active(self):
        if "rack3_active" in self._entity_data:
            return bool(self._entity_data.get('rack3_active'))
        return bool(None)

    @property
    def tank0_start_missing(self):
        if "tank0_start_missing" in self._entity_data:
            return bool(self._entity_data.get('tank0_start_missing'))
        return bool(None)

    @property
    def tank1_start_missing(self):
        if "tank1_start_missing" in self._entity_data:
            return bool(self._entity_data.get('tank1_start_missing'))
        return bool(None)

    @property
    def tank2_start_missing(self):
        if "tank2_start_missing" in self._entity_data:
            return bool(self._entity_data.get('tank2_start_missing'))
        return bool(None)

    @property
    def tank3_start_missing(self):
        if "tank3_start_missing" in self._entity_data:
            return bool(self._entity_data.get('tank3_start_missing'))
        return bool(None)

    @property
    def objective_model(self):
        if "objective_model" in self._entity_data:
            return self._entity_data.get('objective_model')
        return ""

    @property
    def lightmapstatic(self):
        if "lightmapstatic" in self._entity_data:
            return self._entity_data.get('lightmapstatic')
        return "0"


class item_combine_tank_locker(prop_animinteractable):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/props_combine/combine_lockers/combine_locker_standing.vmdl"

    @property
    def starting_tanks(self):
        if "starting_tanks" in self._entity_data:
            return self._entity_data.get('starting_tanks')
        return "0"


class hlvr_vault_tractor_beam_console(Targetname):
    pass


class info_hlvr_toner_port(Targetname, EnableDisable):
    pass

    @property
    def StartPortVisible(self):
        if "StartPortVisible" in self._entity_data:
            return bool(self._entity_data.get('StartPortVisible'))
        return bool(0)

    @property
    def StartVisible(self):
        if "StartVisible" in self._entity_data:
            return bool(self._entity_data.get('StartVisible'))
        return bool(0)

    @property
    def initial_orientation(self):
        if "initial_orientation" in self._entity_data:
            return self._entity_data.get('initial_orientation')
        return "0"

    @property
    def desired_orientation(self):
        if "desired_orientation" in self._entity_data:
            return self._entity_data.get('desired_orientation')
        return "2"


class info_hlvr_toner_path(Targetname):
    pass

    icon_sprite =  "editor/info_hlvr_toner_path.vmat"

    @property
    def first_path_node(self):
        if "first_path_node" in self._entity_data:
            return self._entity_data.get('first_path_node')
        return ""

    @property
    def start_entity(self):
        if "start_entity" in self._entity_data:
            return self._entity_data.get('start_entity')
        return ""

    @property
    def end_entity(self):
        if "end_entity" in self._entity_data:
            return self._entity_data.get('end_entity')
        return ""


class info_hlvr_toner_path_node(Targetname):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return None

    @property
    def is_spline_node(self):
        if "is_spline_node" in self._entity_data:
            return bool(self._entity_data.get('is_spline_node'))
        return bool(1)

    @property
    def inset_distance(self):
        if "inset_distance" in self._entity_data:
            return float(self._entity_data.get('inset_distance'))
        return float(-1)


class info_hlvr_toner_junction(Targetname):
    pass

    @property
    def junction_toplogy(self):
        if "junction_toplogy" in self._entity_data:
            return self._entity_data.get('junction_toplogy')
        return "0"

    @property
    def junction_orientation(self):
        if "junction_orientation" in self._entity_data:
            return self._entity_data.get('junction_orientation')
        return "0"

    @property
    def inset_distance(self):
        if "inset_distance" in self._entity_data:
            return float(self._entity_data.get('inset_distance'))
        return float(-1)

    @property
    def connection_0(self):
        if "connection_0" in self._entity_data:
            return self._entity_data.get('connection_0')
        return ""

    @property
    def connection_1(self):
        if "connection_1" in self._entity_data:
            return self._entity_data.get('connection_1')
        return ""

    @property
    def connection_2(self):
        if "connection_2" in self._entity_data:
            return self._entity_data.get('connection_2')
        return ""

    @property
    def connection_3(self):
        if "connection_3" in self._entity_data:
            return self._entity_data.get('connection_3')
        return ""


class info_hlvr_offscreen_particle_texture(Targetname):
    pass

    icon_sprite =  "editor/info_hlvr_offscreen_particle_texture.vmat"

    @property
    def StartDisabled(self):
        if "StartDisabled" in self._entity_data:
            return bool(self._entity_data.get('StartDisabled'))
        return bool(1)

    @property
    def effect_name(self):
        if "effect_name" in self._entity_data:
            return self._entity_data.get('effect_name')
        return None

    @property
    def control_point_a_index(self):
        if "control_point_a_index" in self._entity_data:
            return int(self._entity_data.get('control_point_a_index'))
        return int(0)

    @property
    def input_a_remap(self):
        if "input_a_remap" in self._entity_data:
            return self._entity_data.get('input_a_remap')
        return "0"

    @property
    def input_a_remap_param_s(self):
        if "input_a_remap_param_s" in self._entity_data:
            return float(self._entity_data.get('input_a_remap_param_s'))
        return float(1.0)

    @property
    def input_a_remap_param_t(self):
        if "input_a_remap_param_t" in self._entity_data:
            return float(self._entity_data.get('input_a_remap_param_t'))
        return float(0.0)

    @property
    def control_point_b_index(self):
        if "control_point_b_index" in self._entity_data:
            return int(self._entity_data.get('control_point_b_index'))
        return int(1)

    @property
    def input_b_remap(self):
        if "input_b_remap" in self._entity_data:
            return self._entity_data.get('input_b_remap')
        return "0"

    @property
    def input_b_remap_param_s(self):
        if "input_b_remap_param_s" in self._entity_data:
            return float(self._entity_data.get('input_b_remap_param_s'))
        return float(1.0)

    @property
    def input_b_remap_param_t(self):
        if "input_b_remap_param_t" in self._entity_data:
            return float(self._entity_data.get('input_b_remap_param_t'))
        return float(0.0)

    @property
    def control_point_c_index(self):
        if "control_point_c_index" in self._entity_data:
            return int(self._entity_data.get('control_point_c_index'))
        return int(-1)

    @property
    def input_c_remap(self):
        if "input_c_remap" in self._entity_data:
            return self._entity_data.get('input_c_remap')
        return "0"

    @property
    def input_c_remap_param_s(self):
        if "input_c_remap_param_s" in self._entity_data:
            return float(self._entity_data.get('input_c_remap_param_s'))
        return float(1.0)

    @property
    def input_c_remap_param_t(self):
        if "input_c_remap_param_t" in self._entity_data:
            return float(self._entity_data.get('input_c_remap_param_t'))
        return float(0.0)

    @property
    def control_point_d_index(self):
        if "control_point_d_index" in self._entity_data:
            return int(self._entity_data.get('control_point_d_index'))
        return int(-1)

    @property
    def input_d_remap(self):
        if "input_d_remap" in self._entity_data:
            return self._entity_data.get('input_d_remap')
        return "0"

    @property
    def input_d_remap_param_s(self):
        if "input_d_remap_param_s" in self._entity_data:
            return float(self._entity_data.get('input_d_remap_param_s'))
        return float(1.0)

    @property
    def input_d_remap_param_t(self):
        if "input_d_remap_param_t" in self._entity_data:
            return float(self._entity_data.get('input_d_remap_param_t'))
        return float(0.0)

    @property
    def target_entity(self):
        if "target_entity" in self._entity_data:
            return self._entity_data.get('target_entity')
        return ""

    @property
    def material_param(self):
        if "material_param" in self._entity_data:
            return self._entity_data.get('material_param')
        return ""

    @property
    def clear_color(self):
        if "clear_color" in self._entity_data:
            return parse_int_vector(self._entity_data.get('clear_color'))
        return parse_int_vector("128 128 128")

    @property
    def texture_resolution(self):
        if "texture_resolution" in self._entity_data:
            return int(self._entity_data.get('texture_resolution'))
        return int(512)

    @property
    def visible_range_check(self):
        if "visible_range_check" in self._entity_data:
            return float(self._entity_data.get('visible_range_check'))
        return float(300)


class trigger_zap_module(Trigger):
    pass


class env_gradient_fog(Targetname, EnableDisable):
    pass

    icon_sprite =  "materials/editor/env_fog_controller.vmat"

    @property
    def fogstart(self):
        if "fogstart" in self._entity_data:
            return float(self._entity_data.get('fogstart'))
        return float(0.0)

    @property
    def fogend(self):
        if "fogend" in self._entity_data:
            return float(self._entity_data.get('fogend'))
        return float(4000.0)

    @property
    def fogstartheight(self):
        if "fogstartheight" in self._entity_data:
            return float(self._entity_data.get('fogstartheight'))
        return float(0.0)

    @property
    def fogendheight(self):
        if "fogendheight" in self._entity_data:
            return float(self._entity_data.get('fogendheight'))
        return float(200.0)

    @property
    def fogmaxopacity(self):
        if "fogmaxopacity" in self._entity_data:
            return float(self._entity_data.get('fogmaxopacity'))
        return float(0.5)

    @property
    def fogcolor(self):
        if "fogcolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('fogcolor'))
        return parse_int_vector("255 255 255")

    @property
    def fogstrength(self):
        if "fogstrength" in self._entity_data:
            return float(self._entity_data.get('fogstrength'))
        return float(1.0)

    @property
    def fogfalloffexponent(self):
        if "fogfalloffexponent" in self._entity_data:
            return float(self._entity_data.get('fogfalloffexponent'))
        return float(2.0)

    @property
    def fogverticalexponent(self):
        if "fogverticalexponent" in self._entity_data:
            return float(self._entity_data.get('fogverticalexponent'))
        return float(1.0)

    @property
    def fadetime(self):
        if "fadetime" in self._entity_data:
            return float(self._entity_data.get('fadetime'))
        return float(1.0)


class env_spherical_vignette(Targetname, EnableDisable):
    pass

    icon_sprite =  "materials/editor/env_fog_controller.vmat"

    @property
    def vignettestart(self):
        if "vignettestart" in self._entity_data:
            return float(self._entity_data.get('vignettestart'))
        return float(30.0)

    @property
    def vignetteend(self):
        if "vignetteend" in self._entity_data:
            return float(self._entity_data.get('vignetteend'))
        return float(120.0)

    @property
    def farz(self):
        if "farz" in self._entity_data:
            return float(self._entity_data.get('farz'))
        return float(0.0)

    @property
    def vignettemaxopacity(self):
        if "vignettemaxopacity" in self._entity_data:
            return float(self._entity_data.get('vignettemaxopacity'))
        return float(1.0)

    @property
    def vignettecolor(self):
        if "vignettecolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('vignettecolor'))
        return parse_int_vector("0 0 0")

    @property
    def vignettestrength(self):
        if "vignettestrength" in self._entity_data:
            return float(self._entity_data.get('vignettestrength'))
        return float(1.0)

    @property
    def vignettefalloffexponent(self):
        if "vignettefalloffexponent" in self._entity_data:
            return float(self._entity_data.get('vignettefalloffexponent'))
        return float(1.0)

    @property
    def fadetime(self):
        if "fadetime" in self._entity_data:
            return float(self._entity_data.get('fadetime'))
        return float(1.0)


class env_cubemap_fog(Targetname, EnableDisable):
    pass

    icon_sprite =  "materials/editor/env_cubemap_fog.vmat"

    @property
    def cubemapfogtexture(self):
        if "cubemapfogtexture" in self._entity_data:
            return self._entity_data.get('cubemapfogtexture')
        return "materials/skybox/tests/src/light_test_sky_sunset.vtex"

    @property
    def cubemapfoglodbiase(self):
        if "cubemapfoglodbiase" in self._entity_data:
            return float(self._entity_data.get('cubemapfoglodbiase'))
        return float(0.5)

    @property
    def cubemapfogstartdistance(self):
        if "cubemapfogstartdistance" in self._entity_data:
            return float(self._entity_data.get('cubemapfogstartdistance'))
        return float(0.0)

    @property
    def cubemapfogenddistance(self):
        if "cubemapfogenddistance" in self._entity_data:
            return float(self._entity_data.get('cubemapfogenddistance'))
        return float(6000.0)

    @property
    def cubemapfogfalloffexponent(self):
        if "cubemapfogfalloffexponent" in self._entity_data:
            return float(self._entity_data.get('cubemapfogfalloffexponent'))
        return float(2.0)

    @property
    def cubemapfogheightwidth(self):
        if "cubemapfogheightwidth" in self._entity_data:
            return float(self._entity_data.get('cubemapfogheightwidth'))
        return float(0.0)

    @property
    def cubemapfogheightstart(self):
        if "cubemapfogheightstart" in self._entity_data:
            return float(self._entity_data.get('cubemapfogheightstart'))
        return float(2000.0)

    @property
    def cubemapfogheightexponent(self):
        if "cubemapfogheightexponent" in self._entity_data:
            return float(self._entity_data.get('cubemapfogheightexponent'))
        return float(2.0)


class trigger_resource_analyzer(Trigger):
    pass

    @property
    def data_name(self):
        if "data_name" in self._entity_data:
            return self._entity_data.get('data_name')
        return ""


class trigger_player_out_of_ammo(Trigger):
    pass

    @property
    def ammotype(self):
        if "ammotype" in self._entity_data:
            return self._entity_data.get('ammotype')
        return "0"


class npc_turret_citizen(Targetname):
    pass

    @property
    def battery_placement_trigger(self):
        if "battery_placement_trigger" in self._entity_data:
            return self._entity_data.get('battery_placement_trigger')
        return ""

    @property
    def spawnflags(self):
        flags = []
        if "spawnflags" in self._entity_data:
            value = self._entity_data.get("spawnflags", None)
            for name, (key, _) in {'Autostart': (32, 0), 'Start Inactive': (64, 0), 'Fast Retire': (128, 0),
                                   'Out of Ammo': (256, 0), 'Citizen modified (Friendly)': (512, 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def SkinNumber(self):
        if "SkinNumber" in self._entity_data:
            return int(self._entity_data.get('SkinNumber'))
        return int(0)


class trigger_turret_citizen_battery(trigger_multiple):
    pass


class trigger_blind_zombie_crash(Trigger):
    pass


class trigger_blind_zombie_sound_area(Trigger):
    pass

    @property
    def scripted_sequence_name(self):
        if "scripted_sequence_name" in self._entity_data:
            return self._entity_data.get('scripted_sequence_name')
        return ""

    @property
    def target_sound_entity_name(self):
        if "target_sound_entity_name" in self._entity_data:
            return self._entity_data.get('target_sound_entity_name')
        return ""

    @property
    def fire_output_immediately(self):
        if "fire_output_immediately" in self._entity_data:
            return bool(self._entity_data.get('fire_output_immediately'))
        return bool(0)


class func_xen_membrane_barrier(func_brush):
    pass


class trigger_blind_zombie_wander_area(Trigger):
    pass

    @property
    def companion_trigger(self):
        if "companion_trigger" in self._entity_data:
            return self._entity_data.get('companion_trigger')
        return ""


class trigger_xen_membrane_door(Trigger):
    pass

    @property
    def portal_membrane_name(self):
        if "portal_membrane_name" in self._entity_data:
            return self._entity_data.get('portal_membrane_name')
        return ""

    @property
    def scripted_sequence_name(self):
        if "scripted_sequence_name" in self._entity_data:
            return self._entity_data.get('scripted_sequence_name')
        return ""

    @property
    def portal_collision_name(self):
        if "portal_collision_name" in self._entity_data:
            return self._entity_data.get('portal_collision_name')
        return ""

    @property
    def portal_nav_blocker_name(self):
        if "portal_nav_blocker_name" in self._entity_data:
            return self._entity_data.get('portal_nav_blocker_name')
        return ""

    @property
    def portal_name(self):
        if "portal_name" in self._entity_data:
            return self._entity_data.get('portal_name')
        return ""


class trigger_player_peephole(Trigger):
    pass

    @property
    def peephole_axis(self):
        if "peephole_axis" in self._entity_data:
            return self._entity_data.get('peephole_axis')
        return "0 0 0, 1 0 0"


class npc_zombie_blind(BaseNPC, BaseZombie):
    pass

    @property
    def freezer_target_name(self):
        if "freezer_target_name" in self._entity_data:
            return self._entity_data.get('freezer_target_name')
        return ""


class item_hlvr_prop_battery(BasePropPhysics):
    pass

    @property
    def battery_level(self):
        if "battery_level" in self._entity_data:
            return float(self._entity_data.get('battery_level'))
        return float(1)

    @property
    def show_battery_level(self):
        if "show_battery_level" in self._entity_data:
            return self._entity_data.get('show_battery_level')
        return "0"


class item_hlvr_health_station_vial(BasePropPhysics):
    pass

    @property
    def vial_level(self):
        if "vial_level" in self._entity_data:
            return float(self._entity_data.get('vial_level'))
        return float(1)


class item_hlvr_combine_console_tank(BasePropPhysics):
    pass


class logic_distance_autosave(Targetname):
    pass

    icon_sprite =  "editor/logic_autosave.vmat"

    @property
    def TargetEntityName(self):
        if "TargetEntityName" in self._entity_data:
            return self._entity_data.get('TargetEntityName')
        return ""

    @property
    def DistanceToPlayer(self):
        if "DistanceToPlayer" in self._entity_data:
            return float(self._entity_data.get('DistanceToPlayer'))
        return float(128)

    @property
    def NewLevelUnit(self):
        if "NewLevelUnit" in self._entity_data:
            return bool(self._entity_data.get('NewLevelUnit'))
        return bool(0)

    @property
    def CheckCough(self):
        if "CheckCough" in self._entity_data:
            return bool(self._entity_data.get('CheckCough'))
        return bool(0)


class logic_multilight_proxy(Targetname, Parentname):
    pass

    @property
    def light_name(self):
        if "light_name" in self._entity_data:
            return self._entity_data.get('light_name')
        return ""

    @property
    def light_class(self):
        if "light_class" in self._entity_data:
            return self._entity_data.get('light_class')
        return ""

    @property
    def light_radius(self):
        if "light_radius" in self._entity_data:
            return float(self._entity_data.get('light_radius'))
        return float(0)

    @property
    def brightness_delta(self):
        if "brightness_delta" in self._entity_data:
            return float(self._entity_data.get('brightness_delta'))
        return float(0.05)

    @property
    def screen_fade(self):
        if "screen_fade" in self._entity_data:
            return bool(self._entity_data.get('screen_fade'))
        return bool(0)


class point_lightmodifier(Targetname, Parentname):
    pass

    @property
    def light_names(self):
        if "light_names" in self._entity_data:
            return self._entity_data.get('light_names')
        return ""

    @property
    def filter_name(self):
        if "filter_name" in self._entity_data:
            return self._entity_data.get('filter_name')
        return ""

    @property
    def filter_radius(self):
        if "filter_radius" in self._entity_data:
            return float(self._entity_data.get('filter_radius'))
        return float(512)

    @property
    def light_level(self):
        if "light_level" in self._entity_data:
            return float(self._entity_data.get('light_level'))
        return float(2.5)

    @property
    def light_time_in(self):
        if "light_time_in" in self._entity_data:
            return float(self._entity_data.get('light_time_in'))
        return float(1.0)

    @property
    def light_time_out(self):
        if "light_time_out" in self._entity_data:
            return float(self._entity_data.get('light_time_out'))
        return float(1.0)

    @property
    def light_noise_interval(self):
        if "light_noise_interval" in self._entity_data:
            return float(self._entity_data.get('light_noise_interval'))
        return float(0.1)

    @property
    def light_noise_min(self):
        if "light_noise_min" in self._entity_data:
            return float(self._entity_data.get('light_noise_min'))
        return float(0.1)

    @property
    def light_noise_max(self):
        if "light_noise_max" in self._entity_data:
            return float(self._entity_data.get('light_noise_max'))
        return float(0.5)

    @property
    def effect_name(self):
        if "effect_name" in self._entity_data:
            return self._entity_data.get('effect_name')
        return None

    @property
    def effect_sound_name(self):
        if "effect_sound_name" in self._entity_data:
            return self._entity_data.get('effect_sound_name')
        return ""

    @property
    def effect_target_name(self):
        if "effect_target_name" in self._entity_data:
            return self._entity_data.get('effect_target_name')
        return ""


class prop_handpose(BasePropPhysics, EnableDisable):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/props/max/handposes/handpose_flatpalm.vmdl"

    @property
    def DistanceMin(self):
        if "DistanceMin" in self._entity_data:
            return float(self._entity_data.get('DistanceMin'))
        return float(4)

    @property
    def DistanceMax(self):
        if "DistanceMax" in self._entity_data:
            return float(self._entity_data.get('DistanceMax'))
        return float(6)

    @property
    def DistanceBias(self):
        if "DistanceBias" in self._entity_data:
            return float(self._entity_data.get('DistanceBias'))
        return float(0.3)

    @property
    def DisengageDistance(self):
        if "DisengageDistance" in self._entity_data:
            return float(self._entity_data.get('DisengageDistance'))
        return float(16)

    @property
    def UseProximityBone(self):
        if "UseProximityBone" in self._entity_data:
            return bool(self._entity_data.get('UseProximityBone'))
        return bool(0)

    @property
    def AddToSpatialPartition(self):
        if "AddToSpatialPartition" in self._entity_data:
            return bool(self._entity_data.get('AddToSpatialPartition'))
        return bool(1)

    @property
    def DeleteAfterSpawn(self):
        if "DeleteAfterSpawn" in self._entity_data:
            return bool(self._entity_data.get('DeleteAfterSpawn'))
        return bool(0)

    @property
    def AutoGrip(self):
        if "AutoGrip" in self._entity_data:
            return bool(self._entity_data.get('AutoGrip'))
        return bool(0)

    @property
    def Extent(self):
        if "Extent" in self._entity_data:
            return parse_int_vector(self._entity_data.get('Extent'))
        return parse_int_vector("")

    @property
    def IgnoreHand(self):
        if "IgnoreHand" in self._entity_data:
            return self._entity_data.get('IgnoreHand')
        return "-1"


class skybox_reference(Targetname):
    pass

    @property
    def targetMapName(self):
        if "targetMapName" in self._entity_data:
            return self._entity_data.get('targetMapName')
        return None

    @property
    def fixupNames(self):
        if "fixupNames" in self._entity_data:
            return bool(self._entity_data.get('fixupNames'))
        return bool(0)

    @property
    def worldGroupID(self):
        if "worldGroupID" in self._entity_data:
            return self._entity_data.get('worldGroupID')
        return "skyboxWorldGroup0"


class ghost_speaker(npc_furniture):
    pass

    icon_sprite =  "editor/ghost_speaker.vmat"

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/props/choreo/ghost_speaker.vmdl"


class ghost_actor(npc_furniture):
    pass

    icon_sprite =  "editor/ghost_actor.vmat"

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/props/choreo/ghost_actor.vmdl"


class prop_russell_headset(BasePropPhysics):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/props/choreo_office/headset_prop.vmdl"


class prop_physics_interactive(prop_physics):
    pass


class trigger_physics(Trigger):
    pass

    @property
    def StartDisabled(self):
        if "StartDisabled" in self._entity_data:
            return bool(self._entity_data.get('StartDisabled'))
        return bool(0)

    @property
    def filtername(self):
        if "filtername" in self._entity_data:
            return self._entity_data.get('filtername')
        return "Filter to use to see if activator triggers me. See filter_activator_name for more explanation."

    @property
    def SetGravityScale(self):
        if "SetGravityScale" in self._entity_data:
            return float(self._entity_data.get('SetGravityScale'))
        return float(1.0)

    @property
    def SetVelocityLimit(self):
        if "SetVelocityLimit" in self._entity_data:
            return float(self._entity_data.get('SetVelocityLimit'))
        return float(-1.0)

    @property
    def SetVelocityDamping(self):
        if "SetVelocityDamping" in self._entity_data:
            return float(self._entity_data.get('SetVelocityDamping'))
        return float(0.0)

    @property
    def SetAngVelocityLimit(self):
        if "SetAngVelocityLimit" in self._entity_data:
            return float(self._entity_data.get('SetAngVelocityLimit'))
        return float(-1.0)

    @property
    def SetAngVelocityDamping(self):
        if "SetAngVelocityDamping" in self._entity_data:
            return float(self._entity_data.get('SetAngVelocityDamping'))
        return float(0.0)

    @property
    def SetLinearForce(self):
        if "SetLinearForce" in self._entity_data:
            return float(self._entity_data.get('SetLinearForce'))
        return float(0.0)

    @property
    def LinearForcePointAt(self):
        if "LinearForcePointAt" in self._entity_data:
            return parse_int_vector(self._entity_data.get('LinearForcePointAt'))
        return parse_int_vector("0 0 0")

    @property
    def CollapseToForcePoint(self):
        if "CollapseToForcePoint" in self._entity_data:
            return bool(self._entity_data.get('CollapseToForcePoint'))
        return bool(1)

    @property
    def SetDampingRatio(self):
        if "SetDampingRatio" in self._entity_data:
            return float(self._entity_data.get('SetDampingRatio'))
        return float(1.0)

    @property
    def SetFrequency(self):
        if "SetFrequency" in self._entity_data:
            return float(self._entity_data.get('SetFrequency'))
        return float(0.1)

    @property
    def ConvertToDebrisWhenPossible(self):
        if "ConvertToDebrisWhenPossible" in self._entity_data:
            return self._entity_data.get('ConvertToDebrisWhenPossible')
        return "0"


class info_teleport_magnet(Targetname, Parentname):
    pass

    @property
    def magnet_radius(self):
        if "magnet_radius" in self._entity_data:
            return float(self._entity_data.get('magnet_radius'))
        return float(0.0)

    @property
    def start_enabled(self):
        if "start_enabled" in self._entity_data:
            return bool(self._entity_data.get('start_enabled'))
        return bool(1)


class func_combine_barrier(func_brush):
    pass

    @property
    def effect_name(self):
        if "effect_name" in self._entity_data:
            return self._entity_data.get('effect_name')
        return None

    @property
    def effect_interpenetrate_name(self):
        if "effect_interpenetrate_name" in self._entity_data:
            return self._entity_data.get('effect_interpenetrate_name')
        return "particles/combine_tech/combine_filter_field_interpenetrate.vpcf"

    @property
    def filtername(self):
        if "filtername" in self._entity_data:
            return self._entity_data.get('filtername')
        return None

    @property
    def barrier_state(self):
        if "barrier_state" in self._entity_data:
            return self._entity_data.get('barrier_state')
        return "BARRIER_COMBINE_BLOCKER"


class func_electrified_volume(func_brush):
    pass

    @property
    def effect_name(self):
        if "effect_name" in self._entity_data:
            return self._entity_data.get('effect_name')
        return None

    @property
    def effect_interpenetrate_name(self):
        if "effect_interpenetrate_name" in self._entity_data:
            return self._entity_data.get('effect_interpenetrate_name')
        return "particles/environment/player_hand_electricity.vpcf"

    @property
    def effect_zap_name(self):
        if "effect_zap_name" in self._entity_data:
            return self._entity_data.get('effect_zap_name')
        return ""

    @property
    def effect_zap_source(self):
        if "effect_zap_source" in self._entity_data:
            return self._entity_data.get('effect_zap_source')
        return ""


class hl_vr_texture_based_animatable(Targetname):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return ""

    @property
    def fps(self):
        if "fps" in self._entity_data:
            return float(self._entity_data.get('fps'))
        return float(60)

    @property
    def loop(self):
        if "loop" in self._entity_data:
            return bool(self._entity_data.get('loop'))
        return bool(0)

    @property
    def texture_based_animation_preview_sequence(self):
        if "texture_based_animation_preview_sequence" in self._entity_data:
            return float(self._entity_data.get('texture_based_animation_preview_sequence'))
        return float(0)

    @property
    def texture_based_animation_position_keys(self):
        if "texture_based_animation_position_keys" in self._entity_data:
            return self._entity_data.get('texture_based_animation_position_keys')
        return None

    @property
    def texture_based_animation_rotation_keys(self):
        if "texture_based_animation_rotation_keys" in self._entity_data:
            return self._entity_data.get('texture_based_animation_rotation_keys')
        return None

    @property
    def anim_bounds_min(self):
        if "anim_bounds_min" in self._entity_data:
            return parse_int_vector(self._entity_data.get('anim_bounds_min'))
        return parse_int_vector("0 0 0")

    @property
    def anim_bounds_max(self):
        if "anim_bounds_max" in self._entity_data:
            return parse_int_vector(self._entity_data.get('anim_bounds_max'))
        return parse_int_vector("0 0 0")

    @property
    def lightingorigin(self):
        if "lightingorigin" in self._entity_data:
            return self._entity_data.get('lightingorigin')
        return ""

    @property
    def StartDisabled(self):
        if "StartDisabled" in self._entity_data:
            return bool(self._entity_data.get('StartDisabled'))
        return bool(0)


class info_hlvr_holo_hacking_plug(Targetname, Parentname, EnableDisable):
    pass

    @property
    def PuzzleSpawnTarget(self):
        if "PuzzleSpawnTarget" in self._entity_data:
            return self._entity_data.get('PuzzleSpawnTarget')
        return ""

    @property
    def PuzzleType(self):
        if "PuzzleType" in self._entity_data:
            return self._entity_data.get('PuzzleType')
        return "0"

    @property
    def IntroVariation(self):
        if "IntroVariation" in self._entity_data:
            return int(self._entity_data.get('IntroVariation'))
        return int(0)

    @property
    def ShowIntro(self):
        if "ShowIntro" in self._entity_data:
            return bool(self._entity_data.get('ShowIntro'))
        return bool(1)

    @property
    def HackDifficultyName(self):
        if "HackDifficultyName" in self._entity_data:
            return self._entity_data.get('HackDifficultyName')
        return "Medium"

    @property
    def StartHacked(self):
        if "StartHacked" in self._entity_data:
            return int(self._entity_data.get('StartHacked'))
        return int(0)


class info_hlvr_holo_hacking_spawn_target(Targetname, Parentname):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/hacking/holo_hacking_sphere_prop_editor.vmdl"

    @property
    def Radius(self):
        if "Radius" in self._entity_data:
            return float(self._entity_data.get('Radius'))
        return float(5)

    @property
    def Hidden(self):
        if "Hidden" in self._entity_data:
            return bool(self._entity_data.get('Hidden'))
        return bool(0)


class combine_attached_armor_prop(Targetname, Parentname):
    pass

    @property
    def StartDisabled(self):
        if "StartDisabled" in self._entity_data:
            return bool(self._entity_data.get('StartDisabled'))
        return bool(0)

    @property
    def SubtleEffects(self):
        if "SubtleEffects" in self._entity_data:
            return bool(self._entity_data.get('SubtleEffects'))
        return bool(1)


class hlvr_piano(Targetname):
    pass


class func_dry_erase_board(Targetname, RenderFields):
    pass

    @property
    def StampName(self):
        if "StampName" in self._entity_data:
            return self._entity_data.get('StampName')
        return ""

    @property
    def DisableFrontOfBoardCheck(self):
        if "DisableFrontOfBoardCheck" in self._entity_data:
            return bool(self._entity_data.get('DisableFrontOfBoardCheck'))
        return bool(0)


class prop_dry_erase_marker(Targetname, Studiomodel):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/props/alyx_hideout/dry_erase_marker.vmdl"

    @property
    def MarkerType(self):
        if "MarkerType" in self._entity_data:
            return self._entity_data.get('MarkerType')
        return "0"

    @property
    def MarkerTipSize(self):
        if "MarkerTipSize" in self._entity_data:
            return float(self._entity_data.get('MarkerTipSize'))
        return float(10)

    @property
    def rendercolor(self):
        if "rendercolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('rendercolor'))
        return parse_int_vector("255 0 0")

    @property
    def MarkerColor(self):
        if "MarkerColor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('MarkerColor'))
        return parse_int_vector("255 0 0")

    @property
    def interactAs(self):
        if "interactAs" in self._entity_data:
            return self._entity_data.get('interactAs')
        return ""


class XenFoliageBase(Targetname, Parentname):
    pass

    @property
    def interaction_trigger(self):
        if "interaction_trigger" in self._entity_data:
            return self._entity_data.get('interaction_trigger')
        return ""

    @property
    def interaction_inner_trigger(self):
        if "interaction_inner_trigger" in self._entity_data:
            return self._entity_data.get('interaction_inner_trigger')
        return ""

    @property
    def noise_generator(self):
        if "noise_generator" in self._entity_data:
            return self._entity_data.get('noise_generator')
        return ""

    @property
    def soundName(self):
        if "soundName" in self._entity_data:
            return self._entity_data.get('soundName')
        return ""

    @property
    def interactive_distance(self):
        if "interactive_distance" in self._entity_data:
            return float(self._entity_data.get('interactive_distance'))
        return float(128)

    @property
    def interactive_close(self):
        if "interactive_close" in self._entity_data:
            return float(self._entity_data.get('interactive_close'))
        return float(24)

    @property
    def interaction_attachment_name(self):
        if "interaction_attachment_name" in self._entity_data:
            return self._entity_data.get('interaction_attachment_name')
        return ""


class xen_foliage(XenFoliageBase):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return ""

    @property
    def max_health_category(self):
        if "max_health_category" in self._entity_data:
            return self._entity_data.get('max_health_category')
        return "0"

    @property
    def nav_ignore(self):
        if "nav_ignore" in self._entity_data:
            return bool(self._entity_data.get('nav_ignore'))
        return bool(0)

    @property
    def randomize_start(self):
        if "randomize_start" in self._entity_data:
            return bool(self._entity_data.get('randomize_start'))
        return bool(0)


class xen_hearing_flower(Targetname, Parentname):
    pass


class xen_foliage_bloater(XenFoliageBase):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/props/xen_infestation/boomerplant_01.vmdl"

    @property
    def explosion_radius(self):
        if "explosion_radius" in self._entity_data:
            return int(self._entity_data.get('explosion_radius'))
        return int(128)

    @property
    def explosion_magnitude(self):
        if "explosion_magnitude" in self._entity_data:
            return int(self._entity_data.get('explosion_magnitude'))
        return int(80)

    @property
    def ignore_entity(self):
        if "ignore_entity" in self._entity_data:
            return self._entity_data.get('ignore_entity')
        return ""


class xen_foliage_grenade_spawner(XenFoliageBase):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/props/xen_infestation/xen_grenade_plant.vmdl"

    @property
    def StartWithGrenade(self):
        if "StartWithGrenade" in self._entity_data:
            return bool(self._entity_data.get('StartWithGrenade'))
        return bool(1)


class xen_foliage_turret(XenFoliageBase):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/props/xen_infestation/xen_grenade_plant.vmdl"

    @property
    def burst_min(self):
        if "burst_min" in self._entity_data:
            return int(self._entity_data.get('burst_min'))
        return int(1)

    @property
    def burst_max(self):
        if "burst_max" in self._entity_data:
            return int(self._entity_data.get('burst_max'))
        return int(3)

    @property
    def postfiredelay(self):
        if "postfiredelay" in self._entity_data:
            return float(self._entity_data.get('postfiredelay'))
        return float(4)


class xen_flora_animatedmover(Targetname, Parentname):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return "models/props/xen_infestation_v2/xen_v2_floater_jellybobber.vmdl"

    @property
    def intercept_radius(self):
        if "intercept_radius" in self._entity_data:
            return float(self._entity_data.get('intercept_radius'))
        return float(15)

    @property
    def speed(self):
        if "speed" in self._entity_data:
            return float(self._entity_data.get('speed'))
        return float(5.0)

    @property
    def face_forward(self):
        if "face_forward" in self._entity_data:
            return bool(self._entity_data.get('face_forward'))
        return bool(1)

    @property
    def min_delay(self):
        if "min_delay" in self._entity_data:
            return float(self._entity_data.get('min_delay'))
        return float(0)

    @property
    def max_delay(self):
        if "max_delay" in self._entity_data:
            return float(self._entity_data.get('max_delay'))
        return float(0)

    @property
    def loop(self):
        if "loop" in self._entity_data:
            return bool(self._entity_data.get('loop'))
        return bool(1)

    @property
    def path_start(self):
        if "path_start" in self._entity_data:
            return self._entity_data.get('path_start')
        return ""

    @property
    def StartDisabled(self):
        if "StartDisabled" in self._entity_data:
            return bool(self._entity_data.get('StartDisabled'))
        return bool(0)

    @property
    def particle_effect(self):
        if "particle_effect" in self._entity_data:
            return self._entity_data.get('particle_effect')
        return "particles/environment/xen_gnats_3.vpcf"

    @property
    def disable_shadows(self):
        if "disable_shadows" in self._entity_data:
            return bool(self._entity_data.get('disable_shadows'))
        return bool(0)


class hl_vr_environmental_interaction:
    pass

    def __init__(self, entity_data: dict):
        self._entity_data = entity_data

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return None

    @property
    def test_type(self):
        if "test_type" in self._entity_data:
            return self._entity_data.get('test_type')
        return "0"


class info_notepad(Targetname):
    pass

    icon_sprite =  "editor/info_notepad.vmat"

    @property
    def message(self):
        if "message" in self._entity_data:
            return self._entity_data.get('message')
        return None

    @property
    def entity_01(self):
        if "entity_01" in self._entity_data:
            return self._entity_data.get('entity_01')
        return None

    @property
    def entity_02(self):
        if "entity_02" in self._entity_data:
            return self._entity_data.get('entity_02')
        return None

    @property
    def entity_03(self):
        if "entity_03" in self._entity_data:
            return self._entity_data.get('entity_03')
        return None

    @property
    def entity_04(self):
        if "entity_04" in self._entity_data:
            return self._entity_data.get('entity_04')
        return None

    @property
    def entity_05(self):
        if "entity_05" in self._entity_data:
            return self._entity_data.get('entity_05')
        return None

    @property
    def entity_06(self):
        if "entity_06" in self._entity_data:
            return self._entity_data.get('entity_06')
        return None

    @property
    def entity_07(self):
        if "entity_07" in self._entity_data:
            return self._entity_data.get('entity_07')
        return None

    @property
    def entity_08(self):
        if "entity_08" in self._entity_data:
            return self._entity_data.get('entity_08')
        return None

    @property
    def entity_09(self):
        if "entity_09" in self._entity_data:
            return self._entity_data.get('entity_09')
        return None

    @property
    def entity_10(self):
        if "entity_10" in self._entity_data:
            return self._entity_data.get('entity_10')
        return None


class trigger_xen_foliage_interaction(Trigger):
    pass


class trigger_foliage_interaction(Trigger):
    pass


class info_dynamic_shadow_hint_base(Targetname, EnableDisable):
    pass

    @property
    def importance(self):
        if "importance" in self._entity_data:
            return self._entity_data.get('importance')
        return "0"

    @property
    def lightchoice(self):
        if "lightchoice" in self._entity_data:
            return self._entity_data.get('lightchoice')
        return "0"

    @property
    def light(self):
        if "light" in self._entity_data:
            return self._entity_data.get('light')
        return ""


class info_dynamic_shadow_hint(info_dynamic_shadow_hint_base):
    pass

    @property
    def range(self):
        if "range" in self._entity_data:
            return float(self._entity_data.get('range'))
        return float(256)


class info_dynamic_shadow_hint_box(info_dynamic_shadow_hint_base):
    pass

    @property
    def box_mins(self):
        if "box_mins" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_mins'))
        return parse_int_vector("-128 -128 -128")

    @property
    def box_maxs(self):
        if "box_maxs" in self._entity_data:
            return parse_int_vector(self._entity_data.get('box_maxs'))
        return parse_int_vector("128 128 128")


class point_clientui_world_movie_panel(Targetname, Parentname):
    pass

    @property
    def src_movie(self):
        if "src_movie" in self._entity_data:
            return self._entity_data.get('src_movie')
        return None

    @property
    def override_sound_event(self):
        if "override_sound_event" in self._entity_data:
            return self._entity_data.get('override_sound_event')
        return None

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
    def auto_play(self):
        if "auto_play" in self._entity_data:
            return bool(self._entity_data.get('auto_play'))
        return bool(0)

    @property
    def repeat(self):
        if "repeat" in self._entity_data:
            return bool(self._entity_data.get('repeat'))
        return bool(0)

    @property
    def horizontal_align(self):
        if "horizontal_align" in self._entity_data:
            return self._entity_data.get('horizontal_align')
        return "0"

    @property
    def vertical_align(self):
        if "vertical_align" in self._entity_data:
            return self._entity_data.get('vertical_align')
        return "0"

    @property
    def orientation(self):
        if "orientation" in self._entity_data:
            return self._entity_data.get('orientation')
        return "0"


class logic_handsup_listener(Targetname, shared_enable_disable):
    pass

    icon_sprite =  "editor/logic_hands_up.vmat"

    @property
    def one_handed(self):
        if "one_handed" in self._entity_data:
            return bool(self._entity_data.get('one_handed'))
        return bool(0)


class logic_door_barricade(Targetname, EnableDisable):
    pass

    icon_sprite =  "editor/logic_door_barricade.vmat"


class logic_gameevent_listener(Targetname, shared_enable_disable):
    pass

    icon_sprite =  "editor/game_event_listener.vmat"

    @property
    def gameeventname(self):
        if "gameeventname" in self._entity_data:
            return self._entity_data.get('gameeventname')
        return None

    @property
    def gameeventitem(self):
        if "gameeventitem" in self._entity_data:
            return self._entity_data.get('gameeventitem')
        return None


class ai_markup_hlvr(markup_volume_with_ref):
    pass

    @property
    def aiProperty(self):
        flags = []
        if "aiProperty" in self._entity_data:
            value = self._entity_data.get("aiProperty", None)
            for name, (key, _) in {'cover_strong': ('COVER_STRONG', 0)}.items():
                if value & key > 0:
                    flags.append(name)
        return flags

    @property
    def tagFieldNames(self):
        if "tagFieldNames" in self._entity_data:
            return self._entity_data.get('tagFieldNames')
        return "aiProperty"


class prop_welded_physics(BasePropPhysics, RenderFields):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return None

    @property
    def start_welded(self):
        if "start_welded" in self._entity_data:
            return bool(self._entity_data.get('start_welded'))
        return bool(1)

    @property
    def drive_to_weld(self):
        if "drive_to_weld" in self._entity_data:
            return bool(self._entity_data.get('drive_to_weld'))
        return bool(1)

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


class prop_welded_physics_to_target(prop_welded_physics):
    pass

    @property
    def weld_target(self):
        if "weld_target" in self._entity_data:
            return self._entity_data.get('weld_target')
        return ""


class prop_welded_physics_target(Targetname, Studiomodel):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return None

    @property
    def rendercolor(self):
        if "rendercolor" in self._entity_data:
            return parse_int_vector(self._entity_data.get('rendercolor'))
        return parse_int_vector("255 255 255")


class prop_reviver_heart(Targetname):
    pass


class point_grabbable(Targetname, Parentname):
    pass

    icon_sprite =  "materials/editor/point_grabbable.vmat"

    @property
    def grab_radius(self):
        if "grab_radius" in self._entity_data:
            return float(self._entity_data.get('grab_radius'))
        return float(4)

    @property
    def shape(self):
        if "shape" in self._entity_data:
            return self._entity_data.get('shape')
        return "0"

    @property
    def limit_mins(self):
        if "limit_mins" in self._entity_data:
            return parse_int_vector(self._entity_data.get('limit_mins'))
        return parse_int_vector("0 -4 -4")

    @property
    def limit_maxs(self):
        if "limit_maxs" in self._entity_data:
            return parse_int_vector(self._entity_data.get('limit_maxs'))
        return parse_int_vector("0 4 4")

    @property
    def sphere_center(self):
        if "sphere_center" in self._entity_data:
            return parse_int_vector(self._entity_data.get('sphere_center'))
        return parse_int_vector("0 0 0")

    @property
    def constrain_angle(self):
        if "constrain_angle" in self._entity_data:
            return float(self._entity_data.get('constrain_angle'))
        return float(20)

    @property
    def detach_angle(self):
        if "detach_angle" in self._entity_data:
            return float(self._entity_data.get('detach_angle'))
        return float(30)

    @property
    def thickness(self):
        if "thickness" in self._entity_data:
            return float(self._entity_data.get('thickness'))
        return float(4)

    @property
    def handpose_entity_name(self):
        if "handpose_entity_name" in self._entity_data:
            return self._entity_data.get('handpose_entity_name')
        return ""

    @property
    def StartDisabled(self):
        if "StartDisabled" in self._entity_data:
            return bool(self._entity_data.get('StartDisabled'))
        return bool(0)


class point_aimat(Targetname, Parentname):
    pass

    @property
    def aim_target(self):
        if "aim_target" in self._entity_data:
            return self._entity_data.get('aim_target')
        return None

    @property
    def aim_offset(self):
        if "aim_offset" in self._entity_data:
            return parse_int_vector(self._entity_data.get('aim_offset'))
        return parse_int_vector("16 0 0")

    @property
    def max_angular_velocity(self):
        if "max_angular_velocity" in self._entity_data:
            return float(self._entity_data.get('max_angular_velocity'))
        return float(-1)

    @property
    def yaw_only(self):
        if "yaw_only" in self._entity_data:
            return bool(self._entity_data.get('yaw_only'))
        return bool(0)


class item_hlvr_headcrab_gland(Item):
    pass


class npc_vr_citizen_base(BaseNPC, Parentname):
    pass

    @property
    def model_state(self):
        if "model_state" in self._entity_data:
            return self._entity_data.get('model_state')
        return ""

    @property
    def background_character(self):
        if "background_character" in self._entity_data:
            return bool(self._entity_data.get('background_character'))
        return bool(0)


class npc_vr_citizen_male(npc_vr_citizen_base):
    pass


class npc_vr_citizen_female(npc_vr_citizen_base):
    pass


class point_hlvr_player_input_modifier(Targetname, shared_enable_disable):
    pass

    icon_sprite =  "materials/editor/point_hlvr_player_input_modifier.vmat"

    @property
    def disable_teleport(self):
        if "disable_teleport" in self._entity_data:
            return bool(self._entity_data.get('disable_teleport'))
        return bool(1)

    @property
    def hide_hands(self):
        if "hide_hands" in self._entity_data:
            return bool(self._entity_data.get('hide_hands'))
        return bool(1)


class npc_headcrab_armored(BaseHeadcrab, Parentname):
    pass


class prop_animating_breakable(Targetname, Parentname, Studiomodel, RenderFields, Glow):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return ""


class point_vort_energy(Targetname, shared_enable_disable):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return ""


class point_render_attr_curve(Targetname):
    pass

    @property
    def attribute_name(self):
        if "attribute_name" in self._entity_data:
            return self._entity_data.get('attribute_name')
        return ""

    @property
    def active_combo_name(self):
        if "active_combo_name" in self._entity_data:
            return self._entity_data.get('active_combo_name')
        return ""

    @property
    def curve(self):
        if "curve" in self._entity_data:
            return parse_int_vector(self._entity_data.get('curve'))
        return parse_int_vector("")


class point_entity_fader(Targetname):
    pass

    @property
    def target(self):
        if "target" in self._entity_data:
            return self._entity_data.get('target')
        return ""

    @property
    def curve(self):
        if "curve" in self._entity_data:
            return parse_int_vector(self._entity_data.get('curve'))
        return parse_int_vector("")


class trigger_lerp_object(Trigger):
    pass

    @property
    def lerp_target(self):
        if "lerp_target" in self._entity_data:
            return self._entity_data.get('lerp_target')
        return ""

    @property
    def lerp_target_attachment(self):
        if "lerp_target_attachment" in self._entity_data:
            return self._entity_data.get('lerp_target_attachment')
        return ""

    @property
    def lerp_duration(self):
        if "lerp_duration" in self._entity_data:
            return float(self._entity_data.get('lerp_duration'))
        return float(1)

    @property
    def lerp_restore_movetype(self):
        if "lerp_restore_movetype" in self._entity_data:
            return bool(self._entity_data.get('lerp_restore_movetype'))
        return bool(0)

    @property
    def lerp_effect(self):
        if "lerp_effect" in self._entity_data:
            return self._entity_data.get('lerp_effect')
        return "particles/entity/trigger_lerp_default.vpcf"

    @property
    def lerp_sound(self):
        if "lerp_sound" in self._entity_data:
            return self._entity_data.get('lerp_sound')
        return ""


class trigger_detect_bullet_fire(Trigger):
    pass

    @property
    def player_fire_only(self):
        if "player_fire_only" in self._entity_data:
            return self._entity_data.get('player_fire_only')
        return "0"


class trigger_detect_explosion(Trigger):
    pass


class save_photogrammetry_anchor(Targetname):
    pass

    icon_sprite =  "editor/save_photogrammetry_anchor.vmat"

    @property
    def photogrammetry_name(self):
        if "photogrammetry_name" in self._entity_data:
            return self._entity_data.get('photogrammetry_name')
        return ""


class info_offscreen_panorama_texture(Targetname):
    pass

    @property
    def layout_file(self):
        if "layout_file" in self._entity_data:
            return self._entity_data.get('layout_file')
        return None

    @property
    def targets(self):
        if "targets" in self._entity_data:
            return self._entity_data.get('targets')
        return ""

    @property
    def render_attr_name(self):
        if "render_attr_name" in self._entity_data:
            return self._entity_data.get('render_attr_name')
        return ""

    @property
    def resolution_x(self):
        if "resolution_x" in self._entity_data:
            return int(self._entity_data.get('resolution_x'))
        return int(512)

    @property
    def resolution_y(self):
        if "resolution_y" in self._entity_data:
            return int(self._entity_data.get('resolution_y'))
        return int(512)


class info_offscreen_movie_texture(Targetname):
    pass

    @property
    def src_movie(self):
        if "src_movie" in self._entity_data:
            return self._entity_data.get('src_movie')
        return None

    @property
    def targets(self):
        if "targets" in self._entity_data:
            return self._entity_data.get('targets')
        return ""

    @property
    def render_attr_name(self):
        if "render_attr_name" in self._entity_data:
            return self._entity_data.get('render_attr_name')
        return ""

    @property
    def resolution_x(self):
        if "resolution_x" in self._entity_data:
            return int(self._entity_data.get('resolution_x'))
        return int(512)

    @property
    def resolution_y(self):
        if "resolution_y" in self._entity_data:
            return int(self._entity_data.get('resolution_y'))
        return int(512)

    @property
    def override_sound_event(self):
        if "override_sound_event" in self._entity_data:
            return self._entity_data.get('override_sound_event')
        return None

    @property
    def auto_play(self):
        if "auto_play" in self._entity_data:
            return bool(self._entity_data.get('auto_play'))
        return bool(0)

    @property
    def repeat(self):
        if "repeat" in self._entity_data:
            return bool(self._entity_data.get('repeat'))
        return bool(0)

    @property
    def visible_range_check(self):
        if "visible_range_check" in self._entity_data:
            return float(self._entity_data.get('visible_range_check'))
        return float(300)


class hl_vr_environmental_interaction_volume(Trigger):
    pass


class point_player_speak(Targetname, Parentname):
    pass


class point_training_gravity_gloves(Targetname):
    pass


class filter_vr_grenade(BaseFilter):
    pass

    icon_sprite =  "editor/filter_name.vmat"

    @property
    def IsNotArmed(self):
        if "IsNotArmed" in self._entity_data:
            return bool(self._entity_data.get('IsNotArmed'))
        return bool(1)

    @property
    def IsArmed(self):
        if "IsArmed" in self._entity_data:
            return bool(self._entity_data.get('IsArmed'))
        return bool(1)


class hl_vr_accessibility(Targetname):
    pass


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
    "light_base": light_base,
    "light_base_legacy_params": light_base_legacy_params,
    "light_base_attenuation_params": light_base_attenuation_params,
    "light_environment": light_environment,
    "light_irradvolume": light_irradvolume,
    "TimeOfDay": TimeOfDay,
    "RealisticDayNightCycle": RealisticDayNightCycle,
    "env_time_of_day": env_time_of_day,
    "light_omni": light_omni,
    "light_spot": light_spot,
    "light_ortho": light_ortho,
    "light_importance_volume": light_importance_volume,
    "IndoorOutdoorLevel": IndoorOutdoorLevel,
    "SetBrightnessColor": SetBrightnessColor,
    "BaseLightProbeVolume": BaseLightProbeVolume,
    "BaseCubemap": BaseCubemap,
    "env_light_probe_volume": env_light_probe_volume,
    "env_cubemap": env_cubemap,
    "env_cubemap_box": env_cubemap_box,
    "env_combined_light_probe_volume": env_combined_light_probe_volume,
    "TalkNPC": TalkNPC,
    "PlayerCompanion": PlayerCompanion,
    "RappelNPC": RappelNPC,
    "VehicleDriverNPC": VehicleDriverNPC,
    "npc_vehicledriver": npc_vehicledriver,
    "npc_bullseye": npc_bullseye,
    "npc_enemyfinder": npc_enemyfinder,
    "ai_goal_operator": ai_goal_operator,
    "monster_generic": monster_generic,
    "generic_actor": generic_actor,
    "cycler_actor": cycler_actor,
    "npc_maker": npc_maker,
    "BaseScripted": BaseScripted,
    "scripted_sentence": scripted_sentence,
    "scripted_target": scripted_target,
    "base_ai_relationship": base_ai_relationship,
    "ai_relationship": ai_relationship,
    "LeadGoalBase": LeadGoalBase,
    "ai_goal_lead": ai_goal_lead,
    "FollowGoal": FollowGoal,
    "ai_goal_follow": ai_goal_follow,
    "ai_goal_injured_follow": ai_goal_injured_follow,
    "ai_battle_line": ai_battle_line,
    "ai_goal_fightfromcover": ai_goal_fightfromcover,
    "ai_goal_standoff": ai_goal_standoff,
    "ai_goal_police": ai_goal_police,
    "assault_rallypoint": assault_rallypoint,
    "assault_assaultpoint": assault_assaultpoint,
    "ai_goal_assault": ai_goal_assault,
    "BaseActBusy": BaseActBusy,
    "ai_goal_actbusy": ai_goal_actbusy,
    "ai_goal_actbusy_queue": ai_goal_actbusy_queue,
    "ai_changetarget": ai_changetarget,
    "ai_npc_eventresponsesystem": ai_npc_eventresponsesystem,
    "ai_changehintgroup": ai_changehintgroup,
    "ai_script_conditions": ai_script_conditions,
    "scripted_sequence": scripted_sequence,
    "aiscripted_schedule": aiscripted_schedule,
    "logic_choreographed_scene": logic_choreographed_scene,
    "logic_scene_list_manager": logic_scene_list_manager,
    "ai_sound": ai_sound,
    "ai_attached_item_manager": ai_attached_item_manager,
    "ai_addon": ai_addon,
    "ai_addon_builder": ai_addon_builder,
    "AlyxInteractable": AlyxInteractable,
    "CombineBallSpawners": CombineBallSpawners,
    "prop_combine_ball": prop_combine_ball,
    "trigger_physics_trap": trigger_physics_trap,
    "trigger_weapon_dissolve": trigger_weapon_dissolve,
    "trigger_weapon_strip": trigger_weapon_strip,
    "func_combine_ball_spawner": func_combine_ball_spawner,
    "point_combine_ball_launcher": point_combine_ball_launcher,
    "npc_grenade_frag": npc_grenade_frag,
    "npc_combine_cannon": npc_combine_cannon,
    "npc_combine_camera": npc_combine_camera,
    "npc_turret_ground": npc_turret_ground,
    "npc_turret_ceiling": npc_turret_ceiling,
    "npc_turret_floor": npc_turret_floor,
    "npc_cranedriver": npc_cranedriver,
    "npc_apcdriver": npc_apcdriver,
    "npc_rollermine": npc_rollermine,
    "npc_missiledefense": npc_missiledefense,
    "npc_sniper": npc_sniper,
    "info_radar_target": info_radar_target,
    "info_target_vehicle_transition": info_target_vehicle_transition,
    "info_snipertarget": info_snipertarget,
    "prop_thumper": prop_thumper,
    "npc_antlion": npc_antlion,
    "npc_antlionguard": npc_antlionguard,
    "BaseBird": BaseBird,
    "npc_crow": npc_crow,
    "npc_seagull": npc_seagull,
    "npc_pigeon": npc_pigeon,
    "npc_ichthyosaur": npc_ichthyosaur,
    "BaseHeadcrab": BaseHeadcrab,
    "npc_headcrab": npc_headcrab,
    "npc_headcrab_fast": npc_headcrab_fast,
    "npc_headcrab_black": npc_headcrab_black,
    "npc_stalker": npc_stalker,
    "npc_enemyfinder_combinecannon": npc_enemyfinder_combinecannon,
    "npc_citizen": npc_citizen,
    "npc_fisherman": npc_fisherman,
    "npc_barney": npc_barney,
    "BaseCombine": BaseCombine,
    "npc_combine_s": npc_combine_s,
    "npc_launcher": npc_launcher,
    "npc_hunter": npc_hunter,
    "npc_hunter_maker": npc_hunter_maker,
    "npc_advisor": npc_advisor,
    "env_sporeexplosion": env_sporeexplosion,
    "env_gunfire": env_gunfire,
    "env_headcrabcanister": env_headcrabcanister,
    "npc_vortigaunt": npc_vortigaunt,
    "npc_spotlight": npc_spotlight,
    "npc_strider": npc_strider,
    "npc_barnacle": npc_barnacle,
    "npc_combinegunship": npc_combinegunship,
    "info_target_helicopter_crash": info_target_helicopter_crash,
    "info_target_gunshipcrash": info_target_gunshipcrash,
    "npc_combinedropship": npc_combinedropship,
    "npc_helicopter": npc_helicopter,
    "grenade_helicopter": grenade_helicopter,
    "npc_heli_avoidsphere": npc_heli_avoidsphere,
    "npc_heli_avoidbox": npc_heli_avoidbox,
    "npc_heli_nobomb": npc_heli_nobomb,
    "info_target_advisor_roaming_crash": info_target_advisor_roaming_crash,
    "npc_combine_advisor_roaming": npc_combine_advisor_roaming,
    "BaseZombie": BaseZombie,
    "npc_fastzombie": npc_fastzombie,
    "npc_fastzombie_torso": npc_fastzombie_torso,
    "npc_zombie": npc_zombie,
    "npc_zombie_torso": npc_zombie_torso,
    "point_zombie_noise_generator": point_zombie_noise_generator,
    "npc_zombine": npc_zombine,
    "npc_poisonzombie": npc_poisonzombie,
    "npc_cscanner": npc_cscanner,
    "npc_clawscanner": npc_clawscanner,
    "npc_manhack": npc_manhack,
    "npc_mortarsynth": npc_mortarsynth,
    "npc_metropolice": npc_metropolice,
    "npc_crabsynth": npc_crabsynth,
    "npc_monk": npc_monk,
    "npc_alyx": npc_alyx,
    "info_darknessmode_lightsource": info_darknessmode_lightsource,
    "npc_kleiner": npc_kleiner,
    "npc_eli": npc_eli,
    "npc_magnusson": npc_magnusson,
    "npc_breen": npc_breen,
    "npc_mossman": npc_mossman,
    "npc_gman": npc_gman,
    "npc_dog": npc_dog,
    "npc_antlion_template_maker": npc_antlion_template_maker,
    "point_antlion_repellant": point_antlion_repellant,
    "player_control": player_control,
    "ai_ally_manager": ai_ally_manager,
    "ai_goal_lead_weapon": ai_goal_lead_weapon,
    "ai_citizen_response_system": ai_citizen_response_system,
    "func_healthcharger": func_healthcharger,
    "func_recharge": func_recharge,
    "func_vehicleclip": func_vehicleclip,
    "func_lookdoor": func_lookdoor,
    "trigger_waterydeath": trigger_waterydeath,
    "env_global": env_global,
    "BaseTank": BaseTank,
    "func_tank": func_tank,
    "func_tankpulselaser": func_tankpulselaser,
    "func_tank_gatling": func_tank_gatling,
    "func_tanklaser": func_tanklaser,
    "func_tankrocket": func_tankrocket,
    "func_tankairboatgun": func_tankairboatgun,
    "func_tankapcrocket": func_tankapcrocket,
    "func_tankmortar": func_tankmortar,
    "func_tankphyscannister": func_tankphyscannister,
    "func_tank_combine_cannon": func_tank_combine_cannon,
    "Item": Item,
    "item_dynamic_resupply": item_dynamic_resupply,
    "item_ammo_pistol": item_ammo_pistol,
    "item_ammo_pistol_large": item_ammo_pistol_large,
    "item_ammo_smg1": item_ammo_smg1,
    "item_ammo_smg1_large": item_ammo_smg1_large,
    "item_ammo_ar2": item_ammo_ar2,
    "item_ammo_ar2_large": item_ammo_ar2_large,
    "item_ammo_357": item_ammo_357,
    "item_ammo_357_large": item_ammo_357_large,
    "item_ammo_crossbow": item_ammo_crossbow,
    "item_box_buckshot": item_box_buckshot,
    "item_rpg_round": item_rpg_round,
    "item_ammo_smg1_grenade": item_ammo_smg1_grenade,
    "item_battery": item_battery,
    "item_healthkit": item_healthkit,
    "item_healthvial_DEPRECATED": item_healthvial_DEPRECATED,
    "item_ammo_ar2_altfire": item_ammo_ar2_altfire,
    "item_suit": item_suit,
    "item_ammo_crate": item_ammo_crate,
    "item_healthcharger_DEPRECATED": item_healthcharger_DEPRECATED,
    "item_suitcharger": item_suitcharger,
    "Weapon": Weapon,
    "weapon_crowbar": weapon_crowbar,
    "weapon_stunstick": weapon_stunstick,
    "weapon_pistol": weapon_pistol,
    "weapon_ar2": weapon_ar2,
    "weapon_rpg": weapon_rpg,
    "weapon_smg1": weapon_smg1,
    "weapon_357": weapon_357,
    "weapon_crossbow": weapon_crossbow,
    "weapon_zipline": weapon_zipline,
    "weapon_shotgun": weapon_shotgun,
    "weapon_frag": weapon_frag,
    "weapon_physcannon": weapon_physcannon,
    "weapon_bugbait": weapon_bugbait,
    "weapon_alyxgun": weapon_alyxgun,
    "weapon_annabelle": weapon_annabelle,
    "trigger_rpgfire": trigger_rpgfire,
    "trigger_vphysics_motion": trigger_vphysics_motion,
    "point_bugbait": point_bugbait,
    "weapon_brickbat": weapon_brickbat,
    "path_corner_crash": path_corner_crash,
    "player_loadsaved": player_loadsaved,
    "player_weaponstrip": player_weaponstrip,
    "player_speedmod": player_speedmod,
    "env_rotorwash": env_rotorwash,
    "combine_mine": combine_mine,
    "env_ar2explosion": env_ar2explosion,
    "env_starfield": env_starfield,
    "env_flare": env_flare,
    "env_muzzleflash": env_muzzleflash,
    "logic_achievement": logic_achievement,
    "func_monitor": func_monitor,
    "func_bulletshield": func_bulletshield,
    "BaseVehicle": BaseVehicle,
    "BaseDriveableVehicle": BaseDriveableVehicle,
    "prop_vehicle": prop_vehicle,
    "prop_vehicle_driveable": prop_vehicle_driveable,
    "point_apc_controller": point_apc_controller,
    "prop_vehicle_apc": prop_vehicle_apc,
    "info_apc_missile_hint": info_apc_missile_hint,
    "prop_vehicle_jeep": prop_vehicle_jeep,
    "vehicle_viewcontroller": vehicle_viewcontroller,
    "prop_vehicle_airboat": prop_vehicle_airboat,
    "prop_vehicle_cannon": prop_vehicle_cannon,
    "prop_vehicle_crane": prop_vehicle_crane,
    "prop_vehicle_prisoner_pod": prop_vehicle_prisoner_pod,
    "env_speaker": env_speaker,
    "script_tauremoval": script_tauremoval,
    "script_intro": script_intro,
    "env_citadel_energy_core": env_citadel_energy_core,
    "env_alyxemp": env_alyxemp,
    "test_sidelist": test_sidelist,
    "info_teleporter_countdown": info_teleporter_countdown,
    "prop_vehicle_choreo_generic": prop_vehicle_choreo_generic,
    "filter_combineball_type": filter_combineball_type,
    "env_entity_dissolver": env_entity_dissolver,
    "prop_coreball": prop_coreball,
    "prop_scalable": prop_scalable,
    "point_push": point_push,
    "npc_antlion_grub": npc_antlion_grub,
    "weapon_striderbuster": weapon_striderbuster,
    "point_flesh_effect_target": point_flesh_effect_target,
    "prop_door_rotating": prop_door_rotating,
    "prop_door_rotating_physics": prop_door_rotating_physics,
    "markup_volume": markup_volume,
    "markup_volume_tagged": markup_volume_tagged,
    "markup_group": markup_group,
    "func_nav_markup": func_nav_markup,
    "markup_volume_with_ref": markup_volume_with_ref,
    "post_processing_volume": post_processing_volume,
    "info_player_start": info_player_start,
    "worldspawn": worldspawn,
    "shared_enable_disable": shared_enable_disable,
    "trigger_traversal_modifier": trigger_traversal_modifier,
    "trigger_traversal_modifier_to_line": trigger_traversal_modifier_to_line,
    "trigger_traversal_no_teleport": trigger_traversal_no_teleport,
    "trigger_traversal_invalid_spot": trigger_traversal_invalid_spot,
    "trigger_traversal_tp_interrupt": trigger_traversal_tp_interrupt,
    "VRHandAttachment": VRHandAttachment,
    "BaseItemPhysics": BaseItemPhysics,
    "item_hlvr_prop_flashlight": item_hlvr_prop_flashlight,
    "item_hlvr_weapon_energygun": item_hlvr_weapon_energygun,
    "hlvr_weapon_energygun": hlvr_weapon_energygun,
    "item_hlvr_weapon_shotgun": item_hlvr_weapon_shotgun,
    "item_hlvr_weapon_rapidfire": item_hlvr_weapon_rapidfire,
    "item_hlvr_weapon_generic_pistol": item_hlvr_weapon_generic_pistol,
    "hlvr_weapon_crowbar": hlvr_weapon_crowbar,
    "item_hlvr_grenade_frag": item_hlvr_grenade_frag,
    "item_hlvr_grenade_remote_sticky": item_hlvr_grenade_remote_sticky,
    "item_hlvr_grenade_bomb": item_hlvr_grenade_bomb,
    "item_hlvr_grenade_xen": item_hlvr_grenade_xen,
    "HLVRAmmo": HLVRAmmo,
    "item_hlvr_clip_energygun": item_hlvr_clip_energygun,
    "item_hlvr_clip_energygun_multiple": item_hlvr_clip_energygun_multiple,
    "item_hlvr_clip_rapidfire": item_hlvr_clip_rapidfire,
    "item_hlvr_clip_shotgun_single": item_hlvr_clip_shotgun_single,
    "item_hlvr_clip_shotgun_multiple": item_hlvr_clip_shotgun_multiple,
    "item_hlvr_clip_generic_pistol": item_hlvr_clip_generic_pistol,
    "item_hlvr_clip_generic_pistol_multiple": item_hlvr_clip_generic_pistol_multiple,
    "item_healthvial": item_healthvial,
    "item_hlvr_weapon_grabbity_glove": item_hlvr_weapon_grabbity_glove,
    "item_hlvr_weapon_grabbity_slingshot": item_hlvr_weapon_grabbity_slingshot,
    "item_hlvr_weapon_tripmine": item_hlvr_weapon_tripmine,
    "item_hlvr_weapon_radio": item_hlvr_weapon_radio,
    "item_hlvr_multitool": item_hlvr_multitool,
    "npc_headcrab_runner": npc_headcrab_runner,
    "item_hlvr_weaponmodule_rapidfire": item_hlvr_weaponmodule_rapidfire,
    "item_hlvr_weaponmodule_ricochet": item_hlvr_weaponmodule_ricochet,
    "item_hlvr_weaponmodule_snark": item_hlvr_weaponmodule_snark,
    "item_hlvr_weaponmodule_zapper": item_hlvr_weaponmodule_zapper,
    "item_hlvr_weaponmodule_guidedmissle": item_hlvr_weaponmodule_guidedmissle,
    "item_hlvr_weaponmodule_guidedmissle_cluster": item_hlvr_weaponmodule_guidedmissle_cluster,
    "item_hlvr_weaponmodule_physcannon": item_hlvr_weaponmodule_physcannon,
    "hlvr_grenadepin_proxy": hlvr_grenadepin_proxy,
    "func_hlvr_nav_markup": func_hlvr_nav_markup,
    "func_nav_blocker": func_nav_blocker,
    "prop_animinteractable": prop_animinteractable,
    "info_hlvr_equip_player": info_hlvr_equip_player,
    "point_hlvr_strip_player": point_hlvr_strip_player,
    "item_item_crate": item_item_crate,
    "npc_cscanner": npc_cscanner,
    "npc_strider": npc_strider,
    "ai_attached_item_manager": ai_attached_item_manager,
    "item_hlvr_crafting_currency_large": item_hlvr_crafting_currency_large,
    "item_hlvr_crafting_currency_small": item_hlvr_crafting_currency_small,
    "item_hlvr_prop_discovery": item_hlvr_prop_discovery,
    "item_hlvr_prop_ammobag": item_hlvr_prop_ammobag,
    "prop_hlvr_crafting_station": prop_hlvr_crafting_station,
    "trigger_crafting_station_object_placement": trigger_crafting_station_object_placement,
    "item_healthcharger": item_healthcharger,
    "item_combine_console": item_combine_console,
    "item_combine_tank_locker": item_combine_tank_locker,
    "hlvr_vault_tractor_beam_console": hlvr_vault_tractor_beam_console,
    "info_hlvr_toner_port": info_hlvr_toner_port,
    "info_hlvr_toner_path": info_hlvr_toner_path,
    "info_hlvr_toner_path_node": info_hlvr_toner_path_node,
    "info_hlvr_toner_junction": info_hlvr_toner_junction,
    "info_hlvr_offscreen_particle_texture": info_hlvr_offscreen_particle_texture,
    "trigger_zap_module": trigger_zap_module,
    "env_gradient_fog": env_gradient_fog,
    "env_spherical_vignette": env_spherical_vignette,
    "env_cubemap_fog": env_cubemap_fog,
    "trigger_resource_analyzer": trigger_resource_analyzer,
    "trigger_player_out_of_ammo": trigger_player_out_of_ammo,
    "npc_turret_citizen": npc_turret_citizen,
    "trigger_turret_citizen_battery": trigger_turret_citizen_battery,
    "trigger_blind_zombie_crash": trigger_blind_zombie_crash,
    "trigger_blind_zombie_sound_area": trigger_blind_zombie_sound_area,
    "func_xen_membrane_barrier": func_xen_membrane_barrier,
    "trigger_blind_zombie_wander_area": trigger_blind_zombie_wander_area,
    "trigger_xen_membrane_door": trigger_xen_membrane_door,
    "trigger_player_peephole": trigger_player_peephole,
    "npc_zombie_blind": npc_zombie_blind,
    "npc_zombie": npc_zombie,
    "item_hlvr_prop_battery": item_hlvr_prop_battery,
    "item_hlvr_health_station_vial": item_hlvr_health_station_vial,
    "item_hlvr_combine_console_tank": item_hlvr_combine_console_tank,
    "prop_physics": prop_physics,
    "prop_physics_override": prop_physics_override,
    "logic_distance_autosave": logic_distance_autosave,
    "logic_multilight_proxy": logic_multilight_proxy,
    "point_lightmodifier": point_lightmodifier,
    "prop_handpose": prop_handpose,
    "skybox_reference": skybox_reference,
    "ghost_speaker": ghost_speaker,
    "ghost_actor": ghost_actor,
    "generic_actor": generic_actor,
    "prop_russell_headset": prop_russell_headset,
    "prop_physics_interactive": prop_physics_interactive,
    "trigger_physics": trigger_physics,
    "info_teleport_magnet": info_teleport_magnet,
    "func_combine_barrier": func_combine_barrier,
    "func_electrified_volume": func_electrified_volume,
    "hl_vr_texture_based_animatable": hl_vr_texture_based_animatable,
    "light_environment": light_environment,
    "light_spot": light_spot,
    "light_ortho": light_ortho,
    "light_omni": light_omni,
    "env_combined_light_probe_volume": env_combined_light_probe_volume,
    "prop_ragdoll": prop_ragdoll,
    "info_hlvr_holo_hacking_plug": info_hlvr_holo_hacking_plug,
    "info_hlvr_holo_hacking_spawn_target": info_hlvr_holo_hacking_spawn_target,
    "combine_attached_armor_prop": combine_attached_armor_prop,
    "hlvr_piano": hlvr_piano,
    "func_dry_erase_board": func_dry_erase_board,
    "prop_dry_erase_marker": prop_dry_erase_marker,
    "XenFoliageBase": XenFoliageBase,
    "xen_foliage": xen_foliage,
    "xen_hearing_flower": xen_hearing_flower,
    "xen_foliage_bloater": xen_foliage_bloater,
    "xen_foliage_grenade_spawner": xen_foliage_grenade_spawner,
    "xen_foliage_turret": xen_foliage_turret,
    "xen_flora_animatedmover": xen_flora_animatedmover,
    "BaseNPC": BaseNPC,
    "hl_vr_environmental_interaction": hl_vr_environmental_interaction,
    "info_notepad": info_notepad,
    "trigger_xen_foliage_interaction": trigger_xen_foliage_interaction,
    "trigger_foliage_interaction": trigger_foliage_interaction,
    "info_dynamic_shadow_hint_base": info_dynamic_shadow_hint_base,
    "info_dynamic_shadow_hint": info_dynamic_shadow_hint,
    "info_dynamic_shadow_hint_box": info_dynamic_shadow_hint_box,
    "point_clientui_world_movie_panel": point_clientui_world_movie_panel,
    "logic_handsup_listener": logic_handsup_listener,
    "logic_door_barricade": logic_door_barricade,
    "logic_gameevent_listener": logic_gameevent_listener,
    "ai_markup_hlvr": ai_markup_hlvr,
    "logic_playerproxy": logic_playerproxy,
    "prop_welded_physics": prop_welded_physics,
    "prop_welded_physics_to_target": prop_welded_physics_to_target,
    "prop_welded_physics_target": prop_welded_physics_target,
    "prop_reviver_heart": prop_reviver_heart,
    "point_grabbable": point_grabbable,
    "point_aimat": point_aimat,
    "item_hlvr_headcrab_gland": item_hlvr_headcrab_gland,
    "npc_vr_citizen_base": npc_vr_citizen_base,
    "npc_vr_citizen_male": npc_vr_citizen_male,
    "npc_vr_citizen_female": npc_vr_citizen_female,
    "point_hlvr_player_input_modifier": point_hlvr_player_input_modifier,
    "npc_headcrab_armored": npc_headcrab_armored,
    "prop_animating_breakable": prop_animating_breakable,
    "point_vort_energy": point_vort_energy,
    "logic_achievement": logic_achievement,
    "point_render_attr_curve": point_render_attr_curve,
    "point_entity_fader": point_entity_fader,
    "trigger_lerp_object": trigger_lerp_object,
    "trigger_detect_bullet_fire": trigger_detect_bullet_fire,
    "trigger_detect_explosion": trigger_detect_explosion,
    "save_photogrammetry_anchor": save_photogrammetry_anchor,
    "info_offscreen_panorama_texture": info_offscreen_panorama_texture,
    "info_offscreen_movie_texture": info_offscreen_movie_texture,
    "hl_vr_environmental_interaction_volume": hl_vr_environmental_interaction_volume,
    "point_player_speak": point_player_speak,
    "point_training_gravity_gloves": point_training_gravity_gloves,
    "filter_vr_grenade": filter_vr_grenade,
    "prop_dynamic": prop_dynamic,
    "func_physbox": func_physbox,
    "prop_dynamic_override": prop_dynamic_override,
    "prop_door_rotating_physics": prop_door_rotating_physics,
    "hl_vr_accessibility": hl_vr_accessibility,
    "env_volumetric_fog_controller": env_volumetric_fog_controller,
    "env_volumetric_fog_volume": env_volumetric_fog_volume,
    "env_sky": env_sky,
}
