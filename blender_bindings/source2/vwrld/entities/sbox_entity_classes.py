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
        return int(None)


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


class snd_event_point(Targetname, Parentname):
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


class snd_event_alignedbox(snd_event_point):
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


class env_screeneffect(Targetname):
    pass

    @property
    def type(self):
        if "type" in self._entity_data:
            return self._entity_data.get('type')
        return "0"


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
        return False

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


class gibshooter(gibshooterbase):
    pass

    icon_sprite = "editor/gibshooter.vmat"


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
        return "materials/decals/decalgraffiti001c.vmat"

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


class info_radial_link_controller(Targetname, Parentname):
    pass

    @property
    def radius(self):
        if "radius" in self._entity_data:
            return float(self._entity_data.get('radius'))
        return float(120)


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
    def _minlight(self):
        if "_minlight" in self._entity_data:
            return self._entity_data.get('_minlight')
        return None

    @property
    def loopmovesound(self):
        if "loopmovesound" in self._entity_data:
            return bool(self._entity_data.get('loopmovesound'))
        return bool(0)


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
        return int(None)

    @property
    def CompareValue(self):
        if "CompareValue" in self._entity_data:
            return int(self._entity_data.get('CompareValue'))
        return int(None)


class logic_branch(Targetname):
    pass

    icon_sprite = "editor/logic_branch.vmat"

    @property
    def InitialValue(self):
        if "InitialValue" in self._entity_data:
            return int(self._entity_data.get('InitialValue'))
        return int(None)


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
        return float(None)

    @property
    def UpperRandomBound(self):
        if "UpperRandomBound" in self._entity_data:
            return float(self._entity_data.get('UpperRandomBound'))
        return float(None)

    @property
    def RefireTime(self):
        if "RefireTime" in self._entity_data:
            return float(self._entity_data.get('RefireTime'))
        return float(None)

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
        return int(None)

    @property
    def out2(self):
        if "out2" in self._entity_data:
            return int(self._entity_data.get('out2'))
        return int(None)


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


class filter_activator_class(BaseFilter):
    pass

    icon_sprite = "editor/filter_class.vmat"

    @property
    def filterclass(self):
        if "filterclass" in self._entity_data:
            return self._entity_data.get('filterclass')
        return None


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
        return float(None)

    @property
    def tolerance(self):
        if "tolerance" in self._entity_data:
            return int(self._entity_data.get('tolerance'))
        return int(None)

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
        return float(None)


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
        return float(None)


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
        return float(None)

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


class trigger_remove(Trigger):
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


class trigger_transition(Targetname):
    pass

    @property
    def filtername(self):
        if "filtername" in self._entity_data:
            return self._entity_data.get('filtername')
        return None


class trigger_serverragdoll(Targetname):
    pass


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


class func_reflective_glass(func_brush):
    pass


class point_gamestats_counter(Targetname, EnableDisable):
    pass

    @property
    def Name(self):
        if "Name" in self._entity_data:
            return self._entity_data.get('Name')
        return None


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


class light_omni(light_base, light_base_legacy_params, light_base_attenuation_params):
    pass

    icon_sprite = "materials/editor/light.vmat"

    @property
    def castshadows(self):
        if "castshadows" in self._entity_data:
            return self._entity_data.get('castshadows')
        return "1"


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

    icon_sprite = "materials/editor/light_importance_volume.vmat"

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


class func_nav_blocker(Targetname):
    pass

    @property
    def StartDisabled(self):
        if "StartDisabled" in self._entity_data:
            return bool(self._entity_data.get('StartDisabled'))
        return bool(0)


class env_gradient_fog(Targetname, EnableDisable):
    pass

    icon_sprite = "materials/editor/env_fog_controller.vmat"

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

    icon_sprite = "materials/editor/env_fog_controller.vmat"

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

    icon_sprite = "materials/editor/env_cubemap_fog.vmat"

    @property
    def cubemapfogtexture(self):
        if "cubemapfogtexture" in self._entity_data:
            return self._entity_data.get('cubemapfogtexture')
        return "materials/skybox/colorscript/skybox_sky_sunset.vtex"

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


class logic_distance_autosave(Targetname):
    pass

    icon_sprite = "editor/logic_autosave.vmat"

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


class info_notepad(Targetname):
    pass

    icon_sprite = "editor/info_notepad.vmat"

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


class logic_door_barricade(Targetname, EnableDisable):
    pass

    icon_sprite = "editor/logic_door_barricade.vmat"


class logic_gameevent_listener(Targetname, shared_enable_disable):
    pass

    icon_sprite = "editor/game_event_listener.vmat"

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


class prop_animating_breakable(Targetname, Parentname, Studiomodel, RenderFields, Glow):
    pass

    @property
    def model(self):
        if "model" in self._entity_data:
            return self._entity_data.get('model')
        return ""


class logic_achievement(Targetname, EnableDisable):
    pass

    icon_sprite = "editor/logic_achievement.vmat"

    @property
    def AchievementEvent(self):
        if "AchievementEvent" in self._entity_data:
            return self._entity_data.get('AchievementEvent')
        return "0"


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

    icon_sprite = "editor/save_photogrammetry_anchor.vmat"

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
    "DamageFilter": DamageFilter,
    "ResponseContext": ResponseContext,
    "Breakable": Breakable,
    "BreakableBrush": BreakableBrush,
    "CanBeClientOnly": CanBeClientOnly,
    "BreakableProp": BreakableProp,
    "PlayerClass": PlayerClass,
    "Light": Light,
    "Node": Node,
    "HintNode": HintNode,
    "TriggerOnce": TriggerOnce,
    "Trigger": Trigger,
    "worldbase": worldbase,
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
    "env_screeneffect": env_screeneffect,
    "env_tonemap_controller": env_tonemap_controller,
    "game_ragdoll_manager": game_ragdoll_manager,
    "game_gib_manager": game_gib_manager,
    "env_fade": env_fade,
    "trigger_tonemap": trigger_tonemap,
    "func_useableladder": func_useableladder,
    "func_ladderendpoint": func_ladderendpoint,
    "info_ladder_dismount": info_ladder_dismount,
    "func_wall": func_wall,
    "func_clip_interaction_layer": func_clip_interaction_layer,
    "cycler": cycler,
    "func_orator": func_orator,
    "gibshooterbase": gibshooterbase,
    "env_beam": env_beam,
    "env_explosion": env_explosion,
    "env_physexplosion": env_physexplosion,
    "env_physimpact": env_physimpact,
    "env_fire": env_fire,
    "env_entity_igniter": env_entity_igniter,
    "env_laser": env_laser,
    "env_message": env_message,
    "env_shake": env_shake,
    "gibshooter": gibshooter,
    "env_soundscape_proxy": env_soundscape_proxy,
    "snd_soundscape_proxy": snd_soundscape_proxy,
    "env_soundscape": env_soundscape,
    "snd_soundscape": snd_soundscape,
    "env_soundscape_triggerable": env_soundscape_triggerable,
    "snd_soundscape_triggerable": snd_soundscape_triggerable,
    "env_spark": env_spark,
    "env_sprite": env_sprite,
    "BaseEnvWind": BaseEnvWind,
    "env_wind": env_wind,
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
    "info_overlay": info_overlay,
    "info_overlay_transition": info_overlay_transition,
    "info_intermission": info_intermission,
    "info_landmark": info_landmark,
    "info_spawngroup_load_unload": info_spawngroup_load_unload,
    "info_null": info_null,
    "info_target": info_target,
    "info_particle_target": info_particle_target,
    "phys_ragdollmagnet": phys_ragdollmagnet,
    "AiHullFlags": AiHullFlags,
    "BaseNodeLink": BaseNodeLink,
    "info_radial_link_controller": info_radial_link_controller,
    "color_correction": color_correction,
    "KeyFrame": KeyFrame,
    "Mover": Mover,
    "func_movelinear": func_movelinear,
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
    "func_physical_button": func_physical_button,
    "momentary_rot_button": momentary_rot_button,
    "Door": Door,
    "BaseFadeProp": BaseFadeProp,
    "BasePropDoorRotating": BasePropDoorRotating,
    "BModelParticleSpawner": BModelParticleSpawner,
    "func_dustmotes": func_dustmotes,
    "func_dustcloud": func_dustcloud,
    "logic_auto": logic_auto,
    "point_viewcontrol": point_viewcontrol,
    "point_posecontroller": point_posecontroller,
    "logic_compare": logic_compare,
    "logic_branch": logic_branch,
    "logic_branch_listener": logic_branch_listener,
    "logic_case": logic_case,
    "LogicNPCCounterPointBase": LogicNPCCounterPointBase,
    "logic_npc_counter_radius": logic_npc_counter_radius,
    "logic_npc_counter_aabb": logic_npc_counter_aabb,
    "logic_npc_counter_obb": logic_npc_counter_obb,
    "logic_script": logic_script,
    "logic_relay": logic_relay,
    "logic_timer": logic_timer,
    "hammer_updateignorelist": hammer_updateignorelist,
    "logic_collision_pair": logic_collision_pair,
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
    "filter_activator_class": filter_activator_class,
    "filter_damage_type": filter_damage_type,
    "filter_enemy": filter_enemy,
    "point_anglesensor": point_anglesensor,
    "point_angularvelocitysensor": point_angularvelocitysensor,
    "point_velocitysensor": point_velocitysensor,
    "point_proximity_sensor": point_proximity_sensor,
    "point_teleport": point_teleport,
    "point_hurt": point_hurt,
    "point_playermoveconstraint": point_playermoveconstraint,
    "BasePhysicsSimulated": BasePhysicsSimulated,
    "BasePhysicsNoSettleAttached": BasePhysicsNoSettleAttached,
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
    "prop_static": prop_static,
    "prop_dynamic": prop_dynamic,
    "prop_dynamic_override": prop_dynamic_override,
    "BasePropPhysics": BasePropPhysics,
    "prop_ragdoll": prop_ragdoll,
    "prop_dynamic_ornament": prop_dynamic_ornament,
    "func_breakable": func_breakable,
    "func_viscluster": func_viscluster,
    "func_illusionary": func_illusionary,
    "func_precipitation": func_precipitation,
    "func_precipitation_blocker": func_precipitation_blocker,
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
    "trigger_remove": trigger_remove,
    "trigger_snd_sos_opvar": trigger_snd_sos_opvar,
    "trigger_look": trigger_look,
    "trigger_push": trigger_push,
    "trigger_wind": trigger_wind,
    "trigger_impact": trigger_impact,
    "trigger_proximity": trigger_proximity,
    "trigger_transition": trigger_transition,
    "trigger_serverragdoll": trigger_serverragdoll,
    "point_camera": point_camera,
    "info_camera_link": info_camera_link,
    "logic_measure_movement": logic_measure_movement,
    "material_modify_control": material_modify_control,
    "point_devshot_camera": point_devshot_camera,
    "logic_playerproxy": logic_playerproxy,
    "func_brush": func_brush,
    "func_reflective_glass": func_reflective_glass,
    "point_gamestats_counter": point_gamestats_counter,
    "func_instance": func_instance,
    "func_timescale": func_timescale,
    "prop_hallucination": prop_hallucination,
    "point_worldtext": point_worldtext,
    "func_occluder": func_occluder,
    "func_distance_occluder": func_distance_occluder,
    "path_corner": path_corner,
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
    "commentary_auto": commentary_auto,
    "point_commentary_node": point_commentary_node,
    "light_base": light_base,
    "light_base_legacy_params": light_base_legacy_params,
    "light_base_attenuation_params": light_base_attenuation_params,
    "light_environment": light_environment,
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
    "markup_volume": markup_volume,
    "markup_volume_tagged": markup_volume_tagged,
    "markup_group": markup_group,
    "func_nav_markup": func_nav_markup,
    "markup_volume_with_ref": markup_volume_with_ref,
    "post_processing_volume": post_processing_volume,
    "worldspawn": worldspawn,
    "shared_enable_disable": shared_enable_disable,
    "trigger_traversal_modifier": trigger_traversal_modifier,
    "trigger_traversal_modifier_to_line": trigger_traversal_modifier_to_line,
    "trigger_traversal_no_teleport": trigger_traversal_no_teleport,
    "trigger_traversal_invalid_spot": trigger_traversal_invalid_spot,
    "trigger_traversal_tp_interrupt": trigger_traversal_tp_interrupt,
    "BaseItemPhysics": BaseItemPhysics,
    "func_nav_blocker": func_nav_blocker,
    "env_gradient_fog": env_gradient_fog,
    "env_spherical_vignette": env_spherical_vignette,
    "env_cubemap_fog": env_cubemap_fog,
    "logic_distance_autosave": logic_distance_autosave,
    "logic_multilight_proxy": logic_multilight_proxy,
    "point_lightmodifier": point_lightmodifier,
    "skybox_reference": skybox_reference,
    "trigger_physics": trigger_physics,
    "info_teleport_magnet": info_teleport_magnet,
    "light_environment": light_environment,
    "light_spot": light_spot,
    "light_ortho": light_ortho,
    "light_omni": light_omni,
    "env_combined_light_probe_volume": env_combined_light_probe_volume,
    "env_light_probe_volume": env_light_probe_volume,
    "prop_ragdoll": prop_ragdoll,
    "info_notepad": info_notepad,
    "trigger_xen_foliage_interaction": trigger_xen_foliage_interaction,
    "trigger_foliage_interaction": trigger_foliage_interaction,
    "info_dynamic_shadow_hint_base": info_dynamic_shadow_hint_base,
    "info_dynamic_shadow_hint": info_dynamic_shadow_hint,
    "info_dynamic_shadow_hint_box": info_dynamic_shadow_hint_box,
    "point_clientui_world_movie_panel": point_clientui_world_movie_panel,
    "logic_door_barricade": logic_door_barricade,
    "logic_gameevent_listener": logic_gameevent_listener,
    "logic_playerproxy": logic_playerproxy,
    "point_aimat": point_aimat,
    "prop_animating_breakable": prop_animating_breakable,
    "logic_achievement": logic_achievement,
    "point_render_attr_curve": point_render_attr_curve,
    "point_entity_fader": point_entity_fader,
    "trigger_lerp_object": trigger_lerp_object,
    "trigger_detect_bullet_fire": trigger_detect_bullet_fire,
    "trigger_detect_explosion": trigger_detect_explosion,
    "save_photogrammetry_anchor": save_photogrammetry_anchor,
    "info_offscreen_panorama_texture": info_offscreen_panorama_texture,
    "info_offscreen_movie_texture": info_offscreen_movie_texture,
    "prop_dynamic": prop_dynamic,
    "prop_dynamic_override": prop_dynamic_override,
    "env_volumetric_fog_controller": env_volumetric_fog_controller,
    "env_volumetric_fog_volume": env_volumetric_fog_volume,
    "env_sky": env_sky,
    "point_worldtext": point_worldtext,
}
