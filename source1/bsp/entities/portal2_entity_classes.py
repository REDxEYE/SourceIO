
def parse_source_value(value):
    if type(value) is str:
        value: str
        if value.replace('.', '', 1).replace('-', '', 1).isdecimal():
            return float(value) if '.' in value else int(value)
        return 0
    else:
        return value


def parse_int_vector(string):
    return [parse_source_value(val) for val in string.replace('  ', ' ').split(' ')]


def parse_float_vector(string):
    return [float(val) for val in string.replace('  ', ' ').split(' ')]


class Base:
    hammer_id_counter = 0

    def __init__(self):
        self.hammer_id = 0
        self.class_name = 'ANY'

    @classmethod
    def new_hammer_id(cls):
        new_id = cls.hammer_id_counter
        cls.hammer_id_counter += 1
        return new_id

    @staticmethod
    def from_dict(instance, entity_data: dict):
        if 'hammerid' in entity_data:
            instance.hammer_id = int(entity_data.get('hammerid'))
        else:  # Titanfall
            instance.hammer_id = Base.new_hammer_id()
        instance.class_name = entity_data.get('classname')


class PaintableBrush(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class Angles(Base):
    def __init__(self):
        super().__init__()
        self.angles = [0.0, 0.0, 0.0]  # Type: angle

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.angles = parse_float_vector(entity_data.get('angles', "0 0 0"))  # Type: angle


class Origin(Base):
    def __init__(self):
        super().__init__()
        self.origin = None  # Type: origin

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))  # Type: origin


class Reflection(Base):
    def __init__(self):
        super().__init__()
        self.drawinfastreflection = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.drawinfastreflection = entity_data.get('drawinfastreflection', None)  # Type: boolean


class ToggleDraw(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class Shadow(Base):
    def __init__(self):
        super().__init__()
        self.disableshadows = None  # Type: boolean
        self.disableshadowdepth = None  # Type: boolean
        self.shadowdepthnocache = None  # Type: choices
        self.disableflashlight = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.disableshadows = entity_data.get('disableshadows', None)  # Type: boolean
        instance.disableshadowdepth = entity_data.get('disableshadowdepth', None)  # Type: boolean
        instance.shadowdepthnocache = entity_data.get('shadowdepthnocache', None)  # Type: choices
        instance.disableflashlight = entity_data.get('disableflashlight', None)  # Type: boolean


class Studiomodel(Shadow, Reflection, ToggleDraw):
    def __init__(self):
        super(Shadow).__init__()
        super(Reflection).__init__()
        super(ToggleDraw).__init__()
        self.model = None  # Type: studio
        self.skin = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Shadow.from_dict(instance, entity_data)
        Reflection.from_dict(instance, entity_data)
        ToggleDraw.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', None)  # Type: studio
        instance.skin = parse_source_value(entity_data.get('skin', 0))  # Type: integer


class BasePlat(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class Targetname(Base):
    def __init__(self):
        super().__init__()
        self.targetname = None  # Type: target_source
        self.vscripts = None  # Type: scriptlist
        self.thinkfunction = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.targetname = entity_data.get('targetname', None)  # Type: target_source
        instance.vscripts = entity_data.get('vscripts', None)  # Type: scriptlist
        instance.thinkfunction = entity_data.get('thinkfunction', None)  # Type: string


class Parentname(Base):
    def __init__(self):
        super().__init__()
        self.parentname = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.parentname = entity_data.get('parentname', None)  # Type: target_destination


class BaseBrush(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class EnableDisable(Base):
    def __init__(self):
        super().__init__()
        self.StartDisabled = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.StartDisabled = entity_data.get('startdisabled', None)  # Type: boolean


class RenderFxChoices(Base):
    def __init__(self):
        super().__init__()
        self.renderfx = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.renderfx = entity_data.get('renderfx', None)  # Type: choices


class SpatialEntity(Base):
    def __init__(self):
        super().__init__()
        self.minfalloff = 0.0  # Type: float
        self.maxfalloff = 200.0  # Type: float
        self.maxweight = 1.0  # Type: float
        self.fadeInDuration = 0.0  # Type: float
        self.fadeOutDuration = 0.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.minfalloff = float(entity_data.get('minfalloff', 0.0))  # Type: float
        instance.maxfalloff = float(entity_data.get('maxfalloff', 200.0))  # Type: float
        instance.maxweight = float(entity_data.get('maxweight', 1.0))  # Type: float
        instance.fadeInDuration = float(entity_data.get('fadeinduration', 0.0))  # Type: float
        instance.fadeOutDuration = float(entity_data.get('fadeoutduration', 0.0))  # Type: float


class RenderFields(RenderFxChoices):
    def __init__(self):
        super(RenderFxChoices).__init__()
        self.rendermode = None  # Type: choices
        self.renderamt = 255  # Type: integer
        self.rendercolor = [255, 255, 255]  # Type: color255
        self.disablereceiveshadows = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        RenderFxChoices.from_dict(instance, entity_data)
        instance.rendermode = entity_data.get('rendermode', None)  # Type: choices
        instance.renderamt = parse_source_value(entity_data.get('renderamt', 255))  # Type: integer
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "255 255 255"))  # Type: color255
        instance.disablereceiveshadows = entity_data.get('disablereceiveshadows', None)  # Type: boolean


class SystemLevelChoice(Base):
    def __init__(self):
        super().__init__()
        self.mincpulevel = None  # Type: choices
        self.maxcpulevel = None  # Type: choices
        self.mingpulevel = None  # Type: choices
        self.maxgpulevel = None  # Type: choices
        self.disableX360 = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.mincpulevel = entity_data.get('mincpulevel', None)  # Type: choices
        instance.maxcpulevel = entity_data.get('maxcpulevel', None)  # Type: choices
        instance.mingpulevel = entity_data.get('mingpulevel', None)  # Type: choices
        instance.maxgpulevel = entity_data.get('maxgpulevel', None)  # Type: choices
        instance.disableX360 = entity_data.get('disablex360', None)  # Type: choices


class Inputfilter(Base):
    def __init__(self):
        super().__init__()
        self.InputFilter = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.InputFilter = entity_data.get('inputfilter', None)  # Type: choices


class Global(Base):
    def __init__(self):
        super().__init__()
        self.globalname = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.globalname = entity_data.get('globalname', None)  # Type: string


class EnvGlobal(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.initialstate = None  # Type: choices
        self.counter = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.initialstate = entity_data.get('initialstate', None)  # Type: choices
        instance.counter = parse_source_value(entity_data.get('counter', 0))  # Type: integer


class DamageFilter(Base):
    def __init__(self):
        super().__init__()
        self.damagefilter = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.damagefilter = entity_data.get('damagefilter', None)  # Type: target_destination


class ResponseContext(Base):
    def __init__(self):
        super().__init__()
        self.ResponseContext = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.ResponseContext = entity_data.get('responsecontext', None)  # Type: string


class Breakable(Targetname, Shadow, DamageFilter, Reflection):
    def __init__(self):
        super(Targetname).__init__()
        super(Shadow).__init__()
        super(DamageFilter).__init__()
        super(Reflection).__init__()
        self.ExplodeDamage = None  # Type: float
        self.ExplodeRadius = None  # Type: float
        self.PerformanceMode = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        DamageFilter.from_dict(instance, entity_data)
        Reflection.from_dict(instance, entity_data)
        instance.ExplodeDamage = float(entity_data.get('explodedamage', 0))  # Type: float
        instance.ExplodeRadius = float(entity_data.get('exploderadius', 0))  # Type: float
        instance.PerformanceMode = entity_data.get('performancemode', None)  # Type: choices


class BreakableBrush(Parentname, Breakable, Global):
    def __init__(self):
        super(Breakable).__init__()
        super(Parentname).__init__()
        super(Global).__init__()
        self.propdata = None  # Type: choices
        self.health = 1  # Type: integer
        self.material = None  # Type: choices
        self.explosion = None  # Type: choices
        self.gibdir = [0.0, 0.0, 0.0]  # Type: angle
        self.nodamageforces = None  # Type: boolean
        self.gibmodel = None  # Type: string
        self.spawnobject = None  # Type: choices
        self.explodemagnitude = None  # Type: integer
        self.pressuredelay = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Breakable.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        instance.propdata = entity_data.get('propdata', None)  # Type: choices
        instance.health = parse_source_value(entity_data.get('health', 1))  # Type: integer
        instance.material = entity_data.get('material', None)  # Type: choices
        instance.explosion = entity_data.get('explosion', None)  # Type: choices
        instance.gibdir = parse_float_vector(entity_data.get('gibdir', "0 0 0"))  # Type: angle
        instance.nodamageforces = entity_data.get('nodamageforces', None)  # Type: boolean
        instance.gibmodel = entity_data.get('gibmodel', None)  # Type: string
        instance.spawnobject = entity_data.get('spawnobject', None)  # Type: choices
        instance.explodemagnitude = parse_source_value(entity_data.get('explodemagnitude', 0))  # Type: integer
        instance.pressuredelay = float(entity_data.get('pressuredelay', 0))  # Type: float


class BreakableProp(Breakable):
    def __init__(self):
        super(Breakable).__init__()
        self.pressuredelay = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Breakable.from_dict(instance, entity_data)
        instance.pressuredelay = float(entity_data.get('pressuredelay', 0))  # Type: float


class BaseNPC(Shadow, Targetname, RenderFields, ResponseContext, ToggleDraw, Angles, DamageFilter):
    def __init__(self):
        super(RenderFields).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        super(ResponseContext).__init__()
        super(ToggleDraw).__init__()
        super(Angles).__init__()
        super(DamageFilter).__init__()
        self.target = None  # Type: target_destination
        self.squadname = None  # Type: string
        self.hintgroup = None  # Type: string
        self.hintlimiting = None  # Type: boolean
        self.sleepstate = None  # Type: choices
        self.wakeradius = None  # Type: float
        self.wakesquad = None  # Type: boolean
        self.enemyfilter = None  # Type: target_destination
        self.ignoreunseenenemies = None  # Type: boolean
        self.physdamagescale = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        ResponseContext.from_dict(instance, entity_data)
        ToggleDraw.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        DamageFilter.from_dict(instance, entity_data)
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.squadname = entity_data.get('squadname', None)  # Type: string
        instance.hintgroup = entity_data.get('hintgroup', None)  # Type: string
        instance.hintlimiting = entity_data.get('hintlimiting', None)  # Type: boolean
        instance.sleepstate = entity_data.get('sleepstate', None)  # Type: choices
        instance.wakeradius = float(entity_data.get('wakeradius', 0))  # Type: float
        instance.wakesquad = entity_data.get('wakesquad', None)  # Type: boolean
        instance.enemyfilter = entity_data.get('enemyfilter', None)  # Type: target_destination
        instance.ignoreunseenenemies = entity_data.get('ignoreunseenenemies', None)  # Type: boolean
        instance.physdamagescale = float(entity_data.get('physdamagescale', 1.0))  # Type: float


class info_npc_spawn_destination(Targetname, Angles, Parentname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.ReuseDelay = 1  # Type: float
        self.RenameNPC = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.ReuseDelay = float(entity_data.get('reusedelay', 1))  # Type: float
        instance.RenameNPC = entity_data.get('renamenpc', None)  # Type: string


class BaseNPCMaker(Targetname, Angles, EnableDisable):
    icon_sprite = "editor/npc_maker.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(EnableDisable).__init__()
        self.MaxNPCCount = 1  # Type: integer
        self.SpawnFrequency = "5"  # Type: string
        self.MaxLiveChildren = 5  # Type: integer
        self.HullCheckMode = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.MaxNPCCount = parse_source_value(entity_data.get('maxnpccount', 1))  # Type: integer
        instance.SpawnFrequency = entity_data.get('spawnfrequency', "5")  # Type: string
        instance.MaxLiveChildren = parse_source_value(entity_data.get('maxlivechildren', 5))  # Type: integer
        instance.HullCheckMode = entity_data.get('hullcheckmode', None)  # Type: choices


class npc_template_maker(BaseNPCMaker):
    icon_sprite = "editor/npc_maker.vmt"
    def __init__(self):
        super(BaseNPCMaker).__init__()
        self.origin = [0, 0, 0]
        self.TemplateName = None  # Type: target_destination
        self.Radius = 256  # Type: float
        self.DestinationGroup = None  # Type: target_destination
        self.CriterionVisibility = "CHOICES NOT SUPPORTED"  # Type: choices
        self.CriterionDistance = "CHOICES NOT SUPPORTED"  # Type: choices
        self.MinSpawnDistance = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPCMaker.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.TemplateName = entity_data.get('templatename', None)  # Type: target_destination
        instance.Radius = float(entity_data.get('radius', 256))  # Type: float
        instance.DestinationGroup = entity_data.get('destinationgroup', None)  # Type: target_destination
        instance.CriterionVisibility = entity_data.get('criterionvisibility', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.CriterionDistance = entity_data.get('criteriondistance', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.MinSpawnDistance = parse_source_value(entity_data.get('minspawndistance', 0))  # Type: integer


class BaseHelicopter(BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        self.InitialSpeed = "0"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.InitialSpeed = entity_data.get('initialspeed', "0")  # Type: string


class PlayerClass(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class Light(Base):
    def __init__(self):
        super().__init__()
        self._light = [255, 255, 255, 200]  # Type: color255
        self._lightHDR = [-1, -1, -1, 1]  # Type: color255
        self._lightscaleHDR = 1  # Type: float
        self.style = None  # Type: choices
        self.pattern = None  # Type: string
        self._constant_attn = "0"  # Type: string
        self._linear_attn = "0"  # Type: string
        self._quadratic_attn = "1"  # Type: string
        self._fifty_percent_distance = "0"  # Type: string
        self._zero_percent_distance = "0"  # Type: string
        self._hardfalloff = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance._light = parse_int_vector(entity_data.get('_light', "255 255 255 200"))  # Type: color255
        instance._lightHDR = parse_int_vector(entity_data.get('_lighthdr', "-1 -1 -1 1"))  # Type: color255
        instance._lightscaleHDR = float(entity_data.get('_lightscalehdr', 1))  # Type: float
        instance.style = entity_data.get('style', None)  # Type: choices
        instance.pattern = entity_data.get('pattern', None)  # Type: string
        instance._constant_attn = entity_data.get('_constant_attn', "0")  # Type: string
        instance._linear_attn = entity_data.get('_linear_attn', "0")  # Type: string
        instance._quadratic_attn = entity_data.get('_quadratic_attn', "1")  # Type: string
        instance._fifty_percent_distance = entity_data.get('_fifty_percent_distance', "0")  # Type: string
        instance._zero_percent_distance = entity_data.get('_zero_percent_distance', "0")  # Type: string
        instance._hardfalloff = parse_source_value(entity_data.get('_hardfalloff', 0))  # Type: integer


class Node(Base):
    def __init__(self):
        super().__init__()
        self.nodeid = None  # Type: node_id

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.nodeid = entity_data.get('nodeid', None)  # Type: node_id


class HintNode(Node):
    def __init__(self):
        super(Node).__init__()
        self.hinttype = None  # Type: choices
        self.generictype = None  # Type: string
        self.hintactivity = None  # Type: string
        self.nodeFOV = "CHOICES NOT SUPPORTED"  # Type: choices
        self.StartHintDisabled = None  # Type: boolean
        self.Group = None  # Type: string
        self.TargetNode = -1  # Type: node_dest
        self.radius = None  # Type: integer
        self.IgnoreFacing = "CHOICES NOT SUPPORTED"  # Type: choices
        self.MinimumState = "CHOICES NOT SUPPORTED"  # Type: choices
        self.MaximumState = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Node.from_dict(instance, entity_data)
        instance.hinttype = entity_data.get('hinttype', None)  # Type: choices
        instance.generictype = entity_data.get('generictype', None)  # Type: string
        instance.hintactivity = entity_data.get('hintactivity', None)  # Type: string
        instance.nodeFOV = entity_data.get('nodefov', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.StartHintDisabled = entity_data.get('starthintdisabled', None)  # Type: boolean
        instance.Group = entity_data.get('group', None)  # Type: string
        instance.TargetNode = parse_source_value(entity_data.get('targetnode', -1))  # Type: node_dest
        instance.radius = parse_source_value(entity_data.get('radius', 0))  # Type: integer
        instance.IgnoreFacing = entity_data.get('ignorefacing', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.MinimumState = entity_data.get('minimumstate', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.MaximumState = entity_data.get('maximumstate', "CHOICES NOT SUPPORTED")  # Type: choices


class TriggerOnce(Parentname, EnableDisable, Targetname, Global, Origin):
    def __init__(self):
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
        super(Origin).__init__()
        self.filtername = None  # Type: filterclass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.filtername = entity_data.get('filtername', None)  # Type: filterclass


class Trigger(TriggerOnce):
    def __init__(self):
        super(TriggerOnce).__init__()
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        super(Origin).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        TriggerOnce.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)


class worldbase(Base):
    def __init__(self):
        super().__init__()
        self.message = None  # Type: string
        self.skyname = "sky_black_nofog"  # Type: string
        self.chaptertitle = None  # Type: string
        self.startdark = None  # Type: boolean
        self.gametitle = None  # Type: boolean
        self.newunit = None  # Type: choices
        self.maxoccludeearea = 0  # Type: float
        self.minoccluderarea = 0  # Type: float
        self.maxoccludeearea_x360 = 0  # Type: float
        self.minoccluderarea_x360 = 0  # Type: float
        self.maxpropscreenwidth = -1  # Type: float
        self.minpropscreenwidth = None  # Type: float
        self.detailvbsp = "detail.vbsp"  # Type: string
        self.detailmaterial = "detail/detailsprites"  # Type: string
        self.coldworld = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.message = entity_data.get('message', None)  # Type: string
        instance.skyname = entity_data.get('skyname', "sky_black_nofog")  # Type: string
        instance.chaptertitle = entity_data.get('chaptertitle', None)  # Type: string
        instance.startdark = entity_data.get('startdark', None)  # Type: boolean
        instance.gametitle = entity_data.get('gametitle', None)  # Type: boolean
        instance.newunit = entity_data.get('newunit', None)  # Type: choices
        instance.maxoccludeearea = float(entity_data.get('maxoccludeearea', 0))  # Type: float
        instance.minoccluderarea = float(entity_data.get('minoccluderarea', 0))  # Type: float
        instance.maxoccludeearea_x360 = float(entity_data.get('maxoccludeearea_x360', 0))  # Type: float
        instance.minoccluderarea_x360 = float(entity_data.get('minoccluderarea_x360', 0))  # Type: float
        instance.maxpropscreenwidth = float(entity_data.get('maxpropscreenwidth', -1))  # Type: float
        instance.minpropscreenwidth = float(entity_data.get('minpropscreenwidth', 0))  # Type: float
        instance.detailvbsp = entity_data.get('detailvbsp', "detail.vbsp")  # Type: string
        instance.detailmaterial = entity_data.get('detailmaterial', "detail/detailsprites")  # Type: string
        instance.coldworld = entity_data.get('coldworld', None)  # Type: boolean


class worldspawn(Targetname, worldbase, ResponseContext):
    def __init__(self):
        super(Targetname).__init__()
        super(worldbase).__init__()
        super(ResponseContext).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        worldbase.from_dict(instance, entity_data)
        ResponseContext.from_dict(instance, entity_data)


class ambient_generic(Targetname):
    icon_sprite = "editor/ambient_generic.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.message = None  # Type: sound
        self.health = 10  # Type: integer
        self.preset = None  # Type: choices
        self.volstart = None  # Type: integer
        self.fadeinsecs = None  # Type: integer
        self.fadeoutsecs = None  # Type: integer
        self.pitch = 100  # Type: integer
        self.pitchstart = 100  # Type: integer
        self.spinup = None  # Type: integer
        self.spindown = None  # Type: integer
        self.lfotype = None  # Type: integer
        self.lforate = None  # Type: integer
        self.lfomodpitch = None  # Type: integer
        self.lfomodvol = None  # Type: integer
        self.cspinup = None  # Type: integer
        self.radius = "1250"  # Type: string
        self.SourceEntityName = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.message = entity_data.get('message', None)  # Type: sound
        instance.health = parse_source_value(entity_data.get('health', 10))  # Type: integer
        instance.preset = entity_data.get('preset', None)  # Type: choices
        instance.volstart = parse_source_value(entity_data.get('volstart', 0))  # Type: integer
        instance.fadeinsecs = parse_source_value(entity_data.get('fadeinsecs', 0))  # Type: integer
        instance.fadeoutsecs = parse_source_value(entity_data.get('fadeoutsecs', 0))  # Type: integer
        instance.pitch = parse_source_value(entity_data.get('pitch', 100))  # Type: integer
        instance.pitchstart = parse_source_value(entity_data.get('pitchstart', 100))  # Type: integer
        instance.spinup = parse_source_value(entity_data.get('spinup', 0))  # Type: integer
        instance.spindown = parse_source_value(entity_data.get('spindown', 0))  # Type: integer
        instance.lfotype = parse_source_value(entity_data.get('lfotype', 0))  # Type: integer
        instance.lforate = parse_source_value(entity_data.get('lforate', 0))  # Type: integer
        instance.lfomodpitch = parse_source_value(entity_data.get('lfomodpitch', 0))  # Type: integer
        instance.lfomodvol = parse_source_value(entity_data.get('lfomodvol', 0))  # Type: integer
        instance.cspinup = parse_source_value(entity_data.get('cspinup', 0))  # Type: integer
        instance.radius = entity_data.get('radius', "1250")  # Type: string
        instance.SourceEntityName = entity_data.get('sourceentityname', None)  # Type: target_destination


class func_lod(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.DisappearDist = 2000  # Type: integer
        self.Solid = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.DisappearDist = parse_source_value(entity_data.get('disappeardist', 2000))  # Type: integer
        instance.Solid = entity_data.get('solid', None)  # Type: choices


class env_zoom(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.Rate = 1.0  # Type: float
        self.FOV = 75  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Rate = float(entity_data.get('rate', 1.0))  # Type: float
        instance.FOV = parse_source_value(entity_data.get('fov', 75))  # Type: integer


class env_screenoverlay(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.OverlayName1 = None  # Type: string
        self.OverlayTime1 = 1.0  # Type: float
        self.OverlayName2 = None  # Type: string
        self.OverlayTime2 = 1.0  # Type: float
        self.OverlayName3 = None  # Type: string
        self.OverlayTime3 = 1.0  # Type: float
        self.OverlayName4 = None  # Type: string
        self.OverlayTime4 = 1.0  # Type: float
        self.OverlayName5 = None  # Type: string
        self.OverlayTime5 = 1.0  # Type: float
        self.OverlayName6 = None  # Type: string
        self.OverlayTime6 = 1.0  # Type: float
        self.OverlayName7 = None  # Type: string
        self.OverlayTime7 = 1.0  # Type: float
        self.OverlayName8 = None  # Type: string
        self.OverlayTime8 = 1.0  # Type: float
        self.OverlayName9 = None  # Type: string
        self.OverlayTime9 = 1.0  # Type: float
        self.OverlayName10 = None  # Type: string
        self.OverlayTime10 = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.OverlayName1 = entity_data.get('overlayname1', None)  # Type: string
        instance.OverlayTime1 = float(entity_data.get('overlaytime1', 1.0))  # Type: float
        instance.OverlayName2 = entity_data.get('overlayname2', None)  # Type: string
        instance.OverlayTime2 = float(entity_data.get('overlaytime2', 1.0))  # Type: float
        instance.OverlayName3 = entity_data.get('overlayname3', None)  # Type: string
        instance.OverlayTime3 = float(entity_data.get('overlaytime3', 1.0))  # Type: float
        instance.OverlayName4 = entity_data.get('overlayname4', None)  # Type: string
        instance.OverlayTime4 = float(entity_data.get('overlaytime4', 1.0))  # Type: float
        instance.OverlayName5 = entity_data.get('overlayname5', None)  # Type: string
        instance.OverlayTime5 = float(entity_data.get('overlaytime5', 1.0))  # Type: float
        instance.OverlayName6 = entity_data.get('overlayname6', None)  # Type: string
        instance.OverlayTime6 = float(entity_data.get('overlaytime6', 1.0))  # Type: float
        instance.OverlayName7 = entity_data.get('overlayname7', None)  # Type: string
        instance.OverlayTime7 = float(entity_data.get('overlaytime7', 1.0))  # Type: float
        instance.OverlayName8 = entity_data.get('overlayname8', None)  # Type: string
        instance.OverlayTime8 = float(entity_data.get('overlaytime8', 1.0))  # Type: float
        instance.OverlayName9 = entity_data.get('overlayname9', None)  # Type: string
        instance.OverlayTime9 = float(entity_data.get('overlaytime9', 1.0))  # Type: float
        instance.OverlayName10 = entity_data.get('overlayname10', None)  # Type: string
        instance.OverlayTime10 = float(entity_data.get('overlaytime10', 1.0))  # Type: float


class env_screeneffect(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.type = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.type = entity_data.get('type', None)  # Type: choices


class env_texturetoggle(Targetname):
    icon_sprite = "editor/env_texturetoggle.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination


class env_splash(Targetname, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.scale = 8.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.scale = float(entity_data.get('scale', 8.0))  # Type: float


class env_particlelight(Parentname):
    def __init__(self):
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.Color = [255, 0, 0]  # Type: color255
        self.Intensity = 5000  # Type: integer
        self.directional = None  # Type: boolean
        self.PSName = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Color = parse_int_vector(entity_data.get('color', "255 0 0"))  # Type: color255
        instance.Intensity = parse_source_value(entity_data.get('intensity', 5000))  # Type: integer
        instance.directional = entity_data.get('directional', None)  # Type: boolean
        instance.PSName = entity_data.get('psname', None)  # Type: string


class env_sun(Targetname, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.use_angles = None  # Type: boolean
        self.pitch = None  # Type: integer
        self.rendercolor = [100, 80, 80]  # Type: color255
        self.overlaycolor = [0, 0, 0]  # Type: color255
        self.size = 16  # Type: integer
        self.overlaysize = -1  # Type: integer
        self.material = "sprites/light_glow02_add_noz"  # Type: sprite
        self.overlaymaterial = "sprites/light_glow02_add_noz"  # Type: sprite
        self.HDRColorScale = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.use_angles = entity_data.get('use_angles', None)  # Type: boolean
        instance.pitch = parse_source_value(entity_data.get('pitch', 0))  # Type: integer
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "100 80 80"))  # Type: color255
        instance.overlaycolor = parse_int_vector(entity_data.get('overlaycolor', "0 0 0"))  # Type: color255
        instance.size = parse_source_value(entity_data.get('size', 16))  # Type: integer
        instance.overlaysize = parse_source_value(entity_data.get('overlaysize', -1))  # Type: integer
        instance.material = entity_data.get('material', "sprites/light_glow02_add_noz")  # Type: sprite
        instance.overlaymaterial = entity_data.get('overlaymaterial', "sprites/light_glow02_add_noz")  # Type: sprite
        instance.HDRColorScale = float(entity_data.get('hdrcolorscale', 1.0))  # Type: float


class game_ragdoll_manager(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.MaxRagdollCount = -1  # Type: integer
        self.MaxRagdollCountDX8 = -1  # Type: integer
        self.SaveImportant = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.MaxRagdollCount = parse_source_value(entity_data.get('maxragdollcount', -1))  # Type: integer
        instance.MaxRagdollCountDX8 = parse_source_value(entity_data.get('maxragdollcountdx8', -1))  # Type: integer
        instance.SaveImportant = entity_data.get('saveimportant', None)  # Type: boolean


class game_gib_manager(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.maxpieces = -1  # Type: integer
        self.maxpiecesdx8 = -1  # Type: integer
        self.allownewgibs = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.maxpieces = parse_source_value(entity_data.get('maxpieces', -1))  # Type: integer
        instance.maxpiecesdx8 = parse_source_value(entity_data.get('maxpiecesdx8', -1))  # Type: integer
        instance.allownewgibs = entity_data.get('allownewgibs', None)  # Type: boolean


class env_dof_controller(Targetname):
    icon_sprite = "editor/env_dof_controller.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.enabled = None  # Type: boolean
        self.near_blur = 20  # Type: float
        self.near_focus = 100  # Type: float
        self.near_radius = 8  # Type: float
        self.far_blur = 1000  # Type: float
        self.far_focus = 500  # Type: float
        self.far_radius = 8  # Type: float
        self.focus_target = None  # Type: target_source
        self.focus_range = 200  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.enabled = entity_data.get('enabled', None)  # Type: boolean
        instance.near_blur = float(entity_data.get('near_blur', 20))  # Type: float
        instance.near_focus = float(entity_data.get('near_focus', 100))  # Type: float
        instance.near_radius = float(entity_data.get('near_radius', 8))  # Type: float
        instance.far_blur = float(entity_data.get('far_blur', 1000))  # Type: float
        instance.far_focus = float(entity_data.get('far_focus', 500))  # Type: float
        instance.far_radius = float(entity_data.get('far_radius', 8))  # Type: float
        instance.focus_target = entity_data.get('focus_target', None)  # Type: target_source
        instance.focus_range = float(entity_data.get('focus_range', 200))  # Type: float


class env_lightglow(Parentname, Angles, Targetname):
    model = "models/editor/axis_helper_thick.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.rendercolor = [255, 255, 255]  # Type: color255
        self.VerticalGlowSize = 30  # Type: integer
        self.HorizontalGlowSize = 30  # Type: integer
        self.MinDist = 500  # Type: integer
        self.MaxDist = 2000  # Type: integer
        self.OuterMaxDist = None  # Type: integer
        self.GlowProxySize = 2.0  # Type: float
        self.HDRColorScale = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "255 255 255"))  # Type: color255
        instance.VerticalGlowSize = parse_source_value(entity_data.get('verticalglowsize', 30))  # Type: integer
        instance.HorizontalGlowSize = parse_source_value(entity_data.get('horizontalglowsize', 30))  # Type: integer
        instance.MinDist = parse_source_value(entity_data.get('mindist', 500))  # Type: integer
        instance.MaxDist = parse_source_value(entity_data.get('maxdist', 2000))  # Type: integer
        instance.OuterMaxDist = parse_source_value(entity_data.get('outermaxdist', 0))  # Type: integer
        instance.GlowProxySize = float(entity_data.get('glowproxysize', 2.0))  # Type: float
        instance.HDRColorScale = float(entity_data.get('hdrcolorscale', 1.0))  # Type: float


class env_smokestack(Parentname, Angles):
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.targetname = None  # Type: target_source
        self.InitialState = None  # Type: choices
        self.BaseSpread = 20  # Type: integer
        self.SpreadSpeed = 15  # Type: integer
        self.Speed = 30  # Type: integer
        self.StartSize = 20  # Type: integer
        self.EndSize = 30  # Type: integer
        self.Rate = 20  # Type: integer
        self.JetLength = 180  # Type: integer
        self.WindAngle = None  # Type: integer
        self.WindSpeed = None  # Type: integer
        self.SmokeMaterial = "particle/SmokeStack.vmt"  # Type: string
        self.twist = None  # Type: integer
        self.roll = None  # Type: float
        self.rendercolor = [255, 255, 255]  # Type: color255
        self.renderamt = 255  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.targetname = entity_data.get('targetname', None)  # Type: target_source
        instance.InitialState = entity_data.get('initialstate', None)  # Type: choices
        instance.BaseSpread = parse_source_value(entity_data.get('basespread', 20))  # Type: integer
        instance.SpreadSpeed = parse_source_value(entity_data.get('spreadspeed', 15))  # Type: integer
        instance.Speed = parse_source_value(entity_data.get('speed', 30))  # Type: integer
        instance.StartSize = parse_source_value(entity_data.get('startsize', 20))  # Type: integer
        instance.EndSize = parse_source_value(entity_data.get('endsize', 30))  # Type: integer
        instance.Rate = parse_source_value(entity_data.get('rate', 20))  # Type: integer
        instance.JetLength = parse_source_value(entity_data.get('jetlength', 180))  # Type: integer
        instance.WindAngle = parse_source_value(entity_data.get('windangle', 0))  # Type: integer
        instance.WindSpeed = parse_source_value(entity_data.get('windspeed', 0))  # Type: integer
        instance.SmokeMaterial = entity_data.get('smokematerial', "particle/SmokeStack.vmt")  # Type: string
        instance.twist = parse_source_value(entity_data.get('twist', 0))  # Type: integer
        instance.roll = float(entity_data.get('roll', 0))  # Type: float
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "255 255 255"))  # Type: color255
        instance.renderamt = parse_source_value(entity_data.get('renderamt', 255))  # Type: integer


class env_fade(Targetname):
    icon_sprite = "editor/env_fade"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.duration = "2"  # Type: string
        self.holdtime = "0"  # Type: string
        self.renderamt = 255  # Type: integer
        self.rendercolor = [0, 0, 0]  # Type: color255
        self.ReverseFadeDuration = 2  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.duration = entity_data.get('duration', "2")  # Type: string
        instance.holdtime = entity_data.get('holdtime', "0")  # Type: string
        instance.renderamt = parse_source_value(entity_data.get('renderamt', 255))  # Type: integer
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "0 0 0"))  # Type: color255
        instance.ReverseFadeDuration = float(entity_data.get('reversefadeduration', 2))  # Type: float


class env_player_surface_trigger(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.gamematerial = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.gamematerial = entity_data.get('gamematerial', "CHOICES NOT SUPPORTED")  # Type: choices


class env_tonemap_controller(Targetname):
    icon_sprite = "editor/env_tonemap_controller.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class func_areaportalwindow(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.target = None  # Type: target_destination
        self.FadeStartDist = 128  # Type: integer
        self.FadeDist = 512  # Type: integer
        self.TranslucencyLimit = "0.2"  # Type: string
        self.BackgroundBModel = None  # Type: string
        self.PortalVersion = 1  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.FadeStartDist = parse_source_value(entity_data.get('fadestartdist', 128))  # Type: integer
        instance.FadeDist = parse_source_value(entity_data.get('fadedist', 512))  # Type: integer
        instance.TranslucencyLimit = entity_data.get('translucencylimit', "0.2")  # Type: string
        instance.BackgroundBModel = entity_data.get('backgroundbmodel', None)  # Type: string
        instance.PortalVersion = parse_source_value(entity_data.get('portalversion', 1))  # Type: integer


class func_wall(Targetname, Shadow, Global, RenderFields):
    def __init__(self):
        super(RenderFields).__init__()
        super(Targetname).__init__()
        super(Shadow).__init__()
        super(Global).__init__()
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class func_clip_vphysics(Targetname, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.filtername = None  # Type: filterclass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.filtername = entity_data.get('filtername', None)  # Type: filterclass


class func_brush(Parentname, Shadow, EnableDisable, Targetname, RenderFields, Inputfilter, PaintableBrush, Global, Reflection, Origin):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Shadow).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        super(Inputfilter).__init__()
        super(PaintableBrush).__init__()
        super(Global).__init__()
        super(Reflection).__init__()
        super(Origin).__init__()
        self._minlight = None  # Type: string
        self.Solidity = None  # Type: choices
        self.excludednpc = None  # Type: string
        self.invert_exclusion = None  # Type: choices
        self.solidbsp = None  # Type: boolean
        self.vrad_brush_cast_shadows = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Inputfilter.from_dict(instance, entity_data)
        PaintableBrush.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Reflection.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.Solidity = entity_data.get('solidity', None)  # Type: choices
        instance.excludednpc = entity_data.get('excludednpc', None)  # Type: string
        instance.invert_exclusion = entity_data.get('invert_exclusion', None)  # Type: choices
        instance.solidbsp = entity_data.get('solidbsp', None)  # Type: boolean
        instance.vrad_brush_cast_shadows = entity_data.get('vrad_brush_cast_shadows', None)  # Type: boolean


class vgui_screen_base(Targetname, Angles, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.panelname = None  # Type: string
        self.overlaymaterial = None  # Type: string
        self.width = 32  # Type: integer
        self.height = 32  # Type: integer
        self.IsTransparent = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.panelname = entity_data.get('panelname', None)  # Type: string
        instance.overlaymaterial = entity_data.get('overlaymaterial', None)  # Type: string
        instance.width = parse_source_value(entity_data.get('width', 32))  # Type: integer
        instance.height = parse_source_value(entity_data.get('height', 32))  # Type: integer
        instance.IsTransparent = entity_data.get('istransparent', None)  # Type: boolean


class vgui_screen(vgui_screen_base):
    def __init__(self):
        super(vgui_screen_base).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        vgui_screen_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class vgui_slideshow_display(Targetname, Angles, Parentname):
    model = "models/editor/axis_helper_thick.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.displaytext = None  # Type: string
        self.directory = "slideshow"  # Type: string
        self.minslidetime = 0.5  # Type: float
        self.maxslidetime = 0.5  # Type: float
        self.cycletype = None  # Type: choices
        self.nolistrepeat = None  # Type: choices
        self.width = 256  # Type: integer
        self.height = 128  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.displaytext = entity_data.get('displaytext', None)  # Type: string
        instance.directory = entity_data.get('directory', "slideshow")  # Type: string
        instance.minslidetime = float(entity_data.get('minslidetime', 0.5))  # Type: float
        instance.maxslidetime = float(entity_data.get('maxslidetime', 0.5))  # Type: float
        instance.cycletype = entity_data.get('cycletype', None)  # Type: choices
        instance.nolistrepeat = entity_data.get('nolistrepeat', None)  # Type: choices
        instance.width = parse_source_value(entity_data.get('width', 256))  # Type: integer
        instance.height = parse_source_value(entity_data.get('height', 128))  # Type: integer


class vgui_movie_display(Targetname, Angles, Parentname):
    model = "models/editor/axis_helper_thick.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.displaytext = None  # Type: string
        self.moviefilename = "media/"  # Type: string
        self.groupname = None  # Type: string
        self.looping = None  # Type: boolean
        self.stretch = None  # Type: boolean
        self.forcedslave = None  # Type: boolean
        self.forceprecache = None  # Type: boolean
        self.width = 256  # Type: integer
        self.height = 128  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.displaytext = entity_data.get('displaytext', None)  # Type: string
        instance.moviefilename = entity_data.get('moviefilename', "media/")  # Type: string
        instance.groupname = entity_data.get('groupname', None)  # Type: string
        instance.looping = entity_data.get('looping', None)  # Type: boolean
        instance.stretch = entity_data.get('stretch', None)  # Type: boolean
        instance.forcedslave = entity_data.get('forcedslave', None)  # Type: boolean
        instance.forceprecache = entity_data.get('forceprecache', None)  # Type: boolean
        instance.width = parse_source_value(entity_data.get('width', 256))  # Type: integer
        instance.height = parse_source_value(entity_data.get('height', 128))  # Type: integer


class cycler(Parentname, Targetname, RenderFields, Angles):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(RenderFxChoices).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.model = None  # Type: studio
        self.skin = None  # Type: integer
        self.sequence = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        RenderFxChoices.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', None)  # Type: studio
        instance.skin = parse_source_value(entity_data.get('skin', 0))  # Type: integer
        instance.sequence = parse_source_value(entity_data.get('sequence', 0))  # Type: integer


class gibshooterbase(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.angles = "0 0 0"  # Type: string
        self.m_iGibs = 3  # Type: integer
        self.delay = "0"  # Type: string
        self.gibangles = "0 0 0"  # Type: string
        self.gibanglevelocity = "0"  # Type: string
        self.m_flVelocity = 200  # Type: integer
        self.m_flVariance = "0.15"  # Type: string
        self.m_flGibLife = "4"  # Type: string
        self.lightingorigin = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.angles = entity_data.get('angles', "0 0 0")  # Type: string
        instance.m_iGibs = parse_source_value(entity_data.get('m_igibs', 3))  # Type: integer
        instance.delay = entity_data.get('delay', "0")  # Type: string
        instance.gibangles = entity_data.get('gibangles', "0 0 0")  # Type: string
        instance.gibanglevelocity = entity_data.get('gibanglevelocity', "0")  # Type: string
        instance.m_flVelocity = parse_source_value(entity_data.get('m_flvelocity', 200))  # Type: integer
        instance.m_flVariance = entity_data.get('m_flvariance', "0.15")  # Type: string
        instance.m_flGibLife = entity_data.get('m_flgiblife', "4")  # Type: string
        instance.lightingorigin = entity_data.get('lightingorigin', None)  # Type: target_destination


class env_beam(Targetname, Parentname, RenderFxChoices):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(RenderFxChoices).__init__()
        self.origin = [0, 0, 0]
        self.renderamt = 100  # Type: integer
        self.rendercolor = [255, 255, 255]  # Type: color255
        self.Radius = 256  # Type: integer
        self.life = "1"  # Type: string
        self.BoltWidth = 2  # Type: float
        self.NoiseAmplitude = None  # Type: float
        self.texture = "sprites/laserbeam.spr"  # Type: sprite
        self.TextureScroll = 35  # Type: integer
        self.framerate = None  # Type: integer
        self.framestart = None  # Type: integer
        self.StrikeTime = "1"  # Type: string
        self.damage = "0"  # Type: string
        self.LightningStart = None  # Type: target_destination
        self.LightningEnd = None  # Type: target_destination
        self.decalname = "Bigshot"  # Type: string
        self.HDRColorScale = 1.0  # Type: float
        self.targetpoint = [0, 0, 0]  # Type: vecline
        self.TouchType = None  # Type: choices
        self.ClipStyle = None  # Type: choices
        self.filtername = None  # Type: filterclass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        RenderFxChoices.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "[0 0 0]"))
        instance.renderamt = parse_source_value(entity_data.get('renderamt', 100))  # Type: integer
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "255 255 255"))  # Type: color255
        instance.Radius = parse_source_value(entity_data.get('radius', 256))  # Type: integer
        instance.life = entity_data.get('life', "1")  # Type: string
        instance.BoltWidth = float(entity_data.get('boltwidth', 2))  # Type: float
        instance.NoiseAmplitude = float(entity_data.get('noiseamplitude', 0))  # Type: float
        instance.texture = entity_data.get('texture', "sprites/laserbeam.spr")  # Type: sprite
        instance.TextureScroll = parse_source_value(entity_data.get('texturescroll', 35))  # Type: integer
        instance.framerate = parse_source_value(entity_data.get('framerate', 0))  # Type: integer
        instance.framestart = parse_source_value(entity_data.get('framestart', 0))  # Type: integer
        instance.StrikeTime = entity_data.get('striketime', "1")  # Type: string
        instance.damage = entity_data.get('damage', "0")  # Type: string
        instance.LightningStart = entity_data.get('lightningstart', None)  # Type: target_destination
        instance.LightningEnd = entity_data.get('lightningend', None)  # Type: target_destination
        instance.decalname = entity_data.get('decalname', "Bigshot")  # Type: string
        instance.HDRColorScale = float(entity_data.get('hdrcolorscale', 1.0))  # Type: float
        instance.targetpoint = entity_data.get('targetpoint', None)  # Type: vecline
        instance.TouchType = entity_data.get('touchtype', None)  # Type: choices
        instance.ClipStyle = entity_data.get('clipstyle', None)  # Type: choices
        instance.filtername = entity_data.get('filtername', None)  # Type: filterclass


class env_embers(Targetname, Angles, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.particletype = None  # Type: choices
        self.density = 50  # Type: integer
        self.lifetime = 4  # Type: integer
        self.speed = 32  # Type: integer
        self.rendercolor = [255, 255, 255]  # Type: color255

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.particletype = entity_data.get('particletype', None)  # Type: choices
        instance.density = parse_source_value(entity_data.get('density', 50))  # Type: integer
        instance.lifetime = parse_source_value(entity_data.get('lifetime', 4))  # Type: integer
        instance.speed = parse_source_value(entity_data.get('speed', 32))  # Type: integer
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "255 255 255"))  # Type: color255


class env_funnel(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_blood(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.spraydir = [0.0, 0.0, 0.0]  # Type: angle
        self.color = None  # Type: choices
        self.amount = "100"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.spraydir = parse_float_vector(entity_data.get('spraydir', "0 0 0"))  # Type: angle
        instance.color = entity_data.get('color', None)  # Type: choices
        instance.amount = entity_data.get('amount', "100")  # Type: string


class env_bubbles(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.density = 2  # Type: integer
        self.frequency = 2  # Type: integer
        self.current = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.density = parse_source_value(entity_data.get('density', 2))  # Type: integer
        instance.frequency = parse_source_value(entity_data.get('frequency', 2))  # Type: integer
        instance.current = parse_source_value(entity_data.get('current', 0))  # Type: integer


class env_explosion(Targetname, Parentname):
    icon_sprite = "editor/env_explosion.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.iMagnitude = 100  # Type: integer
        self.iRadiusOverride = None  # Type: integer
        self.fireballsprite = "sprites/zerogxplode.spr"  # Type: sprite
        self.rendermode = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ignoredEntity = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.iMagnitude = parse_source_value(entity_data.get('imagnitude', 100))  # Type: integer
        instance.iRadiusOverride = parse_source_value(entity_data.get('iradiusoverride', 0))  # Type: integer
        instance.fireballsprite = entity_data.get('fireballsprite', "sprites/zerogxplode.spr")  # Type: sprite
        instance.rendermode = entity_data.get('rendermode', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ignoredEntity = entity_data.get('ignoredentity', None)  # Type: target_destination


class env_smoketrail(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.opacity = 0.75  # Type: float
        self.spawnrate = 20  # Type: float
        self.lifetime = 5.0  # Type: float
        self.startcolor = [192, 192, 192]  # Type: color255
        self.endcolor = [160, 160, 160]  # Type: color255
        self.emittime = 0  # Type: float
        self.minspeed = 10  # Type: float
        self.maxspeed = 20  # Type: float
        self.mindirectedspeed = 0  # Type: float
        self.maxdirectedspeed = 0  # Type: float
        self.startsize = 15  # Type: float
        self.endsize = 50  # Type: float
        self.spawnradius = 15  # Type: float
        self.firesprite = "sprites/firetrail.spr"  # Type: sprite
        self.smokesprite = "sprites/whitepuff.spr"  # Type: sprite

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.opacity = float(entity_data.get('opacity', 0.75))  # Type: float
        instance.spawnrate = float(entity_data.get('spawnrate', 20))  # Type: float
        instance.lifetime = float(entity_data.get('lifetime', 5.0))  # Type: float
        instance.startcolor = parse_int_vector(entity_data.get('startcolor', "192 192 192"))  # Type: color255
        instance.endcolor = parse_int_vector(entity_data.get('endcolor', "160 160 160"))  # Type: color255
        instance.emittime = float(entity_data.get('emittime', 0))  # Type: float
        instance.minspeed = float(entity_data.get('minspeed', 10))  # Type: float
        instance.maxspeed = float(entity_data.get('maxspeed', 20))  # Type: float
        instance.mindirectedspeed = float(entity_data.get('mindirectedspeed', 0))  # Type: float
        instance.maxdirectedspeed = float(entity_data.get('maxdirectedspeed', 0))  # Type: float
        instance.startsize = float(entity_data.get('startsize', 15))  # Type: float
        instance.endsize = float(entity_data.get('endsize', 50))  # Type: float
        instance.spawnradius = float(entity_data.get('spawnradius', 15))  # Type: float
        instance.firesprite = entity_data.get('firesprite', "sprites/firetrail.spr")  # Type: sprite
        instance.smokesprite = entity_data.get('smokesprite', "sprites/whitepuff.spr")  # Type: sprite


class env_physexplosion(Targetname, Parentname):
    icon_sprite = "editor/env_physexplosion.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.magnitude = "100"  # Type: string
        self.radius = "0"  # Type: string
        self.targetentityname = None  # Type: target_destination
        self.inner_radius = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.magnitude = entity_data.get('magnitude', "100")  # Type: string
        instance.radius = entity_data.get('radius', "0")  # Type: string
        instance.targetentityname = entity_data.get('targetentityname', None)  # Type: target_destination
        instance.inner_radius = float(entity_data.get('inner_radius', 0))  # Type: float


class env_physimpact(Targetname, Parentname):
    icon_sprite = "editor/env_physexplosion.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.angles = "0 0 0"  # Type: string
        self.magnitude = 100  # Type: integer
        self.distance = None  # Type: integer
        self.directionentityname = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.angles = entity_data.get('angles', "0 0 0")  # Type: string
        instance.magnitude = parse_source_value(entity_data.get('magnitude', 100))  # Type: integer
        instance.distance = parse_source_value(entity_data.get('distance', 0))  # Type: integer
        instance.directionentityname = entity_data.get('directionentityname', None)  # Type: target_destination


class env_fire(Targetname, Parentname, EnableDisable):
    icon_sprite = "editor/env_fire"
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.health = 30  # Type: integer
        self.firesize = 64  # Type: integer
        self.fireattack = 4  # Type: integer
        self.firetype = None  # Type: choices
        self.ignitionpoint = 32  # Type: float
        self.damagescale = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.health = parse_source_value(entity_data.get('health', 30))  # Type: integer
        instance.firesize = parse_source_value(entity_data.get('firesize', 64))  # Type: integer
        instance.fireattack = parse_source_value(entity_data.get('fireattack', 4))  # Type: integer
        instance.firetype = entity_data.get('firetype', None)  # Type: choices
        instance.ignitionpoint = float(entity_data.get('ignitionpoint', 32))  # Type: float
        instance.damagescale = float(entity_data.get('damagescale', 1.0))  # Type: float


class env_firesource(Targetname, Parentname):
    icon_sprite = "editor/env_firesource"
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.fireradius = 128  # Type: float
        self.firedamage = 10  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.fireradius = float(entity_data.get('fireradius', 128))  # Type: float
        instance.firedamage = float(entity_data.get('firedamage', 10))  # Type: float


class env_firesensor(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.fireradius = 128  # Type: float
        self.heatlevel = 32  # Type: float
        self.heattime = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.fireradius = float(entity_data.get('fireradius', 128))  # Type: float
        instance.heatlevel = float(entity_data.get('heatlevel', 32))  # Type: float
        instance.heattime = float(entity_data.get('heattime', 0))  # Type: float


class env_entity_igniter(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.lifetime = 10  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.lifetime = float(entity_data.get('lifetime', 10))  # Type: float


class env_fog_controller(Targetname, Angles, SystemLevelChoice):
    icon_sprite = "editor/fog_controller.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(SystemLevelChoice).__init__()
        self.origin = [0, 0, 0]
        self.fogenable = None  # Type: boolean
        self.fogblend = None  # Type: boolean
        self.use_angles = None  # Type: boolean
        self.fogcolor = [255, 255, 255]  # Type: color255
        self.fogcolor2 = [255, 255, 255]  # Type: color255
        self.fogdir = "1 0 0"  # Type: string
        self.fogstart = "500.0"  # Type: string
        self.fogend = "2000.0"  # Type: string
        self.fogmaxdensity = 1  # Type: float
        self.foglerptime = 0  # Type: float
        self.farz = "-1"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        SystemLevelChoice.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.fogenable = entity_data.get('fogenable', None)  # Type: boolean
        instance.fogblend = entity_data.get('fogblend', None)  # Type: boolean
        instance.use_angles = entity_data.get('use_angles', None)  # Type: boolean
        instance.fogcolor = parse_int_vector(entity_data.get('fogcolor', "255 255 255"))  # Type: color255
        instance.fogcolor2 = parse_int_vector(entity_data.get('fogcolor2', "255 255 255"))  # Type: color255
        instance.fogdir = entity_data.get('fogdir', "1 0 0")  # Type: string
        instance.fogstart = entity_data.get('fogstart', "500.0")  # Type: string
        instance.fogend = entity_data.get('fogend', "2000.0")  # Type: string
        instance.fogmaxdensity = float(entity_data.get('fogmaxdensity', 1))  # Type: float
        instance.foglerptime = float(entity_data.get('foglerptime', 0))  # Type: float
        instance.farz = entity_data.get('farz', "-1")  # Type: string


class env_steam(Targetname, Angles, Parentname):
    viewport_model = "models/editor/spot_cone.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.InitialState = None  # Type: choices
        self.type = None  # Type: choices
        self.SpreadSpeed = 15  # Type: integer
        self.Speed = 120  # Type: integer
        self.StartSize = 10  # Type: integer
        self.EndSize = 25  # Type: integer
        self.Rate = 26  # Type: integer
        self.rendercolor = [255, 255, 255]  # Type: color255
        self.JetLength = 80  # Type: integer
        self.renderamt = 255  # Type: integer
        self.rollspeed = 8  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.InitialState = entity_data.get('initialstate', None)  # Type: choices
        instance.type = entity_data.get('type', None)  # Type: choices
        instance.SpreadSpeed = parse_source_value(entity_data.get('spreadspeed', 15))  # Type: integer
        instance.Speed = parse_source_value(entity_data.get('speed', 120))  # Type: integer
        instance.StartSize = parse_source_value(entity_data.get('startsize', 10))  # Type: integer
        instance.EndSize = parse_source_value(entity_data.get('endsize', 25))  # Type: integer
        instance.Rate = parse_source_value(entity_data.get('rate', 26))  # Type: integer
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "255 255 255"))  # Type: color255
        instance.JetLength = parse_source_value(entity_data.get('jetlength', 80))  # Type: integer
        instance.renderamt = parse_source_value(entity_data.get('renderamt', 255))  # Type: integer
        instance.rollspeed = float(entity_data.get('rollspeed', 8))  # Type: float


class env_laser(Targetname, Parentname, RenderFxChoices):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(RenderFxChoices).__init__()
        self.origin = [0, 0, 0]
        self.LaserTarget = None  # Type: target_destination
        self.renderamt = 100  # Type: integer
        self.rendercolor = [255, 255, 255]  # Type: color255
        self.width = 2  # Type: float
        self.NoiseAmplitude = None  # Type: integer
        self.texture = "sprites/laserbeam.spr"  # Type: sprite
        self.EndSprite = None  # Type: sprite
        self.TextureScroll = 35  # Type: integer
        self.framestart = None  # Type: integer
        self.damage = "100"  # Type: string
        self.dissolvetype = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        RenderFxChoices.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.LaserTarget = entity_data.get('lasertarget', None)  # Type: target_destination
        instance.renderamt = parse_source_value(entity_data.get('renderamt', 100))  # Type: integer
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "255 255 255"))  # Type: color255
        instance.width = float(entity_data.get('width', 2))  # Type: float
        instance.NoiseAmplitude = parse_source_value(entity_data.get('noiseamplitude', 0))  # Type: integer
        instance.texture = entity_data.get('texture', "sprites/laserbeam.spr")  # Type: sprite
        instance.EndSprite = entity_data.get('endsprite', None)  # Type: sprite
        instance.TextureScroll = parse_source_value(entity_data.get('texturescroll', 35))  # Type: integer
        instance.framestart = parse_source_value(entity_data.get('framestart', 0))  # Type: integer
        instance.damage = entity_data.get('damage', "100")  # Type: string
        instance.dissolvetype = entity_data.get('dissolvetype', "CHOICES NOT SUPPORTED")  # Type: choices


class env_message(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.message = None  # Type: string
        self.messagesound = None  # Type: sound
        self.messagevolume = "10"  # Type: string
        self.messageattenuation = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.message = entity_data.get('message', None)  # Type: string
        instance.messagesound = entity_data.get('messagesound', None)  # Type: sound
        instance.messagevolume = entity_data.get('messagevolume', "10")  # Type: string
        instance.messageattenuation = entity_data.get('messageattenuation', None)  # Type: choices


class env_hudhint(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.message = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.message = entity_data.get('message', None)  # Type: string


class env_shake(Targetname, Parentname):
    icon_sprite = "editor/env_shake.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.amplitude = 4  # Type: float
        self.radius = 500  # Type: float
        self.duration = 1  # Type: float
        self.frequency = 2.5  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.amplitude = float(entity_data.get('amplitude', 4))  # Type: float
        instance.radius = float(entity_data.get('radius', 500))  # Type: float
        instance.duration = float(entity_data.get('duration', 1))  # Type: float
        instance.frequency = float(entity_data.get('frequency', 2.5))  # Type: float


class env_tilt(Targetname, Angles, Parentname):
    model = "models/editor/axis_helper_thick.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.radius = 500  # Type: float
        self.duration = 1  # Type: float
        self.tilttime = 2.5  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.radius = float(entity_data.get('radius', 500))  # Type: float
        instance.duration = float(entity_data.get('duration', 1))  # Type: float
        instance.tilttime = float(entity_data.get('tilttime', 2.5))  # Type: float


class env_viewpunch(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.punchangle = [0.0, 0.0, 90.0]  # Type: angle
        self.radius = 500  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.punchangle = parse_float_vector(entity_data.get('punchangle', "0 0 90"))  # Type: angle
        instance.radius = float(entity_data.get('radius', 500))  # Type: float


class gibshooter(gibshooterbase):
    icon_sprite = "editor/gibshooter.vmt"
    def __init__(self):
        super(gibshooterbase).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        gibshooterbase.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_shooter(gibshooterbase, RenderFields):
    icon_sprite = "editor/env_shooter.vmt"
    def __init__(self):
        super(gibshooterbase).__init__()
        super(RenderFields).__init__()
        self.origin = [0, 0, 0]
        self.shootmodel = None  # Type: studio
        self.shootsounds = "CHOICES NOT SUPPORTED"  # Type: choices
        self.simulation = None  # Type: choices
        self.skin = None  # Type: integer
        self.nogibshadows = None  # Type: boolean
        self.gibgravityscale = 1  # Type: float
        self.massoverride = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        gibshooterbase.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.shootmodel = entity_data.get('shootmodel', None)  # Type: studio
        instance.shootsounds = entity_data.get('shootsounds', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.simulation = entity_data.get('simulation', None)  # Type: choices
        instance.skin = parse_source_value(entity_data.get('skin', 0))  # Type: integer
        instance.nogibshadows = entity_data.get('nogibshadows', None)  # Type: boolean
        instance.gibgravityscale = float(entity_data.get('gibgravityscale', 1))  # Type: float
        instance.massoverride = float(entity_data.get('massoverride', 0))  # Type: float


class env_rotorshooter(gibshooterbase, RenderFields):
    icon_sprite = "editor/env_shooter.vmt"
    def __init__(self):
        super(gibshooterbase).__init__()
        super(RenderFields).__init__()
        self.origin = [0, 0, 0]
        self.shootmodel = None  # Type: studio
        self.shootsounds = "CHOICES NOT SUPPORTED"  # Type: choices
        self.simulation = None  # Type: choices
        self.skin = None  # Type: integer
        self.rotortime = 1  # Type: float
        self.rotortimevariance = 0.3  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        gibshooterbase.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.shootmodel = entity_data.get('shootmodel', None)  # Type: studio
        instance.shootsounds = entity_data.get('shootsounds', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.simulation = entity_data.get('simulation', None)  # Type: choices
        instance.skin = parse_source_value(entity_data.get('skin', 0))  # Type: integer
        instance.rotortime = float(entity_data.get('rotortime', 1))  # Type: float
        instance.rotortimevariance = float(entity_data.get('rotortimevariance', 0.3))  # Type: float


class env_soundscape_proxy(Targetname, Parentname):
    icon_sprite = "editor/env_soundscape.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.MainSoundscapeName = None  # Type: target_destination
        self.radius = 128  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.MainSoundscapeName = entity_data.get('mainsoundscapename', None)  # Type: target_destination
        instance.radius = parse_source_value(entity_data.get('radius', 128))  # Type: integer


class env_soundscape(Targetname, Parentname, EnableDisable):
    icon_sprite = "editor/env_soundscape.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.radius = 128  # Type: integer
        self.soundscape = "CHOICES NOT SUPPORTED"  # Type: choices
        self.position0 = None  # Type: target_destination
        self.position1 = None  # Type: target_destination
        self.position2 = None  # Type: target_destination
        self.position3 = None  # Type: target_destination
        self.position4 = None  # Type: target_destination
        self.position5 = None  # Type: target_destination
        self.position6 = None  # Type: target_destination
        self.position7 = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.radius = parse_source_value(entity_data.get('radius', 128))  # Type: integer
        instance.soundscape = entity_data.get('soundscape', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.position0 = entity_data.get('position0', None)  # Type: target_destination
        instance.position1 = entity_data.get('position1', None)  # Type: target_destination
        instance.position2 = entity_data.get('position2', None)  # Type: target_destination
        instance.position3 = entity_data.get('position3', None)  # Type: target_destination
        instance.position4 = entity_data.get('position4', None)  # Type: target_destination
        instance.position5 = entity_data.get('position5', None)  # Type: target_destination
        instance.position6 = entity_data.get('position6', None)  # Type: target_destination
        instance.position7 = entity_data.get('position7', None)  # Type: target_destination


class env_soundscape_triggerable(env_soundscape):
    icon_sprite = "editor/env_soundscape.vmt"
    def __init__(self):
        super(env_soundscape).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        env_soundscape.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_spark(Targetname, Angles, Parentname):
    icon_sprite = "editor/env_spark.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.MaxDelay = "0"  # Type: string
        self.Magnitude = "CHOICES NOT SUPPORTED"  # Type: choices
        self.TrailLength = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.MaxDelay = entity_data.get('maxdelay', "0")  # Type: string
        instance.Magnitude = entity_data.get('magnitude', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.TrailLength = entity_data.get('traillength', "CHOICES NOT SUPPORTED")  # Type: choices


class env_sprite(Targetname, Parentname, SystemLevelChoice, RenderFields):
    def __init__(self):
        super(RenderFields).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(SystemLevelChoice).__init__()
        self.origin = [0, 0, 0]
        self.framerate = "10.0"  # Type: string
        self.model = "sprites/glow01.spr"  # Type: sprite
        self.scale = None  # Type: string
        self.GlowProxySize = 2.0  # Type: float
        self.HDRColorScale = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        SystemLevelChoice.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.framerate = entity_data.get('framerate', "10.0")  # Type: string
        instance.model = entity_data.get('model', "sprites/glow01.spr")  # Type: sprite
        instance.scale = entity_data.get('scale', None)  # Type: string
        instance.GlowProxySize = float(entity_data.get('glowproxysize', 2.0))  # Type: float
        instance.HDRColorScale = float(entity_data.get('hdrcolorscale', 1.0))  # Type: float


class env_sprite_clientside(env_sprite):
    def __init__(self):
        super(env_sprite).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        env_sprite.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_sprite_oriented(env_sprite, Angles):
    def __init__(self):
        super(env_sprite).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        env_sprite.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_wind(Targetname, Angles):
    icon_sprite = "editor/env_wind.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.minwind = 20  # Type: integer
        self.maxwind = 50  # Type: integer
        self.mingust = 100  # Type: integer
        self.maxgust = 250  # Type: integer
        self.mingustdelay = 10  # Type: integer
        self.maxgustdelay = 20  # Type: integer
        self.gustduration = 5  # Type: integer
        self.gustdirchange = 20  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.minwind = parse_source_value(entity_data.get('minwind', 20))  # Type: integer
        instance.maxwind = parse_source_value(entity_data.get('maxwind', 50))  # Type: integer
        instance.mingust = parse_source_value(entity_data.get('mingust', 100))  # Type: integer
        instance.maxgust = parse_source_value(entity_data.get('maxgust', 250))  # Type: integer
        instance.mingustdelay = parse_source_value(entity_data.get('mingustdelay', 10))  # Type: integer
        instance.maxgustdelay = parse_source_value(entity_data.get('maxgustdelay', 20))  # Type: integer
        instance.gustduration = parse_source_value(entity_data.get('gustduration', 5))  # Type: integer
        instance.gustdirchange = parse_source_value(entity_data.get('gustdirchange', 20))  # Type: integer


class sky_camera(Angles):
    def __init__(self):
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.scale = 16  # Type: integer
        self.fogenable = None  # Type: boolean
        self.fogblend = None  # Type: boolean
        self.use_angles = None  # Type: boolean
        self.fogcolor = [255, 255, 255]  # Type: color255
        self.fogcolor2 = [255, 255, 255]  # Type: color255
        self.fogdir = "1 0 0"  # Type: string
        self.fogstart = "500.0"  # Type: string
        self.fogend = "2000.0"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.scale = parse_source_value(entity_data.get('scale', 16))  # Type: integer
        instance.fogenable = entity_data.get('fogenable', None)  # Type: boolean
        instance.fogblend = entity_data.get('fogblend', None)  # Type: boolean
        instance.use_angles = entity_data.get('use_angles', None)  # Type: boolean
        instance.fogcolor = parse_int_vector(entity_data.get('fogcolor', "255 255 255"))  # Type: color255
        instance.fogcolor2 = parse_int_vector(entity_data.get('fogcolor2', "255 255 255"))  # Type: color255
        instance.fogdir = entity_data.get('fogdir', "1 0 0")  # Type: string
        instance.fogstart = entity_data.get('fogstart', "500.0")  # Type: string
        instance.fogend = entity_data.get('fogend', "2000.0")  # Type: string


class BaseSpeaker(Targetname, ResponseContext):
    def __init__(self):
        super(Targetname).__init__()
        super(ResponseContext).__init__()
        self.delaymin = "15"  # Type: string
        self.delaymax = "135"  # Type: string
        self.rulescript = None  # Type: string
        self.concept = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        ResponseContext.from_dict(instance, entity_data)
        instance.delaymin = entity_data.get('delaymin', "15")  # Type: string
        instance.delaymax = entity_data.get('delaymax', "135")  # Type: string
        instance.rulescript = entity_data.get('rulescript', None)  # Type: string
        instance.concept = entity_data.get('concept', None)  # Type: string


class game_weapon_manager(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.weaponname = None  # Type: string
        self.maxpieces = None  # Type: integer
        self.ammomod = 1  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.weaponname = entity_data.get('weaponname', None)  # Type: string
        instance.maxpieces = parse_source_value(entity_data.get('maxpieces', 0))  # Type: integer
        instance.ammomod = float(entity_data.get('ammomod', 1))  # Type: float


class game_end(Targetname):
    icon_sprite = "editor/game_end.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.master = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.master = entity_data.get('master', None)  # Type: string


class game_player_equip(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.master = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.master = entity_data.get('master', None)  # Type: string


class game_player_team(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: string
        self.master = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: string
        instance.master = entity_data.get('master', None)  # Type: string


class game_score(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.points = 1  # Type: integer
        self.master = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.points = parse_source_value(entity_data.get('points', 1))  # Type: integer
        instance.master = entity_data.get('master', None)  # Type: string


class game_text(Targetname):
    icon_sprite = "editor/game_text.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.message = None  # Type: string
        self.x = "-1"  # Type: string
        self.y = "-1"  # Type: string
        self.effect = None  # Type: choices
        self.color = [100, 100, 100]  # Type: color255
        self.color2 = [240, 110, 0]  # Type: color255
        self.fadein = "1.5"  # Type: string
        self.fadeout = "0.5"  # Type: string
        self.holdtime = "1.2"  # Type: string
        self.fxtime = "0.25"  # Type: string
        self.channel = "CHOICES NOT SUPPORTED"  # Type: choices
        self.master = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.message = entity_data.get('message', None)  # Type: string
        instance.x = entity_data.get('x', "-1")  # Type: string
        instance.y = entity_data.get('y', "-1")  # Type: string
        instance.effect = entity_data.get('effect', None)  # Type: choices
        instance.color = parse_int_vector(entity_data.get('color', "100 100 100"))  # Type: color255
        instance.color2 = parse_int_vector(entity_data.get('color2', "240 110 0"))  # Type: color255
        instance.fadein = entity_data.get('fadein', "1.5")  # Type: string
        instance.fadeout = entity_data.get('fadeout', "0.5")  # Type: string
        instance.holdtime = entity_data.get('holdtime', "1.2")  # Type: string
        instance.fxtime = entity_data.get('fxtime', "0.25")  # Type: string
        instance.channel = entity_data.get('channel', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.master = entity_data.get('master', None)  # Type: string


class point_enable_motion_fixup(Parentname, Angles):
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class point_message(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.message = None  # Type: string
        self.radius = 128  # Type: integer
        self.developeronly = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.message = entity_data.get('message', None)  # Type: string
        instance.radius = parse_source_value(entity_data.get('radius', 128))  # Type: integer
        instance.developeronly = entity_data.get('developeronly', None)  # Type: boolean


class point_spotlight(Parentname, SystemLevelChoice, Targetname, RenderFields, Angles):
    model = "models/editor/cone_helper.mdl"
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(SystemLevelChoice).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.spotlightlength = 500  # Type: integer
        self.spotlightwidth = 50  # Type: integer
        self.HDRColorScale = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        SystemLevelChoice.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.spotlightlength = parse_source_value(entity_data.get('spotlightlength', 500))  # Type: integer
        instance.spotlightwidth = parse_source_value(entity_data.get('spotlightwidth', 50))  # Type: integer
        instance.HDRColorScale = float(entity_data.get('hdrcolorscale', 1.0))  # Type: float


class point_tesla(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.m_SourceEntityName = None  # Type: string
        self.m_SoundName = "DoSpark"  # Type: string
        self.texture = "sprites/physbeam.vmt"  # Type: sprite
        self.m_Color = [255, 255, 255]  # Type: color255
        self.m_flRadius = 200  # Type: integer
        self.beamcount_min = 6  # Type: integer
        self.beamcount_max = 8  # Type: integer
        self.thick_min = "4"  # Type: string
        self.thick_max = "5"  # Type: string
        self.lifetime_min = "0.3"  # Type: string
        self.lifetime_max = "0.3"  # Type: string
        self.interval_min = "0.5"  # Type: string
        self.interval_max = "2"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.m_SourceEntityName = entity_data.get('m_sourceentityname', None)  # Type: string
        instance.m_SoundName = entity_data.get('m_soundname', "DoSpark")  # Type: string
        instance.texture = entity_data.get('texture', "sprites/physbeam.vmt")  # Type: sprite
        instance.m_Color = parse_int_vector(entity_data.get('m_color', "255 255 255"))  # Type: color255
        instance.m_flRadius = parse_source_value(entity_data.get('m_flradius', 200))  # Type: integer
        instance.beamcount_min = parse_source_value(entity_data.get('beamcount_min', 6))  # Type: integer
        instance.beamcount_max = parse_source_value(entity_data.get('beamcount_max', 8))  # Type: integer
        instance.thick_min = entity_data.get('thick_min', "4")  # Type: string
        instance.thick_max = entity_data.get('thick_max', "5")  # Type: string
        instance.lifetime_min = entity_data.get('lifetime_min', "0.3")  # Type: string
        instance.lifetime_max = entity_data.get('lifetime_max', "0.3")  # Type: string
        instance.interval_min = entity_data.get('interval_min', "0.5")  # Type: string
        instance.interval_max = entity_data.get('interval_max', "2")  # Type: string


class point_clientcommand(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class point_servercommand(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class point_bonusmaps_accessor(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.filename = None  # Type: string
        self.mapname = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.filename = entity_data.get('filename', None)  # Type: string
        instance.mapname = entity_data.get('mapname', None)  # Type: string


class game_ui(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.FieldOfView = -1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.FieldOfView = float(entity_data.get('fieldofview', -1.0))  # Type: float


class point_entity_finder(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.filtername = None  # Type: filterclass
        self.referencename = None  # Type: target_destination
        self.Method = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.filtername = entity_data.get('filtername', None)  # Type: filterclass
        instance.referencename = entity_data.get('referencename', None)  # Type: target_destination
        instance.Method = entity_data.get('method', "CHOICES NOT SUPPORTED")  # Type: choices


class game_zone_player(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)


class infodecal(Targetname):
    model = "models/editor/axis_helper_thick.mdl"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.texture = None  # Type: decal
        self.LowPriority = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.texture = entity_data.get('texture', None)  # Type: decal
        instance.LowPriority = entity_data.get('lowpriority', None)  # Type: boolean


class info_projecteddecal(Angles, Targetname):
    model = "models/editor/axis_helper_thick.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.texture = None  # Type: decal
        self.Distance = 64  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.texture = entity_data.get('texture', None)  # Type: decal
        instance.Distance = float(entity_data.get('distance', 64))  # Type: float


class info_no_dynamic_shadow(Base):
    def __init__(self):
        super().__init__()
        self.origin = [0, 0, 0]
        self.sides = None  # Type: sidelist

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.sides = entity_data.get('sides', None)  # Type: sidelist


class info_player_start(PlayerClass, Angles):
    model = "models/editor/playerstart.mdl"
    def __init__(self):
        super(PlayerClass).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        PlayerClass.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_overlay(Targetname):
    model = "models/editor/overlay_helper.mdl"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.material = None  # Type: material
        self.sides = None  # Type: sidelist
        self.RenderOrder = None  # Type: integer
        self.StartU = 0.0  # Type: float
        self.EndU = 1.0  # Type: float
        self.StartV = 0.0  # Type: float
        self.EndV = 1.0  # Type: float
        self.BasisOrigin = None  # Type: vector
        self.BasisU = None  # Type: vector
        self.BasisV = None  # Type: vector
        self.BasisNormal = None  # Type: vector
        self.uv0 = None  # Type: vector
        self.uv1 = None  # Type: vector
        self.uv2 = None  # Type: vector
        self.uv3 = None  # Type: vector
        self.fademindist = -1  # Type: float
        self.fademaxdist = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.material = entity_data.get('material', None)  # Type: material
        instance.sides = entity_data.get('sides', None)  # Type: sidelist
        instance.RenderOrder = parse_source_value(entity_data.get('renderorder', 0))  # Type: integer
        instance.StartU = float(entity_data.get('startu', 0.0))  # Type: float
        instance.EndU = float(entity_data.get('endu', 1.0))  # Type: float
        instance.StartV = float(entity_data.get('startv', 0.0))  # Type: float
        instance.EndV = float(entity_data.get('endv', 1.0))  # Type: float
        instance.BasisOrigin = parse_float_vector(entity_data.get('basisorigin', "0 0 0"))  # Type: vector
        instance.BasisU = parse_float_vector(entity_data.get('basisu', "0 0 0"))  # Type: vector
        instance.BasisV = parse_float_vector(entity_data.get('basisv', "0 0 0"))  # Type: vector
        instance.BasisNormal = parse_float_vector(entity_data.get('basisnormal', "0 0 0"))  # Type: vector
        instance.uv0 = parse_float_vector(entity_data.get('uv0', "0 0 0"))  # Type: vector
        instance.uv1 = parse_float_vector(entity_data.get('uv1', "0 0 0"))  # Type: vector
        instance.uv2 = parse_float_vector(entity_data.get('uv2', "0 0 0"))  # Type: vector
        instance.uv3 = parse_float_vector(entity_data.get('uv3', "0 0 0"))  # Type: vector
        instance.fademindist = float(entity_data.get('fademindist', -1))  # Type: float
        instance.fademaxdist = float(entity_data.get('fademaxdist', 0))  # Type: float


class info_overlay_transition(Base):
    def __init__(self):
        super().__init__()
        self.origin = [0, 0, 0]
        self.material = None  # Type: material
        self.sides = None  # Type: sidelist
        self.sides2 = None  # Type: sidelist
        self.LengthTexcoordStart = 0.0  # Type: float
        self.LengthTexcoordEnd = 1.0  # Type: float
        self.WidthTexcoordStart = 0.0  # Type: float
        self.WidthTexcoordEnd = 1.0  # Type: float
        self.Width1 = 25.0  # Type: float
        self.Width2 = 25.0  # Type: float
        self.DebugDraw = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.material = entity_data.get('material', None)  # Type: material
        instance.sides = entity_data.get('sides', None)  # Type: sidelist
        instance.sides2 = entity_data.get('sides2', None)  # Type: sidelist
        instance.LengthTexcoordStart = float(entity_data.get('lengthtexcoordstart', 0.0))  # Type: float
        instance.LengthTexcoordEnd = float(entity_data.get('lengthtexcoordend', 1.0))  # Type: float
        instance.WidthTexcoordStart = float(entity_data.get('widthtexcoordstart', 0.0))  # Type: float
        instance.WidthTexcoordEnd = float(entity_data.get('widthtexcoordend', 1.0))  # Type: float
        instance.Width1 = float(entity_data.get('width1', 25.0))  # Type: float
        instance.Width2 = float(entity_data.get('width2', 25.0))  # Type: float
        instance.DebugDraw = parse_source_value(entity_data.get('debugdraw', 0))  # Type: integer


class info_intermission(Base):
    def __init__(self):
        super().__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination


class info_landmark(Targetname):
    icon_sprite = "editor/info_landmark"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_null(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_target(Targetname, Angles, Parentname):
    icon_sprite = "editor/info_target.vmt"
    model = "models/editor/axis_helper.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_particle_system(Targetname, Angles, Parentname, Reflection):
    model = "models/editor/cone_helper.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        super(Reflection).__init__()
        self.origin = [0, 0, 0]
        self.effect_name = None  # Type: particlesystem
        self.start_active = None  # Type: boolean
        self.cpoint1 = None  # Type: target_destination
        self.cpoint2 = None  # Type: target_destination
        self.cpoint3 = None  # Type: target_destination
        self.cpoint4 = None  # Type: target_destination
        self.cpoint5 = None  # Type: target_destination
        self.cpoint6 = None  # Type: target_destination
        self.cpoint7 = None  # Type: target_destination
        self.cpoint8 = None  # Type: target_destination
        self.cpoint9 = None  # Type: target_destination
        self.cpoint10 = None  # Type: target_destination
        self.cpoint11 = None  # Type: target_destination
        self.cpoint12 = None  # Type: target_destination
        self.cpoint13 = None  # Type: target_destination
        self.cpoint14 = None  # Type: target_destination
        self.cpoint15 = None  # Type: target_destination
        self.cpoint16 = None  # Type: target_destination
        self.cpoint17 = None  # Type: target_destination
        self.cpoint18 = None  # Type: target_destination
        self.cpoint19 = None  # Type: target_destination
        self.cpoint20 = None  # Type: target_destination
        self.cpoint21 = None  # Type: target_destination
        self.cpoint22 = None  # Type: target_destination
        self.cpoint23 = None  # Type: target_destination
        self.cpoint24 = None  # Type: target_destination
        self.cpoint25 = None  # Type: target_destination
        self.cpoint26 = None  # Type: target_destination
        self.cpoint27 = None  # Type: target_destination
        self.cpoint28 = None  # Type: target_destination
        self.cpoint29 = None  # Type: target_destination
        self.cpoint30 = None  # Type: target_destination
        self.cpoint31 = None  # Type: target_destination
        self.cpoint32 = None  # Type: target_destination
        self.cpoint33 = None  # Type: target_destination
        self.cpoint34 = None  # Type: target_destination
        self.cpoint35 = None  # Type: target_destination
        self.cpoint36 = None  # Type: target_destination
        self.cpoint37 = None  # Type: target_destination
        self.cpoint38 = None  # Type: target_destination
        self.cpoint39 = None  # Type: target_destination
        self.cpoint40 = None  # Type: target_destination
        self.cpoint41 = None  # Type: target_destination
        self.cpoint42 = None  # Type: target_destination
        self.cpoint43 = None  # Type: target_destination
        self.cpoint44 = None  # Type: target_destination
        self.cpoint45 = None  # Type: target_destination
        self.cpoint46 = None  # Type: target_destination
        self.cpoint47 = None  # Type: target_destination
        self.cpoint48 = None  # Type: target_destination
        self.cpoint49 = None  # Type: target_destination
        self.cpoint50 = None  # Type: target_destination
        self.cpoint51 = None  # Type: target_destination
        self.cpoint52 = None  # Type: target_destination
        self.cpoint53 = None  # Type: target_destination
        self.cpoint54 = None  # Type: target_destination
        self.cpoint55 = None  # Type: target_destination
        self.cpoint56 = None  # Type: target_destination
        self.cpoint57 = None  # Type: target_destination
        self.cpoint58 = None  # Type: target_destination
        self.cpoint59 = None  # Type: target_destination
        self.cpoint60 = None  # Type: target_destination
        self.cpoint61 = None  # Type: target_destination
        self.cpoint62 = None  # Type: target_destination
        self.cpoint63 = None  # Type: target_destination
        self.cpoint1_parent = None  # Type: integer
        self.cpoint2_parent = None  # Type: integer
        self.cpoint3_parent = None  # Type: integer
        self.cpoint4_parent = None  # Type: integer
        self.cpoint5_parent = None  # Type: integer
        self.cpoint6_parent = None  # Type: integer
        self.cpoint7_parent = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Reflection.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.effect_name = entity_data.get('effect_name', None)  # Type: particlesystem
        instance.start_active = entity_data.get('start_active', None)  # Type: boolean
        instance.cpoint1 = entity_data.get('cpoint1', None)  # Type: target_destination
        instance.cpoint2 = entity_data.get('cpoint2', None)  # Type: target_destination
        instance.cpoint3 = entity_data.get('cpoint3', None)  # Type: target_destination
        instance.cpoint4 = entity_data.get('cpoint4', None)  # Type: target_destination
        instance.cpoint5 = entity_data.get('cpoint5', None)  # Type: target_destination
        instance.cpoint6 = entity_data.get('cpoint6', None)  # Type: target_destination
        instance.cpoint7 = entity_data.get('cpoint7', None)  # Type: target_destination
        instance.cpoint8 = entity_data.get('cpoint8', None)  # Type: target_destination
        instance.cpoint9 = entity_data.get('cpoint9', None)  # Type: target_destination
        instance.cpoint10 = entity_data.get('cpoint10', None)  # Type: target_destination
        instance.cpoint11 = entity_data.get('cpoint11', None)  # Type: target_destination
        instance.cpoint12 = entity_data.get('cpoint12', None)  # Type: target_destination
        instance.cpoint13 = entity_data.get('cpoint13', None)  # Type: target_destination
        instance.cpoint14 = entity_data.get('cpoint14', None)  # Type: target_destination
        instance.cpoint15 = entity_data.get('cpoint15', None)  # Type: target_destination
        instance.cpoint16 = entity_data.get('cpoint16', None)  # Type: target_destination
        instance.cpoint17 = entity_data.get('cpoint17', None)  # Type: target_destination
        instance.cpoint18 = entity_data.get('cpoint18', None)  # Type: target_destination
        instance.cpoint19 = entity_data.get('cpoint19', None)  # Type: target_destination
        instance.cpoint20 = entity_data.get('cpoint20', None)  # Type: target_destination
        instance.cpoint21 = entity_data.get('cpoint21', None)  # Type: target_destination
        instance.cpoint22 = entity_data.get('cpoint22', None)  # Type: target_destination
        instance.cpoint23 = entity_data.get('cpoint23', None)  # Type: target_destination
        instance.cpoint24 = entity_data.get('cpoint24', None)  # Type: target_destination
        instance.cpoint25 = entity_data.get('cpoint25', None)  # Type: target_destination
        instance.cpoint26 = entity_data.get('cpoint26', None)  # Type: target_destination
        instance.cpoint27 = entity_data.get('cpoint27', None)  # Type: target_destination
        instance.cpoint28 = entity_data.get('cpoint28', None)  # Type: target_destination
        instance.cpoint29 = entity_data.get('cpoint29', None)  # Type: target_destination
        instance.cpoint30 = entity_data.get('cpoint30', None)  # Type: target_destination
        instance.cpoint31 = entity_data.get('cpoint31', None)  # Type: target_destination
        instance.cpoint32 = entity_data.get('cpoint32', None)  # Type: target_destination
        instance.cpoint33 = entity_data.get('cpoint33', None)  # Type: target_destination
        instance.cpoint34 = entity_data.get('cpoint34', None)  # Type: target_destination
        instance.cpoint35 = entity_data.get('cpoint35', None)  # Type: target_destination
        instance.cpoint36 = entity_data.get('cpoint36', None)  # Type: target_destination
        instance.cpoint37 = entity_data.get('cpoint37', None)  # Type: target_destination
        instance.cpoint38 = entity_data.get('cpoint38', None)  # Type: target_destination
        instance.cpoint39 = entity_data.get('cpoint39', None)  # Type: target_destination
        instance.cpoint40 = entity_data.get('cpoint40', None)  # Type: target_destination
        instance.cpoint41 = entity_data.get('cpoint41', None)  # Type: target_destination
        instance.cpoint42 = entity_data.get('cpoint42', None)  # Type: target_destination
        instance.cpoint43 = entity_data.get('cpoint43', None)  # Type: target_destination
        instance.cpoint44 = entity_data.get('cpoint44', None)  # Type: target_destination
        instance.cpoint45 = entity_data.get('cpoint45', None)  # Type: target_destination
        instance.cpoint46 = entity_data.get('cpoint46', None)  # Type: target_destination
        instance.cpoint47 = entity_data.get('cpoint47', None)  # Type: target_destination
        instance.cpoint48 = entity_data.get('cpoint48', None)  # Type: target_destination
        instance.cpoint49 = entity_data.get('cpoint49', None)  # Type: target_destination
        instance.cpoint50 = entity_data.get('cpoint50', None)  # Type: target_destination
        instance.cpoint51 = entity_data.get('cpoint51', None)  # Type: target_destination
        instance.cpoint52 = entity_data.get('cpoint52', None)  # Type: target_destination
        instance.cpoint53 = entity_data.get('cpoint53', None)  # Type: target_destination
        instance.cpoint54 = entity_data.get('cpoint54', None)  # Type: target_destination
        instance.cpoint55 = entity_data.get('cpoint55', None)  # Type: target_destination
        instance.cpoint56 = entity_data.get('cpoint56', None)  # Type: target_destination
        instance.cpoint57 = entity_data.get('cpoint57', None)  # Type: target_destination
        instance.cpoint58 = entity_data.get('cpoint58', None)  # Type: target_destination
        instance.cpoint59 = entity_data.get('cpoint59', None)  # Type: target_destination
        instance.cpoint60 = entity_data.get('cpoint60', None)  # Type: target_destination
        instance.cpoint61 = entity_data.get('cpoint61', None)  # Type: target_destination
        instance.cpoint62 = entity_data.get('cpoint62', None)  # Type: target_destination
        instance.cpoint63 = entity_data.get('cpoint63', None)  # Type: target_destination
        instance.cpoint1_parent = parse_source_value(entity_data.get('cpoint1_parent', 0))  # Type: integer
        instance.cpoint2_parent = parse_source_value(entity_data.get('cpoint2_parent', 0))  # Type: integer
        instance.cpoint3_parent = parse_source_value(entity_data.get('cpoint3_parent', 0))  # Type: integer
        instance.cpoint4_parent = parse_source_value(entity_data.get('cpoint4_parent', 0))  # Type: integer
        instance.cpoint5_parent = parse_source_value(entity_data.get('cpoint5_parent', 0))  # Type: integer
        instance.cpoint6_parent = parse_source_value(entity_data.get('cpoint6_parent', 0))  # Type: integer
        instance.cpoint7_parent = parse_source_value(entity_data.get('cpoint7_parent', 0))  # Type: integer


class phys_ragdollmagnet(Targetname, Angles, Parentname, EnableDisable):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.axis = None  # Type: vecline
        self.radius = 512  # Type: float
        self.force = 5000  # Type: float
        self.target = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.axis = entity_data.get('axis', None)  # Type: vecline
        instance.radius = float(entity_data.get('radius', 512))  # Type: float
        instance.force = float(entity_data.get('force', 5000))  # Type: float
        instance.target = entity_data.get('target', None)  # Type: string


class info_lighting(Targetname):
    icon_sprite = "editor/info_lighting.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_teleport_destination(Targetname, PlayerClass, Angles, Parentname):
    model = "models/editor/playerstart.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(PlayerClass).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        PlayerClass.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_node(Node):
    model = "models/editor/ground_node.mdl"
    def __init__(self):
        super(Node).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Node.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_node_hint(Targetname, Angles, HintNode):
    model = "models/editor/ground_node_hint.mdl"
    def __init__(self):
        super(HintNode).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        HintNode.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_node_air(Node):
    model = "models/editor/air_node.mdl"
    def __init__(self):
        super(Node).__init__()
        self.origin = [0, 0, 0]
        self.nodeheight = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Node.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.nodeheight = parse_source_value(entity_data.get('nodeheight', 0))  # Type: integer


class info_node_air_hint(Angles, Targetname, HintNode):
    model = "models/editor/air_node_hint.mdl"
    def __init__(self):
        super(HintNode).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.nodeheight = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        HintNode.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.nodeheight = parse_source_value(entity_data.get('nodeheight', 0))  # Type: integer


class info_hint(Targetname, Angles, HintNode):
    model = "models/editor/node_hint.mdl"
    def __init__(self):
        super(HintNode).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        HintNode.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_node_link(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.StartNode = None  # Type: node_dest
        self.EndNode = None  # Type: node_dest
        self.initialstate = "CHOICES NOT SUPPORTED"  # Type: choices
        self.linktype = "CHOICES NOT SUPPORTED"  # Type: choices
        self.AllowUse = None  # Type: string
        self.InvertAllow = None  # Type: boolean
        self.preciseMovement = None  # Type: boolean
        self.priority = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.StartNode = parse_source_value(entity_data.get('startnode', 0))  # Type: node_dest
        instance.EndNode = parse_source_value(entity_data.get('endnode', 0))  # Type: node_dest
        instance.initialstate = entity_data.get('initialstate', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.linktype = entity_data.get('linktype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.AllowUse = entity_data.get('allowuse', None)  # Type: string
        instance.InvertAllow = entity_data.get('invertallow', None)  # Type: boolean
        instance.preciseMovement = entity_data.get('precisemovement', None)  # Type: boolean
        instance.priority = entity_data.get('priority', None)  # Type: choices


class info_node_link_controller(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.mins = [-8.0, -32.0, -36.0]  # Type: vector
        self.maxs = [8.0, 32.0, 36.0]  # Type: vector
        self.initialstate = "CHOICES NOT SUPPORTED"  # Type: choices
        self.useairlinkradius = None  # Type: boolean
        self.AllowUse = None  # Type: string
        self.InvertAllow = None  # Type: boolean
        self.priority = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.mins = parse_float_vector(entity_data.get('mins', "-8 -32 -36"))  # Type: vector
        instance.maxs = parse_float_vector(entity_data.get('maxs', "8 32 36"))  # Type: vector
        instance.initialstate = entity_data.get('initialstate', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.useairlinkradius = entity_data.get('useairlinkradius', None)  # Type: boolean
        instance.AllowUse = entity_data.get('allowuse', None)  # Type: string
        instance.InvertAllow = entity_data.get('invertallow', None)  # Type: boolean
        instance.priority = entity_data.get('priority', None)  # Type: choices


class info_radial_link_controller(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.radius = 120  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.radius = float(entity_data.get('radius', 120))  # Type: float


class info_node_climb(Targetname, Angles, HintNode):
    model = "models/editor/climb_node.mdl"
    def __init__(self):
        super(HintNode).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        HintNode.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class light(Targetname, Light):
    icon_sprite = "editor/light.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(Light).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self._distance = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Light.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance._distance = parse_source_value(entity_data.get('_distance', 0))  # Type: integer


class light_environment(Angles):
    icon_sprite = "editor/light_env.vmt"
    def __init__(self):
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.pitch = None  # Type: integer
        self._light = [255, 255, 255, 200]  # Type: color255
        self._ambient = [255, 255, 255, 20]  # Type: color255
        self._lightHDR = [-1, -1, -1, 1]  # Type: color255
        self._lightscaleHDR = 1  # Type: float
        self._ambientHDR = [-1, -1, -1, 1]  # Type: color255
        self._AmbientScaleHDR = 1  # Type: float
        self.SunSpreadAngle = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.pitch = parse_source_value(entity_data.get('pitch', 0))  # Type: integer
        instance._light = parse_int_vector(entity_data.get('_light', "255 255 255 200"))  # Type: color255
        instance._ambient = parse_int_vector(entity_data.get('_ambient', "255 255 255 20"))  # Type: color255
        instance._lightHDR = parse_int_vector(entity_data.get('_lighthdr', "-1 -1 -1 1"))  # Type: color255
        instance._lightscaleHDR = float(entity_data.get('_lightscalehdr', 1))  # Type: float
        instance._ambientHDR = parse_int_vector(entity_data.get('_ambienthdr', "-1 -1 -1 1"))  # Type: color255
        instance._AmbientScaleHDR = float(entity_data.get('_ambientscalehdr', 1))  # Type: float
        instance.SunSpreadAngle = float(entity_data.get('sunspreadangle', 0))  # Type: float


class light_spot(Targetname, Angles, Light):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Light).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self._inner_cone = 30  # Type: integer
        self._cone = 45  # Type: integer
        self._exponent = 1  # Type: integer
        self._distance = None  # Type: integer
        self.pitch = -90  # Type: angle_negative_pitch

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Light.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance._inner_cone = parse_source_value(entity_data.get('_inner_cone', 30))  # Type: integer
        instance._cone = parse_source_value(entity_data.get('_cone', 45))  # Type: integer
        instance._exponent = parse_source_value(entity_data.get('_exponent', 1))  # Type: integer
        instance._distance = parse_source_value(entity_data.get('_distance', 0))  # Type: integer
        instance.pitch = float(entity_data.get('pitch', -90))  # Type: angle_negative_pitch


class light_dynamic(Targetname, Angles, Parentname):
    icon_sprite = "editor/light.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self._light = [255, 255, 255, 200]  # Type: color255
        self.brightness = None  # Type: integer
        self._inner_cone = 30  # Type: integer
        self._cone = 45  # Type: integer
        self.pitch = -90  # Type: integer
        self.distance = 120  # Type: float
        self.spotlight_radius = 80  # Type: float
        self.style = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance._light = parse_int_vector(entity_data.get('_light', "255 255 255 200"))  # Type: color255
        instance.brightness = parse_source_value(entity_data.get('brightness', 0))  # Type: integer
        instance._inner_cone = parse_source_value(entity_data.get('_inner_cone', 30))  # Type: integer
        instance._cone = parse_source_value(entity_data.get('_cone', 45))  # Type: integer
        instance.pitch = parse_source_value(entity_data.get('pitch', -90))  # Type: integer
        instance.distance = float(entity_data.get('distance', 120))  # Type: float
        instance.spotlight_radius = float(entity_data.get('spotlight_radius', 80))  # Type: float
        instance.style = entity_data.get('style', None)  # Type: choices


class shadow_control(Targetname):
    icon_sprite = "editor/shadow_control.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.angles = "80 30 0"  # Type: string
        self.color = [128, 128, 128]  # Type: color255
        self.distance = 75  # Type: float
        self.disableallshadows = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.angles = entity_data.get('angles', "80 30 0")  # Type: string
        instance.color = parse_int_vector(entity_data.get('color', "128 128 128"))  # Type: color255
        instance.distance = float(entity_data.get('distance', 75))  # Type: float
        instance.disableallshadows = entity_data.get('disableallshadows', None)  # Type: boolean


class sunlight_shadow_control(Targetname, EnableDisable):
    icon_sprite = "editor/shadow_control.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.angles = "50 40 0"  # Type: string
        self.color = [255, 255, 255, 1]  # Type: color255
        self.colortransitiontime = 0.5  # Type: float
        self.distance = 10000  # Type: float
        self.fov = 5  # Type: float
        self.nearz = 512  # Type: float
        self.northoffset = 200  # Type: float
        self.texturename = "effects/flashlight001"  # Type: material
        self.enableshadows = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.angles = entity_data.get('angles', "50 40 0")  # Type: string
        instance.color = parse_int_vector(entity_data.get('color', "255 255 255 1"))  # Type: color255
        instance.colortransitiontime = float(entity_data.get('colortransitiontime', 0.5))  # Type: float
        instance.distance = float(entity_data.get('distance', 10000))  # Type: float
        instance.fov = float(entity_data.get('fov', 5))  # Type: float
        instance.nearz = float(entity_data.get('nearz', 512))  # Type: float
        instance.northoffset = float(entity_data.get('northoffset', 200))  # Type: float
        instance.texturename = entity_data.get('texturename', "effects/flashlight001")  # Type: material
        instance.enableshadows = entity_data.get('enableshadows', None)  # Type: boolean


class env_ambient_light(Targetname, EnableDisable, SpatialEntity):
    icon_sprite = "editor/color_correction.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(SpatialEntity).__init__()
        self.origin = [0, 0, 0]
        self.color = [255, 255, 255]  # Type: color255

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        SpatialEntity.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.color = parse_int_vector(entity_data.get('color', "255 255 255"))  # Type: color255


class color_correction(Targetname, EnableDisable):
    icon_sprite = "editor/color_correction.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.minfalloff = 0.0  # Type: float
        self.maxfalloff = 200.0  # Type: float
        self.maxweight = 1.0  # Type: float
        self.filename = None  # Type: string
        self.fadeInDuration = 0.0  # Type: float
        self.fadeOutDuration = 0.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.minfalloff = float(entity_data.get('minfalloff', 0.0))  # Type: float
        instance.maxfalloff = float(entity_data.get('maxfalloff', 200.0))  # Type: float
        instance.maxweight = float(entity_data.get('maxweight', 1.0))  # Type: float
        instance.filename = entity_data.get('filename', None)  # Type: string
        instance.fadeInDuration = float(entity_data.get('fadeinduration', 0.0))  # Type: float
        instance.fadeOutDuration = float(entity_data.get('fadeoutduration', 0.0))  # Type: float


class color_correction_volume(Targetname, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.fadeDuration = 10.0  # Type: float
        self.maxweight = 1.0  # Type: float
        self.filename = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.fadeDuration = float(entity_data.get('fadeduration', 10.0))  # Type: float
        instance.maxweight = float(entity_data.get('maxweight', 1.0))  # Type: float
        instance.filename = entity_data.get('filename', None)  # Type: string


class KeyFrame(Base):
    def __init__(self):
        super().__init__()
        self.NextKey = None  # Type: target_destination
        self.MoveSpeed = 64  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.NextKey = entity_data.get('nextkey', None)  # Type: target_destination
        instance.MoveSpeed = parse_source_value(entity_data.get('movespeed', 64))  # Type: integer


class Mover(Base):
    def __init__(self):
        super().__init__()
        self.PositionInterpolator = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.PositionInterpolator = entity_data.get('positioninterpolator', None)  # Type: choices


class func_movelinear(Targetname, Parentname, Origin, RenderFields):
    def __init__(self):
        super(RenderFields).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(Origin).__init__()
        self.movedir = [0.0, 0.0, 0.0]  # Type: angle
        self.startposition = None  # Type: float
        self.speed = 100  # Type: integer
        self.movedistance = 100  # Type: float
        self.blockdamage = None  # Type: float
        self.startsound = None  # Type: sound
        self.stopsound = None  # Type: sound

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance.movedir = parse_float_vector(entity_data.get('movedir', "0 0 0"))  # Type: angle
        instance.startposition = float(entity_data.get('startposition', 0))  # Type: float
        instance.speed = parse_source_value(entity_data.get('speed', 100))  # Type: integer
        instance.movedistance = float(entity_data.get('movedistance', 100))  # Type: float
        instance.blockdamage = float(entity_data.get('blockdamage', 0))  # Type: float
        instance.startsound = entity_data.get('startsound', None)  # Type: sound
        instance.stopsound = entity_data.get('stopsound', None)  # Type: sound


class func_water_analog(Targetname, Parentname, Origin):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(Origin).__init__()
        self.movedir = [0.0, 0.0, 0.0]  # Type: angle
        self.startposition = None  # Type: float
        self.speed = 100  # Type: integer
        self.movedistance = 100  # Type: float
        self.startsound = None  # Type: sound
        self.stopsound = None  # Type: sound
        self.WaveHeight = "3.0"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.movedir = parse_float_vector(entity_data.get('movedir', "0 0 0"))  # Type: angle
        instance.startposition = float(entity_data.get('startposition', 0))  # Type: float
        instance.speed = parse_source_value(entity_data.get('speed', 100))  # Type: integer
        instance.movedistance = float(entity_data.get('movedistance', 100))  # Type: float
        instance.startsound = entity_data.get('startsound', None)  # Type: sound
        instance.stopsound = entity_data.get('stopsound', None)  # Type: sound
        instance.WaveHeight = entity_data.get('waveheight', "3.0")  # Type: string


class func_rotating(Parentname, Shadow, Targetname, RenderFields, Reflection, Angles, Origin):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        super(Reflection).__init__()
        super(Angles).__init__()
        super(Origin).__init__()
        self.maxspeed = 100  # Type: integer
        self.fanfriction = 20  # Type: integer
        self.message = None  # Type: sound
        self.volume = 10  # Type: integer
        self._minlight = None  # Type: string
        self.dmg = None  # Type: integer
        self.solidbsp = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Reflection.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.maxspeed = parse_source_value(entity_data.get('maxspeed', 100))  # Type: integer
        instance.fanfriction = parse_source_value(entity_data.get('fanfriction', 20))  # Type: integer
        instance.message = entity_data.get('message', None)  # Type: sound
        instance.volume = parse_source_value(entity_data.get('volume', 10))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.dmg = parse_source_value(entity_data.get('dmg', 0))  # Type: integer
        instance.solidbsp = entity_data.get('solidbsp', None)  # Type: choices


class func_platrot(Parentname, Shadow, Targetname, RenderFields, BasePlat, Reflection, Angles, Origin):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        super(BasePlat).__init__()
        super(Reflection).__init__()
        super(Angles).__init__()
        super(Origin).__init__()
        self.noise1 = None  # Type: sound
        self.noise2 = None  # Type: sound
        self.speed = 50  # Type: integer
        self.height = None  # Type: integer
        self.rotation = None  # Type: integer
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        BasePlat.from_dict(instance, entity_data)
        Reflection.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.noise1 = entity_data.get('noise1', None)  # Type: sound
        instance.noise2 = entity_data.get('noise2', None)  # Type: sound
        instance.speed = parse_source_value(entity_data.get('speed', 50))  # Type: integer
        instance.height = parse_source_value(entity_data.get('height', 0))  # Type: integer
        instance.rotation = parse_source_value(entity_data.get('rotation', 0))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class keyframe_track(Targetname, Angles, Parentname, KeyFrame):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        super(KeyFrame).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        KeyFrame.from_dict(instance, entity_data)


class move_keyframed(Targetname, Mover, Parentname, KeyFrame):
    def __init__(self):
        super(Targetname).__init__()
        super(Mover).__init__()
        super(Parentname).__init__()
        super(KeyFrame).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Mover.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        KeyFrame.from_dict(instance, entity_data)


class move_track(Targetname, KeyFrame, Parentname, Mover):
    def __init__(self):
        super(Targetname).__init__()
        super(KeyFrame).__init__()
        super(Parentname).__init__()
        super(Mover).__init__()
        self.WheelBaseLength = 50  # Type: integer
        self.Damage = None  # Type: integer
        self.NoRotate = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        KeyFrame.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Mover.from_dict(instance, entity_data)
        instance.WheelBaseLength = parse_source_value(entity_data.get('wheelbaselength', 50))  # Type: integer
        instance.Damage = parse_source_value(entity_data.get('damage', 0))  # Type: integer
        instance.NoRotate = entity_data.get('norotate', None)  # Type: boolean


class RopeKeyFrame(SystemLevelChoice):
    def __init__(self):
        super(SystemLevelChoice).__init__()
        self.Slack = 25  # Type: integer
        self.Type = None  # Type: choices
        self.Subdiv = 2  # Type: integer
        self.Barbed = None  # Type: boolean
        self.Width = "2"  # Type: string
        self.TextureScale = "1"  # Type: string
        self.Collide = None  # Type: boolean
        self.Dangling = None  # Type: boolean
        self.Breakable = None  # Type: boolean
        self.RopeMaterial = "cable/cable.vmt"  # Type: material

    @staticmethod
    def from_dict(instance, entity_data: dict):
        SystemLevelChoice.from_dict(instance, entity_data)
        instance.Slack = parse_source_value(entity_data.get('slack', 25))  # Type: integer
        instance.Type = entity_data.get('type', None)  # Type: choices
        instance.Subdiv = parse_source_value(entity_data.get('subdiv', 2))  # Type: integer
        instance.Barbed = entity_data.get('barbed', None)  # Type: boolean
        instance.Width = entity_data.get('width', "2")  # Type: string
        instance.TextureScale = entity_data.get('texturescale', "1")  # Type: string
        instance.Collide = entity_data.get('collide', None)  # Type: boolean
        instance.Dangling = entity_data.get('dangling', None)  # Type: boolean
        instance.Breakable = entity_data.get('breakable', None)  # Type: boolean
        instance.RopeMaterial = entity_data.get('ropematerial', "cable/cable.vmt")  # Type: material


class keyframe_rope(Targetname, RopeKeyFrame, Parentname, KeyFrame):
    model = "models/editor/axis_helper_thick.mdl"
    def __init__(self):
        super(RopeKeyFrame).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(KeyFrame).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        RopeKeyFrame.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        KeyFrame.from_dict(instance, entity_data)


class move_rope(Targetname, RopeKeyFrame, Parentname, KeyFrame):
    model = "models/editor/axis_helper.mdl"
    def __init__(self):
        super(RopeKeyFrame).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(KeyFrame).__init__()
        self.PositionInterpolator = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        RopeKeyFrame.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        KeyFrame.from_dict(instance, entity_data)
        instance.PositionInterpolator = entity_data.get('positioninterpolator', "CHOICES NOT SUPPORTED")  # Type: choices


class Button(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class func_button(Parentname, Targetname, RenderFields, Button, DamageFilter, Origin):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Button).__init__()
        super(DamageFilter).__init__()
        super(Origin).__init__()
        self.movedir = [0.0, 0.0, 0.0]  # Type: angle
        self.speed = 5  # Type: integer
        self.health = None  # Type: integer
        self.lip = None  # Type: integer
        self.master = None  # Type: string
        self.sounds = None  # Type: choices
        self.wait = 3  # Type: integer
        self.locked_sound = None  # Type: choices
        self.unlocked_sound = None  # Type: choices
        self.locked_sentence = None  # Type: choices
        self.unlocked_sentence = None  # Type: choices
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Button.from_dict(instance, entity_data)
        DamageFilter.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.movedir = parse_float_vector(entity_data.get('movedir', "0 0 0"))  # Type: angle
        instance.speed = parse_source_value(entity_data.get('speed', 5))  # Type: integer
        instance.health = parse_source_value(entity_data.get('health', 0))  # Type: integer
        instance.lip = parse_source_value(entity_data.get('lip', 0))  # Type: integer
        instance.master = entity_data.get('master', None)  # Type: string
        instance.sounds = entity_data.get('sounds', None)  # Type: choices
        instance.wait = parse_source_value(entity_data.get('wait', 3))  # Type: integer
        instance.locked_sound = entity_data.get('locked_sound', None)  # Type: choices
        instance.unlocked_sound = entity_data.get('unlocked_sound', None)  # Type: choices
        instance.locked_sentence = entity_data.get('locked_sentence', None)  # Type: choices
        instance.unlocked_sentence = entity_data.get('unlocked_sentence', None)  # Type: choices
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class func_rot_button(Parentname, EnableDisable, Targetname, Button, Global, Angles, Origin):
    def __init__(self):
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        super(Button).__init__()
        super(Global).__init__()
        super(Angles).__init__()
        super(Origin).__init__()
        self.master = None  # Type: string
        self.speed = 50  # Type: integer
        self.health = None  # Type: integer
        self.sounds = "CHOICES NOT SUPPORTED"  # Type: choices
        self.wait = 3  # Type: integer
        self.distance = 90  # Type: integer
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Button.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.master = entity_data.get('master', None)  # Type: string
        instance.speed = parse_source_value(entity_data.get('speed', 50))  # Type: integer
        instance.health = parse_source_value(entity_data.get('health', 0))  # Type: integer
        instance.sounds = entity_data.get('sounds', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.wait = parse_source_value(entity_data.get('wait', 3))  # Type: integer
        instance.distance = parse_source_value(entity_data.get('distance', 90))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class momentary_rot_button(Parentname, Targetname, RenderFields, Angles, Origin):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Origin).__init__()
        self.speed = 50  # Type: integer
        self.master = None  # Type: string
        self.sounds = None  # Type: choices
        self.distance = 90  # Type: integer
        self.returnspeed = None  # Type: integer
        self._minlight = None  # Type: string
        self.startposition = None  # Type: float
        self.startdirection = "CHOICES NOT SUPPORTED"  # Type: choices
        self.solidbsp = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.speed = parse_source_value(entity_data.get('speed', 50))  # Type: integer
        instance.master = entity_data.get('master', None)  # Type: string
        instance.sounds = entity_data.get('sounds', None)  # Type: choices
        instance.distance = parse_source_value(entity_data.get('distance', 90))  # Type: integer
        instance.returnspeed = parse_source_value(entity_data.get('returnspeed', 0))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.startposition = float(entity_data.get('startposition', 0))  # Type: float
        instance.startdirection = entity_data.get('startdirection', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.solidbsp = entity_data.get('solidbsp', None)  # Type: boolean


class Door(Parentname, Shadow, Targetname, RenderFields, Global, Reflection):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
        super(Reflection).__init__()
        self.speed = 100  # Type: integer
        self.master = None  # Type: string
        self.noise1 = None  # Type: sound
        self.noise2 = None  # Type: sound
        self.startclosesound = None  # Type: sound
        self.closesound = None  # Type: sound
        self.wait = 4  # Type: integer
        self.lip = None  # Type: integer
        self.dmg = None  # Type: integer
        self.forceclosed = None  # Type: boolean
        self.ignoredebris = None  # Type: boolean
        self.message = None  # Type: string
        self.health = None  # Type: integer
        self.locked_sound = None  # Type: sound
        self.unlocked_sound = None  # Type: sound
        self.spawnpos = None  # Type: choices
        self.locked_sentence = None  # Type: choices
        self.unlocked_sentence = None  # Type: choices
        self._minlight = None  # Type: string
        self.loopmovesound = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Reflection.from_dict(instance, entity_data)
        instance.speed = parse_source_value(entity_data.get('speed', 100))  # Type: integer
        instance.master = entity_data.get('master', None)  # Type: string
        instance.noise1 = entity_data.get('noise1', None)  # Type: sound
        instance.noise2 = entity_data.get('noise2', None)  # Type: sound
        instance.startclosesound = entity_data.get('startclosesound', None)  # Type: sound
        instance.closesound = entity_data.get('closesound', None)  # Type: sound
        instance.wait = parse_source_value(entity_data.get('wait', 4))  # Type: integer
        instance.lip = parse_source_value(entity_data.get('lip', 0))  # Type: integer
        instance.dmg = parse_source_value(entity_data.get('dmg', 0))  # Type: integer
        instance.forceclosed = entity_data.get('forceclosed', None)  # Type: boolean
        instance.ignoredebris = entity_data.get('ignoredebris', None)  # Type: boolean
        instance.message = entity_data.get('message', None)  # Type: string
        instance.health = parse_source_value(entity_data.get('health', 0))  # Type: integer
        instance.locked_sound = entity_data.get('locked_sound', None)  # Type: sound
        instance.unlocked_sound = entity_data.get('unlocked_sound', None)  # Type: sound
        instance.spawnpos = entity_data.get('spawnpos', None)  # Type: choices
        instance.locked_sentence = entity_data.get('locked_sentence', None)  # Type: choices
        instance.unlocked_sentence = entity_data.get('unlocked_sentence', None)  # Type: choices
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.loopmovesound = entity_data.get('loopmovesound', None)  # Type: boolean


class func_door(Door, Origin):
    def __init__(self):
        super(Door).__init__()
        super(Origin).__init__()
        self.movedir = [0.0, 0.0, 0.0]  # Type: angle
        self.filtername = None  # Type: filterclass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Door.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.movedir = parse_float_vector(entity_data.get('movedir', "0 0 0"))  # Type: angle
        instance.filtername = entity_data.get('filtername', None)  # Type: filterclass


class func_door_rotating(Angles, Door, Origin):
    def __init__(self):
        super(Door).__init__()
        super(Angles).__init__()
        super(Origin).__init__()
        self.distance = 90  # Type: integer
        self.solidbsp = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Door.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.distance = parse_source_value(entity_data.get('distance', 90))  # Type: integer
        instance.solidbsp = entity_data.get('solidbsp', None)  # Type: choices


class prop_door_rotating(Parentname, Studiomodel, Targetname, Global, Angles):
    def __init__(self):
        super(Studiomodel).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.slavename = None  # Type: target_destination
        self.hardware = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ajarangles = [0.0, 0.0, 0.0]  # Type: angle
        self.spawnpos = None  # Type: choices
        self.axis = None  # Type: axis
        self.distance = 90  # Type: float
        self.speed = 100  # Type: integer
        self.soundopenoverride = None  # Type: sound
        self.soundcloseoverride = None  # Type: sound
        self.soundmoveoverride = None  # Type: sound
        self.returndelay = -1  # Type: integer
        self.dmg = None  # Type: integer
        self.health = None  # Type: integer
        self.soundlockedoverride = None  # Type: sound
        self.soundunlockedoverride = None  # Type: sound
        self.forceclosed = None  # Type: boolean
        self.opendir = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.slavename = entity_data.get('slavename', None)  # Type: target_destination
        instance.hardware = entity_data.get('hardware', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ajarangles = parse_float_vector(entity_data.get('ajarangles', "0 0 0"))  # Type: angle
        instance.spawnpos = entity_data.get('spawnpos', None)  # Type: choices
        instance.axis = entity_data.get('axis', None)  # Type: axis
        instance.distance = float(entity_data.get('distance', 90))  # Type: float
        instance.speed = parse_source_value(entity_data.get('speed', 100))  # Type: integer
        instance.soundopenoverride = entity_data.get('soundopenoverride', None)  # Type: sound
        instance.soundcloseoverride = entity_data.get('soundcloseoverride', None)  # Type: sound
        instance.soundmoveoverride = entity_data.get('soundmoveoverride', None)  # Type: sound
        instance.returndelay = parse_source_value(entity_data.get('returndelay', -1))  # Type: integer
        instance.dmg = parse_source_value(entity_data.get('dmg', 0))  # Type: integer
        instance.health = parse_source_value(entity_data.get('health', 0))  # Type: integer
        instance.soundlockedoverride = entity_data.get('soundlockedoverride', None)  # Type: sound
        instance.soundunlockedoverride = entity_data.get('soundunlockedoverride', None)  # Type: sound
        instance.forceclosed = entity_data.get('forceclosed', None)  # Type: boolean
        instance.opendir = entity_data.get('opendir', None)  # Type: choices


class env_cubemap(Base):
    icon_sprite = "editor/env_cubemap.vmt"
    def __init__(self):
        super().__init__()
        self.origin = [0, 0, 0]
        self.cubemapsize = None  # Type: choices
        self.sides = None  # Type: sidelist

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.cubemapsize = entity_data.get('cubemapsize', None)  # Type: choices
        instance.sides = entity_data.get('sides', None)  # Type: sidelist


class BModelParticleSpawner(Base):
    def __init__(self):
        super().__init__()
        self.StartDisabled = None  # Type: boolean
        self.Color = [255, 255, 255]  # Type: color255
        self.SpawnRate = 40  # Type: integer
        self.SpeedMax = "13"  # Type: string
        self.LifetimeMin = "3"  # Type: string
        self.LifetimeMax = "5"  # Type: string
        self.DistMax = 1024  # Type: integer
        self.Frozen = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.StartDisabled = entity_data.get('startdisabled', None)  # Type: boolean
        instance.Color = parse_int_vector(entity_data.get('color', "255 255 255"))  # Type: color255
        instance.SpawnRate = parse_source_value(entity_data.get('spawnrate', 40))  # Type: integer
        instance.SpeedMax = entity_data.get('speedmax', "13")  # Type: string
        instance.LifetimeMin = entity_data.get('lifetimemin', "3")  # Type: string
        instance.LifetimeMax = entity_data.get('lifetimemax', "5")  # Type: string
        instance.DistMax = parse_source_value(entity_data.get('distmax', 1024))  # Type: integer
        instance.Frozen = entity_data.get('frozen', None)  # Type: boolean


class func_dustmotes(Targetname, BModelParticleSpawner):
    def __init__(self):
        super(Targetname).__init__()
        super(BModelParticleSpawner).__init__()
        self.SizeMin = "10"  # Type: string
        self.SizeMax = "20"  # Type: string
        self.Alpha = 255  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        BModelParticleSpawner.from_dict(instance, entity_data)
        instance.SizeMin = entity_data.get('sizemin', "10")  # Type: string
        instance.SizeMax = entity_data.get('sizemax', "20")  # Type: string
        instance.Alpha = parse_source_value(entity_data.get('alpha', 255))  # Type: integer


class func_smokevolume(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.Color1 = [255, 255, 255]  # Type: color255
        self.Color2 = [255, 255, 255]  # Type: color255
        self.material = "particle/particle_smokegrenade"  # Type: material
        self.ParticleDrawWidth = 120  # Type: float
        self.ParticleSpacingDistance = 80  # Type: float
        self.DensityRampSpeed = 1  # Type: float
        self.RotationSpeed = 10  # Type: float
        self.MovementSpeed = 10  # Type: float
        self.Density = 1  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.Color1 = parse_int_vector(entity_data.get('color1', "255 255 255"))  # Type: color255
        instance.Color2 = parse_int_vector(entity_data.get('color2', "255 255 255"))  # Type: color255
        instance.material = entity_data.get('material', "particle/particle_smokegrenade")  # Type: material
        instance.ParticleDrawWidth = float(entity_data.get('particledrawwidth', 120))  # Type: float
        instance.ParticleSpacingDistance = float(entity_data.get('particlespacingdistance', 80))  # Type: float
        instance.DensityRampSpeed = float(entity_data.get('densityrampspeed', 1))  # Type: float
        instance.RotationSpeed = float(entity_data.get('rotationspeed', 10))  # Type: float
        instance.MovementSpeed = float(entity_data.get('movementspeed', 10))  # Type: float
        instance.Density = float(entity_data.get('density', 1))  # Type: float


class func_dustcloud(Targetname, BModelParticleSpawner):
    def __init__(self):
        super(Targetname).__init__()
        super(BModelParticleSpawner).__init__()
        self.Alpha = 30  # Type: integer
        self.SizeMin = "100"  # Type: string
        self.SizeMax = "200"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        BModelParticleSpawner.from_dict(instance, entity_data)
        instance.Alpha = parse_source_value(entity_data.get('alpha', 30))  # Type: integer
        instance.SizeMin = entity_data.get('sizemin', "100")  # Type: string
        instance.SizeMax = entity_data.get('sizemax', "200")  # Type: string


class env_dustpuff(Targetname, Angles, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.scale = 8  # Type: float
        self.speed = 16  # Type: float
        self.color = [128, 128, 128]  # Type: color255

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.scale = float(entity_data.get('scale', 8))  # Type: float
        instance.speed = float(entity_data.get('speed', 16))  # Type: float
        instance.color = parse_int_vector(entity_data.get('color', "128 128 128"))  # Type: color255


class env_particlescript(Targetname, Angles, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.model = "models/Ambient_citadel_paths.mdl"  # Type: studio

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', "models/Ambient_citadel_paths.mdl")  # Type: studio


class env_effectscript(Targetname, Angles, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.model = "models/Effects/teleporttrail.mdl"  # Type: studio
        self.scriptfile = "scripts/effects/testeffect.txt"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', "models/Effects/teleporttrail.mdl")  # Type: studio
        instance.scriptfile = entity_data.get('scriptfile', "scripts/effects/testeffect.txt")  # Type: string


class logic_auto(Base):
    icon_sprite = "editor/logic_auto.vmt"
    def __init__(self):
        super().__init__()
        self.origin = [0, 0, 0]
        self.globalstate = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.globalstate = entity_data.get('globalstate', None)  # Type: choices


class point_viewcontrol(Targetname, Angles, Parentname):
    viewport_model = "models/editor/camera.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.targetattachment = None  # Type: string
        self.wait = 10  # Type: integer
        self.moveto = None  # Type: target_destination
        self.interpolatepositiontoplayer = None  # Type: boolean
        self.trackspeed = 40  # Type: float
        self.fov = 90  # Type: float
        self.fov_rate = 1  # Type: float
        self.speed = "0"  # Type: string
        self.acceleration = "500"  # Type: string
        self.deceleration = "500"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.targetattachment = entity_data.get('targetattachment', None)  # Type: string
        instance.wait = parse_source_value(entity_data.get('wait', 10))  # Type: integer
        instance.moveto = entity_data.get('moveto', None)  # Type: target_destination
        instance.interpolatepositiontoplayer = entity_data.get('interpolatepositiontoplayer', None)  # Type: boolean
        instance.trackspeed = float(entity_data.get('trackspeed', 40))  # Type: float
        instance.fov = float(entity_data.get('fov', 90))  # Type: float
        instance.fov_rate = float(entity_data.get('fov_rate', 1))  # Type: float
        instance.speed = entity_data.get('speed', "0")  # Type: string
        instance.acceleration = entity_data.get('acceleration', "500")  # Type: string
        instance.deceleration = entity_data.get('deceleration', "500")  # Type: string


class point_viewcontrol_multiplayer(Targetname, Angles, Parentname):
    viewport_model = "models/editor/camera.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.fov = 90  # Type: float
        self.fov_rate = 1.0  # Type: float
        self.target_entity = None  # Type: target_destination
        self.interp_time = 1.0  # Type: float
        self.target_team = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.fov = float(entity_data.get('fov', 90))  # Type: float
        instance.fov_rate = float(entity_data.get('fov_rate', 1.0))  # Type: float
        instance.target_entity = entity_data.get('target_entity', None)  # Type: target_destination
        instance.interp_time = float(entity_data.get('interp_time', 1.0))  # Type: float
        instance.target_team = entity_data.get('target_team', "CHOICES NOT SUPPORTED")  # Type: choices


class point_viewproxy(Targetname, Angles, Parentname):
    viewport_model = "models/editor/camera.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.offsettype = None  # Type: choices
        self.proxy = None  # Type: target_destination
        self.proxyattachment = None  # Type: target_destination
        self.tiltfraction = 0.5  # Type: float
        self.usefakeacceleration = None  # Type: boolean
        self.skewaccelerationforward = 1  # Type: boolean
        self.accelerationscalar = 1.0  # Type: float
        self.easeanglestocamera = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.offsettype = entity_data.get('offsettype', None)  # Type: choices
        instance.proxy = entity_data.get('proxy', None)  # Type: target_destination
        instance.proxyattachment = entity_data.get('proxyattachment', None)  # Type: target_destination
        instance.tiltfraction = float(entity_data.get('tiltfraction', 0.5))  # Type: float
        instance.usefakeacceleration = entity_data.get('usefakeacceleration', None)  # Type: boolean
        instance.skewaccelerationforward = entity_data.get('skewaccelerationforward', None)  # Type: boolean
        instance.accelerationscalar = float(entity_data.get('accelerationscalar', 1.0))  # Type: float
        instance.easeanglestocamera = entity_data.get('easeanglestocamera', None)  # Type: boolean


class point_posecontroller(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.PropName = None  # Type: string
        self.PoseParameterName = None  # Type: string
        self.PoseValue = 0.0  # Type: float
        self.InterpolationTime = 0.0  # Type: float
        self.InterpolationWrap = None  # Type: boolean
        self.CycleFrequency = 0.0  # Type: float
        self.FModulationType = None  # Type: choices
        self.FModTimeOffset = 0.0  # Type: float
        self.FModRate = 0.0  # Type: float
        self.FModAmplitude = 0.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.PropName = entity_data.get('propname', None)  # Type: string
        instance.PoseParameterName = entity_data.get('poseparametername', None)  # Type: string
        instance.PoseValue = float(entity_data.get('posevalue', 0.0))  # Type: float
        instance.InterpolationTime = float(entity_data.get('interpolationtime', 0.0))  # Type: float
        instance.InterpolationWrap = entity_data.get('interpolationwrap', None)  # Type: boolean
        instance.CycleFrequency = float(entity_data.get('cyclefrequency', 0.0))  # Type: float
        instance.FModulationType = entity_data.get('fmodulationtype', None)  # Type: choices
        instance.FModTimeOffset = float(entity_data.get('fmodtimeoffset', 0.0))  # Type: float
        instance.FModRate = float(entity_data.get('fmodrate', 0.0))  # Type: float
        instance.FModAmplitude = float(entity_data.get('fmodamplitude', 0.0))  # Type: float


class logic_compare(Targetname):
    icon_sprite = "editor/logic_compare.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.InitialValue = None  # Type: integer
        self.CompareValue = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.InitialValue = parse_source_value(entity_data.get('initialvalue', 0))  # Type: integer
        instance.CompareValue = parse_source_value(entity_data.get('comparevalue', 0))  # Type: integer


class logic_branch(Targetname):
    icon_sprite = "editor/logic_branch.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.InitialValue = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.InitialValue = parse_source_value(entity_data.get('initialvalue', 0))  # Type: integer


class logic_branch_listener(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.Branch01 = None  # Type: target_destination
        self.Branch02 = None  # Type: target_destination
        self.Branch03 = None  # Type: target_destination
        self.Branch04 = None  # Type: target_destination
        self.Branch05 = None  # Type: target_destination
        self.Branch06 = None  # Type: target_destination
        self.Branch07 = None  # Type: target_destination
        self.Branch08 = None  # Type: target_destination
        self.Branch09 = None  # Type: target_destination
        self.Branch10 = None  # Type: target_destination
        self.Branch11 = None  # Type: target_destination
        self.Branch12 = None  # Type: target_destination
        self.Branch13 = None  # Type: target_destination
        self.Branch14 = None  # Type: target_destination
        self.Branch15 = None  # Type: target_destination
        self.Branch16 = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Branch01 = entity_data.get('branch01', None)  # Type: target_destination
        instance.Branch02 = entity_data.get('branch02', None)  # Type: target_destination
        instance.Branch03 = entity_data.get('branch03', None)  # Type: target_destination
        instance.Branch04 = entity_data.get('branch04', None)  # Type: target_destination
        instance.Branch05 = entity_data.get('branch05', None)  # Type: target_destination
        instance.Branch06 = entity_data.get('branch06', None)  # Type: target_destination
        instance.Branch07 = entity_data.get('branch07', None)  # Type: target_destination
        instance.Branch08 = entity_data.get('branch08', None)  # Type: target_destination
        instance.Branch09 = entity_data.get('branch09', None)  # Type: target_destination
        instance.Branch10 = entity_data.get('branch10', None)  # Type: target_destination
        instance.Branch11 = entity_data.get('branch11', None)  # Type: target_destination
        instance.Branch12 = entity_data.get('branch12', None)  # Type: target_destination
        instance.Branch13 = entity_data.get('branch13', None)  # Type: target_destination
        instance.Branch14 = entity_data.get('branch14', None)  # Type: target_destination
        instance.Branch15 = entity_data.get('branch15', None)  # Type: target_destination
        instance.Branch16 = entity_data.get('branch16', None)  # Type: target_destination


class logic_case(Targetname):
    icon_sprite = "editor/logic_case.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.Case01 = None  # Type: string
        self.Case02 = None  # Type: string
        self.Case03 = None  # Type: string
        self.Case04 = None  # Type: string
        self.Case05 = None  # Type: string
        self.Case06 = None  # Type: string
        self.Case07 = None  # Type: string
        self.Case08 = None  # Type: string
        self.Case09 = None  # Type: string
        self.Case10 = None  # Type: string
        self.Case11 = None  # Type: string
        self.Case12 = None  # Type: string
        self.Case13 = None  # Type: string
        self.Case14 = None  # Type: string
        self.Case15 = None  # Type: string
        self.Case16 = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Case01 = entity_data.get('case01', None)  # Type: string
        instance.Case02 = entity_data.get('case02', None)  # Type: string
        instance.Case03 = entity_data.get('case03', None)  # Type: string
        instance.Case04 = entity_data.get('case04', None)  # Type: string
        instance.Case05 = entity_data.get('case05', None)  # Type: string
        instance.Case06 = entity_data.get('case06', None)  # Type: string
        instance.Case07 = entity_data.get('case07', None)  # Type: string
        instance.Case08 = entity_data.get('case08', None)  # Type: string
        instance.Case09 = entity_data.get('case09', None)  # Type: string
        instance.Case10 = entity_data.get('case10', None)  # Type: string
        instance.Case11 = entity_data.get('case11', None)  # Type: string
        instance.Case12 = entity_data.get('case12', None)  # Type: string
        instance.Case13 = entity_data.get('case13', None)  # Type: string
        instance.Case14 = entity_data.get('case14', None)  # Type: string
        instance.Case15 = entity_data.get('case15', None)  # Type: string
        instance.Case16 = entity_data.get('case16', None)  # Type: string


class logic_multicompare(Targetname):
    icon_sprite = "editor/logic_multicompare.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.IntegerValue = None  # Type: integer
        self.ShouldComparetoValue = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.IntegerValue = parse_source_value(entity_data.get('integervalue', 0))  # Type: integer
        instance.ShouldComparetoValue = entity_data.get('shouldcomparetovalue', None)  # Type: boolean


class logic_relay(Targetname, EnableDisable):
    icon_sprite = "editor/logic_relay.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class logic_register_activator(Targetname, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class logic_random_outputs(Targetname, EnableDisable):
    icon_sprite = "editor/logic_random_outputs.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.OnTriggerChance1 = 1.0  # Type: float
        self.OnTriggerChance2 = 1.0  # Type: float
        self.OnTriggerChance3 = 1.0  # Type: float
        self.OnTriggerChance4 = 1.0  # Type: float
        self.OnTriggerChance5 = 1.0  # Type: float
        self.OnTriggerChance6 = 1.0  # Type: float
        self.OnTriggerChance7 = 1.0  # Type: float
        self.OnTriggerChance8 = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.OnTriggerChance1 = float(entity_data.get('ontriggerchance1', 1.0))  # Type: float
        instance.OnTriggerChance2 = float(entity_data.get('ontriggerchance2', 1.0))  # Type: float
        instance.OnTriggerChance3 = float(entity_data.get('ontriggerchance3', 1.0))  # Type: float
        instance.OnTriggerChance4 = float(entity_data.get('ontriggerchance4', 1.0))  # Type: float
        instance.OnTriggerChance5 = float(entity_data.get('ontriggerchance5', 1.0))  # Type: float
        instance.OnTriggerChance6 = float(entity_data.get('ontriggerchance6', 1.0))  # Type: float
        instance.OnTriggerChance7 = float(entity_data.get('ontriggerchance7', 1.0))  # Type: float
        instance.OnTriggerChance8 = float(entity_data.get('ontriggerchance8', 1.0))  # Type: float


class logic_script(Targetname):
    icon_sprite = "editor/logic_script.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.Group00 = None  # Type: target_destination
        self.Group01 = None  # Type: target_destination
        self.Group02 = None  # Type: target_destination
        self.Group03 = None  # Type: target_destination
        self.Group04 = None  # Type: target_destination
        self.Group05 = None  # Type: target_destination
        self.Group06 = None  # Type: target_destination
        self.Group07 = None  # Type: target_destination
        self.Group08 = None  # Type: target_destination
        self.Group09 = None  # Type: target_destination
        self.Group10 = None  # Type: target_destination
        self.Group11 = None  # Type: target_destination
        self.Group12 = None  # Type: target_destination
        self.Group13 = None  # Type: target_destination
        self.Group14 = None  # Type: target_destination
        self.Group15 = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Group00 = entity_data.get('group00', None)  # Type: target_destination
        instance.Group01 = entity_data.get('group01', None)  # Type: target_destination
        instance.Group02 = entity_data.get('group02', None)  # Type: target_destination
        instance.Group03 = entity_data.get('group03', None)  # Type: target_destination
        instance.Group04 = entity_data.get('group04', None)  # Type: target_destination
        instance.Group05 = entity_data.get('group05', None)  # Type: target_destination
        instance.Group06 = entity_data.get('group06', None)  # Type: target_destination
        instance.Group07 = entity_data.get('group07', None)  # Type: target_destination
        instance.Group08 = entity_data.get('group08', None)  # Type: target_destination
        instance.Group09 = entity_data.get('group09', None)  # Type: target_destination
        instance.Group10 = entity_data.get('group10', None)  # Type: target_destination
        instance.Group11 = entity_data.get('group11', None)  # Type: target_destination
        instance.Group12 = entity_data.get('group12', None)  # Type: target_destination
        instance.Group13 = entity_data.get('group13', None)  # Type: target_destination
        instance.Group14 = entity_data.get('group14', None)  # Type: target_destination
        instance.Group15 = entity_data.get('group15', None)  # Type: target_destination


class logic_timer(Targetname, EnableDisable):
    icon_sprite = "editor/logic_timer.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.UseRandomTime = None  # Type: boolean
        self.LowerRandomBound = None  # Type: string
        self.UpperRandomBound = None  # Type: string
        self.RefireTime = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.UseRandomTime = entity_data.get('userandomtime', None)  # Type: boolean
        instance.LowerRandomBound = entity_data.get('lowerrandombound', None)  # Type: string
        instance.UpperRandomBound = entity_data.get('upperrandombound', None)  # Type: string
        instance.RefireTime = entity_data.get('refiretime', None)  # Type: string


class hammer_updateignorelist(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.IgnoredName01 = None  # Type: target_destination
        self.IgnoredName02 = None  # Type: target_destination
        self.IgnoredName03 = None  # Type: target_destination
        self.IgnoredName04 = None  # Type: target_destination
        self.IgnoredName05 = None  # Type: target_destination
        self.IgnoredName06 = None  # Type: target_destination
        self.IgnoredName07 = None  # Type: target_destination
        self.IgnoredName08 = None  # Type: target_destination
        self.IgnoredName09 = None  # Type: target_destination
        self.IgnoredName10 = None  # Type: target_destination
        self.IgnoredName11 = None  # Type: target_destination
        self.IgnoredName12 = None  # Type: target_destination
        self.IgnoredName13 = None  # Type: target_destination
        self.IgnoredName14 = None  # Type: target_destination
        self.IgnoredName15 = None  # Type: target_destination
        self.IgnoredName16 = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.IgnoredName01 = entity_data.get('ignoredname01', None)  # Type: target_destination
        instance.IgnoredName02 = entity_data.get('ignoredname02', None)  # Type: target_destination
        instance.IgnoredName03 = entity_data.get('ignoredname03', None)  # Type: target_destination
        instance.IgnoredName04 = entity_data.get('ignoredname04', None)  # Type: target_destination
        instance.IgnoredName05 = entity_data.get('ignoredname05', None)  # Type: target_destination
        instance.IgnoredName06 = entity_data.get('ignoredname06', None)  # Type: target_destination
        instance.IgnoredName07 = entity_data.get('ignoredname07', None)  # Type: target_destination
        instance.IgnoredName08 = entity_data.get('ignoredname08', None)  # Type: target_destination
        instance.IgnoredName09 = entity_data.get('ignoredname09', None)  # Type: target_destination
        instance.IgnoredName10 = entity_data.get('ignoredname10', None)  # Type: target_destination
        instance.IgnoredName11 = entity_data.get('ignoredname11', None)  # Type: target_destination
        instance.IgnoredName12 = entity_data.get('ignoredname12', None)  # Type: target_destination
        instance.IgnoredName13 = entity_data.get('ignoredname13', None)  # Type: target_destination
        instance.IgnoredName14 = entity_data.get('ignoredname14', None)  # Type: target_destination
        instance.IgnoredName15 = entity_data.get('ignoredname15', None)  # Type: target_destination
        instance.IgnoredName16 = entity_data.get('ignoredname16', None)  # Type: target_destination


class logic_collision_pair(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.attach1 = None  # Type: target_destination
        self.attach2 = None  # Type: target_destination
        self.startdisabled = 1  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.attach1 = entity_data.get('attach1', None)  # Type: target_destination
        instance.attach2 = entity_data.get('attach2', None)  # Type: target_destination
        instance.startdisabled = entity_data.get('startdisabled', None)  # Type: boolean


class env_microphone(Targetname, Parentname, EnableDisable):
    icon_sprite = "editor/env_microphone.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.SpeakerName = None  # Type: target_destination
        self.ListenFilter = None  # Type: filterclass
        self.speaker_dsp_preset = None  # Type: choices
        self.Sensitivity = 1  # Type: float
        self.SmoothFactor = None  # Type: float
        self.MaxRange = 240  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.SpeakerName = entity_data.get('speakername', None)  # Type: target_destination
        instance.ListenFilter = entity_data.get('listenfilter', None)  # Type: filterclass
        instance.speaker_dsp_preset = entity_data.get('speaker_dsp_preset', None)  # Type: choices
        instance.Sensitivity = float(entity_data.get('sensitivity', 1))  # Type: float
        instance.SmoothFactor = float(entity_data.get('smoothfactor', 0))  # Type: float
        instance.MaxRange = float(entity_data.get('maxrange', 240))  # Type: float


class math_remap(Targetname, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.in1 = None  # Type: integer
        self.in2 = 1  # Type: integer
        self.out1 = None  # Type: integer
        self.out2 = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.in1 = parse_source_value(entity_data.get('in1', 0))  # Type: integer
        instance.in2 = parse_source_value(entity_data.get('in2', 1))  # Type: integer
        instance.out1 = parse_source_value(entity_data.get('out1', 0))  # Type: integer
        instance.out2 = parse_source_value(entity_data.get('out2', 0))  # Type: integer


class math_colorblend(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.inmin = None  # Type: integer
        self.inmax = 1  # Type: integer
        self.colormin = [0, 0, 0]  # Type: color255
        self.colormax = [255, 255, 255]  # Type: color255

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.inmin = parse_source_value(entity_data.get('inmin', 0))  # Type: integer
        instance.inmax = parse_source_value(entity_data.get('inmax', 1))  # Type: integer
        instance.colormin = parse_int_vector(entity_data.get('colormin', "0 0 0"))  # Type: color255
        instance.colormax = parse_int_vector(entity_data.get('colormax', "255 255 255"))  # Type: color255


class math_counter(Targetname, EnableDisable):
    icon_sprite = "editor/math_counter.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.startvalue = None  # Type: integer
        self.min = None  # Type: integer
        self.max = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.startvalue = parse_source_value(entity_data.get('startvalue', 0))  # Type: integer
        instance.min = parse_source_value(entity_data.get('min', 0))  # Type: integer
        instance.max = parse_source_value(entity_data.get('max', 0))  # Type: integer


class logic_lineto(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.source = None  # Type: target_destination
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.source = entity_data.get('source', None)  # Type: target_destination
        instance.target = entity_data.get('target', None)  # Type: target_destination


class logic_navigation(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Set to none due to bug in BlackMesa base.fgd file  # Type: target_destination
        self.navprop = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Set to none due to bug in BlackMesa base.fgd file  # Type: target_destination
        instance.navprop = entity_data.get('navprop', "CHOICES NOT SUPPORTED")  # Type: choices


class logic_autosave(Targetname):
    icon_sprite = "editor/logic_autosave.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.NewLevelUnit = None  # Type: boolean
        self.MinimumHitPoints = None  # Type: integer
        self.MinHitPointsToCommit = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.NewLevelUnit = entity_data.get('newlevelunit', None)  # Type: boolean
        instance.MinimumHitPoints = parse_source_value(entity_data.get('minimumhitpoints', 0))  # Type: integer
        instance.MinHitPointsToCommit = parse_source_value(entity_data.get('minhitpointstocommit', 0))  # Type: integer


class logic_active_autosave(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.MinimumHitPoints = 30  # Type: integer
        self.TriggerHitPoints = 75  # Type: integer
        self.TimeToTrigget = None  # Type: float
        self.DangerousTime = 10  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.MinimumHitPoints = parse_source_value(entity_data.get('minimumhitpoints', 30))  # Type: integer
        instance.TriggerHitPoints = parse_source_value(entity_data.get('triggerhitpoints', 75))  # Type: integer
        instance.TimeToTrigget = float(entity_data.get('timetotrigget', 0))  # Type: float
        instance.DangerousTime = float(entity_data.get('dangeroustime', 10))  # Type: float


class logic_playmovie(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.MovieFilename = None  # Type: string
        self.allowskip = None  # Type: boolean
        self.loopvideo = None  # Type: boolean
        self.fadeintime = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.MovieFilename = entity_data.get('moviefilename', None)  # Type: string
        instance.allowskip = entity_data.get('allowskip', None)  # Type: boolean
        instance.loopvideo = entity_data.get('loopvideo', None)  # Type: boolean
        instance.fadeintime = float(entity_data.get('fadeintime', 0))  # Type: float


class point_template(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.Template01 = None  # Type: target_destination
        self.Template02 = None  # Type: target_destination
        self.Template03 = None  # Type: target_destination
        self.Template04 = None  # Type: target_destination
        self.Template05 = None  # Type: target_destination
        self.Template06 = None  # Type: target_destination
        self.Template07 = None  # Type: target_destination
        self.Template08 = None  # Type: target_destination
        self.Template09 = None  # Type: target_destination
        self.Template10 = None  # Type: target_destination
        self.Template11 = None  # Type: target_destination
        self.Template12 = None  # Type: target_destination
        self.Template13 = None  # Type: target_destination
        self.Template14 = None  # Type: target_destination
        self.Template15 = None  # Type: target_destination
        self.Template16 = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Template01 = entity_data.get('template01', None)  # Type: target_destination
        instance.Template02 = entity_data.get('template02', None)  # Type: target_destination
        instance.Template03 = entity_data.get('template03', None)  # Type: target_destination
        instance.Template04 = entity_data.get('template04', None)  # Type: target_destination
        instance.Template05 = entity_data.get('template05', None)  # Type: target_destination
        instance.Template06 = entity_data.get('template06', None)  # Type: target_destination
        instance.Template07 = entity_data.get('template07', None)  # Type: target_destination
        instance.Template08 = entity_data.get('template08', None)  # Type: target_destination
        instance.Template09 = entity_data.get('template09', None)  # Type: target_destination
        instance.Template10 = entity_data.get('template10', None)  # Type: target_destination
        instance.Template11 = entity_data.get('template11', None)  # Type: target_destination
        instance.Template12 = entity_data.get('template12', None)  # Type: target_destination
        instance.Template13 = entity_data.get('template13', None)  # Type: target_destination
        instance.Template14 = entity_data.get('template14', None)  # Type: target_destination
        instance.Template15 = entity_data.get('template15', None)  # Type: target_destination
        instance.Template16 = entity_data.get('template16', None)  # Type: target_destination


class env_entity_maker(Targetname, Angles, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.EntityTemplate = None  # Type: target_destination
        self.PostSpawnSpeed = 0  # Type: float
        self.PostSpawnDirection = [0.0, 0.0, 0.0]  # Type: angle
        self.PostSpawnDirectionVariance = 0.15  # Type: float
        self.PostSpawnInheritAngles = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.EntityTemplate = entity_data.get('entitytemplate', None)  # Type: target_destination
        instance.PostSpawnSpeed = float(entity_data.get('postspawnspeed', 0))  # Type: float
        instance.PostSpawnDirection = parse_float_vector(entity_data.get('postspawndirection', "0 0 0"))  # Type: angle
        instance.PostSpawnDirectionVariance = float(entity_data.get('postspawndirectionvariance', 0.15))  # Type: float
        instance.PostSpawnInheritAngles = entity_data.get('postspawninheritangles', None)  # Type: boolean


class BaseFilter(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.Negated = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.Negated = entity_data.get('negated', "CHOICES NOT SUPPORTED")  # Type: choices


class filter_multi(BaseFilter):
    icon_sprite = "editor/filter_multiple.vmt"
    def __init__(self):
        super(BaseFilter).__init__()
        self.filtertype = None  # Type: choices
        self.Filter01 = None  # Type: filterclass
        self.Filter02 = None  # Type: filterclass
        self.Filter03 = None  # Type: filterclass
        self.Filter04 = None  # Type: filterclass
        self.Filter05 = None  # Type: filterclass
        self.Filter06 = None  # Type: filterclass
        self.Filter07 = None  # Type: filterclass
        self.Filter08 = None  # Type: filterclass
        self.Filter09 = None  # Type: filterclass
        self.Filter10 = None  # Type: filterclass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.filtertype = entity_data.get('filtertype', None)  # Type: choices
        instance.Filter01 = entity_data.get('filter01', None)  # Type: filterclass
        instance.Filter02 = entity_data.get('filter02', None)  # Type: filterclass
        instance.Filter03 = entity_data.get('filter03', None)  # Type: filterclass
        instance.Filter04 = entity_data.get('filter04', None)  # Type: filterclass
        instance.Filter05 = entity_data.get('filter05', None)  # Type: filterclass
        instance.Filter06 = entity_data.get('filter06', None)  # Type: filterclass
        instance.Filter07 = entity_data.get('filter07', None)  # Type: filterclass
        instance.Filter08 = entity_data.get('filter08', None)  # Type: filterclass
        instance.Filter09 = entity_data.get('filter09', None)  # Type: filterclass
        instance.Filter10 = entity_data.get('filter10', None)  # Type: filterclass


class filter_activator_name(BaseFilter):
    icon_sprite = "editor/filter_name.vmt"
    def __init__(self):
        super(BaseFilter).__init__()
        self.filtername = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.filtername = entity_data.get('filtername', None)  # Type: target_destination


class filter_activator_model(BaseFilter):
    icon_sprite = "editor/filter_name.vmt"
    def __init__(self):
        super(BaseFilter).__init__()
        self.model = None  # Type: studio

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', None)  # Type: studio


class filter_activator_context(BaseFilter):
    icon_sprite = "editor/filter_name.vmt"
    def __init__(self):
        super(BaseFilter).__init__()
        self.ResponseContext = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.ResponseContext = entity_data.get('responsecontext', None)  # Type: string


class filter_activator_class(BaseFilter):
    icon_sprite = "editor/filter_class.vmt"
    def __init__(self):
        super(BaseFilter).__init__()
        self.filterclass = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.filterclass = entity_data.get('filterclass', None)  # Type: string


class filter_activator_mass_greater(BaseFilter):
    icon_sprite = "editor/filter_class.vmt"
    def __init__(self):
        super(BaseFilter).__init__()
        self.filtermass = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.filtermass = float(entity_data.get('filtermass', 0))  # Type: float


class filter_damage_type(BaseFilter):
    def __init__(self):
        super(BaseFilter).__init__()
        self.damagetype = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.damagetype = entity_data.get('damagetype', "CHOICES NOT SUPPORTED")  # Type: choices


class filter_enemy(BaseFilter):
    icon_sprite = "editor/filter_class.vmt"
    def __init__(self):
        super(BaseFilter).__init__()
        self.filtername = None  # Type: string
        self.filter_radius = None  # Type: float
        self.filter_outer_radius = None  # Type: float
        self.filter_max_per_enemy = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.filtername = entity_data.get('filtername', None)  # Type: string
        instance.filter_radius = float(entity_data.get('filter_radius', 0))  # Type: float
        instance.filter_outer_radius = float(entity_data.get('filter_outer_radius', 0))  # Type: float
        instance.filter_max_per_enemy = parse_source_value(entity_data.get('filter_max_per_enemy', 0))  # Type: integer


class point_anglesensor(Targetname, Parentname, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.lookatname = None  # Type: target_destination
        self.duration = None  # Type: float
        self.tolerance = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.lookatname = entity_data.get('lookatname', None)  # Type: target_destination
        instance.duration = float(entity_data.get('duration', 0))  # Type: float
        instance.tolerance = parse_source_value(entity_data.get('tolerance', 0))  # Type: integer


class point_angularvelocitysensor(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.threshold = None  # Type: float
        self.fireinterval = 0.2  # Type: float
        self.axis = None  # Type: vecline
        self.usehelper = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.threshold = float(entity_data.get('threshold', 0))  # Type: float
        instance.fireinterval = float(entity_data.get('fireinterval', 0.2))  # Type: float
        instance.axis = entity_data.get('axis', None)  # Type: vecline
        instance.usehelper = entity_data.get('usehelper', None)  # Type: boolean


class point_velocitysensor(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.axis = None  # Type: vecline
        self.enabled = 1  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.axis = entity_data.get('axis', None)  # Type: vecline
        instance.enabled = entity_data.get('enabled', None)  # Type: boolean


class point_proximity_sensor(Targetname, Angles, Parentname, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination


class point_teleport(Targetname, Angles):
    model = "models/editor/axis_helper_thick.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination


class point_hurt(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.DamageTarget = None  # Type: string
        self.DamageRadius = 256  # Type: float
        self.Damage = 5  # Type: integer
        self.DamageDelay = 1  # Type: float
        self.DamageType = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.DamageTarget = entity_data.get('damagetarget', None)  # Type: string
        instance.DamageRadius = float(entity_data.get('damageradius', 256))  # Type: float
        instance.Damage = parse_source_value(entity_data.get('damage', 5))  # Type: integer
        instance.DamageDelay = float(entity_data.get('damagedelay', 1))  # Type: float
        instance.DamageType = entity_data.get('damagetype', None)  # Type: choices


class point_playermoveconstraint(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.radius = 256  # Type: float
        self.width = 75.0  # Type: float
        self.speedfactor = 0.15  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.radius = float(entity_data.get('radius', 256))  # Type: float
        instance.width = float(entity_data.get('width', 75.0))  # Type: float
        instance.speedfactor = float(entity_data.get('speedfactor', 0.15))  # Type: float


class func_physbox(BreakableBrush, RenderFields, Origin):
    def __init__(self):
        super(BreakableBrush).__init__()
        super(RenderFields).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        super(Reflection).__init__()
        super(Origin).__init__()
        self._minlight = None  # Type: string
        self.Damagetype = None  # Type: choices
        self.massScale = 0  # Type: float
        self.overridescript = None  # Type: string
        self.damagetoenablemotion = None  # Type: integer
        self.forcetoenablemotion = None  # Type: float
        self.preferredcarryangles = [0.0, 0.0, 0.0]  # Type: vector
        self.notsolid = None  # Type: choices
        self.ExploitableByPlayer = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Shadow.from_dict(instance, entity_data)
        BreakableBrush.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Reflection.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.Damagetype = entity_data.get('damagetype', None)  # Type: choices
        instance.massScale = float(entity_data.get('massscale', 0))  # Type: float
        instance.overridescript = entity_data.get('overridescript', None)  # Type: string
        instance.damagetoenablemotion = parse_source_value(entity_data.get('damagetoenablemotion', 0))  # Type: integer
        instance.forcetoenablemotion = float(entity_data.get('forcetoenablemotion', 0))  # Type: float
        instance.preferredcarryangles = parse_float_vector(entity_data.get('preferredcarryangles', "0 0 0"))  # Type: vector
        instance.notsolid = entity_data.get('notsolid', None)  # Type: choices
        instance.ExploitableByPlayer = entity_data.get('exploitablebyplayer', None)  # Type: choices


class TwoObjectPhysics(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.attach1 = None  # Type: target_destination
        self.attach2 = None  # Type: target_destination
        self.constraintsystem = None  # Type: target_destination
        self.forcelimit = 0  # Type: float
        self.torquelimit = 0  # Type: float
        self.breaksound = None  # Type: sound
        self.teleportfollowdistance = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.attach1 = entity_data.get('attach1', None)  # Type: target_destination
        instance.attach2 = entity_data.get('attach2', None)  # Type: target_destination
        instance.constraintsystem = entity_data.get('constraintsystem', None)  # Type: target_destination
        instance.forcelimit = float(entity_data.get('forcelimit', 0))  # Type: float
        instance.torquelimit = float(entity_data.get('torquelimit', 0))  # Type: float
        instance.breaksound = entity_data.get('breaksound', None)  # Type: sound
        instance.teleportfollowdistance = float(entity_data.get('teleportfollowdistance', 0))  # Type: float


class phys_constraintsystem(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.additionaliterations = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.additionaliterations = parse_source_value(entity_data.get('additionaliterations', 0))  # Type: integer


class phys_keepupright(Targetname, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.attach1 = None  # Type: target_destination
        self.angularlimit = 15  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.attach1 = entity_data.get('attach1', None)  # Type: target_destination
        instance.angularlimit = float(entity_data.get('angularlimit', 15))  # Type: float


class physics_cannister(Targetname, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.model = "models/fire_equipment/w_weldtank.mdl"  # Type: studio
        self.expdamage = "200.0"  # Type: string
        self.expradius = "250.0"  # Type: string
        self.health = 25  # Type: integer
        self.thrust = "3000.0"  # Type: string
        self.fuel = "12.0"  # Type: string
        self.rendercolor = [255, 255, 255]  # Type: color255
        self.renderamt = 128  # Type: integer
        self.gassound = "ambient/objects/cannister_loop.wav"  # Type: sound

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', "models/fire_equipment/w_weldtank.mdl")  # Type: studio
        instance.expdamage = entity_data.get('expdamage', "200.0")  # Type: string
        instance.expradius = entity_data.get('expradius', "250.0")  # Type: string
        instance.health = parse_source_value(entity_data.get('health', 25))  # Type: integer
        instance.thrust = entity_data.get('thrust', "3000.0")  # Type: string
        instance.fuel = entity_data.get('fuel', "12.0")  # Type: string
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "255 255 255"))  # Type: color255
        instance.renderamt = parse_source_value(entity_data.get('renderamt', 128))  # Type: integer
        instance.gassound = entity_data.get('gassound', "ambient/objects/cannister_loop.wav")  # Type: sound


class info_constraint_anchor(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.massScale = 1  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.massScale = float(entity_data.get('massscale', 1))  # Type: float


class info_mass_center(Base):
    def __init__(self):
        super().__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination


class phys_spring(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.attach1 = None  # Type: target_destination
        self.attach2 = None  # Type: target_destination
        self.springaxis = None  # Type: vecline
        self.length = "0"  # Type: string
        self.constant = "50"  # Type: string
        self.damping = "2.0"  # Type: string
        self.relativedamping = "0.1"  # Type: string
        self.breaklength = "0"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.attach1 = entity_data.get('attach1', None)  # Type: target_destination
        instance.attach2 = entity_data.get('attach2', None)  # Type: target_destination
        instance.springaxis = entity_data.get('springaxis', None)  # Type: vecline
        instance.length = entity_data.get('length', "0")  # Type: string
        instance.constant = entity_data.get('constant', "50")  # Type: string
        instance.damping = entity_data.get('damping', "2.0")  # Type: string
        instance.relativedamping = entity_data.get('relativedamping', "0.1")  # Type: string
        instance.breaklength = entity_data.get('breaklength', "0")  # Type: string


class phys_hinge(TwoObjectPhysics):
    def __init__(self):
        super(TwoObjectPhysics).__init__()
        self.origin = [0, 0, 0]
        self.hingefriction = 0  # Type: float
        self.hingeaxis = None  # Type: vecline
        self.SystemLoadScale = 1  # Type: float
        self.minSoundThreshold = 6  # Type: float
        self.maxSoundThreshold = 80  # Type: float
        self.slidesoundfwd = None  # Type: sound
        self.slidesoundback = None  # Type: sound
        self.reversalsoundthresholdSmall = 0  # Type: float
        self.reversalsoundthresholdMedium = 0  # Type: float
        self.reversalsoundthresholdLarge = 0  # Type: float
        self.reversalsoundSmall = None  # Type: sound
        self.reversalsoundMedium = None  # Type: sound
        self.reversalsoundLarge = None  # Type: sound

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TwoObjectPhysics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.hingefriction = float(entity_data.get('hingefriction', 0))  # Type: float
        instance.hingeaxis = entity_data.get('hingeaxis', None)  # Type: vecline
        instance.SystemLoadScale = float(entity_data.get('systemloadscale', 1))  # Type: float
        instance.minSoundThreshold = float(entity_data.get('minsoundthreshold', 6))  # Type: float
        instance.maxSoundThreshold = float(entity_data.get('maxsoundthreshold', 80))  # Type: float
        instance.slidesoundfwd = entity_data.get('slidesoundfwd', None)  # Type: sound
        instance.slidesoundback = entity_data.get('slidesoundback', None)  # Type: sound
        instance.reversalsoundthresholdSmall = float(entity_data.get('reversalsoundthresholdsmall', 0))  # Type: float
        instance.reversalsoundthresholdMedium = float(entity_data.get('reversalsoundthresholdmedium', 0))  # Type: float
        instance.reversalsoundthresholdLarge = float(entity_data.get('reversalsoundthresholdlarge', 0))  # Type: float
        instance.reversalsoundSmall = entity_data.get('reversalsoundsmall', None)  # Type: sound
        instance.reversalsoundMedium = entity_data.get('reversalsoundmedium', None)  # Type: sound
        instance.reversalsoundLarge = entity_data.get('reversalsoundlarge', None)  # Type: sound


class phys_ballsocket(TwoObjectPhysics):
    icon_sprite = "editor/phys_ballsocket.vmt"
    def __init__(self):
        super(TwoObjectPhysics).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TwoObjectPhysics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class phys_constraint(TwoObjectPhysics):
    model = "models/editor/axis_helper.mdl"
    def __init__(self):
        super(TwoObjectPhysics).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TwoObjectPhysics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class phys_pulleyconstraint(TwoObjectPhysics):
    model = "models/editor/axis_helper.mdl"
    def __init__(self):
        super(TwoObjectPhysics).__init__()
        self.origin = [0, 0, 0]
        self.addlength = 0  # Type: float
        self.gearratio = 1  # Type: float
        self.position2 = None  # Type: vecline

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TwoObjectPhysics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.addlength = float(entity_data.get('addlength', 0))  # Type: float
        instance.gearratio = float(entity_data.get('gearratio', 1))  # Type: float
        instance.position2 = entity_data.get('position2', None)  # Type: vecline


class phys_slideconstraint(TwoObjectPhysics):
    model = "models/editor/axis_helper.mdl"
    def __init__(self):
        super(TwoObjectPhysics).__init__()
        self.origin = [0, 0, 0]
        self.slideaxis = None  # Type: vecline
        self.slidefriction = 0  # Type: float
        self.SystemLoadScale = 1  # Type: float
        self.minSoundThreshold = 6  # Type: float
        self.maxSoundThreshold = 80  # Type: float
        self.slidesoundfwd = None  # Type: sound
        self.slidesoundback = None  # Type: sound
        self.reversalsoundthresholdSmall = 0  # Type: float
        self.reversalsoundthresholdMedium = 0  # Type: float
        self.reversalsoundthresholdLarge = 0  # Type: float
        self.reversalsoundSmall = None  # Type: sound
        self.reversalsoundMedium = None  # Type: sound
        self.reversalsoundLarge = None  # Type: sound

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TwoObjectPhysics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.slideaxis = entity_data.get('slideaxis', None)  # Type: vecline
        instance.slidefriction = float(entity_data.get('slidefriction', 0))  # Type: float
        instance.SystemLoadScale = float(entity_data.get('systemloadscale', 1))  # Type: float
        instance.minSoundThreshold = float(entity_data.get('minsoundthreshold', 6))  # Type: float
        instance.maxSoundThreshold = float(entity_data.get('maxsoundthreshold', 80))  # Type: float
        instance.slidesoundfwd = entity_data.get('slidesoundfwd', None)  # Type: sound
        instance.slidesoundback = entity_data.get('slidesoundback', None)  # Type: sound
        instance.reversalsoundthresholdSmall = float(entity_data.get('reversalsoundthresholdsmall', 0))  # Type: float
        instance.reversalsoundthresholdMedium = float(entity_data.get('reversalsoundthresholdmedium', 0))  # Type: float
        instance.reversalsoundthresholdLarge = float(entity_data.get('reversalsoundthresholdlarge', 0))  # Type: float
        instance.reversalsoundSmall = entity_data.get('reversalsoundsmall', None)  # Type: sound
        instance.reversalsoundMedium = entity_data.get('reversalsoundmedium', None)  # Type: sound
        instance.reversalsoundLarge = entity_data.get('reversalsoundlarge', None)  # Type: sound


class phys_lengthconstraint(TwoObjectPhysics):
    model = "models/editor/axis_helper.mdl"
    def __init__(self):
        super(TwoObjectPhysics).__init__()
        self.origin = [0, 0, 0]
        self.addlength = 0  # Type: float
        self.minlength = 0  # Type: float
        self.attachpoint = None  # Set to none due to bug in BlackMesa base.fgd file  # Type: vecline

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TwoObjectPhysics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.addlength = float(entity_data.get('addlength', 0))  # Type: float
        instance.minlength = float(entity_data.get('minlength', 0))  # Type: float
        instance.attachpoint = entity_data.get('attachpoint', None)  # Set to none due to bug in BlackMesa base.fgd file  # Type: vecline


class phys_ragdollconstraint(TwoObjectPhysics):
    model = "models/editor/axis_helper.mdl"
    def __init__(self):
        super(TwoObjectPhysics).__init__()
        self.origin = [0, 0, 0]
        self.xmin = -90  # Type: float
        self.xmax = 90  # Type: float
        self.ymin = 0  # Type: float
        self.ymax = 0  # Type: float
        self.zmin = 0  # Type: float
        self.zmax = 0  # Type: float
        self.xfriction = 0  # Type: float
        self.yfriction = 0  # Type: float
        self.zfriction = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TwoObjectPhysics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.xmin = float(entity_data.get('xmin', -90))  # Type: float
        instance.xmax = float(entity_data.get('xmax', 90))  # Type: float
        instance.ymin = float(entity_data.get('ymin', 0))  # Type: float
        instance.ymax = float(entity_data.get('ymax', 0))  # Type: float
        instance.zmin = float(entity_data.get('zmin', 0))  # Type: float
        instance.zmax = float(entity_data.get('zmax', 0))  # Type: float
        instance.xfriction = float(entity_data.get('xfriction', 0))  # Type: float
        instance.yfriction = float(entity_data.get('yfriction', 0))  # Type: float
        instance.zfriction = float(entity_data.get('zfriction', 0))  # Type: float


class phys_convert(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.swapmodel = None  # Type: string
        self.massoverride = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.swapmodel = entity_data.get('swapmodel', None)  # Type: string
        instance.massoverride = float(entity_data.get('massoverride', 0))  # Type: float


class ForceController(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.attach1 = None  # Type: target_destination
        self.forcetime = "0"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.attach1 = entity_data.get('attach1', None)  # Type: target_destination
        instance.forcetime = entity_data.get('forcetime', "0")  # Type: string


class phys_thruster(Angles, ForceController):
    def __init__(self):
        super(ForceController).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.force = "0"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        ForceController.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.force = entity_data.get('force', "0")  # Type: string


class phys_torque(ForceController):
    def __init__(self):
        super(ForceController).__init__()
        self.origin = [0, 0, 0]
        self.force = "0"  # Type: string
        self.axis = None  # Type: vecline

    @staticmethod
    def from_dict(instance, entity_data: dict):
        ForceController.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.force = entity_data.get('force', "0")  # Type: string
        instance.axis = entity_data.get('axis', None)  # Type: vecline


class phys_motor(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.speed = "0"  # Type: string
        self.spinup = "1"  # Type: string
        self.inertiafactor = 1.0  # Type: float
        self.axis = None  # Type: vecline
        self.attach1 = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.speed = entity_data.get('speed', "0")  # Type: string
        instance.spinup = entity_data.get('spinup', "1")  # Type: string
        instance.inertiafactor = float(entity_data.get('inertiafactor', 1.0))  # Type: float
        instance.axis = entity_data.get('axis', None)  # Type: vecline
        instance.attach1 = entity_data.get('attach1', None)  # Type: target_destination


class phys_magnet(Targetname, Angles, Parentname, Studiomodel):
    def __init__(self):
        super(Studiomodel).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.forcelimit = 0  # Type: float
        self.torquelimit = 0  # Type: float
        self.massScale = 0  # Type: float
        self.overridescript = None  # Type: string
        self.maxobjects = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.forcelimit = float(entity_data.get('forcelimit', 0))  # Type: float
        instance.torquelimit = float(entity_data.get('torquelimit', 0))  # Type: float
        instance.massScale = float(entity_data.get('massscale', 0))  # Type: float
        instance.overridescript = entity_data.get('overridescript', None)  # Type: string
        instance.maxobjects = parse_source_value(entity_data.get('maxobjects', 0))  # Type: integer


class prop_detail_base(Base):
    def __init__(self):
        super().__init__()
        self.model = None  # Type: studio

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', None)  # Type: studio


class prop_static_base(Angles, Reflection, SystemLevelChoice, Shadow):
    def __init__(self):
        super(Angles).__init__()
        super(Reflection).__init__()
        super(SystemLevelChoice).__init__()
        super(Shadow).__init__()
        self.model = None  # Type: studio
        self.skin = None  # Type: integer
        self.solid = "CHOICES NOT SUPPORTED"  # Type: choices
        self.screenspacefade = None  # Type: boolean
        self.fademindist = -1  # Type: float
        self.fademaxdist = None  # Type: float
        self.fadescale = 1  # Type: float
        self.lightingorigin = None  # Type: target_destination
        self.disablevertexlighting = None  # Type: boolean
        self.disableselfshadowing = None  # Type: boolean
        self.ignorenormals = None  # Type: boolean
        self.renderamt = 255  # Type: integer
        self.rendercolor = [255, 255, 255]  # Type: color255

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Reflection.from_dict(instance, entity_data)
        SystemLevelChoice.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', None)  # Type: studio
        instance.skin = parse_source_value(entity_data.get('skin', 0))  # Type: integer
        instance.solid = entity_data.get('solid', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.screenspacefade = entity_data.get('screenspacefade', None)  # Type: boolean
        instance.fademindist = float(entity_data.get('fademindist', -1))  # Type: float
        instance.fademaxdist = float(entity_data.get('fademaxdist', 0))  # Type: float
        instance.fadescale = float(entity_data.get('fadescale', 1))  # Type: float
        instance.lightingorigin = entity_data.get('lightingorigin', None)  # Type: target_destination
        instance.disablevertexlighting = entity_data.get('disablevertexlighting', None)  # Type: boolean
        instance.disableselfshadowing = entity_data.get('disableselfshadowing', None)  # Type: boolean
        instance.ignorenormals = entity_data.get('ignorenormals', None)  # Type: boolean
        instance.renderamt = parse_source_value(entity_data.get('renderamt', 255))  # Type: integer
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "255 255 255"))  # Type: color255


class BaseFadeProp(Base):
    def __init__(self):
        super().__init__()
        self.fademindist = -1  # Type: float
        self.fademaxdist = None  # Type: float
        self.fadescale = 1  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.fademindist = float(entity_data.get('fademindist', -1))  # Type: float
        instance.fademaxdist = float(entity_data.get('fademaxdist', 0))  # Type: float
        instance.fadescale = float(entity_data.get('fadescale', 1))  # Type: float


class prop_dynamic_base(Parentname, SystemLevelChoice, Studiomodel, BreakableProp, RenderFields, Global, Angles, BaseFadeProp):
    def __init__(self):
        super(Studiomodel).__init__()
        super(BreakableProp).__init__()
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(SystemLevelChoice).__init__()
        super(Shadow).__init__()
        super(Global).__init__()
        super(Angles).__init__()
        super(BaseFadeProp).__init__()
        self.solid = "CHOICES NOT SUPPORTED"  # Type: choices
        self.DefaultAnim = None  # Type: string
        self.RandomAnimation = None  # Type: boolean
        self.DisableBoneFollowers = None  # Type: boolean
        self.SuppressAnimSounds = None  # Type: boolean
        self.HoldAnimation = None  # Type: boolean
        self.MinAnimTime = 5  # Type: float
        self.MaxAnimTime = 10  # Type: float
        self.SetBodyGroup = None  # Type: integer
        self.lightingorigin = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        SystemLevelChoice.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        BreakableProp.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        instance.solid = entity_data.get('solid', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.DefaultAnim = entity_data.get('defaultanim', None)  # Type: string
        instance.RandomAnimation = entity_data.get('randomanimation', None)  # Type: boolean
        instance.DisableBoneFollowers = entity_data.get('disablebonefollowers', None)  # Type: boolean
        instance.SuppressAnimSounds = entity_data.get('suppressanimsounds', None)  # Type: boolean
        instance.HoldAnimation = entity_data.get('holdanimation', None)  # Type: boolean
        instance.MinAnimTime = float(entity_data.get('minanimtime', 5))  # Type: float
        instance.MaxAnimTime = float(entity_data.get('maxanimtime', 10))  # Type: float
        instance.SetBodyGroup = parse_source_value(entity_data.get('setbodygroup', 0))  # Type: integer
        instance.lightingorigin = entity_data.get('lightingorigin', None)  # Type: target_destination


class prop_detail(prop_detail_base):
    def __init__(self):
        super(prop_detail_base).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_detail_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class prop_static(prop_static_base):
    def __init__(self):
        super(prop_static_base).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_static_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class prop_dynamic(prop_dynamic_base, EnableDisable):
    def __init__(self):
        super(prop_dynamic_base).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_dynamic_base.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class prop_dynamic_override(prop_dynamic_base):
    def __init__(self):
        super(prop_dynamic_base).__init__()
        self.origin = [0, 0, 0]
        self.health = None  # Type: integer
        self.AnimateEveryFrame = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.health = parse_source_value(entity_data.get('health', 0))  # Type: integer
        instance.AnimateEveryFrame = entity_data.get('animateeveryframe', None)  # Type: boolean


class BasePropPhysics(SystemLevelChoice, Studiomodel, BreakableProp, Global, Angles, BaseFadeProp):
    def __init__(self):
        super(Studiomodel).__init__()
        super(BreakableProp).__init__()
        super(SystemLevelChoice).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
        super(Angles).__init__()
        super(BaseFadeProp).__init__()
        self.minhealthdmg = None  # Type: integer
        self.shadowcastdist = None  # Type: integer
        self.physdamagescale = 0.1  # Type: float
        self.Damagetype = None  # Type: choices
        self.nodamageforces = None  # Type: choices
        self.inertiaScale = 1.0  # Type: float
        self.massScale = 0  # Type: float
        self.overridescript = None  # Type: string
        self.damagetoenablemotion = None  # Type: integer
        self.forcetoenablemotion = None  # Type: float
        self.puntsound = None  # Type: sound
        self.addon = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        SystemLevelChoice.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        BreakableProp.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        instance.minhealthdmg = parse_source_value(entity_data.get('minhealthdmg', 0))  # Type: integer
        instance.shadowcastdist = parse_source_value(entity_data.get('shadowcastdist', 0))  # Type: integer
        instance.physdamagescale = float(entity_data.get('physdamagescale', 0.1))  # Type: float
        instance.Damagetype = entity_data.get('damagetype', None)  # Type: choices
        instance.nodamageforces = entity_data.get('nodamageforces', None)  # Type: choices
        instance.inertiaScale = float(entity_data.get('inertiascale', 1.0))  # Type: float
        instance.massScale = float(entity_data.get('massscale', 0))  # Type: float
        instance.overridescript = entity_data.get('overridescript', None)  # Type: string
        instance.damagetoenablemotion = parse_source_value(entity_data.get('damagetoenablemotion', 0))  # Type: integer
        instance.forcetoenablemotion = float(entity_data.get('forcetoenablemotion', 0))  # Type: float
        instance.puntsound = entity_data.get('puntsound', None)  # Type: sound
        instance.addon = entity_data.get('addon', None)  # Type: string


class prop_physics_override(BasePropPhysics):
    def __init__(self):
        super(BasePropPhysics).__init__()
        self.origin = [0, 0, 0]
        self.health = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePropPhysics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.health = parse_source_value(entity_data.get('health', 0))  # Type: integer


class prop_physics(BasePropPhysics, RenderFields):
    def __init__(self):
        super(BasePropPhysics).__init__()
        super(RenderFields).__init__()
        self.origin = [0, 0, 0]
        self.ExploitableByPlayer = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePropPhysics.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.ExploitableByPlayer = entity_data.get('exploitablebyplayer', None)  # Type: choices


class prop_physics_multiplayer(prop_physics):
    def __init__(self):
        super(prop_physics).__init__()
        self.origin = [0, 0, 0]
        self.physicsmode = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_physics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.physicsmode = entity_data.get('physicsmode', None)  # Type: choices


class prop_ragdoll(SystemLevelChoice, Studiomodel, EnableDisable, Targetname, Angles, BaseFadeProp):
    def __init__(self):
        super(Studiomodel).__init__()
        super(SystemLevelChoice).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        super(BaseFadeProp).__init__()
        self.origin = [0, 0, 0]
        self.angleOverride = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        SystemLevelChoice.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.angleOverride = entity_data.get('angleoverride', None)  # Type: string


class prop_dynamic_ornament(prop_dynamic_base):
    def __init__(self):
        super(prop_dynamic_base).__init__()
        self.origin = [0, 0, 0]
        self.InitialOwner = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.InitialOwner = entity_data.get('initialowner', None)  # Type: string


class func_areaportal(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.target = None  # Type: target_destination
        self.StartOpen = "CHOICES NOT SUPPORTED"  # Type: choices
        self.PortalVersion = 1  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.StartOpen = entity_data.get('startopen', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.PortalVersion = parse_source_value(entity_data.get('portalversion', 1))  # Type: integer


class func_occluder(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.StartActive = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.StartActive = entity_data.get('startactive', "CHOICES NOT SUPPORTED")  # Type: choices


class func_breakable(BreakableBrush, RenderFields, Origin):
    def __init__(self):
        super(BreakableBrush).__init__()
        super(RenderFields).__init__()
        super(Shadow).__init__()
        super(Reflection).__init__()
        super(Origin).__init__()
        self.minhealthdmg = None  # Type: integer
        self._minlight = None  # Type: string
        self.physdamagescale = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Shadow.from_dict(instance, entity_data)
        BreakableBrush.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Reflection.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.minhealthdmg = parse_source_value(entity_data.get('minhealthdmg', 0))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.physdamagescale = float(entity_data.get('physdamagescale', 1.0))  # Type: float


class func_breakable_surf(BreakableBrush, RenderFields):
    def __init__(self):
        super(BreakableBrush).__init__()
        super(RenderFields).__init__()
        super(Reflection).__init__()
        super(Shadow).__init__()
        self.fragility = 100  # Type: integer
        self.surfacetype = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BreakableBrush.from_dict(instance, entity_data)
        Reflection.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance.fragility = parse_source_value(entity_data.get('fragility', 100))  # Type: integer
        instance.surfacetype = entity_data.get('surfacetype', None)  # Type: choices


class func_conveyor(Targetname, Parentname, Shadow, RenderFields):
    def __init__(self):
        super(RenderFields).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(Shadow).__init__()
        self.movedir = [0.0, 0.0, 0.0]  # Type: angle
        self.speed = "100"  # Type: string
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance.movedir = parse_float_vector(entity_data.get('movedir', "0 0 0"))  # Type: angle
        instance.speed = entity_data.get('speed', "100")  # Type: string
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class func_detail(SystemLevelChoice):
    def __init__(self):
        super(SystemLevelChoice).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        SystemLevelChoice.from_dict(instance, entity_data)


class func_viscluster(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class func_illusionary(Parentname, Shadow, Targetname, RenderFields, Origin):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        super(Origin).__init__()
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class func_precipitation(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.renderamt = 100  # Type: integer
        self.rendercolor = [100, 100, 100]  # Type: color255
        self.preciptype = None  # Type: choices
        self.minSpeed = 25  # Type: float
        self.maxSpeed = 35  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.renderamt = parse_source_value(entity_data.get('renderamt', 100))  # Type: integer
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "100 100 100"))  # Type: color255
        instance.preciptype = entity_data.get('preciptype', None)  # Type: choices
        instance.minSpeed = float(entity_data.get('minspeed', 25))  # Type: float
        instance.maxSpeed = float(entity_data.get('maxspeed', 35))  # Type: float


class func_wall_toggle(func_wall):
    def __init__(self):
        super(func_wall).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        func_wall.from_dict(instance, entity_data)


class func_guntarget(Targetname, Parentname, Global, RenderFields):
    def __init__(self):
        super(RenderFields).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(Global).__init__()
        self.speed = 100  # Type: integer
        self.target = None  # Type: target_destination
        self.health = None  # Type: integer
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance.speed = parse_source_value(entity_data.get('speed', 100))  # Type: integer
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.health = parse_source_value(entity_data.get('health', 0))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class PlatSounds(Base):
    def __init__(self):
        super().__init__()
        self.movesnd = None  # Type: choices
        self.stopsnd = None  # Type: choices
        self.volume = "0.85"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.movesnd = entity_data.get('movesnd', None)  # Type: choices
        instance.stopsnd = entity_data.get('stopsnd', None)  # Type: choices
        instance.volume = entity_data.get('volume', "0.85")  # Type: string


class Trackchange(Parentname, PlatSounds, Targetname, RenderFields, Global):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(PlatSounds).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
        self.height = None  # Type: integer
        self.rotation = None  # Type: integer
        self.train = None  # Type: target_destination
        self.toptrack = None  # Type: target_destination
        self.bottomtrack = None  # Type: target_destination
        self.speed = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        PlatSounds.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        instance.height = parse_source_value(entity_data.get('height', 0))  # Type: integer
        instance.rotation = parse_source_value(entity_data.get('rotation', 0))  # Type: integer
        instance.train = entity_data.get('train', None)  # Type: target_destination
        instance.toptrack = entity_data.get('toptrack', None)  # Type: target_destination
        instance.bottomtrack = entity_data.get('bottomtrack', None)  # Type: target_destination
        instance.speed = parse_source_value(entity_data.get('speed', 0))  # Type: integer


class BaseTrain(Parentname, Shadow, Targetname, RenderFields, Global, Reflection, Origin):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
        super(Reflection).__init__()
        super(Origin).__init__()
        self.target = None  # Type: target_destination
        self.startspeed = 100  # Type: integer
        self.speed = None  # Type: integer
        self.velocitytype = None  # Type: choices
        self.orientationtype = "CHOICES NOT SUPPORTED"  # Type: choices
        self.wheels = 50  # Type: integer
        self.height = 4  # Type: integer
        self.bank = "0"  # Type: string
        self.dmg = None  # Type: integer
        self._minlight = None  # Type: string
        self.MoveSound = None  # Type: sound
        self.MovePingSound = None  # Type: sound
        self.StartSound = None  # Type: sound
        self.StopSound = None  # Type: sound
        self.volume = 10  # Type: integer
        self.MoveSoundMinPitch = 60  # Type: integer
        self.MoveSoundMaxPitch = 200  # Type: integer
        self.MoveSoundMinTime = None  # Type: float
        self.MoveSoundMaxTime = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Reflection.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.startspeed = parse_source_value(entity_data.get('startspeed', 100))  # Type: integer
        instance.speed = parse_source_value(entity_data.get('speed', 0))  # Type: integer
        instance.velocitytype = entity_data.get('velocitytype', None)  # Type: choices
        instance.orientationtype = entity_data.get('orientationtype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.wheels = parse_source_value(entity_data.get('wheels', 50))  # Type: integer
        instance.height = parse_source_value(entity_data.get('height', 4))  # Type: integer
        instance.bank = entity_data.get('bank', "0")  # Type: string
        instance.dmg = parse_source_value(entity_data.get('dmg', 0))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.MoveSound = entity_data.get('movesound', None)  # Type: sound
        instance.MovePingSound = entity_data.get('movepingsound', None)  # Type: sound
        instance.StartSound = entity_data.get('startsound', None)  # Type: sound
        instance.StopSound = entity_data.get('stopsound', None)  # Type: sound
        instance.volume = parse_source_value(entity_data.get('volume', 10))  # Type: integer
        instance.MoveSoundMinPitch = parse_source_value(entity_data.get('movesoundminpitch', 60))  # Type: integer
        instance.MoveSoundMaxPitch = parse_source_value(entity_data.get('movesoundmaxpitch', 200))  # Type: integer
        instance.MoveSoundMinTime = float(entity_data.get('movesoundmintime', 0))  # Type: float
        instance.MoveSoundMaxTime = float(entity_data.get('movesoundmaxtime', 0))  # Type: float


class func_trackautochange(Trackchange):
    def __init__(self):
        super(Trackchange).__init__()
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trackchange.from_dict(instance, entity_data)
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class func_trackchange(Trackchange):
    def __init__(self):
        super(Trackchange).__init__()
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trackchange.from_dict(instance, entity_data)
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class func_tracktrain(PaintableBrush, BaseTrain):
    def __init__(self):
        super(BaseTrain).__init__()
        super(PaintableBrush).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        PaintableBrush.from_dict(instance, entity_data)
        BaseTrain.from_dict(instance, entity_data)


class func_tanktrain(BaseTrain):
    def __init__(self):
        super(BaseTrain).__init__()
        self.health = 100  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseTrain.from_dict(instance, entity_data)
        instance.health = parse_source_value(entity_data.get('health', 100))  # Type: integer


class func_traincontrols(Parentname, Global):
    def __init__(self):
        super(Parentname).__init__()
        super(Global).__init__()
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        instance.target = entity_data.get('target', None)  # Type: target_destination


class tanktrain_aitarget(Targetname):
    icon_sprite = "editor/tanktrain_aitarget.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.newtarget = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.newtarget = entity_data.get('newtarget', None)  # Type: target_destination


class tanktrain_ai(Targetname):
    icon_sprite = "editor/tanktrain_ai.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.startsound = "vehicles/diesel_start1.wav"  # Type: sound
        self.enginesound = "vehicles/diesel_turbo_loop1.wav"  # Type: sound
        self.movementsound = "vehicles/tank_treads_loop1.wav"  # Type: sound

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.startsound = entity_data.get('startsound', "vehicles/diesel_start1.wav")  # Type: sound
        instance.enginesound = entity_data.get('enginesound', "vehicles/diesel_turbo_loop1.wav")  # Type: sound
        instance.movementsound = entity_data.get('movementsound', "vehicles/tank_treads_loop1.wav")  # Type: sound


class path_track(Targetname, Angles, Parentname):
    model = "models/editor/angle_helper.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.altpath = None  # Type: target_destination
        self.speed = None  # Type: float
        self.radius = None  # Type: float
        self.orientationtype = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.altpath = entity_data.get('altpath', None)  # Type: target_destination
        instance.speed = float(entity_data.get('speed', 0))  # Type: float
        instance.radius = float(entity_data.get('radius', 0))  # Type: float
        instance.orientationtype = entity_data.get('orientationtype', "CHOICES NOT SUPPORTED")  # Type: choices


class test_traceline(Angles):
    def __init__(self):
        super(Angles).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class trigger_autosave(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.master = None  # Type: string
        self.NewLevelUnit = None  # Type: boolean
        self.DangerousTimer = None  # Type: float
        self.MinimumHitPoints = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.master = entity_data.get('master', None)  # Type: string
        instance.NewLevelUnit = entity_data.get('newlevelunit', None)  # Type: boolean
        instance.DangerousTimer = float(entity_data.get('dangeroustimer', 0))  # Type: float
        instance.MinimumHitPoints = parse_source_value(entity_data.get('minimumhitpoints', 0))  # Type: integer


class trigger_changelevel(EnableDisable):
    def __init__(self):
        super(EnableDisable).__init__()
        self.targetname = None  # Type: target_source
        self.map = None  # Type: string
        self.landmark = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        instance.targetname = entity_data.get('targetname', None)  # Type: target_source
        instance.map = entity_data.get('map', None)  # Type: string
        instance.landmark = entity_data.get('landmark', None)  # Type: target_destination


class trigger_gravity(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        self.gravity = 1  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.gravity = parse_source_value(entity_data.get('gravity', 1))  # Type: integer


class trigger_playermovement(Trigger):
    def __init__(self):
        super(Trigger).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)


class trigger_soundscape(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        self.soundscape = None  # Type: target_source

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.soundscape = entity_data.get('soundscape', None)  # Type: target_source


class trigger_hurt(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        super(Targetname).__init__()
        self.master = None  # Type: string
        self.damage = 10  # Type: integer
        self.damagecap = 20  # Type: integer
        self.damagetype = None  # Type: choices
        self.damagemodel = None  # Type: choices
        self.nodmgforce = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.master = entity_data.get('master', None)  # Type: string
        instance.damage = parse_source_value(entity_data.get('damage', 10))  # Type: integer
        instance.damagecap = parse_source_value(entity_data.get('damagecap', 20))  # Type: integer
        instance.damagetype = entity_data.get('damagetype', None)  # Type: choices
        instance.damagemodel = entity_data.get('damagemodel', None)  # Type: choices
        instance.nodmgforce = entity_data.get('nodmgforce', None)  # Type: boolean


class trigger_remove(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)


class trigger_multiple(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        self.wait = 1  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.wait = parse_source_value(entity_data.get('wait', 1))  # Type: integer


class trigger_once(TriggerOnce):
    def __init__(self):
        super(TriggerOnce).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TriggerOnce.from_dict(instance, entity_data)


class trigger_look(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        self.target = None  # Type: target_destination
        self.LookTime = "0.5"  # Type: string
        self.FieldOfView = "0.9"  # Type: string
        self.Timeout = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.LookTime = entity_data.get('looktime', "0.5")  # Type: string
        instance.FieldOfView = entity_data.get('fieldofview', "0.9")  # Type: string
        instance.Timeout = float(entity_data.get('timeout', 0))  # Type: float


class trigger_push(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        self.pushdir = [0.0, 0.0, 0.0]  # Type: angle
        self.speed = 40  # Type: integer
        self.alternateticksfix = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.pushdir = parse_float_vector(entity_data.get('pushdir', "0 0 0"))  # Type: angle
        instance.speed = parse_source_value(entity_data.get('speed', 40))  # Type: integer
        instance.alternateticksfix = float(entity_data.get('alternateticksfix', 0))  # Type: float


class trigger_wind(Trigger, Angles):
    def __init__(self):
        super(Trigger).__init__()
        super(Angles).__init__()
        self.Speed = 200  # Type: integer
        self.SpeedNoise = None  # Type: integer
        self.DirectionNoise = 10  # Type: integer
        self.HoldTime = None  # Type: integer
        self.HoldNoise = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.Speed = parse_source_value(entity_data.get('speed', 200))  # Type: integer
        instance.SpeedNoise = parse_source_value(entity_data.get('speednoise', 0))  # Type: integer
        instance.DirectionNoise = parse_source_value(entity_data.get('directionnoise', 10))  # Type: integer
        instance.HoldTime = parse_source_value(entity_data.get('holdtime', 0))  # Type: integer
        instance.HoldNoise = parse_source_value(entity_data.get('holdnoise', 0))  # Type: integer


class trigger_hierarchy(trigger_multiple):
    def __init__(self):
        super(trigger_multiple).__init__()
        self.childfiltername = None  # Type: filterclass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        trigger_multiple.from_dict(instance, entity_data)
        instance.childfiltername = entity_data.get('childfiltername', None)  # Type: filterclass


class trigger_impact(Targetname, Angles, Origin):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Origin).__init__()
        self.Magnitude = 200  # Type: float
        self.noise = 0.1  # Type: float
        self.viewkick = 0.05  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.Magnitude = float(entity_data.get('magnitude', 200))  # Type: float
        instance.noise = float(entity_data.get('noise', 0.1))  # Type: float
        instance.viewkick = float(entity_data.get('viewkick', 0.05))  # Type: float


class trigger_proximity(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        self.measuretarget = None  # Type: target_destination
        self.radius = "256"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.measuretarget = entity_data.get('measuretarget', None)  # Type: target_destination
        instance.radius = entity_data.get('radius', "256")  # Type: string


class trigger_teleport(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        self.target = None  # Type: target_destination
        self.landmark = None  # Type: target_destination
        self.UseLandmarkAngles = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.landmark = entity_data.get('landmark', None)  # Type: target_destination
        instance.UseLandmarkAngles = entity_data.get('uselandmarkangles', None)  # Type: boolean


class trigger_transition(Targetname):
    def __init__(self):
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)


class trigger_serverragdoll(Targetname):
    def __init__(self):
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)


class ai_speechfilter(Targetname, ResponseContext, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(ResponseContext).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.subject = None  # Type: target_destination
        self.IdleModifier = 1.0  # Type: float
        self.NeverSayHello = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        ResponseContext.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.subject = entity_data.get('subject', None)  # Type: target_destination
        instance.IdleModifier = float(entity_data.get('idlemodifier', 1.0))  # Type: float
        instance.NeverSayHello = entity_data.get('neversayhello', None)  # Type: choices


class ai_addon(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.target = entity_data.get('target', None)  # Type: target_destination


class ai_addon_builder(Targetname, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.NPCName = None  # Type: string
        self.AddOnName = None  # Type: string
        self.NpcPoints = 10  # Type: integer
        self.AddonPoints = 10  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.NPCName = entity_data.get('npcname', None)  # Type: string
        instance.AddOnName = entity_data.get('addonname', None)  # Type: string
        instance.NpcPoints = parse_source_value(entity_data.get('npcpoints', 10))  # Type: integer
        instance.AddonPoints = parse_source_value(entity_data.get('addonpoints', 10))  # Type: integer


class water_lod_control(Targetname):
    icon_sprite = "editor/waterlodcontrol.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.cheapwaterstartdistance = 1000  # Type: float
        self.cheapwaterenddistance = 2000  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.cheapwaterstartdistance = float(entity_data.get('cheapwaterstartdistance', 1000))  # Type: float
        instance.cheapwaterenddistance = float(entity_data.get('cheapwaterenddistance', 2000))  # Type: float


class info_camera_link(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.PointCamera = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.PointCamera = entity_data.get('pointcamera', None)  # Type: target_destination


class logic_eventlistener(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.EventName = None  # Type: string
        self.IsEnabled = "CHOICES NOT SUPPORTED"  # Type: choices
        self.TeamNum = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.EventName = entity_data.get('eventname', None)  # Type: string
        instance.IsEnabled = entity_data.get('isenabled', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.TeamNum = entity_data.get('teamnum', "CHOICES NOT SUPPORTED")  # Type: choices


class logic_measure_movement(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.MeasureTarget = None  # Type: target_destination
        self.MeasureReference = None  # Type: target_destination
        self.Target = None  # Type: target_destination
        self.TargetReference = None  # Type: target_destination
        self.TargetScale = 1  # Type: float
        self.MeasureType = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.MeasureTarget = entity_data.get('measuretarget', None)  # Type: target_destination
        instance.MeasureReference = entity_data.get('measurereference', None)  # Type: target_destination
        instance.Target = entity_data.get('target', None)  # Type: target_destination
        instance.TargetReference = entity_data.get('targetreference', None)  # Type: target_destination
        instance.TargetScale = float(entity_data.get('targetscale', 1))  # Type: float
        instance.MeasureType = entity_data.get('measuretype', None)  # Type: choices


class npc_furniture(Parentname, BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.model = None  # Type: studio

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        BaseNPC.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', None)  # Type: studio


class env_credits(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class material_modify_control(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.materialName = None  # Type: string
        self.materialVar = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.materialName = entity_data.get('materialname', None)  # Type: string
        instance.materialVar = entity_data.get('materialvar', None)  # Type: string


class point_devshot_camera(Angles):
    viewport_model = "models/editor/camera.mdl"
    def __init__(self):
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.cameraname = None  # Type: string
        self.FOV = 75  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.cameraname = entity_data.get('cameraname', None)  # Type: string
        instance.FOV = parse_source_value(entity_data.get('fov', 75))  # Type: integer


class logic_playerproxy(Targetname, DamageFilter):
    def __init__(self):
        super(Targetname).__init__()
        super(DamageFilter).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        DamageFilter.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_spritetrail(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.lifetime = 0.5  # Type: float
        self.startwidth = 8.0  # Type: float
        self.endwidth = 1.0  # Type: float
        self.spritename = "sprites/bluelaser1.vmt"  # Type: string
        self.renderamt = 255  # Type: integer
        self.rendercolor = [255, 255, 255]  # Type: color255
        self.rendermode = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.lifetime = float(entity_data.get('lifetime', 0.5))  # Type: float
        instance.startwidth = float(entity_data.get('startwidth', 8.0))  # Type: float
        instance.endwidth = float(entity_data.get('endwidth', 1.0))  # Type: float
        instance.spritename = entity_data.get('spritename', "sprites/bluelaser1.vmt")  # Type: string
        instance.renderamt = parse_source_value(entity_data.get('renderamt', 255))  # Type: integer
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "255 255 255"))  # Type: color255
        instance.rendermode = entity_data.get('rendermode', "CHOICES NOT SUPPORTED")  # Type: choices


class env_projectedtexture(Targetname, Angles, Parentname, SystemLevelChoice):
    model = "models/editor/cone_helper.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        super(SystemLevelChoice).__init__()
        self.origin = [0, 0, 0]
        self.style = None  # Type: choices
        self.pattern = None  # Type: string
        self.target = None  # Type: target_destination
        self.lightfov = 90.0  # Type: float
        self.nearz = 4.0  # Type: float
        self.farz = 750.0  # Type: float
        self.enableshadows = None  # Type: boolean
        self.shadowquality = "CHOICES NOT SUPPORTED"  # Type: choices
        self.lightonlytarget = None  # Type: boolean
        self.lightworld = 1  # Type: boolean
        self.simpleprojection = None  # Type: boolean
        self.brightnessscale = 1.0  # Type: float
        self.lightcolor = [255, 255, 255, 200]  # Type: color255
        self.colortransitiontime = 0.5  # Type: float
        self.cameraspace = None  # Type: integer
        self.texturename = "effects/flashlight001"  # Type: material

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        SystemLevelChoice.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.style = entity_data.get('style', None)  # Type: choices
        instance.pattern = entity_data.get('pattern', None)  # Type: string
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.lightfov = float(entity_data.get('lightfov', 90.0))  # Type: float
        instance.nearz = float(entity_data.get('nearz', 4.0))  # Type: float
        instance.farz = float(entity_data.get('farz', 750.0))  # Type: float
        instance.enableshadows = entity_data.get('enableshadows', None)  # Type: boolean
        instance.shadowquality = entity_data.get('shadowquality', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.lightonlytarget = entity_data.get('lightonlytarget', None)  # Type: boolean
        instance.lightworld = entity_data.get('lightworld', None)  # Type: boolean
        instance.simpleprojection = entity_data.get('simpleprojection', None)  # Type: boolean
        instance.brightnessscale = float(entity_data.get('brightnessscale', 1.0))  # Type: float
        instance.lightcolor = parse_int_vector(entity_data.get('lightcolor', "255 255 255 200"))  # Type: color255
        instance.colortransitiontime = float(entity_data.get('colortransitiontime', 0.5))  # Type: float
        instance.cameraspace = parse_source_value(entity_data.get('cameraspace', 0))  # Type: integer
        instance.texturename = entity_data.get('texturename', "effects/flashlight001")  # Type: material


class func_reflective_glass(func_brush):
    def __init__(self):
        super(func_brush).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        func_brush.from_dict(instance, entity_data)


class env_particle_performance_monitor(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class npc_puppet(Studiomodel, Parentname, BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        super(Studiomodel).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.animationtarget = None  # Type: target_source
        self.attachmentname = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        BaseNPC.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.animationtarget = entity_data.get('animationtarget', None)  # Type: target_source
        instance.attachmentname = entity_data.get('attachmentname', None)  # Type: string


class point_gamestats_counter(Targetname, Origin, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(Origin).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.Name = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Name = entity_data.get('name', None)  # Type: string


class beam_spotlight(Parentname, SystemLevelChoice, Targetname, RenderFields, Angles):
    model = "models/editor/cone_helper.mdl"
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(SystemLevelChoice).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.maxspeed = 100  # Type: integer
        self.spotlightlength = 500  # Type: integer
        self.spotlightwidth = 50  # Type: integer
        self.HDRColorScale = 0.7  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        SystemLevelChoice.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.maxspeed = parse_source_value(entity_data.get('maxspeed', 100))  # Type: integer
        instance.spotlightlength = parse_source_value(entity_data.get('spotlightlength', 500))  # Type: integer
        instance.spotlightwidth = parse_source_value(entity_data.get('spotlightwidth', 50))  # Type: integer
        instance.HDRColorScale = float(entity_data.get('hdrcolorscale', 0.7))  # Type: float


class func_instance(Angles):
    def __init__(self):
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.targetname = None  # Type: target_source
        self.file = None  # Type: instance_file
        self.fixup_style = None  # Type: choices
        self.replace01 = None  # Type: instance_variable
        self.replace02 = None  # Type: instance_variable
        self.replace03 = None  # Type: instance_variable
        self.replace04 = None  # Type: instance_variable
        self.replace05 = None  # Type: instance_variable
        self.replace06 = None  # Type: instance_variable
        self.replace07 = None  # Type: instance_variable
        self.replace08 = None  # Type: instance_variable
        self.replace09 = None  # Type: instance_variable
        self.replace10 = None  # Type: instance_variable

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.targetname = entity_data.get('targetname', None)  # Type: target_source
        instance.file = entity_data.get('file', None)  # Type: instance_file
        instance.fixup_style = entity_data.get('fixup_style', None)  # Type: choices
        instance.replace01 = entity_data.get('replace01', None)  # Type: instance_variable
        instance.replace02 = entity_data.get('replace02', None)  # Type: instance_variable
        instance.replace03 = entity_data.get('replace03', None)  # Type: instance_variable
        instance.replace04 = entity_data.get('replace04', None)  # Type: instance_variable
        instance.replace05 = entity_data.get('replace05', None)  # Type: instance_variable
        instance.replace06 = entity_data.get('replace06', None)  # Type: instance_variable
        instance.replace07 = entity_data.get('replace07', None)  # Type: instance_variable
        instance.replace08 = entity_data.get('replace08', None)  # Type: instance_variable
        instance.replace09 = entity_data.get('replace09', None)  # Type: instance_variable
        instance.replace10 = entity_data.get('replace10', None)  # Type: instance_variable


class func_instance_parms(Base):
    icon_sprite = "editor/func_instance_parms.vmt"
    def __init__(self):
        super().__init__()
        self.origin = [0, 0, 0]
        self.parm1 = None  # Type: instance_parm
        self.parm2 = None  # Type: instance_parm
        self.parm3 = None  # Type: instance_parm
        self.parm4 = None  # Type: instance_parm
        self.parm5 = None  # Type: instance_parm
        self.parm6 = None  # Type: instance_parm
        self.parm7 = None  # Type: instance_parm
        self.parm8 = None  # Type: instance_parm
        self.parm9 = None  # Type: instance_parm
        self.parm10 = None  # Type: instance_parm

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.parm1 = entity_data.get('parm1', None)  # Type: instance_parm
        instance.parm2 = entity_data.get('parm2', None)  # Type: instance_parm
        instance.parm3 = entity_data.get('parm3', None)  # Type: instance_parm
        instance.parm4 = entity_data.get('parm4', None)  # Type: instance_parm
        instance.parm5 = entity_data.get('parm5', None)  # Type: instance_parm
        instance.parm6 = entity_data.get('parm6', None)  # Type: instance_parm
        instance.parm7 = entity_data.get('parm7', None)  # Type: instance_parm
        instance.parm8 = entity_data.get('parm8', None)  # Type: instance_parm
        instance.parm9 = entity_data.get('parm9', None)  # Type: instance_parm
        instance.parm10 = entity_data.get('parm10', None)  # Type: instance_parm


class func_instance_io_proxy(Base):
    icon_sprite = "editor/func_instance_parms.vmt"
    def __init__(self):
        super().__init__()
        self.origin = [0, 0, 0]
        self.targetname = None  # Type: target_source

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.targetname = entity_data.get('targetname', None)  # Type: target_source


class func_instance_origin(Base):
    def __init__(self):
        super().__init__()
        self.origin = [0, 0, 0]
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_instructor_hint(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.hint_replace_key = None  # Type: string
        self.hint_target = None  # Type: target_destination
        self.hint_static = None  # Type: choices
        self.hint_allow_nodraw_target = "CHOICES NOT SUPPORTED"  # Type: choices
        self.hint_caption = None  # Type: string
        self.hint_activator_caption = None  # Type: string
        self.hint_color = [255, 255, 255]  # Type: color255
        self.hint_forcecaption = None  # Type: choices
        self.hint_icon_onscreen = "CHOICES NOT SUPPORTED"  # Type: choices
        self.hint_icon_offscreen = "CHOICES NOT SUPPORTED"  # Type: choices
        self.hint_nooffscreen = None  # Type: choices
        self.hint_binding = None  # Type: string
        self.hint_gamepad_binding = None  # Type: string
        self.hint_icon_offset = None  # Type: float
        self.hint_pulseoption = None  # Type: choices
        self.hint_alphaoption = None  # Type: choices
        self.hint_shakeoption = None  # Type: choices
        self.hint_local_player_only = False # Type: boolean
        self.hint_timeout = None  # Type: integer
        self.hint_range = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.hint_replace_key = entity_data.get('hint_replace_key', None)  # Type: string
        instance.hint_target = entity_data.get('hint_target', None)  # Type: target_destination
        instance.hint_static = entity_data.get('hint_static', None)  # Type: choices
        instance.hint_allow_nodraw_target = entity_data.get('hint_allow_nodraw_target', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.hint_caption = entity_data.get('hint_caption', None)  # Type: string
        instance.hint_activator_caption = entity_data.get('hint_activator_caption', None)  # Type: string
        instance.hint_color = parse_int_vector(entity_data.get('hint_color', "255 255 255"))  # Type: color255
        instance.hint_forcecaption = entity_data.get('hint_forcecaption', None)  # Type: choices
        instance.hint_icon_onscreen = entity_data.get('hint_icon_onscreen', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.hint_icon_offscreen = entity_data.get('hint_icon_offscreen', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.hint_nooffscreen = entity_data.get('hint_nooffscreen', None)  # Type: choices
        instance.hint_binding = entity_data.get('hint_binding', None)  # Type: string
        instance.hint_gamepad_binding = entity_data.get('hint_gamepad_binding', None)  # Type: string
        instance.hint_icon_offset = float(entity_data.get('hint_icon_offset', 0))  # Type: float
        instance.hint_pulseoption = entity_data.get('hint_pulseoption', None)  # Type: choices
        instance.hint_alphaoption = entity_data.get('hint_alphaoption', None)  # Type: choices
        instance.hint_shakeoption = entity_data.get('hint_shakeoption', None)  # Type: choices
        instance.hint_local_player_only = entity_data.get('hint_local_player_only', None)  # Type: boolean
        instance.hint_timeout = parse_source_value(entity_data.get('hint_timeout', 0))  # Type: integer
        instance.hint_range = float(entity_data.get('hint_range', 0))  # Type: float


class info_target_instructor_hint(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_game_event_proxy(Targetname, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.event_name = None  # Type: string
        self.range = 512  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.event_name = entity_data.get('event_name', None)  # Type: string
        instance.range = float(entity_data.get('range', 512))  # Type: float


class light_directional(Angles):
    icon_sprite = "editor/light_directional.vmt"
    def __init__(self):
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.pitch = None  # Type: integer
        self._light = [255, 255, 255, 200]  # Type: color255
        self._lightHDR = [-1, -1, -1, 1]  # Type: color255
        self._lightscaleHDR = 0.7  # Type: float
        self.SunSpreadAngle = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.pitch = parse_source_value(entity_data.get('pitch', 0))  # Type: integer
        instance._light = parse_int_vector(entity_data.get('_light', "255 255 255 200"))  # Type: color255
        instance._lightHDR = parse_int_vector(entity_data.get('_lighthdr', "-1 -1 -1 1"))  # Type: color255
        instance._lightscaleHDR = float(entity_data.get('_lightscalehdr', 0.7))  # Type: float
        instance.SunSpreadAngle = float(entity_data.get('sunspreadangle', 0))  # Type: float


class postprocess_controller(Targetname):
    icon_sprite = "editor/postprocess_controller.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.fadetime = 2  # Type: float
        self.localcontraststrength = 0  # Type: float
        self.localcontrastedgestrength = 0  # Type: float
        self.vignettestart = 0.8  # Type: float
        self.vignetteend = 1.1  # Type: float
        self.vignetteblurstrength = 0  # Type: float
        self.fadetoblackstrength = 0  # Type: float
        self.depthblurfocaldistance = 0  # Type: float
        self.depthblurstrength = 0  # Type: float
        self.screenblurstrength = 0  # Type: float
        self.filmgrainstrength = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.fadetime = float(entity_data.get('fadetime', 2))  # Type: float
        instance.localcontraststrength = float(entity_data.get('localcontraststrength', 0))  # Type: float
        instance.localcontrastedgestrength = float(entity_data.get('localcontrastedgestrength', 0))  # Type: float
        instance.vignettestart = float(entity_data.get('vignettestart', 0.8))  # Type: float
        instance.vignetteend = float(entity_data.get('vignetteend', 1.1))  # Type: float
        instance.vignetteblurstrength = float(entity_data.get('vignetteblurstrength', 0))  # Type: float
        instance.fadetoblackstrength = float(entity_data.get('fadetoblackstrength', 0))  # Type: float
        instance.depthblurfocaldistance = float(entity_data.get('depthblurfocaldistance', 0))  # Type: float
        instance.depthblurstrength = float(entity_data.get('depthblurstrength', 0))  # Type: float
        instance.screenblurstrength = float(entity_data.get('screenblurstrength', 0))  # Type: float
        instance.filmgrainstrength = float(entity_data.get('filmgrainstrength', 0))  # Type: float


class fog_volume(Targetname, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.FogName = None  # Type: target_destination
        self.PostProcessName = None  # Type: target_destination
        self.ColorCorrectionName = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.FogName = entity_data.get('fogname', None)  # Type: target_destination
        instance.PostProcessName = entity_data.get('postprocessname', None)  # Type: target_destination
        instance.ColorCorrectionName = entity_data.get('colorcorrectionname', None)  # Type: target_destination


class TalkNPC(BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        self.UseSentence = None  # Type: string
        self.UnUseSentence = None  # Type: string
        self.DontUseSpeechSemaphore = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.UseSentence = entity_data.get('usesentence', None)  # Type: string
        instance.UnUseSentence = entity_data.get('unusesentence', None)  # Type: string
        instance.DontUseSpeechSemaphore = entity_data.get('dontusespeechsemaphore', None)  # Type: choices


class PlayerCompanion(BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        self.AlwaysTransition = False # Type: boolean
        self.DontPickupWeapons = False # Type: boolean
        self.GameEndAlly = False # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.AlwaysTransition = entity_data.get('alwaystransition', None)  # Type: boolean
        instance.DontPickupWeapons = entity_data.get('dontpickupweapons', None)  # Type: boolean
        instance.GameEndAlly = entity_data.get('gameendally', None)  # Type: boolean


class RappelNPC(BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        self.waitingtorappel = False # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.waitingtorappel = entity_data.get('waitingtorappel', None)  # Type: boolean


class AlyxInteractable(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class CombineBallSpawners(Origin, Angles, Global, Targetname):
    def __init__(self):
        super(Origin).__init__()
        super(Angles).__init__()
        super(Global).__init__()
        super(Targetname).__init__()
        self.ballcount = 3  # Type: integer
        self.minspeed = 300.0  # Type: float
        self.maxspeed = 600.0  # Type: float
        self.ballradius = 20.0  # Type: float
        self.balltype = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ballrespawntime = 4.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Origin.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.ballcount = parse_source_value(entity_data.get('ballcount', 3))  # Type: integer
        instance.minspeed = float(entity_data.get('minspeed', 300.0))  # Type: float
        instance.maxspeed = float(entity_data.get('maxspeed', 600.0))  # Type: float
        instance.ballradius = float(entity_data.get('ballradius', 20.0))  # Type: float
        instance.balltype = entity_data.get('balltype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ballrespawntime = float(entity_data.get('ballrespawntime', 4.0))  # Type: float


class prop_combine_ball(BasePropPhysics):
    def __init__(self):
        super(BasePropPhysics).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePropPhysics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class trigger_physics_trap(Trigger, Angles):
    def __init__(self):
        super(Trigger).__init__()
        super(Angles).__init__()
        self.dissolvetype = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.dissolvetype = entity_data.get('dissolvetype', "CHOICES NOT SUPPORTED")  # Type: choices


class trigger_weapon_strip(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        self.KillWeapons = False # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.KillWeapons = entity_data.get('killweapons', None)  # Type: boolean


class func_combine_ball_spawner(CombineBallSpawners):
    def __init__(self):
        super(CombineBallSpawners).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        CombineBallSpawners.from_dict(instance, entity_data)


class point_combine_ball_launcher(CombineBallSpawners):
    def __init__(self):
        super(CombineBallSpawners).__init__()
        self.origin = [0, 0, 0]
        self.launchconenoise = 0.0  # Type: float
        self.bullseyename = None  # Type: string
        self.maxballbounces = 8  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        CombineBallSpawners.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.launchconenoise = float(entity_data.get('launchconenoise', 0.0))  # Type: float
        instance.bullseyename = entity_data.get('bullseyename', None)  # Type: string
        instance.maxballbounces = parse_source_value(entity_data.get('maxballbounces', 8))  # Type: integer


class npc_turret_ground(BaseNPC, AlyxInteractable, Parentname):
    model = "models/combine_turrets/ground_turret.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        super(AlyxInteractable).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        AlyxInteractable.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class npc_turret_floor(Angles, Targetname):
    model = "models/combine_turrets/floor_turret.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.SkinNumber = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.SkinNumber = parse_source_value(entity_data.get('skinnumber', 0))  # Type: integer


class npc_bullseye(BaseNPC, Parentname):
    icon_sprite = "editor/bullseye.vmt"
    def __init__(self):
        super(BaseNPC).__init__()
        super(Parentname).__init__()
        self.health = 35  # Type: integer
        self.minangle = "360"  # Type: string
        self.mindist = "0"  # Type: string
        self.alwaystransmit = 0  # Type: boolean
        self.autoaimradius = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.health = parse_source_value(entity_data.get('health', 35))  # Type: integer
        instance.minangle = entity_data.get('minangle', "360")  # Type: string
        instance.mindist = entity_data.get('mindist', "0")  # Type: string
        instance.alwaystransmit = entity_data.get('alwaystransmit', None)  # Type: boolean
        instance.autoaimradius = float(entity_data.get('autoaimradius', 0))  # Type: float


class npc_enemyfinder(BaseNPC, Parentname):
    def __init__(self):
        super(BaseNPC).__init__()
        super(Parentname).__init__()
        self.FieldOfView = "0.2"  # Type: string
        self.MinSearchDist = None  # Type: integer
        self.MaxSearchDist = 2048  # Type: integer
        self.freepass_timetotrigger = None  # Type: float
        self.freepass_duration = None  # Type: float
        self.freepass_movetolerance = 120  # Type: float
        self.freepass_refillrate = 0.5  # Type: float
        self.freepass_peektime = None  # Type: float
        self.StartOn = 1  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.FieldOfView = entity_data.get('fieldofview', "0.2")  # Type: string
        instance.MinSearchDist = parse_source_value(entity_data.get('minsearchdist', 0))  # Type: integer
        instance.MaxSearchDist = parse_source_value(entity_data.get('maxsearchdist', 2048))  # Type: integer
        instance.freepass_timetotrigger = float(entity_data.get('freepass_timetotrigger', 0))  # Type: float
        instance.freepass_duration = float(entity_data.get('freepass_duration', 0))  # Type: float
        instance.freepass_movetolerance = float(entity_data.get('freepass_movetolerance', 120))  # Type: float
        instance.freepass_refillrate = float(entity_data.get('freepass_refillrate', 0.5))  # Type: float
        instance.freepass_peektime = float(entity_data.get('freepass_peektime', 0))  # Type: float
        instance.StartOn = entity_data.get('starton', None)  # Type: boolean


class npc_spotlight(BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        self.health = 100  # Type: integer
        self.YawRange = 90  # Type: integer
        self.PitchMin = 35  # Type: integer
        self.PitchMax = 50  # Type: integer
        self.IdleSpeed = 2  # Type: integer
        self.AlertSpeed = 5  # Type: integer
        self.spotlightlength = 500  # Type: integer
        self.spotlightwidth = 50  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.health = parse_source_value(entity_data.get('health', 100))  # Type: integer
        instance.YawRange = parse_source_value(entity_data.get('yawrange', 90))  # Type: integer
        instance.PitchMin = parse_source_value(entity_data.get('pitchmin', 35))  # Type: integer
        instance.PitchMax = parse_source_value(entity_data.get('pitchmax', 50))  # Type: integer
        instance.IdleSpeed = parse_source_value(entity_data.get('idlespeed', 2))  # Type: integer
        instance.AlertSpeed = parse_source_value(entity_data.get('alertspeed', 5))  # Type: integer
        instance.spotlightlength = parse_source_value(entity_data.get('spotlightlength', 500))  # Type: integer
        instance.spotlightwidth = parse_source_value(entity_data.get('spotlightwidth', 50))  # Type: integer


class monster_generic(BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        super(RenderFields).__init__()
        super(Shadow).__init__()
        self.origin = [0, 0, 0]
        self.model = None  # Type: studio
        self.body = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', None)  # Type: studio
        instance.body = parse_source_value(entity_data.get('body', 0))  # Type: integer


class generic_actor(BaseNPC, Parentname):
    def __init__(self):
        super(BaseNPC).__init__()
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Shadow).__init__()
        self.origin = [0, 0, 0]
        self.model = None  # Type: studio
        self.hull_name = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', None)  # Type: studio
        instance.hull_name = entity_data.get('hull_name', "CHOICES NOT SUPPORTED")  # Type: choices


class cycler_actor(BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        super(RenderFields).__init__()
        super(Shadow).__init__()
        self.origin = [0, 0, 0]
        self.model = None  # Type: studio
        self.Sentence = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', None)  # Type: studio
        instance.Sentence = entity_data.get('sentence', None)  # Type: string


class npc_maker(BaseNPCMaker):
    icon_sprite = "editor/npc_maker.vmt"
    def __init__(self):
        super(BaseNPCMaker).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.NPCType = None  # Type: npcclass
        self.NPCTargetname = None  # Type: string
        self.NPCSquadname = None  # Type: string
        self.NPCHintGroup = None  # Type: string
        self.additionalequipment = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        BaseNPCMaker.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.NPCType = entity_data.get('npctype', None)  # Type: npcclass
        instance.NPCTargetname = entity_data.get('npctargetname', None)  # Type: string
        instance.NPCSquadname = entity_data.get('npcsquadname', None)  # Type: string
        instance.NPCHintGroup = entity_data.get('npchintgroup', None)  # Type: string
        instance.additionalequipment = entity_data.get('additionalequipment', "CHOICES NOT SUPPORTED")  # Type: choices


class player_control(Targetname):
    def __init__(self):
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)


class BaseScripted(Angles, Targetname, Parentname):
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.m_iszEntity = None  # Type: target_destination
        self.m_iszIdle = None  # Type: string
        self.m_iszEntry = None  # Type: string
        self.m_iszPlay = None  # Type: string
        self.m_iszPostIdle = None  # Type: string
        self.m_iszCustomMove = None  # Type: string
        self.m_bLoopActionSequence = None  # Type: boolean
        self.m_bSynchPostIdles = None  # Type: boolean
        self.m_flRadius = None  # Type: integer
        self.m_flRepeat = None  # Type: integer
        self.m_fMoveTo = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_iszNextScript = None  # Type: target_destination
        self.m_bIgnoreGravity = None  # Type: boolean
        self.m_bDisableNPCCollisions = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.m_iszEntity = entity_data.get('m_iszentity', None)  # Type: target_destination
        instance.m_iszIdle = entity_data.get('m_iszidle', None)  # Type: string
        instance.m_iszEntry = entity_data.get('m_iszentry', None)  # Type: string
        instance.m_iszPlay = entity_data.get('m_iszplay', None)  # Type: string
        instance.m_iszPostIdle = entity_data.get('m_iszpostidle', None)  # Type: string
        instance.m_iszCustomMove = entity_data.get('m_iszcustommove', None)  # Type: string
        instance.m_bLoopActionSequence = entity_data.get('m_bloopactionsequence', None)  # Type: boolean
        instance.m_bSynchPostIdles = entity_data.get('m_bsynchpostidles', None)  # Type: boolean
        instance.m_flRadius = parse_source_value(entity_data.get('m_flradius', 0))  # Type: integer
        instance.m_flRepeat = parse_source_value(entity_data.get('m_flrepeat', 0))  # Type: integer
        instance.m_fMoveTo = entity_data.get('m_fmoveto', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.m_iszNextScript = entity_data.get('m_isznextscript', None)  # Type: target_destination
        instance.m_bIgnoreGravity = entity_data.get('m_bignoregravity', None)  # Type: boolean
        instance.m_bDisableNPCCollisions = entity_data.get('m_bdisablenpccollisions', None)  # Type: boolean


class scripted_sentence(Targetname):
    icon_sprite = "editor/scripted_sentence.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.sentence = None  # Type: string
        self.entity = None  # Type: string
        self.delay = "0"  # Type: string
        self.radius = 512  # Type: integer
        self.refire = "3"  # Type: string
        self.listener = None  # Type: string
        self.volume = "10"  # Type: string
        self.attenuation = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.sentence = entity_data.get('sentence', None)  # Type: string
        instance.entity = entity_data.get('entity', None)  # Type: string
        instance.delay = entity_data.get('delay', "0")  # Type: string
        instance.radius = parse_source_value(entity_data.get('radius', 512))  # Type: integer
        instance.refire = entity_data.get('refire', "3")  # Type: string
        instance.listener = entity_data.get('listener', None)  # Type: string
        instance.volume = entity_data.get('volume', "10")  # Type: string
        instance.attenuation = entity_data.get('attenuation', None)  # Type: choices


class scripted_target(Targetname, Parentname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.StartDisabled = 1  # Type: boolean
        self.m_iszEntity = None  # Type: npcclass
        self.m_flRadius = None  # Type: integer
        self.MoveSpeed = 5  # Type: integer
        self.PauseDuration = None  # Type: integer
        self.EffectDuration = 2  # Type: integer
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.StartDisabled = entity_data.get('startdisabled', None)  # Type: boolean
        instance.m_iszEntity = entity_data.get('m_iszentity', None)  # Type: npcclass
        instance.m_flRadius = parse_source_value(entity_data.get('m_flradius', 0))  # Type: integer
        instance.MoveSpeed = parse_source_value(entity_data.get('movespeed', 5))  # Type: integer
        instance.PauseDuration = parse_source_value(entity_data.get('pauseduration', 0))  # Type: integer
        instance.EffectDuration = parse_source_value(entity_data.get('effectduration', 2))  # Type: integer
        instance.target = entity_data.get('target', None)  # Type: target_destination


class ai_relationship(Targetname):
    icon_sprite = "editor/ai_relationship.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.subject = None  # Type: target_name_or_class
        self.target = None  # Type: target_name_or_class
        self.disposition = "CHOICES NOT SUPPORTED"  # Type: choices
        self.radius = None  # Type: float
        self.rank = None  # Type: integer
        self.StartActive = None  # Type: boolean
        self.Reciprocal = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.subject = entity_data.get('subject', None)  # Type: target_name_or_class
        instance.target = entity_data.get('target', None)  # Type: target_name_or_class
        instance.disposition = entity_data.get('disposition', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.radius = float(entity_data.get('radius', 0))  # Type: float
        instance.rank = parse_source_value(entity_data.get('rank', 0))  # Type: integer
        instance.StartActive = entity_data.get('startactive', None)  # Type: boolean
        instance.Reciprocal = entity_data.get('reciprocal', None)  # Type: boolean


class ai_ally_manager(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.maxallies = 5  # Type: integer
        self.maxmedics = 1  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.maxallies = parse_source_value(entity_data.get('maxallies', 5))  # Type: integer
        instance.maxmedics = parse_source_value(entity_data.get('maxmedics', 1))  # Type: integer


class LeadGoalBase(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.actor = None  # Type: target_name_or_class
        self.goal = None  # Type: string
        self.WaitPointName = None  # Type: target_destination
        self.WaitDistance = None  # Type: float
        self.LeadDistance = 64  # Type: float
        self.RetrieveDistance = 96  # Type: float
        self.SuccessDistance = 0  # Type: float
        self.Run = 0  # Type: boolean
        self.Retrieve = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ComingBackWaitForSpeak = "CHOICES NOT SUPPORTED"  # Type: choices
        self.RetrieveWaitForSpeak = "CHOICES NOT SUPPORTED"  # Type: choices
        self.DontSpeakStart = None  # Type: choices
        self.LeadDuringCombat = None  # Type: choices
        self.GagLeader = None  # Type: choices
        self.AttractPlayerConceptModifier = None  # Type: string
        self.WaitOverConceptModifier = None  # Type: string
        self.ArrivalConceptModifier = None  # Type: string
        self.PostArrivalConceptModifier = None  # Type: string
        self.SuccessConceptModifier = None  # Type: string
        self.FailureConceptModifier = None  # Type: string
        self.ComingBackConceptModifier = None  # Type: string
        self.RetrieveConceptModifier = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.actor = entity_data.get('actor', None)  # Type: target_name_or_class
        instance.goal = entity_data.get('goal', None)  # Type: string
        instance.WaitPointName = entity_data.get('waitpointname', None)  # Type: target_destination
        instance.WaitDistance = float(entity_data.get('waitdistance', 0))  # Type: float
        instance.LeadDistance = float(entity_data.get('leaddistance', 64))  # Type: float
        instance.RetrieveDistance = float(entity_data.get('retrievedistance', 96))  # Type: float
        instance.SuccessDistance = float(entity_data.get('successdistance', 0))  # Type: float
        instance.Run = entity_data.get('run', None)  # Type: boolean
        instance.Retrieve = entity_data.get('retrieve', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ComingBackWaitForSpeak = entity_data.get('comingbackwaitforspeak', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.RetrieveWaitForSpeak = entity_data.get('retrievewaitforspeak', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.DontSpeakStart = entity_data.get('dontspeakstart', None)  # Type: choices
        instance.LeadDuringCombat = entity_data.get('leadduringcombat', None)  # Type: choices
        instance.GagLeader = entity_data.get('gagleader', None)  # Type: choices
        instance.AttractPlayerConceptModifier = entity_data.get('attractplayerconceptmodifier', None)  # Type: string
        instance.WaitOverConceptModifier = entity_data.get('waitoverconceptmodifier', None)  # Type: string
        instance.ArrivalConceptModifier = entity_data.get('arrivalconceptmodifier', None)  # Type: string
        instance.PostArrivalConceptModifier = entity_data.get('postarrivalconceptmodifier', None)  # Type: string
        instance.SuccessConceptModifier = entity_data.get('successconceptmodifier', None)  # Type: string
        instance.FailureConceptModifier = entity_data.get('failureconceptmodifier', None)  # Type: string
        instance.ComingBackConceptModifier = entity_data.get('comingbackconceptmodifier', None)  # Type: string
        instance.RetrieveConceptModifier = entity_data.get('retrieveconceptmodifier', None)  # Type: string


class ai_goal_lead(LeadGoalBase):
    icon_sprite = "editor/ai_goal_lead.vmt"
    def __init__(self):
        super(LeadGoalBase).__init__()
        self.origin = [0, 0, 0]
        self.SearchType = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        LeadGoalBase.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.SearchType = entity_data.get('searchtype', None)  # Type: choices


class ai_goal_lead_weapon(LeadGoalBase):
    icon_sprite = "editor/ai_goal_lead.vmt"
    def __init__(self):
        super(LeadGoalBase).__init__()
        self.origin = [0, 0, 0]
        self.WeaponName = "CHOICES NOT SUPPORTED"  # Type: choices
        self.MissingWeaponConceptModifier = None  # Type: string
        self.SearchType = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        LeadGoalBase.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.WeaponName = entity_data.get('weaponname', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.MissingWeaponConceptModifier = entity_data.get('missingweaponconceptmodifier', None)  # Type: string
        instance.SearchType = entity_data.get('searchtype', None)  # Type: choices


class FollowGoal(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.actor = None  # Type: target_name_or_class
        self.goal = None  # Type: string
        self.SearchType = None  # Type: choices
        self.StartActive = None  # Type: boolean
        self.MaximumState = "CHOICES NOT SUPPORTED"  # Type: choices
        self.Formation = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.actor = entity_data.get('actor', None)  # Type: target_name_or_class
        instance.goal = entity_data.get('goal', None)  # Type: string
        instance.SearchType = entity_data.get('searchtype', None)  # Type: choices
        instance.StartActive = entity_data.get('startactive', None)  # Type: boolean
        instance.MaximumState = entity_data.get('maximumstate', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.Formation = entity_data.get('formation', None)  # Type: choices


class ai_goal_follow(FollowGoal):
    icon_sprite = "editor/ai_goal_follow.vmt"
    def __init__(self):
        super(FollowGoal).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        FollowGoal.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class ai_goal_injured_follow(FollowGoal):
    icon_sprite = "editor/ai_goal_follow.vmt"
    def __init__(self):
        super(FollowGoal).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        FollowGoal.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class point_energy_ball_launcher(CombineBallSpawners, Parentname):
    def __init__(self):
        super(CombineBallSpawners).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.BallLifetime = 12  # Type: float
        self.MinLifeAfterPortal = 6  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        CombineBallSpawners.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.BallLifetime = float(entity_data.get('balllifetime', 12))  # Type: float
        instance.MinLifeAfterPortal = float(entity_data.get('minlifeafterportal', 6))  # Type: float


class npc_rocket_turret(Angles, Targetname, Parentname):
    model = "models/props_bts/rocket_sentry.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.TripwireMode = None  # Type: choices
        self.TripwireAimTarget = None  # Type: target_destination
        self.RocketSpeed = 450  # Type: float
        self.RocketLifetime = 20  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.TripwireMode = entity_data.get('tripwiremode', None)  # Type: choices
        instance.TripwireAimTarget = entity_data.get('tripwireaimtarget', None)  # Type: target_destination
        instance.RocketSpeed = float(entity_data.get('rocketspeed', 450))  # Type: float
        instance.RocketLifetime = float(entity_data.get('rocketlifetime', 20))  # Type: float


class env_portal_path_track(Angles, Targetname, Parentname):
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.Track_beam_scale = None  # Type: float
        self.End_point_scale = None  # Type: float
        self.End_point_fadeout = None  # Type: float
        self.End_point_fadein = None  # Type: float
        self.target = None  # Type: target_destination
        self.altpath = None  # Type: target_destination
        self.speed = None  # Type: float
        self.radius = None  # Type: float
        self.orientationtype = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Track_beam_scale = float(entity_data.get('track_beam_scale', 0))  # Type: float
        instance.End_point_scale = float(entity_data.get('end_point_scale', 0))  # Type: float
        instance.End_point_fadeout = float(entity_data.get('end_point_fadeout', 0))  # Type: float
        instance.End_point_fadein = float(entity_data.get('end_point_fadein', 0))  # Type: float
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.altpath = entity_data.get('altpath', None)  # Type: target_destination
        instance.speed = float(entity_data.get('speed', 0))  # Type: float
        instance.radius = float(entity_data.get('radius', 0))  # Type: float
        instance.orientationtype = entity_data.get('orientationtype', "CHOICES NOT SUPPORTED")  # Type: choices


class trigger_portal_cleanser(Trigger, Reflection):
    def __init__(self):
        super(Trigger).__init__()
        super(Targetname).__init__()
        super(Reflection).__init__()
        self.Visible = 0  # Type: boolean
        self.UseScanline = 1  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Reflection.from_dict(instance, entity_data)
        instance.Visible = entity_data.get('visible', None)  # Type: boolean
        instance.UseScanline = entity_data.get('usescanline', None)  # Type: boolean


class func_portal_orientation(EnableDisable, Targetname, Parentname):
    def __init__(self):
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.AnglesToFace = [0.0, 0.0, 0.0]  # Type: angle
        self.MatchLinkedAngles = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.AnglesToFace = parse_float_vector(entity_data.get('anglestoface', "0 0 0"))  # Type: angle
        instance.MatchLinkedAngles = entity_data.get('matchlinkedangles', None)  # Type: choices


class func_weight_button(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.WeightToActivate = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.WeightToActivate = float(entity_data.get('weighttoactivate', 0))  # Type: float


class func_noportal_volume(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)


class func_portal_bumper(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)


class func_portal_detector(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.LinkageGroupID = None  # Type: integer
        self.CheckAllIDs = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.LinkageGroupID = parse_source_value(entity_data.get('linkagegroupid', 0))  # Type: integer
        instance.CheckAllIDs = entity_data.get('checkallids', None)  # Type: boolean


class PortalBase(Base):
    def __init__(self):
        super().__init__()
        self.Activated = "CHOICES NOT SUPPORTED"  # Type: choices
        self.PortalTwo = None  # Type: choices
        self.HalfWidth = None  # Type: float
        self.HalfHeight = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.Activated = entity_data.get('activated', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.PortalTwo = entity_data.get('portaltwo', None)  # Type: choices
        instance.HalfWidth = float(entity_data.get('halfwidth', 0))  # Type: float
        instance.HalfHeight = float(entity_data.get('halfheight', 0))  # Type: float


class prop_portal(Angles, PortalBase, Targetname):
    model = "models/editor/axis_helper_thick.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(PortalBase).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.LinkageGroupID = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        PortalBase.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.LinkageGroupID = parse_source_value(entity_data.get('linkagegroupid', 0))  # Type: integer


class weapon_portalgun(Targetname, Parentname):
    model = "models/weapons/w_portalgun.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.CanFirePortal1 = 1  # Type: boolean
        self.CanFirePortal2 = 1  # Type: boolean
        self.ShowingPotatos = None  # Type: boolean
        self.StartingTeamNum = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.CanFirePortal1 = entity_data.get('canfireportal1', None)  # Type: boolean
        instance.CanFirePortal2 = entity_data.get('canfireportal2', None)  # Type: boolean
        instance.ShowingPotatos = entity_data.get('showingpotatos', None)  # Type: boolean
        instance.StartingTeamNum = entity_data.get('startingteamnum', None)  # Type: choices


class npc_portal_turret_ground(npc_turret_ground):
    model = "models/combine_turrets/ground_turret.mdl"
    def __init__(self):
        super(npc_turret_ground).__init__()
        self.origin = [0, 0, 0]
        self.ConeOfFire = 60  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        npc_turret_ground.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.ConeOfFire = float(entity_data.get('coneoffire', 60))  # Type: float


class prop_glados_core(BasePropPhysics):
    model = "models/npcs/personality_sphere/personality_sphere.mdl"
    def __init__(self):
        super(BasePropPhysics).__init__()
        self.origin = [0, 0, 0]
        self.CoreType = "CHOICES NOT SUPPORTED"  # Type: choices
        self.DelayBetweenLines = 0.4  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePropPhysics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.CoreType = entity_data.get('coretype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.DelayBetweenLines = float(entity_data.get('delaybetweenlines', 0.4))  # Type: float


class npc_portal_turret_floor(npc_turret_floor):
    model = "models/props/turret_01.mdl"
    def __init__(self):
        super(npc_turret_floor).__init__()
        self.origin = [0, 0, 0]
        self.Gagged = None  # Type: boolean
        self.UsedAsActor = None  # Type: boolean
        self.PickupEnabled = 1  # Type: boolean
        self.DisableMotion = None  # Type: boolean
        self.AllowShootThroughPortals = None  # Type: boolean
        self.TurretRange = 1024  # Type: float
        self.LoadAlternativeModels = None  # Type: boolean
        self.UseSuperDamageScale = None  # Type: boolean
        self.CollisionType = None  # Type: choices
        self.ModelIndex = None  # Type: choices
        self.DamageForce = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        npc_turret_floor.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Gagged = entity_data.get('gagged', None)  # Type: boolean
        instance.UsedAsActor = entity_data.get('usedasactor', None)  # Type: boolean
        instance.PickupEnabled = entity_data.get('pickupenabled', None)  # Type: boolean
        instance.DisableMotion = entity_data.get('disablemotion', None)  # Type: boolean
        instance.AllowShootThroughPortals = entity_data.get('allowshootthroughportals', None)  # Type: boolean
        instance.TurretRange = float(entity_data.get('turretrange', 1024))  # Type: float
        instance.LoadAlternativeModels = entity_data.get('loadalternativemodels', None)  # Type: boolean
        instance.UseSuperDamageScale = entity_data.get('usesuperdamagescale', None)  # Type: boolean
        instance.CollisionType = entity_data.get('collisiontype', None)  # Type: choices
        instance.ModelIndex = entity_data.get('modelindex', None)  # Type: choices
        instance.DamageForce = entity_data.get('damageforce', None)  # Type: boolean


class npc_security_camera(Angles, Studiomodel, Targetname):
    model = "models/props/security_camera.mdl"
    def __init__(self):
        super(Studiomodel).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.LookAtPlayerPings = 0  # Type: boolean
        self.TeamToLookAt = "CHOICES NOT SUPPORTED"  # Type: choices
        self.TeamPlayerToLookAt = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.LookAtPlayerPings = entity_data.get('lookatplayerpings', None)  # Type: boolean
        instance.TeamToLookAt = entity_data.get('teamtolookat', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.TeamPlayerToLookAt = entity_data.get('teamplayertolookat', None)  # Type: choices


class prop_telescopic_arm(Angles, Studiomodel, Targetname):
    model = "models/props/telescopic_arm.mdl"
    def __init__(self):
        super(Studiomodel).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class prop_portal_stats_display(Angles, Global, Targetname, Parentname):
    model = "models/props/Round_elevator_body.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Global).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class vgui_neurotoxin_countdown(Angles, Targetname, Parentname):
    model = "models/editor/axis_helper_thick.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.width = 256  # Type: integer
        self.height = 128  # Type: integer
        self.countdown = 60  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.width = parse_source_value(entity_data.get('width', 256))  # Type: integer
        instance.height = parse_source_value(entity_data.get('height', 128))  # Type: integer
        instance.countdown = parse_source_value(entity_data.get('countdown', 60))  # Type: integer


class env_lightrail_endpoint(Angles, Targetname, Parentname):
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.small_fx_scale = 1  # Type: float
        self.large_fx_scale = 1  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.small_fx_scale = float(entity_data.get('small_fx_scale', 1))  # Type: float
        instance.large_fx_scale = float(entity_data.get('large_fx_scale', 1))  # Type: float


class env_portal_credits(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_lighting_relative(Targetname, Parentname):
    icon_sprite = "editor/info_lighting.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.LightingLandmark = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.LightingLandmark = entity_data.get('lightinglandmark', None)  # Type: target_destination


class prop_mirror(Studiomodel, Angles, Targetname, Parentname):
    def __init__(self):
        super(Studiomodel).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.Width = 64.0  # Type: float
        self.Height = 108.0  # Type: float
        self.PhysicsEnabled = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Studiomodel.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Width = float(entity_data.get('width', 64.0))  # Type: float
        instance.Height = float(entity_data.get('height', 108.0))  # Type: float
        instance.PhysicsEnabled = entity_data.get('physicsenabled', None)  # Type: boolean


class point_futbol_shooter(Targetname, Angles, Parentname):
    model = "models/editor/angle_helper.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.launchSpeed = 100  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.launchSpeed = float(entity_data.get('launchspeed', 100))  # Type: float


class LinkedPortalDoor(Base):
    def __init__(self):
        super().__init__()
        self.partnername = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.partnername = entity_data.get('partnername', None)  # Type: target_destination


class BaseProjector(Base):
    def __init__(self):
        super().__init__()
        self.StartEnabled = 1  # Type: boolean
        self.DisableHelper = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.StartEnabled = entity_data.get('startenabled', None)  # Type: boolean
        instance.DisableHelper = entity_data.get('disablehelper', None)  # Type: boolean


class info_target_personality_sphere(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.sphereLine = None  # Type: string
        self.radius = 16  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.sphereLine = entity_data.get('sphereline', None)  # Type: string
        instance.radius = float(entity_data.get('radius', 16))  # Type: float


class prop_rocket_tripwire(Targetname, Angles, Parentname):
    viewport_model = "models/props/tripwire_turret.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.RocketSpeed = 450  # Type: float
        self.RocketLifetime = 20  # Type: float
        self.StartDisabled = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.RocketSpeed = float(entity_data.get('rocketspeed', 450))  # Type: float
        instance.RocketLifetime = float(entity_data.get('rocketlifetime', 20))  # Type: float
        instance.StartDisabled = entity_data.get('startdisabled', None)  # Type: choices


class func_camera_target(Targetname):
    def __init__(self):
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)


class linked_portal_door(Targetname, Angles, Parentname, LinkedPortalDoor):
    model = "models/editor/axis_helper_thick.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        super(LinkedPortalDoor).__init__()
        self.origin = [0, 0, 0]
        self.width = 128  # Type: integer
        self.height = 128  # Type: integer
        self.startactive = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        LinkedPortalDoor.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.width = parse_source_value(entity_data.get('width', 128))  # Type: integer
        instance.height = parse_source_value(entity_data.get('height', 128))  # Type: integer
        instance.startactive = entity_data.get('startactive', None)  # Type: boolean


class prop_linked_portal_door(Targetname, Angles, Parentname, LinkedPortalDoor):
    model = "models/props/portal_door.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        super(LinkedPortalDoor).__init__()
        self.origin = [0, 0, 0]
        self.lightingorigin = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        LinkedPortalDoor.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.lightingorigin = entity_data.get('lightingorigin', None)  # Type: target_destination


class prop_button(Targetname, Angles, Parentname):
    model = "models/props/switch001.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.Delay = 1  # Type: float
        self.istimer = None  # Type: boolean
        self.preventfastreset = None  # Type: boolean
        self.skin = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Delay = float(entity_data.get('delay', 1))  # Type: float
        instance.istimer = entity_data.get('istimer', None)  # Type: boolean
        instance.preventfastreset = entity_data.get('preventfastreset', None)  # Type: boolean
        instance.skin = entity_data.get('skin', None)  # Type: choices


class prop_under_button(prop_button):
    model = "models/props_underground/underground_testchamber_button.mdl"
    def __init__(self):
        super(prop_button).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_button.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class prop_floor_button(Targetname, Angles, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.model = "models/props/portal_button.mdl"  # Type: studio

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', "models/props/portal_button.mdl")  # Type: studio


class prop_floor_cube_button(Targetname, Angles, Parentname):
    model = "models/props/box_socket.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.AcceptsBall = 1  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.AcceptsBall = entity_data.get('acceptsball', None)  # Type: boolean


class prop_floor_ball_button(Targetname, Angles, Parentname):
    model = "models/props/ball_button.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class prop_under_floor_button(Targetname, Angles, Parentname):
    model = "models/props_underground/underground_floor_button.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class prop_wall_projector(Targetname, Angles, BaseProjector):
    model = "models/props/wall_emitter.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(BaseProjector).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        BaseProjector.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_placement_helper(Targetname, Angles, Parentname, EnableDisable):
    model = "models/editor/angle_helper.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.radius = 16  # Type: float
        self.proxy_name = None  # Type: target_destination
        self.attach_target_name = None  # Type: string
        self.snap_to_helper_angles = None  # Type: boolean
        self.force_placement = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.radius = float(entity_data.get('radius', 16))  # Type: float
        instance.proxy_name = entity_data.get('proxy_name', None)  # Type: target_destination
        instance.attach_target_name = entity_data.get('attach_target_name', None)  # Type: string
        instance.snap_to_helper_angles = entity_data.get('snap_to_helper_angles', None)  # Type: boolean
        instance.force_placement = entity_data.get('force_placement', None)  # Type: boolean


class info_player_ping_detector(Targetname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.FuncTankName = None  # Type: target_destination
        self.TeamToLookAt = "CHOICES NOT SUPPORTED"  # Type: choices
        self.Enabled = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.FuncTankName = entity_data.get('functankname', None)  # Type: target_destination
        instance.TeamToLookAt = entity_data.get('teamtolookat', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.Enabled = entity_data.get('enabled', "CHOICES NOT SUPPORTED")  # Type: choices


class func_placement_clip(Trigger):
    def __init__(self):
        super(Trigger).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)


class filter_player_held(BaseFilter):
    icon_sprite = "editor/filter_name.vmt"
    def __init__(self):
        super(BaseFilter).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)


class env_portal_laser(Parentname, Angles, Targetname, Reflection):
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        super(Reflection).__init__()
        self.origin = [0, 0, 0]
        self.NoPlacementHelper = None  # Type: boolean
        self.model = "models/props/laser_emitter.mdl"  # Type: studio
        self.StartState = None  # Type: choices
        self.LethalDamage = None  # Type: choices
        self.AutoAimEnabled = 1  # Type: boolean
        self.skin = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Reflection.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.NoPlacementHelper = entity_data.get('noplacementhelper', None)  # Type: boolean
        instance.model = entity_data.get('model', "models/props/laser_emitter.mdl")  # Type: studio
        instance.StartState = entity_data.get('startstate', None)  # Type: choices
        instance.LethalDamage = entity_data.get('lethaldamage', None)  # Type: choices
        instance.AutoAimEnabled = entity_data.get('autoaimenabled', None)  # Type: boolean
        instance.skin = entity_data.get('skin', None)  # Type: choices


class point_laser_target(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.terminalpoint = 1  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.terminalpoint = entity_data.get('terminalpoint', None)  # Type: boolean


class prop_laser_catcher(Parentname, Targetname, Reflection):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Reflection).__init__()
        self.origin = [0, 0, 0]
        self.SkinType = None  # Type: choices
        self.model = "models/props/laser_catcher.mdl"  # Type: studio

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Reflection.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.SkinType = entity_data.get('skintype', None)  # Type: choices
        instance.model = entity_data.get('model', "models/props/laser_catcher.mdl")  # Type: studio


class prop_laser_relay(Parentname, Targetname, Reflection):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Reflection).__init__()
        self.origin = [0, 0, 0]
        self.model = "models/props/laser_receptacle.mdl"  # Type: studio

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Reflection.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', "models/props/laser_receptacle.mdl")  # Type: studio


class prop_weighted_cube(Targetname, Angles, Reflection):
    model = "models/props/metal_box.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Reflection).__init__()
        self.origin = [0, 0, 0]
        self.skin = None  # Type: choices
        self.CubeType = None  # Type: choices
        self.SkinType = None  # Type: choices
        self.PaintPower = "CHOICES NOT SUPPORTED"  # Type: choices
        self.NewSkins = None  # Type: boolean
        self.allowfunnel = 1  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Reflection.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.skin = entity_data.get('skin', None)  # Type: choices
        instance.CubeType = entity_data.get('cubetype', None)  # Type: choices
        instance.SkinType = entity_data.get('skintype', None)  # Type: choices
        instance.PaintPower = entity_data.get('paintpower', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.NewSkins = entity_data.get('newskins', None)  # Type: boolean
        instance.allowfunnel = entity_data.get('allowfunnel', None)  # Type: boolean


class trigger_catapult(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        self.playerSpeed = 450  # Type: float
        self.physicsSpeed = 450  # Type: float
        self.useThresholdCheck = None  # Type: boolean
        self.entryAngleTolerance = 0.0  # Type: float
        self.useExactVelocity = None  # Type: boolean
        self.exactVelocityChoiceType = None  # Type: choices
        self.lowerThreshold = 0.15  # Type: float
        self.upperThreshold = 0.30  # Type: float
        self.launchDirection = [0.0, 0.0, 0.0]  # Type: angle
        self.launchTarget = None  # Type: target_destination
        self.onlyVelocityCheck = None  # Type: boolean
        self.applyAngularImpulse = 1  # Type: boolean
        self.AirCtrlSupressionTime = -1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.playerSpeed = float(entity_data.get('playerspeed', 450))  # Type: float
        instance.physicsSpeed = float(entity_data.get('physicsspeed', 450))  # Type: float
        instance.useThresholdCheck = entity_data.get('usethresholdcheck', None)  # Type: boolean
        instance.entryAngleTolerance = float(entity_data.get('entryangletolerance', 0.0))  # Type: float
        instance.useExactVelocity = entity_data.get('useexactvelocity', None)  # Type: boolean
        instance.exactVelocityChoiceType = entity_data.get('exactvelocitychoicetype', None)  # Type: choices
        instance.lowerThreshold = float(entity_data.get('lowerthreshold', 0.15))  # Type: float
        instance.upperThreshold = float(entity_data.get('upperthreshold', 0.30))  # Type: float
        instance.launchDirection = parse_float_vector(entity_data.get('launchdirection', "0 0 0"))  # Type: angle
        instance.launchTarget = entity_data.get('launchtarget', None)  # Type: target_destination
        instance.onlyVelocityCheck = entity_data.get('onlyvelocitycheck', None)  # Type: boolean
        instance.applyAngularImpulse = entity_data.get('applyangularimpulse', None)  # Type: boolean
        instance.AirCtrlSupressionTime = float(entity_data.get('airctrlsupressiontime', -1.0))  # Type: float


class prop_glass_futbol(BasePropPhysics):
    model = "models/props/futbol.mdl"
    def __init__(self):
        super(BasePropPhysics).__init__()
        self.origin = [0, 0, 0]
        self.SpawnerName = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePropPhysics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.SpawnerName = entity_data.get('spawnername', None)  # Type: string


class prop_glass_futbol_spawner(Targetname):
    model = "models/props/futbol_dispenser.mdl"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.StartWithFutbol = 1  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.StartWithFutbol = entity_data.get('startwithfutbol', None)  # Type: boolean


class prop_glass_futbol_socket(Targetname):
    model = "models/props/futbol_socket.mdl"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class portalmp_gamerules(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_nugget(Parentname, Angles, Targetname):
    model = "models/effects/cappoint_hologram.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.GroupName = None  # Type: string
        self.RespawnTime = 30  # Type: float
        self.PointValue = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.GroupName = entity_data.get('groupname', None)  # Type: string
        instance.RespawnTime = float(entity_data.get('respawntime', 30))  # Type: float
        instance.PointValue = entity_data.get('pointvalue', "CHOICES NOT SUPPORTED")  # Type: choices


class func_portalled(func_portal_detector):
    def __init__(self):
        super(func_portal_detector).__init__()
        self.FireOnDeparture = 1  # Type: boolean
        self.FireOnArrival = 1  # Type: boolean
        self.FireOnPlayer = 1  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        func_portal_detector.from_dict(instance, entity_data)
        instance.FireOnDeparture = entity_data.get('fireondeparture', None)  # Type: boolean
        instance.FireOnArrival = entity_data.get('fireonarrival', None)  # Type: boolean
        instance.FireOnPlayer = entity_data.get('fireonplayer', None)  # Type: boolean


class logic_player_slowtime(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class logic_timescale(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.BlendTime = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.BlendTime = float(entity_data.get('blendtime', 0))  # Type: float


class env_player_viewfinder(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_coop_spawn(Targetname, PlayerClass, Angles):
    model = "models/editor/playerstart.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(PlayerClass).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.Enabled = None  # Type: choices
        self.StartingTeam = None  # Type: choices
        self.ForceGunOnSpawn = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        PlayerClass.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Enabled = entity_data.get('enabled', None)  # Type: choices
        instance.StartingTeam = entity_data.get('startingteam', None)  # Type: choices
        instance.ForceGunOnSpawn = entity_data.get('forcegunonspawn', None)  # Type: boolean


class npc_personality_core(TalkNPC, Parentname):
    model = "models/npcs/personality_sphere/personality_sphere.mdl"
    def __init__(self):
        super(TalkNPC).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.ModelSkin = None  # Type: choices
        self.altmodel = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TalkNPC.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.ModelSkin = entity_data.get('modelskin', None)  # Type: choices
        instance.altmodel = entity_data.get('altmodel', None)  # Type: choices


class prop_monster_box(BasePropPhysics, Parentname):
    model = "models/npcs/monsters/monster_a.mdl"
    def __init__(self):
        super(BasePropPhysics).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.StartAsBox = None  # Type: boolean
        self.BoxSwitchSpeed = 400  # Type: float
        self.AllowSilentDissolve = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePropPhysics.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.StartAsBox = entity_data.get('startasbox', None)  # Type: boolean
        instance.BoxSwitchSpeed = float(entity_data.get('boxswitchspeed', 400))  # Type: float
        instance.AllowSilentDissolve = entity_data.get('allowsilentdissolve', None)  # Type: boolean


class prop_indicator_panel(Targetname, Angles, Parentname):
    model = "models/props/sign_frame01/sign_frame01.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.TimerDuration = None  # Type: float
        self.Enabled = None  # Type: boolean
        self.IsTimer = None  # Type: boolean
        self.IsChecked = None  # Type: boolean
        self.IndicatorLights = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.TimerDuration = float(entity_data.get('timerduration', 0))  # Type: float
        instance.Enabled = entity_data.get('enabled', None)  # Type: boolean
        instance.IsTimer = entity_data.get('istimer', None)  # Type: boolean
        instance.IsChecked = entity_data.get('ischecked', None)  # Type: boolean
        instance.IndicatorLights = entity_data.get('indicatorlights', None)  # Type: target_destination


class prop_tic_tac_toe_panel(Targetname, Angles, Parentname):
    model = "models/props/sign_frame01/sign_frame01.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class trigger_playerteam(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        self.target_team = None  # Type: choices
        self.trigger_once = False # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.target_team = entity_data.get('target_team', None)  # Type: choices
        instance.trigger_once = entity_data.get('trigger_once', None)  # Type: boolean


class trigger_ping_detector(Targetname):
    def __init__(self):
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)


class info_landmark_entry(Targetname):
    icon_sprite = "editor/info_landmark"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_landmark_exit(Targetname):
    icon_sprite = "editor/info_landmark"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class point_changelevel(Targetname):
    icon_sprite = "editor/game_end.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class prop_tractor_beam(Parentname, Shadow, Targetname, BaseProjector, Reflection, Angles):
    model = "models/props/tractor_beam_emitter.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        super(BaseProjector).__init__()
        super(Reflection).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.linearForce = 250  # Type: float
        self.noemitterparticles = None  # Type: boolean
        self.use128model = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        BaseProjector.from_dict(instance, entity_data)
        Reflection.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.linearForce = float(entity_data.get('linearforce', 250))  # Type: float
        instance.noemitterparticles = entity_data.get('noemitterparticles', None)  # Type: boolean
        instance.use128model = entity_data.get('use128model', None)  # Type: boolean


class info_paint_sprayer(Targetname, Angles, Parentname):
    model = "models/editor/cone_helper.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.maxblobcount = 250  # Type: integer
        self.light_position_name = None  # Type: string
        self.start_active = None  # Type: boolean
        self.silent = 0  # Type: boolean
        self.DrawOnly = 0  # Type: boolean
        self.PaintType = None  # Type: choices
        self.RenderMode = None  # Type: choices
        self.AmbientSound = None  # Type: choices
        self.blobs_per_second = 1  # Type: float
        self.min_speed = 100  # Type: float
        self.max_speed = 100  # Type: float
        self.blob_spread_radius = 0  # Type: float
        self.blob_spread_angle = 0  # Type: float
        self.blob_streak_percentage = 0  # Type: float
        self.min_streak_time = 0.2  # Type: float
        self.max_streak_time = 0.5  # Type: float
        self.min_streak_speed_dampen = 500  # Type: float
        self.max_streak_speed_dampen = 1000  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.maxblobcount = parse_source_value(entity_data.get('maxblobcount', 250))  # Type: integer
        instance.light_position_name = entity_data.get('light_position_name', None)  # Type: string
        instance.start_active = entity_data.get('start_active', None)  # Type: boolean
        instance.silent = entity_data.get('silent', None)  # Type: boolean
        instance.DrawOnly = entity_data.get('drawonly', None)  # Type: boolean
        instance.PaintType = entity_data.get('painttype', None)  # Type: choices
        instance.RenderMode = entity_data.get('rendermode', None)  # Type: choices
        instance.AmbientSound = entity_data.get('ambientsound', None)  # Type: choices
        instance.blobs_per_second = float(entity_data.get('blobs_per_second', 1))  # Type: float
        instance.min_speed = float(entity_data.get('min_speed', 100))  # Type: float
        instance.max_speed = float(entity_data.get('max_speed', 100))  # Type: float
        instance.blob_spread_radius = float(entity_data.get('blob_spread_radius', 0))  # Type: float
        instance.blob_spread_angle = float(entity_data.get('blob_spread_angle', 0))  # Type: float
        instance.blob_streak_percentage = float(entity_data.get('blob_streak_percentage', 0))  # Type: float
        instance.min_streak_time = float(entity_data.get('min_streak_time', 0.2))  # Type: float
        instance.max_streak_time = float(entity_data.get('max_streak_time', 0.5))  # Type: float
        instance.min_streak_speed_dampen = float(entity_data.get('min_streak_speed_dampen', 500))  # Type: float
        instance.max_streak_speed_dampen = float(entity_data.get('max_streak_speed_dampen', 1000))  # Type: float


class prop_paint_bomb(Targetname, Angles):
    model = "models/props/futbol.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.PaintType = "CHOICES NOT SUPPORTED"  # Type: choices
        self.BombType = None  # Type: choices
        self.allowfunnel = 1  # Type: boolean
        self.AllowSilentDissolve = None  # Type: boolean
        self.playspawnsound = 1  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.PaintType = entity_data.get('painttype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.BombType = entity_data.get('bombtype', None)  # Type: choices
        instance.allowfunnel = entity_data.get('allowfunnel', None)  # Type: boolean
        instance.AllowSilentDissolve = entity_data.get('allowsilentdissolve', None)  # Type: boolean
        instance.playspawnsound = entity_data.get('playspawnsound', None)  # Type: boolean


class item_paint_power_pickup(Base):
    model = "models/items/healthkit.mdl"
    def __init__(self):
        super().__init__()
        self.origin = [0, 0, 0]
        self.PaintType = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.PaintType = entity_data.get('painttype', None)  # Type: choices


class trigger_paint_cleanser(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)


class weapon_paintgun(Base):
    model = "models/weapons/w_portalgun.mdl"
    def __init__(self):
        super().__init__()
        self.origin = [0, 0, 0]
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class point_survey(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.surveyname = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.surveyname = entity_data.get('surveyname', None)  # Type: string


class portal_race_checkpoint(Targetname, Angles):
    model = "models/effects/cappoint_hologram.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.ResetTime = 5.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.ResetTime = float(entity_data.get('resettime', 5.0))  # Type: float


class vgui_level_placard_display(Targetname, Angles, Parentname):
    model = "models/editor/axis_helper_thick.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class vgui_mp_lobby_display(Targetname, Angles, Parentname):
    model = "models/editor/axis_helper_thick.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class prop_exploding_futbol(BasePropPhysics):
    model = "models/props/futbol.mdl"
    def __init__(self):
        super(BasePropPhysics).__init__()
        self.origin = [0, 0, 0]
        self.SpawnerName = None  # Type: string
        self.ShouldRespawn = 0  # Type: boolean
        self.ExplodeOnTouch = 1  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePropPhysics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.SpawnerName = entity_data.get('spawnername', None)  # Type: string
        instance.ShouldRespawn = entity_data.get('shouldrespawn', None)  # Type: boolean
        instance.ExplodeOnTouch = entity_data.get('explodeontouch', None)  # Type: boolean


class prop_exploding_futbol_spawner(Targetname):
    model = "models/props/futbol_dispenser.mdl"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.StartWithFutbol = 1  # Type: boolean
        self.IsTimed = 0  # Type: boolean
        self.Timer = 0  # Type: float
        self.TimerIndicatorName = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.StartWithFutbol = entity_data.get('startwithfutbol', None)  # Type: boolean
        instance.IsTimed = entity_data.get('istimed', None)  # Type: boolean
        instance.Timer = float(entity_data.get('timer', 0))  # Type: float
        instance.TimerIndicatorName = entity_data.get('timerindicatorname', None)  # Type: target_destination


class prop_exploding_futbol_socket(Targetname):
    model = "models/props/futbol_socket.mdl"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class point_push(Targetname, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.enabled = 1  # Type: boolean
        self.magnitude = 100  # Type: float
        self.radius = 128  # Type: float
        self.inner_radius = 0  # Type: float
        self.influence_cone = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.enabled = entity_data.get('enabled', None)  # Type: boolean
        instance.magnitude = float(entity_data.get('magnitude', 100))  # Type: float
        instance.radius = float(entity_data.get('radius', 128))  # Type: float
        instance.inner_radius = float(entity_data.get('inner_radius', 0))  # Type: float
        instance.influence_cone = float(entity_data.get('influence_cone', 0))  # Type: float


class prop_physics_paintable(prop_physics):
    def __init__(self):
        super(prop_physics).__init__()
        self.origin = [0, 0, 0]
        self.PaintPower = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_physics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.PaintPower = entity_data.get('paintpower', "CHOICES NOT SUPPORTED")  # Type: choices


class logic_coop_manager(Targetname):
    icon_sprite = "editor/logic_coop_manager.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.DefaultPlayerStateA = None  # Type: choices
        self.DefaultPlayerStateB = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.DefaultPlayerStateA = entity_data.get('defaultplayerstatea', None)  # Type: choices
        instance.DefaultPlayerStateB = entity_data.get('defaultplayerstateb', None)  # Type: choices


class prop_testchamber_door(Targetname, Angles, Parentname):
    model = "models/props/portal_door_combined.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.lightingorigin = None  # Type: target_destination
        self.AreaPortalWindow = None  # Type: target_destination
        self.UseAreaPortalFade = None  # Type: boolean
        self.AreaPortalFadeStart = None  # Type: float
        self.AreaPortalFadeEnd = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.lightingorigin = entity_data.get('lightingorigin', None)  # Type: target_destination
        instance.AreaPortalWindow = entity_data.get('areaportalwindow', None)  # Type: target_destination
        instance.UseAreaPortalFade = entity_data.get('useareaportalfade', None)  # Type: boolean
        instance.AreaPortalFadeStart = float(entity_data.get('areaportalfadestart', 0))  # Type: float
        instance.AreaPortalFadeEnd = float(entity_data.get('areaportalfadeend', 0))  # Type: float


class npc_wheatley_boss(Parentname, BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Shadow).__init__()
        self.origin = [0, 0, 0]
        self.model = None  # Type: studio

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        BaseNPC.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', None)  # Type: studio


class paint_sphere(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.paint_type = None  # Type: choices
        self.radius = 60.0 # Type: float
        self.alpha_percent = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.paint_type = entity_data.get('paint_type', None)  # Type: choices
        instance.radius = float(entity_data.get('radius', 60.0))  # Type: float
        instance.alpha_percent = float(entity_data.get('alpha_percent', 1.0))  # Type: float



entity_class_handle = {
    'PaintableBrush': PaintableBrush,
    'Angles': Angles,
    'Origin': Origin,
    'Reflection': Reflection,
    'ToggleDraw': ToggleDraw,
    'Shadow': Shadow,
    'Studiomodel': Studiomodel,
    'BasePlat': BasePlat,
    'Targetname': Targetname,
    'Parentname': Parentname,
    'BaseBrush': BaseBrush,
    'EnableDisable': EnableDisable,
    'RenderFxChoices': RenderFxChoices,
    'SpatialEntity': SpatialEntity,
    'RenderFields': RenderFields,
    'SystemLevelChoice': SystemLevelChoice,
    'Inputfilter': Inputfilter,
    'Global': Global,
    'EnvGlobal': EnvGlobal,
    'DamageFilter': DamageFilter,
    'ResponseContext': ResponseContext,
    'Breakable': Breakable,
    'BreakableBrush': BreakableBrush,
    'BreakableProp': BreakableProp,
    'BaseNPC': BaseNPC,
    'info_npc_spawn_destination': info_npc_spawn_destination,
    'BaseNPCMaker': BaseNPCMaker,
    'npc_template_maker': npc_template_maker,
    'BaseHelicopter': BaseHelicopter,
    'PlayerClass': PlayerClass,
    'Light': Light,
    'Node': Node,
    'HintNode': HintNode,
    'TriggerOnce': TriggerOnce,
    'Trigger': Trigger,
    'worldbase': worldbase,
    'worldspawn': worldspawn,
    'ambient_generic': ambient_generic,
    'func_lod': func_lod,
    'env_zoom': env_zoom,
    'env_screenoverlay': env_screenoverlay,
    'env_screeneffect': env_screeneffect,
    'env_texturetoggle': env_texturetoggle,
    'env_splash': env_splash,
    'env_particlelight': env_particlelight,
    'env_sun': env_sun,
    'game_ragdoll_manager': game_ragdoll_manager,
    'game_gib_manager': game_gib_manager,
    'env_dof_controller': env_dof_controller,
    'env_lightglow': env_lightglow,
    'env_smokestack': env_smokestack,
    'env_fade': env_fade,
    'env_player_surface_trigger': env_player_surface_trigger,
    'env_tonemap_controller': env_tonemap_controller,
    'func_areaportalwindow': func_areaportalwindow,
    'func_wall': func_wall,
    'func_clip_vphysics': func_clip_vphysics,
    'func_brush': func_brush,
    'vgui_screen_base': vgui_screen_base,
    'vgui_screen': vgui_screen,
    'vgui_slideshow_display': vgui_slideshow_display,
    'vgui_movie_display': vgui_movie_display,
    'cycler': cycler,
    'gibshooterbase': gibshooterbase,
    'env_beam': env_beam,
    'env_embers': env_embers,
    'env_funnel': env_funnel,
    'env_blood': env_blood,
    'env_bubbles': env_bubbles,
    'env_explosion': env_explosion,
    'env_smoketrail': env_smoketrail,
    'env_physexplosion': env_physexplosion,
    'env_physimpact': env_physimpact,
    'env_fire': env_fire,
    'env_firesource': env_firesource,
    'env_firesensor': env_firesensor,
    'env_entity_igniter': env_entity_igniter,
    'env_fog_controller': env_fog_controller,
    'env_steam': env_steam,
    'env_laser': env_laser,
    'env_message': env_message,
    'env_hudhint': env_hudhint,
    'env_shake': env_shake,
    'env_tilt': env_tilt,
    'env_viewpunch': env_viewpunch,
    'gibshooter': gibshooter,
    'env_shooter': env_shooter,
    'env_rotorshooter': env_rotorshooter,
    'env_soundscape_proxy': env_soundscape_proxy,
    'env_soundscape': env_soundscape,
    'env_soundscape_triggerable': env_soundscape_triggerable,
    'env_spark': env_spark,
    'env_sprite': env_sprite,
    'env_sprite_clientside': env_sprite_clientside,
    'env_sprite_oriented': env_sprite_oriented,
    'env_wind': env_wind,
    'sky_camera': sky_camera,
    'BaseSpeaker': BaseSpeaker,
    'game_weapon_manager': game_weapon_manager,
    'game_end': game_end,
    'game_player_equip': game_player_equip,
    'game_player_team': game_player_team,
    'game_score': game_score,
    'game_text': game_text,
    'point_enable_motion_fixup': point_enable_motion_fixup,
    'point_message': point_message,
    'point_spotlight': point_spotlight,
    'point_tesla': point_tesla,
    'point_clientcommand': point_clientcommand,
    'point_servercommand': point_servercommand,
    'point_bonusmaps_accessor': point_bonusmaps_accessor,
    'game_ui': game_ui,
    'point_entity_finder': point_entity_finder,
    'game_zone_player': game_zone_player,
    'infodecal': infodecal,
    'info_projecteddecal': info_projecteddecal,
    'info_no_dynamic_shadow': info_no_dynamic_shadow,
    'info_player_start': info_player_start,
    'info_overlay': info_overlay,
    'info_overlay_transition': info_overlay_transition,
    'info_intermission': info_intermission,
    'info_landmark': info_landmark,
    'info_null': info_null,
    'info_target': info_target,
    'info_particle_system': info_particle_system,
    'phys_ragdollmagnet': phys_ragdollmagnet,
    'info_lighting': info_lighting,
    'info_teleport_destination': info_teleport_destination,
    'info_node': info_node,
    'info_node_hint': info_node_hint,
    'info_node_air': info_node_air,
    'info_node_air_hint': info_node_air_hint,
    'info_hint': info_hint,
    'info_node_link': info_node_link,
    'info_node_link_controller': info_node_link_controller,
    'info_radial_link_controller': info_radial_link_controller,
    'info_node_climb': info_node_climb,
    'light': light,
    'light_environment': light_environment,
    'light_spot': light_spot,
    'light_dynamic': light_dynamic,
    'shadow_control': shadow_control,
    'sunlight_shadow_control': sunlight_shadow_control,
    'env_ambient_light': env_ambient_light,
    'color_correction': color_correction,
    'color_correction_volume': color_correction_volume,
    'KeyFrame': KeyFrame,
    'Mover': Mover,
    'func_movelinear': func_movelinear,
    'func_water_analog': func_water_analog,
    'func_rotating': func_rotating,
    'func_platrot': func_platrot,
    'keyframe_track': keyframe_track,
    'move_keyframed': move_keyframed,
    'move_track': move_track,
    'RopeKeyFrame': RopeKeyFrame,
    'keyframe_rope': keyframe_rope,
    'move_rope': move_rope,
    'Button': Button,
    'func_button': func_button,
    'func_rot_button': func_rot_button,
    'momentary_rot_button': momentary_rot_button,
    'Door': Door,
    'func_door': func_door,
    'func_door_rotating': func_door_rotating,
    'prop_door_rotating': prop_door_rotating,
    'env_cubemap': env_cubemap,
    'BModelParticleSpawner': BModelParticleSpawner,
    'func_dustmotes': func_dustmotes,
    'func_smokevolume': func_smokevolume,
    'func_dustcloud': func_dustcloud,
    'env_dustpuff': env_dustpuff,
    'env_particlescript': env_particlescript,
    'env_effectscript': env_effectscript,
    'logic_auto': logic_auto,
    'point_viewcontrol': point_viewcontrol,
    'point_viewcontrol_multiplayer': point_viewcontrol_multiplayer,
    'point_viewproxy': point_viewproxy,
    'point_posecontroller': point_posecontroller,
    'logic_compare': logic_compare,
    'logic_branch': logic_branch,
    'logic_branch_listener': logic_branch_listener,
    'logic_case': logic_case,
    'logic_multicompare': logic_multicompare,
    'logic_relay': logic_relay,
    'logic_register_activator': logic_register_activator,
    'logic_random_outputs': logic_random_outputs,
    'logic_script': logic_script,
    'logic_timer': logic_timer,
    'hammer_updateignorelist': hammer_updateignorelist,
    'logic_collision_pair': logic_collision_pair,
    'env_microphone': env_microphone,
    'math_remap': math_remap,
    'math_colorblend': math_colorblend,
    'math_counter': math_counter,
    'logic_lineto': logic_lineto,
    'logic_navigation': logic_navigation,
    'logic_autosave': logic_autosave,
    'logic_active_autosave': logic_active_autosave,
    'logic_playmovie': logic_playmovie,
    'point_template': point_template,
    'env_entity_maker': env_entity_maker,
    'BaseFilter': BaseFilter,
    'filter_multi': filter_multi,
    'filter_activator_name': filter_activator_name,
    'filter_activator_model': filter_activator_model,
    'filter_activator_context': filter_activator_context,
    'filter_activator_class': filter_activator_class,
    'filter_activator_mass_greater': filter_activator_mass_greater,
    'filter_damage_type': filter_damage_type,
    'filter_enemy': filter_enemy,
    'point_anglesensor': point_anglesensor,
    'point_angularvelocitysensor': point_angularvelocitysensor,
    'point_velocitysensor': point_velocitysensor,
    'point_proximity_sensor': point_proximity_sensor,
    'point_teleport': point_teleport,
    'point_hurt': point_hurt,
    'point_playermoveconstraint': point_playermoveconstraint,
    'func_physbox': func_physbox,
    'TwoObjectPhysics': TwoObjectPhysics,
    'phys_constraintsystem': phys_constraintsystem,
    'phys_keepupright': phys_keepupright,
    'physics_cannister': physics_cannister,
    'info_constraint_anchor': info_constraint_anchor,
    'info_mass_center': info_mass_center,
    'phys_spring': phys_spring,
    'phys_hinge': phys_hinge,
    'phys_ballsocket': phys_ballsocket,
    'phys_constraint': phys_constraint,
    'phys_pulleyconstraint': phys_pulleyconstraint,
    'phys_slideconstraint': phys_slideconstraint,
    'phys_lengthconstraint': phys_lengthconstraint,
    'phys_ragdollconstraint': phys_ragdollconstraint,
    'phys_convert': phys_convert,
    'ForceController': ForceController,
    'phys_thruster': phys_thruster,
    'phys_torque': phys_torque,
    'phys_motor': phys_motor,
    'phys_magnet': phys_magnet,
    'prop_detail_base': prop_detail_base,
    'prop_static_base': prop_static_base,
    'BaseFadeProp': BaseFadeProp,
    'prop_dynamic_base': prop_dynamic_base,
    'prop_detail': prop_detail,
    'prop_static': prop_static,
    'prop_dynamic': prop_dynamic,
    'prop_dynamic_override': prop_dynamic_override,
    'BasePropPhysics': BasePropPhysics,
    'prop_physics_override': prop_physics_override,
    'prop_physics': prop_physics,
    'prop_physics_multiplayer': prop_physics_multiplayer,
    'prop_ragdoll': prop_ragdoll,
    'prop_dynamic_ornament': prop_dynamic_ornament,
    'func_areaportal': func_areaportal,
    'func_occluder': func_occluder,
    'func_breakable': func_breakable,
    'func_breakable_surf': func_breakable_surf,
    'func_conveyor': func_conveyor,
    'func_detail': func_detail,
    'func_viscluster': func_viscluster,
    'func_illusionary': func_illusionary,
    'func_precipitation': func_precipitation,
    'func_wall_toggle': func_wall_toggle,
    'func_guntarget': func_guntarget,
    'PlatSounds': PlatSounds,
    'Trackchange': Trackchange,
    'BaseTrain': BaseTrain,
    'func_trackautochange': func_trackautochange,
    'func_trackchange': func_trackchange,
    'func_tracktrain': func_tracktrain,
    'func_tanktrain': func_tanktrain,
    'func_traincontrols': func_traincontrols,
    'tanktrain_aitarget': tanktrain_aitarget,
    'tanktrain_ai': tanktrain_ai,
    'path_track': path_track,
    'test_traceline': test_traceline,
    'trigger_autosave': trigger_autosave,
    'trigger_changelevel': trigger_changelevel,
    'trigger_gravity': trigger_gravity,
    'trigger_playermovement': trigger_playermovement,
    'trigger_soundscape': trigger_soundscape,
    'trigger_hurt': trigger_hurt,
    'trigger_remove': trigger_remove,
    'trigger_multiple': trigger_multiple,
    'trigger_once': trigger_once,
    'trigger_look': trigger_look,
    'trigger_push': trigger_push,
    'trigger_wind': trigger_wind,
    'trigger_hierarchy': trigger_hierarchy,
    'trigger_impact': trigger_impact,
    'trigger_proximity': trigger_proximity,
    'trigger_teleport': trigger_teleport,
    'trigger_transition': trigger_transition,
    'trigger_serverragdoll': trigger_serverragdoll,
    'ai_speechfilter': ai_speechfilter,
    'ai_addon': ai_addon,
    'ai_addon_builder': ai_addon_builder,
    'water_lod_control': water_lod_control,
    'info_camera_link': info_camera_link,
    'logic_eventlistener': logic_eventlistener,
    'logic_measure_movement': logic_measure_movement,
    'npc_furniture': npc_furniture,
    'env_credits': env_credits,
    'material_modify_control': material_modify_control,
    'point_devshot_camera': point_devshot_camera,
    'logic_playerproxy': logic_playerproxy,
    'env_spritetrail': env_spritetrail,
    'env_projectedtexture': env_projectedtexture,
    'func_reflective_glass': func_reflective_glass,
    'env_particle_performance_monitor': env_particle_performance_monitor,
    'npc_puppet': npc_puppet,
    'point_gamestats_counter': point_gamestats_counter,
    'beam_spotlight': beam_spotlight,
    'func_instance': func_instance,
    'func_instance_parms': func_instance_parms,
    'func_instance_io_proxy': func_instance_io_proxy,
    'func_instance_origin': func_instance_origin,
    'env_instructor_hint': env_instructor_hint,
    'info_target_instructor_hint': info_target_instructor_hint,
    'info_game_event_proxy': info_game_event_proxy,
    'light_directional': light_directional,
    'postprocess_controller': postprocess_controller,
    'fog_volume': fog_volume,
    'TalkNPC': TalkNPC,
    'PlayerCompanion': PlayerCompanion,
    'RappelNPC': RappelNPC,
    'AlyxInteractable': AlyxInteractable,
    'CombineBallSpawners': CombineBallSpawners,
    'prop_combine_ball': prop_combine_ball,
    'trigger_physics_trap': trigger_physics_trap,
    'trigger_weapon_strip': trigger_weapon_strip,
    'func_combine_ball_spawner': func_combine_ball_spawner,
    'point_combine_ball_launcher': point_combine_ball_launcher,
    'npc_turret_ground': npc_turret_ground,
    'npc_turret_floor': npc_turret_floor,
    'npc_bullseye': npc_bullseye,
    'npc_enemyfinder': npc_enemyfinder,
    'npc_spotlight': npc_spotlight,
    'monster_generic': monster_generic,
    'generic_actor': generic_actor,
    'cycler_actor': cycler_actor,
    'npc_maker': npc_maker,
    'player_control': player_control,
    'BaseScripted': BaseScripted,
    'scripted_sentence': scripted_sentence,
    'scripted_target': scripted_target,
    'ai_relationship': ai_relationship,
    'ai_ally_manager': ai_ally_manager,
    'LeadGoalBase': LeadGoalBase,
    'ai_goal_lead': ai_goal_lead,
    'ai_goal_lead_weapon': ai_goal_lead_weapon,
    'FollowGoal': FollowGoal,
    'ai_goal_follow': ai_goal_follow,
    'ai_goal_injured_follow': ai_goal_injured_follow,
    'point_energy_ball_launcher': point_energy_ball_launcher,
    'npc_rocket_turret': npc_rocket_turret,
    'env_portal_path_track': env_portal_path_track,
    'trigger_portal_cleanser': trigger_portal_cleanser,
    'func_portal_orientation': func_portal_orientation,
    'func_weight_button': func_weight_button,
    'func_noportal_volume': func_noportal_volume,
    'func_portal_bumper': func_portal_bumper,
    'func_portal_detector': func_portal_detector,
    'PortalBase': PortalBase,
    'prop_portal': prop_portal,
    'weapon_portalgun': weapon_portalgun,
    'npc_portal_turret_ground': npc_portal_turret_ground,
    'prop_glados_core': prop_glados_core,
    'npc_portal_turret_floor': npc_portal_turret_floor,
    'npc_security_camera': npc_security_camera,
    'prop_telescopic_arm': prop_telescopic_arm,
    'prop_portal_stats_display': prop_portal_stats_display,
    'vgui_neurotoxin_countdown': vgui_neurotoxin_countdown,
    'env_lightrail_endpoint': env_lightrail_endpoint,
    'env_portal_credits': env_portal_credits,
    'info_lighting_relative': info_lighting_relative,
    'prop_mirror': prop_mirror,
    'point_futbol_shooter': point_futbol_shooter,
    'LinkedPortalDoor': LinkedPortalDoor,
    'BaseProjector': BaseProjector,
    'info_target_personality_sphere': info_target_personality_sphere,
    'prop_rocket_tripwire': prop_rocket_tripwire,
    'func_camera_target': func_camera_target,
    'linked_portal_door': linked_portal_door,
    'prop_linked_portal_door': prop_linked_portal_door,
    'prop_button': prop_button,
    'prop_under_button': prop_under_button,
    'prop_floor_button': prop_floor_button,
    'prop_floor_cube_button': prop_floor_cube_button,
    'prop_floor_ball_button': prop_floor_ball_button,
    'prop_under_floor_button': prop_under_floor_button,
    'prop_wall_projector': prop_wall_projector,
    'info_placement_helper': info_placement_helper,
    'info_player_ping_detector': info_player_ping_detector,
    'func_placement_clip': func_placement_clip,
    'filter_player_held': filter_player_held,
    'env_portal_laser': env_portal_laser,
    'point_laser_target': point_laser_target,
    'prop_laser_catcher': prop_laser_catcher,
    'prop_laser_relay': prop_laser_relay,
    'prop_weighted_cube': prop_weighted_cube,
    'trigger_catapult': trigger_catapult,
    'prop_glass_futbol': prop_glass_futbol,
    'prop_glass_futbol_spawner': prop_glass_futbol_spawner,
    'prop_glass_futbol_socket': prop_glass_futbol_socket,
    'portalmp_gamerules': portalmp_gamerules,
    'item_nugget': item_nugget,
    'func_portalled': func_portalled,
    'logic_player_slowtime': logic_player_slowtime,
    'logic_timescale': logic_timescale,
    'env_player_viewfinder': env_player_viewfinder,
    'info_coop_spawn': info_coop_spawn,
    'npc_personality_core': npc_personality_core,
    'prop_monster_box': prop_monster_box,
    'prop_indicator_panel': prop_indicator_panel,
    'prop_tic_tac_toe_panel': prop_tic_tac_toe_panel,
    'trigger_playerteam': trigger_playerteam,
    'trigger_ping_detector': trigger_ping_detector,
    'info_landmark_entry': info_landmark_entry,
    'info_landmark_exit': info_landmark_exit,
    'point_changelevel': point_changelevel,
    'prop_tractor_beam': prop_tractor_beam,
    'info_paint_sprayer': info_paint_sprayer,
    'prop_paint_bomb': prop_paint_bomb,
    'item_paint_power_pickup': item_paint_power_pickup,
    'trigger_paint_cleanser': trigger_paint_cleanser,
    'weapon_paintgun': weapon_paintgun,
    'point_survey': point_survey,
    'portal_race_checkpoint': portal_race_checkpoint,
    'vgui_level_placard_display': vgui_level_placard_display,
    'vgui_mp_lobby_display': vgui_mp_lobby_display,
    'prop_exploding_futbol': prop_exploding_futbol,
    'prop_exploding_futbol_spawner': prop_exploding_futbol_spawner,
    'prop_exploding_futbol_socket': prop_exploding_futbol_socket,
    'point_push': point_push,
    'prop_physics_paintable': prop_physics_paintable,
    'logic_coop_manager': logic_coop_manager,
    'prop_testchamber_door': prop_testchamber_door,
    'npc_wheatley_boss': npc_wheatley_boss,
    'paint_sphere': paint_sphere,
}