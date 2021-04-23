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


class Studiomodel(Base):
    def __init__(self):
        super().__init__()
        self.model = None  # Type: studio
        self.skin = None  # Type: integer
        self.body = None  # Type: integer
        self.disableshadows = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', None)  # Type: studio
        instance.skin = parse_source_value(entity_data.get('skin', 0))  # Type: integer
        instance.body = parse_source_value(entity_data.get('body', 0))  # Type: integer
        instance.disableshadows = entity_data.get('disableshadows', None)  # Type: choices


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
        self.StartDisabled = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.StartDisabled = entity_data.get('startdisabled', None)  # Type: choices


class RenderFxChoices(Base):
    def __init__(self):
        super().__init__()
        self.renderfx = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.renderfx = entity_data.get('renderfx', None)  # Type: choices


class Shadow(Base):
    def __init__(self):
        super().__init__()
        self.disableshadows = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.disableshadows = entity_data.get('disableshadows', None)  # Type: choices


class Glow(Base):
    def __init__(self):
        super().__init__()
        self.glowstate = None  # Type: choices
        self.glowrange = None  # Type: integer
        self.glowrangemin = None  # Type: integer
        self.glowcolor = [0, 0, 0]  # Type: color255

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.glowstate = entity_data.get('glowstate', None)  # Type: choices
        instance.glowrange = parse_source_value(entity_data.get('glowrange', 0))  # Type: integer
        instance.glowrangemin = parse_source_value(entity_data.get('glowrangemin', 0))  # Type: integer
        instance.glowcolor = parse_int_vector(entity_data.get('glowcolor', "0 0 0"))  # Type: color255


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


class RenderFields(SystemLevelChoice, RenderFxChoices):
    def __init__(self):
        super(SystemLevelChoice).__init__()
        super(RenderFxChoices).__init__()
        self.rendermode = None  # Type: choices
        self.renderamt = 255  # Type: integer
        self.rendercolor = [255, 255, 255]  # Type: color255
        self.disablereceiveshadows = None  # Type: choices
        self.fademindist = -1  # Type: float
        self.fademaxdist = None  # Type: float
        self.fadescale = 1  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        SystemLevelChoice.from_dict(instance, entity_data)
        RenderFxChoices.from_dict(instance, entity_data)
        instance.rendermode = entity_data.get('rendermode', None)  # Type: choices
        instance.renderamt = parse_source_value(entity_data.get('renderamt', 255))  # Type: integer
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "255 255 255"))  # Type: color255
        instance.disablereceiveshadows = entity_data.get('disablereceiveshadows', None)  # Type: choices
        instance.fademindist = float(entity_data.get('fademindist', -1))  # Type: float
        instance.fademaxdist = float(entity_data.get('fademaxdist', 0))  # Type: float
        instance.fadescale = float(entity_data.get('fadescale', 1))  # Type: float


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


class Breakable(Targetname, Shadow, DamageFilter):
    def __init__(self):
        super(Targetname).__init__()
        super(Shadow).__init__()
        super(DamageFilter).__init__()
        self.ExplodeDamage = None  # Type: float
        self.ExplodeRadius = None  # Type: float
        self.PerformanceMode = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        DamageFilter.from_dict(instance, entity_data)
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
        self.nodamageforces = None  # Type: choices
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
        instance.nodamageforces = entity_data.get('nodamageforces', None)  # Type: choices
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


class BaseNPC(RenderFields, Angles, ResponseContext, DamageFilter, Shadow, Targetname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Angles).__init__()
        super(ResponseContext).__init__()
        super(DamageFilter).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        self.target = None  # Type: target_destination
        self.squadname = None  # Type: string
        self.hintgroup = None  # Type: string
        self.hintlimiting = None  # Type: choices
        self.sleepstate = None  # Type: choices
        self.wakeradius = None  # Type: float
        self.wakesquad = None  # Type: choices
        self.enemyfilter = None  # Type: target_destination
        self.ignoreunseenenemies = None  # Type: choices
        self.physdamagescale = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        RenderFields.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        ResponseContext.from_dict(instance, entity_data)
        DamageFilter.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.squadname = entity_data.get('squadname', None)  # Type: string
        instance.hintgroup = entity_data.get('hintgroup', None)  # Type: string
        instance.hintlimiting = entity_data.get('hintlimiting', None)  # Type: choices
        instance.sleepstate = entity_data.get('sleepstate', None)  # Type: choices
        instance.wakeradius = float(entity_data.get('wakeradius', 0))  # Type: float
        instance.wakesquad = entity_data.get('wakesquad', None)  # Type: choices
        instance.enemyfilter = entity_data.get('enemyfilter', None)  # Type: target_destination
        instance.ignoreunseenenemies = entity_data.get('ignoreunseenenemies', None)  # Type: choices
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


class BaseNPCMaker(Targetname, EnableDisable, Angles):
    icon_sprite = "editor/npc_maker.vmt"

    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(Angles).__init__()
        self.MaxNPCCount = 1  # Type: integer
        self.SpawnFrequency = "5"  # Type: string
        self.MaxLiveChildren = 5  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.MaxNPCCount = parse_source_value(entity_data.get('maxnpccount', 1))  # Type: integer
        instance.SpawnFrequency = entity_data.get('spawnfrequency', "5")  # Type: string
        instance.MaxLiveChildren = parse_source_value(entity_data.get('maxlivechildren', 5))  # Type: integer


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
        self._lightscaleHDR = 0.5  # Type: float
        self.style = None  # Type: choices
        self.pattern = None  # Type: string
        self._constant_attn = "0"  # Type: string
        self._linear_attn = "0"  # Type: string
        self._quadratic_attn = "1"  # Type: string
        self._fifty_percent_distance = "0"  # Type: string
        self._zero_percent_distance = "0"  # Type: string
        self._hardfalloff = None  # Type: integer
        self._castentityshadow = "CHOICES NOT SUPPORTED"  # Type: choices
        self._shadoworiginoffset = [0.0, 0.0, 0.0]  # Type: vector

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance._light = parse_int_vector(entity_data.get('_light', "255 255 255 200"))  # Type: color255
        instance._lightHDR = parse_int_vector(entity_data.get('_lighthdr', "-1 -1 -1 1"))  # Type: color255
        instance._lightscaleHDR = float(entity_data.get('_lightscalehdr', 0.5))  # Type: float
        instance.style = entity_data.get('style', None)  # Type: choices
        instance.pattern = entity_data.get('pattern', None)  # Type: string
        instance._constant_attn = entity_data.get('_constant_attn', "0")  # Type: string
        instance._linear_attn = entity_data.get('_linear_attn', "0")  # Type: string
        instance._quadratic_attn = entity_data.get('_quadratic_attn', "1")  # Type: string
        instance._fifty_percent_distance = entity_data.get('_fifty_percent_distance', "0")  # Type: string
        instance._zero_percent_distance = entity_data.get('_zero_percent_distance', "0")  # Type: string
        instance._hardfalloff = parse_source_value(entity_data.get('_hardfalloff', 0))  # Type: integer
        instance._castentityshadow = entity_data.get('_castentityshadow', "CHOICES NOT SUPPORTED")  # Type: choices
        instance._shadoworiginoffset = parse_float_vector(
            entity_data.get('_shadoworiginoffset', "0 0 0"))  # Type: vector


class Node(Base):
    def __init__(self):
        super().__init__()
        self.nodeid = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.nodeid = parse_source_value(entity_data.get('nodeid', 0))  # Type: integer


class HintNode(Node):
    def __init__(self):
        super(Node).__init__()
        self.hinttype = None  # Type: choices
        self.hintactivity = None  # Type: string
        self.nodeFOV = "CHOICES NOT SUPPORTED"  # Type: choices
        self.StartHintDisabled = None  # Type: choices
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
        instance.hintactivity = entity_data.get('hintactivity', None)  # Type: string
        instance.nodeFOV = entity_data.get('nodefov', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.StartHintDisabled = entity_data.get('starthintdisabled', None)  # Type: choices
        instance.Group = entity_data.get('group', None)  # Type: string
        instance.TargetNode = parse_source_value(entity_data.get('targetnode', -1))  # Type: node_dest
        instance.radius = parse_source_value(entity_data.get('radius', 0))  # Type: integer
        instance.IgnoreFacing = entity_data.get('ignorefacing', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.MinimumState = entity_data.get('minimumstate', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.MaximumState = entity_data.get('maximumstate', "CHOICES NOT SUPPORTED")  # Type: choices


class TriggerOnce(Parentname, Origin, EnableDisable, Targetname, Global):
    def __init__(self):
        super(Parentname).__init__()
        super(Origin).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
        self.filtername = None  # Type: filterclass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        instance.filtername = entity_data.get('filtername', None)  # Type: filterclass


class Trigger(TriggerOnce):
    def __init__(self):
        super(TriggerOnce).__init__()
        super(Parentname).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        super(EnableDisable).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        TriggerOnce.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)


class worldbase(Base):
    def __init__(self):
        super().__init__()
        self.message = None  # Type: string
        self.skyname = "sky_l4d_rural02_hdr"  # Type: string
        self.chaptertitle = None  # Type: string
        self.startdark = None  # Type: choices
        self.gametitle = None  # Type: choices
        self.newunit = None  # Type: choices
        self.maxoccludeearea = 0  # Type: float
        self.minoccluderarea = 0  # Type: float
        self.maxoccludeearea_x360 = 0  # Type: float
        self.minoccluderarea_x360 = 0  # Type: float
        self.maxpropscreenwidth = -1  # Type: float
        self.minpropscreenwidth = None  # Type: float
        self.detailvbsp = "detail.vbsp"  # Type: string
        self.detailmaterial = "detail/detailsprites"  # Type: string
        self.coldworld = None  # Type: choices
        self.timeofday = None  # Type: choices
        self.startmusictype = None  # Type: choices
        self.musicpostfix = "Waterfront"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.message = entity_data.get('message', None)  # Type: string
        instance.skyname = entity_data.get('skyname', "sky_l4d_rural02_hdr")  # Type: string
        instance.chaptertitle = entity_data.get('chaptertitle', None)  # Type: string
        instance.startdark = entity_data.get('startdark', None)  # Type: choices
        instance.gametitle = entity_data.get('gametitle', None)  # Type: choices
        instance.newunit = entity_data.get('newunit', None)  # Type: choices
        instance.maxoccludeearea = float(entity_data.get('maxoccludeearea', 0))  # Type: float
        instance.minoccluderarea = float(entity_data.get('minoccluderarea', 0))  # Type: float
        instance.maxoccludeearea_x360 = float(entity_data.get('maxoccludeearea_x360', 0))  # Type: float
        instance.minoccluderarea_x360 = float(entity_data.get('minoccluderarea_x360', 0))  # Type: float
        instance.maxpropscreenwidth = float(entity_data.get('maxpropscreenwidth', -1))  # Type: float
        instance.minpropscreenwidth = float(entity_data.get('minpropscreenwidth', 0))  # Type: float
        instance.detailvbsp = entity_data.get('detailvbsp', "detail.vbsp")  # Type: string
        instance.detailmaterial = entity_data.get('detailmaterial', "detail/detailsprites")  # Type: string
        instance.coldworld = entity_data.get('coldworld', None)  # Type: choices
        instance.timeofday = entity_data.get('timeofday', None)  # Type: choices
        instance.startmusictype = entity_data.get('startmusictype', None)  # Type: choices
        instance.musicpostfix = entity_data.get('musicpostfix', "Waterfront")  # Type: string


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


class ambient_music(Targetname):
    icon_sprite = "editor/ambient_generic.vmt"

    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.message = None  # Type: sound

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.message = entity_data.get('message', None)  # Type: sound


class sound_mix_layer(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.MixLayerName = None  # Type: string
        self.Level = 0.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.MixLayerName = entity_data.get('mixlayername', None)  # Type: string
        instance.Level = float(entity_data.get('level', 0.0))  # Type: float


class func_lod(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.DisappearMinDist = 2000  # Type: integer
        self.DisappearMaxDist = 2200  # Type: integer
        self.Solid = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.DisappearMinDist = parse_source_value(entity_data.get('disappearmindist', 2000))  # Type: integer
        instance.DisappearMaxDist = parse_source_value(entity_data.get('disappearmaxdist', 2200))  # Type: integer
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
        self.directional = None  # Type: choices
        self.PSName = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Color = parse_int_vector(entity_data.get('color', "255 0 0"))  # Type: color255
        instance.Intensity = parse_source_value(entity_data.get('intensity', 5000))  # Type: integer
        instance.directional = entity_data.get('directional', None)  # Type: choices
        instance.PSName = entity_data.get('psname', None)  # Type: string


class env_sun(Targetname, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.use_angles = None  # Type: choices
        self.pitch = None  # Type: integer
        self.rendercolor = [100, 80, 80]  # Type: color255
        self.overlaycolor = [0, 0, 0]  # Type: color255
        self.size = 16  # Type: integer
        self.overlaysize = -1  # Type: integer
        self.material = "sprites/light_glow02_add_noz"  # Type: sprite
        self.overlaymaterial = "sprites/light_glow02_add_noz"  # Type: sprite
        self.HDRColorScale = 0.5  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.use_angles = entity_data.get('use_angles', None)  # Type: choices
        instance.pitch = parse_source_value(entity_data.get('pitch', 0))  # Type: integer
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "100 80 80"))  # Type: color255
        instance.overlaycolor = parse_int_vector(entity_data.get('overlaycolor', "0 0 0"))  # Type: color255
        instance.size = parse_source_value(entity_data.get('size', 16))  # Type: integer
        instance.overlaysize = parse_source_value(entity_data.get('overlaysize', -1))  # Type: integer
        instance.material = entity_data.get('material', "sprites/light_glow02_add_noz")  # Type: sprite
        instance.overlaymaterial = entity_data.get('overlaymaterial', "sprites/light_glow02_add_noz")  # Type: sprite
        instance.HDRColorScale = float(entity_data.get('hdrcolorscale', 0.5))  # Type: float


class game_ragdoll_manager(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.MaxRagdollCount = -1  # Type: integer
        self.MaxRagdollCountDX8 = -1  # Type: integer
        self.SaveImportant = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.MaxRagdollCount = parse_source_value(entity_data.get('maxragdollcount', -1))  # Type: integer
        instance.MaxRagdollCountDX8 = parse_source_value(entity_data.get('maxragdollcountdx8', -1))  # Type: integer
        instance.SaveImportant = entity_data.get('saveimportant', None)  # Type: choices


class game_gib_manager(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.maxpieces = -1  # Type: integer
        self.maxpiecesdx8 = -1  # Type: integer
        self.allownewgibs = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.maxpieces = parse_source_value(entity_data.get('maxpieces', -1))  # Type: integer
        instance.maxpiecesdx8 = parse_source_value(entity_data.get('maxpiecesdx8', -1))  # Type: integer
        instance.allownewgibs = entity_data.get('allownewgibs', None)  # Type: choices


class env_dof_controller(Targetname):
    icon_sprite = "editor/env_dof_controller.vmt"

    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


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
        self.HDRColorScale = 0.5  # Type: float

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
        instance.HDRColorScale = float(entity_data.get('hdrcolorscale', 0.5))  # Type: float


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

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.duration = entity_data.get('duration', "2")  # Type: string
        instance.holdtime = entity_data.get('holdtime', "0")  # Type: string
        instance.renderamt = parse_source_value(entity_data.get('renderamt', 255))  # Type: integer
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "0 0 0"))  # Type: color255


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


class trigger_tonemap(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.TonemapName = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.TonemapName = entity_data.get('tonemapname', None)  # Type: target_destination


class env_tonemap_controller(Targetname):
    icon_sprite = "editor/env_tonemap_controller.vmt"

    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_tonemap_controller_infected(env_tonemap_controller):
    icon_sprite = "editor/env_tonemap_controller.vmt"

    def __init__(self):
        super(env_tonemap_controller).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        env_tonemap_controller.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_tonemap_controller_ghost(env_tonemap_controller):
    icon_sprite = "editor/env_tonemap_controller.vmt"

    def __init__(self):
        super(env_tonemap_controller).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        env_tonemap_controller.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class func_useableladder(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.point0 = None  # Type: vector
        self.point1 = None  # Type: vector
        self.StartDisabled = None  # Type: choices
        self.ladderSurfaceProperties = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.point0 = parse_float_vector(entity_data.get('point0', "0 0 0"))  # Type: vector
        instance.point1 = parse_float_vector(entity_data.get('point1', "0 0 0"))  # Type: vector
        instance.StartDisabled = entity_data.get('startdisabled', None)  # Type: choices
        instance.ladderSurfaceProperties = entity_data.get('laddersurfaceproperties', None)  # Type: string


class func_ladderendpoint(Targetname, Angles, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination


class info_ladder_dismount(Parentname):
    def __init__(self):
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination


class func_areaportalwindow(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.target = None  # Type: target_destination
        self.FadeStartDist = 128  # Type: integer
        self.FadeDist = 512  # Type: integer
        self.TranslucencyLimit = "0"  # Type: string
        self.BackgroundBModel = None  # Type: string
        self.PortalVersion = 1  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.FadeStartDist = parse_source_value(entity_data.get('fadestartdist', 128))  # Type: integer
        instance.FadeDist = parse_source_value(entity_data.get('fadedist', 512))  # Type: integer
        instance.TranslucencyLimit = entity_data.get('translucencylimit', "0")  # Type: string
        instance.BackgroundBModel = entity_data.get('backgroundbmodel', None)  # Type: string
        instance.PortalVersion = parse_source_value(entity_data.get('portalversion', 1))  # Type: integer


class func_wall(Targetname, RenderFields, Global, Shadow):
    def __init__(self):
        super(RenderFields).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
        super(Shadow).__init__()
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
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


class func_brush(Parentname, RenderFields, Origin, Inputfilter, EnableDisable, Shadow, Targetname, Global):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Origin).__init__()
        super(Inputfilter).__init__()
        super(EnableDisable).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
        self._minlight = None  # Type: string
        self.Solidity = "CHOICES NOT SUPPORTED"  # Type: choices
        self.excludednpc = None  # Type: string
        self.invert_exclusion = None  # Type: choices
        self.solidbsp = None  # Type: choices
        self.vrad_brush_cast_shadows = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Inputfilter.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.Solidity = entity_data.get('solidity', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.excludednpc = entity_data.get('excludednpc', None)  # Type: string
        instance.invert_exclusion = entity_data.get('invert_exclusion', None)  # Type: choices
        instance.solidbsp = entity_data.get('solidbsp', None)  # Type: choices
        instance.vrad_brush_cast_shadows = entity_data.get('vrad_brush_cast_shadows', None)  # Type: choices


class vgui_screen_base(Targetname, Angles, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.panelname = None  # Type: string
        self.overlaymaterial = None  # Type: string
        self.width = 32  # Type: integer
        self.height = 32  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.panelname = entity_data.get('panelname', None)  # Type: string
        instance.overlaymaterial = entity_data.get('overlaymaterial', None)  # Type: string
        instance.width = parse_source_value(entity_data.get('width', 32))  # Type: integer
        instance.height = parse_source_value(entity_data.get('height', 32))  # Type: integer


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


class cycler(Parentname, RenderFields, Angles, Targetname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(RenderFxChoices).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.model = None  # Type: studio
        self.skin = None  # Type: integer
        self.sequence = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        RenderFxChoices.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', None)  # Type: studio
        instance.skin = parse_source_value(entity_data.get('skin', 0))  # Type: integer
        instance.sequence = parse_source_value(entity_data.get('sequence', 0))  # Type: integer


class func_orator(Parentname, RenderFields, Angles, Studiomodel, Targetname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Studiomodel).__init__()
        super(RenderFxChoices).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.maxThenAnyDispatchDist = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        RenderFxChoices.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.maxThenAnyDispatchDist = float(entity_data.get('maxthenanydispatchdist', 0))  # Type: float


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


class env_beam(Targetname, RenderFxChoices, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(RenderFxChoices).__init__()
        super(Parentname).__init__()
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
        self.TouchType = None  # Type: choices
        self.filtername = None  # Type: filterclass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        RenderFxChoices.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
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
        instance.TouchType = entity_data.get('touchtype', None)  # Type: choices
        instance.filtername = entity_data.get('filtername', None)  # Type: filterclass


class env_beverage(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.health = 10  # Type: integer
        self.beveragetype = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.health = parse_source_value(entity_data.get('health', 10))  # Type: integer
        instance.beveragetype = entity_data.get('beveragetype', None)  # Type: choices


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
        self.ignoredClass = None  # Type: integer

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
        instance.ignoredClass = parse_source_value(entity_data.get('ignoredclass', 0))  # Type: integer


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


class env_fire(Targetname, EnableDisable, Parentname):
    icon_sprite = "editor/env_fire"

    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(Parentname).__init__()
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
        EnableDisable.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
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
        self.fogenable = None  # Type: choices
        self.fogblend = None  # Type: choices
        self.use_angles = None  # Type: choices
        self.fogcolor = [255, 255, 255]  # Type: color255
        self.fogcolor2 = [255, 255, 255]  # Type: color255
        self.fogdir = "1 0 0"  # Type: string
        self.fogstart = "500.0"  # Type: string
        self.fogend = "2000.0"  # Type: string
        self.fogmaxdensity = 1  # Type: float
        self.foglerptime = 0  # Type: float
        self.farz = "-1"  # Type: string
        self.HDRColorScale = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        SystemLevelChoice.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.fogenable = entity_data.get('fogenable', None)  # Type: choices
        instance.fogblend = entity_data.get('fogblend', None)  # Type: choices
        instance.use_angles = entity_data.get('use_angles', None)  # Type: choices
        instance.fogcolor = parse_int_vector(entity_data.get('fogcolor', "255 255 255"))  # Type: color255
        instance.fogcolor2 = parse_int_vector(entity_data.get('fogcolor2', "255 255 255"))  # Type: color255
        instance.fogdir = entity_data.get('fogdir', "1 0 0")  # Type: string
        instance.fogstart = entity_data.get('fogstart', "500.0")  # Type: string
        instance.fogend = entity_data.get('fogend', "2000.0")  # Type: string
        instance.fogmaxdensity = float(entity_data.get('fogmaxdensity', 1))  # Type: float
        instance.foglerptime = float(entity_data.get('foglerptime', 0))  # Type: float
        instance.farz = entity_data.get('farz', "-1")  # Type: string
        instance.HDRColorScale = float(entity_data.get('hdrcolorscale', 1.0))  # Type: float


class postprocess_controller(Targetname):
    icon_sprite = "editor/fog_controller.vmt"

    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.localcontraststrength = 0  # Type: float
        self.localcontrastedgestrength = 0  # Type: float
        self.vignettestart = 1  # Type: float
        self.vignetteend = 2  # Type: float
        self.vignetteblurstrength = 0  # Type: float
        self.grainstrength = 1  # Type: float
        self.topvignettestrength = 1  # Type: float
        self.fadetime = 2  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.localcontraststrength = float(entity_data.get('localcontraststrength', 0))  # Type: float
        instance.localcontrastedgestrength = float(entity_data.get('localcontrastedgestrength', 0))  # Type: float
        instance.vignettestart = float(entity_data.get('vignettestart', 1))  # Type: float
        instance.vignetteend = float(entity_data.get('vignetteend', 2))  # Type: float
        instance.vignetteblurstrength = float(entity_data.get('vignetteblurstrength', 0))  # Type: float
        instance.grainstrength = float(entity_data.get('grainstrength', 1))  # Type: float
        instance.topvignettestrength = float(entity_data.get('topvignettestrength', 1))  # Type: float
        instance.fadetime = float(entity_data.get('fadetime', 2))  # Type: float


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
        self.StartNoise = None  # Type: string

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
        instance.StartNoise = entity_data.get('startnoise', None)  # Type: string


class env_laser(Targetname, RenderFxChoices, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(RenderFxChoices).__init__()
        super(Parentname).__init__()
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
        RenderFxChoices.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
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


class env_rotorwash_emitter(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.altitude = 1024  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.altitude = float(entity_data.get('altitude', 1024))  # Type: float


class gibshooter(gibshooterbase):
    icon_sprite = "editor/gibshooter.vmt"

    def __init__(self):
        super(gibshooterbase).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        gibshooterbase.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_shooter(RenderFields, gibshooterbase):
    icon_sprite = "editor/env_shooter.vmt"

    def __init__(self):
        super(RenderFields).__init__()
        super(gibshooterbase).__init__()
        self.origin = [0, 0, 0]
        self.shootmodel = None  # Type: studio
        self.shootsounds = "CHOICES NOT SUPPORTED"  # Type: choices
        self.simulation = None  # Type: choices
        self.skin = None  # Type: integer
        self.nogibshadows = None  # Type: choices
        self.gibgravityscale = 1  # Type: float
        self.massoverride = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        RenderFields.from_dict(instance, entity_data)
        gibshooterbase.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.shootmodel = entity_data.get('shootmodel', None)  # Type: studio
        instance.shootsounds = entity_data.get('shootsounds', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.simulation = entity_data.get('simulation', None)  # Type: choices
        instance.skin = parse_source_value(entity_data.get('skin', 0))  # Type: integer
        instance.nogibshadows = entity_data.get('nogibshadows', None)  # Type: choices
        instance.gibgravityscale = float(entity_data.get('gibgravityscale', 1))  # Type: float
        instance.massoverride = float(entity_data.get('massoverride', 0))  # Type: float


class env_rotorshooter(RenderFields, gibshooterbase):
    icon_sprite = "editor/env_shooter.vmt"

    def __init__(self):
        super(RenderFields).__init__()
        super(gibshooterbase).__init__()
        self.origin = [0, 0, 0]
        self.shootmodel = None  # Type: studio
        self.shootsounds = "CHOICES NOT SUPPORTED"  # Type: choices
        self.simulation = None  # Type: choices
        self.skin = None  # Type: integer
        self.rotortime = 1  # Type: float
        self.rotortimevariance = 0.3  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        RenderFields.from_dict(instance, entity_data)
        gibshooterbase.from_dict(instance, entity_data)
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


class env_soundscape(Targetname, EnableDisable, Parentname):
    icon_sprite = "editor/env_soundscape.vmt"

    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(Parentname).__init__()
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
        EnableDisable.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
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


class env_sprite(Targetname, RenderFields, Parentname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.framerate = "10.0"  # Type: string
        self.model = "sprites/glow01.spr"  # Type: sprite
        self.scale = None  # Type: string
        self.GlowProxySize = 2.0  # Type: float
        self.HDRColorScale = 0.7  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.framerate = entity_data.get('framerate', "10.0")  # Type: string
        instance.model = entity_data.get('model', "sprites/glow01.spr")  # Type: sprite
        instance.scale = entity_data.get('scale', None)  # Type: string
        instance.GlowProxySize = float(entity_data.get('glowproxysize', 2.0))  # Type: float
        instance.HDRColorScale = float(entity_data.get('hdrcolorscale', 0.7))  # Type: float


class env_sprite_oriented(Angles, env_sprite):
    def __init__(self):
        super(env_sprite).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        env_sprite.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_wind(Targetname, Angles):
    icon_sprite = "editor/env_wind.vmt"

    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.minwind = 20  # Type: integer
        self.maxwind = 50  # Type: integer
        self.windradius = -1  # Type: float
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
        instance.windradius = float(entity_data.get('windradius', -1))  # Type: float
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
        self.fogenable = None  # Type: choices
        self.fogblend = None  # Type: choices
        self.use_angles = None  # Type: choices
        self.clip_3D_skybox_near_to_world_far = None  # Type: choices
        self.clip_3D_skybox_near_to_world_far_offset = "0.0"  # Type: string
        self.fogcolor = [255, 255, 255]  # Type: color255
        self.fogcolor2 = [255, 255, 255]  # Type: color255
        self.fogdir = "1 0 0"  # Type: string
        self.fogstart = "500.0"  # Type: string
        self.fogend = "2000.0"  # Type: string
        self.fogmaxdensity = 1  # Type: float
        self.HDRColorScale = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.scale = parse_source_value(entity_data.get('scale', 16))  # Type: integer
        instance.fogenable = entity_data.get('fogenable', None)  # Type: choices
        instance.fogblend = entity_data.get('fogblend', None)  # Type: choices
        instance.use_angles = entity_data.get('use_angles', None)  # Type: choices
        instance.clip_3D_skybox_near_to_world_far = entity_data.get('clip_3d_skybox_near_to_world_far',
                                                                    None)  # Type: choices
        instance.clip_3D_skybox_near_to_world_far_offset = entity_data.get('clip_3d_skybox_near_to_world_far_offset',
                                                                           "0.0")  # Type: string
        instance.fogcolor = parse_int_vector(entity_data.get('fogcolor', "255 255 255"))  # Type: color255
        instance.fogcolor2 = parse_int_vector(entity_data.get('fogcolor2', "255 255 255"))  # Type: color255
        instance.fogdir = entity_data.get('fogdir', "1 0 0")  # Type: string
        instance.fogstart = entity_data.get('fogstart', "500.0")  # Type: string
        instance.fogend = entity_data.get('fogend', "2000.0")  # Type: string
        instance.fogmaxdensity = float(entity_data.get('fogmaxdensity', 1))  # Type: float
        instance.HDRColorScale = float(entity_data.get('hdrcolorscale', 1.0))  # Type: float


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
        self.developeronly = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.message = entity_data.get('message', None)  # Type: string
        instance.radius = parse_source_value(entity_data.get('radius', 128))  # Type: integer
        instance.developeronly = entity_data.get('developeronly', None)  # Type: choices


class point_spotlight(Targetname, RenderFields, Angles, Parentname):
    model = "models/editor/cone_helper.mdl"

    def __init__(self):
        super(RenderFields).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.spotlightlength = 500  # Type: integer
        self.spotlightwidth = 50  # Type: integer
        self.HaloScale = 60  # Type: float
        self.HDRColorScale = 0.7  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.spotlightlength = parse_source_value(entity_data.get('spotlightlength', 500))  # Type: integer
        instance.spotlightwidth = parse_source_value(entity_data.get('spotlightwidth', 50))  # Type: integer
        instance.HaloScale = float(entity_data.get('haloscale', 60))  # Type: float
        instance.HDRColorScale = float(entity_data.get('hdrcolorscale', 0.7))  # Type: float


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


class point_broadcastclientcommand(Targetname):
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
        self.LowPriority = None  # Type: choices
        self.ApplyEntity = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.texture = entity_data.get('texture', None)  # Type: decal
        instance.LowPriority = entity_data.get('lowpriority', None)  # Type: choices
        instance.ApplyEntity = entity_data.get('applyentity', None)  # Type: target_destination


class info_projecteddecal(Targetname, Angles):
    model = "models/editor/axis_helper_thick.mdl"

    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.texture = None  # Type: decal
        self.Distance = 64  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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


class info_player_start(Angles, PlayerClass):
    model = "models/editor/playerstart.mdl"

    def __init__(self):
        super(Angles).__init__()
        super(PlayerClass).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        PlayerClass.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_overlay(Targetname, SystemLevelChoice):
    model = "models/editor/overlay_helper.mdl"

    def __init__(self):
        super(Targetname).__init__()
        super(SystemLevelChoice).__init__()
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
        SystemLevelChoice.from_dict(instance, entity_data)
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


class info_particle_target(Targetname, Angles, Parentname):
    model = "models/editor/cone_helper.mdl"

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


class info_particle_system(Targetname, Angles, Parentname):
    model = "models/editor/cone_helper.mdl"

    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.effect_name = None  # Type: particlesystem
        self.start_active = None  # Type: choices
        self.render_in_front = None  # Type: choices
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
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.effect_name = entity_data.get('effect_name', None)  # Type: particlesystem
        instance.start_active = entity_data.get('start_active', None)  # Type: choices
        instance.render_in_front = entity_data.get('render_in_front', None)  # Type: choices
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


class phys_ragdollmagnet(Targetname, Angles, EnableDisable, Parentname):
    icon_sprite = "editor/info_target.vmt"

    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(EnableDisable).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.axis = None  # Type: vecline
        self.radius = 512  # Type: float
        self.force = 5000  # Type: float
        self.target = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
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


class info_teleport_destination(Targetname, Angles, Parentname, PlayerClass):
    model = "models/editor/playerstart.mdl"

    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        super(PlayerClass).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        PlayerClass.from_dict(instance, entity_data)
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


class info_node_hint(Targetname, HintNode, Angles):
    model = "models/editor/ground_node_hint.mdl"

    def __init__(self):
        super(HintNode).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        HintNode.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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


class info_node_air_hint(Targetname, HintNode, Angles):
    model = "models/editor/air_node_hint.mdl"

    def __init__(self):
        super(HintNode).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.nodeheight = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        HintNode.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.nodeheight = parse_source_value(entity_data.get('nodeheight', 0))  # Type: integer


class info_hint(Targetname, HintNode, Angles):
    model = "models/editor/node_hint.mdl"

    def __init__(self):
        super(HintNode).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        HintNode.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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
        self.InvertAllow = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.StartNode = parse_source_value(entity_data.get('startnode', 0))  # Type: node_dest
        instance.EndNode = parse_source_value(entity_data.get('endnode', 0))  # Type: node_dest
        instance.initialstate = entity_data.get('initialstate', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.linktype = entity_data.get('linktype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.AllowUse = entity_data.get('allowuse', None)  # Type: string
        instance.InvertAllow = entity_data.get('invertallow', None)  # Type: choices


class info_node_link_controller(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.mins = [-8.0, -32.0, -36.0]  # Type: vector
        self.maxs = [8.0, 32.0, 36.0]  # Type: vector
        self.initialstate = "CHOICES NOT SUPPORTED"  # Type: choices
        self.useairlinkradius = None  # Type: choices
        self.AllowUse = None  # Type: string
        self.InvertAllow = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.mins = parse_float_vector(entity_data.get('mins', "-8 -32 -36"))  # Type: vector
        instance.maxs = parse_float_vector(entity_data.get('maxs', "8 32 36"))  # Type: vector
        instance.initialstate = entity_data.get('initialstate', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.useairlinkradius = entity_data.get('useairlinkradius', None)  # Type: choices
        instance.AllowUse = entity_data.get('allowuse', None)  # Type: string
        instance.InvertAllow = entity_data.get('invertallow', None)  # Type: choices


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


class info_node_climb(Targetname, HintNode, Angles):
    model = "models/editor/climb_node.mdl"

    def __init__(self):
        super(HintNode).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        HintNode.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class light(Targetname, Light):
    icon_sprite = "editor/light.vmt"

    def __init__(self):
        super(Targetname).__init__()
        super(Light).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self._distance = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Light.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance._distance = float(entity_data.get('_distance', 0))  # Type: float


class light_environment(Angles):
    icon_sprite = "editor/light_env.vmt"

    def __init__(self):
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.pitch = None  # Type: integer
        self._light = [255, 255, 255, 200]  # Type: color255
        self._ambient = [255, 255, 255, 20]  # Type: color255
        self._lightHDR = [-1, -1, -1, 1]  # Type: color255
        self._lightscaleHDR = 0.7  # Type: float
        self._ambientHDR = [-1, -1, -1, 1]  # Type: color255
        self._AmbientScaleHDR = 0.7  # Type: float
        self.SunSpreadAngle = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.pitch = parse_source_value(entity_data.get('pitch', 0))  # Type: integer
        instance._light = parse_int_vector(entity_data.get('_light', "255 255 255 200"))  # Type: color255
        instance._ambient = parse_int_vector(entity_data.get('_ambient', "255 255 255 20"))  # Type: color255
        instance._lightHDR = parse_int_vector(entity_data.get('_lighthdr', "-1 -1 -1 1"))  # Type: color255
        instance._lightscaleHDR = float(entity_data.get('_lightscalehdr', 0.7))  # Type: float
        instance._ambientHDR = parse_int_vector(entity_data.get('_ambienthdr', "-1 -1 -1 1"))  # Type: color255
        instance._AmbientScaleHDR = float(entity_data.get('_ambientscalehdr', 0.7))  # Type: float
        instance.SunSpreadAngle = float(entity_data.get('sunspreadangle', 0))  # Type: float


class light_directional(Angles):
    icon_sprite = "editor/light_env.vmt"

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
        self._distance = None  # Type: float
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
        instance._distance = float(entity_data.get('_distance', 0))  # Type: float
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
        self.disableallshadows = None  # Type: choices
        self.enableshadowsfromlocallights = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.angles = entity_data.get('angles', "80 30 0")  # Type: string
        instance.color = parse_int_vector(entity_data.get('color', "128 128 128"))  # Type: color255
        instance.distance = float(entity_data.get('distance', 75))  # Type: float
        instance.disableallshadows = entity_data.get('disableallshadows', None)  # Type: choices
        instance.enableshadowsfromlocallights = entity_data.get('enableshadowsfromlocallights', None)  # Type: choices


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
        self.exclusive = None  # Type: choices

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
        instance.exclusive = entity_data.get('exclusive', None)  # Type: choices


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


class func_movelinear(Targetname, RenderFields, Parentname, Origin):
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
        RenderFields.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
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


class func_rotating(Parentname, RenderFields, Angles, Origin, Shadow, Targetname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Origin).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
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
        RenderFields.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.maxspeed = parse_source_value(entity_data.get('maxspeed', 100))  # Type: integer
        instance.fanfriction = parse_source_value(entity_data.get('fanfriction', 20))  # Type: integer
        instance.message = entity_data.get('message', None)  # Type: sound
        instance.volume = parse_source_value(entity_data.get('volume', 10))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.dmg = parse_source_value(entity_data.get('dmg', 0))  # Type: integer
        instance.solidbsp = entity_data.get('solidbsp', None)  # Type: choices


class func_platrot(Parentname, RenderFields, Angles, Origin, BasePlat, Shadow, Targetname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Origin).__init__()
        super(BasePlat).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        self.noise1 = None  # Type: sound
        self.noise2 = None  # Type: sound
        self.speed = 50  # Type: integer
        self.height = None  # Type: integer
        self.rotation = None  # Type: integer
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        BasePlat.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.noise1 = entity_data.get('noise1', None)  # Type: sound
        instance.noise2 = entity_data.get('noise2', None)  # Type: sound
        instance.speed = parse_source_value(entity_data.get('speed', 50))  # Type: integer
        instance.height = parse_source_value(entity_data.get('height', 0))  # Type: integer
        instance.rotation = parse_source_value(entity_data.get('rotation', 0))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class keyframe_track(Targetname, KeyFrame, Angles, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(KeyFrame).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        KeyFrame.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)


class move_keyframed(Targetname, KeyFrame, Mover, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(KeyFrame).__init__()
        super(Mover).__init__()
        super(Parentname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        KeyFrame.from_dict(instance, entity_data)
        Mover.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)


class move_track(Targetname, KeyFrame, Mover, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(KeyFrame).__init__()
        super(Mover).__init__()
        super(Parentname).__init__()
        self.WheelBaseLength = 50  # Type: integer
        self.Damage = None  # Type: integer
        self.NoRotate = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        KeyFrame.from_dict(instance, entity_data)
        Mover.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.WheelBaseLength = parse_source_value(entity_data.get('wheelbaselength', 50))  # Type: integer
        instance.Damage = parse_source_value(entity_data.get('damage', 0))  # Type: integer
        instance.NoRotate = entity_data.get('norotate', None)  # Type: choices


class RopeKeyFrame(SystemLevelChoice):
    def __init__(self):
        super(SystemLevelChoice).__init__()
        self.Slack = 25  # Type: integer
        self.Type = None  # Type: choices
        self.Subdiv = 2  # Type: integer
        self.Barbed = None  # Type: choices
        self.Width = "2"  # Type: string
        self.TextureScale = "1"  # Type: string
        self.Collide = None  # Type: choices
        self.Dangling = None  # Type: choices
        self.Breakable = None  # Type: choices
        self.UseWind = None  # Type: choices
        self.RopeMaterial = "cable/cable.vmt"  # Type: material

    @staticmethod
    def from_dict(instance, entity_data: dict):
        SystemLevelChoice.from_dict(instance, entity_data)
        instance.Slack = parse_source_value(entity_data.get('slack', 25))  # Type: integer
        instance.Type = entity_data.get('type', None)  # Type: choices
        instance.Subdiv = parse_source_value(entity_data.get('subdiv', 2))  # Type: integer
        instance.Barbed = entity_data.get('barbed', None)  # Type: choices
        instance.Width = entity_data.get('width', "2")  # Type: string
        instance.TextureScale = entity_data.get('texturescale', "1")  # Type: string
        instance.Collide = entity_data.get('collide', None)  # Type: choices
        instance.Dangling = entity_data.get('dangling', None)  # Type: choices
        instance.Breakable = entity_data.get('breakable', None)  # Type: choices
        instance.UseWind = entity_data.get('usewind', None)  # Type: choices
        instance.RopeMaterial = entity_data.get('ropematerial', "cable/cable.vmt")  # Type: material


class keyframe_rope(Targetname, RopeKeyFrame, KeyFrame, Parentname):
    model = "models/editor/axis_helper_thick.mdl"

    def __init__(self):
        super(RopeKeyFrame).__init__()
        super(Targetname).__init__()
        super(KeyFrame).__init__()
        super(Parentname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        RopeKeyFrame.from_dict(instance, entity_data)
        KeyFrame.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)


class move_rope(Targetname, RopeKeyFrame, KeyFrame, Parentname):
    model = "models/editor/axis_helper.mdl"

    def __init__(self):
        super(RopeKeyFrame).__init__()
        super(Targetname).__init__()
        super(KeyFrame).__init__()
        super(Parentname).__init__()
        self.PositionInterpolator = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        RopeKeyFrame.from_dict(instance, entity_data)
        KeyFrame.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.PositionInterpolator = entity_data.get('positioninterpolator',
                                                        "CHOICES NOT SUPPORTED")  # Type: choices


class Button(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class func_button(Parentname, RenderFields, Origin, Button, DamageFilter, Targetname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Origin).__init__()
        super(Button).__init__()
        super(DamageFilter).__init__()
        super(Targetname).__init__()
        self.movedir = [0.0, 0.0, 0.0]  # Type: angle
        self.speed = 5  # Type: integer
        self.health = None  # Type: integer
        self.lip = None  # Type: integer
        self.master = None  # Type: string
        self.glow = None  # Type: target_destination
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
        RenderFields.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Button.from_dict(instance, entity_data)
        DamageFilter.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.movedir = parse_float_vector(entity_data.get('movedir', "0 0 0"))  # Type: angle
        instance.speed = parse_source_value(entity_data.get('speed', 5))  # Type: integer
        instance.health = parse_source_value(entity_data.get('health', 0))  # Type: integer
        instance.lip = parse_source_value(entity_data.get('lip', 0))  # Type: integer
        instance.master = entity_data.get('master', None)  # Type: string
        instance.glow = entity_data.get('glow', None)  # Type: target_destination
        instance.sounds = entity_data.get('sounds', None)  # Type: choices
        instance.wait = parse_source_value(entity_data.get('wait', 3))  # Type: integer
        instance.locked_sound = entity_data.get('locked_sound', None)  # Type: choices
        instance.unlocked_sound = entity_data.get('unlocked_sound', None)  # Type: choices
        instance.locked_sentence = entity_data.get('locked_sentence', None)  # Type: choices
        instance.unlocked_sentence = entity_data.get('unlocked_sentence', None)  # Type: choices
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class func_rot_button(Parentname, Angles, Origin, Button, EnableDisable, Targetname, Global):
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Origin).__init__()
        super(Button).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
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
        Angles.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Button.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        instance.master = entity_data.get('master', None)  # Type: string
        instance.speed = parse_source_value(entity_data.get('speed', 50))  # Type: integer
        instance.health = parse_source_value(entity_data.get('health', 0))  # Type: integer
        instance.sounds = entity_data.get('sounds', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.wait = parse_source_value(entity_data.get('wait', 3))  # Type: integer
        instance.distance = parse_source_value(entity_data.get('distance', 90))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class momentary_rot_button(Parentname, RenderFields, Angles, Origin, Targetname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        self.speed = 50  # Type: integer
        self.master = None  # Type: string
        self.glow = None  # Type: target_destination
        self.sounds = None  # Type: choices
        self.distance = 90  # Type: integer
        self.returnspeed = None  # Type: integer
        self._minlight = None  # Type: string
        self.startposition = None  # Type: float
        self.startdirection = "CHOICES NOT SUPPORTED"  # Type: choices
        self.solidbsp = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.speed = parse_source_value(entity_data.get('speed', 50))  # Type: integer
        instance.master = entity_data.get('master', None)  # Type: string
        instance.glow = entity_data.get('glow', None)  # Type: target_destination
        instance.sounds = entity_data.get('sounds', None)  # Type: choices
        instance.distance = parse_source_value(entity_data.get('distance', 90))  # Type: integer
        instance.returnspeed = parse_source_value(entity_data.get('returnspeed', 0))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.startposition = float(entity_data.get('startposition', 0))  # Type: float
        instance.startdirection = entity_data.get('startdirection', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.solidbsp = entity_data.get('solidbsp', None)  # Type: choices


class Door(Parentname, RenderFields, Shadow, Targetname, Global):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
        self.speed = 100  # Type: integer
        self.master = None  # Type: string
        self.noise1 = None  # Type: sound
        self.noise2 = None  # Type: sound
        self.startclosesound = None  # Type: sound
        self.closesound = None  # Type: sound
        self.wait = 4  # Type: integer
        self.lip = None  # Type: integer
        self.dmg = None  # Type: integer
        self.forceclosed = None  # Type: choices
        self.ignoredebris = None  # Type: choices
        self.message = None  # Type: string
        self.health = None  # Type: integer
        self.locked_sound = None  # Type: sound
        self.unlocked_sound = None  # Type: sound
        self.spawnpos = None  # Type: choices
        self.locked_sentence = None  # Type: choices
        self.unlocked_sentence = None  # Type: choices
        self._minlight = None  # Type: string
        self.loopmovesound = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        instance.speed = parse_source_value(entity_data.get('speed', 100))  # Type: integer
        instance.master = entity_data.get('master', None)  # Type: string
        instance.noise1 = entity_data.get('noise1', None)  # Type: sound
        instance.noise2 = entity_data.get('noise2', None)  # Type: sound
        instance.startclosesound = entity_data.get('startclosesound', None)  # Type: sound
        instance.closesound = entity_data.get('closesound', None)  # Type: sound
        instance.wait = parse_source_value(entity_data.get('wait', 4))  # Type: integer
        instance.lip = parse_source_value(entity_data.get('lip', 0))  # Type: integer
        instance.dmg = parse_source_value(entity_data.get('dmg', 0))  # Type: integer
        instance.forceclosed = entity_data.get('forceclosed', None)  # Type: choices
        instance.ignoredebris = entity_data.get('ignoredebris', None)  # Type: choices
        instance.message = entity_data.get('message', None)  # Type: string
        instance.health = parse_source_value(entity_data.get('health', 0))  # Type: integer
        instance.locked_sound = entity_data.get('locked_sound', None)  # Type: sound
        instance.unlocked_sound = entity_data.get('unlocked_sound', None)  # Type: sound
        instance.spawnpos = entity_data.get('spawnpos', None)  # Type: choices
        instance.locked_sentence = entity_data.get('locked_sentence', None)  # Type: choices
        instance.unlocked_sentence = entity_data.get('unlocked_sentence', None)  # Type: choices
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.loopmovesound = entity_data.get('loopmovesound', None)  # Type: choices


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


class func_door_rotating(Door, Angles, Origin):
    def __init__(self):
        super(Door).__init__()
        super(Angles).__init__()
        super(Origin).__init__()
        self.distance = 90  # Type: integer
        self.always_fire_blocked_outputs = None  # Type: choices
        self.solidbsp = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Door.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.distance = parse_source_value(entity_data.get('distance', 90))  # Type: integer
        instance.always_fire_blocked_outputs = entity_data.get('always_fire_blocked_outputs', None)  # Type: choices
        instance.solidbsp = entity_data.get('solidbsp', None)  # Type: choices


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


class prop_door_rotating(Parentname, BaseFadeProp, Angles, Glow, Studiomodel, Targetname, Global):
    def __init__(self):
        super(Parentname).__init__()
        super(BaseFadeProp).__init__()
        super(Angles).__init__()
        super(Glow).__init__()
        super(Studiomodel).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
        self.origin = [0, 0, 0]
        self.slavename = None  # Type: target_destination
        self.hardware = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ajarangles = [0.0, 0.0, 0.0]  # Type: angle
        self.spawnpos = None  # Type: choices
        self.axis = None  # Type: axis
        self.distance = 90  # Type: float
        self.speed = 200  # Type: integer
        self.soundopenoverride = None  # Type: sound
        self.soundcloseoverride = None  # Type: sound
        self.soundmoveoverride = None  # Type: sound
        self.returndelay = -1  # Type: integer
        self.dmg = None  # Type: integer
        self.health = None  # Type: integer
        self.soundlockedoverride = None  # Type: sound
        self.soundunlockedoverride = None  # Type: sound
        self.rendercolor = [255, 255, 255]  # Type: color255
        self.forceclosed = None  # Type: choices
        self.opendir = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Glow.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.slavename = entity_data.get('slavename', None)  # Type: target_destination
        instance.hardware = entity_data.get('hardware', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ajarangles = parse_float_vector(entity_data.get('ajarangles', "0 0 0"))  # Type: angle
        instance.spawnpos = entity_data.get('spawnpos', None)  # Type: choices
        instance.axis = entity_data.get('axis', None)  # Type: axis
        instance.distance = float(entity_data.get('distance', 90))  # Type: float
        instance.speed = parse_source_value(entity_data.get('speed', 200))  # Type: integer
        instance.soundopenoverride = entity_data.get('soundopenoverride', None)  # Type: sound
        instance.soundcloseoverride = entity_data.get('soundcloseoverride', None)  # Type: sound
        instance.soundmoveoverride = entity_data.get('soundmoveoverride', None)  # Type: sound
        instance.returndelay = parse_source_value(entity_data.get('returndelay', -1))  # Type: integer
        instance.dmg = parse_source_value(entity_data.get('dmg', 0))  # Type: integer
        instance.health = parse_source_value(entity_data.get('health', 0))  # Type: integer
        instance.soundlockedoverride = entity_data.get('soundlockedoverride', None)  # Type: sound
        instance.soundunlockedoverride = entity_data.get('soundunlockedoverride', None)  # Type: sound
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "255 255 255"))  # Type: color255
        instance.forceclosed = entity_data.get('forceclosed', None)  # Type: choices
        instance.opendir = entity_data.get('opendir', None)  # Type: choices


class prop_wall_breakable(Parentname, BaseFadeProp, Angles, Studiomodel, Targetname, Global):
    def __init__(self):
        super(Parentname).__init__()
        super(BaseFadeProp).__init__()
        super(Angles).__init__()
        super(Studiomodel).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


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
        self.StartDisabled = None  # Type: choices
        self.Color = [255, 255, 255]  # Type: color255
        self.SpawnRate = 40  # Type: integer
        self.SpeedMax = "13"  # Type: string
        self.LifetimeMin = "3"  # Type: string
        self.LifetimeMax = "5"  # Type: string
        self.DistMax = 1024  # Type: integer
        self.Frozen = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.StartDisabled = entity_data.get('startdisabled', None)  # Type: choices
        instance.Color = parse_int_vector(entity_data.get('color', "255 255 255"))  # Type: color255
        instance.SpawnRate = parse_source_value(entity_data.get('spawnrate', 40))  # Type: integer
        instance.SpeedMax = entity_data.get('speedmax', "13")  # Type: string
        instance.LifetimeMin = entity_data.get('lifetimemin', "3")  # Type: string
        instance.LifetimeMax = entity_data.get('lifetimemax', "5")  # Type: string
        instance.DistMax = parse_source_value(entity_data.get('distmax', 1024))  # Type: integer
        instance.Frozen = entity_data.get('frozen', None)  # Type: choices


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
        self.MaxDrawDistance = None  # Type: float

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
        instance.MaxDrawDistance = float(entity_data.get('maxdrawdistance', 0))  # Type: float


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
        self.fov = 90  # Type: float
        self.fov_rate = 1.0  # Type: float
        self.target = None  # Type: target_destination
        self.targetattachment = None  # Type: string
        self.wait = 10  # Type: integer
        self.moveto = None  # Type: target_destination
        self.interpolatepositiontoplayer = None  # Type: choices
        self.speed = "0"  # Type: string
        self.acceleration = "500"  # Type: string
        self.deceleration = "500"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.fov = float(entity_data.get('fov', 90))  # Type: float
        instance.fov_rate = float(entity_data.get('fov_rate', 1.0))  # Type: float
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.targetattachment = entity_data.get('targetattachment', None)  # Type: string
        instance.wait = parse_source_value(entity_data.get('wait', 10))  # Type: integer
        instance.moveto = entity_data.get('moveto', None)  # Type: target_destination
        instance.interpolatepositiontoplayer = entity_data.get('interpolatepositiontoplayer', None)  # Type: choices
        instance.speed = entity_data.get('speed', "0")  # Type: string
        instance.acceleration = entity_data.get('acceleration', "500")  # Type: string
        instance.deceleration = entity_data.get('deceleration', "500")  # Type: string


class point_posecontroller(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.PropName = None  # Type: string
        self.PoseParameterName = None  # Type: string
        self.PoseValue = 0.0  # Type: float
        self.InterpolationTime = 0.0  # Type: float
        self.InterpolationWrap = None  # Type: choices
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
        instance.InterpolationWrap = entity_data.get('interpolationwrap', None)  # Type: choices
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
        self.ShouldComparetoValue = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.IntegerValue = parse_source_value(entity_data.get('integervalue', 0))  # Type: integer
        instance.ShouldComparetoValue = entity_data.get('shouldcomparetovalue', None)  # Type: choices


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


class logic_timer(Targetname, EnableDisable):
    icon_sprite = "editor/logic_timer.vmt"

    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.UseRandomTime = None  # Type: choices
        self.LowerRandomBound = None  # Type: string
        self.UpperRandomBound = None  # Type: string
        self.RefireTime = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.UseRandomTime = entity_data.get('userandomtime', None)  # Type: choices
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
        self.startdisabled = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.attach1 = entity_data.get('attach1', None)  # Type: target_destination
        instance.attach2 = entity_data.get('attach2', None)  # Type: target_destination
        instance.startdisabled = entity_data.get('startdisabled', "CHOICES NOT SUPPORTED")  # Type: choices


class env_microphone(Targetname, EnableDisable, Parentname):
    icon_sprite = "editor/env_microphone.vmt"

    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(Parentname).__init__()
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
        EnableDisable.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
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
        instance.target = entity_data.get('target',
                                          None)  # Set to none due to bug in BlackMesa base.fgd file  # Type: target_destination
        instance.navprop = entity_data.get('navprop', "CHOICES NOT SUPPORTED")  # Type: choices


class logic_autosave(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.NewLevelUnit = None  # Type: choices
        self.MinimumHitPoints = None  # Type: integer
        self.MinHitPointsToCommit = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.NewLevelUnit = entity_data.get('newlevelunit', None)  # Type: choices
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


class point_template(Targetname):
    icon_sprite = "editor/point_template.vmt"

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
        self.PostSpawnInheritAngles = None  # Type: choices

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
        instance.PostSpawnInheritAngles = entity_data.get('postspawninheritangles', None)  # Type: choices


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


class point_anglesensor(Targetname, EnableDisable, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.lookatname = None  # Type: target_destination
        self.duration = None  # Type: float
        self.tolerance = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
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
        self.usehelper = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.threshold = float(entity_data.get('threshold', 0))  # Type: float
        instance.fireinterval = float(entity_data.get('fireinterval', 0.2))  # Type: float
        instance.axis = entity_data.get('axis', None)  # Type: vecline
        instance.usehelper = entity_data.get('usehelper', None)  # Type: choices


class point_velocitysensor(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.axis = None  # Type: vecline
        self.enabled = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.axis = entity_data.get('axis', None)  # Type: vecline
        instance.enabled = entity_data.get('enabled', "CHOICES NOT SUPPORTED")  # Type: choices


class point_proximity_sensor(Targetname, Angles, EnableDisable, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(EnableDisable).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination


class point_teleport(Targetname, Angles):
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


class point_push(Targetname, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.enabled = "CHOICES NOT SUPPORTED"  # Type: choices
        self.magnitude = 100  # Type: float
        self.radius = 128  # Type: float
        self.inner_radius = 0  # Type: float
        self.influence_cone = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.enabled = entity_data.get('enabled', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.magnitude = float(entity_data.get('magnitude', 100))  # Type: float
        instance.radius = float(entity_data.get('radius', 128))  # Type: float
        instance.inner_radius = float(entity_data.get('inner_radius', 0))  # Type: float
        instance.influence_cone = float(entity_data.get('influence_cone', 0))  # Type: float


class func_physbox(RenderFields, Origin, BreakableBrush):
    def __init__(self):
        super(BreakableBrush).__init__()
        super(RenderFields).__init__()
        super(Origin).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        self._minlight = None  # Type: string
        self.Damagetype = None  # Type: choices
        self.massScale = 0  # Type: float
        self.overridescript = None  # Type: string
        self.damagetoenablemotion = None  # Type: integer
        self.forcetoenablemotion = None  # Type: float
        self.preferredcarryangles = [0.0, 0.0, 0.0]  # Type: vector
        self.notsolid = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        RenderFields.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        BreakableBrush.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.Damagetype = entity_data.get('damagetype', None)  # Type: choices
        instance.massScale = float(entity_data.get('massscale', 0))  # Type: float
        instance.overridescript = entity_data.get('overridescript', None)  # Type: string
        instance.damagetoenablemotion = parse_source_value(entity_data.get('damagetoenablemotion', 0))  # Type: integer
        instance.forcetoenablemotion = float(entity_data.get('forcetoenablemotion', 0))  # Type: float
        instance.preferredcarryangles = parse_float_vector(
            entity_data.get('preferredcarryangles', "0 0 0"))  # Type: vector
        instance.notsolid = entity_data.get('notsolid', None)  # Type: choices


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
        instance.attachpoint = entity_data.get('attachpoint',
                                               None)  # Set to none due to bug in BlackMesa base.fgd file  # Type: vecline


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


class phys_magnet(Targetname, Studiomodel, Angles, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Studiomodel).__init__()
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
        Studiomodel.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
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


class prop_static_base(SystemLevelChoice, Angles):
    def __init__(self):
        super(SystemLevelChoice).__init__()
        super(Angles).__init__()
        self.model = None  # Type: studio
        self.skin = None  # Type: integer
        self.solid = "CHOICES NOT SUPPORTED"  # Type: choices
        self.disableshadows = None  # Type: choices
        self.fademindist = -1  # Type: float
        self.fademaxdist = None  # Type: float
        self.fadescale = 1  # Type: float
        self.lightingorigin = None  # Type: target_destination
        self.disablevertexlighting = "CHOICES NOT SUPPORTED"  # Type: choices
        self.disableselfshadowing = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ignorenormals = None  # Type: choices
        self.renderamt = 255  # Type: integer
        self.rendercolor = [255, 255, 255]  # Type: color255

    @staticmethod
    def from_dict(instance, entity_data: dict):
        SystemLevelChoice.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', None)  # Type: studio
        instance.skin = parse_source_value(entity_data.get('skin', 0))  # Type: integer
        instance.solid = entity_data.get('solid', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.disableshadows = entity_data.get('disableshadows', None)  # Type: choices
        instance.fademindist = float(entity_data.get('fademindist', -1))  # Type: float
        instance.fademaxdist = float(entity_data.get('fademaxdist', 0))  # Type: float
        instance.fadescale = float(entity_data.get('fadescale', 1))  # Type: float
        instance.lightingorigin = entity_data.get('lightingorigin', None)  # Type: target_destination
        instance.disablevertexlighting = entity_data.get('disablevertexlighting',
                                                         "CHOICES NOT SUPPORTED")  # Type: choices
        instance.disableselfshadowing = entity_data.get('disableselfshadowing',
                                                        "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ignorenormals = entity_data.get('ignorenormals', None)  # Type: choices
        instance.renderamt = parse_source_value(entity_data.get('renderamt', 255))  # Type: integer
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "255 255 255"))  # Type: color255


class prop_dynamic_base(Parentname, Angles, Glow, BreakableProp, Studiomodel, RenderFields, Global):
    def __init__(self):
        super(RenderFields).__init__()
        super(BreakableProp).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Glow).__init__()
        super(Studiomodel).__init__()
        super(Global).__init__()
        self.solid = "CHOICES NOT SUPPORTED"  # Type: choices
        self.DefaultAnim = None  # Type: string
        self.RandomAnimation = None  # Type: choices
        self.MinAnimTime = 5  # Type: float
        self.MaxAnimTime = 10  # Type: float
        self.SetBodyGroup = None  # Type: integer
        self.LagCompensate = None  # Type: choices
        self.glowbackfacemult = 1.0  # Type: float
        self.lightingorigin = None  # Type: target_destination
        self.updatechildren = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Glow.from_dict(instance, entity_data)
        BreakableProp.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        instance.solid = entity_data.get('solid', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.DefaultAnim = entity_data.get('defaultanim', None)  # Type: string
        instance.RandomAnimation = entity_data.get('randomanimation', None)  # Type: choices
        instance.MinAnimTime = float(entity_data.get('minanimtime', 5))  # Type: float
        instance.MaxAnimTime = float(entity_data.get('maxanimtime', 10))  # Type: float
        instance.SetBodyGroup = parse_source_value(entity_data.get('setbodygroup', 0))  # Type: integer
        instance.LagCompensate = entity_data.get('lagcompensate', None)  # Type: choices
        instance.glowbackfacemult = float(entity_data.get('glowbackfacemult', 1.0))  # Type: float
        instance.lightingorigin = entity_data.get('lightingorigin', None)  # Type: target_destination
        instance.updatechildren = entity_data.get('updatechildren', None)  # Type: choices


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


class prop_dynamic(EnableDisable, prop_dynamic_base):
    def __init__(self):
        super(prop_dynamic_base).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class prop_dynamic_override(prop_dynamic_base):
    def __init__(self):
        super(prop_dynamic_base).__init__()
        self.origin = [0, 0, 0]
        self.health = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.health = parse_source_value(entity_data.get('health', 0))  # Type: integer


class BasePropPhysics(Angles, Glow, BreakableProp, Studiomodel, SystemLevelChoice, Global):
    def __init__(self):
        super(BreakableProp).__init__()
        super(Angles).__init__()
        super(Glow).__init__()
        super(Studiomodel).__init__()
        super(SystemLevelChoice).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
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

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Glow.from_dict(instance, entity_data)
        BreakableProp.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        SystemLevelChoice.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
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


class prop_physics_override(BasePropPhysics, BaseFadeProp):
    def __init__(self):
        super(BasePropPhysics).__init__()
        super(BaseFadeProp).__init__()
        self.origin = [0, 0, 0]
        self.health = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePropPhysics.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.health = parse_source_value(entity_data.get('health', 0))  # Type: integer


class prop_physics(BasePropPhysics, RenderFields):
    def __init__(self):
        super(BasePropPhysics).__init__()
        super(RenderFields).__init__()
        self.origin = [0, 0, 0]
        self.BreakableType = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePropPhysics.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.BreakableType = entity_data.get('breakabletype', None)  # Type: choices


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


class prop_ragdoll(BaseFadeProp, Angles, Studiomodel, SystemLevelChoice, Targetname, EnableDisable):
    def __init__(self):
        super(BaseFadeProp).__init__()
        super(Angles).__init__()
        super(Studiomodel).__init__()
        super(SystemLevelChoice).__init__()
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.angleOverride = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFadeProp.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        SystemLevelChoice.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
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


class func_breakable(Origin, RenderFields, BreakableBrush):
    def __init__(self):
        super(BreakableBrush).__init__()
        super(RenderFields).__init__()
        super(Origin).__init__()
        super(Shadow).__init__()
        self.minhealthdmg = None  # Type: integer
        self._minlight = None  # Type: string
        self.physdamagescale = 1.0  # Type: float
        self.BreakableType = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Origin.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        BreakableBrush.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        instance.minhealthdmg = parse_source_value(entity_data.get('minhealthdmg', 0))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.physdamagescale = float(entity_data.get('physdamagescale', 1.0))  # Type: float
        instance.BreakableType = entity_data.get('breakabletype', None)  # Type: choices


class func_breakable_surf(RenderFields, BreakableBrush):
    def __init__(self):
        super(BreakableBrush).__init__()
        super(RenderFields).__init__()
        super(Shadow).__init__()
        self.fragility = 100  # Type: integer
        self.surfacetype = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        RenderFields.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        BreakableBrush.from_dict(instance, entity_data)
        instance.fragility = parse_source_value(entity_data.get('fragility', 100))  # Type: integer
        instance.surfacetype = entity_data.get('surfacetype', None)  # Type: choices


class func_conveyor(Targetname, RenderFields, Shadow, Parentname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Targetname).__init__()
        super(Shadow).__init__()
        super(Parentname).__init__()
        self.movedir = [0.0, 0.0, 0.0]  # Type: angle
        self.speed = "100"  # Type: string
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.movedir = parse_float_vector(entity_data.get('movedir', "0 0 0"))  # Type: angle
        instance.speed = entity_data.get('speed', "100")  # Type: string
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class func_detail(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class func_viscluster(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class func_illusionary(Parentname, RenderFields, Origin, Shadow, Targetname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Origin).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class func_precipitation_blocker(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)


class func_detail_blocker(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)


class func_wall_toggle(func_wall):
    def __init__(self):
        super(func_wall).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        func_wall.from_dict(instance, entity_data)


class func_guntarget(Targetname, RenderFields, Global, Parentname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
        super(Parentname).__init__()
        self.speed = 100  # Type: integer
        self.target = None  # Type: target_destination
        self.health = None  # Type: integer
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.speed = parse_source_value(entity_data.get('speed', 100))  # Type: integer
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.health = parse_source_value(entity_data.get('health', 0))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class func_fish_pool(Base):
    def __init__(self):
        super().__init__()
        self.origin = [0, 0, 0]
        self.model = "models/Junkola.mdl"  # Type: studio
        self.fish_count = 10  # Type: integer
        self.max_range = 150  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', "models/Junkola.mdl")  # Type: studio
        instance.fish_count = parse_source_value(entity_data.get('fish_count', 10))  # Type: integer
        instance.max_range = float(entity_data.get('max_range', 150))  # Type: float


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


class Trackchange(PlatSounds, RenderFields, Parentname, Targetname, Global):
    def __init__(self):
        super(RenderFields).__init__()
        super(PlatSounds).__init__()
        super(Parentname).__init__()
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
        PlatSounds.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        instance.height = parse_source_value(entity_data.get('height', 0))  # Type: integer
        instance.rotation = parse_source_value(entity_data.get('rotation', 0))  # Type: integer
        instance.train = entity_data.get('train', None)  # Type: target_destination
        instance.toptrack = entity_data.get('toptrack', None)  # Type: target_destination
        instance.bottomtrack = entity_data.get('bottomtrack', None)  # Type: target_destination
        instance.speed = parse_source_value(entity_data.get('speed', 0))  # Type: integer


class BaseTrain(Parentname, RenderFields, Origin, Shadow, Targetname, Global):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Origin).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
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
        RenderFields.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
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


class func_tracktrain(BaseTrain):
    def __init__(self):
        super(BaseTrain).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
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
        self.NewLevelUnit = None  # Type: choices
        self.DangerousTimer = None  # Type: float
        self.MinimumHitPoints = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.master = entity_data.get('master', None)  # Type: string
        instance.NewLevelUnit = entity_data.get('newlevelunit', None)  # Type: choices
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
        self.nodmgforce = None  # Type: choices
        self.damageforce = None  # Type: vector
        self.thinkalways = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.master = entity_data.get('master', None)  # Type: string
        instance.damage = parse_source_value(entity_data.get('damage', 10))  # Type: integer
        instance.damagecap = parse_source_value(entity_data.get('damagecap', 20))  # Type: integer
        instance.damagetype = entity_data.get('damagetype', None)  # Type: choices
        instance.damagemodel = entity_data.get('damagemodel', None)  # Type: choices
        instance.nodmgforce = entity_data.get('nodmgforce', None)  # Type: choices
        instance.damageforce = parse_float_vector(entity_data.get('damageforce', "0 0 0"))  # Type: vector
        instance.thinkalways = entity_data.get('thinkalways', None)  # Type: choices


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
        self.entireteam = None  # Type: choices
        self.allowincap = None  # Type: choices
        self.allowghost = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.wait = parse_source_value(entity_data.get('wait', 1))  # Type: integer
        instance.entireteam = entity_data.get('entireteam', None)  # Type: choices
        instance.allowincap = entity_data.get('allowincap', None)  # Type: choices
        instance.allowghost = entity_data.get('allowghost', None)  # Type: choices


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
        self.triggeronstarttouch = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.pushdir = parse_float_vector(entity_data.get('pushdir', "0 0 0"))  # Type: angle
        instance.speed = parse_source_value(entity_data.get('speed', 40))  # Type: integer
        instance.alternateticksfix = float(entity_data.get('alternateticksfix', 0))  # Type: float
        instance.triggeronstarttouch = entity_data.get('triggeronstarttouch', None)  # Type: choices


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

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.landmark = entity_data.get('landmark', None)  # Type: target_destination


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


class ai_speechfilter(Targetname, EnableDisable, ResponseContext):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(ResponseContext).__init__()
        self.origin = [0, 0, 0]
        self.subject = None  # Type: target_destination
        self.IdleModifier = 1.0  # Type: float
        self.NeverSayHello = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        ResponseContext.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.subject = entity_data.get('subject', None)  # Type: target_destination
        instance.IdleModifier = float(entity_data.get('idlemodifier', 1.0))  # Type: float
        instance.NeverSayHello = entity_data.get('neversayhello', None)  # Type: choices


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


class npc_furniture(BaseNPC, Parentname):
    def __init__(self):
        super(BaseNPC).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.model = None  # Type: studio

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
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


class env_projectedtexture(Targetname, Angles, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.lightfov = 90.0  # Type: float
        self.nearz = 4.0  # Type: float
        self.farz = 750.0  # Type: float
        self.enableshadows = None  # Type: choices
        self.shadowquality = "CHOICES NOT SUPPORTED"  # Type: choices
        self.lightonlytarget = None  # Type: choices
        self.lightworld = "CHOICES NOT SUPPORTED"  # Type: choices
        self.lightcolor = [255, 255, 255, 200]  # Type: color255
        self.cameraspace = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.lightfov = float(entity_data.get('lightfov', 90.0))  # Type: float
        instance.nearz = float(entity_data.get('nearz', 4.0))  # Type: float
        instance.farz = float(entity_data.get('farz', 750.0))  # Type: float
        instance.enableshadows = entity_data.get('enableshadows', None)  # Type: choices
        instance.shadowquality = entity_data.get('shadowquality', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.lightonlytarget = entity_data.get('lightonlytarget', None)  # Type: choices
        instance.lightworld = entity_data.get('lightworld', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.lightcolor = parse_int_vector(entity_data.get('lightcolor', "255 255 255 200"))  # Type: color255
        instance.cameraspace = parse_source_value(entity_data.get('cameraspace', 0))  # Type: integer


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


class npc_puppet(BaseNPC, Studiomodel, Parentname):
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
        BaseNPC.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.animationtarget = entity_data.get('animationtarget', None)  # Type: target_source
        instance.attachmentname = entity_data.get('attachmentname', None)  # Type: string


class point_gamestats_counter(Origin, EnableDisable, Targetname):
    def __init__(self):
        super(Origin).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.Name = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Origin.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Name = entity_data.get('name', None)  # Type: string


class func_instance(Angles):
    def __init__(self):
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.targetname = None  # Type: target_source
        self.spawnpositionname = None  # Type: string
        self.file = None  # Type: instance_file
        self.fixup_style = None  # Type: choices
        self.propagate_fixup = None  # Type: choices
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
        instance.spawnpositionname = entity_data.get('spawnpositionname', None)  # Type: string
        instance.file = entity_data.get('file', None)  # Type: instance_file
        instance.fixup_style = entity_data.get('fixup_style', None)  # Type: choices
        instance.propagate_fixup = entity_data.get('propagate_fixup', None)  # Type: choices
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


class env_instructor_hint(Targetname):
    icon_sprite = "editor/env_instructor_hint.vmt"

    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.hint_target = None  # Type: target_destination
        self.hint_name = None  # Type: string
        self.hint_static = None  # Type: choices
        self.hint_allow_nodraw_target = "CHOICES NOT SUPPORTED"  # Type: choices
        self.hint_caption = None  # Type: string
        self.hint_color = [255, 255, 255]  # Type: color255
        self.hint_forcecaption = None  # Type: choices
        self.hint_icon_onscreen = "CHOICES NOT SUPPORTED"  # Type: choices
        self.hint_icon_offscreen = "CHOICES NOT SUPPORTED"  # Type: choices
        self.hint_nooffscreen = None  # Type: choices
        self.hint_binding = None  # Type: string
        self.hint_icon_offset = None  # Type: float
        self.hint_pulseoption = None  # Type: choices
        self.hint_alphaoption = None  # Type: choices
        self.hint_shakeoption = None  # Type: choices
        self.hint_timeout = None  # Type: integer
        self.hint_display_limit = None  # Type: integer
        self.hint_range = None  # Type: float
        self.hint_instance_type = "CHOICES NOT SUPPORTED"  # Type: choices
        self.hint_auto_start = "CHOICES NOT SUPPORTED"  # Type: choices
        self.hint_suppress_rest = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.hint_target = entity_data.get('hint_target', None)  # Type: target_destination
        instance.hint_name = entity_data.get('hint_name', None)  # Type: string
        instance.hint_static = entity_data.get('hint_static', None)  # Type: choices
        instance.hint_allow_nodraw_target = entity_data.get('hint_allow_nodraw_target',
                                                            "CHOICES NOT SUPPORTED")  # Type: choices
        instance.hint_caption = entity_data.get('hint_caption', None)  # Type: string
        instance.hint_color = parse_int_vector(entity_data.get('hint_color', "255 255 255"))  # Type: color255
        instance.hint_forcecaption = entity_data.get('hint_forcecaption', None)  # Type: choices
        instance.hint_icon_onscreen = entity_data.get('hint_icon_onscreen', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.hint_icon_offscreen = entity_data.get('hint_icon_offscreen', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.hint_nooffscreen = entity_data.get('hint_nooffscreen', None)  # Type: choices
        instance.hint_binding = entity_data.get('hint_binding', None)  # Type: string
        instance.hint_icon_offset = float(entity_data.get('hint_icon_offset', 0))  # Type: float
        instance.hint_pulseoption = entity_data.get('hint_pulseoption', None)  # Type: choices
        instance.hint_alphaoption = entity_data.get('hint_alphaoption', None)  # Type: choices
        instance.hint_shakeoption = entity_data.get('hint_shakeoption', None)  # Type: choices
        instance.hint_timeout = parse_source_value(entity_data.get('hint_timeout', 0))  # Type: integer
        instance.hint_display_limit = parse_source_value(entity_data.get('hint_display_limit', 0))  # Type: integer
        instance.hint_range = float(entity_data.get('hint_range', 0))  # Type: float
        instance.hint_instance_type = entity_data.get('hint_instance_type', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.hint_auto_start = entity_data.get('hint_auto_start', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.hint_suppress_rest = entity_data.get('hint_suppress_rest', None)  # Type: choices


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


class func_timescale(Targetname, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.desiredTimescale = 1.0  # Type: float
        self.acceleration = 0.05  # Type: float
        self.minBlendRate = 0.1  # Type: float
        self.blendDeltaMultiplier = 3.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.desiredTimescale = float(entity_data.get('desiredtimescale', 1.0))  # Type: float
        instance.acceleration = float(entity_data.get('acceleration', 0.05))  # Type: float
        instance.minBlendRate = float(entity_data.get('minblendrate', 0.1))  # Type: float
        instance.blendDeltaMultiplier = float(entity_data.get('blenddeltamultiplier', 3.0))  # Type: float


class func_block_charge(func_brush):
    def __init__(self):
        super(func_brush).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        func_brush.from_dict(instance, entity_data)


class info_ambient_mob_start(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_ambient_mob_end(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_ambient_mob(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_item_position(Targetname, Angles, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.model = None  # Type: studio
        self.group = None  # Type: integer
        self.rarity = None  # Type: choices
        self.replace01 = None  # Type: string
        self.replace02 = None  # Type: string
        self.replace03 = None  # Type: string
        self.replace04 = None  # Type: string
        self.replace05 = None  # Type: string
        self.replace06 = None  # Type: string
        self.replace07 = None  # Type: string
        self.replace08 = None  # Type: string
        self.replace09 = None  # Type: string
        self.replace10 = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', None)  # Type: studio
        instance.group = parse_source_value(entity_data.get('group', 0))  # Type: integer
        instance.rarity = entity_data.get('rarity', None)  # Type: choices
        instance.replace01 = entity_data.get('replace01', None)  # Type: string
        instance.replace02 = entity_data.get('replace02', None)  # Type: string
        instance.replace03 = entity_data.get('replace03', None)  # Type: string
        instance.replace04 = entity_data.get('replace04', None)  # Type: string
        instance.replace05 = entity_data.get('replace05', None)  # Type: string
        instance.replace06 = entity_data.get('replace06', None)  # Type: string
        instance.replace07 = entity_data.get('replace07', None)  # Type: string
        instance.replace08 = entity_data.get('replace08', None)  # Type: string
        instance.replace09 = entity_data.get('replace09', None)  # Type: string
        instance.replace10 = entity_data.get('replace10', None)  # Type: string


class info_l4d1_survivor_spawn(Targetname):
    model = "models/survivors/survivor_biker.mdl"

    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.character = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.character = entity_data.get('character', "CHOICES NOT SUPPORTED")  # Type: choices


class env_airstrike_indoors(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.height = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.height = entity_data.get('height', "CHOICES NOT SUPPORTED")  # Type: choices


class env_airstrike_outdoors(Targetname, Angles, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.model = "models/props_destruction/general_dest_roof_set.mdl"  # Type: studio
        self.modelgroup = None  # Type: target_destination
        self.sequence1 = None  # Type: string
        self.sequence2 = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', "models/props_destruction/general_dest_roof_set.mdl")  # Type: studio
        instance.modelgroup = entity_data.get('modelgroup', None)  # Type: target_destination
        instance.sequence1 = entity_data.get('sequence1', None)  # Type: string
        instance.sequence2 = entity_data.get('sequence2', None)  # Type: string


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


class point_viewcontrol_survivor(Targetname, Angles, Parentname):
    viewport_model = "models/editor/camera.mdl"

    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.fov = 90  # Type: float
        self.fov_rate = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.fov = float(entity_data.get('fov', 90))  # Type: float
        instance.fov_rate = float(entity_data.get('fov_rate', 1.0))  # Type: float


class point_deathfall_camera(Targetname, Angles, Parentname):
    viewport_model = "models/editor/camera.mdl"

    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.fov = 90  # Type: float
        self.fov_rate = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.fov = float(entity_data.get('fov', 90))  # Type: float
        instance.fov_rate = float(entity_data.get('fov_rate', 1.0))  # Type: float


class logic_choreographed_scene(Targetname):
    icon_sprite = "editor/choreo_scene.vmt"

    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.SceneFile = None  # Type: scene
        self.target1 = None  # Type: target_destination
        self.target2 = None  # Type: target_destination
        self.target3 = None  # Type: target_destination
        self.target4 = None  # Type: target_destination
        self.target5 = None  # Type: target_destination
        self.target6 = None  # Type: target_destination
        self.target7 = None  # Type: target_destination
        self.target8 = None  # Type: target_destination
        self.busyactor = "CHOICES NOT SUPPORTED"  # Type: choices
        self.onplayerdeath = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.SceneFile = entity_data.get('scenefile', None)  # Type: scene
        instance.target1 = entity_data.get('target1', None)  # Type: target_destination
        instance.target2 = entity_data.get('target2', None)  # Type: target_destination
        instance.target3 = entity_data.get('target3', None)  # Type: target_destination
        instance.target4 = entity_data.get('target4', None)  # Type: target_destination
        instance.target5 = entity_data.get('target5', None)  # Type: target_destination
        instance.target6 = entity_data.get('target6', None)  # Type: target_destination
        instance.target7 = entity_data.get('target7', None)  # Type: target_destination
        instance.target8 = entity_data.get('target8', None)  # Type: target_destination
        instance.busyactor = entity_data.get('busyactor', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.onplayerdeath = entity_data.get('onplayerdeath', None)  # Type: choices


class logic_scene_list_manager(Targetname):
    icon_sprite = "editor/choreo_manager.vmt"

    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.scene0 = None  # Type: target_destination
        self.scene1 = None  # Type: target_destination
        self.scene2 = None  # Type: target_destination
        self.scene3 = None  # Type: target_destination
        self.scene4 = None  # Type: target_destination
        self.scene5 = None  # Type: target_destination
        self.scene6 = None  # Type: target_destination
        self.scene7 = None  # Type: target_destination
        self.scene8 = None  # Type: target_destination
        self.scene9 = None  # Type: target_destination
        self.scene10 = None  # Type: target_destination
        self.scene11 = None  # Type: target_destination
        self.scene12 = None  # Type: target_destination
        self.scene13 = None  # Type: target_destination
        self.scene14 = None  # Type: target_destination
        self.scene15 = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.scene0 = entity_data.get('scene0', None)  # Type: target_destination
        instance.scene1 = entity_data.get('scene1', None)  # Type: target_destination
        instance.scene2 = entity_data.get('scene2', None)  # Type: target_destination
        instance.scene3 = entity_data.get('scene3', None)  # Type: target_destination
        instance.scene4 = entity_data.get('scene4', None)  # Type: target_destination
        instance.scene5 = entity_data.get('scene5', None)  # Type: target_destination
        instance.scene6 = entity_data.get('scene6', None)  # Type: target_destination
        instance.scene7 = entity_data.get('scene7', None)  # Type: target_destination
        instance.scene8 = entity_data.get('scene8', None)  # Type: target_destination
        instance.scene9 = entity_data.get('scene9', None)  # Type: target_destination
        instance.scene10 = entity_data.get('scene10', None)  # Type: target_destination
        instance.scene11 = entity_data.get('scene11', None)  # Type: target_destination
        instance.scene12 = entity_data.get('scene12', None)  # Type: target_destination
        instance.scene13 = entity_data.get('scene13', None)  # Type: target_destination
        instance.scene14 = entity_data.get('scene14', None)  # Type: target_destination
        instance.scene15 = entity_data.get('scene15', None)  # Type: target_destination


class generic_actor(BaseNPC, Parentname):
    def __init__(self):
        super(BaseNPC).__init__()
        super(RenderFields).__init__()
        super(Shadow).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.model = None  # Type: studio
        self.hull_name = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', None)  # Type: studio
        instance.hull_name = entity_data.get('hull_name', "CHOICES NOT SUPPORTED")  # Type: choices


class prop_car_glass(prop_dynamic):
    def __init__(self):
        super(prop_dynamic).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_dynamic.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class prop_car_alarm(prop_physics, EnableDisable):
    def __init__(self):
        super(prop_physics).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_physics.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class func_ladder(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class trigger_auto_crouch(Trigger):
    def __init__(self):
        super(Trigger).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)


class trigger_active_weapon_detect(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        self.weaponclassname = "weapon_dieselcan"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.weaponclassname = entity_data.get('weaponclassname', "weapon_dieselcan")  # Type: string


class player_weaponstrip(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class NavBlocker(Base):
    def __init__(self):
        super().__init__()
        self.teamToBlock = "CHOICES NOT SUPPORTED"  # Type: choices
        self.affectsFlow = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.teamToBlock = entity_data.get('teamtoblock', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.affectsFlow = entity_data.get('affectsflow', None)  # Type: choices


class func_nav_blocker(Targetname, NavBlocker):
    def __init__(self):
        super(Targetname).__init__()
        super(NavBlocker).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        NavBlocker.from_dict(instance, entity_data)


class func_nav_avoidance_obstacle(Targetname, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)


class NavAttributeRegion(Base):
    def __init__(self):
        super().__init__()
        self.precise = None  # Type: choices
        self.crouch = None  # Type: choices
        self.stairs = None  # Type: choices
        self.remove_attributes = None  # Type: integer
        self.tank_only = None  # Type: choices
        self.mob_only = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.precise = entity_data.get('precise', None)  # Type: choices
        instance.crouch = entity_data.get('crouch', None)  # Type: choices
        instance.stairs = entity_data.get('stairs', None)  # Type: choices
        instance.remove_attributes = parse_source_value(entity_data.get('remove_attributes', 0))  # Type: integer
        instance.tank_only = entity_data.get('tank_only', None)  # Type: choices
        instance.mob_only = entity_data.get('mob_only', None)  # Type: choices


class func_nav_attribute_region(Targetname, NavAttributeRegion):
    def __init__(self):
        super(Targetname).__init__()
        super(NavAttributeRegion).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        NavAttributeRegion.from_dict(instance, entity_data)


class point_nav_attribute_region(Targetname, NavAttributeRegion):
    def __init__(self):
        super(Targetname).__init__()
        super(NavAttributeRegion).__init__()
        self.origin = [0, 0, 0]
        self.mins = [-4.0, -128.0, -80.0]  # Type: vector
        self.maxs = [4.0, 128.0, 80.0]  # Type: vector

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        NavAttributeRegion.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.mins = parse_float_vector(entity_data.get('mins', "-4 -128 -80"))  # Type: vector
        instance.maxs = parse_float_vector(entity_data.get('maxs', "4 128 80"))  # Type: vector


class func_elevator(Targetname, RenderFields, Parentname, Origin):
    def __init__(self):
        super(RenderFields).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(Origin).__init__()
        self.top = None  # Type: vecline
        self.bottom = None  # Type: vecline
        self.speed = 100  # Type: integer
        self.acceleration = 100  # Type: integer
        self.blockdamage = None  # Type: float
        self.startsound = None  # Type: sound
        self.stopsound = None  # Type: sound
        self.disablesound = None  # Type: sound

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.top = entity_data.get('top', None)  # Type: vecline
        instance.bottom = entity_data.get('bottom', None)  # Type: vecline
        instance.speed = parse_source_value(entity_data.get('speed', 100))  # Type: integer
        instance.acceleration = parse_source_value(entity_data.get('acceleration', 100))  # Type: integer
        instance.blockdamage = float(entity_data.get('blockdamage', 0))  # Type: float
        instance.startsound = entity_data.get('startsound', None)  # Type: sound
        instance.stopsound = entity_data.get('stopsound', None)  # Type: sound
        instance.disablesound = entity_data.get('disablesound', None)  # Type: sound


class info_elevator_floor(Targetname, Angles, Parentname):
    icon_sprite = "editor/info_target.vmt"

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


class logic_director_query(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.minAngerRange = 1  # Type: integer
        self.maxAngerRange = 10  # Type: integer
        self.noise = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.minAngerRange = parse_source_value(entity_data.get('minangerrange', 1))  # Type: integer
        instance.maxAngerRange = parse_source_value(entity_data.get('maxangerrange', 10))  # Type: integer
        instance.noise = entity_data.get('noise', None)  # Type: choices


class info_director(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_game_event_proxy(Targetname):
    icon_sprite = "editor/info_game_event_proxy.vmt"

    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.event_name = None  # Type: string
        self.range = 50  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.event_name = entity_data.get('event_name', None)  # Type: string
        instance.range = float(entity_data.get('range', 50))  # Type: float


class game_scavenge_progress_display(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.Max = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Max = float(entity_data.get('max', 0))  # Type: float


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


class filter_activator_team(BaseFilter):
    icon_sprite = "editor/filter_team.vmt"

    def __init__(self):
        super(BaseFilter).__init__()
        self.filterteam = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.filterteam = entity_data.get('filterteam', "CHOICES NOT SUPPORTED")  # Type: choices


class filter_activator_infected_class(BaseFilter):
    icon_sprite = "editor/filter_team.vmt"

    def __init__(self):
        super(BaseFilter).__init__()
        self.filterinfectedclass = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.filterinfectedclass = entity_data.get('filterinfectedclass', "CHOICES NOT SUPPORTED")  # Type: choices


class filter_melee_damage(BaseFilter):
    def __init__(self):
        super(BaseFilter).__init__()
        self.damagetype = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.damagetype = entity_data.get('damagetype', "CHOICES NOT SUPPORTED")  # Type: choices


class filter_health(BaseFilter):
    def __init__(self):
        super(BaseFilter).__init__()
        self.adrenalinepresence = "CHOICES NOT SUPPORTED"  # Type: choices
        self.healthmin = None  # Type: integer
        self.healthmax = 100  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.adrenalinepresence = entity_data.get('adrenalinepresence', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.healthmin = parse_source_value(entity_data.get('healthmin', 0))  # Type: integer
        instance.healthmax = parse_source_value(entity_data.get('healthmax', 100))  # Type: integer


class prop_minigun(EnableDisable, prop_dynamic_base):
    viewport_model = "models/w_models/weapons/w_minigun.mdl"

    def __init__(self):
        super(prop_dynamic_base).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.MaxYaw = 90  # Type: float
        self.MaxPitch = 60  # Type: float
        self.MinPitch = -30  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.MaxYaw = float(entity_data.get('maxyaw', 90))  # Type: float
        instance.MaxPitch = float(entity_data.get('maxpitch', 60))  # Type: float
        instance.MinPitch = float(entity_data.get('minpitch', -30))  # Type: float


class prop_mounted_machine_gun(EnableDisable, prop_dynamic_base):
    viewport_model = "models/w_models/weapons/50cal.mdl"

    def __init__(self):
        super(prop_dynamic_base).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.MaxYaw = 90  # Type: float
        self.MaxPitch = 60  # Type: float
        self.MinPitch = -30  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.MaxYaw = float(entity_data.get('maxyaw', 90))  # Type: float
        instance.MaxPitch = float(entity_data.get('maxpitch', 60))  # Type: float
        instance.MinPitch = float(entity_data.get('minpitch', -30))  # Type: float


class prop_health_cabinet(EnableDisable, prop_dynamic_base):
    def __init__(self):
        super(prop_dynamic_base).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.HealthCount = 1  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.HealthCount = parse_source_value(entity_data.get('healthcount', 1))  # Type: integer


class info_survivor_position(Targetname, Angles, Parentname):
    model = "models/survivors/survivor_coach.mdl"

    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.Order = 1  # Type: integer
        self.SurvivorName = None  # Type: string
        self.SurvivorIntroSequence = None  # Type: string
        self.GameMode = None  # Type: string
        self.SurvivorConcept = None  # Type: string
        self.HideWeapons = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Order = parse_source_value(entity_data.get('order', 1))  # Type: integer
        instance.SurvivorName = entity_data.get('survivorname', None)  # Type: string
        instance.SurvivorIntroSequence = entity_data.get('survivorintrosequence', None)  # Type: string
        instance.GameMode = entity_data.get('gamemode', None)  # Type: string
        instance.SurvivorConcept = entity_data.get('survivorconcept', None)  # Type: string
        instance.HideWeapons = entity_data.get('hideweapons', None)  # Type: choices


class info_survivor_rescue(Targetname, Angles, PlayerClass):
    model = "models/survivors/survivor_coach.mdl"

    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(PlayerClass).__init__()
        self.origin = [0, 0, 0]
        self.rescueEyePos = None  # Type: vecline
        self.model = "models/editor/playerstart.mdl"  # Type: studio

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        PlayerClass.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.rescueEyePos = entity_data.get('rescueeyepos', None)  # Type: vecline
        instance.model = entity_data.get('model', "models/editor/playerstart.mdl")  # Type: studio


class trigger_finale(Targetname, EnableDisable, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.model = "models/props_misc/german_radio.mdl"  # Type: studio
        self.disableshadows = None  # Type: choices
        self.FirstUseDelay = 0  # Type: float
        self.UseDelay = 0  # Type: float
        self.type = None  # Type: choices
        self.ScriptFile = None  # Type: string
        self.VersusTravelCompletion = 0.2  # Type: float
        self.IsSacrificeFinale = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', "models/props_misc/german_radio.mdl")  # Type: studio
        instance.disableshadows = entity_data.get('disableshadows', None)  # Type: choices
        instance.FirstUseDelay = float(entity_data.get('firstusedelay', 0))  # Type: float
        instance.UseDelay = float(entity_data.get('usedelay', 0))  # Type: float
        instance.type = entity_data.get('type', None)  # Type: choices
        instance.ScriptFile = entity_data.get('scriptfile', None)  # Type: string
        instance.VersusTravelCompletion = float(entity_data.get('versustravelcompletion', 0.2))  # Type: float
        instance.IsSacrificeFinale = entity_data.get('issacrificefinale', None)  # Type: choices


class trigger_standoff(Targetname, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.model = "models/props_misc/german_radio.mdl"  # Type: studio
        self.disableshadows = None  # Type: choices
        self.UseDuration = 0  # Type: float
        self.UseDelay = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', "models/props_misc/german_radio.mdl")  # Type: studio
        instance.disableshadows = entity_data.get('disableshadows', None)  # Type: choices
        instance.UseDuration = float(entity_data.get('useduration', 0))  # Type: float
        instance.UseDelay = float(entity_data.get('usedelay', 0))  # Type: float


class info_changelevel(Base):
    def __init__(self):
        super().__init__()
        self.targetname = None  # Type: target_source
        self.map = None  # Type: string
        self.landmark = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.targetname = entity_data.get('targetname', None)  # Type: target_source
        instance.map = entity_data.get('map', None)  # Type: string
        instance.landmark = entity_data.get('landmark', None)  # Type: target_destination


class prop_door_rotating_checkpoint(prop_door_rotating):
    def __init__(self):
        super(prop_door_rotating).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_door_rotating.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_zombie_spawn(Targetname, Angles, Parentname):
    model = "models/infected/common_male01.mdl"

    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.population = "default"  # Type: string
        self.offer_tank = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.population = entity_data.get('population', "default")  # Type: string
        instance.offer_tank = entity_data.get('offer_tank', None)  # Type: choices


class info_zombie_border(Targetname, EnableDisable, Angles, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_remarkable(Origin, Targetname):
    def __init__(self):
        super(Origin).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.contextsubject = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.contextsubject = entity_data.get('contextsubject', None)  # Type: string


class Weapon(Targetname, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)


class WeaponSpawnSingle(Parentname, Angles, Studiomodel, Targetname, Global):
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Studiomodel).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
        self.weaponskin = -1  # Type: integer
        self.glowrange = None  # Type: float
        self.solid = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        instance.weaponskin = parse_source_value(entity_data.get('weaponskin', -1))  # Type: integer
        instance.glowrange = float(entity_data.get('glowrange', 0))  # Type: float
        instance.solid = entity_data.get('solid', "CHOICES NOT SUPPORTED")  # Type: choices


class WeaponSpawn(WeaponSpawnSingle):
    def __init__(self):
        super(WeaponSpawnSingle).__init__()
        self.count = 5  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawnSingle.from_dict(instance, entity_data)
        instance.count = parse_source_value(entity_data.get('count', 5))  # Type: integer


class weapon_item_spawn(Targetname, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.weaponskin = -1  # Type: integer
        self.glowrange = None  # Type: float
        self.item1 = 1  # Type: integer
        self.item2 = None  # Type: integer
        self.item3 = 1  # Type: integer
        self.item4 = 1  # Type: integer
        self.item5 = 1  # Type: integer
        self.item6 = None  # Type: integer
        self.item7 = None  # Type: integer
        self.item8 = None  # Type: integer
        self.item11 = 1  # Type: integer
        self.item12 = None  # Type: integer
        self.item13 = None  # Type: integer
        self.item16 = None  # Type: integer
        self.item17 = None  # Type: integer
        self.item18 = None  # Type: integer
        self.melee_weapon = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.weaponskin = parse_source_value(entity_data.get('weaponskin', -1))  # Type: integer
        instance.glowrange = float(entity_data.get('glowrange', 0))  # Type: float
        instance.item1 = parse_source_value(entity_data.get('item1', 1))  # Type: integer
        instance.item2 = parse_source_value(entity_data.get('item2', 0))  # Type: integer
        instance.item3 = parse_source_value(entity_data.get('item3', 1))  # Type: integer
        instance.item4 = parse_source_value(entity_data.get('item4', 1))  # Type: integer
        instance.item5 = parse_source_value(entity_data.get('item5', 1))  # Type: integer
        instance.item6 = parse_source_value(entity_data.get('item6', 0))  # Type: integer
        instance.item7 = parse_source_value(entity_data.get('item7', 0))  # Type: integer
        instance.item8 = parse_source_value(entity_data.get('item8', 0))  # Type: integer
        instance.item11 = parse_source_value(entity_data.get('item11', 1))  # Type: integer
        instance.item12 = parse_source_value(entity_data.get('item12', 0))  # Type: integer
        instance.item13 = parse_source_value(entity_data.get('item13', 0))  # Type: integer
        instance.item16 = parse_source_value(entity_data.get('item16', 0))  # Type: integer
        instance.item17 = parse_source_value(entity_data.get('item17', 0))  # Type: integer
        instance.item18 = parse_source_value(entity_data.get('item18', 0))  # Type: integer
        instance.melee_weapon = entity_data.get('melee_weapon', None)  # Type: string


class upgrade_spawn(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.laser_sight = 1  # Type: integer
        self.upgradepack_incendiary = 1  # Type: integer
        self.upgradepack_explosive = 1  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.laser_sight = parse_source_value(entity_data.get('laser_sight', 1))  # Type: integer
        instance.upgradepack_incendiary = parse_source_value(
            entity_data.get('upgradepack_incendiary', 1))  # Type: integer
        instance.upgradepack_explosive = parse_source_value(
            entity_data.get('upgradepack_explosive', 1))  # Type: integer


class upgrade_ammo_explosive(Targetname, Angles):
    viewport_model = "models/props/terror/exploding_ammo.mdl"

    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.count = 4  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.count = parse_source_value(entity_data.get('count', 4))  # Type: integer


class upgrade_ammo_incendiary(Targetname, Angles):
    viewport_model = "models/props/terror/incendiary_ammo.mdl"

    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.count = 4  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.count = parse_source_value(entity_data.get('count', 4))  # Type: integer


class upgrade_laser_sight(Targetname, Angles):
    viewport_model = "models/w_models/Weapons/w_laser_sights.mdl"

    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_pistol_spawn(WeaponSpawn):
    viewport_model = "models/w_models/Weapons/w_pistol_a.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_pistol_magnum_spawn(WeaponSpawn):
    viewport_model = "models/w_models/Weapons/w_desert_eagle.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_smg_spawn(WeaponSpawn):
    viewport_model = "models/w_models/Weapons/w_smg_uzi.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_pumpshotgun_spawn(WeaponSpawn):
    viewport_model = "models/w_models/Weapons/w_shotgun.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_autoshotgun_spawn(WeaponSpawn):
    viewport_model = "models/w_models/Weapons/w_autoshot_m4super.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_rifle_spawn(WeaponSpawn):
    viewport_model = "models/w_models/Weapons/w_rifle_m16a2.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_hunting_rifle_spawn(WeaponSpawn):
    viewport_model = "models/w_models/Weapons/w_sniper_mini14.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_smg_silenced_spawn(WeaponSpawn):
    viewport_model = "models/w_models/Weapons/w_smg_a.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_shotgun_chrome_spawn(WeaponSpawn):
    viewport_model = "models/w_models/Weapons/w_pumpshotgun_A.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_shotgun_spas_spawn(WeaponSpawn):
    viewport_model = "models/w_models/Weapons/w_shotgun_spas.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_rifle_desert_spawn(WeaponSpawn):
    viewport_model = "models/w_models/Weapons/w_rifle_B.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_rifle_ak47_spawn(WeaponSpawn):
    viewport_model = "models/w_models/Weapons/w_rifle_ak47.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_sniper_military_spawn(WeaponSpawn):
    viewport_model = "models/w_models/Weapons/w_sniper_military.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_chainsaw_spawn(WeaponSpawn):
    viewport_model = "models/weapons/melee/w_chainsaw.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_grenade_launcher_spawn(WeaponSpawn):
    viewport_model = "models/w_models/weapons/w_grenade_launcher.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_rifle_m60_spawn(WeaponSpawn):
    viewport_model = "models/w_models/Weapons/w_m60.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_smg_mp5_spawn(WeaponSpawn):
    viewport_model = "models/w_models/Weapons/w_smg_mp5.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_rifle_sg552_spawn(WeaponSpawn):
    viewport_model = "models/w_models/Weapons/w_rifle_sg552.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_sniper_awp_spawn(WeaponSpawn):
    viewport_model = "models/w_models/Weapons/w_sniper_awp.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_sniper_scout_spawn(WeaponSpawn):
    viewport_model = "models/w_models/Weapons/w_sniper_scout.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_pipe_bomb_spawn(WeaponSpawn):
    viewport_model = "models/w_models/weapons/w_eq_pipebomb.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_molotov_spawn(WeaponSpawn):
    viewport_model = "models/w_models/weapons/w_eq_molotov.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_vomitjar_spawn(WeaponSpawn):
    viewport_model = "models/w_models/weapons/w_eq_bile_flask.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_first_aid_kit_spawn(WeaponSpawnSingle):
    viewport_model = "models/w_models/weapons/w_eq_Medkit.mdl"

    def __init__(self):
        super(WeaponSpawnSingle).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawnSingle.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_pain_pills_spawn(WeaponSpawnSingle):
    viewport_model = "models/w_models/weapons/w_eq_painpills.mdl"

    def __init__(self):
        super(WeaponSpawnSingle).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawnSingle.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_adrenaline_spawn(WeaponSpawnSingle):
    viewport_model = "models/w_models/weapons/w_eq_adrenaline.mdl"

    def __init__(self):
        super(WeaponSpawnSingle).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawnSingle.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_defibrillator_spawn(WeaponSpawnSingle):
    viewport_model = "models/w_models/weapons/w_eq_defibrillator.mdl"

    def __init__(self):
        super(WeaponSpawnSingle).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawnSingle.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_gascan_spawn(WeaponSpawnSingle):
    viewport_model = "models/props_junk/gascan001a.mdl"

    def __init__(self):
        super(WeaponSpawnSingle).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawnSingle.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_upgradepack_incendiary_spawn(WeaponSpawnSingle):
    viewport_model = "models/w_models/weapons/w_eq_incendiary_ammopack.mdl"

    def __init__(self):
        super(WeaponSpawnSingle).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawnSingle.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_upgradepack_explosive_spawn(WeaponSpawnSingle):
    viewport_model = "models/w_models/weapons/w_eq_explosive_ammopack.mdl"

    def __init__(self):
        super(WeaponSpawnSingle).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawnSingle.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_first_aid_kit(WeaponSpawnSingle):
    viewport_model = "models/w_models/weapons/w_eq_Medkit.mdl"

    def __init__(self):
        super(WeaponSpawnSingle).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawnSingle.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_grenade_launcher(WeaponSpawnSingle):
    viewport_model = "models/w_models/weapons/w_grenade_launcher.mdl"

    def __init__(self):
        super(WeaponSpawnSingle).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawnSingle.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_melee_spawn(WeaponSpawn):
    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]
        self.melee_weapon = "any"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.melee_weapon = entity_data.get('melee_weapon', "any")  # Type: string


class weapon_scavenge_item_spawn(WeaponSpawnSingle):
    viewport_model = "models/props_junk/gascan001a.mdl"

    def __init__(self):
        super(WeaponSpawnSingle).__init__()
        self.origin = [0, 0, 0]
        self.glowstate = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawnSingle.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.glowstate = entity_data.get('glowstate', "CHOICES NOT SUPPORTED")  # Type: choices


class point_prop_use_target(Origin, Targetname):
    def __init__(self):
        super(Origin).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.nozzle = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.nozzle = entity_data.get('nozzle', None)  # Type: target_destination


class weapon_spawn(WeaponSpawn):
    viewport_model = "models/w_models/Weapons/w_rifle_m16a2.mdl"

    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]
        self.weapon_selection = "CHOICES NOT SUPPORTED"  # Type: choices
        self.spawn_without_director = None  # Type: choices
        self.no_cs_weapons = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.weapon_selection = entity_data.get('weapon_selection', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.spawn_without_director = entity_data.get('spawn_without_director', None)  # Type: choices
        instance.no_cs_weapons = entity_data.get('no_cs_weapons', None)  # Type: choices


class weapon_ammo_spawn(WeaponSpawn):
    def __init__(self):
        super(WeaponSpawn).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        WeaponSpawn.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_map_parameters(Targetname, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.AmmoDensity = 6.48  # Type: float
        self.PainPillDensity = 6.48  # Type: float
        self.MolotovDensity = 6.48  # Type: float
        self.PipeBombDensity = 6.48  # Type: float
        self.PistolDensity = 6.48  # Type: float
        self.GasCanDensity = 6.48  # Type: float
        self.OxygenTankDensity = 6.48  # Type: float
        self.PropaneTankDensity = 6.48  # Type: float
        self.MeleeWeaponDensity = 6.48  # Type: float
        self.AdrenalineDensity = 6.48  # Type: float
        self.DefibrillatorDensity = 3.0  # Type: float
        self.VomitJarDensity = 6.48  # Type: float
        self.UpgradepackDensity = 1.0  # Type: float
        self.ChainsawDensity = 1.0  # Type: float
        self.ConfigurableWeaponDensity = -1.0  # Type: float
        self.ConfigurableWeaponClusterRange = 100  # Type: float
        self.MagnumDensity = -1.0  # Type: float
        self.ItemClusterRange = 50  # Type: float
        self.FinaleItemClusterCount = 3  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.AmmoDensity = float(entity_data.get('ammodensity', 6.48))  # Type: float
        instance.PainPillDensity = float(entity_data.get('painpilldensity', 6.48))  # Type: float
        instance.MolotovDensity = float(entity_data.get('molotovdensity', 6.48))  # Type: float
        instance.PipeBombDensity = float(entity_data.get('pipebombdensity', 6.48))  # Type: float
        instance.PistolDensity = float(entity_data.get('pistoldensity', 6.48))  # Type: float
        instance.GasCanDensity = float(entity_data.get('gascandensity', 6.48))  # Type: float
        instance.OxygenTankDensity = float(entity_data.get('oxygentankdensity', 6.48))  # Type: float
        instance.PropaneTankDensity = float(entity_data.get('propanetankdensity', 6.48))  # Type: float
        instance.MeleeWeaponDensity = float(entity_data.get('meleeweapondensity', 6.48))  # Type: float
        instance.AdrenalineDensity = float(entity_data.get('adrenalinedensity', 6.48))  # Type: float
        instance.DefibrillatorDensity = float(entity_data.get('defibrillatordensity', 3.0))  # Type: float
        instance.VomitJarDensity = float(entity_data.get('vomitjardensity', 6.48))  # Type: float
        instance.UpgradepackDensity = float(entity_data.get('upgradepackdensity', 1.0))  # Type: float
        instance.ChainsawDensity = float(entity_data.get('chainsawdensity', 1.0))  # Type: float
        instance.ConfigurableWeaponDensity = float(entity_data.get('configurableweapondensity', -1.0))  # Type: float
        instance.ConfigurableWeaponClusterRange = float(
            entity_data.get('configurableweaponclusterrange', 100))  # Type: float
        instance.MagnumDensity = float(entity_data.get('magnumdensity', -1.0))  # Type: float
        instance.ItemClusterRange = float(entity_data.get('itemclusterrange', 50))  # Type: float
        instance.FinaleItemClusterCount = parse_source_value(
            entity_data.get('finaleitemclustercount', 3))  # Type: integer


class info_map_parameters_versus(Targetname, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.AmmoDensity = 6.48  # Type: float
        self.PainPillDensity = 6.48  # Type: float
        self.MolotovDensity = 6.48  # Type: float
        self.PipeBombDensity = 6.48  # Type: float
        self.PistolDensity = 6.48  # Type: float
        self.GasCanDensity = 6.48  # Type: float
        self.OxygenTankDensity = 6.48  # Type: float
        self.PropaneTankDensity = 6.48  # Type: float
        self.MeleeWeaponDensity = 6.48  # Type: float
        self.AdrenalineDensity = 6.48  # Type: float
        self.DefibrillatorDensity = 2.50  # Type: float
        self.VomitJarDensity = 6.48  # Type: float
        self.UpgradepackDensity = 1.0  # Type: float
        self.ChainsawDensity = 1.0  # Type: float
        self.ConfigurableWeaponDensity = -1.0  # Type: float
        self.ConfigurableWeaponClusterRange = 100  # Type: float
        self.MagnumDensity = -1.0  # Type: float
        self.ItemClusterRange = 50  # Type: float
        self.FinaleItemClusterCount = 3  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.AmmoDensity = float(entity_data.get('ammodensity', 6.48))  # Type: float
        instance.PainPillDensity = float(entity_data.get('painpilldensity', 6.48))  # Type: float
        instance.MolotovDensity = float(entity_data.get('molotovdensity', 6.48))  # Type: float
        instance.PipeBombDensity = float(entity_data.get('pipebombdensity', 6.48))  # Type: float
        instance.PistolDensity = float(entity_data.get('pistoldensity', 6.48))  # Type: float
        instance.GasCanDensity = float(entity_data.get('gascandensity', 6.48))  # Type: float
        instance.OxygenTankDensity = float(entity_data.get('oxygentankdensity', 6.48))  # Type: float
        instance.PropaneTankDensity = float(entity_data.get('propanetankdensity', 6.48))  # Type: float
        instance.MeleeWeaponDensity = float(entity_data.get('meleeweapondensity', 6.48))  # Type: float
        instance.AdrenalineDensity = float(entity_data.get('adrenalinedensity', 6.48))  # Type: float
        instance.DefibrillatorDensity = float(entity_data.get('defibrillatordensity', 2.50))  # Type: float
        instance.VomitJarDensity = float(entity_data.get('vomitjardensity', 6.48))  # Type: float
        instance.UpgradepackDensity = float(entity_data.get('upgradepackdensity', 1.0))  # Type: float
        instance.ChainsawDensity = float(entity_data.get('chainsawdensity', 1.0))  # Type: float
        instance.ConfigurableWeaponDensity = float(entity_data.get('configurableweapondensity', -1.0))  # Type: float
        instance.ConfigurableWeaponClusterRange = float(
            entity_data.get('configurableweaponclusterrange', 100))  # Type: float
        instance.MagnumDensity = float(entity_data.get('magnumdensity', -1.0))  # Type: float
        instance.ItemClusterRange = float(entity_data.get('itemclusterrange', 50))  # Type: float
        instance.FinaleItemClusterCount = parse_source_value(
            entity_data.get('finaleitemclustercount', 3))  # Type: integer


class info_gamemode(Targetname, Angles):
    icon_sprite = "editor/info_gamemode.vmt"

    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class beam_spotlight(Targetname, RenderFields, Angles, Parentname):
    model = "models/editor/cone_helper.mdl"

    def __init__(self):
        super(RenderFields).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.maxspeed = 100  # Type: integer
        self.spotlightlength = 500  # Type: integer
        self.spotlightwidth = 50  # Type: integer
        self.HDRColorScale = 0.7  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.maxspeed = parse_source_value(entity_data.get('maxspeed', 100))  # Type: integer
        instance.spotlightlength = parse_source_value(entity_data.get('spotlightlength', 500))  # Type: integer
        instance.spotlightwidth = parse_source_value(entity_data.get('spotlightwidth', 50))  # Type: integer
        instance.HDRColorScale = float(entity_data.get('hdrcolorscale', 0.7))  # Type: float


class env_detail_controller(Angles):
    icon_sprite = "editor/env_particles.vmt"

    def __init__(self):
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.fademindist = 512  # Type: integer
        self.fademaxdist = 1024  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.fademindist = parse_source_value(entity_data.get('fademindist', 512))  # Type: integer
        instance.fademaxdist = parse_source_value(entity_data.get('fademaxdist', 1024))  # Type: integer


class info_goal_infected_chase(Targetname, Parentname):
    icon_sprite = "editor/info_target.vmt"

    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class func_playerinfected_clip(Parentname, RenderFields, Inputfilter, EnableDisable, Shadow, Targetname, Global):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Inputfilter).__init__()
        super(EnableDisable).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
        self.Solidity = "CHOICES NOT SUPPORTED"  # Type: choices
        self.vrad_brush_cast_shadows = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Inputfilter.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        instance.Solidity = entity_data.get('solidity', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.vrad_brush_cast_shadows = entity_data.get('vrad_brush_cast_shadows', None)  # Type: choices


class func_playerghostinfected_clip(Parentname, RenderFields, Inputfilter, EnableDisable, Shadow, Targetname, Global):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Inputfilter).__init__()
        super(EnableDisable).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
        self.Solidity = "CHOICES NOT SUPPORTED"  # Type: choices
        self.vrad_brush_cast_shadows = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Inputfilter.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        instance.Solidity = entity_data.get('solidity', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.vrad_brush_cast_shadows = entity_data.get('vrad_brush_cast_shadows', None)  # Type: choices


class commentary_dummy(Targetname, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.model = "models/survivors/survivor_coach.mdl"  # Type: studio
        self.EyeHeight = 64  # Type: integer
        self.StartingAnim = "Idle_Calm_Pistol"  # Type: string
        self.StartingWeapons = "weapon_pistol"  # Type: string
        self.LookAtPlayers = None  # Type: choices
        self.HeadYawPoseParam = "Head_Yaw"  # Type: string
        self.HeadPitchPoseParam = "Head_Pitch"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', "models/survivors/survivor_coach.mdl")  # Type: studio
        instance.EyeHeight = parse_source_value(entity_data.get('eyeheight', 64))  # Type: integer
        instance.StartingAnim = entity_data.get('startinganim', "Idle_Calm_Pistol")  # Type: string
        instance.StartingWeapons = entity_data.get('startingweapons', "weapon_pistol")  # Type: string
        instance.LookAtPlayers = entity_data.get('lookatplayers', None)  # Type: choices
        instance.HeadYawPoseParam = entity_data.get('headyawposeparam', "Head_Yaw")  # Type: string
        instance.HeadPitchPoseParam = entity_data.get('headpitchposeparam', "Head_Pitch")  # Type: string


class commentary_zombie_spawner(Targetname, Angles, Parentname):
    model = "models/infected/smoker.mdl"

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


class env_outtro_stats(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class trigger_hurt_ghost(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        super(Targetname).__init__()
        self.master = None  # Type: string
        self.damage = 10  # Type: integer
        self.damagecap = 20  # Type: integer
        self.damagetype = None  # Type: choices
        self.damagemodel = None  # Type: choices
        self.nodmgforce = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.master = entity_data.get('master', None)  # Type: string
        instance.damage = parse_source_value(entity_data.get('damage', 10))  # Type: integer
        instance.damagecap = parse_source_value(entity_data.get('damagecap', 20))  # Type: integer
        instance.damagetype = entity_data.get('damagetype', None)  # Type: choices
        instance.damagemodel = entity_data.get('damagemodel', None)  # Type: choices
        instance.nodmgforce = entity_data.get('nodmgforce', None)  # Type: choices


class func_nav_connection_blocker(Targetname, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)


class env_player_blocker(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.mins = [-4.0, -128.0, -80.0]  # Type: vector
        self.maxs = [4.0, 128.0, 80.0]  # Type: vector
        self.initialstate = "CHOICES NOT SUPPORTED"  # Type: choices
        self.BlockType = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.mins = parse_float_vector(entity_data.get('mins', "-4 -128 -80"))  # Type: vector
        instance.maxs = parse_float_vector(entity_data.get('maxs', "4 128 80"))  # Type: vector
        instance.initialstate = entity_data.get('initialstate', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.BlockType = entity_data.get('blocktype', None)  # Type: choices


class env_physics_blocker(Targetname, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.boxmins = [-8.0, -8.0, -8.0]  # Type: vector
        self.boxmaxs = [8.0, 8.0, 8.0]  # Type: vector
        self.initialstate = "CHOICES NOT SUPPORTED"  # Type: choices
        self.BlockType = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.boxmins = parse_float_vector(entity_data.get('boxmins', "-8 -8 -8"))  # Type: vector
        instance.boxmaxs = parse_float_vector(entity_data.get('boxmaxs', "8 8 8"))  # Type: vector
        instance.initialstate = entity_data.get('initialstate', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.BlockType = entity_data.get('blocktype', None)  # Type: choices


class trigger_upgrade_laser_sight(Trigger):
    def __init__(self):
        super(Trigger).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)


class logic_game_event(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.eventName = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.eventName = entity_data.get('eventname', None)  # Type: string


class func_button_timed(Parentname, RenderFields, Origin, DamageFilter, Targetname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Origin).__init__()
        super(DamageFilter).__init__()
        super(Targetname).__init__()
        self.use_time = 5  # Type: integer
        self.use_string = "Using...."  # Type: string
        self.use_sub_string = None  # Type: string
        self.glow = None  # Type: target_destination
        self.auto_disable = "CHOICES NOT SUPPORTED"  # Type: choices
        self.locked_sound = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        DamageFilter.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.use_time = parse_source_value(entity_data.get('use_time', 5))  # Type: integer
        instance.use_string = entity_data.get('use_string', "Using....")  # Type: string
        instance.use_sub_string = entity_data.get('use_sub_string', None)  # Type: string
        instance.glow = entity_data.get('glow', None)  # Type: target_destination
        instance.auto_disable = entity_data.get('auto_disable', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.locked_sound = entity_data.get('locked_sound', None)  # Type: choices


class prop_fuel_barrel(Targetname, Studiomodel, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(Studiomodel).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.fademindist = -1  # Type: float
        self.fademaxdist = None  # Type: float
        self.fadescale = 1  # Type: float
        self.BasePiece = "models/props_industrial/barrel_fuel_partb.mdl"  # Type: studio
        self.FlyingPiece01 = "models/props_industrial/barrel_fuel_parta.mdl"  # Type: studio
        self.FlyingPiece02 = None  # Type: studio
        self.FlyingPiece03 = None  # Type: studio
        self.FlyingPiece04 = None  # Type: studio
        self.DetonateParticles = "weapon_pipebomb"  # Type: string
        self.FlyingParticles = "barrel_fly"  # Type: string
        self.DetonateSound = "BaseGrenade.Explode"  # Type: sound

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.fademindist = float(entity_data.get('fademindist', -1))  # Type: float
        instance.fademaxdist = float(entity_data.get('fademaxdist', 0))  # Type: float
        instance.fadescale = float(entity_data.get('fadescale', 1))  # Type: float
        instance.BasePiece = entity_data.get('basepiece',
                                             "models/props_industrial/barrel_fuel_partb.mdl")  # Type: studio
        instance.FlyingPiece01 = entity_data.get('flyingpiece01',
                                                 "models/props_industrial/barrel_fuel_parta.mdl")  # Type: studio
        instance.FlyingPiece02 = entity_data.get('flyingpiece02', None)  # Type: studio
        instance.FlyingPiece03 = entity_data.get('flyingpiece03', None)  # Type: studio
        instance.FlyingPiece04 = entity_data.get('flyingpiece04', None)  # Type: studio
        instance.DetonateParticles = entity_data.get('detonateparticles', "weapon_pipebomb")  # Type: string
        instance.FlyingParticles = entity_data.get('flyingparticles', "barrel_fly")  # Type: string
        instance.DetonateSound = entity_data.get('detonatesound', "BaseGrenade.Explode")  # Type: sound


class logic_versus_random(Targetname):
    icon_sprite = "editor/logic_auto.vmt"

    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_weaponfire(Targetname, EnableDisable, Angles, Parentname):
    model = "models/editor/cone_helper.mdl"

    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.TargetArc = 40  # Type: float
        self.TargetRange = 3600  # Type: float
        self.filtername = None  # Type: filterclass
        self.DamageMod = 1.0  # Type: float
        self.WeaponType = "CHOICES NOT SUPPORTED"  # Type: choices
        self.TargetTeam = "CHOICES NOT SUPPORTED"  # Type: choices
        self.IgnorePlayers = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.TargetArc = float(entity_data.get('targetarc', 40))  # Type: float
        instance.TargetRange = float(entity_data.get('targetrange', 3600))  # Type: float
        instance.filtername = entity_data.get('filtername', None)  # Type: filterclass
        instance.DamageMod = float(entity_data.get('damagemod', 1.0))  # Type: float
        instance.WeaponType = entity_data.get('weapontype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.TargetTeam = entity_data.get('targetteam', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.IgnorePlayers = entity_data.get('ignoreplayers', None)  # Type: choices


class env_rock_launcher(Targetname, Angles):
    model = "models/editor/cone_helper.mdl"

    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.RockTargetName = None  # Type: target_destination
        self.RockDamageOverride = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.RockTargetName = entity_data.get('rocktargetname', None)  # Type: target_destination
        instance.RockDamageOverride = parse_source_value(entity_data.get('rockdamageoverride', 0))  # Type: integer


class func_extinguisher(Targetname, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)


class func_ragdoll_fader(Targetname, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)


class prop_minigun_l4d1(EnableDisable, prop_dynamic_base):
    viewport_model = "models/w_models/weapons/w_minigun.mdl"

    def __init__(self):
        super(prop_dynamic_base).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.MaxYaw = 90  # Type: float
        self.MaxPitch = 60  # Type: float
        self.MinPitch = -30  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.MaxYaw = float(entity_data.get('maxyaw', 90))  # Type: float
        instance.MaxPitch = float(entity_data.get('maxpitch', 60))  # Type: float
        instance.MinPitch = float(entity_data.get('minpitch', -30))  # Type: float


class trigger_escape(Trigger):
    def __init__(self):
        super(Trigger).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)


class func_buildable_button(Targetname, Parentname, Origin):
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(Origin).__init__()
        self.is_cumulative_use = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.is_cumulative_use = entity_data.get('is_cumulative_use', None)  # Type: choices


class point_script_use_target(Origin, Targetname):
    def __init__(self):
        super(Origin).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.model = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', None)  # Type: target_destination


class scripted_item_drop(Targetname, Studiomodel, Angles, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(Studiomodel).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


entity_class_handle = {
    'Angles': Angles,
    'Origin': Origin,
    'Studiomodel': Studiomodel,
    'BasePlat': BasePlat,
    'Targetname': Targetname,
    'Parentname': Parentname,
    'BaseBrush': BaseBrush,
    'EnableDisable': EnableDisable,
    'RenderFxChoices': RenderFxChoices,
    'Shadow': Shadow,
    'Glow': Glow,
    'SystemLevelChoice': SystemLevelChoice,
    'RenderFields': RenderFields,
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
    'ambient_music': ambient_music,
    'sound_mix_layer': sound_mix_layer,
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
    'trigger_tonemap': trigger_tonemap,
    'env_tonemap_controller': env_tonemap_controller,
    'env_tonemap_controller_infected': env_tonemap_controller_infected,
    'env_tonemap_controller_ghost': env_tonemap_controller_ghost,
    'func_useableladder': func_useableladder,
    'func_ladderendpoint': func_ladderendpoint,
    'info_ladder_dismount': info_ladder_dismount,
    'func_areaportalwindow': func_areaportalwindow,
    'func_wall': func_wall,
    'func_clip_vphysics': func_clip_vphysics,
    'func_brush': func_brush,
    'vgui_screen_base': vgui_screen_base,
    'vgui_screen': vgui_screen,
    'vgui_slideshow_display': vgui_slideshow_display,
    'cycler': cycler,
    'func_orator': func_orator,
    'gibshooterbase': gibshooterbase,
    'env_beam': env_beam,
    'env_beverage': env_beverage,
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
    'postprocess_controller': postprocess_controller,
    'env_steam': env_steam,
    'env_laser': env_laser,
    'env_message': env_message,
    'env_hudhint': env_hudhint,
    'env_shake': env_shake,
    'env_viewpunch': env_viewpunch,
    'env_rotorwash_emitter': env_rotorwash_emitter,
    'gibshooter': gibshooter,
    'env_shooter': env_shooter,
    'env_rotorshooter': env_rotorshooter,
    'env_soundscape_proxy': env_soundscape_proxy,
    'env_soundscape': env_soundscape,
    'env_soundscape_triggerable': env_soundscape_triggerable,
    'env_spark': env_spark,
    'env_sprite': env_sprite,
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
    'point_broadcastclientcommand': point_broadcastclientcommand,
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
    'info_particle_target': info_particle_target,
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
    'light_directional': light_directional,
    'light_spot': light_spot,
    'light_dynamic': light_dynamic,
    'shadow_control': shadow_control,
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
    'BaseFadeProp': BaseFadeProp,
    'prop_door_rotating': prop_door_rotating,
    'prop_wall_breakable': prop_wall_breakable,
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
    'point_posecontroller': point_posecontroller,
    'logic_compare': logic_compare,
    'logic_branch': logic_branch,
    'logic_branch_listener': logic_branch_listener,
    'logic_case': logic_case,
    'logic_multicompare': logic_multicompare,
    'logic_relay': logic_relay,
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
    'point_push': point_push,
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
    'func_precipitation_blocker': func_precipitation_blocker,
    'func_detail_blocker': func_detail_blocker,
    'func_wall_toggle': func_wall_toggle,
    'func_guntarget': func_guntarget,
    'func_fish_pool': func_fish_pool,
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
    'trigger_impact': trigger_impact,
    'trigger_proximity': trigger_proximity,
    'trigger_teleport': trigger_teleport,
    'trigger_transition': trigger_transition,
    'trigger_serverragdoll': trigger_serverragdoll,
    'ai_speechfilter': ai_speechfilter,
    'water_lod_control': water_lod_control,
    'info_camera_link': info_camera_link,
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
    'func_instance': func_instance,
    'func_instance_parms': func_instance_parms,
    'env_instructor_hint': env_instructor_hint,
    'info_target_instructor_hint': info_target_instructor_hint,
    'logic_script': logic_script,
    'func_timescale': func_timescale,
    'func_block_charge': func_block_charge,
    'info_ambient_mob_start': info_ambient_mob_start,
    'info_ambient_mob_end': info_ambient_mob_end,
    'info_ambient_mob': info_ambient_mob,
    'info_item_position': info_item_position,
    'info_l4d1_survivor_spawn': info_l4d1_survivor_spawn,
    'env_airstrike_indoors': env_airstrike_indoors,
    'env_airstrike_outdoors': env_airstrike_outdoors,
    'point_viewcontrol_multiplayer': point_viewcontrol_multiplayer,
    'point_viewcontrol_survivor': point_viewcontrol_survivor,
    'point_deathfall_camera': point_deathfall_camera,
    'logic_choreographed_scene': logic_choreographed_scene,
    'logic_scene_list_manager': logic_scene_list_manager,
    'generic_actor': generic_actor,
    'prop_car_glass': prop_car_glass,
    'prop_car_alarm': prop_car_alarm,
    'func_ladder': func_ladder,
    'trigger_auto_crouch': trigger_auto_crouch,
    'trigger_active_weapon_detect': trigger_active_weapon_detect,
    'player_weaponstrip': player_weaponstrip,
    'NavBlocker': NavBlocker,
    'func_nav_blocker': func_nav_blocker,
    'func_nav_avoidance_obstacle': func_nav_avoidance_obstacle,
    'NavAttributeRegion': NavAttributeRegion,
    'func_nav_attribute_region': func_nav_attribute_region,
    'point_nav_attribute_region': point_nav_attribute_region,
    'func_elevator': func_elevator,
    'info_elevator_floor': info_elevator_floor,
    'logic_director_query': logic_director_query,
    'info_director': info_director,
    'info_game_event_proxy': info_game_event_proxy,
    'game_scavenge_progress_display': game_scavenge_progress_display,
    'fog_volume': fog_volume,
    'filter_activator_team': filter_activator_team,
    'filter_activator_infected_class': filter_activator_infected_class,
    'filter_melee_damage': filter_melee_damage,
    'filter_health': filter_health,
    'prop_minigun': prop_minigun,
    'prop_mounted_machine_gun': prop_mounted_machine_gun,
    'prop_health_cabinet': prop_health_cabinet,
    'info_survivor_position': info_survivor_position,
    'info_survivor_rescue': info_survivor_rescue,
    'trigger_finale': trigger_finale,
    'trigger_standoff': trigger_standoff,
    'info_changelevel': info_changelevel,
    'prop_door_rotating_checkpoint': prop_door_rotating_checkpoint,
    'info_zombie_spawn': info_zombie_spawn,
    'info_zombie_border': info_zombie_border,
    'info_remarkable': info_remarkable,
    'Weapon': Weapon,
    'WeaponSpawnSingle': WeaponSpawnSingle,
    'WeaponSpawn': WeaponSpawn,
    'weapon_item_spawn': weapon_item_spawn,
    'upgrade_spawn': upgrade_spawn,
    'upgrade_ammo_explosive': upgrade_ammo_explosive,
    'upgrade_ammo_incendiary': upgrade_ammo_incendiary,
    'upgrade_laser_sight': upgrade_laser_sight,
    'weapon_pistol_spawn': weapon_pistol_spawn,
    'weapon_pistol_magnum_spawn': weapon_pistol_magnum_spawn,
    'weapon_smg_spawn': weapon_smg_spawn,
    'weapon_pumpshotgun_spawn': weapon_pumpshotgun_spawn,
    'weapon_autoshotgun_spawn': weapon_autoshotgun_spawn,
    'weapon_rifle_spawn': weapon_rifle_spawn,
    'weapon_hunting_rifle_spawn': weapon_hunting_rifle_spawn,
    'weapon_smg_silenced_spawn': weapon_smg_silenced_spawn,
    'weapon_shotgun_chrome_spawn': weapon_shotgun_chrome_spawn,
    'weapon_shotgun_spas_spawn': weapon_shotgun_spas_spawn,
    'weapon_rifle_desert_spawn': weapon_rifle_desert_spawn,
    'weapon_rifle_ak47_spawn': weapon_rifle_ak47_spawn,
    'weapon_sniper_military_spawn': weapon_sniper_military_spawn,
    'weapon_chainsaw_spawn': weapon_chainsaw_spawn,
    'weapon_grenade_launcher_spawn': weapon_grenade_launcher_spawn,
    'weapon_rifle_m60_spawn': weapon_rifle_m60_spawn,
    'weapon_smg_mp5_spawn': weapon_smg_mp5_spawn,
    'weapon_rifle_sg552_spawn': weapon_rifle_sg552_spawn,
    'weapon_sniper_awp_spawn': weapon_sniper_awp_spawn,
    'weapon_sniper_scout_spawn': weapon_sniper_scout_spawn,
    'weapon_pipe_bomb_spawn': weapon_pipe_bomb_spawn,
    'weapon_molotov_spawn': weapon_molotov_spawn,
    'weapon_vomitjar_spawn': weapon_vomitjar_spawn,
    'weapon_first_aid_kit_spawn': weapon_first_aid_kit_spawn,
    'weapon_pain_pills_spawn': weapon_pain_pills_spawn,
    'weapon_adrenaline_spawn': weapon_adrenaline_spawn,
    'weapon_defibrillator_spawn': weapon_defibrillator_spawn,
    'weapon_gascan_spawn': weapon_gascan_spawn,
    'weapon_upgradepack_incendiary_spawn': weapon_upgradepack_incendiary_spawn,
    'weapon_upgradepack_explosive_spawn': weapon_upgradepack_explosive_spawn,
    'weapon_first_aid_kit': weapon_first_aid_kit,
    'weapon_grenade_launcher': weapon_grenade_launcher,
    'weapon_melee_spawn': weapon_melee_spawn,
    'weapon_scavenge_item_spawn': weapon_scavenge_item_spawn,
    'point_prop_use_target': point_prop_use_target,
    'weapon_spawn': weapon_spawn,
    'weapon_ammo_spawn': weapon_ammo_spawn,
    'info_map_parameters': info_map_parameters,
    'info_map_parameters_versus': info_map_parameters_versus,
    'info_gamemode': info_gamemode,
    'beam_spotlight': beam_spotlight,
    'env_detail_controller': env_detail_controller,
    'info_goal_infected_chase': info_goal_infected_chase,
    'func_playerinfected_clip': func_playerinfected_clip,
    'func_playerghostinfected_clip': func_playerghostinfected_clip,
    'commentary_dummy': commentary_dummy,
    'commentary_zombie_spawner': commentary_zombie_spawner,
    'env_outtro_stats': env_outtro_stats,
    'trigger_hurt_ghost': trigger_hurt_ghost,
    'func_nav_connection_blocker': func_nav_connection_blocker,
    'env_player_blocker': env_player_blocker,
    'env_physics_blocker': env_physics_blocker,
    'trigger_upgrade_laser_sight': trigger_upgrade_laser_sight,
    'logic_game_event': logic_game_event,
    'func_button_timed': func_button_timed,
    'prop_fuel_barrel': prop_fuel_barrel,
    'logic_versus_random': logic_versus_random,
    'env_weaponfire': env_weaponfire,
    'env_rock_launcher': env_rock_launcher,
    'func_extinguisher': func_extinguisher,
    'func_ragdoll_fader': func_ragdoll_fader,
    'prop_minigun_l4d1': prop_minigun_l4d1,
    'trigger_escape': trigger_escape,
    'func_buildable_button': func_buildable_button,
    'point_script_use_target': point_script_use_target,
    'scripted_item_drop': scripted_item_drop,
}
