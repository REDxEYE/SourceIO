
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


class Empty(Base):
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


class Studiomodel(Base):
    def __init__(self):
        super().__init__()
        self.model = None  # Type: studio
        self.skin = None  # Type: integer
        self.modelscale = 1.0  # Type: float
        self.disableshadows = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', None)  # Type: studio
        instance.skin = parse_source_value(entity_data.get('skin', 0))  # Type: integer
        instance.modelscale = float(entity_data.get('modelscale', 1.0))  # Type: float
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

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.targetname = entity_data.get('targetname', None)  # Type: target_source


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


class RenderFields(RenderFxChoices):
    def __init__(self):
        super(RenderFxChoices).__init__()
        self.rendermode = None  # Type: choices
        self.renderamt = 255  # Type: integer
        self.rendercolor = [255, 255, 255]  # Type: color255
        self.disablereceiveshadows = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        RenderFxChoices.from_dict(instance, entity_data)
        instance.rendermode = entity_data.get('rendermode', None)  # Type: choices
        instance.renderamt = parse_source_value(entity_data.get('renderamt', 255))  # Type: integer
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "255 255 255"))  # Type: color255
        instance.disablereceiveshadows = entity_data.get('disablereceiveshadows', None)  # Type: choices


class DXLevelChoice(Base):
    def __init__(self):
        super().__init__()
        self.mindxlevel = None  # Type: choices
        self.maxdxlevel = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.mindxlevel = entity_data.get('mindxlevel', None)  # Type: choices
        instance.maxdxlevel = entity_data.get('maxdxlevel', None)  # Type: choices


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


class Breakable(Shadow, Targetname, DamageFilter):
    def __init__(self):
        super(Shadow).__init__()
        super(Targetname).__init__()
        super(DamageFilter).__init__()
        self.ExplodeDamage = None  # Type: float
        self.ExplodeRadius = None  # Type: float
        self.PerformanceMode = None  # Type: choices
        self.BreakModelMessage = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        DamageFilter.from_dict(instance, entity_data)
        instance.ExplodeDamage = float(entity_data.get('explodedamage', 0))  # Type: float
        instance.ExplodeRadius = float(entity_data.get('exploderadius', 0))  # Type: float
        instance.PerformanceMode = entity_data.get('performancemode', None)  # Type: choices
        instance.BreakModelMessage = entity_data.get('breakmodelmessage', None)  # Type: string


class BreakableBrush(Parentname, Global, Breakable):
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
        Global.from_dict(instance, entity_data)
        Breakable.from_dict(instance, entity_data)
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


class BaseNPC(Shadow, RenderFields, ResponseContext, DamageFilter, Angles, Targetname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Shadow).__init__()
        super(ResponseContext).__init__()
        super(DamageFilter).__init__()
        super(Angles).__init__()
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
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        ResponseContext.from_dict(instance, entity_data)
        DamageFilter.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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


class BaseNPCAssault(BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class info_npc_spawn_destination(Parentname, Angles, Targetname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.ReuseDelay = 1  # Type: float
        self.RenameNPC = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.ReuseDelay = float(entity_data.get('reusedelay', 1))  # Type: float
        instance.RenameNPC = entity_data.get('renamenpc', None)  # Type: string


class BaseNPCMaker(Angles, Targetname, EnableDisable):
    icon_sprite = "editor/npc_maker.vmt"
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.MaxNPCCount = 1  # Type: integer
        self.SpawnFrequency = "5"  # Type: string
        self.MaxLiveChildren = 5  # Type: integer
        self.ForceScheduleOnSpawn = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.MaxNPCCount = parse_source_value(entity_data.get('maxnpccount', 1))  # Type: integer
        instance.SpawnFrequency = entity_data.get('spawnfrequency', "5")  # Type: string
        instance.MaxLiveChildren = parse_source_value(entity_data.get('maxlivechildren', 5))  # Type: integer
        instance.ForceScheduleOnSpawn = entity_data.get('forcescheduleonspawn', None)  # Type: string


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
        instance.IgnoreFacing = entity_data.get('ignorefacing', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.MinimumState = entity_data.get('minimumstate', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.MaximumState = entity_data.get('maximumstate', "CHOICES NOT SUPPORTED")  # Type: choices


class TriggerOnce(Global, Origin, EnableDisable, Parentname, Targetname):
    def __init__(self):
        super(Global).__init__()
        super(Origin).__init__()
        super(EnableDisable).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.filtername = None  # Type: filterclass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Global.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.filtername = entity_data.get('filtername', None)  # Type: filterclass


class Trigger(TriggerOnce):
    def __init__(self):
        super(TriggerOnce).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(Parentname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        TriggerOnce.from_dict(instance, entity_data)


class worldbase(Base):
    def __init__(self):
        super().__init__()
        self.message = None  # Type: string
        self.skyname = "sky_day01_01"  # Type: string
        self.chaptertitle = None  # Type: string
        self.startdark = None  # Type: choices
        self.underwaterparticle = "CHOICES NOT SUPPORTED"  # Type: choices
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

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.message = entity_data.get('message', None)  # Type: string
        instance.skyname = entity_data.get('skyname', "sky_day01_01")  # Type: string
        instance.chaptertitle = entity_data.get('chaptertitle', None)  # Type: string
        instance.startdark = entity_data.get('startdark', None)  # Type: choices
        instance.underwaterparticle = entity_data.get('underwaterparticle', "CHOICES NOT SUPPORTED")  # Type: choices
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


class worldspawn(worldbase, Targetname, ResponseContext):
    def __init__(self):
        super(worldbase).__init__()
        super(Targetname).__init__()
        super(ResponseContext).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        worldbase.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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
        self.m_bDontModifyPitchVolOnSpawn = None  # Type: choices

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
        instance.m_bDontModifyPitchVolOnSpawn = entity_data.get('m_bdontmodifypitchvolonspawn', None)  # Type: choices


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
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination


class env_splash(Angles, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.scale = 8.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class env_sun(Angles, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
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
        self.HDRColorScale = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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
        instance.HDRColorScale = float(entity_data.get('hdrcolorscale', 1.0))  # Type: float


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


class env_lightglow(Parentname, Angles, Targetname):
    model_ = "models/editor/axis_helper_thick.mdl"
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


class env_tonemap_controller(Targetname):
    icon_sprite = "editor/env_tonemap_controller.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.bUseNextGenBloom = None  # Type: choices
        self.bUseCusBloomNG_Threshold = None  # Type: choices
        self.fCusBloomNG_Threshold = None  # Type: float
        self.bUseCusBloomNG_tintExponent = None  # Type: choices
        self.m_fCustomBloomNextGen_r = 1.0  # Type: float
        self.m_fCustomBloomNextGen_g = 1.0  # Type: float
        self.m_fCustomBloomNextGen_b = 1.0  # Type: float
        self.m_fCusBloomNG_exponent = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.bUseNextGenBloom = entity_data.get('busenextgenbloom', None)  # Type: choices
        instance.bUseCusBloomNG_Threshold = entity_data.get('busecusbloomng_threshold', None)  # Type: choices
        instance.fCusBloomNG_Threshold = float(entity_data.get('fcusbloomng_threshold', 0))  # Type: float
        instance.bUseCusBloomNG_tintExponent = entity_data.get('busecusbloomng_tintexponent', None)  # Type: choices
        instance.m_fCustomBloomNextGen_r = float(entity_data.get('m_fcustombloomnextgen_r', 1.0))  # Type: float
        instance.m_fCustomBloomNextGen_g = float(entity_data.get('m_fcustombloomnextgen_g', 1.0))  # Type: float
        instance.m_fCustomBloomNextGen_b = float(entity_data.get('m_fcustombloomnextgen_b', 1.0))  # Type: float
        instance.m_fCusBloomNG_exponent = float(entity_data.get('m_fcusbloomng_exponent', 1.0))  # Type: float


class func_useableladder(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.point0 = None  # Type: vector
        self.point1 = None  # Type: vector
        self.StartDisabled = None  # Type: choices
        self.ladderSurfaceProperties = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.point0 = parse_float_vector(entity_data.get('point0', "0 0 0"))  # Type: vector
        instance.point1 = parse_float_vector(entity_data.get('point1', "0 0 0"))  # Type: vector
        instance.StartDisabled = entity_data.get('startdisabled', None)  # Type: choices
        instance.ladderSurfaceProperties = entity_data.get('laddersurfaceproperties', None)  # Type: string


class func_ladderendpoint(Parentname, Angles, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class func_wall(RenderFields, Global, Shadow, Targetname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Global).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        RenderFields.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class func_clip_vphysics(EnableDisable, Targetname):
    def __init__(self):
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        self.filtername = None  # Type: filterclass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.filtername = entity_data.get('filtername', None)  # Type: filterclass


class func_brush(Global, Shadow, RenderFields, Origin, EnableDisable, Inputfilter, Parentname, Targetname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Global).__init__()
        super(Shadow).__init__()
        super(Origin).__init__()
        super(EnableDisable).__init__()
        super(Inputfilter).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
        self._minlight = None  # Type: string
        self.Solidity = None  # Type: choices
        self.excludednpc = None  # Type: string
        self.invert_exclusion = None  # Type: choices
        self.solidbsp = None  # Type: choices
        self.vrad_brush_cast_shadows = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Global.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Inputfilter.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.Solidity = entity_data.get('solidity', None)  # Type: choices
        instance.excludednpc = entity_data.get('excludednpc', None)  # Type: string
        instance.invert_exclusion = entity_data.get('invert_exclusion', None)  # Type: choices
        instance.solidbsp = entity_data.get('solidbsp', None)  # Type: choices
        instance.vrad_brush_cast_shadows = entity_data.get('vrad_brush_cast_shadows', None)  # Type: choices


class vgui_screen_base(Parentname, Angles, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.panelname = None  # Type: string
        self.overlaymaterial = None  # Type: string
        self.width = 32  # Type: integer
        self.height = 32  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class vgui_slideshow_display(Angles, Targetname, Parentname):
    model_ = "models/editor/axis_helper_thick.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
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
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class cycler(RenderFields, Targetname, Parentname, Angles):
    def __init__(self):
        super(RenderFields).__init__()
        super(RenderFxChoices).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.model = None  # Type: studio
        self.skin = None  # Type: integer
        self.sequence = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        RenderFxChoices.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', None)  # Type: studio
        instance.skin = parse_source_value(entity_data.get('skin', 0))  # Type: integer
        instance.sequence = parse_source_value(entity_data.get('sequence', 0))  # Type: integer


class gibshooterbase(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
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
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.angles = entity_data.get('angles', "0 0 0")  # Type: string
        instance.m_iGibs = parse_source_value(entity_data.get('m_igibs', 3))  # Type: integer
        instance.delay = entity_data.get('delay', "0")  # Type: string
        instance.gibangles = entity_data.get('gibangles', "0 0 0")  # Type: string
        instance.gibanglevelocity = entity_data.get('gibanglevelocity', "0")  # Type: string
        instance.m_flVelocity = parse_source_value(entity_data.get('m_flvelocity', 200))  # Type: integer
        instance.m_flVariance = entity_data.get('m_flvariance', "0.15")  # Type: string
        instance.m_flGibLife = entity_data.get('m_flgiblife', "4")  # Type: string
        instance.lightingorigin = entity_data.get('lightingorigin', None)  # Type: target_destination


class env_beam(Parentname, RenderFxChoices, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(RenderFxChoices).__init__()
        super(Targetname).__init__()
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
        self.dissolvetype = "CHOICES NOT SUPPORTED"  # Type: choices
        self.TouchType = None  # Type: choices
        self.filtername = None  # Type: filterclass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        RenderFxChoices.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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
        instance.dissolvetype = entity_data.get('dissolvetype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.TouchType = entity_data.get('touchtype', None)  # Type: choices
        instance.filtername = entity_data.get('filtername', None)  # Type: filterclass


class env_beverage(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.health = 10  # Type: integer
        self.beveragetype = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.health = parse_source_value(entity_data.get('health', 10))  # Type: integer
        instance.beveragetype = entity_data.get('beveragetype', None)  # Type: choices


class env_embers(Parentname, Angles, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.particletype = None  # Type: choices
        self.density = 50  # Type: integer
        self.lifetime = 4  # Type: integer
        self.speed = 32  # Type: integer
        self.rendercolor = [255, 255, 255]  # Type: color255

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.particletype = entity_data.get('particletype', None)  # Type: choices
        instance.density = parse_source_value(entity_data.get('density', 50))  # Type: integer
        instance.lifetime = parse_source_value(entity_data.get('lifetime', 4))  # Type: integer
        instance.speed = parse_source_value(entity_data.get('speed', 32))  # Type: integer
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "255 255 255"))  # Type: color255


class env_funnel(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_blood(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.spraydir = [0.0, 0.0, 0.0]  # Type: angle
        self.color = None  # Type: choices
        self.amount = "100"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.spraydir = parse_float_vector(entity_data.get('spraydir', "0 0 0"))  # Type: angle
        instance.color = entity_data.get('color', None)  # Type: choices
        instance.amount = entity_data.get('amount', "100")  # Type: string


class env_bubbles(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.density = 2  # Type: integer
        self.frequency = 2  # Type: integer
        self.current = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.density = parse_source_value(entity_data.get('density', 2))  # Type: integer
        instance.frequency = parse_source_value(entity_data.get('frequency', 2))  # Type: integer
        instance.current = parse_source_value(entity_data.get('current', 0))  # Type: integer


class env_explosion(Parentname, Targetname):
    icon_sprite = "editor/env_explosion.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.iMagnitude = 100  # Type: integer
        self.iRadiusOverride = None  # Type: integer
        self.fireballsprite = "sprites/zerogxplode.spr"  # Type: sprite
        self.rendermode = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ignoredEntity = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.iMagnitude = parse_source_value(entity_data.get('imagnitude', 100))  # Type: integer
        instance.iRadiusOverride = parse_source_value(entity_data.get('iradiusoverride', 0))  # Type: integer
        instance.fireballsprite = entity_data.get('fireballsprite', "sprites/zerogxplode.spr")  # Type: sprite
        instance.rendermode = entity_data.get('rendermode', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ignoredEntity = entity_data.get('ignoredentity', None)  # Type: target_destination


class env_smoketrail(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
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
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class env_physexplosion(Parentname, Targetname):
    icon_sprite = "editor/env_physexplosion.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.magnitude = "100"  # Type: string
        self.radius = "0"  # Type: string
        self.targetentityname = None  # Type: target_destination
        self.inner_radius = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.magnitude = entity_data.get('magnitude', "100")  # Type: string
        instance.radius = entity_data.get('radius', "0")  # Type: string
        instance.targetentityname = entity_data.get('targetentityname', None)  # Type: target_destination
        instance.inner_radius = float(entity_data.get('inner_radius', 0))  # Type: float


class env_physimpact(Parentname, Targetname):
    icon_sprite = "editor/env_physexplosion.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.angles = "0 0 0"  # Type: string
        self.magnitude = 100  # Type: integer
        self.distance = None  # Type: integer
        self.directionentityname = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.angles = entity_data.get('angles', "0 0 0")  # Type: string
        instance.magnitude = parse_source_value(entity_data.get('magnitude', 100))  # Type: integer
        instance.distance = parse_source_value(entity_data.get('distance', 0))  # Type: integer
        instance.directionentityname = entity_data.get('directionentityname', None)  # Type: target_destination


class env_fire(Parentname, Targetname, EnableDisable):
    icon_sprite = "editor/env_fire"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
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
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.health = parse_source_value(entity_data.get('health', 30))  # Type: integer
        instance.firesize = parse_source_value(entity_data.get('firesize', 64))  # Type: integer
        instance.fireattack = parse_source_value(entity_data.get('fireattack', 4))  # Type: integer
        instance.firetype = entity_data.get('firetype', None)  # Type: choices
        instance.ignitionpoint = float(entity_data.get('ignitionpoint', 32))  # Type: float
        instance.damagescale = float(entity_data.get('damagescale', 1.0))  # Type: float


class env_firesource(Parentname, Targetname):
    icon_sprite = "editor/env_firesource"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.fireradius = 128  # Type: float
        self.firedamage = 10  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.fireradius = float(entity_data.get('fireradius', 128))  # Type: float
        instance.firedamage = float(entity_data.get('firedamage', 10))  # Type: float


class env_firesensor(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.fireradius = 128  # Type: float
        self.heatlevel = 32  # Type: float
        self.heattime = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class env_fog_controller(DXLevelChoice, Angles, Targetname):
    icon_sprite = "editor/fog_controller.vmt"
    def __init__(self):
        super(DXLevelChoice).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
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

    @staticmethod
    def from_dict(instance, entity_data: dict):
        DXLevelChoice.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class env_steam(Parentname, Angles, Targetname):
    viewport_model = "models/editor/spot_cone.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
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
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class env_laser(Parentname, RenderFxChoices, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(RenderFxChoices).__init__()
        super(Targetname).__init__()
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
        self.decalname = "FadingScorch"  # Type: string
        self.dissolvetype = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        RenderFxChoices.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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
        instance.decalname = entity_data.get('decalname', "FadingScorch")  # Type: string
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


class env_shake(Parentname, Targetname):
    icon_sprite = "editor/env_shake.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.amplitude = 4  # Type: float
        self.radius = 500  # Type: float
        self.duration = 1  # Type: float
        self.frequency = 2.5  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.amplitude = float(entity_data.get('amplitude', 4))  # Type: float
        instance.radius = float(entity_data.get('radius', 500))  # Type: float
        instance.duration = float(entity_data.get('duration', 1))  # Type: float
        instance.frequency = float(entity_data.get('frequency', 2.5))  # Type: float


class env_viewpunch(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.punchangle = [0.0, 0.0, 90.0]  # Type: angle
        self.radius = 500  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.punchangle = parse_float_vector(entity_data.get('punchangle', "0 0 90"))  # Type: angle
        instance.radius = float(entity_data.get('radius', 500))  # Type: float


class env_rotorwash_emitter(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.altitude = 1024  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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
        self.nogibshadows = None  # Type: choices
        self.gibgravityscale = 1  # Type: float
        self.massoverride = 0  # Type: float
        self.bloodcolor = "CHOICES NOT SUPPORTED"  # Type: choices
        self.touchkill = None  # Type: choices
        self.gibdamage = 0  # Type: float
        self.gibsound = None  # Type: sound

    @staticmethod
    def from_dict(instance, entity_data: dict):
        gibshooterbase.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.shootmodel = entity_data.get('shootmodel', None)  # Type: studio
        instance.shootsounds = entity_data.get('shootsounds', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.simulation = entity_data.get('simulation', None)  # Type: choices
        instance.skin = parse_source_value(entity_data.get('skin', 0))  # Type: integer
        instance.nogibshadows = entity_data.get('nogibshadows', None)  # Type: choices
        instance.gibgravityscale = float(entity_data.get('gibgravityscale', 1))  # Type: float
        instance.massoverride = float(entity_data.get('massoverride', 0))  # Type: float
        instance.bloodcolor = entity_data.get('bloodcolor', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.touchkill = entity_data.get('touchkill', None)  # Type: choices
        instance.gibdamage = float(entity_data.get('gibdamage', 0))  # Type: float
        instance.gibsound = entity_data.get('gibsound', None)  # Type: sound


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


class env_soundscape_proxy(Parentname, Targetname):
    icon_sprite = "editor/env_soundscape.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.MainSoundscapeName = None  # Type: target_destination
        self.radius = 128  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.MainSoundscapeName = entity_data.get('mainsoundscapename', None)  # Type: target_destination
        instance.radius = parse_source_value(entity_data.get('radius', 128))  # Type: integer


class env_soundscape(Parentname, Targetname, EnableDisable):
    icon_sprite = "editor/env_soundscape.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
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
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class env_spark(Parentname, Angles, Targetname):
    icon_sprite = "editor/env_spark.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.MaxDelay = "0"  # Type: string
        self.Magnitude = "CHOICES NOT SUPPORTED"  # Type: choices
        self.TrailLength = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.MaxDelay = entity_data.get('maxdelay', "0")  # Type: string
        instance.Magnitude = entity_data.get('magnitude', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.TrailLength = entity_data.get('traillength', "CHOICES NOT SUPPORTED")  # Type: choices


class env_sprite(Parentname, Targetname, DXLevelChoice, RenderFields):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(DXLevelChoice).__init__()
        self.origin = [0, 0, 0]
        self.framerate = "10.0"  # Type: string
        self.model = "sprites/glow01.spr"  # Type: sprite
        self.scale = None  # Type: string
        self.GlowProxySize = 2.0  # Type: float
        self.HDRColorScale = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        DXLevelChoice.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.framerate = entity_data.get('framerate', "10.0")  # Type: string
        instance.model = entity_data.get('model', "sprites/glow01.spr")  # Type: sprite
        instance.scale = entity_data.get('scale', None)  # Type: string
        instance.GlowProxySize = float(entity_data.get('glowproxysize', 2.0))  # Type: float
        instance.HDRColorScale = float(entity_data.get('hdrcolorscale', 1.0))  # Type: float


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


class env_wind(Angles, Targetname):
    icon_sprite = "editor/env_wind.vmt"
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
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
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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
        self.fogenable = None  # Type: choices
        self.fogblend = None  # Type: choices
        self.use_angles = None  # Type: choices
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
        instance.fogenable = entity_data.get('fogenable', None)  # Type: choices
        instance.fogblend = entity_data.get('fogblend', None)  # Type: choices
        instance.use_angles = entity_data.get('use_angles', None)  # Type: choices
        instance.fogcolor = parse_int_vector(entity_data.get('fogcolor', "255 255 255"))  # Type: color255
        instance.fogcolor2 = parse_int_vector(entity_data.get('fogcolor2', "255 255 255"))  # Type: color255
        instance.fogdir = entity_data.get('fogdir', "1 0 0")  # Type: string
        instance.fogstart = entity_data.get('fogstart', "500.0")  # Type: string
        instance.fogend = entity_data.get('fogend', "2000.0")  # Type: string


class BaseSpeaker(ResponseContext, Targetname):
    def __init__(self):
        super(ResponseContext).__init__()
        super(Targetname).__init__()
        self.delaymin = "15"  # Type: string
        self.delaymax = "135"  # Type: string
        self.rulescript = "scripts/talker/announcements.txt"  # Type: string
        self.concept = "announcement"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        ResponseContext.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.delaymin = entity_data.get('delaymin', "15")  # Type: string
        instance.delaymax = entity_data.get('delaymax', "135")  # Type: string
        instance.rulescript = entity_data.get('rulescript', "scripts/talker/announcements.txt")  # Type: string
        instance.concept = entity_data.get('concept', "announcement")  # Type: string


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


class point_message(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.message = None  # Type: string
        self.radius = 128  # Type: integer
        self.developeronly = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.message = entity_data.get('message', None)  # Type: string
        instance.radius = parse_source_value(entity_data.get('radius', 128))  # Type: integer
        instance.developeronly = entity_data.get('developeronly', None)  # Type: choices


class point_spotlight(RenderFields, Targetname, Parentname, Angles, DXLevelChoice):
    model_ = "models/editor/cone_helper.mdl"
    def __init__(self):
        super(RenderFields).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(DXLevelChoice).__init__()
        self.origin = [0, 0, 0]
        self.spotlightlength = 500  # Type: integer
        self.spotlightwidth = 50  # Type: integer
        self.HDRColorScale = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        RenderFields.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        DXLevelChoice.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.spotlightlength = parse_source_value(entity_data.get('spotlightlength', 500))  # Type: integer
        instance.spotlightwidth = parse_source_value(entity_data.get('spotlightwidth', 50))  # Type: integer
        instance.HDRColorScale = float(entity_data.get('hdrcolorscale', 1.0))  # Type: float


class point_tesla(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
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
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class game_zone_player(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)


class infodecal(Targetname):
    model_ = "models/editor/axis_helper_thick.mdl"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.texture = None  # Type: decal
        self.LowPriority = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.texture = entity_data.get('texture', None)  # Type: decal
        instance.LowPriority = entity_data.get('lowpriority', None)  # Type: choices


class info_projecteddecal(Angles, Targetname):
    model_ = "models/editor/axis_helper_thick.mdl"
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


class info_player_start(Angles, PlayerClass):
    model_ = "models/editor/playerstart.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(PlayerClass).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        PlayerClass.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_overlay(Targetname):
    model_ = "models/editor/overlay_helper.mdl"
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


class info_target(Parentname, Angles, Targetname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_particle_system(Parentname, Angles, Targetname):
    model_ = "models/editor/cone_helper.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.orientation_follows_viewer = None  # Type: choices
        self.effect_name = None  # Type: particlesystem
        self.start_active = None  # Type: choices
        self.flag_as_weather = None  # Type: choices
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
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.orientation_follows_viewer = entity_data.get('orientation_follows_viewer', None)  # Type: choices
        instance.effect_name = entity_data.get('effect_name', None)  # Type: particlesystem
        instance.start_active = entity_data.get('start_active', None)  # Type: choices
        instance.flag_as_weather = entity_data.get('flag_as_weather', None)  # Type: choices
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


class phys_ragdollmagnet(EnableDisable, Angles, Targetname, Parentname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(EnableDisable).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.axis = None  # Type: vecline
        self.radius = 512  # Type: float
        self.force = 5000  # Type: float
        self.target = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class info_teleport_destination(Parentname, Angles, Targetname, PlayerClass):
    model_ = "models/editor/playerstart.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        super(PlayerClass).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        PlayerClass.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_node(Node):
    model_ = "models/editor/ground_node.mdl"
    def __init__(self):
        super(Node).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Node.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_node_hint(Angles, HintNode, Targetname):
    model_ = "models/editor/ground_node_hint.mdl"
    def __init__(self):
        super(HintNode).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        HintNode.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_node_air(Node):
    model_ = "models/editor/air_node.mdl"
    def __init__(self):
        super(Node).__init__()
        self.origin = [0, 0, 0]
        self.nodeheight = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Node.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.nodeheight = parse_source_value(entity_data.get('nodeheight', 0))  # Type: integer


class info_node_air_hint(Angles, HintNode, Targetname):
    model_ = "models/editor/air_node_hint.mdl"
    def __init__(self):
        super(HintNode).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.nodeheight = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        HintNode.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.nodeheight = parse_source_value(entity_data.get('nodeheight', 0))  # Type: integer


class info_hint(Angles, HintNode, Targetname):
    model_ = "models/editor/node_hint.mdl"
    def __init__(self):
        super(HintNode).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        HintNode.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class info_radial_link_controller(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.radius = 120  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.radius = float(entity_data.get('radius', 120))  # Type: float


class info_node_climb(Angles, HintNode, Targetname):
    model_ = "models/editor/climb_node.mdl"
    def __init__(self):
        super(HintNode).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        HintNode.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_cascade_light(EnableDisable, Targetname):
    icon_sprite = "editor/shadow_control.vmt"
    def __init__(self):
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.angles = "50 40 0"  # Type: string
        self.color = [255, 255, 255, 1]  # Type: color255
        self.maxshadowdistance = 400  # Type: float
        self.uselightenvangles = 1  # Type: integer
        self.LightRadius1 = 0.001  # Type: float
        self.LightRadius2 = 0.001  # Type: float
        self.LightRadius3 = 0.001  # Type: float
        self.Depthbias1 = 0.00025  # Type: float
        self.Depthbias2 = 0.00005  # Type: float
        self.Depthbias3 = 0.00005  # Type: float
        self.Slopescaledepthbias1 = 2.0  # Type: float
        self.Slopescaledepthbias2 = 2.0  # Type: float
        self.Slopescaledepthbias3 = 2.0  # Type: float
        self.ViewModelDepthbias = 0.000009  # Type: float
        self.ViewModelSlopescaledepthbias = 0.9  # Type: float
        self.CSMVolumeMode = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.angles = entity_data.get('angles', "50 40 0")  # Type: string
        instance.color = parse_int_vector(entity_data.get('color', "255 255 255 1"))  # Type: color255
        instance.maxshadowdistance = float(entity_data.get('maxshadowdistance', 400))  # Type: float
        instance.uselightenvangles = parse_source_value(entity_data.get('uselightenvangles', 1))  # Type: integer
        instance.LightRadius1 = float(entity_data.get('lightradius1', 0.001))  # Type: float
        instance.LightRadius2 = float(entity_data.get('lightradius2', 0.001))  # Type: float
        instance.LightRadius3 = float(entity_data.get('lightradius3', 0.001))  # Type: float
        instance.Depthbias1 = float(entity_data.get('depthbias1', 0.00025))  # Type: float
        instance.Depthbias2 = float(entity_data.get('depthbias2', 0.00005))  # Type: float
        instance.Depthbias3 = float(entity_data.get('depthbias3', 0.00005))  # Type: float
        instance.Slopescaledepthbias1 = float(entity_data.get('slopescaledepthbias1', 2.0))  # Type: float
        instance.Slopescaledepthbias2 = float(entity_data.get('slopescaledepthbias2', 2.0))  # Type: float
        instance.Slopescaledepthbias3 = float(entity_data.get('slopescaledepthbias3', 2.0))  # Type: float
        instance.ViewModelDepthbias = float(entity_data.get('viewmodeldepthbias', 0.000009))  # Type: float
        instance.ViewModelSlopescaledepthbias = float(entity_data.get('viewmodelslopescaledepthbias', 0.9))  # Type: float
        instance.CSMVolumeMode = entity_data.get('csmvolumemode', None)  # Type: choices


class newLight_Dir(Parentname, Angles, Targetname):
    icon_sprite = "editor/light_new.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.LightEnvEnabled = "CHOICES NOT SUPPORTED"  # Type: choices
        self.PushbackDist = 9999999  # Type: float
        self.EnableGodRays = "CHOICES NOT SUPPORTED"  # Type: choices
        self.Density = 1.0  # Type: float
        self.Weight = 1.0  # Type: float
        self.Decay = 1.0  # Type: float
        self.Exposure = 2.5  # Type: float
        self.DistFactor = 1.0  # Type: float
        self.DiskRadius = 0.02  # Type: float
        self.DiskInnerSizePercent = 0.75  # Type: float
        self.ColorInner = [128, 200, 255, 255]  # Type: color255
        self.ColorOuter = [255, 255, 164, 255]  # Type: color255
        self.ColorRays = [200, 200, 255, 255]  # Type: color255
        self.m_bUseToneMapRays = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.LightEnvEnabled = entity_data.get('lightenvenabled', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.PushbackDist = float(entity_data.get('pushbackdist', 9999999))  # Type: float
        instance.EnableGodRays = entity_data.get('enablegodrays', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.Density = float(entity_data.get('density', 1.0))  # Type: float
        instance.Weight = float(entity_data.get('weight', 1.0))  # Type: float
        instance.Decay = float(entity_data.get('decay', 1.0))  # Type: float
        instance.Exposure = float(entity_data.get('exposure', 2.5))  # Type: float
        instance.DistFactor = float(entity_data.get('distfactor', 1.0))  # Type: float
        instance.DiskRadius = float(entity_data.get('diskradius', 0.02))  # Type: float
        instance.DiskInnerSizePercent = float(entity_data.get('diskinnersizepercent', 0.75))  # Type: float
        instance.ColorInner = parse_int_vector(entity_data.get('colorinner', "128 200 255 255"))  # Type: color255
        instance.ColorOuter = parse_int_vector(entity_data.get('colorouter', "255 255 164 255"))  # Type: color255
        instance.ColorRays = parse_int_vector(entity_data.get('colorrays', "200 200 255 255"))  # Type: color255
        instance.m_bUseToneMapRays = entity_data.get('m_busetonemaprays', "CHOICES NOT SUPPORTED")  # Type: choices


class newLight_Point(Parentname, Targetname):
    icon_sprite = "editor/light_new.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.Enabled = "CHOICES NOT SUPPORTED"  # Type: choices
        self.style = None  # Type: choices
        self.LightColorAmbient = [0, 0, 0, 0]  # Type: color255
        self.LightColor = [255, 255, 255, 1]  # Type: color255
        self.Intensity = 8000  # Type: float
        self.SpecMultiplier = 1  # Type: float
        self.Range = 1000  # Type: float
        self.LightType = None  # Type: choices
        self.HasShadow = None  # Type: choices
        self.ShadowLod = None  # Type: choices
        self.ShadowFaceX = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ShadowFaceX_Minus = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ShadowFaceY = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ShadowFaceY_Minus = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ShadowFaceZ = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ShadowFaceZ_Minus = "CHOICES NOT SUPPORTED"  # Type: choices
        self.NearZ = 2  # Type: float
        self.DepthBias = 0.0002  # Type: float
        self.SlopeDepthBias = 0.2  # Type: float
        self.NormalBias = 1.0  # Type: float
        self.ShadowRadius = -1.0  # Type: float
        self.bTexLight = None  # Type: choices
        self.texName = None  # Type: string
        self.bNegLight = None  # Type: choices
        self.LightnGodRayMode = None  # Type: choices
        self.EnableGodRays = None  # Type: choices
        self.Density = 1.0  # Type: float
        self.Weight = 1.0  # Type: float
        self.Decay = 1.0  # Type: float
        self.Exposure = 2.5  # Type: float
        self.DistFactor = 1.0  # Type: float
        self.DiskRadius = 0.02  # Type: float
        self.ColorInner = [128, 200, 255, 255]  # Type: color255
        self.ColorRays = [200, 200, 255, 255]  # Type: color255
        self.GodRaysType = None  # Type: choices
        self.DiskInnerSizePercent = 0.75  # Type: float
        self.ColorOuter = [255, 255, 164, 1]  # Type: color255
        self.Ell_FR_ConstA = 0.9  # Type: float
        self.Ell_FR_ConstB = 0.1  # Type: float
        self.Ell_SR_ConstA = 0.9  # Type: float
        self.Ell_SR_ConstB = 0.1  # Type: float
        self.Ell_RRF_ConstA = 0.9  # Type: float
        self.Ell_RRF_ConstB = 0.1  # Type: float
        self.RotSpeed = 3.14  # Type: float
        self.RotPatternFreq = 10.0  # Type: float
        self.m_bEnableWorldSpace = None  # Type: choices
        self.m_fAlphaDiskInner = 1  # Type: float
        self.m_fAlphaDiskOuter = 1  # Type: float
        self.m_bUseToneMapRays = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_bUseToneMapDisk = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_bSRO_Brush = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_bSRO_StaticProp = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_bSRO_DynProps = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_bSRO_Trans = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Enabled = entity_data.get('enabled', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.style = entity_data.get('style', None)  # Type: choices
        instance.LightColorAmbient = parse_int_vector(entity_data.get('lightcolorambient', "0 0 0 0"))  # Type: color255
        instance.LightColor = parse_int_vector(entity_data.get('lightcolor', "255 255 255 1"))  # Type: color255
        instance.Intensity = float(entity_data.get('intensity', 8000))  # Type: float
        instance.SpecMultiplier = float(entity_data.get('specmultiplier', 1))  # Type: float
        instance.Range = float(entity_data.get('range', 1000))  # Type: float
        instance.LightType = entity_data.get('lighttype', None)  # Type: choices
        instance.HasShadow = entity_data.get('hasshadow', None)  # Type: choices
        instance.ShadowLod = entity_data.get('shadowlod', None)  # Type: choices
        instance.ShadowFaceX = entity_data.get('shadowfacex', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ShadowFaceX_Minus = entity_data.get('shadowfacex_minus', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ShadowFaceY = entity_data.get('shadowfacey', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ShadowFaceY_Minus = entity_data.get('shadowfacey_minus', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ShadowFaceZ = entity_data.get('shadowfacez', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ShadowFaceZ_Minus = entity_data.get('shadowfacez_minus', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.NearZ = float(entity_data.get('nearz', 2))  # Type: float
        instance.DepthBias = float(entity_data.get('depthbias', 0.0002))  # Type: float
        instance.SlopeDepthBias = float(entity_data.get('slopedepthbias', 0.2))  # Type: float
        instance.NormalBias = float(entity_data.get('normalbias', 1.0))  # Type: float
        instance.ShadowRadius = float(entity_data.get('shadowradius', -1.0))  # Type: float
        instance.bTexLight = entity_data.get('btexlight', None)  # Type: choices
        instance.texName = entity_data.get('texname', None)  # Type: string
        instance.bNegLight = entity_data.get('bneglight', None)  # Type: choices
        instance.LightnGodRayMode = entity_data.get('lightngodraymode', None)  # Type: choices
        instance.EnableGodRays = entity_data.get('enablegodrays', None)  # Type: choices
        instance.Density = float(entity_data.get('density', 1.0))  # Type: float
        instance.Weight = float(entity_data.get('weight', 1.0))  # Type: float
        instance.Decay = float(entity_data.get('decay', 1.0))  # Type: float
        instance.Exposure = float(entity_data.get('exposure', 2.5))  # Type: float
        instance.DistFactor = float(entity_data.get('distfactor', 1.0))  # Type: float
        instance.DiskRadius = float(entity_data.get('diskradius', 0.02))  # Type: float
        instance.ColorInner = parse_int_vector(entity_data.get('colorinner', "128 200 255 255"))  # Type: color255
        instance.ColorRays = parse_int_vector(entity_data.get('colorrays', "200 200 255 255"))  # Type: color255
        instance.GodRaysType = entity_data.get('godraystype', None)  # Type: choices
        instance.DiskInnerSizePercent = float(entity_data.get('diskinnersizepercent', 0.75))  # Type: float
        instance.ColorOuter = parse_int_vector(entity_data.get('colorouter', "255 255 164 1"))  # Type: color255
        instance.Ell_FR_ConstA = float(entity_data.get('ell_fr_consta', 0.9))  # Type: float
        instance.Ell_FR_ConstB = float(entity_data.get('ell_fr_constb', 0.1))  # Type: float
        instance.Ell_SR_ConstA = float(entity_data.get('ell_sr_consta', 0.9))  # Type: float
        instance.Ell_SR_ConstB = float(entity_data.get('ell_sr_constb', 0.1))  # Type: float
        instance.Ell_RRF_ConstA = float(entity_data.get('ell_rrf_consta', 0.9))  # Type: float
        instance.Ell_RRF_ConstB = float(entity_data.get('ell_rrf_constb', 0.1))  # Type: float
        instance.RotSpeed = float(entity_data.get('rotspeed', 3.14))  # Type: float
        instance.RotPatternFreq = float(entity_data.get('rotpatternfreq', 10.0))  # Type: float
        instance.m_bEnableWorldSpace = entity_data.get('m_benableworldspace', None)  # Type: choices
        instance.m_fAlphaDiskInner = float(entity_data.get('m_falphadiskinner', 1))  # Type: float
        instance.m_fAlphaDiskOuter = float(entity_data.get('m_falphadiskouter', 1))  # Type: float
        instance.m_bUseToneMapRays = entity_data.get('m_busetonemaprays', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.m_bUseToneMapDisk = entity_data.get('m_busetonemapdisk', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.m_bSRO_Brush = entity_data.get('m_bsro_brush', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.m_bSRO_StaticProp = entity_data.get('m_bsro_staticprop', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.m_bSRO_DynProps = entity_data.get('m_bsro_dynprops', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.m_bSRO_Trans = entity_data.get('m_bsro_trans', "CHOICES NOT SUPPORTED")  # Type: choices


class newLight_Spot(Parentname, Angles, Targetname):
    icon_sprite = "editor/light_new.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.ReverseDir = None  # Type: choices
        self.Enabled = "CHOICES NOT SUPPORTED"  # Type: choices
        self.style = None  # Type: choices
        self.LightColorAmbient = [0, 0, 0, 0]  # Type: color255
        self.LightColor = [255, 255, 255, 1]  # Type: color255
        self.Intensity = 8000  # Type: float
        self.SpecMultiplier = 1  # Type: float
        self.Range = 1000  # Type: float
        self.phi = 60  # Type: float
        self.theta = 30  # Type: float
        self.angularFallOff = 2  # Type: float
        self.LightType = None  # Type: choices
        self.HasShadow = None  # Type: choices
        self.ShadowLod = None  # Type: choices
        self.NearZ = 2  # Type: float
        self.DepthBias = 0.0002  # Type: float
        self.SlopeDepthBias = 0.2  # Type: float
        self.NormalBias = 1.0  # Type: float
        self.ShadowFOV = 0  # Type: float
        self.ShadowRadius = -1.0  # Type: float
        self.bTexLight = None  # Type: choices
        self.texName = None  # Type: string
        self.TexCookieFramesX = 1  # Type: integer
        self.TexCookieFramesY = 1  # Type: integer
        self.TexCookieFps = None  # Type: float
        self.bTexCookieScrollMode = None  # Type: choices
        self.fScrollSpeedU = None  # Type: float
        self.fScrollSpeedV = None  # Type: float
        self.bNegLight = None  # Type: choices
        self.m_bSRO_Brush = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_bSRO_StaticProp = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_bSRO_DynProps = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_bSRO_Trans = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.ReverseDir = entity_data.get('reversedir', None)  # Type: choices
        instance.Enabled = entity_data.get('enabled', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.style = entity_data.get('style', None)  # Type: choices
        instance.LightColorAmbient = parse_int_vector(entity_data.get('lightcolorambient', "0 0 0 0"))  # Type: color255
        instance.LightColor = parse_int_vector(entity_data.get('lightcolor', "255 255 255 1"))  # Type: color255
        instance.Intensity = float(entity_data.get('intensity', 8000))  # Type: float
        instance.SpecMultiplier = float(entity_data.get('specmultiplier', 1))  # Type: float
        instance.Range = float(entity_data.get('range', 1000))  # Type: float
        instance.phi = float(entity_data.get('phi', 60))  # Type: float
        instance.theta = float(entity_data.get('theta', 30))  # Type: float
        instance.angularFallOff = float(entity_data.get('angularfalloff', 2))  # Type: float
        instance.LightType = entity_data.get('lighttype', None)  # Type: choices
        instance.HasShadow = entity_data.get('hasshadow', None)  # Type: choices
        instance.ShadowLod = entity_data.get('shadowlod', None)  # Type: choices
        instance.NearZ = float(entity_data.get('nearz', 2))  # Type: float
        instance.DepthBias = float(entity_data.get('depthbias', 0.0002))  # Type: float
        instance.SlopeDepthBias = float(entity_data.get('slopedepthbias', 0.2))  # Type: float
        instance.NormalBias = float(entity_data.get('normalbias', 1.0))  # Type: float
        instance.ShadowFOV = float(entity_data.get('shadowfov', 0))  # Type: float
        instance.ShadowRadius = float(entity_data.get('shadowradius', -1.0))  # Type: float
        instance.bTexLight = entity_data.get('btexlight', None)  # Type: choices
        instance.texName = entity_data.get('texname', None)  # Type: string
        instance.TexCookieFramesX = parse_source_value(entity_data.get('texcookieframesx', 1))  # Type: integer
        instance.TexCookieFramesY = parse_source_value(entity_data.get('texcookieframesy', 1))  # Type: integer
        instance.TexCookieFps = float(entity_data.get('texcookiefps', 0))  # Type: float
        instance.bTexCookieScrollMode = entity_data.get('btexcookiescrollmode', None)  # Type: choices
        instance.fScrollSpeedU = float(entity_data.get('fscrollspeedu', 0))  # Type: float
        instance.fScrollSpeedV = float(entity_data.get('fscrollspeedv', 0))  # Type: float
        instance.bNegLight = entity_data.get('bneglight', None)  # Type: choices
        instance.m_bSRO_Brush = entity_data.get('m_bsro_brush', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.m_bSRO_StaticProp = entity_data.get('m_bsro_staticprop', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.m_bSRO_DynProps = entity_data.get('m_bsro_dynprops', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.m_bSRO_Trans = entity_data.get('m_bsro_trans', "CHOICES NOT SUPPORTED")  # Type: choices


class godrays_settings(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.TargetGodRays = None  # Type: string
        self.TransitionTime = None  # Type: integer
        self.LightType = "CHOICES NOT SUPPORTED"  # Type: choices
        self.EnableGodRays = 1  # Type: integer
        self.Density = 1.0  # Type: float
        self.Weight = 1.0  # Type: float
        self.Decay = 1.0  # Type: float
        self.Exposure = 2.5  # Type: float
        self.DistFactor = 1.0  # Type: float
        self.DiskRadius = 0.02  # Type: float
        self.DiskInnerSizePercent = 0.75  # Type: float
        self.ColorInner = [128, 200, 255, 1]  # Type: color255
        self.ColorOuter = [255, 255, 164, 1]  # Type: color255
        self.ColorRays = [200, 200, 255, 1]  # Type: color255

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.TargetGodRays = entity_data.get('targetgodrays', None)  # Type: string
        instance.TransitionTime = parse_source_value(entity_data.get('transitiontime', 0))  # Type: integer
        instance.LightType = entity_data.get('lighttype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.EnableGodRays = parse_source_value(entity_data.get('enablegodrays', 1))  # Type: integer
        instance.Density = float(entity_data.get('density', 1.0))  # Type: float
        instance.Weight = float(entity_data.get('weight', 1.0))  # Type: float
        instance.Decay = float(entity_data.get('decay', 1.0))  # Type: float
        instance.Exposure = float(entity_data.get('exposure', 2.5))  # Type: float
        instance.DistFactor = float(entity_data.get('distfactor', 1.0))  # Type: float
        instance.DiskRadius = float(entity_data.get('diskradius', 0.02))  # Type: float
        instance.DiskInnerSizePercent = float(entity_data.get('diskinnersizepercent', 0.75))  # Type: float
        instance.ColorInner = parse_int_vector(entity_data.get('colorinner', "128 200 255 1"))  # Type: color255
        instance.ColorOuter = parse_int_vector(entity_data.get('colorouter', "255 255 164 1"))  # Type: color255
        instance.ColorRays = parse_int_vector(entity_data.get('colorrays', "200 200 255 1"))  # Type: color255


class newLights_settings(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.TargetNewLightEntity = None  # Type: string
        self.TransitionTime = None  # Type: integer
        self.LightType = None  # Type: choices
        self.Enabled = 1  # Type: integer
        self.LightColorAmbient = [0, 0, 0, 0]  # Type: color255
        self.LightColor = [255, 255, 255, 1]  # Type: color255
        self.style = None  # Type: choices
        self.Intensity = 8000  # Type: float
        self.SpecMultiplier = 1  # Type: float
        self.Range = 1000  # Type: float
        self.falloffQuadratic = None  # Type: float
        self.falloffLinear = None  # Type: float
        self.falloffConstant = 1  # Type: float
        self.phi = 60  # Type: float
        self.theta = 30  # Type: float
        self.angularFallOff = 2  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.TargetNewLightEntity = entity_data.get('targetnewlightentity', None)  # Type: string
        instance.TransitionTime = parse_source_value(entity_data.get('transitiontime', 0))  # Type: integer
        instance.LightType = entity_data.get('lighttype', None)  # Type: choices
        instance.Enabled = parse_source_value(entity_data.get('enabled', 1))  # Type: integer
        instance.LightColorAmbient = parse_int_vector(entity_data.get('lightcolorambient', "0 0 0 0"))  # Type: color255
        instance.LightColor = parse_int_vector(entity_data.get('lightcolor', "255 255 255 1"))  # Type: color255
        instance.style = entity_data.get('style', None)  # Type: choices
        instance.Intensity = float(entity_data.get('intensity', 8000))  # Type: float
        instance.SpecMultiplier = float(entity_data.get('specmultiplier', 1))  # Type: float
        instance.Range = float(entity_data.get('range', 1000))  # Type: float
        instance.falloffQuadratic = float(entity_data.get('falloffquadratic', 0))  # Type: float
        instance.falloffLinear = float(entity_data.get('fallofflinear', 0))  # Type: float
        instance.falloffConstant = float(entity_data.get('falloffconstant', 1))  # Type: float
        instance.phi = float(entity_data.get('phi', 60))  # Type: float
        instance.theta = float(entity_data.get('theta', 30))  # Type: float
        instance.angularFallOff = float(entity_data.get('angularfalloff', 2))  # Type: float


class newlights_gbuffersettings(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.Enable4WaysFastPath = "CHOICES NOT SUPPORTED"  # Type: choices
        self.DisableGbufferOnSecondaryCams = None  # Type: choices
        self.DisableGbufferOnRefractions = None  # Type: choices
        self.DisableGbufferOnReflections = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Enable4WaysFastPath = entity_data.get('enable4waysfastpath', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.DisableGbufferOnSecondaryCams = entity_data.get('disablegbufferonsecondarycams', None)  # Type: choices
        instance.DisableGbufferOnRefractions = entity_data.get('disablegbufferonrefractions', None)  # Type: choices
        instance.DisableGbufferOnReflections = entity_data.get('disablegbufferonreflections', None)  # Type: choices


class newLights_Spawner(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.LightType = None  # Type: choices
        self.NumLights = 100  # Type: integer
        self.NumLightsInRow = 10  # Type: integer
        self.LightRange = 250  # Type: float
        self.LightIntensity = 4000  # Type: float
        self.RowSpaceing = 100  # Type: float
        self.ColSpacing = 100  # Type: float
        self.RandomColor = "CHOICES NOT SUPPORTED"  # Type: choices
        self.LightColor = [255, 255, 255, 1]  # Type: color255
        self.SpawnDir_Forward = [1.0, 0.0, 0.0]  # Type: vector
        self.SpawnDir_Right = [0.0, 1.0, 0.0]  # Type: vector

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.LightType = entity_data.get('lighttype', None)  # Type: choices
        instance.NumLights = parse_source_value(entity_data.get('numlights', 100))  # Type: integer
        instance.NumLightsInRow = parse_source_value(entity_data.get('numlightsinrow', 10))  # Type: integer
        instance.LightRange = float(entity_data.get('lightrange', 250))  # Type: float
        instance.LightIntensity = float(entity_data.get('lightintensity', 4000))  # Type: float
        instance.RowSpaceing = float(entity_data.get('rowspaceing', 100))  # Type: float
        instance.ColSpacing = float(entity_data.get('colspacing', 100))  # Type: float
        instance.RandomColor = entity_data.get('randomcolor', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.LightColor = parse_int_vector(entity_data.get('lightcolor', "255 255 255 1"))  # Type: color255
        instance.SpawnDir_Forward = parse_float_vector(entity_data.get('spawndir_forward', "1.0 0.0 0.0"))  # Type: vector
        instance.SpawnDir_Right = parse_float_vector(entity_data.get('spawndir_right', "0.0 1.0 0.0"))  # Type: vector


class newxog_global(Parentname, Targetname):
    icon_sprite = "editor/xog_global.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.Enabled = "CHOICES NOT SUPPORTED"  # Type: choices
        self.XogType = None  # Type: choices
        self.skyColor = [0, 0, 255, 255]  # Type: color255
        self.skyBlendFactor = 0.4  # Type: float
        self.colorMode = None  # Type: choices
        self.texName = None  # Type: string
        self.colorTop = [255, 0, 0, 255]  # Type: color255
        self.colorBottom = [0, 255, 0, 255]  # Type: color255
        self.distStart = 50  # Type: float
        self.distEnd = 2000  # Type: float
        self.distDensity = 1.0  # Type: float
        self.opacityOffsetTop = None  # Type: float
        self.opacityOffsetBottom = None  # Type: float
        self.htZStart = None  # Type: float
        self.htZEnd = 400  # Type: float
        self.htZColStart = None  # Type: float
        self.htZColEnd = 400  # Type: float
        self.noise1ScrollSpeed = None  # Failed to parse value type due to could not convert string to float: '0.006,'  # Type: vector
        self.noise1Tiling = [0.34, 0.34, 0.34, 0.0]  # Type: vector
        self.noise2ScrollSpeed = None  # Failed to parse value type due to could not convert string to float: '0.0035,'  # Type: vector
        self.noise2Tiling = [0.24, 0.24, 0.24, 0.0]  # Type: vector
        self.noiseContrast = 1.0  # Type: float
        self.noiseMultiplier = 1.0  # Type: float
        self.RadiusX = None  # Type: float
        self.RadiusY = None  # Type: float
        self.RadiusZ = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Enabled = entity_data.get('enabled', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.XogType = entity_data.get('xogtype', None)  # Type: choices
        instance.skyColor = parse_int_vector(entity_data.get('skycolor', "0 0 255 255"))  # Type: color255
        instance.skyBlendFactor = float(entity_data.get('skyblendfactor', 0.4))  # Type: float
        instance.colorMode = entity_data.get('colormode', None)  # Type: choices
        instance.texName = entity_data.get('texname', None)  # Type: string
        instance.colorTop = parse_int_vector(entity_data.get('colortop', "255 0 0 255"))  # Type: color255
        instance.colorBottom = parse_int_vector(entity_data.get('colorbottom', "0 255 0 255"))  # Type: color255
        instance.distStart = float(entity_data.get('diststart', 50))  # Type: float
        instance.distEnd = float(entity_data.get('distend', 2000))  # Type: float
        instance.distDensity = float(entity_data.get('distdensity', 1.0))  # Type: float
        instance.opacityOffsetTop = float(entity_data.get('opacityoffsettop', 0))  # Type: float
        instance.opacityOffsetBottom = float(entity_data.get('opacityoffsetbottom', 0))  # Type: float
        instance.htZStart = float(entity_data.get('htzstart', 0))  # Type: float
        instance.htZEnd = float(entity_data.get('htzend', 400))  # Type: float
        instance.htZColStart = float(entity_data.get('htzcolstart', 0))  # Type: float
        instance.htZColEnd = float(entity_data.get('htzcolend', 400))  # Type: float
        instance.noise1ScrollSpeed = parse_float_vector(entity_data.get('noise1scrollspeed', "0.007 0.006, -0.01 0.0"))  # Type: vector
        instance.noise1Tiling = parse_float_vector(entity_data.get('noise1tiling', "0.34 0.34 0.34 0.0"))  # Type: vector
        instance.noise2ScrollSpeed = parse_float_vector(entity_data.get('noise2scrollspeed', "0.0035, 0.003, -0.005 0.0"))  # Type: vector
        instance.noise2Tiling = parse_float_vector(entity_data.get('noise2tiling', "0.24 0.24 0.24 0.0"))  # Type: vector
        instance.noiseContrast = float(entity_data.get('noisecontrast', 1.0))  # Type: float
        instance.noiseMultiplier = float(entity_data.get('noisemultiplier', 1.0))  # Type: float
        instance.RadiusX = float(entity_data.get('radiusx', 0))  # Type: float
        instance.RadiusY = float(entity_data.get('radiusy', 0))  # Type: float
        instance.RadiusZ = float(entity_data.get('radiusz', 0))  # Type: float


class newxog_settings(Parentname, Targetname):
    icon_sprite = "editor/xog_settings.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.TargetNewLightEntity = None  # Type: string
        self.TransitionTime = None  # Type: integer
        self.Enabled = 1  # Type: integer
        self.skyColorTop = [112, 104, 255, 255]  # Type: color255
        self.skyBlendType = None  # Type: integer
        self.skyBlendFactor = 0.25  # Type: float
        self.colorTop = [61, 255, 235, 255]  # Type: color255
        self.colorBottom = [255, 62, 235, 255]  # Type: color255
        self.distStart = 50  # Type: float
        self.distEnd = 2000  # Type: float
        self.distDensity = 1.0  # Type: float
        self.opacityOffsetTop = None  # Type: float
        self.opacityOffsetBottom = None  # Type: float
        self.htZStart = None  # Type: float
        self.htZEnd = 2000  # Type: float
        self.htZColStart = None  # Type: float
        self.htZColEnd = 400  # Type: float
        self.noise1ScrollSpeed = [0.007, 0.006, -0.01, 0.0]  # Type: vector
        self.noise1Tiling = [0.34, 0.34, 0.34, 0.0]  # Type: vector
        self.noise2ScrollSpeed = [0.0035, 0.003, -0.005, 0.0]  # Type: vector
        self.noise2Tiling = [0.24, 0.24, 0.24, 0.0]  # Type: vector
        self.noiseContrast = 1.0  # Type: float
        self.noiseMultiplier = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.TargetNewLightEntity = entity_data.get('targetnewlightentity', None)  # Type: string
        instance.TransitionTime = parse_source_value(entity_data.get('transitiontime', 0))  # Type: integer
        instance.Enabled = parse_source_value(entity_data.get('enabled', 1))  # Type: integer
        instance.skyColorTop = parse_int_vector(entity_data.get('skycolortop', "112 104 255 255"))  # Type: color255
        instance.skyBlendType = parse_source_value(entity_data.get('skyblendtype', 0))  # Type: integer
        instance.skyBlendFactor = float(entity_data.get('skyblendfactor', 0.25))  # Type: float
        instance.colorTop = parse_int_vector(entity_data.get('colortop', "61 255 235 255"))  # Type: color255
        instance.colorBottom = parse_int_vector(entity_data.get('colorbottom', "255 62 235 255"))  # Type: color255
        instance.distStart = float(entity_data.get('diststart', 50))  # Type: float
        instance.distEnd = float(entity_data.get('distend', 2000))  # Type: float
        instance.distDensity = float(entity_data.get('distdensity', 1.0))  # Type: float
        instance.opacityOffsetTop = float(entity_data.get('opacityoffsettop', 0))  # Type: float
        instance.opacityOffsetBottom = float(entity_data.get('opacityoffsetbottom', 0))  # Type: float
        instance.htZStart = float(entity_data.get('htzstart', 0))  # Type: float
        instance.htZEnd = float(entity_data.get('htzend', 2000))  # Type: float
        instance.htZColStart = float(entity_data.get('htzcolstart', 0))  # Type: float
        instance.htZColEnd = float(entity_data.get('htzcolend', 400))  # Type: float
        instance.noise1ScrollSpeed = parse_float_vector(entity_data.get('noise1scrollspeed', "0.007 0.006 -0.01 0.0"))  # Type: vector
        instance.noise1Tiling = parse_float_vector(entity_data.get('noise1tiling', "0.34 0.34 0.34 0.0"))  # Type: vector
        instance.noise2ScrollSpeed = parse_float_vector(entity_data.get('noise2scrollspeed', "0.0035 0.003 -0.005 0.0"))  # Type: vector
        instance.noise2Tiling = parse_float_vector(entity_data.get('noise2tiling', "0.24 0.24 0.24 0.0"))  # Type: vector
        instance.noiseContrast = float(entity_data.get('noisecontrast', 1.0))  # Type: float
        instance.noiseMultiplier = float(entity_data.get('noisemultiplier', 1.0))  # Type: float


class newxog_volume(TriggerOnce):
    def __init__(self):
        super(TriggerOnce).__init__()
        self.Enabled = "CHOICES NOT SUPPORTED"  # Type: choices
        self.XogType = None  # Type: choices
        self.colorMode = None  # Type: choices
        self.texName = None  # Type: string
        self.colorTop = [255, 0, 0, 255]  # Type: color255
        self.colorBottom = [0, 255, 0, 255]  # Type: color255
        self.distStart = 50  # Type: float
        self.distEnd = 2000  # Type: float
        self.distDensity = 1.0  # Type: float
        self.opacityOffsetTop = None  # Type: float
        self.opacityOffsetBottom = None  # Type: float
        self.htZStart = None  # Type: float
        self.htZEnd = 400  # Type: float
        self.htZColStart = None  # Type: float
        self.htZColEnd = 400  # Type: float
        self.noise1ScrollSpeed = [0.01095, 0.00855, -0.02265, 0.0]  # Type: vector
        self.noise1Tiling = [1.32, 1.32, 1.32, 0.0]  # Type: vector
        self.noise2ScrollSpeed = [0.00525, 0.00495, -0.0075, 0.0]  # Type: vector
        self.noise2Tiling = [0.96, 0.96, 0.96, 0.0]  # Type: vector
        self.noiseContrast = 1.0  # Type: float
        self.noiseMultiplier = 1.0  # Type: float
        self.EnableVol_Height = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TriggerOnce.from_dict(instance, entity_data)
        instance.Enabled = entity_data.get('enabled', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.XogType = entity_data.get('xogtype', None)  # Type: choices
        instance.colorMode = entity_data.get('colormode', None)  # Type: choices
        instance.texName = entity_data.get('texname', None)  # Type: string
        instance.colorTop = parse_int_vector(entity_data.get('colortop', "255 0 0 255"))  # Type: color255
        instance.colorBottom = parse_int_vector(entity_data.get('colorbottom', "0 255 0 255"))  # Type: color255
        instance.distStart = float(entity_data.get('diststart', 50))  # Type: float
        instance.distEnd = float(entity_data.get('distend', 2000))  # Type: float
        instance.distDensity = float(entity_data.get('distdensity', 1.0))  # Type: float
        instance.opacityOffsetTop = float(entity_data.get('opacityoffsettop', 0))  # Type: float
        instance.opacityOffsetBottom = float(entity_data.get('opacityoffsetbottom', 0))  # Type: float
        instance.htZStart = float(entity_data.get('htzstart', 0))  # Type: float
        instance.htZEnd = float(entity_data.get('htzend', 400))  # Type: float
        instance.htZColStart = float(entity_data.get('htzcolstart', 0))  # Type: float
        instance.htZColEnd = float(entity_data.get('htzcolend', 400))  # Type: float
        instance.noise1ScrollSpeed = parse_float_vector(entity_data.get('noise1scrollspeed', "0.01095 0.00855 -0.02265 0.0"))  # Type: vector
        instance.noise1Tiling = parse_float_vector(entity_data.get('noise1tiling', "1.32 1.32 1.32 0.0"))  # Type: vector
        instance.noise2ScrollSpeed = parse_float_vector(entity_data.get('noise2scrollspeed', "0.00525 0.00495 -0.0075 0.0"))  # Type: vector
        instance.noise2Tiling = parse_float_vector(entity_data.get('noise2tiling', "0.96 0.96 0.96 0.0"))  # Type: vector
        instance.noiseContrast = float(entity_data.get('noisecontrast', 1.0))  # Type: float
        instance.noiseMultiplier = float(entity_data.get('noisemultiplier', 1.0))  # Type: float
        instance.EnableVol_Height = entity_data.get('enablevol_height', None)  # Type: choices


class light(Light, Targetname):
    icon_sprite = "editor/light.vmt"
    def __init__(self):
        super(Light).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self._distance = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Light.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class light_spot(Angles, Light, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Light).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self._inner_cone = 30  # Type: integer
        self._cone = 45  # Type: integer
        self._exponent = 1  # Type: integer
        self._distance = None  # Type: integer
        self.pitch = -90  # Type: angle_negative_pitch

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Light.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance._inner_cone = parse_source_value(entity_data.get('_inner_cone', 30))  # Type: integer
        instance._cone = parse_source_value(entity_data.get('_cone', 45))  # Type: integer
        instance._exponent = parse_source_value(entity_data.get('_exponent', 1))  # Type: integer
        instance._distance = parse_source_value(entity_data.get('_distance', 0))  # Type: integer
        instance.pitch = float(entity_data.get('pitch', -90))  # Type: angle_negative_pitch


class light_dynamic(Parentname, Angles, Targetname):
    icon_sprite = "editor/light.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
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
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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
        self.disableallshadows = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ForceBlobShadows = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.angles = entity_data.get('angles', "80 30 0")  # Type: string
        instance.color = parse_int_vector(entity_data.get('color', "128 128 128"))  # Type: color255
        instance.distance = float(entity_data.get('distance', 75))  # Type: float
        instance.disableallshadows = entity_data.get('disableallshadows', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ForceBlobShadows = entity_data.get('forceblobshadows', "CHOICES NOT SUPPORTED")  # Type: choices


class color_correction(EnableDisable, Targetname):
    icon_sprite = "editor/color_correction.vmt"
    def __init__(self):
        super(EnableDisable).__init__()
        super(Targetname).__init__()
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
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.minfalloff = float(entity_data.get('minfalloff', 0.0))  # Type: float
        instance.maxfalloff = float(entity_data.get('maxfalloff', 200.0))  # Type: float
        instance.maxweight = float(entity_data.get('maxweight', 1.0))  # Type: float
        instance.filename = entity_data.get('filename', None)  # Type: string
        instance.fadeInDuration = float(entity_data.get('fadeinduration', 0.0))  # Type: float
        instance.fadeOutDuration = float(entity_data.get('fadeoutduration', 0.0))  # Type: float
        instance.exclusive = entity_data.get('exclusive', None)  # Type: choices


class fog_volume(EnableDisable, Targetname):
    def __init__(self):
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        self.FogName = None  # Type: target_destination
        self.ColorCorrectionName = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.FogName = entity_data.get('fogname', None)  # Type: target_destination
        instance.ColorCorrectionName = entity_data.get('colorcorrectionname', None)  # Type: target_destination


class color_correction_volume(EnableDisable, Targetname):
    def __init__(self):
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        self.fadeDuration = 10.0  # Type: float
        self.maxweight = 1.0  # Type: float
        self.filename = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class func_movelinear(Parentname, Targetname, Origin, RenderFields):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
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
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance.movedir = parse_float_vector(entity_data.get('movedir', "0 0 0"))  # Type: angle
        instance.startposition = float(entity_data.get('startposition', 0))  # Type: float
        instance.speed = parse_source_value(entity_data.get('speed', 100))  # Type: integer
        instance.movedistance = float(entity_data.get('movedistance', 100))  # Type: float
        instance.blockdamage = float(entity_data.get('blockdamage', 0))  # Type: float
        instance.startsound = entity_data.get('startsound', None)  # Type: sound
        instance.stopsound = entity_data.get('stopsound', None)  # Type: sound


class func_water_analog(Parentname, Targetname, Origin):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Origin).__init__()
        self.movedir = [0.0, 0.0, 0.0]  # Type: angle
        self.startposition = None  # Type: float
        self.speed = 100  # Type: integer
        self.movedistance = 100  # Type: float
        self.startsound = None  # Type: sound
        self.stopsound = None  # Type: sound
        self.WaterMaterial = "liquids/c4a1_water_green"  # Type: material
        self.WaveHeight = "3.0"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.movedir = parse_float_vector(entity_data.get('movedir', "0 0 0"))  # Type: angle
        instance.startposition = float(entity_data.get('startposition', 0))  # Type: float
        instance.speed = parse_source_value(entity_data.get('speed', 100))  # Type: integer
        instance.movedistance = float(entity_data.get('movedistance', 100))  # Type: float
        instance.startsound = entity_data.get('startsound', None)  # Type: sound
        instance.stopsound = entity_data.get('stopsound', None)  # Type: sound
        instance.WaterMaterial = entity_data.get('watermaterial', "liquids/c4a1_water_green")  # Type: material
        instance.WaveHeight = entity_data.get('waveheight', "3.0")  # Type: string


class func_rotating(Shadow, RenderFields, Origin, Targetname, Parentname, Angles):
    def __init__(self):
        super(RenderFields).__init__()
        super(Shadow).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        self.maxspeed = 100  # Type: integer
        self.fanfriction = 20  # Type: integer
        self.message = None  # Type: sound
        self.volume = 10  # Type: integer
        self._minlight = None  # Type: string
        self.dmg = None  # Type: integer
        self.solidbsp = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.maxspeed = parse_source_value(entity_data.get('maxspeed', 100))  # Type: integer
        instance.fanfriction = parse_source_value(entity_data.get('fanfriction', 20))  # Type: integer
        instance.message = entity_data.get('message', None)  # Type: sound
        instance.volume = parse_source_value(entity_data.get('volume', 10))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.dmg = parse_source_value(entity_data.get('dmg', 0))  # Type: integer
        instance.solidbsp = entity_data.get('solidbsp', None)  # Type: choices


class func_platrot(Shadow, RenderFields, BasePlat, Origin, Targetname, Parentname, Angles):
    def __init__(self):
        super(RenderFields).__init__()
        super(Shadow).__init__()
        super(BasePlat).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        self.noise1 = None  # Type: sound
        self.noise2 = None  # Type: sound
        self.speed = 50  # Type: integer
        self.height = None  # Type: integer
        self.rotation = None  # Type: integer
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        BasePlat.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.noise1 = entity_data.get('noise1', None)  # Type: sound
        instance.noise2 = entity_data.get('noise2', None)  # Type: sound
        instance.speed = parse_source_value(entity_data.get('speed', 50))  # Type: integer
        instance.height = parse_source_value(entity_data.get('height', 0))  # Type: integer
        instance.rotation = parse_source_value(entity_data.get('rotation', 0))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class keyframe_track(Parentname, Angles, KeyFrame, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(KeyFrame).__init__()
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        KeyFrame.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)


class move_keyframed(Parentname, Mover, KeyFrame, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Mover).__init__()
        super(KeyFrame).__init__()
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Mover.from_dict(instance, entity_data)
        KeyFrame.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)


class move_track(Parentname, Mover, KeyFrame, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Mover).__init__()
        super(KeyFrame).__init__()
        super(Targetname).__init__()
        self.WheelBaseLength = 50  # Type: integer
        self.Damage = None  # Type: integer
        self.NoRotate = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Mover.from_dict(instance, entity_data)
        KeyFrame.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.WheelBaseLength = parse_source_value(entity_data.get('wheelbaselength', 50))  # Type: integer
        instance.Damage = parse_source_value(entity_data.get('damage', 0))  # Type: integer
        instance.NoRotate = entity_data.get('norotate', None)  # Type: choices


class RopeKeyFrame(DXLevelChoice):
    def __init__(self):
        super(DXLevelChoice).__init__()
        self.Slack = 25  # Type: integer
        self.Type = None  # Type: choices
        self.Subdiv = 2  # Type: integer
        self.Barbed = None  # Type: choices
        self.Width = "2"  # Type: string
        self.TextureScale = "1"  # Type: string
        self.Collide = None  # Type: choices
        self.Dangling = None  # Type: choices
        self.Breakable = None  # Type: choices
        self.RopeMaterial = "cable/cable.vmt"  # Type: material
        self.NoWind = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        DXLevelChoice.from_dict(instance, entity_data)
        instance.Slack = parse_source_value(entity_data.get('slack', 25))  # Type: integer
        instance.Type = entity_data.get('type', None)  # Type: choices
        instance.Subdiv = parse_source_value(entity_data.get('subdiv', 2))  # Type: integer
        instance.Barbed = entity_data.get('barbed', None)  # Type: choices
        instance.Width = entity_data.get('width', "2")  # Type: string
        instance.TextureScale = entity_data.get('texturescale', "1")  # Type: string
        instance.Collide = entity_data.get('collide', None)  # Type: choices
        instance.Dangling = entity_data.get('dangling', None)  # Type: choices
        instance.Breakable = entity_data.get('breakable', None)  # Type: choices
        instance.RopeMaterial = entity_data.get('ropematerial', "cable/cable.vmt")  # Type: material
        instance.NoWind = entity_data.get('nowind', None)  # Type: choices


class keyframe_rope(Parentname, KeyFrame, Targetname, RopeKeyFrame):
    model_ = "models/editor/axis_helper_thick.mdl"
    def __init__(self):
        super(RopeKeyFrame).__init__()
        super(Parentname).__init__()
        super(KeyFrame).__init__()
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        KeyFrame.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RopeKeyFrame.from_dict(instance, entity_data)


class move_rope(Parentname, KeyFrame, Targetname, RopeKeyFrame):
    model_ = "models/editor/axis_helper.mdl"
    def __init__(self):
        super(RopeKeyFrame).__init__()
        super(Parentname).__init__()
        super(KeyFrame).__init__()
        super(Targetname).__init__()
        self.PositionInterpolator = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        KeyFrame.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RopeKeyFrame.from_dict(instance, entity_data)
        instance.PositionInterpolator = entity_data.get('positioninterpolator', "CHOICES NOT SUPPORTED")  # Type: choices


class Button(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class func_button(RenderFields, Button, Origin, DamageFilter, Parentname, Targetname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Button).__init__()
        super(Origin).__init__()
        super(DamageFilter).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
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
        RenderFields.from_dict(instance, entity_data)
        Button.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        DamageFilter.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class func_detail_blocker(Empty):
    def __init__(self):
        super(Empty).__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Empty.from_dict(instance, entity_data)


class func_rot_button(Global, Button, Origin, Targetname, EnableDisable, Parentname, Angles):
    def __init__(self):
        super(Global).__init__()
        super(Button).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        self.master = None  # Type: string
        self.speed = 50  # Type: integer
        self.health = None  # Type: integer
        self.sounds = "CHOICES NOT SUPPORTED"  # Type: choices
        self.wait = 3  # Type: integer
        self.distance = 90  # Type: integer
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Global.from_dict(instance, entity_data)
        Button.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.master = entity_data.get('master', None)  # Type: string
        instance.speed = parse_source_value(entity_data.get('speed', 50))  # Type: integer
        instance.health = parse_source_value(entity_data.get('health', 0))  # Type: integer
        instance.sounds = entity_data.get('sounds', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.wait = parse_source_value(entity_data.get('wait', 3))  # Type: integer
        instance.distance = parse_source_value(entity_data.get('distance', 90))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class momentary_rot_button(RenderFields, Origin, Targetname, Parentname, Angles):
    def __init__(self):
        super(RenderFields).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        self.speed = 50  # Type: integer
        self.master = None  # Type: string
        self.sounds = None  # Type: choices
        self.distance = 90  # Type: integer
        self.returnspeed = None  # Type: integer
        self._minlight = None  # Type: string
        self.startposition = None  # Type: float
        self.startdirection = "CHOICES NOT SUPPORTED"  # Type: choices
        self.solidbsp = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        RenderFields.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.speed = parse_source_value(entity_data.get('speed', 50))  # Type: integer
        instance.master = entity_data.get('master', None)  # Type: string
        instance.sounds = entity_data.get('sounds', None)  # Type: choices
        instance.distance = parse_source_value(entity_data.get('distance', 90))  # Type: integer
        instance.returnspeed = parse_source_value(entity_data.get('returnspeed', 0))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.startposition = float(entity_data.get('startposition', 0))  # Type: float
        instance.startdirection = entity_data.get('startdirection', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.solidbsp = entity_data.get('solidbsp', None)  # Type: choices


class Door(Global, Shadow, RenderFields, Parentname, Targetname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Global).__init__()
        super(Shadow).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
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
        Global.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class prop_door_rotating(Global, Targetname, Studiomodel, Parentname, Angles):
    def __init__(self):
        super(Global).__init__()
        super(Targetname).__init__()
        super(Studiomodel).__init__()
        super(Parentname).__init__()
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
        self.forceclosed = None  # Type: choices
        self.opendir = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Global.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
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
        instance.forceclosed = entity_data.get('forceclosed', None)  # Type: choices
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


class func_dustmotes(BModelParticleSpawner, Targetname):
    def __init__(self):
        super(BModelParticleSpawner).__init__()
        super(Targetname).__init__()
        self.SizeMin = "10"  # Type: string
        self.SizeMax = "20"  # Type: string
        self.Alpha = 255  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BModelParticleSpawner.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class func_dustcloud(BModelParticleSpawner, Targetname):
    def __init__(self):
        super(BModelParticleSpawner).__init__()
        super(Targetname).__init__()
        self.Alpha = 30  # Type: integer
        self.SizeMin = "100"  # Type: string
        self.SizeMax = "200"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BModelParticleSpawner.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.Alpha = parse_source_value(entity_data.get('alpha', 30))  # Type: integer
        instance.SizeMin = entity_data.get('sizemin', "100")  # Type: string
        instance.SizeMax = entity_data.get('sizemax', "200")  # Type: string


class env_dustpuff(Parentname, Angles, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.scale = 8  # Type: float
        self.speed = 16  # Type: float
        self.color = [128, 128, 128]  # Type: color255

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.scale = float(entity_data.get('scale', 8))  # Type: float
        instance.speed = float(entity_data.get('speed', 16))  # Type: float
        instance.color = parse_int_vector(entity_data.get('color', "128 128 128"))  # Type: color255


class env_particlescript(Parentname, Angles, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.model = "models/Ambient_citadel_paths.mdl"  # Type: studio

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', "models/Ambient_citadel_paths.mdl")  # Type: studio


class env_effectscript(Parentname, Angles, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.model = "models/Effects/teleporttrail.mdl"  # Type: studio
        self.scriptfile = "scripts/effects/testeffect.txt"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class point_viewcontrol(Angles, Targetname, Parentname):
    viewport_model = "models/editor/camera.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
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
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
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


class logic_relay(EnableDisable, Targetname):
    icon_sprite = "editor/logic_relay.vmt"
    def __init__(self):
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class logic_timer(EnableDisable, Targetname):
    icon_sprite = "editor/logic_timer.vmt"
    def __init__(self):
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.UseRandomTime = None  # Type: choices
        self.LowerRandomBound = None  # Type: string
        self.UpperRandomBound = None  # Type: string
        self.RefireTime = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class env_microphone(Parentname, Targetname, EnableDisable):
    icon_sprite = "editor/env_microphone.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
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
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.SpeakerName = entity_data.get('speakername', None)  # Type: target_destination
        instance.ListenFilter = entity_data.get('listenfilter', None)  # Type: filterclass
        instance.speaker_dsp_preset = entity_data.get('speaker_dsp_preset', None)  # Type: choices
        instance.Sensitivity = float(entity_data.get('sensitivity', 1))  # Type: float
        instance.SmoothFactor = float(entity_data.get('smoothfactor', 0))  # Type: float
        instance.MaxRange = float(entity_data.get('maxrange', 240))  # Type: float


class math_remap(EnableDisable, Targetname):
    def __init__(self):
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.in1 = None  # Type: integer
        self.in2 = 1  # Type: integer
        self.out1 = None  # Type: integer
        self.out2 = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class math_counter(EnableDisable, Targetname):
    icon_sprite = "editor/math_counter.vmt"
    def __init__(self):
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.startvalue = None  # Type: integer
        self.min = None  # Type: integer
        self.max = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class env_entity_maker(Parentname, Angles, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.EntityTemplate = None  # Type: target_destination
        self.PostSpawnSpeed = 0  # Type: float
        self.PostSpawnDirection = [0.0, 0.0, 0.0]  # Type: angle
        self.PostSpawnDirectionVariance = 0.15  # Type: float
        self.PostSpawnInheritAngles = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.filtertype = entity_data.get('filtertype', None)  # Type: choices
        instance.Filter01 = entity_data.get('filter01', None)  # Type: filterclass
        instance.Filter02 = entity_data.get('filter02', None)  # Type: filterclass
        instance.Filter03 = entity_data.get('filter03', None)  # Type: filterclass
        instance.Filter04 = entity_data.get('filter04', None)  # Type: filterclass
        instance.Filter05 = entity_data.get('filter05', None)  # Type: filterclass


class filter_activator_model(BaseFilter):
    icon_sprite = "editor/filter_name.vmt"
    def __init__(self):
        super(BaseFilter).__init__()
        self.model = None  # Type: studio

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', None)  # Type: studio


class filter_activator_name(BaseFilter):
    icon_sprite = "editor/filter_name.vmt"
    def __init__(self):
        super(BaseFilter).__init__()
        self.filtername = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.filtername = entity_data.get('filtername', None)  # Type: target_destination


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


class point_anglesensor(Parentname, Targetname, EnableDisable):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.lookatname = None  # Type: target_destination
        self.duration = None  # Type: float
        self.tolerance = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class point_proximity_sensor(Parentname, Angles, Targetname, EnableDisable):
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination


class point_teleport(Angles, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class func_physbox(RenderFields, BreakableBrush, Origin):
    def __init__(self):
        super(BreakableBrush).__init__()
        super(RenderFields).__init__()
        super(Shadow).__init__()
        super(Origin).__init__()
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
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        BreakableBrush.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.Damagetype = entity_data.get('damagetype', None)  # Type: choices
        instance.massScale = float(entity_data.get('massscale', 0))  # Type: float
        instance.overridescript = entity_data.get('overridescript', None)  # Type: string
        instance.damagetoenablemotion = parse_source_value(entity_data.get('damagetoenablemotion', 0))  # Type: integer
        instance.forcetoenablemotion = float(entity_data.get('forcetoenablemotion', 0))  # Type: float
        instance.preferredcarryangles = parse_float_vector(entity_data.get('preferredcarryangles', "0 0 0"))  # Type: vector
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


class phys_keepupright(Angles, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.attach1 = None  # Type: target_destination
        self.angularlimit = 15  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.attach1 = entity_data.get('attach1', None)  # Type: target_destination
        instance.angularlimit = float(entity_data.get('angularlimit', 15))  # Type: float


class physics_cannister(Angles, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
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
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class info_constraint_anchor(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.massScale = 1  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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
    model_ = "models/editor/axis_helper.mdl"
    def __init__(self):
        super(TwoObjectPhysics).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TwoObjectPhysics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class phys_pulleyconstraint(TwoObjectPhysics):
    model_ = "models/editor/axis_helper.mdl"
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
    model_ = "models/editor/axis_helper.mdl"
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
    model_ = "models/editor/axis_helper.mdl"
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
    model_ = "models/editor/axis_helper.mdl"
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


class phys_magnet(Parentname, Studiomodel, Angles, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Studiomodel).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.forcelimit = 0  # Type: float
        self.torquelimit = 0  # Type: float
        self.massScale = 0  # Type: float
        self.overridescript = None  # Type: string
        self.maxobjects = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class prop_static_base(Angles, DXLevelChoice):
    def __init__(self):
        super(Angles).__init__()
        super(DXLevelChoice).__init__()
        self.model = None  # Type: studio
        self.skin = None  # Type: integer
        self.solid = "CHOICES NOT SUPPORTED"  # Type: choices
        self.disableshadows = None  # Type: choices
        self.screenspacefade = None  # Type: choices
        self.fademindist = -1  # Type: float
        self.fademaxdist = None  # Type: float
        self.fadescale = 1  # Type: float
        self.lightingorigin = None  # Type: target_destination
        self.disablevertexlighting = None  # Type: choices
        self.disableselfshadowing = None  # Type: choices
        self.ignorenormals = None  # Type: choices
        self.generatelightmaps = None  # Type: choices
        self.lightmapresolutionx = 32  # Type: integer
        self.lightmapresolutiony = 32  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        DXLevelChoice.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', None)  # Type: studio
        instance.skin = parse_source_value(entity_data.get('skin', 0))  # Type: integer
        instance.solid = entity_data.get('solid', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.disableshadows = entity_data.get('disableshadows', None)  # Type: choices
        instance.screenspacefade = entity_data.get('screenspacefade', None)  # Type: choices
        instance.fademindist = float(entity_data.get('fademindist', -1))  # Type: float
        instance.fademaxdist = float(entity_data.get('fademaxdist', 0))  # Type: float
        instance.fadescale = float(entity_data.get('fadescale', 1))  # Type: float
        instance.lightingorigin = entity_data.get('lightingorigin', None)  # Type: target_destination
        instance.disablevertexlighting = entity_data.get('disablevertexlighting', None)  # Type: choices
        instance.disableselfshadowing = entity_data.get('disableselfshadowing', None)  # Type: choices
        instance.ignorenormals = entity_data.get('ignorenormals', None)  # Type: choices
        instance.generatelightmaps = entity_data.get('generatelightmaps', None)  # Type: choices
        instance.lightmapresolutionx = parse_source_value(entity_data.get('lightmapresolutionx', 32))  # Type: integer
        instance.lightmapresolutiony = parse_source_value(entity_data.get('lightmapresolutiony', 32))  # Type: integer


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


class prop_dynamic_base(Global, BreakableProp, RenderFields, BaseFadeProp, Studiomodel, Parentname, Angles, DXLevelChoice):
    def __init__(self):
        super(BreakableProp).__init__()
        super(RenderFields).__init__()
        super(Global).__init__()
        super(BaseFadeProp).__init__()
        super(Studiomodel).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(DXLevelChoice).__init__()
        self.solid = "CHOICES NOT SUPPORTED"  # Type: choices
        self.DefaultAnim = None  # Type: string
        self.RandomAnimation = None  # Type: choices
        self.MinAnimTime = 5  # Type: float
        self.MaxAnimTime = 10  # Type: float
        self.SetBodyGroup = None  # Type: integer
        self.DisableBoneFollowers = None  # Type: choices
        self.lightingorigin = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Global.from_dict(instance, entity_data)
        BreakableProp.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        DXLevelChoice.from_dict(instance, entity_data)
        instance.solid = entity_data.get('solid', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.DefaultAnim = entity_data.get('defaultanim', None)  # Type: string
        instance.RandomAnimation = entity_data.get('randomanimation', None)  # Type: choices
        instance.MinAnimTime = float(entity_data.get('minanimtime', 5))  # Type: float
        instance.MaxAnimTime = float(entity_data.get('maxanimtime', 10))  # Type: float
        instance.SetBodyGroup = parse_source_value(entity_data.get('setbodygroup', 0))  # Type: integer
        instance.DisableBoneFollowers = entity_data.get('disablebonefollowers', None)  # Type: choices
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


class prop_dynamic(EnableDisable, prop_dynamic_base):
    def __init__(self):
        super(prop_dynamic_base).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.IgnoreNPCCollisions = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.IgnoreNPCCollisions = entity_data.get('ignorenpccollisions', None)  # Type: choices


class prop_dynamic_playertouch(EnableDisable, prop_dynamic_base):
    def __init__(self):
        super(prop_dynamic_base).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.IgnoreNPCCollisions = None  # Type: choices
        self.health = None  # Type: integer
        self.m_szParticlesOnBreak = None  # Type: string
        self.m_szSoundOnBreak = None  # Type: string
        self.m_FDamageToPlayerOnTouch = 20  # Type: float
        self.fGluonDmgMultiplier = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.IgnoreNPCCollisions = entity_data.get('ignorenpccollisions', None)  # Type: choices
        instance.health = parse_source_value(entity_data.get('health', 0))  # Type: integer
        instance.m_szParticlesOnBreak = entity_data.get('m_szparticlesonbreak', None)  # Type: string
        instance.m_szSoundOnBreak = entity_data.get('m_szsoundonbreak', None)  # Type: string
        instance.m_FDamageToPlayerOnTouch = float(entity_data.get('m_fdamagetoplayerontouch', 20))  # Type: float
        instance.fGluonDmgMultiplier = float(entity_data.get('fgluondmgmultiplier', 1.0))  # Type: float


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


class BasePropPhysics(Global, BreakableProp, BaseFadeProp, Studiomodel, Angles, DXLevelChoice):
    def __init__(self):
        super(BreakableProp).__init__()
        super(Global).__init__()
        super(BaseFadeProp).__init__()
        super(Studiomodel).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        super(DXLevelChoice).__init__()
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
        self.physicsdamagescale = 1.0  # Type: float
        self.physicsdamagelimit = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Global.from_dict(instance, entity_data)
        BreakableProp.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        DXLevelChoice.from_dict(instance, entity_data)
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
        instance.physicsdamagescale = float(entity_data.get('physicsdamagescale', 1.0))  # Type: float
        instance.physicsdamagelimit = float(entity_data.get('physicsdamagelimit', 0))  # Type: float


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


class prop_physics(RenderFields, BasePropPhysics):
    def __init__(self):
        super(BasePropPhysics).__init__()
        super(RenderFields).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        RenderFields.from_dict(instance, entity_data)
        BasePropPhysics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class prop_physics_teleprop(RenderFields, BasePropPhysics):
    def __init__(self):
        super(BasePropPhysics).__init__()
        super(RenderFields).__init__()
        self.origin = [0, 0, 0]
        self.m_bHammerEntity = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_szOwnnerPortalName = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        RenderFields.from_dict(instance, entity_data)
        BasePropPhysics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.m_bHammerEntity = entity_data.get('m_bhammerentity', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.m_szOwnnerPortalName = entity_data.get('m_szownnerportalname', None)  # Type: string


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


class prop_ragdoll(BaseFadeProp, EnableDisable, Studiomodel, Angles, Targetname, DXLevelChoice):
    def __init__(self):
        super(BaseFadeProp).__init__()
        super(EnableDisable).__init__()
        super(Studiomodel).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        super(DXLevelChoice).__init__()
        self.origin = [0, 0, 0]
        self.MimicName = None  # Type: string
        self.angleOverride = None  # Type: string
        self.health = 100  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFadeProp.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        DXLevelChoice.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.MimicName = entity_data.get('mimicname', None)  # Type: string
        instance.angleOverride = entity_data.get('angleoverride', None)  # Type: string
        instance.health = parse_source_value(entity_data.get('health', 100))  # Type: integer


class prop_ragdoll_original(BaseFadeProp, EnableDisable, Studiomodel, Angles, Targetname, DXLevelChoice):
    def __init__(self):
        super(BaseFadeProp).__init__()
        super(EnableDisable).__init__()
        super(Studiomodel).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        super(DXLevelChoice).__init__()
        self.origin = [0, 0, 0]
        self.MimicName = None  # Type: string
        self.angleOverride = None  # Type: string
        self.health = 100  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFadeProp.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        DXLevelChoice.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.MimicName = entity_data.get('mimicname', None)  # Type: string
        instance.angleOverride = entity_data.get('angleoverride', None)  # Type: string
        instance.health = parse_source_value(entity_data.get('health', 100))  # Type: integer


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


class func_breakable(RenderFields, BreakableBrush, Origin):
    def __init__(self):
        super(BreakableBrush).__init__()
        super(RenderFields).__init__()
        super(Shadow).__init__()
        super(Origin).__init__()
        self.minhealthdmg = None  # Type: integer
        self._minlight = None  # Type: string
        self.physdamagescale = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        RenderFields.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        BreakableBrush.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.minhealthdmg = parse_source_value(entity_data.get('minhealthdmg', 0))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.physdamagescale = float(entity_data.get('physdamagescale', 1.0))  # Type: float


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


class func_conveyor(Parentname, Shadow, Targetname, RenderFields):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        self.movedir = [0.0, 0.0, 0.0]  # Type: angle
        self.speed = "100"  # Type: string
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance.movedir = parse_float_vector(entity_data.get('movedir', "0 0 0"))  # Type: angle
        instance.speed = entity_data.get('speed', "100")  # Type: string
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class func_detail(DXLevelChoice):
    def __init__(self):
        super(DXLevelChoice).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        DXLevelChoice.from_dict(instance, entity_data)


class func_viscluster(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class func_illusionary(Shadow, RenderFields, Origin, Parentname, Targetname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Shadow).__init__()
        super(Origin).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class func_precipitation(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.renderamt = 5  # Type: integer
        self.rendercolor = [100, 100, 100]  # Type: color255
        self.preciptype = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.renderamt = parse_source_value(entity_data.get('renderamt', 5))  # Type: integer
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "100 100 100"))  # Type: color255
        instance.preciptype = entity_data.get('preciptype', None)  # Type: choices


class func_wall_toggle(func_wall):
    def __init__(self):
        super(func_wall).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        func_wall.from_dict(instance, entity_data)


class func_guntarget(Parentname, Global, Targetname, RenderFields):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Global).__init__()
        super(Targetname).__init__()
        self.speed = 100  # Type: integer
        self.target = None  # Type: target_destination
        self.health = None  # Type: integer
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
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


class Trackchange(Global, RenderFields, PlatSounds, Parentname, Targetname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Global).__init__()
        super(PlatSounds).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.height = None  # Type: integer
        self.rotation = None  # Type: integer
        self.train = None  # Type: target_destination
        self.toptrack = None  # Type: target_destination
        self.bottomtrack = None  # Type: target_destination
        self.speed = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Global.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        PlatSounds.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.height = parse_source_value(entity_data.get('height', 0))  # Type: integer
        instance.rotation = parse_source_value(entity_data.get('rotation', 0))  # Type: integer
        instance.train = entity_data.get('train', None)  # Type: target_destination
        instance.toptrack = entity_data.get('toptrack', None)  # Type: target_destination
        instance.bottomtrack = entity_data.get('bottomtrack', None)  # Type: target_destination
        instance.speed = parse_source_value(entity_data.get('speed', 0))  # Type: integer


class BaseTrain(Global, Shadow, RenderFields, Origin, Parentname, Targetname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Global).__init__()
        super(Shadow).__init__()
        super(Origin).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
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
        Global.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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
        self.ManualSpeedChanges = None  # Type: choices
        self.ManualAccelSpeed = None  # Type: float
        self.ManualDecelSpeed = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseTrain.from_dict(instance, entity_data)
        instance.ManualSpeedChanges = entity_data.get('manualspeedchanges', None)  # Type: choices
        instance.ManualAccelSpeed = float(entity_data.get('manualaccelspeed', 0))  # Type: float
        instance.ManualDecelSpeed = float(entity_data.get('manualdecelspeed', 0))  # Type: float


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


class path_track(Parentname, Angles, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.altpath = None  # Type: target_destination
        self.speed = None  # Type: float
        self.radius = None  # Type: float
        self.orientationtype = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class trigger_csm_volume(Trigger):
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
        self.NotLookingFrequency = 0.5  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.LookTime = entity_data.get('looktime', "0.5")  # Type: string
        instance.FieldOfView = entity_data.get('fieldofview', "0.9")  # Type: string
        instance.Timeout = float(entity_data.get('timeout', 0))  # Type: float
        instance.NotLookingFrequency = float(entity_data.get('notlookingfrequency', 0.5))  # Type: float


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


class trigger_impact(Angles, Targetname, Origin):
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        super(Origin).__init__()
        self.Magnitude = 200  # Type: float
        self.noise = 0.1  # Type: float
        self.viewkick = 0.05  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class ai_speechfilter(ResponseContext, Targetname, EnableDisable):
    def __init__(self):
        super(ResponseContext).__init__()
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.subject = None  # Type: target_destination
        self.IdleModifier = 1.0  # Type: float
        self.NeverSayHello = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        ResponseContext.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
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
    icon_sprite = "editor/playerproxy.vmt"
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


class npc_puppet(Parentname, Studiomodel, BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        super(Parentname).__init__()
        super(Studiomodel).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.animationtarget = None  # Type: target_source
        self.attachmentname = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        BaseNPC.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.animationtarget = entity_data.get('animationtarget', None)  # Type: target_source
        instance.attachmentname = entity_data.get('attachmentname', None)  # Type: string


class point_gamestats_counter(EnableDisable, Origin, Targetname):
    def __init__(self):
        super(EnableDisable).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.Name = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Name = entity_data.get('name', None)  # Type: string


class func_instance(Angles):
    def __init__(self):
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.targetname = None  # Type: target_source
        self.file = None  # Type: instance_file
        self.fixup_style = "CHOICES NOT SUPPORTED"  # Type: choices
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
        instance.fixup_style = entity_data.get('fixup_style', "CHOICES NOT SUPPORTED")  # Type: choices
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
        self.AlwaysTransition = "CHOICES NOT SUPPORTED"  # Type: choices
        self.DontPickupWeapons = "CHOICES NOT SUPPORTED"  # Type: choices
        self.GameEndAlly = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.AlwaysTransition = entity_data.get('alwaystransition', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.DontPickupWeapons = entity_data.get('dontpickupweapons', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.GameEndAlly = entity_data.get('gameendally', "CHOICES NOT SUPPORTED")  # Type: choices


class RappelNPC(BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        self.waitingtorappel = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.waitingtorappel = entity_data.get('waitingtorappel', "CHOICES NOT SUPPORTED")  # Type: choices


class trigger_physics_trap(Angles, Trigger):
    def __init__(self):
        super(Trigger).__init__()
        super(Angles).__init__()
        self.dissolvetype = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Trigger.from_dict(instance, entity_data)
        instance.dissolvetype = entity_data.get('dissolvetype', "CHOICES NOT SUPPORTED")  # Type: choices


class trigger_weapon_dissolve(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        self.emittername = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.emittername = entity_data.get('emittername', None)  # Type: target_destination


class trigger_weapon_strip(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        self.KillWeapons = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.KillWeapons = entity_data.get('killweapons', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_crow(BaseNPC):
    model_ = "models/crow.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.deaf = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.deaf = entity_data.get('deaf', None)  # Type: choices


class npc_seagull(BaseNPC):
    model_ = "models/seagull.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.deaf = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.deaf = entity_data.get('deaf', None)  # Type: choices


class npc_pigeon(BaseNPC):
    model_ = "models/pigeon.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.deaf = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.deaf = entity_data.get('deaf', None)  # Type: choices


class npc_bullseye(Parentname, BaseNPC):
    icon_sprite = "editor/bullseye.vmt"
    def __init__(self):
        super(BaseNPC).__init__()
        super(Parentname).__init__()
        self.health = 35  # Type: integer
        self.minangle = "360"  # Type: string
        self.mindist = "0"  # Type: string
        self.autoaimradius = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        BaseNPC.from_dict(instance, entity_data)
        instance.health = parse_source_value(entity_data.get('health', 35))  # Type: integer
        instance.minangle = entity_data.get('minangle', "360")  # Type: string
        instance.mindist = entity_data.get('mindist', "0")  # Type: string
        instance.autoaimradius = float(entity_data.get('autoaimradius', 0))  # Type: float


class npc_enemyfinder(Parentname, BaseNPC):
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
        self.StartOn = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        BaseNPC.from_dict(instance, entity_data)
        instance.FieldOfView = entity_data.get('fieldofview', "0.2")  # Type: string
        instance.MinSearchDist = parse_source_value(entity_data.get('minsearchdist', 0))  # Type: integer
        instance.MaxSearchDist = parse_source_value(entity_data.get('maxsearchdist', 2048))  # Type: integer
        instance.freepass_timetotrigger = float(entity_data.get('freepass_timetotrigger', 0))  # Type: float
        instance.freepass_duration = float(entity_data.get('freepass_duration', 0))  # Type: float
        instance.freepass_movetolerance = float(entity_data.get('freepass_movetolerance', 120))  # Type: float
        instance.freepass_refillrate = float(entity_data.get('freepass_refillrate', 0.5))  # Type: float
        instance.freepass_peektime = float(entity_data.get('freepass_peektime', 0))  # Type: float
        instance.StartOn = entity_data.get('starton', "CHOICES NOT SUPPORTED")  # Type: choices


class env_gunfire(Parentname, Targetname, EnableDisable):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.minburstsize = 2  # Type: integer
        self.maxburstsize = 7  # Type: integer
        self.minburstdelay = 2  # Type: float
        self.maxburstdelay = 5  # Type: float
        self.rateoffire = 10  # Type: float
        self.spread = "CHOICES NOT SUPPORTED"  # Type: choices
        self.bias = "CHOICES NOT SUPPORTED"  # Type: choices
        self.collisions = None  # Type: choices
        self.shootsound = "CHOICES NOT SUPPORTED"  # Type: choices
        self.tracertype = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.minburstsize = parse_source_value(entity_data.get('minburstsize', 2))  # Type: integer
        instance.maxburstsize = parse_source_value(entity_data.get('maxburstsize', 7))  # Type: integer
        instance.minburstdelay = float(entity_data.get('minburstdelay', 2))  # Type: float
        instance.maxburstdelay = float(entity_data.get('maxburstdelay', 5))  # Type: float
        instance.rateoffire = float(entity_data.get('rateoffire', 10))  # Type: float
        instance.spread = entity_data.get('spread', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.bias = entity_data.get('bias', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.collisions = entity_data.get('collisions', None)  # Type: choices
        instance.shootsound = entity_data.get('shootsound', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.tracertype = entity_data.get('tracertype', "CHOICES NOT SUPPORTED")  # Type: choices


class ai_goal_operator(EnableDisable, Targetname):
    def __init__(self):
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.actor = None  # Type: target_name_or_class
        self.target = None  # Type: target_destination
        self.contexttarget = None  # Type: target_destination
        self.state = None  # Type: choices
        self.moveto = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.actor = entity_data.get('actor', None)  # Type: target_name_or_class
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.contexttarget = entity_data.get('contexttarget', None)  # Type: target_destination
        instance.state = entity_data.get('state', None)  # Type: choices
        instance.moveto = entity_data.get('moveto', "CHOICES NOT SUPPORTED")  # Type: choices


class info_darknessmode_lightsource(EnableDisable, Targetname):
    def __init__(self):
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.LightRadius = 256.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.LightRadius = float(entity_data.get('lightradius', 256.0))  # Type: float


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
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', None)  # Type: studio
        instance.body = parse_source_value(entity_data.get('body', 0))  # Type: integer


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
        Shadow.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
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
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
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
        BaseNPCMaker.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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


class BaseScripted(Parentname, Angles, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.m_iszEntity = None  # Type: target_destination
        self.m_iszIdle = None  # Type: string
        self.m_iszEntry = None  # Type: string
        self.m_iszPlay = None  # Type: string
        self.m_iszPostIdle = None  # Type: string
        self.m_iszCustomMove = None  # Type: string
        self.m_bLoopActionSequence = None  # Type: choices
        self.m_bNoBlendedMovement = None  # Type: choices
        self.m_bSynchPostIdles = None  # Type: choices
        self.m_flRadius = None  # Type: integer
        self.m_flRepeat = None  # Type: integer
        self.m_fMoveTo = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_iszNextScript = None  # Type: target_destination
        self.m_bIgnoreGravity = None  # Type: choices
        self.m_bDisableNPCCollisions = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.m_iszEntity = entity_data.get('m_iszentity', None)  # Type: target_destination
        instance.m_iszIdle = entity_data.get('m_iszidle', None)  # Type: string
        instance.m_iszEntry = entity_data.get('m_iszentry', None)  # Type: string
        instance.m_iszPlay = entity_data.get('m_iszplay', None)  # Type: string
        instance.m_iszPostIdle = entity_data.get('m_iszpostidle', None)  # Type: string
        instance.m_iszCustomMove = entity_data.get('m_iszcustommove', None)  # Type: string
        instance.m_bLoopActionSequence = entity_data.get('m_bloopactionsequence', None)  # Type: choices
        instance.m_bNoBlendedMovement = entity_data.get('m_bnoblendedmovement', None)  # Type: choices
        instance.m_bSynchPostIdles = entity_data.get('m_bsynchpostidles', None)  # Type: choices
        instance.m_flRadius = parse_source_value(entity_data.get('m_flradius', 0))  # Type: integer
        instance.m_flRepeat = parse_source_value(entity_data.get('m_flrepeat', 0))  # Type: integer
        instance.m_fMoveTo = entity_data.get('m_fmoveto', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.m_iszNextScript = entity_data.get('m_isznextscript', None)  # Type: target_destination
        instance.m_bIgnoreGravity = entity_data.get('m_bignoregravity', None)  # Type: choices
        instance.m_bDisableNPCCollisions = entity_data.get('m_bdisablenpccollisions', None)  # Type: choices


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


class scripted_target(Parentname, Targetname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.StartDisabled = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_iszEntity = None  # Type: npcclass
        self.m_flRadius = None  # Type: integer
        self.MoveSpeed = 5  # Type: integer
        self.PauseDuration = None  # Type: integer
        self.EffectDuration = 2  # Type: integer
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.StartDisabled = entity_data.get('startdisabled', "CHOICES NOT SUPPORTED")  # Type: choices
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
        self.StartActive = None  # Type: choices
        self.Reciprocal = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.subject = entity_data.get('subject', None)  # Type: target_name_or_class
        instance.target = entity_data.get('target', None)  # Type: target_name_or_class
        instance.disposition = entity_data.get('disposition', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.radius = float(entity_data.get('radius', 0))  # Type: float
        instance.rank = parse_source_value(entity_data.get('rank', 0))  # Type: integer
        instance.StartActive = entity_data.get('startactive', None)  # Type: choices
        instance.Reciprocal = entity_data.get('reciprocal', None)  # Type: choices


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
        self.Run = "CHOICES NOT SUPPORTED"  # Type: choices
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
        instance.Run = entity_data.get('run', "CHOICES NOT SUPPORTED")  # Type: choices
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
        self.StartActive = None  # Type: choices
        self.MaximumState = "CHOICES NOT SUPPORTED"  # Type: choices
        self.Formation = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.actor = entity_data.get('actor', None)  # Type: target_name_or_class
        instance.goal = entity_data.get('goal', None)  # Type: string
        instance.SearchType = entity_data.get('searchtype', None)  # Type: choices
        instance.StartActive = entity_data.get('startactive', None)  # Type: choices
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


class env_detail_controller(Angles):
    def __init__(self):
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.fademindist = 400  # Type: float
        self.fademaxdist = 1200  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.fademindist = float(entity_data.get('fademindist', 400))  # Type: float
        instance.fademaxdist = float(entity_data.get('fademaxdist', 1200))  # Type: float


class env_global(EnvGlobal):
    def __init__(self):
        super(EnvGlobal).__init__()
        self.origin = [0, 0, 0]
        self.globalstate = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnvGlobal.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.globalstate = entity_data.get('globalstate', None)  # Type: choices


class BaseCharger(Angles, BaseFadeProp, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(BaseFadeProp).__init__()
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)


class item_healthcharger(BaseCharger):
    model_ = "models/props_blackmesa/health_charger.mdl"
    def __init__(self):
        super(BaseCharger).__init__()
        self.origin = [0, 0, 0]
        self.charge = 50  # Type: float
        self.skintype = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseCharger.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.charge = float(entity_data.get('charge', 50))  # Type: float
        instance.skintype = entity_data.get('skintype', "CHOICES NOT SUPPORTED")  # Type: choices


class item_suitcharger(BaseCharger):
    model_ = "models/props_blackmesa/hev_charger.mdl"
    def __init__(self):
        super(BaseCharger).__init__()
        self.origin = [0, 0, 0]
        self.charge = 75  # Type: float
        self.skintype = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseCharger.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.charge = float(entity_data.get('charge', 75))  # Type: float
        instance.skintype = entity_data.get('skintype', "CHOICES NOT SUPPORTED")  # Type: choices


class BasePickup(Angles, BaseFadeProp, Shadow, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(BaseFadeProp).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        self.respawntime = 15  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.respawntime = float(entity_data.get('respawntime', 15))  # Type: float


class item_weapon_357(BasePickup):
    model_ = "models/weapons/w_357.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_weapon_crowbar(BasePickup):
    model_ = "models/weapons/w_crowbar.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_weapon_crossbow(BasePickup):
    model_ = "models/weapons/w_crossbow.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_weapon_frag(BasePickup):
    model_ = "models/weapons/w_grenade.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_weapon_glock(BasePickup):
    model_ = "models/weapons/w_glock.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_weapon_gluon(BasePickup):
    model_ = "models/weapons/w_egon_pickup.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_weapon_hivehand(BasePickup):
    model_ = "models/weapons/w_hgun.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_weapon_mp5(BasePickup):
    model_ = "models/weapons/w_mp5.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_weapon_shotgun(BasePickup):
    model_ = "models/weapons/w_shotgun.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_weapon_rpg(BasePickup):
    model_ = "models/weapons/w_rpg.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_weapon_satchel(BasePickup):
    model_ = "models/weapons/w_satchel.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_weapon_snark(BasePickup):
    model_ = "models/xenians/snarknest.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_weapon_tau(BasePickup):
    model_ = "models/weapons/w_gauss.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_weapon_tripmine(BasePickup):
    model_ = "models/weapons/w_tripmine.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammo_357(BasePickup):
    model_ = "models/weapons/w_357ammobox.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammo_crossbow(BasePickup):
    model_ = "models/weapons/w_crossbow_clip.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammo_glock(BasePickup):
    model_ = "models/weapons/w_9mmclip.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammo_energy(BasePickup):
    model_ = "models/weapons/w_gaussammo.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammo_mp5(BasePickup):
    model_ = "models/weapons/w_9mmARclip.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammo_shotgun(BasePickup):
    model_ = "models/weapons/w_shotbox.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_grenade_mp5(BasePickup):
    model_ = "models/weapons/w_argrenade.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_grenade_rpg(BasePickup):
    model_ = "models/weapons/w_rpgammo.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammo_canister(BasePickup):
    model_ = "models/weapons/w_weaponbox.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]
        self.AmmoGlock = None  # Type: integer
        self.AmmoMp5 = None  # Type: integer
        self.Ammo357 = None  # Type: integer
        self.AmmoBolt = None  # Type: integer
        self.AmmoBuckshot = None  # Type: integer
        self.AmmoEnergy = None  # Type: integer
        self.AmmoMp5Grenade = None  # Type: integer
        self.AmmoRPG = None  # Type: integer
        self.AmmoSatchel = None  # Type: integer
        self.AmmoSnark = None  # Type: integer
        self.AmmoTripmine = None  # Type: integer
        self.AmmoFrag = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.AmmoGlock = parse_source_value(entity_data.get('ammoglock', 0))  # Type: integer
        instance.AmmoMp5 = parse_source_value(entity_data.get('ammomp5', 0))  # Type: integer
        instance.Ammo357 = parse_source_value(entity_data.get('ammo357', 0))  # Type: integer
        instance.AmmoBolt = parse_source_value(entity_data.get('ammobolt', 0))  # Type: integer
        instance.AmmoBuckshot = parse_source_value(entity_data.get('ammobuckshot', 0))  # Type: integer
        instance.AmmoEnergy = parse_source_value(entity_data.get('ammoenergy', 0))  # Type: integer
        instance.AmmoMp5Grenade = parse_source_value(entity_data.get('ammomp5grenade', 0))  # Type: integer
        instance.AmmoRPG = parse_source_value(entity_data.get('ammorpg', 0))  # Type: integer
        instance.AmmoSatchel = parse_source_value(entity_data.get('ammosatchel', 0))  # Type: integer
        instance.AmmoSnark = parse_source_value(entity_data.get('ammosnark', 0))  # Type: integer
        instance.AmmoTripmine = parse_source_value(entity_data.get('ammotripmine', 0))  # Type: integer
        instance.AmmoFrag = parse_source_value(entity_data.get('ammofrag', 0))  # Type: integer


class item_ammo_crate(Angles, BaseFadeProp, Targetname):
    model_ = "models/items/ammocrate_rockets.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(BaseFadeProp).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.model = "models/items/ammocrate_rockets.mdl"  # Type: studio
        self.AmmoType = "CHOICES NOT SUPPORTED"  # Type: choices
        self.isDynamicMoving = None  # Type: integer
        self.AmmoCount = 1  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', "models/items/ammocrate_rockets.mdl")  # Type: studio
        instance.AmmoType = entity_data.get('ammotype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.isDynamicMoving = parse_source_value(entity_data.get('isdynamicmoving', 0))  # Type: integer
        instance.AmmoCount = parse_source_value(entity_data.get('ammocount', 1))  # Type: integer


class item_suit(BasePickup):
    model_ = "models/props_am/hev_suit.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_battery(BasePickup):
    model_ = "models/weapons/w_battery.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_healthkit(BasePickup):
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]
        self.model = "models/weapons/w_medkit.mdl"  # Type: studio

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', "models/weapons/w_medkit.mdl")  # Type: studio


class item_longjump(BasePickup):
    model_ = "models/weapons/w_longjump.mdl"
    def __init__(self):
        super(BasePickup).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePickup.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class BaseGrenade(Angles, Shadow, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)


class grenade_satchel(BaseGrenade):
    model_ = "models/weapons/w_satchel.mdl"
    def __init__(self):
        super(BaseGrenade).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseGrenade.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class grenade_tripmine(BaseGrenade):
    model_ = "models/weapons/w_tripmine.mdl"
    def __init__(self):
        super(BaseGrenade).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseGrenade.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class BaseSentry(BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_plantlight(BaseNPC):
    model_ = "models/props_xen/xen_protractinglight.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.planttype = None  # Type: choices
        self.ICanTakeDamage = "CHOICES NOT SUPPORTED"  # Type: choices
        self.LightColor = [255, 223, 43, 255]  # Type: color255
        self.Intensity = 1000  # Type: float
        self.Range = 500  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.planttype = entity_data.get('planttype', None)  # Type: choices
        instance.ICanTakeDamage = entity_data.get('icantakedamage', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.LightColor = parse_int_vector(entity_data.get('lightcolor', "255 223 43 255"))  # Type: color255
        instance.Intensity = float(entity_data.get('intensity', 1000))  # Type: float
        instance.Range = float(entity_data.get('range', 500))  # Type: float


class npc_plantlight_stalker(Base):
    model_ = "models/props_xen/xen_plantlightstalker.mdl"
    def __init__(self):
        super().__init__()
        self.origin = [0, 0, 0]
        self.planttype = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ICanTakeDamage = None  # Type: choices
        self.LightColor = [255, 223, 43, 255]  # Type: color255
        self.Intensity = 1000  # Type: float
        self.Range = 500  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.planttype = entity_data.get('planttype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ICanTakeDamage = entity_data.get('icantakedamage', None)  # Type: choices
        instance.LightColor = parse_int_vector(entity_data.get('lightcolor', "255 223 43 255"))  # Type: color255
        instance.Intensity = float(entity_data.get('intensity', 1000))  # Type: float
        instance.Range = float(entity_data.get('range', 500))  # Type: float


class npc_puffballfungus(BaseNPC):
    model_ = "models/props_xen/xen_puffballfungus.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.RangeOuter = 500  # Type: float
        self.RangeInner = 250  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.RangeOuter = float(entity_data.get('rangeouter', 500))  # Type: float
        instance.RangeInner = float(entity_data.get('rangeinner', 250))  # Type: float


class npc_xentree(BaseNPC):
    model_ = "models/props_xen/foliage/hacker_tree.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_protozoan(BaseNPC):
    model_ = "models/xenians/protozoan.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_sentry_ceiling(BaseSentry):
    model_ = "models/npcs/sentry_ceiling.mdl"
    def __init__(self):
        super(BaseSentry).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseSentry.from_dict(instance, entity_data)


class npc_sentry_ground(BaseSentry):
    model_ = "models/npcs/sentry_ground.mdl"
    def __init__(self):
        super(BaseSentry).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseSentry.from_dict(instance, entity_data)


class npc_xenturret(BaseNPC):
    model_ = "models/props_xen/xen_turret.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.RangeOuter = 999999  # Type: float
        self.RangeInner = 99999  # Type: float
        self.m_Color = [255, 255, 255]  # Type: color255
        self.nShield = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.RangeOuter = float(entity_data.get('rangeouter', 999999))  # Type: float
        instance.RangeInner = float(entity_data.get('rangeinner', 99999))  # Type: float
        instance.m_Color = parse_int_vector(entity_data.get('m_color', "255 255 255"))  # Type: color255
        instance.nShield = parse_source_value(entity_data.get('nshield', 0))  # Type: integer


class npc_alien_slave_dummy(BaseNPC):
    model_ = "models/vortigaunt_slave.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.passive = "CHOICES NOT SUPPORTED"  # Type: choices
        self.bPlayTeleportAnimOnSpawn = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.passive = entity_data.get('passive', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.bPlayTeleportAnimOnSpawn = entity_data.get('bplayteleportanimonspawn', None)  # Type: choices


class npc_alien_slave(BaseNPC):
    model_ = "models/vortigaunt_slave.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.passive = "CHOICES NOT SUPPORTED"  # Type: choices
        self.bPlayTeleportAnimOnSpawn = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.passive = entity_data.get('passive', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.bPlayTeleportAnimOnSpawn = entity_data.get('bplayteleportanimonspawn', None)  # Type: choices


class npc_xort(BaseNPC):
    model_ = "models/vortigaunt_slave.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.XortState = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_nFearLevel = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_nDamageCallEveryone = -1  # Type: integer
        self.m_fDamageCallRadius = -1  # Type: float
        self.m_nAlertCallEveryone = -1  # Type: integer
        self.m_fAlertCallRadius = -1  # Type: float
        self.FearNodesGroupName = None  # Type: string
        self.HealNodesGroupName = None  # Type: string
        self.bPlayTeleportAnimOnSpawn = None  # Type: choices
        self.bMakeThemStationary = None  # Type: choices
        self.bDisableSpells = None  # Type: choices
        self.CanUseHealingNodes = "CHOICES NOT SUPPORTED"  # Type: choices
        self.CanUseFearNodes = "CHOICES NOT SUPPORTED"  # Type: choices
        self.PossesBreakCooldownOVerride = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.XortState = entity_data.get('xortstate', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.m_nFearLevel = entity_data.get('m_nfearlevel', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.m_nDamageCallEveryone = parse_source_value(entity_data.get('m_ndamagecalleveryone', -1))  # Type: integer
        instance.m_fDamageCallRadius = float(entity_data.get('m_fdamagecallradius', -1))  # Type: float
        instance.m_nAlertCallEveryone = parse_source_value(entity_data.get('m_nalertcalleveryone', -1))  # Type: integer
        instance.m_fAlertCallRadius = float(entity_data.get('m_falertcallradius', -1))  # Type: float
        instance.FearNodesGroupName = entity_data.get('fearnodesgroupname', None)  # Type: string
        instance.HealNodesGroupName = entity_data.get('healnodesgroupname', None)  # Type: string
        instance.bPlayTeleportAnimOnSpawn = entity_data.get('bplayteleportanimonspawn', None)  # Type: choices
        instance.bMakeThemStationary = entity_data.get('bmakethemstationary', None)  # Type: choices
        instance.bDisableSpells = entity_data.get('bdisablespells', None)  # Type: choices
        instance.CanUseHealingNodes = entity_data.get('canusehealingnodes', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.CanUseFearNodes = entity_data.get('canusefearnodes', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.PossesBreakCooldownOVerride = float(entity_data.get('possesbreakcooldownoverride', 0))  # Type: float


class npc_xortEB(BaseNPC):
    model_ = "models/vortigaunt_slave.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.XortState = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_nFearLevel = None  # Type: choices
        self.m_nDamageCallEveryone = -1  # Type: integer
        self.m_fDamageCallRadius = -1  # Type: float
        self.m_nAlertCallEveryone = -1  # Type: integer
        self.m_fAlertCallRadius = -1  # Type: float
        self.FearNodesGroupName = None  # Type: string
        self.HealNodesGroupName = None  # Type: string
        self.bPlayTeleportAnimOnSpawn = None  # Type: choices
        self.bDisableSpells = None  # Type: choices
        self.CanUseHealingNodes = None  # Type: choices
        self.CanUseFearNodes = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.XortState = entity_data.get('xortstate', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.m_nFearLevel = entity_data.get('m_nfearlevel', None)  # Type: choices
        instance.m_nDamageCallEveryone = parse_source_value(entity_data.get('m_ndamagecalleveryone', -1))  # Type: integer
        instance.m_fDamageCallRadius = float(entity_data.get('m_fdamagecallradius', -1))  # Type: float
        instance.m_nAlertCallEveryone = parse_source_value(entity_data.get('m_nalertcalleveryone', -1))  # Type: integer
        instance.m_fAlertCallRadius = float(entity_data.get('m_falertcallradius', -1))  # Type: float
        instance.FearNodesGroupName = entity_data.get('fearnodesgroupname', None)  # Type: string
        instance.HealNodesGroupName = entity_data.get('healnodesgroupname', None)  # Type: string
        instance.bPlayTeleportAnimOnSpawn = entity_data.get('bplayteleportanimonspawn', None)  # Type: choices
        instance.bDisableSpells = entity_data.get('bdisablespells', None)  # Type: choices
        instance.CanUseHealingNodes = entity_data.get('canusehealingnodes', None)  # Type: choices
        instance.CanUseFearNodes = entity_data.get('canusefearnodes', None)  # Type: choices


class npc_headcrab(Parentname, BaseNPC):
    model_ = "models/headcrabclassic.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        super(Parentname).__init__()
        self.startburrowed = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        BaseNPC.from_dict(instance, entity_data)
        instance.startburrowed = entity_data.get('startburrowed', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_headcrab_fast(BaseNPC):
    model_ = "models/Headcrab.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_headcrab_black(BaseNPC):
    model_ = "models/Headcrabblack.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_headcrab_baby(BaseNPC):
    model_ = "models/xenians/bebcrab.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_barnacle(BaseNPC):
    model_ = "models/barnacle.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.RestDist = 16  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.RestDist = float(entity_data.get('restdist', 16))  # Type: float


class npc_beneathticle(BaseNPC):
    model_ = "models/xenians/barnacle_underwater.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.TongueLength = None  # Type: float
        self.TonguePullSpeed = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.TongueLength = float(entity_data.get('tonguelength', 0))  # Type: float
        instance.TonguePullSpeed = float(entity_data.get('tonguepullspeed', 0))  # Type: float


class npc_bullsquid(BaseNPCAssault):
    model_ = "models/xenians/bullsquid.mdl"
    def __init__(self):
        super(BaseNPCAssault).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPCAssault.from_dict(instance, entity_data)


class npc_bullsquid_melee(BaseNPCAssault):
    model_ = "models/xenians/bullsquid.mdl"
    def __init__(self):
        super(BaseNPCAssault).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPCAssault.from_dict(instance, entity_data)


class npc_houndeye(BaseNPCAssault):
    model_ = "models/xenians/houndeye.mdl"
    def __init__(self):
        super(BaseNPCAssault).__init__()
        self.m_bEnableMemoryUpdateEveryFrame = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPCAssault.from_dict(instance, entity_data)
        instance.m_bEnableMemoryUpdateEveryFrame = entity_data.get('m_benablememoryupdateeveryframe', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_houndeye_suicide(npc_houndeye):
    model_ = "models/xenians/houndeye_suicide.mdl"
    def __init__(self):
        super(npc_houndeye).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        npc_houndeye.from_dict(instance, entity_data)


class npc_houndeye_knockback(npc_houndeye):
    model_ = "models/xenians/houndeye_knockback.mdl"
    def __init__(self):
        super(npc_houndeye).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        npc_houndeye.from_dict(instance, entity_data)


class npc_human_assassin(BaseNPC):
    model_ = "models/humans/hassassin.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class BaseMarine(RappelNPC, BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        super(RappelNPC).__init__()
        self.NumGrenades = "CHOICES NOT SUPPORTED"  # Type: choices
        self.additionalequipment = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        RappelNPC.from_dict(instance, entity_data)
        BaseNPC.from_dict(instance, entity_data)
        instance.NumGrenades = entity_data.get('numgrenades', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.additionalequipment = entity_data.get('additionalequipment', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_human_commander(BaseMarine):
    model_ = "models/humans/marine.mdl"
    def __init__(self):
        super(BaseMarine).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseMarine.from_dict(instance, entity_data)


class npc_human_grunt(BaseMarine):
    model_ = "models/humans/marine.mdl"
    def __init__(self):
        super(BaseMarine).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseMarine.from_dict(instance, entity_data)


class npc_human_medic(BaseMarine):
    model_ = "models/humans/marine.mdl"
    def __init__(self):
        super(BaseMarine).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseMarine.from_dict(instance, entity_data)


class npc_human_grenadier(BaseMarine):
    model_ = "models/humans/marine.mdl"
    def __init__(self):
        super(BaseMarine).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseMarine.from_dict(instance, entity_data)


class npc_alien_controller(BaseNPC):
    model_ = "models/xenians/controller.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_xontroller(BaseNPC):
    model_ = "models/xenians/controller.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.mainbehaviortreename = None  # Type: string
        self.assaultbehaviortreename = None  # Type: string
        self.preferred_pawns_group_01 = None  # Type: target_destination
        self.preferred_pawns_group_02 = None  # Type: target_destination
        self.preferred_pawns_group_03 = None  # Type: target_destination
        self.preferred_pawns_group_04 = None  # Type: target_destination
        self.preferred_pawns_group_05 = None  # Type: target_destination
        self.preferred_pawns_group_06 = None  # Type: target_destination
        self.preferred_pawns_group_07 = None  # Type: target_destination
        self.preferred_pawns_group_08 = None  # Type: target_destination
        self.preferred_pawns_group_09 = None  # Type: target_destination
        self.preferred_pawns_group_10 = None  # Type: target_destination
        self.preferred_pawns_group_11 = None  # Type: target_destination
        self.preferred_pawns_group_12 = None  # Type: target_destination
        self.preferred_pawns_group_13 = None  # Type: target_destination
        self.preferred_pawns_group_14 = None  # Type: target_destination
        self.preferred_pawns_group_15 = None  # Type: target_destination
        self.preferred_pawns_group_16 = None  # Type: target_destination
        self.select_preferred_pawns_only = None  # Type: choices
        self.mindcontrol_attacks_disabled = None  # Type: choices
        self.attack_mode_energy_enabled = "CHOICES NOT SUPPORTED"  # Type: choices
        self.attack_mode_cluster_enabled = "CHOICES NOT SUPPORTED"  # Type: choices
        self.attack_mode_brainwash_enabled = "CHOICES NOT SUPPORTED"  # Type: choices
        self.attack_mode_throw_enabled = "CHOICES NOT SUPPORTED"  # Type: choices
        self.attack_mode_smash_enabled = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.mainbehaviortreename = entity_data.get('mainbehaviortreename', None)  # Type: string
        instance.assaultbehaviortreename = entity_data.get('assaultbehaviortreename', None)  # Type: string
        instance.preferred_pawns_group_01 = entity_data.get('preferred_pawns_group_01', None)  # Type: target_destination
        instance.preferred_pawns_group_02 = entity_data.get('preferred_pawns_group_02', None)  # Type: target_destination
        instance.preferred_pawns_group_03 = entity_data.get('preferred_pawns_group_03', None)  # Type: target_destination
        instance.preferred_pawns_group_04 = entity_data.get('preferred_pawns_group_04', None)  # Type: target_destination
        instance.preferred_pawns_group_05 = entity_data.get('preferred_pawns_group_05', None)  # Type: target_destination
        instance.preferred_pawns_group_06 = entity_data.get('preferred_pawns_group_06', None)  # Type: target_destination
        instance.preferred_pawns_group_07 = entity_data.get('preferred_pawns_group_07', None)  # Type: target_destination
        instance.preferred_pawns_group_08 = entity_data.get('preferred_pawns_group_08', None)  # Type: target_destination
        instance.preferred_pawns_group_09 = entity_data.get('preferred_pawns_group_09', None)  # Type: target_destination
        instance.preferred_pawns_group_10 = entity_data.get('preferred_pawns_group_10', None)  # Type: target_destination
        instance.preferred_pawns_group_11 = entity_data.get('preferred_pawns_group_11', None)  # Type: target_destination
        instance.preferred_pawns_group_12 = entity_data.get('preferred_pawns_group_12', None)  # Type: target_destination
        instance.preferred_pawns_group_13 = entity_data.get('preferred_pawns_group_13', None)  # Type: target_destination
        instance.preferred_pawns_group_14 = entity_data.get('preferred_pawns_group_14', None)  # Type: target_destination
        instance.preferred_pawns_group_15 = entity_data.get('preferred_pawns_group_15', None)  # Type: target_destination
        instance.preferred_pawns_group_16 = entity_data.get('preferred_pawns_group_16', None)  # Type: target_destination
        instance.select_preferred_pawns_only = entity_data.get('select_preferred_pawns_only', None)  # Type: choices
        instance.mindcontrol_attacks_disabled = entity_data.get('mindcontrol_attacks_disabled', None)  # Type: choices
        instance.attack_mode_energy_enabled = entity_data.get('attack_mode_energy_enabled', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.attack_mode_cluster_enabled = entity_data.get('attack_mode_cluster_enabled', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.attack_mode_brainwash_enabled = entity_data.get('attack_mode_brainwash_enabled', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.attack_mode_throw_enabled = entity_data.get('attack_mode_throw_enabled', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.attack_mode_smash_enabled = entity_data.get('attack_mode_smash_enabled', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_alien_grunt_unarmored(BaseNPC):
    model_ = "models/xenians/agrunt_unarmored.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_alien_grunt_melee(BaseNPC):
    model_ = "models/xenians/agrunt_unarmored.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_alien_grunt(BaseNPC):
    model_ = "models/xenians/agrunt.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_alien_grunt_elite(BaseNPC):
    model_ = "models/xenians/agrunt.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_xen_grunt(BaseNPC):
    model_ = "models/xenians/agrunt.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.mainbehaviortreename = None  # Type: string
        self.assaultbehaviortreename = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.mainbehaviortreename = entity_data.get('mainbehaviortreename', None)  # Type: string
        instance.assaultbehaviortreename = entity_data.get('assaultbehaviortreename', None)  # Type: string


class npc_cockroach(BaseNPC):
    model_ = "models/fauna/roach.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_flyer_flock(BaseNPC):
    model_ = "models/xenians/flock.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_gargantua(BaseNPCAssault):
    model_ = "models/xenians/garg.mdl"
    def __init__(self):
        super(BaseNPCAssault).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPCAssault.from_dict(instance, entity_data)


class info_bigmomma(Node):
    model_ = "models/editor/ground_node.mdl"
    def __init__(self):
        super(Node).__init__()
        self.origin = [0, 0, 0]
        self.m_flRadius = 1  # Type: float
        self.m_flDelay = 1  # Type: float
        self.reachtarget = None  # Type: target_destination
        self.reachsequence = "0"  # Type: string
        self.presequence = "0"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Node.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.m_flRadius = float(entity_data.get('m_flradius', 1))  # Type: float
        instance.m_flDelay = float(entity_data.get('m_fldelay', 1))  # Type: float
        instance.reachtarget = entity_data.get('reachtarget', None)  # Type: target_destination
        instance.reachsequence = entity_data.get('reachsequence', "0")  # Type: string
        instance.presequence = entity_data.get('presequence', "0")  # Type: string


class npc_gonarch(Studiomodel, BaseNPC):
    model_ = "models/xenians/gonarch.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        super(Studiomodel).__init__()
        self.cavernbreed = "CHOICES NOT SUPPORTED"  # Type: choices
        self.shovetargets = None  # Type: string
        self.taunttargets = None  # Type: string
        self.covertargetsHI = None  # Type: string
        self.attacktargetsHI = None  # Type: string
        self.m_tGSState = "CHOICES NOT SUPPORTED"  # Type: choices
        self.bTouchKillActive = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Studiomodel.from_dict(instance, entity_data)
        BaseNPC.from_dict(instance, entity_data)
        instance.cavernbreed = entity_data.get('cavernbreed', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.shovetargets = entity_data.get('shovetargets', None)  # Type: string
        instance.taunttargets = entity_data.get('taunttargets', None)  # Type: string
        instance.covertargetsHI = entity_data.get('covertargetshi', None)  # Type: string
        instance.attacktargetsHI = entity_data.get('attacktargetshi', None)  # Type: string
        instance.m_tGSState = entity_data.get('m_tgsstate', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.bTouchKillActive = entity_data.get('btouchkillactive', "CHOICES NOT SUPPORTED")  # Type: choices


class env_gon_mortar_area(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.Radius = 100  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Radius = float(entity_data.get('radius', 100))  # Type: float


class npc_generic(BaseNPC):
    model_ = "models/gman.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_gman(BaseNPC):
    model_ = "models/gman.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_ichthyosaur(BaseNPC):
    model_ = "models/ichthyosaur.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_maintenance(BaseNPC):
    model_ = "models/humans/maintenance/maintenance_1.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_nihilanth(BaseNPC):
    model_ = "models/xenians/nihilanth.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.m_tNHState = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.m_tNHState = parse_source_value(entity_data.get('m_tnhstate', 0))  # Type: integer


class prop_nihi_shield(Angles, Studiomodel, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Studiomodel).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.m_fStartHP = 1500  # Type: float
        self.m_szGaurdedEntityName = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.m_fStartHP = float(entity_data.get('m_fstarthp', 1500))  # Type: float
        instance.m_szGaurdedEntityName = entity_data.get('m_szgaurdedentityname', None)  # Type: string


class nihilanth_pylon(prop_dynamic_base):
    viewport_model = "models/props_xen/nil_pylon.mdl"
    def __init__(self):
        super(prop_dynamic_base).__init__()
        self.origin = [0, 0, 0]
        self.bHealNihi = 1  # Type: integer
        self.bHealShield = 1  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.bHealNihi = parse_source_value(entity_data.get('bhealnihi', 1))  # Type: integer
        instance.bHealShield = parse_source_value(entity_data.get('bhealshield', 1))  # Type: integer


class BMBaseHelicopter(BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        self.InitialSpeed = "0"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.InitialSpeed = entity_data.get('initialspeed', "0")  # Type: string


class npc_manta(BMBaseHelicopter):
    model_ = "models/xenians/manta_jet.mdl"
    def __init__(self):
        super(BMBaseHelicopter).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BMBaseHelicopter.from_dict(instance, entity_data)


class prop_xen_grunt_pod(Global, BreakableProp, RenderFields, BaseFadeProp, Studiomodel, Parentname, Angles, DXLevelChoice):
    def __init__(self):
        super(BreakableProp).__init__()
        super(RenderFields).__init__()
        super(Global).__init__()
        super(BaseFadeProp).__init__()
        super(Studiomodel).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(DXLevelChoice).__init__()
        self.origin = [0, 0, 0]
        self.solid = "CHOICES NOT SUPPORTED"  # Type: choices
        self.MyCustomMass = None  # Type: float
        self.SetBodyGroup = None  # Type: integer
        self.lightingorigin = None  # Type: target_destination
        self.SpawnEntityName = "npc_alien_grunt_melee"  # Type: string
        self.EnablePodLight = "CHOICES NOT SUPPORTED"  # Type: choices
        self.EnablePodShadows = None  # Type: choices
        self.PodLightColor = [232, 251, 0]  # Type: color255
        self.ShouldKeepUpright = None  # Type: choices
        self.TargetEntityToIgnore01 = None  # Type: target_destination
        self.TargetEntityToIgnore02 = None  # Type: target_destination
        self.TargetEntityToIgnore03 = None  # Type: target_destination
        self.TargetEntityToIgnore04 = None  # Type: target_destination
        self.TargetEntityToIgnore05 = None  # Type: target_destination
        self.TargetEntityToIgnore06 = None  # Type: target_destination
        self.TargetEntityToIgnore07 = None  # Type: target_destination
        self.TargetEntityToIgnore08 = None  # Type: target_destination
        self.TargetEntityToIgnore09 = None  # Type: target_destination
        self.TargetEntityToIgnore10 = None  # Type: target_destination
        self.TargetEntityToIgnore11 = None  # Type: target_destination
        self.TargetEntityToIgnore12 = None  # Type: target_destination
        self.TargetEntityToIgnore13 = None  # Type: target_destination
        self.TargetEntityToIgnore14 = None  # Type: target_destination
        self.TargetEntityToIgnore15 = None  # Type: target_destination
        self.TargetEntityToIgnore16 = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Global.from_dict(instance, entity_data)
        BreakableProp.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        DXLevelChoice.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.solid = entity_data.get('solid', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.MyCustomMass = float(entity_data.get('mycustommass', 0))  # Type: float
        instance.SetBodyGroup = parse_source_value(entity_data.get('setbodygroup', 0))  # Type: integer
        instance.lightingorigin = entity_data.get('lightingorigin', None)  # Type: target_destination
        instance.SpawnEntityName = entity_data.get('spawnentityname', "npc_alien_grunt_melee")  # Type: string
        instance.EnablePodLight = entity_data.get('enablepodlight', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.EnablePodShadows = entity_data.get('enablepodshadows', None)  # Type: choices
        instance.PodLightColor = parse_int_vector(entity_data.get('podlightcolor', "232 251 0"))  # Type: color255
        instance.ShouldKeepUpright = entity_data.get('shouldkeepupright', None)  # Type: choices
        instance.TargetEntityToIgnore01 = entity_data.get('targetentitytoignore01', None)  # Type: target_destination
        instance.TargetEntityToIgnore02 = entity_data.get('targetentitytoignore02', None)  # Type: target_destination
        instance.TargetEntityToIgnore03 = entity_data.get('targetentitytoignore03', None)  # Type: target_destination
        instance.TargetEntityToIgnore04 = entity_data.get('targetentitytoignore04', None)  # Type: target_destination
        instance.TargetEntityToIgnore05 = entity_data.get('targetentitytoignore05', None)  # Type: target_destination
        instance.TargetEntityToIgnore06 = entity_data.get('targetentitytoignore06', None)  # Type: target_destination
        instance.TargetEntityToIgnore07 = entity_data.get('targetentitytoignore07', None)  # Type: target_destination
        instance.TargetEntityToIgnore08 = entity_data.get('targetentitytoignore08', None)  # Type: target_destination
        instance.TargetEntityToIgnore09 = entity_data.get('targetentitytoignore09', None)  # Type: target_destination
        instance.TargetEntityToIgnore10 = entity_data.get('targetentitytoignore10', None)  # Type: target_destination
        instance.TargetEntityToIgnore11 = entity_data.get('targetentitytoignore11', None)  # Type: target_destination
        instance.TargetEntityToIgnore12 = entity_data.get('targetentitytoignore12', None)  # Type: target_destination
        instance.TargetEntityToIgnore13 = entity_data.get('targetentitytoignore13', None)  # Type: target_destination
        instance.TargetEntityToIgnore14 = entity_data.get('targetentitytoignore14', None)  # Type: target_destination
        instance.TargetEntityToIgnore15 = entity_data.get('targetentitytoignore15', None)  # Type: target_destination
        instance.TargetEntityToIgnore16 = entity_data.get('targetentitytoignore16', None)  # Type: target_destination


class prop_xen_grunt_pod_dynamic(Global, BreakableProp, RenderFields, BaseFadeProp, Studiomodel, Parentname, Angles, DXLevelChoice):
    def __init__(self):
        super(BreakableProp).__init__()
        super(RenderFields).__init__()
        super(Global).__init__()
        super(BaseFadeProp).__init__()
        super(Studiomodel).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(DXLevelChoice).__init__()
        self.origin = [0, 0, 0]
        self.solid = "CHOICES NOT SUPPORTED"  # Type: choices
        self.MyCustomMass = None  # Type: float
        self.SetBodyGroup = None  # Type: integer
        self.lightingorigin = None  # Type: target_destination
        self.SpawnEntityName = "npc_alien_grunt_melee"  # Type: string
        self.PodLightColor = [232, 251, 0]  # Type: color255
        self.EnablePodLight = "CHOICES NOT SUPPORTED"  # Type: choices
        self.EnablePodShadows = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Global.from_dict(instance, entity_data)
        BreakableProp.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        DXLevelChoice.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.solid = entity_data.get('solid', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.MyCustomMass = float(entity_data.get('mycustommass', 0))  # Type: float
        instance.SetBodyGroup = parse_source_value(entity_data.get('setbodygroup', 0))  # Type: integer
        instance.lightingorigin = entity_data.get('lightingorigin', None)  # Type: target_destination
        instance.SpawnEntityName = entity_data.get('spawnentityname', "npc_alien_grunt_melee")  # Type: string
        instance.PodLightColor = parse_int_vector(entity_data.get('podlightcolor', "232 251 0"))  # Type: color255
        instance.EnablePodLight = entity_data.get('enablepodlight', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.EnablePodShadows = entity_data.get('enablepodshadows', None)  # Type: choices


class prop_xen_int_barrel(Global, BreakableProp, RenderFields, BaseFadeProp, Studiomodel, Parentname, Angles, DXLevelChoice):
    def __init__(self):
        super(BreakableProp).__init__()
        super(RenderFields).__init__()
        super(Global).__init__()
        super(BaseFadeProp).__init__()
        super(Studiomodel).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(DXLevelChoice).__init__()
        self.origin = [0, 0, 0]
        self.solid = "CHOICES NOT SUPPORTED"  # Type: choices
        self.SetBodyGroup = None  # Type: integer
        self.lightingorigin = None  # Type: target_destination
        self.EnablePodLight = "CHOICES NOT SUPPORTED"  # Type: choices
        self.EnablePodShadows = None  # Type: choices
        self.PodLightColor = [232, 251, 0]  # Type: color255

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Global.from_dict(instance, entity_data)
        BreakableProp.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        DXLevelChoice.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.solid = entity_data.get('solid', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.SetBodyGroup = parse_source_value(entity_data.get('setbodygroup', 0))  # Type: integer
        instance.lightingorigin = entity_data.get('lightingorigin', None)  # Type: target_destination
        instance.EnablePodLight = entity_data.get('enablepodlight', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.EnablePodShadows = entity_data.get('enablepodshadows', None)  # Type: choices
        instance.PodLightColor = parse_int_vector(entity_data.get('podlightcolor', "232 251 0"))  # Type: color255


class prop_barrel_cactus(prop_dynamic_base):
    def __init__(self):
        super(prop_dynamic_base).__init__()
        self.origin = [0, 0, 0]
        self.cactustype = None  # Type: choices
        self.m_bOverrideModel = None  # Type: choices
        self.m_bDisableGibs = None  # Type: choices
        self.lightRadius = 400  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.cactustype = entity_data.get('cactustype', None)  # Type: choices
        instance.m_bOverrideModel = entity_data.get('m_boverridemodel', None)  # Type: choices
        instance.m_bDisableGibs = entity_data.get('m_bdisablegibs', None)  # Type: choices
        instance.lightRadius = float(entity_data.get('lightradius', 400))  # Type: float


class prop_barrel_cactus_semilarge(prop_dynamic_base):
    def __init__(self):
        super(prop_dynamic_base).__init__()
        self.origin = [0, 0, 0]
        self.cactustype = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_bOverrideModel = None  # Type: choices
        self.m_bDisableGibs = None  # Type: choices
        self.lightRadius = 400  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.cactustype = entity_data.get('cactustype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.m_bOverrideModel = entity_data.get('m_boverridemodel', None)  # Type: choices
        instance.m_bDisableGibs = entity_data.get('m_bdisablegibs', None)  # Type: choices
        instance.lightRadius = float(entity_data.get('lightradius', 400))  # Type: float


class prop_barrel_cactus_adolescent(prop_dynamic_base):
    def __init__(self):
        super(prop_dynamic_base).__init__()
        self.origin = [0, 0, 0]
        self.cactustype = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_bOverrideModel = None  # Type: choices
        self.m_bDisableGibs = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.cactustype = entity_data.get('cactustype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.m_bOverrideModel = entity_data.get('m_boverridemodel', None)  # Type: choices
        instance.m_bDisableGibs = entity_data.get('m_bdisablegibs', None)  # Type: choices


class prop_barrel_cactus_infant(prop_dynamic_base):
    def __init__(self):
        super(prop_dynamic_base).__init__()
        self.origin = [0, 0, 0]
        self.cactustype = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_bOverrideModel = None  # Type: choices
        self.m_bDisableGibs = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.cactustype = entity_data.get('cactustype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.m_bOverrideModel = entity_data.get('m_boverridemodel', None)  # Type: choices
        instance.m_bDisableGibs = entity_data.get('m_bdisablegibs', None)  # Type: choices


class prop_barrel_cactus_exploder(prop_dynamic_base):
    def __init__(self):
        super(prop_dynamic_base).__init__()
        self.origin = [0, 0, 0]
        self.cactustype = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_bOverrideModel = None  # Type: choices
        self.m_bDisableGibs = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.cactustype = entity_data.get('cactustype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.m_bOverrideModel = entity_data.get('m_boverridemodel', None)  # Type: choices
        instance.m_bDisableGibs = entity_data.get('m_bdisablegibs', None)  # Type: choices


class prop_barrel_interloper(prop_dynamic_base):
    def __init__(self):
        super(prop_dynamic_base).__init__()
        self.origin = [0, 0, 0]
        self.cactustype = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_bOverrideModel = None  # Type: choices
        self.m_bDisableGibs = None  # Type: choices
        self.lightRadius = 400  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.cactustype = entity_data.get('cactustype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.m_bOverrideModel = entity_data.get('m_boverridemodel', None)  # Type: choices
        instance.m_bDisableGibs = entity_data.get('m_bdisablegibs', None)  # Type: choices
        instance.lightRadius = float(entity_data.get('lightradius', 400))  # Type: float


class prop_barrel_interloper_small(prop_dynamic_base):
    def __init__(self):
        super(prop_dynamic_base).__init__()
        self.origin = [0, 0, 0]
        self.cactustype = "CHOICES NOT SUPPORTED"  # Type: choices
        self.m_bOverrideModel = None  # Type: choices
        self.m_bDisableGibs = None  # Type: choices
        self.lightRadius = 400  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.cactustype = entity_data.get('cactustype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.m_bOverrideModel = entity_data.get('m_boverridemodel', None)  # Type: choices
        instance.m_bDisableGibs = entity_data.get('m_bdisablegibs', None)  # Type: choices
        instance.lightRadius = float(entity_data.get('lightradius', 400))  # Type: float


class npc_apache(BMBaseHelicopter):
    model_ = "models/props_vehicles/apache.mdl"
    def __init__(self):
        super(BMBaseHelicopter).__init__()
        self.bNerfedFireCone = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BMBaseHelicopter.from_dict(instance, entity_data)
        instance.bNerfedFireCone = entity_data.get('bnerfedfirecone', None)  # Type: choices


class npc_osprey(BMBaseHelicopter):
    model_ = "models/props_vehicles/osprey.mdl"
    def __init__(self):
        super(BMBaseHelicopter).__init__()
        self.NPCTemplate1 = None  # Type: target_destination
        self.NPCTemplate2 = None  # Type: target_destination
        self.NPCTemplate3 = None  # Type: target_destination
        self.NPCTemplate4 = None  # Type: target_destination
        self.NPCTemplate5 = None  # Type: target_destination
        self.NPCTemplate6 = None  # Type: target_destination
        self.NPCTemplate7 = None  # Type: target_destination
        self.NPCTemplate8 = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BMBaseHelicopter.from_dict(instance, entity_data)
        instance.NPCTemplate1 = entity_data.get('npctemplate1', None)  # Type: target_destination
        instance.NPCTemplate2 = entity_data.get('npctemplate2', None)  # Type: target_destination
        instance.NPCTemplate3 = entity_data.get('npctemplate3', None)  # Type: target_destination
        instance.NPCTemplate4 = entity_data.get('npctemplate4', None)  # Type: target_destination
        instance.NPCTemplate5 = entity_data.get('npctemplate5', None)  # Type: target_destination
        instance.NPCTemplate6 = entity_data.get('npctemplate6', None)  # Type: target_destination
        instance.NPCTemplate7 = entity_data.get('npctemplate7', None)  # Type: target_destination
        instance.NPCTemplate8 = entity_data.get('npctemplate8', None)  # Type: target_destination


class npc_rat(BaseNPC):
    model_ = "models/fauna/rat.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class BaseColleague(BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        self.expressiontype = "CHOICES NOT SUPPORTED"  # Type: choices
        self.CanSpeakWhileScripting = "CHOICES NOT SUPPORTED"  # Type: choices
        self.AlwaysTransition = "CHOICES NOT SUPPORTED"  # Type: choices
        self.GameEndAlly = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.expressiontype = entity_data.get('expressiontype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.CanSpeakWhileScripting = entity_data.get('canspeakwhilescripting', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.AlwaysTransition = entity_data.get('alwaystransition', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.GameEndAlly = entity_data.get('gameendally', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_human_security(Parentname, TalkNPC, BaseColleague):
    model_ = "models/humans/guard.mdl"
    def __init__(self):
        super(TalkNPC).__init__()
        super(BaseColleague).__init__()
        super(Parentname).__init__()
        self.additionalequipment = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        TalkNPC.from_dict(instance, entity_data)
        BaseColleague.from_dict(instance, entity_data)
        instance.additionalequipment = entity_data.get('additionalequipment', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_human_scientist_kleiner(Parentname, TalkNPC, BaseColleague):
    model_ = "models/humans/scientist_kliener.mdl"
    def __init__(self):
        super(TalkNPC).__init__()
        super(BaseColleague).__init__()
        super(Parentname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        TalkNPC.from_dict(instance, entity_data)
        BaseColleague.from_dict(instance, entity_data)


class npc_human_scientist_eli(Parentname, TalkNPC, BaseColleague):
    model_ = "models/humans/scientist_eli.mdl"
    def __init__(self):
        super(TalkNPC).__init__()
        super(BaseColleague).__init__()
        super(Parentname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        TalkNPC.from_dict(instance, entity_data)
        BaseColleague.from_dict(instance, entity_data)


class npc_human_scientist(Parentname, TalkNPC, BaseColleague):
    model_ = "models/humans/scientist.mdl"
    def __init__(self):
        super(TalkNPC).__init__()
        super(BaseColleague).__init__()
        super(Parentname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        TalkNPC.from_dict(instance, entity_data)
        BaseColleague.from_dict(instance, entity_data)


class npc_human_scientist_female(Parentname, TalkNPC, BaseColleague):
    model_ = "models/humans/scientist_female.mdl"
    def __init__(self):
        super(TalkNPC).__init__()
        super(BaseColleague).__init__()
        super(Parentname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        TalkNPC.from_dict(instance, entity_data)
        BaseColleague.from_dict(instance, entity_data)


class npc_xentacle(BaseNPC):
    model_ = "models/xenians/xentacle.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.radius = 320  # Type: integer
        self.target01 = None  # Type: target_destination
        self.target02 = None  # Type: target_destination
        self.target03 = None  # Type: target_destination
        self.target04 = None  # Type: target_destination
        self.target05 = None  # Type: target_destination
        self.target06 = None  # Type: target_destination
        self.target07 = None  # Type: target_destination
        self.target08 = None  # Type: target_destination
        self.target09 = None  # Type: target_destination
        self.target10 = None  # Type: target_destination
        self.target11 = None  # Type: target_destination
        self.target12 = None  # Type: target_destination
        self.target13 = None  # Type: target_destination
        self.target14 = None  # Type: target_destination
        self.target15 = None  # Type: target_destination
        self.target16 = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.radius = parse_source_value(entity_data.get('radius', 320))  # Type: integer
        instance.target01 = entity_data.get('target01', None)  # Type: target_destination
        instance.target02 = entity_data.get('target02', None)  # Type: target_destination
        instance.target03 = entity_data.get('target03', None)  # Type: target_destination
        instance.target04 = entity_data.get('target04', None)  # Type: target_destination
        instance.target05 = entity_data.get('target05', None)  # Type: target_destination
        instance.target06 = entity_data.get('target06', None)  # Type: target_destination
        instance.target07 = entity_data.get('target07', None)  # Type: target_destination
        instance.target08 = entity_data.get('target08', None)  # Type: target_destination
        instance.target09 = entity_data.get('target09', None)  # Type: target_destination
        instance.target10 = entity_data.get('target10', None)  # Type: target_destination
        instance.target11 = entity_data.get('target11', None)  # Type: target_destination
        instance.target12 = entity_data.get('target12', None)  # Type: target_destination
        instance.target13 = entity_data.get('target13', None)  # Type: target_destination
        instance.target14 = entity_data.get('target14', None)  # Type: target_destination
        instance.target15 = entity_data.get('target15', None)  # Type: target_destination
        instance.target16 = entity_data.get('target16', None)  # Type: target_destination


class npc_tentacle(BaseNPC):
    model_ = "models/xenians/tentacle.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_snark(BaseNPC):
    model_ = "models/xenians/snark.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.m_fLifeTime = 14  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.m_fLifeTime = float(entity_data.get('m_flifetime', 14))  # Type: float


class npc_sniper(BaseNPC):
    model_ = "models/combine_soldier.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.radius = None  # Type: integer
        self.misses = None  # Type: integer
        self.beambrightness = 100  # Type: integer
        self.shootZombiesInChest = None  # Type: choices
        self.shielddistance = 64  # Type: float
        self.shieldradius = 48  # Type: float
        self.PaintInterval = 1  # Type: float
        self.PaintIntervalVariance = 0.75  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.radius = parse_source_value(entity_data.get('radius', 0))  # Type: integer
        instance.misses = parse_source_value(entity_data.get('misses', 0))  # Type: integer
        instance.beambrightness = parse_source_value(entity_data.get('beambrightness', 100))  # Type: integer
        instance.shootZombiesInChest = entity_data.get('shootzombiesinchest', None)  # Type: choices
        instance.shielddistance = float(entity_data.get('shielddistance', 64))  # Type: float
        instance.shieldradius = float(entity_data.get('shieldradius', 48))  # Type: float
        instance.PaintInterval = float(entity_data.get('paintinterval', 1))  # Type: float
        instance.PaintIntervalVariance = float(entity_data.get('paintintervalvariance', 0.75))  # Type: float


class info_target_helicoptercrash(Parentname, Targetname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_dlightmap_update(Parentname, Targetname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_timescale_controller(Parentname, Targetname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_stopallsounds(Parentname, Targetname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_player_deathmatch(Angles, PlayerClass, Targetname):
    model_ = "models/editor/playerstart.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(PlayerClass).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.itemstogive = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        PlayerClass.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.itemstogive = entity_data.get('itemstogive', None)  # Type: string


class info_player_marine(Angles, PlayerClass, Targetname):
    model_ = "models/Player/mp_marine.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(PlayerClass).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.itemstogive = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        PlayerClass.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.itemstogive = entity_data.get('itemstogive', None)  # Type: string


class info_player_scientist(Angles, PlayerClass, Targetname):
    model_ = "models/Player/mp_scientist_hev.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(PlayerClass).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.itemstogive = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        PlayerClass.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.itemstogive = entity_data.get('itemstogive', None)  # Type: string


class material_timer(Parentname, Targetname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.length = 30  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.length = float(entity_data.get('length', 30))  # Type: float


class xen_portal(Base):
    def __init__(self):
        super().__init__()
        self.origin = [0, 0, 0]
        self.size = "CHOICES NOT SUPPORTED"  # Type: choices
        self.sound = "XenPortal.Sound"  # Type: sound
        self.jump_distance = 0  # Type: float
        self.jump_hmaxspeed = 200  # Type: float
        self.min_delay = 0  # Type: float
        self.max_delay = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.size = entity_data.get('size', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.sound = entity_data.get('sound', "XenPortal.Sound")  # Type: sound
        instance.jump_distance = float(entity_data.get('jump_distance', 0))  # Type: float
        instance.jump_hmaxspeed = float(entity_data.get('jump_hmaxspeed', 200))  # Type: float
        instance.min_delay = float(entity_data.get('min_delay', 0))  # Type: float
        instance.max_delay = float(entity_data.get('max_delay', 0))  # Type: float


class env_introcredits(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.startactive = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.startactive = entity_data.get('startactive', "CHOICES NOT SUPPORTED")  # Type: choices


class env_particle_beam(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.particlebeam = "CHOICES NOT SUPPORTED"  # Type: choices
        self.target = None  # Type: target_destination
        self.damage = 1  # Type: float
        self.damagetick = 0.1  # Type: float
        self.burntrail = "effects/gluon_burn_trail.vmt"  # Type: material
        self.burntrail_life = 4  # Type: float
        self.burntrail_size = 16  # Type: float
        self.burntrail_text = 0.01  # Type: float
        self.burntrail_flags = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.particlebeam = entity_data.get('particlebeam', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.damage = float(entity_data.get('damage', 1))  # Type: float
        instance.damagetick = float(entity_data.get('damagetick', 0.1))  # Type: float
        instance.burntrail = entity_data.get('burntrail', "effects/gluon_burn_trail.vmt")  # Type: material
        instance.burntrail_life = float(entity_data.get('burntrail_life', 4))  # Type: float
        instance.burntrail_size = float(entity_data.get('burntrail_size', 16))  # Type: float
        instance.burntrail_text = float(entity_data.get('burntrail_text', 0.01))  # Type: float
        instance.burntrail_flags = entity_data.get('burntrail_flags', "CHOICES NOT SUPPORTED")  # Type: choices


class env_particle_tesla(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.particletesla = "tesla_lc_core"  # Type: string
        self.frequency = 0.1  # Type: float
        self.mincount = 2  # Type: integer
        self.maxcount = 4  # Type: integer
        self.range = 2048  # Type: integer
        self.life = -1  # Type: float
        self.min = [-1.0, -1.0, -1.0]  # Type: vector
        self.max = [1.0, 1.0, 1.0]  # Type: vector
        self.decalname = "ZapScorch"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.particletesla = entity_data.get('particletesla', "tesla_lc_core")  # Type: string
        instance.frequency = float(entity_data.get('frequency', 0.1))  # Type: float
        instance.mincount = parse_source_value(entity_data.get('mincount', 2))  # Type: integer
        instance.maxcount = parse_source_value(entity_data.get('maxcount', 4))  # Type: integer
        instance.range = parse_source_value(entity_data.get('range', 2048))  # Type: integer
        instance.life = float(entity_data.get('life', -1))  # Type: float
        instance.min = parse_float_vector(entity_data.get('min', "-1 -1 -1"))  # Type: vector
        instance.max = parse_float_vector(entity_data.get('max', "1 1 1"))  # Type: vector
        instance.decalname = entity_data.get('decalname', "ZapScorch")  # Type: string


class env_xen_portal(xen_portal, npc_maker):
    icon_sprite = "Editor/Xen_Portal"
    def __init__(self):
        super(npc_maker).__init__()
        super(xen_portal).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        xen_portal.from_dict(instance, entity_data)
        npc_maker.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_xen_portal_template(npc_template_maker, xen_portal):
    icon_sprite = "Editor/Xen_Portal"
    def __init__(self):
        super(npc_template_maker).__init__()
        super(xen_portal).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        npc_template_maker.from_dict(instance, entity_data)
        xen_portal.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_pinch(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.timer = 1.8  # Type: float
        self.startsize = 10  # Type: float
        self.endsize = 30  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.timer = float(entity_data.get('timer', 1.8))  # Type: float
        instance.startsize = float(entity_data.get('startsize', 10))  # Type: float
        instance.endsize = float(entity_data.get('endsize', 30))  # Type: float


class env_dispenser(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.spawnmodel = "models/props_junk/popcan01a.mdl"  # Type: studio
        self.spawnangles = None  # Failed to parse value type due to could not convert string to float: 'Orientation'  # Type: angle
        self.capacity = 15  # Type: integer
        self.skinmin = None  # Type: integer
        self.skinmax = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.spawnmodel = entity_data.get('spawnmodel', "models/props_junk/popcan01a.mdl")  # Type: studio
        instance.spawnangles = parse_float_vector(entity_data.get('spawnangles', "Orientation of the model at spawn (Y Z X)"))  # Type: angle
        instance.capacity = parse_source_value(entity_data.get('capacity', 15))  # Type: integer
        instance.skinmin = parse_source_value(entity_data.get('skinmin', 0))  # Type: integer
        instance.skinmax = parse_source_value(entity_data.get('skinmax', 0))  # Type: integer


class item_crate(BasePropPhysics):
    def __init__(self):
        super(BasePropPhysics).__init__()
        super(BaseFadeProp).__init__()
        super(DamageFilter).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.scriptpreset = None  # Type: choices
        self.spawnonbreak = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFadeProp.from_dict(instance, entity_data)
        BasePropPhysics.from_dict(instance, entity_data)
        DamageFilter.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.scriptpreset = entity_data.get('scriptpreset', None)  # Type: choices
        instance.spawnonbreak = entity_data.get('spawnonbreak', None)  # Type: string


class func_50cal(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class func_tow(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class func_tow_mp(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class func_conveyor_bms(Parentname, Shadow, Targetname, RenderFields):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        self.direction = [0.0, 0.0, 0.0]  # Type: angle
        self.speed = "150"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance.direction = parse_float_vector(entity_data.get('direction', "0 0 0"))  # Type: angle
        instance.speed = entity_data.get('speed', "150")  # Type: string


class item_tow_missile(BasePropPhysics):
    model_ = "models/props_marines/tow_missile_projectile.mdl"
    def __init__(self):
        super(BasePropPhysics).__init__()
        super(BaseFadeProp).__init__()
        super(DamageFilter).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFadeProp.from_dict(instance, entity_data)
        BasePropPhysics.from_dict(instance, entity_data)
        DamageFilter.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_mortar_launcher(Targetname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.firedelay = 1  # Type: integer
        self.rateoffire = 10  # Type: float
        self.radius = 128  # Type: float
        self.target = None  # Type: target_destination
        self.grenadeentityname = "CHOICES NOT SUPPORTED"  # Type: choices
        self.apexheightratio = 1  # Type: float
        self.pathoption = "CHOICES NOT SUPPORTED"  # Type: choices
        self.fireshellscount = 1  # Type: integer
        self.override_damage = -1  # Type: float
        self.override_damageradius = -1  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.firedelay = parse_source_value(entity_data.get('firedelay', 1))  # Type: integer
        instance.rateoffire = float(entity_data.get('rateoffire', 10))  # Type: float
        instance.radius = float(entity_data.get('radius', 128))  # Type: float
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.grenadeentityname = entity_data.get('grenadeentityname', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.apexheightratio = float(entity_data.get('apexheightratio', 1))  # Type: float
        instance.pathoption = entity_data.get('pathoption', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.fireshellscount = parse_source_value(entity_data.get('fireshellscount', 1))  # Type: integer
        instance.override_damage = float(entity_data.get('override_damage', -1))  # Type: float
        instance.override_damageradius = float(entity_data.get('override_damageradius', -1))  # Type: float


class env_mortar_controller(Angles, Targetname):
    model_ = "models/props_st/airstrike_map.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.MortarLauncher = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.MortarLauncher = entity_data.get('mortarlauncher', None)  # Type: target_destination


class npc_abrams(BaseNPC):
    model_ = "models/props_vehicles/abrams.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.enableminiguns = "CHOICES NOT SUPPORTED"  # Type: choices
        self.enablebodyrotation = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.enableminiguns = entity_data.get('enableminiguns', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.enablebodyrotation = entity_data.get('enablebodyrotation', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_lav(BaseNPC):
    model_ = "models/props_vehicles/lav.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class env_tram_screen(Angles, Targetname, Origin):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        super(Origin).__init__()
        self.origin = [0, 0, 0]
        self.panelname = None  # Type: string
        self.functrainname = None  # Type: target_destination
        self.propname = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.panelname = entity_data.get('panelname', None)  # Type: string
        instance.functrainname = entity_data.get('functrainname', None)  # Type: target_destination
        instance.propname = entity_data.get('propname', None)  # Type: target_destination


class prop_retinalscanner(prop_dynamic_base):
    def __init__(self):
        super(prop_dynamic_base).__init__()
        self.origin = [0, 0, 0]
        self.locked = None  # Type: choices
        self.nextlockeduse = 4  # Type: float
        self.nextunlockeduse = 4  # Type: float
        self.lockedsound = None  # Type: sound
        self.unlockedsound = None  # Type: sound
        self.lockedusesound = None  # Type: sound
        self.unlockedusesound = None  # Type: sound
        self.lockedusevox = None  # Type: sound
        self.unlockedusevox = None  # Type: sound
        self.delaylockedvox = None  # Type: float
        self.delayunlockedvox = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.locked = entity_data.get('locked', None)  # Type: choices
        instance.nextlockeduse = float(entity_data.get('nextlockeduse', 4))  # Type: float
        instance.nextunlockeduse = float(entity_data.get('nextunlockeduse', 4))  # Type: float
        instance.lockedsound = entity_data.get('lockedsound', None)  # Type: sound
        instance.unlockedsound = entity_data.get('unlockedsound', None)  # Type: sound
        instance.lockedusesound = entity_data.get('lockedusesound', None)  # Type: sound
        instance.unlockedusesound = entity_data.get('unlockedusesound', None)  # Type: sound
        instance.lockedusevox = entity_data.get('lockedusevox', None)  # Type: sound
        instance.unlockedusevox = entity_data.get('unlockedusevox', None)  # Type: sound
        instance.delaylockedvox = float(entity_data.get('delaylockedvox', 0))  # Type: float
        instance.delayunlockedvox = float(entity_data.get('delayunlockedvox', 0))  # Type: float


class prop_physics_respawnable(prop_physics):
    def __init__(self):
        super(prop_physics).__init__()
        self.origin = [0, 0, 0]
        self.RespawnTime = 60  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_physics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.RespawnTime = float(entity_data.get('respawntime', 60))  # Type: float


class prop_scalable(EnableDisable, prop_dynamic_base):
    def __init__(self):
        super(prop_dynamic_base).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class logic_parent(Targetname):
    icon_sprite = "editor/logic_auto.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_xen_pushpad(Targetname):
    model_ = "models/xenians/jump_pad.mdl"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.nextjumpdelta = 0.5  # Type: float
        self.target = None  # Type: target_destination
        self.height = 512  # Type: float
        self.disableshadows = None  # Type: choices
        self.m_bMuteME = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.nextjumpdelta = float(entity_data.get('nextjumpdelta', 0.5))  # Type: float
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.height = float(entity_data.get('height', 512))  # Type: float
        instance.disableshadows = entity_data.get('disableshadows', None)  # Type: choices
        instance.m_bMuteME = entity_data.get('m_bmuteme', None)  # Type: choices


class trigger_gargantua_shake(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)


class trigger_lift(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        super(Targetname).__init__()
        self.liftaccel = 100  # Type: float
        self.clampspeed = 512  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.liftaccel = float(entity_data.get('liftaccel', 100))  # Type: float
        instance.clampspeed = float(entity_data.get('clampspeed', 512))  # Type: float


class trigger_weaponfire(trigger_multiple):
    def __init__(self):
        super(trigger_multiple).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        trigger_multiple.from_dict(instance, entity_data)


class func_minefield(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        super(Targetname).__init__()
        self.minecount = 25  # Type: integer
        self.ranx = 35  # Type: float
        self.rany = 35  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.minecount = parse_source_value(entity_data.get('minecount', 25))  # Type: integer
        instance.ranx = float(entity_data.get('ranx', 35))  # Type: float
        instance.rany = float(entity_data.get('rany', 35))  # Type: float


class func_friction(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        self.modifier = 100  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.modifier = float(entity_data.get('modifier', 100))  # Type: float


class prop_train_awesome(Shadow, Global, RenderFields, BaseFadeProp, Parentname, Angles, Targetname, DXLevelChoice):
    model_ = "models/props_vehicles/oar_awesome_tram.mdl"
    def __init__(self):
        super(RenderFields).__init__()
        super(Shadow).__init__()
        super(Global).__init__()
        super(BaseFadeProp).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        super(DXLevelChoice).__init__()
        self.origin = [0, 0, 0]
        self.solid = "CHOICES NOT SUPPORTED"  # Type: choices
        self.bTrainDisabled = None  # Type: choices
        self.lightingorigin = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Shadow.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        DXLevelChoice.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.solid = entity_data.get('solid', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.bTrainDisabled = entity_data.get('btraindisabled', None)  # Type: choices
        instance.lightingorigin = entity_data.get('lightingorigin', None)  # Type: target_destination


class prop_train_apprehension(Shadow, Global, RenderFields, BaseFadeProp, Parentname, Angles, Targetname, DXLevelChoice):
    model_ = "models/props_vehicles/oar_tram.mdl"
    def __init__(self):
        super(RenderFields).__init__()
        super(Shadow).__init__()
        super(Global).__init__()
        super(BaseFadeProp).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        super(DXLevelChoice).__init__()
        self.origin = [0, 0, 0]
        self.solid = "CHOICES NOT SUPPORTED"  # Type: choices
        self.bTrainDisabled = None  # Type: choices
        self.lightingorigin = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Shadow.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        DXLevelChoice.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.solid = entity_data.get('solid', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.bTrainDisabled = entity_data.get('btraindisabled', None)  # Type: choices
        instance.lightingorigin = entity_data.get('lightingorigin', None)  # Type: target_destination


class BaseZombie(BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_zombie_scientist(BaseZombie):
    model_ = "models/zombies/zombie_sci.mdl"
    def __init__(self):
        super(BaseZombie).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseZombie.from_dict(instance, entity_data)


class npc_zombie_scientist_torso(BaseZombie):
    model_ = "models/zombies/zombie_sci_torso.mdl"
    def __init__(self):
        super(BaseZombie).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseZombie.from_dict(instance, entity_data)


class npc_zombie_security(BaseZombie):
    model_ = "models/zombies/zombie_guard.mdl"
    def __init__(self):
        super(BaseZombie).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseZombie.from_dict(instance, entity_data)


class npc_zombie_grunt(BaseZombie):
    model_ = "models/zombies/zombie_grunt.mdl"
    def __init__(self):
        super(BaseZombie).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseZombie.from_dict(instance, entity_data)


class npc_zombie_grunt_torso(BaseZombie):
    model_ = "models/zombies/zombie_grunt_torso.mdl"
    def __init__(self):
        super(BaseZombie).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseZombie.from_dict(instance, entity_data)


class npc_zombie_hev(BaseZombie):
    model_ = "models/zombies/zombie_hev.mdl"
    def __init__(self):
        super(BaseZombie).__init__()
        self.flashlight_status = "CHOICES NOT SUPPORTED"  # Type: choices
        self.FlashLight_Shadows = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseZombie.from_dict(instance, entity_data)
        instance.flashlight_status = entity_data.get('flashlight_status', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.FlashLight_Shadows = entity_data.get('flashlight_shadows', None)  # Type: choices


class filter_damage_class(BaseFilter):
    def __init__(self):
        super(BaseFilter).__init__()
        self.filterclass = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.filterclass = entity_data.get('filterclass', None)  # Type: string


class filter_activator_flag(BaseFilter):
    def __init__(self):
        super(BaseFilter).__init__()
        self.flag = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.flag = entity_data.get('flag', None)  # Type: choices


class filter_activator_team(BaseFilter):
    def __init__(self):
        super(BaseFilter).__init__()
        self.filterteam = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.filterteam = entity_data.get('filterteam', "CHOICES NOT SUPPORTED")  # Type: choices


class prop_flare(BasePropPhysics):
    def __init__(self):
        super(BasePropPhysics).__init__()
        super(BaseFadeProp).__init__()
        super(DamageFilter).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFadeProp.from_dict(instance, entity_data)
        BasePropPhysics.from_dict(instance, entity_data)
        DamageFilter.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class prop_surgerybot(Angles, Targetname):
    viewport_model = "models/props_questionableethics/qe_surgery_bot_main.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.startactive = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.startactive = entity_data.get('startactive', "CHOICES NOT SUPPORTED")  # Type: choices


class env_xen_healpool(Angles, Studiomodel, Targetname):
    model_ = "models/props_Xen/xen_healingpool_full.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Studiomodel).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.healRate = 5  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.healRate = float(entity_data.get('healrate', 5))  # Type: float


class env_xen_healshower(env_xen_healpool):
    def __init__(self):
        super(env_xen_healpool).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        env_xen_healpool.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class prop_web_burnable(Angles, Studiomodel, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Studiomodel).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.m_fBurnTime = 2  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.m_fBurnTime = float(entity_data.get('m_fburntime', 2))  # Type: float


class prop_charger_base(Angles, Studiomodel, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Studiomodel).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.radius = 128  # Type: float
        self.chargerate = 1  # Type: float
        self.chargeamount = 10  # Type: float
        self.warmuptime = 5  # Type: float
        self.cooldowntime = 5  # Type: float
        self.warmlightcolor = [245, 154, 52]  # Type: color255
        self.coollightcolor = [128, 255, 255]  # Type: color255
        self.lightpos = [0.0, 0.0, 0.0]  # Type: vector
        self.lightintensity = 16000  # Type: float
        self.lightrange = 512  # Type: float
        self.bPlayIdleSounds = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.radius = float(entity_data.get('radius', 128))  # Type: float
        instance.chargerate = float(entity_data.get('chargerate', 1))  # Type: float
        instance.chargeamount = float(entity_data.get('chargeamount', 10))  # Type: float
        instance.warmuptime = float(entity_data.get('warmuptime', 5))  # Type: float
        instance.cooldowntime = float(entity_data.get('cooldowntime', 5))  # Type: float
        instance.warmlightcolor = parse_int_vector(entity_data.get('warmlightcolor', "245 154 52"))  # Type: color255
        instance.coollightcolor = parse_int_vector(entity_data.get('coollightcolor', "128 255 255"))  # Type: color255
        instance.lightpos = parse_float_vector(entity_data.get('lightpos', "0 0 0"))  # Type: vector
        instance.lightintensity = float(entity_data.get('lightintensity', 16000))  # Type: float
        instance.lightrange = float(entity_data.get('lightrange', 512))  # Type: float
        instance.bPlayIdleSounds = entity_data.get('bplayidlesounds', "CHOICES NOT SUPPORTED")  # Type: choices


class prop_hev_charger(Parentname, prop_charger_base):
    def __init__(self):
        super(prop_charger_base).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        prop_charger_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class prop_radiation_charger(Parentname, prop_charger_base):
    def __init__(self):
        super(prop_charger_base).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        prop_charger_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class camera_satellite(Parentname, Angles):
    viewport_model = "models/editor/camera.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.rendertarget = "CHOICES NOT SUPPORTED"  # Type: choices
        self.targetname = None  # Type: target_source
        self.FOV = 90  # Type: float
        self.UseScreenAspectRatio = None  # Type: choices
        self.fogEnable = None  # Type: choices
        self.fogColor = [0, 0, 0]  # Type: color255
        self.fogStart = 2048  # Type: float
        self.fogEnd = 4096  # Type: float
        self.fogMaxDensity = 1  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.rendertarget = entity_data.get('rendertarget', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.targetname = entity_data.get('targetname', None)  # Type: target_source
        instance.FOV = float(entity_data.get('fov', 90))  # Type: float
        instance.UseScreenAspectRatio = entity_data.get('usescreenaspectratio', None)  # Type: choices
        instance.fogEnable = entity_data.get('fogenable', None)  # Type: choices
        instance.fogColor = parse_int_vector(entity_data.get('fogcolor', "0 0 0"))  # Type: color255
        instance.fogStart = float(entity_data.get('fogstart', 2048))  # Type: float
        instance.fogEnd = float(entity_data.get('fogend', 4096))  # Type: float
        instance.fogMaxDensity = float(entity_data.get('fogmaxdensity', 1))  # Type: float


class logic_achievement(EnableDisable, Targetname):
    def __init__(self):
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.AchievementEvent = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.AchievementEvent = entity_data.get('achievementevent', None)  # Type: choices


class ai_goal_throw_prop(Targetname):
    icon_sprite = "editor/ai_goal_standoff.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.actor = None  # Type: target_name_or_class
        self.SearchType = None  # Type: choices
        self.StartActive = None  # Type: choices
        self.PropName = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.actor = entity_data.get('actor', None)  # Type: target_name_or_class
        instance.SearchType = entity_data.get('searchtype', None)  # Type: choices
        instance.StartActive = entity_data.get('startactive', None)  # Type: choices
        instance.PropName = entity_data.get('propname', None)  # Type: string


class info_observer_menu(Angles):
    viewport_model = "models/editor/camera.mdl"
    def __init__(self):
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.observerid = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.observerid = parse_source_value(entity_data.get('observerid', 0))  # Type: integer


class game_round_win(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.team = None  # Type: choices
        self.force_map_reset = "CHOICES NOT SUPPORTED"  # Type: choices
        self.switch_teams = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.team = entity_data.get('team', None)  # Type: choices
        instance.force_map_reset = entity_data.get('force_map_reset', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.switch_teams = entity_data.get('switch_teams', None)  # Type: choices


class game_round_start(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class game_mp_gamerules(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class mp_round_time(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.WarmupTime = None  # Type: integer
        self.RoundTime = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.WarmupTime = parse_source_value(entity_data.get('warmuptime', 0))  # Type: integer
        instance.RoundTime = parse_source_value(entity_data.get('roundtime', 0))  # Type: integer


class env_gravity(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_godrays_controller(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.Density = 0.5  # Type: float
        self.Decay = 0.5  # Type: float
        self.Weight = 1.0  # Type: float
        self.Exposure = 0.20  # Type: float
        self.DensityUW = 0.5  # Type: float
        self.DecayUW = 0.5  # Type: float
        self.WeightUW = 1.0  # Type: float
        self.ExposureUW = 0.20  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Density = float(entity_data.get('density', 0.5))  # Type: float
        instance.Decay = float(entity_data.get('decay', 0.5))  # Type: float
        instance.Weight = float(entity_data.get('weight', 1.0))  # Type: float
        instance.Exposure = float(entity_data.get('exposure', 0.20))  # Type: float
        instance.DensityUW = float(entity_data.get('densityuw', 0.5))  # Type: float
        instance.DecayUW = float(entity_data.get('decayuw', 0.5))  # Type: float
        instance.WeightUW = float(entity_data.get('weightuw', 1.0))  # Type: float
        instance.ExposureUW = float(entity_data.get('exposureuw', 0.20))  # Type: float


class misc_dead_hev(BaseFadeProp, EnableDisable, Studiomodel, Angles, Targetname, DXLevelChoice):
    def __init__(self):
        super(BaseFadeProp).__init__()
        super(EnableDisable).__init__()
        super(Studiomodel).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        super(DXLevelChoice).__init__()
        self.origin = [0, 0, 0]
        self.spritecolor = [255, 0, 0, 200]  # Type: color255
        self.lightcolor = [255, 0, 0, 4]  # Type: color255
        self.lightradius = 64  # Type: integer
        self.attachmentname = "eyes"  # Type: string
        self.health = 100  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFadeProp.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        DXLevelChoice.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.spritecolor = parse_int_vector(entity_data.get('spritecolor', "255 0 0 200"))  # Type: color255
        instance.lightcolor = parse_int_vector(entity_data.get('lightcolor', "255 0 0 4"))  # Type: color255
        instance.lightradius = parse_source_value(entity_data.get('lightradius', 64))  # Type: integer
        instance.attachmentname = entity_data.get('attachmentname', "eyes")  # Type: string
        instance.health = parse_source_value(entity_data.get('health', 100))  # Type: integer


class env_lensflare(Parentname, Angles, Targetname):
    icon_sprite = "editor/lensflare.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.FlareFile = None  # Type: string
        self.FlareAttenuation = 0.0  # Type: float
        self.FlareType = None  # Type: choices
        self.FlareStyle = None  # Type: choices
        self.GlowProxySize = 2.0  # Type: float
        self.Flare01_texture = None  # Type: string
        self.Flare01_params = None  # Type: string
        self.Flare01_intensity = None  # Type: vector
        self.Flare01_sizes = None  # Type: vector
        self.Flare01_color = None  # Type: color255
        self.Flare02_texture = None  # Type: string
        self.Flare02_params = None  # Type: string
        self.Flare02_intensity = None  # Type: vector
        self.Flare02_sizes = None  # Type: vector
        self.Flare02_color = None  # Type: color255
        self.Flare03_texture = None  # Type: string
        self.Flare03_params = None  # Type: string
        self.Flare03_intensity = None  # Type: vector
        self.Flare03_sizes = None  # Type: vector
        self.Flare03_color = None  # Type: color255
        self.Flare04_texture = None  # Type: string
        self.Flare04_params = None  # Type: string
        self.Flare04_intensity = None  # Type: vector
        self.Flare04_sizes = None  # Type: vector
        self.Flare04_color = None  # Type: color255
        self.Flare05_texture = None  # Type: string
        self.Flare05_params = None  # Type: string
        self.Flare05_intensity = None  # Type: vector
        self.Flare05_sizes = None  # Type: vector
        self.Flare05_color = None  # Type: color255
        self.Flare06_texture = None  # Type: string
        self.Flare06_params = None  # Type: string
        self.Flare06_intensity = None  # Type: vector
        self.Flare06_sizes = None  # Type: vector
        self.Flare06_color = None  # Type: color255
        self.Flare07_texture = None  # Type: string
        self.Flare07_params = None  # Type: string
        self.Flare07_intensity = None  # Type: vector
        self.Flare07_sizes = None  # Type: vector
        self.Flare07_color = None  # Type: color255
        self.Flare08_texture = None  # Type: string
        self.Flare08_params = None  # Type: string
        self.Flare08_intensity = None  # Type: vector
        self.Flare08_sizes = None  # Type: vector
        self.Flare08_color = None  # Type: color255
        self.Flare09_texture = None  # Type: string
        self.Flare09_params = None  # Type: string
        self.Flare09_intensity = None  # Type: vector
        self.Flare09_sizes = None  # Type: vector
        self.Flare09_color = None  # Type: color255
        self.Flare10_texture = None  # Type: string
        self.Flare10_params = None  # Type: string
        self.Flare10_intensity = None  # Type: vector
        self.Flare10_sizes = None  # Type: vector
        self.Flare10_color = None  # Type: color255
        self.Flare11_texture = None  # Type: string
        self.Flare11_params = None  # Type: string
        self.Flare11_intensity = None  # Type: vector
        self.Flare11_sizes = None  # Type: vector
        self.Flare11_color = None  # Type: color255
        self.Flare12_texture = None  # Type: string
        self.Flare12_params = None  # Type: string
        self.Flare12_intensity = None  # Type: vector
        self.Flare12_sizes = None  # Type: vector
        self.Flare12_color = None  # Type: color255
        self.Flare13_texture = None  # Type: string
        self.Flare13_params = None  # Type: string
        self.Flare13_intensity = None  # Type: vector
        self.Flare13_sizes = None  # Type: vector
        self.Flare13_color = None  # Type: color255
        self.Flare14_texture = None  # Type: string
        self.Flare14_params = None  # Type: string
        self.Flare14_intensity = None  # Type: vector
        self.Flare14_sizes = None  # Type: vector
        self.Flare14_color = None  # Type: color255
        self.Flare15_texture = None  # Type: string
        self.Flare15_params = None  # Type: string
        self.Flare15_intensity = None  # Type: vector
        self.Flare15_sizes = None  # Type: vector
        self.Flare15_color = None  # Type: color255
        self.Flare16_texture = None  # Type: string
        self.Flare16_params = None  # Type: string
        self.Flare16_intensity = None  # Type: vector
        self.Flare16_sizes = None  # Type: vector
        self.Flare16_color = None  # Type: color255
        self.Flare17_texture = None  # Type: string
        self.Flare17_params = None  # Type: string
        self.Flare17_intensity = None  # Type: vector
        self.Flare17_sizes = None  # Type: vector
        self.Flare17_color = None  # Type: color255
        self.Flare18_texture = None  # Type: string
        self.Flare18_params = None  # Type: string
        self.Flare18_intensity = None  # Type: vector
        self.Flare18_sizes = None  # Type: vector
        self.Flare18_color = None  # Type: color255
        self.Flare19_texture = None  # Type: string
        self.Flare19_params = None  # Type: string
        self.Flare19_intensity = None  # Type: vector
        self.Flare19_sizes = None  # Type: vector
        self.Flare19_color = None  # Type: color255
        self.Flare20_texture = None  # Type: string
        self.Flare20_params = None  # Type: string
        self.Flare20_intensity = None  # Type: vector
        self.Flare20_sizes = None  # Type: vector
        self.Flare20_color = None  # Type: color255

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.FlareFile = entity_data.get('flarefile', None)  # Type: string
        instance.FlareAttenuation = float(entity_data.get('flareattenuation', 0.0))  # Type: float
        instance.FlareType = entity_data.get('flaretype', None)  # Type: choices
        instance.FlareStyle = entity_data.get('flarestyle', None)  # Type: choices
        instance.GlowProxySize = float(entity_data.get('glowproxysize', 2.0))  # Type: float
        instance.Flare01_texture = entity_data.get('flare01_texture', None)  # Type: string
        instance.Flare01_params = entity_data.get('flare01_params', None)  # Type: string
        instance.Flare01_intensity = parse_float_vector(entity_data.get('flare01_intensity', "0 0 0"))  # Type: vector
        instance.Flare01_sizes = parse_float_vector(entity_data.get('flare01_sizes', "0 0 0"))  # Type: vector
        instance.Flare01_color = parse_int_vector(entity_data.get('flare01_color', "0 0 0"))  # Type: color255
        instance.Flare02_texture = entity_data.get('flare02_texture', None)  # Type: string
        instance.Flare02_params = entity_data.get('flare02_params', None)  # Type: string
        instance.Flare02_intensity = parse_float_vector(entity_data.get('flare02_intensity', "0 0 0"))  # Type: vector
        instance.Flare02_sizes = parse_float_vector(entity_data.get('flare02_sizes', "0 0 0"))  # Type: vector
        instance.Flare02_color = parse_int_vector(entity_data.get('flare02_color', "0 0 0"))  # Type: color255
        instance.Flare03_texture = entity_data.get('flare03_texture', None)  # Type: string
        instance.Flare03_params = entity_data.get('flare03_params', None)  # Type: string
        instance.Flare03_intensity = parse_float_vector(entity_data.get('flare03_intensity', "0 0 0"))  # Type: vector
        instance.Flare03_sizes = parse_float_vector(entity_data.get('flare03_sizes', "0 0 0"))  # Type: vector
        instance.Flare03_color = parse_int_vector(entity_data.get('flare03_color', "0 0 0"))  # Type: color255
        instance.Flare04_texture = entity_data.get('flare04_texture', None)  # Type: string
        instance.Flare04_params = entity_data.get('flare04_params', None)  # Type: string
        instance.Flare04_intensity = parse_float_vector(entity_data.get('flare04_intensity', "0 0 0"))  # Type: vector
        instance.Flare04_sizes = parse_float_vector(entity_data.get('flare04_sizes', "0 0 0"))  # Type: vector
        instance.Flare04_color = parse_int_vector(entity_data.get('flare04_color', "0 0 0"))  # Type: color255
        instance.Flare05_texture = entity_data.get('flare05_texture', None)  # Type: string
        instance.Flare05_params = entity_data.get('flare05_params', None)  # Type: string
        instance.Flare05_intensity = parse_float_vector(entity_data.get('flare05_intensity', "0 0 0"))  # Type: vector
        instance.Flare05_sizes = parse_float_vector(entity_data.get('flare05_sizes', "0 0 0"))  # Type: vector
        instance.Flare05_color = parse_int_vector(entity_data.get('flare05_color', "0 0 0"))  # Type: color255
        instance.Flare06_texture = entity_data.get('flare06_texture', None)  # Type: string
        instance.Flare06_params = entity_data.get('flare06_params', None)  # Type: string
        instance.Flare06_intensity = parse_float_vector(entity_data.get('flare06_intensity', "0 0 0"))  # Type: vector
        instance.Flare06_sizes = parse_float_vector(entity_data.get('flare06_sizes', "0 0 0"))  # Type: vector
        instance.Flare06_color = parse_int_vector(entity_data.get('flare06_color', "0 0 0"))  # Type: color255
        instance.Flare07_texture = entity_data.get('flare07_texture', None)  # Type: string
        instance.Flare07_params = entity_data.get('flare07_params', None)  # Type: string
        instance.Flare07_intensity = parse_float_vector(entity_data.get('flare07_intensity', "0 0 0"))  # Type: vector
        instance.Flare07_sizes = parse_float_vector(entity_data.get('flare07_sizes', "0 0 0"))  # Type: vector
        instance.Flare07_color = parse_int_vector(entity_data.get('flare07_color', "0 0 0"))  # Type: color255
        instance.Flare08_texture = entity_data.get('flare08_texture', None)  # Type: string
        instance.Flare08_params = entity_data.get('flare08_params', None)  # Type: string
        instance.Flare08_intensity = parse_float_vector(entity_data.get('flare08_intensity', "0 0 0"))  # Type: vector
        instance.Flare08_sizes = parse_float_vector(entity_data.get('flare08_sizes', "0 0 0"))  # Type: vector
        instance.Flare08_color = parse_int_vector(entity_data.get('flare08_color', "0 0 0"))  # Type: color255
        instance.Flare09_texture = entity_data.get('flare09_texture', None)  # Type: string
        instance.Flare09_params = entity_data.get('flare09_params', None)  # Type: string
        instance.Flare09_intensity = parse_float_vector(entity_data.get('flare09_intensity', "0 0 0"))  # Type: vector
        instance.Flare09_sizes = parse_float_vector(entity_data.get('flare09_sizes', "0 0 0"))  # Type: vector
        instance.Flare09_color = parse_int_vector(entity_data.get('flare09_color', "0 0 0"))  # Type: color255
        instance.Flare10_texture = entity_data.get('flare10_texture', None)  # Type: string
        instance.Flare10_params = entity_data.get('flare10_params', None)  # Type: string
        instance.Flare10_intensity = parse_float_vector(entity_data.get('flare10_intensity', "0 0 0"))  # Type: vector
        instance.Flare10_sizes = parse_float_vector(entity_data.get('flare10_sizes', "0 0 0"))  # Type: vector
        instance.Flare10_color = parse_int_vector(entity_data.get('flare10_color', "0 0 0"))  # Type: color255
        instance.Flare11_texture = entity_data.get('flare11_texture', None)  # Type: string
        instance.Flare11_params = entity_data.get('flare11_params', None)  # Type: string
        instance.Flare11_intensity = parse_float_vector(entity_data.get('flare11_intensity', "0 0 0"))  # Type: vector
        instance.Flare11_sizes = parse_float_vector(entity_data.get('flare11_sizes', "0 0 0"))  # Type: vector
        instance.Flare11_color = parse_int_vector(entity_data.get('flare11_color', "0 0 0"))  # Type: color255
        instance.Flare12_texture = entity_data.get('flare12_texture', None)  # Type: string
        instance.Flare12_params = entity_data.get('flare12_params', None)  # Type: string
        instance.Flare12_intensity = parse_float_vector(entity_data.get('flare12_intensity', "0 0 0"))  # Type: vector
        instance.Flare12_sizes = parse_float_vector(entity_data.get('flare12_sizes', "0 0 0"))  # Type: vector
        instance.Flare12_color = parse_int_vector(entity_data.get('flare12_color', "0 0 0"))  # Type: color255
        instance.Flare13_texture = entity_data.get('flare13_texture', None)  # Type: string
        instance.Flare13_params = entity_data.get('flare13_params', None)  # Type: string
        instance.Flare13_intensity = parse_float_vector(entity_data.get('flare13_intensity', "0 0 0"))  # Type: vector
        instance.Flare13_sizes = parse_float_vector(entity_data.get('flare13_sizes', "0 0 0"))  # Type: vector
        instance.Flare13_color = parse_int_vector(entity_data.get('flare13_color', "0 0 0"))  # Type: color255
        instance.Flare14_texture = entity_data.get('flare14_texture', None)  # Type: string
        instance.Flare14_params = entity_data.get('flare14_params', None)  # Type: string
        instance.Flare14_intensity = parse_float_vector(entity_data.get('flare14_intensity', "0 0 0"))  # Type: vector
        instance.Flare14_sizes = parse_float_vector(entity_data.get('flare14_sizes', "0 0 0"))  # Type: vector
        instance.Flare14_color = parse_int_vector(entity_data.get('flare14_color', "0 0 0"))  # Type: color255
        instance.Flare15_texture = entity_data.get('flare15_texture', None)  # Type: string
        instance.Flare15_params = entity_data.get('flare15_params', None)  # Type: string
        instance.Flare15_intensity = parse_float_vector(entity_data.get('flare15_intensity', "0 0 0"))  # Type: vector
        instance.Flare15_sizes = parse_float_vector(entity_data.get('flare15_sizes', "0 0 0"))  # Type: vector
        instance.Flare15_color = parse_int_vector(entity_data.get('flare15_color', "0 0 0"))  # Type: color255
        instance.Flare16_texture = entity_data.get('flare16_texture', None)  # Type: string
        instance.Flare16_params = entity_data.get('flare16_params', None)  # Type: string
        instance.Flare16_intensity = parse_float_vector(entity_data.get('flare16_intensity', "0 0 0"))  # Type: vector
        instance.Flare16_sizes = parse_float_vector(entity_data.get('flare16_sizes', "0 0 0"))  # Type: vector
        instance.Flare16_color = parse_int_vector(entity_data.get('flare16_color', "0 0 0"))  # Type: color255
        instance.Flare17_texture = entity_data.get('flare17_texture', None)  # Type: string
        instance.Flare17_params = entity_data.get('flare17_params', None)  # Type: string
        instance.Flare17_intensity = parse_float_vector(entity_data.get('flare17_intensity', "0 0 0"))  # Type: vector
        instance.Flare17_sizes = parse_float_vector(entity_data.get('flare17_sizes', "0 0 0"))  # Type: vector
        instance.Flare17_color = parse_int_vector(entity_data.get('flare17_color', "0 0 0"))  # Type: color255
        instance.Flare18_texture = entity_data.get('flare18_texture', None)  # Type: string
        instance.Flare18_params = entity_data.get('flare18_params', None)  # Type: string
        instance.Flare18_intensity = parse_float_vector(entity_data.get('flare18_intensity', "0 0 0"))  # Type: vector
        instance.Flare18_sizes = parse_float_vector(entity_data.get('flare18_sizes', "0 0 0"))  # Type: vector
        instance.Flare18_color = parse_int_vector(entity_data.get('flare18_color', "0 0 0"))  # Type: color255
        instance.Flare19_texture = entity_data.get('flare19_texture', None)  # Type: string
        instance.Flare19_params = entity_data.get('flare19_params', None)  # Type: string
        instance.Flare19_intensity = parse_float_vector(entity_data.get('flare19_intensity', "0 0 0"))  # Type: vector
        instance.Flare19_sizes = parse_float_vector(entity_data.get('flare19_sizes', "0 0 0"))  # Type: vector
        instance.Flare19_color = parse_int_vector(entity_data.get('flare19_color', "0 0 0"))  # Type: color255
        instance.Flare20_texture = entity_data.get('flare20_texture', None)  # Type: string
        instance.Flare20_params = entity_data.get('flare20_params', None)  # Type: string
        instance.Flare20_intensity = parse_float_vector(entity_data.get('flare20_intensity', "0 0 0"))  # Type: vector
        instance.Flare20_sizes = parse_float_vector(entity_data.get('flare20_sizes', "0 0 0"))  # Type: vector
        instance.Flare20_color = parse_int_vector(entity_data.get('flare20_color', "0 0 0"))  # Type: color255


class env_fumer(Targetname, BaseFadeProp, EnableDisable, Studiomodel, Parentname, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(BaseFadeProp).__init__()
        super(EnableDisable).__init__()
        super(Studiomodel).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.DetectionRadius = 128  # Type: float
        self.ExplodeRaius = 128  # Type: float
        self.ExplodeDmg = 30  # Type: float
        self.ExplodeForce = 1  # Type: float
        self.FlameTime = 10  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.DetectionRadius = float(entity_data.get('detectionradius', 128))  # Type: float
        instance.ExplodeRaius = float(entity_data.get('exploderaius', 128))  # Type: float
        instance.ExplodeDmg = float(entity_data.get('explodedmg', 30))  # Type: float
        instance.ExplodeForce = float(entity_data.get('explodeforce', 1))  # Type: float
        instance.FlameTime = float(entity_data.get('flametime', 10))  # Type: float


class trigger_apply_impulse(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        self.impulse_dir = [0.0, 0.0, 0.0]  # Type: angle
        self.force = 300  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.impulse_dir = parse_float_vector(entity_data.get('impulse_dir', "0 0 0"))  # Type: angle
        instance.force = float(entity_data.get('force', 300))  # Type: float


class info_nihilanth_summon(Parentname, Angles, Targetname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class point_weaponstrip(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.Weapon = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Weapon = entity_data.get('weapon', "CHOICES NOT SUPPORTED")  # Type: choices


class misc_marionettist(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.innerdestinationradius = 32  # Type: integer
        self.innerpullspeed = 448  # Type: integer
        self.outerdestinationradius = 128  # Type: integer
        self.outerpullspeed = 512  # Type: integer
        self.ignorecollisions = None  # Type: choices
        self.target01 = None  # Type: target_destination
        self.target02 = None  # Type: target_destination
        self.target03 = None  # Type: target_destination
        self.target04 = None  # Type: target_destination
        self.target05 = None  # Type: target_destination
        self.target06 = None  # Type: target_destination
        self.target07 = None  # Type: target_destination
        self.target08 = None  # Type: target_destination
        self.target09 = None  # Type: target_destination
        self.target10 = None  # Type: target_destination
        self.target11 = None  # Type: target_destination
        self.target12 = None  # Type: target_destination
        self.target13 = None  # Type: target_destination
        self.target14 = None  # Type: target_destination
        self.target15 = None  # Type: target_destination
        self.target16 = None  # Type: target_destination
        self.soundscriptstart = None  # Type: string
        self.soundscriptloop = None  # Type: string
        self.soundscriptend = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.innerdestinationradius = parse_source_value(entity_data.get('innerdestinationradius', 32))  # Type: integer
        instance.innerpullspeed = parse_source_value(entity_data.get('innerpullspeed', 448))  # Type: integer
        instance.outerdestinationradius = parse_source_value(entity_data.get('outerdestinationradius', 128))  # Type: integer
        instance.outerpullspeed = parse_source_value(entity_data.get('outerpullspeed', 512))  # Type: integer
        instance.ignorecollisions = entity_data.get('ignorecollisions', None)  # Type: choices
        instance.target01 = entity_data.get('target01', None)  # Type: target_destination
        instance.target02 = entity_data.get('target02', None)  # Type: target_destination
        instance.target03 = entity_data.get('target03', None)  # Type: target_destination
        instance.target04 = entity_data.get('target04', None)  # Type: target_destination
        instance.target05 = entity_data.get('target05', None)  # Type: target_destination
        instance.target06 = entity_data.get('target06', None)  # Type: target_destination
        instance.target07 = entity_data.get('target07', None)  # Type: target_destination
        instance.target08 = entity_data.get('target08', None)  # Type: target_destination
        instance.target09 = entity_data.get('target09', None)  # Type: target_destination
        instance.target10 = entity_data.get('target10', None)  # Type: target_destination
        instance.target11 = entity_data.get('target11', None)  # Type: target_destination
        instance.target12 = entity_data.get('target12', None)  # Type: target_destination
        instance.target13 = entity_data.get('target13', None)  # Type: target_destination
        instance.target14 = entity_data.get('target14', None)  # Type: target_destination
        instance.target15 = entity_data.get('target15', None)  # Type: target_destination
        instance.target16 = entity_data.get('target16', None)  # Type: target_destination
        instance.soundscriptstart = entity_data.get('soundscriptstart', None)  # Type: string
        instance.soundscriptloop = entity_data.get('soundscriptloop', None)  # Type: string
        instance.soundscriptend = entity_data.get('soundscriptend', None)  # Type: string


class misc_xen_healing_pylon(Angles, Studiomodel, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Studiomodel).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.max_health = 200  # Type: integer
        self.can_be_damaged_only_healing = None  # Type: choices
        self.danger_recovering_duration = 5.0  # Type: float
        self.healing_request_duration = 5.0  # Type: float
        self.healing_request_hp_per_tick = 16  # Type: integer
        self.healing_request_tick_delta = 0.125  # Type: float
        self.healing_beam_attachment_name = None  # Type: string
        self.healing_beam_spread_radius = 16.0  # Type: float
        self.healing_beam_sprite_model = "sprites/rollermine_shock.vmt"  # Type: sprite
        self.healing_beam_noise_amplitude = 4.0  # Type: float
        self.healing_beam_starting_width = 8.0  # Type: float
        self.healing_beam_ending_width = 32.0  # Type: float
        self.healing_beam_color = [255, 255, 255, 255]  # Type: color255
        self.healing_beam_starting_pfx = "gloun_zap"  # Type: string
        self.healing_beam_ending_pfx = "gloun_zap"  # Type: string
        self.pylon_sequence_opening = "deploy"  # Type: string
        self.pylon_sequence_opened_idle = "idle_deploy"  # Type: string
        self.pylon_sequence_closing = "retract"  # Type: string
        self.pylon_sequence_closed_idle = "idle_retract"  # Type: string
        self.pylon_sequence_dying = "explode"  # Type: string
        self.pylon_sequence_died_idle = "idle_explode"  # Type: string
        self.trace_targetname_filter = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.max_health = parse_source_value(entity_data.get('max_health', 200))  # Type: integer
        instance.can_be_damaged_only_healing = entity_data.get('can_be_damaged_only_healing', None)  # Type: choices
        instance.danger_recovering_duration = float(entity_data.get('danger_recovering_duration', 5.0))  # Type: float
        instance.healing_request_duration = float(entity_data.get('healing_request_duration', 5.0))  # Type: float
        instance.healing_request_hp_per_tick = parse_source_value(entity_data.get('healing_request_hp_per_tick', 16))  # Type: integer
        instance.healing_request_tick_delta = float(entity_data.get('healing_request_tick_delta', 0.125))  # Type: float
        instance.healing_beam_attachment_name = entity_data.get('healing_beam_attachment_name', None)  # Type: string
        instance.healing_beam_spread_radius = float(entity_data.get('healing_beam_spread_radius', 16.0))  # Type: float
        instance.healing_beam_sprite_model = entity_data.get('healing_beam_sprite_model', "sprites/rollermine_shock.vmt")  # Type: sprite
        instance.healing_beam_noise_amplitude = float(entity_data.get('healing_beam_noise_amplitude', 4.0))  # Type: float
        instance.healing_beam_starting_width = float(entity_data.get('healing_beam_starting_width', 8.0))  # Type: float
        instance.healing_beam_ending_width = float(entity_data.get('healing_beam_ending_width', 32.0))  # Type: float
        instance.healing_beam_color = parse_int_vector(entity_data.get('healing_beam_color', "255 255 255 255"))  # Type: color255
        instance.healing_beam_starting_pfx = entity_data.get('healing_beam_starting_pfx', "gloun_zap")  # Type: string
        instance.healing_beam_ending_pfx = entity_data.get('healing_beam_ending_pfx', "gloun_zap")  # Type: string
        instance.pylon_sequence_opening = entity_data.get('pylon_sequence_opening', "deploy")  # Type: string
        instance.pylon_sequence_opened_idle = entity_data.get('pylon_sequence_opened_idle', "idle_deploy")  # Type: string
        instance.pylon_sequence_closing = entity_data.get('pylon_sequence_closing', "retract")  # Type: string
        instance.pylon_sequence_closed_idle = entity_data.get('pylon_sequence_closed_idle', "idle_retract")  # Type: string
        instance.pylon_sequence_dying = entity_data.get('pylon_sequence_dying', "explode")  # Type: string
        instance.pylon_sequence_died_idle = entity_data.get('pylon_sequence_died_idle', "idle_explode")  # Type: string
        instance.trace_targetname_filter = entity_data.get('trace_targetname_filter', None)  # Type: string


class misc_xen_shield(Angles, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.panel_modelname_template = "models/xenians/shield/pentagonal.hexecontahedron/nihilanth/panel.%03d.mdl"  # Type: string
        self.panels_amount = 60  # Type: integer
        self.max_health_for_panel = 75  # Type: integer
        self.max_health = 3000  # Type: integer
        self.healing_per_tick_for_panel = 2  # Type: integer
        self.healing_tick_delta_for_panel = 0.125  # Type: float
        self.healing_request_cooldown = 7.5  # Type: float
        self.hp_amount_to_request_heal = 0.85  # Type: float
        self.pylon01 = None  # Type: target_destination
        self.pylon02 = None  # Type: target_destination
        self.pylon03 = None  # Type: target_destination
        self.pylon04 = None  # Type: target_destination
        self.pylon05 = None  # Type: target_destination
        self.pylon06 = None  # Type: target_destination
        self.pylon07 = None  # Type: target_destination
        self.pylon08 = None  # Type: target_destination
        self.pylon09 = None  # Type: target_destination
        self.pylon10 = None  # Type: target_destination
        self.pylon11 = None  # Type: target_destination
        self.pylon12 = None  # Type: target_destination
        self.pylon13 = None  # Type: target_destination
        self.pylon14 = None  # Type: target_destination
        self.pylon15 = None  # Type: target_destination
        self.pylon16 = None  # Type: target_destination
        self.angular_velocity_value01 = [5.0, 30.0, 15.0]  # Type: angle
        self.angular_velocity_value02 = [-25.0, 45.0, -5.0]  # Type: angle
        self.angular_velocity_value03 = [5.0, 60.0, 15.0]  # Type: angle
        self.angular_velocity_value04 = [25.0, 45.0, 0.0]  # Type: angle
        self.angular_velocity_value05 = [-5.0, 15.0, -15.0]  # Type: angle
        self.angular_velocity_value06 = None  # Type: angle
        self.angular_velocity_value07 = None  # Type: angle
        self.angular_velocity_value08 = None  # Type: angle
        self.angular_velocity_value09 = None  # Type: angle
        self.angular_velocity_value10 = None  # Type: angle
        self.angular_velocity_value11 = None  # Type: angle
        self.angular_velocity_value12 = None  # Type: angle
        self.angular_velocity_value13 = None  # Type: angle
        self.angular_velocity_value14 = None  # Type: angle
        self.angular_velocity_value15 = None  # Type: angle
        self.angular_velocity_value16 = None  # Type: angle
        self.angular_velocity_values_used = 5  # Type: integer
        self.health_color01 = [1.0, 0.0, 0.0]  # Type: color1
        self.health_color02 = [1.0, 1.0, 0.0]  # Type: color1
        self.health_color03 = [0.0, 1.0, 0.0]  # Type: color1
        self.health_color04 = [0.0, 1.0, 1.0]  # Type: color1
        self.health_color05 = [0.0, 0.0, 1.0]  # Type: color1
        self.health_color06 = None  # Type: color1
        self.health_color07 = None  # Type: color1
        self.health_color08 = None  # Type: color1
        self.health_color09 = None  # Type: color1
        self.health_color10 = None  # Type: color1
        self.health_color11 = None  # Type: color1
        self.health_color12 = None  # Type: color1
        self.health_color13 = None  # Type: color1
        self.health_color14 = None  # Type: color1
        self.health_color15 = None  # Type: color1
        self.health_color16 = None  # Type: color1
        self.health_colors_used = 5  # Type: integer
        self.intro_for_panel_minimum = 2.5  # Type: float
        self.intro_for_panel_maximum = 5.0  # Type: float
        self.pause_for_panel_minimum = 0.5  # Type: float
        self.pause_for_panel_maximum = 2.5  # Type: float
        self.death_for_panel = 1.5  # Type: float
        self.per_panel_color_scheme = None  # Type: choices
        self.no_impact_on_alive_pylons = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.panel_modelname_template = entity_data.get('panel_modelname_template', "models/xenians/shield/pentagonal.hexecontahedron/nihilanth/panel.%03d.mdl")  # Type: string
        instance.panels_amount = parse_source_value(entity_data.get('panels_amount', 60))  # Type: integer
        instance.max_health_for_panel = parse_source_value(entity_data.get('max_health_for_panel', 75))  # Type: integer
        instance.max_health = parse_source_value(entity_data.get('max_health', 3000))  # Type: integer
        instance.healing_per_tick_for_panel = parse_source_value(entity_data.get('healing_per_tick_for_panel', 2))  # Type: integer
        instance.healing_tick_delta_for_panel = float(entity_data.get('healing_tick_delta_for_panel', 0.125))  # Type: float
        instance.healing_request_cooldown = float(entity_data.get('healing_request_cooldown', 7.5))  # Type: float
        instance.hp_amount_to_request_heal = float(entity_data.get('hp_amount_to_request_heal', 0.85))  # Type: float
        instance.pylon01 = entity_data.get('pylon01', None)  # Type: target_destination
        instance.pylon02 = entity_data.get('pylon02', None)  # Type: target_destination
        instance.pylon03 = entity_data.get('pylon03', None)  # Type: target_destination
        instance.pylon04 = entity_data.get('pylon04', None)  # Type: target_destination
        instance.pylon05 = entity_data.get('pylon05', None)  # Type: target_destination
        instance.pylon06 = entity_data.get('pylon06', None)  # Type: target_destination
        instance.pylon07 = entity_data.get('pylon07', None)  # Type: target_destination
        instance.pylon08 = entity_data.get('pylon08', None)  # Type: target_destination
        instance.pylon09 = entity_data.get('pylon09', None)  # Type: target_destination
        instance.pylon10 = entity_data.get('pylon10', None)  # Type: target_destination
        instance.pylon11 = entity_data.get('pylon11', None)  # Type: target_destination
        instance.pylon12 = entity_data.get('pylon12', None)  # Type: target_destination
        instance.pylon13 = entity_data.get('pylon13', None)  # Type: target_destination
        instance.pylon14 = entity_data.get('pylon14', None)  # Type: target_destination
        instance.pylon15 = entity_data.get('pylon15', None)  # Type: target_destination
        instance.pylon16 = entity_data.get('pylon16', None)  # Type: target_destination
        instance.angular_velocity_value01 = parse_float_vector(entity_data.get('angular_velocity_value01', "5.0 30.0 15.0"))  # Type: angle
        instance.angular_velocity_value02 = parse_float_vector(entity_data.get('angular_velocity_value02', "-25.0 45.0 -5.0"))  # Type: angle
        instance.angular_velocity_value03 = parse_float_vector(entity_data.get('angular_velocity_value03', "5.0 60.0 15.0"))  # Type: angle
        instance.angular_velocity_value04 = parse_float_vector(entity_data.get('angular_velocity_value04', "25.0 45.0 0.0"))  # Type: angle
        instance.angular_velocity_value05 = parse_float_vector(entity_data.get('angular_velocity_value05', "-5.0 15.0 -15.0"))  # Type: angle
        instance.angular_velocity_value06 = parse_float_vector(entity_data.get('angular_velocity_value06', "0 0 0"))  # Type: angle
        instance.angular_velocity_value07 = parse_float_vector(entity_data.get('angular_velocity_value07', "0 0 0"))  # Type: angle
        instance.angular_velocity_value08 = parse_float_vector(entity_data.get('angular_velocity_value08', "0 0 0"))  # Type: angle
        instance.angular_velocity_value09 = parse_float_vector(entity_data.get('angular_velocity_value09', "0 0 0"))  # Type: angle
        instance.angular_velocity_value10 = parse_float_vector(entity_data.get('angular_velocity_value10', "0 0 0"))  # Type: angle
        instance.angular_velocity_value11 = parse_float_vector(entity_data.get('angular_velocity_value11', "0 0 0"))  # Type: angle
        instance.angular_velocity_value12 = parse_float_vector(entity_data.get('angular_velocity_value12', "0 0 0"))  # Type: angle
        instance.angular_velocity_value13 = parse_float_vector(entity_data.get('angular_velocity_value13', "0 0 0"))  # Type: angle
        instance.angular_velocity_value14 = parse_float_vector(entity_data.get('angular_velocity_value14', "0 0 0"))  # Type: angle
        instance.angular_velocity_value15 = parse_float_vector(entity_data.get('angular_velocity_value15', "0 0 0"))  # Type: angle
        instance.angular_velocity_value16 = parse_float_vector(entity_data.get('angular_velocity_value16', "0 0 0"))  # Type: angle
        instance.angular_velocity_values_used = parse_source_value(entity_data.get('angular_velocity_values_used', 5))  # Type: integer
        instance.health_color01 = parse_float_vector(entity_data.get('health_color01', "1.0 0.0 0.0"))  # Type: color1
        instance.health_color02 = parse_float_vector(entity_data.get('health_color02', "1.0 1.0 0.0"))  # Type: color1
        instance.health_color03 = parse_float_vector(entity_data.get('health_color03', "0.0 1.0 0.0"))  # Type: color1
        instance.health_color04 = parse_float_vector(entity_data.get('health_color04', "0.0 1.0 1.0"))  # Type: color1
        instance.health_color05 = parse_float_vector(entity_data.get('health_color05', "0.0 0.0 1.0"))  # Type: color1
        instance.health_color06 = parse_float_vector(entity_data.get('health_color06', "0 0 0"))  # Type: color1
        instance.health_color07 = parse_float_vector(entity_data.get('health_color07', "0 0 0"))  # Type: color1
        instance.health_color08 = parse_float_vector(entity_data.get('health_color08', "0 0 0"))  # Type: color1
        instance.health_color09 = parse_float_vector(entity_data.get('health_color09', "0 0 0"))  # Type: color1
        instance.health_color10 = parse_float_vector(entity_data.get('health_color10', "0 0 0"))  # Type: color1
        instance.health_color11 = parse_float_vector(entity_data.get('health_color11', "0 0 0"))  # Type: color1
        instance.health_color12 = parse_float_vector(entity_data.get('health_color12', "0 0 0"))  # Type: color1
        instance.health_color13 = parse_float_vector(entity_data.get('health_color13', "0 0 0"))  # Type: color1
        instance.health_color14 = parse_float_vector(entity_data.get('health_color14', "0 0 0"))  # Type: color1
        instance.health_color15 = parse_float_vector(entity_data.get('health_color15', "0 0 0"))  # Type: color1
        instance.health_color16 = parse_float_vector(entity_data.get('health_color16', "0 0 0"))  # Type: color1
        instance.health_colors_used = parse_source_value(entity_data.get('health_colors_used', 5))  # Type: integer
        instance.intro_for_panel_minimum = float(entity_data.get('intro_for_panel_minimum', 2.5))  # Type: float
        instance.intro_for_panel_maximum = float(entity_data.get('intro_for_panel_maximum', 5.0))  # Type: float
        instance.pause_for_panel_minimum = float(entity_data.get('pause_for_panel_minimum', 0.5))  # Type: float
        instance.pause_for_panel_maximum = float(entity_data.get('pause_for_panel_maximum', 2.5))  # Type: float
        instance.death_for_panel = float(entity_data.get('death_for_panel', 1.5))  # Type: float
        instance.per_panel_color_scheme = entity_data.get('per_panel_color_scheme', None)  # Type: choices
        instance.no_impact_on_alive_pylons = entity_data.get('no_impact_on_alive_pylons', "CHOICES NOT SUPPORTED")  # Type: choices


class prop_physics_psychokinesis(BasePropPhysics):
    def __init__(self):
        super(BasePropPhysics).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePropPhysics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class nihiportalsbase(Angles, Studiomodel, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Studiomodel).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.m_fTimeToActivate = None  # Type: float
        self.m_fTimeToDie = 9000  # Type: float
        self.m_bManualAwake = None  # Type: choices
        self.m_bLightNeeded = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.m_fTimeToActivate = float(entity_data.get('m_ftimetoactivate', 0))  # Type: float
        instance.m_fTimeToDie = float(entity_data.get('m_ftimetodie', 9000))  # Type: float
        instance.m_bManualAwake = entity_data.get('m_bmanualawake', None)  # Type: choices
        instance.m_bLightNeeded = entity_data.get('m_blightneeded', None)  # Type: choices


class nihiportals_teleprops(Angles, Studiomodel, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Studiomodel).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.m_fTimeToActivate = None  # Type: float
        self.m_fTimeToDie = 9000  # Type: float
        self.m_bManualAwake = None  # Type: choices
        self.m_bLightNeeded = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.m_fTimeToActivate = float(entity_data.get('m_ftimetoactivate', 0))  # Type: float
        instance.m_fTimeToDie = float(entity_data.get('m_ftimetodie', 9000))  # Type: float
        instance.m_bManualAwake = entity_data.get('m_bmanualawake', None)  # Type: choices
        instance.m_bLightNeeded = entity_data.get('m_blightneeded', None)  # Type: choices


class music_track(Targetname):
    icon_sprite = "editor/ambient_generic.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.track_script_sound = None  # Type: sound
        self.next_track_entity = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.track_script_sound = entity_data.get('track_script_sound', None)  # Type: sound
        instance.next_track_entity = entity_data.get('next_track_entity', None)  # Type: string



entity_class_handle = {
    'Empty': Empty,
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
    'RenderFields': RenderFields,
    'DXLevelChoice': DXLevelChoice,
    'Inputfilter': Inputfilter,
    'Global': Global,
    'EnvGlobal': EnvGlobal,
    'DamageFilter': DamageFilter,
    'ResponseContext': ResponseContext,
    'Breakable': Breakable,
    'BreakableBrush': BreakableBrush,
    'BreakableProp': BreakableProp,
    'BaseNPC': BaseNPC,
    'BaseNPCAssault': BaseNPCAssault,
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
    'env_lightglow': env_lightglow,
    'env_smokestack': env_smokestack,
    'env_fade': env_fade,
    'env_player_surface_trigger': env_player_surface_trigger,
    'env_tonemap_controller': env_tonemap_controller,
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
    'point_bonusmaps_accessor': point_bonusmaps_accessor,
    'game_ui': game_ui,
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
    'env_cascade_light': env_cascade_light,
    'newLight_Dir': newLight_Dir,
    'newLight_Point': newLight_Point,
    'newLight_Spot': newLight_Spot,
    'godrays_settings': godrays_settings,
    'newLights_settings': newLights_settings,
    'newlights_gbuffersettings': newlights_gbuffersettings,
    'newLights_Spawner': newLights_Spawner,
    'newxog_global': newxog_global,
    'newxog_settings': newxog_settings,
    'newxog_volume': newxog_volume,
    'light': light,
    'light_environment': light_environment,
    'light_spot': light_spot,
    'light_dynamic': light_dynamic,
    'shadow_control': shadow_control,
    'color_correction': color_correction,
    'fog_volume': fog_volume,
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
    'func_detail_blocker': func_detail_blocker,
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
    'filter_activator_model': filter_activator_model,
    'filter_activator_name': filter_activator_name,
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
    'prop_dynamic_playertouch': prop_dynamic_playertouch,
    'prop_dynamic_override': prop_dynamic_override,
    'BasePropPhysics': BasePropPhysics,
    'prop_physics_override': prop_physics_override,
    'prop_physics': prop_physics,
    'prop_physics_teleprop': prop_physics_teleprop,
    'prop_physics_multiplayer': prop_physics_multiplayer,
    'prop_ragdoll': prop_ragdoll,
    'prop_ragdoll_original': prop_ragdoll_original,
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
    'trigger_csm_volume': trigger_csm_volume,
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
    'func_reflective_glass': func_reflective_glass,
    'env_particle_performance_monitor': env_particle_performance_monitor,
    'npc_puppet': npc_puppet,
    'point_gamestats_counter': point_gamestats_counter,
    'func_instance': func_instance,
    'func_instance_parms': func_instance_parms,
    'func_instance_io_proxy': func_instance_io_proxy,
    'TalkNPC': TalkNPC,
    'PlayerCompanion': PlayerCompanion,
    'RappelNPC': RappelNPC,
    'trigger_physics_trap': trigger_physics_trap,
    'trigger_weapon_dissolve': trigger_weapon_dissolve,
    'trigger_weapon_strip': trigger_weapon_strip,
    'npc_crow': npc_crow,
    'npc_seagull': npc_seagull,
    'npc_pigeon': npc_pigeon,
    'npc_bullseye': npc_bullseye,
    'npc_enemyfinder': npc_enemyfinder,
    'env_gunfire': env_gunfire,
    'ai_goal_operator': ai_goal_operator,
    'info_darknessmode_lightsource': info_darknessmode_lightsource,
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
    'env_detail_controller': env_detail_controller,
    'env_global': env_global,
    'BaseCharger': BaseCharger,
    'item_healthcharger': item_healthcharger,
    'item_suitcharger': item_suitcharger,
    'BasePickup': BasePickup,
    'item_weapon_357': item_weapon_357,
    'item_weapon_crowbar': item_weapon_crowbar,
    'item_weapon_crossbow': item_weapon_crossbow,
    'item_weapon_frag': item_weapon_frag,
    'item_weapon_glock': item_weapon_glock,
    'item_weapon_gluon': item_weapon_gluon,
    'item_weapon_hivehand': item_weapon_hivehand,
    'item_weapon_mp5': item_weapon_mp5,
    'item_weapon_shotgun': item_weapon_shotgun,
    'item_weapon_rpg': item_weapon_rpg,
    'item_weapon_satchel': item_weapon_satchel,
    'item_weapon_snark': item_weapon_snark,
    'item_weapon_tau': item_weapon_tau,
    'item_weapon_tripmine': item_weapon_tripmine,
    'item_ammo_357': item_ammo_357,
    'item_ammo_crossbow': item_ammo_crossbow,
    'item_ammo_glock': item_ammo_glock,
    'item_ammo_energy': item_ammo_energy,
    'item_ammo_mp5': item_ammo_mp5,
    'item_ammo_shotgun': item_ammo_shotgun,
    'item_grenade_mp5': item_grenade_mp5,
    'item_grenade_rpg': item_grenade_rpg,
    'item_ammo_canister': item_ammo_canister,
    'item_ammo_crate': item_ammo_crate,
    'item_suit': item_suit,
    'item_battery': item_battery,
    'item_healthkit': item_healthkit,
    'item_longjump': item_longjump,
    'BaseGrenade': BaseGrenade,
    'grenade_satchel': grenade_satchel,
    'grenade_tripmine': grenade_tripmine,
    'BaseSentry': BaseSentry,
    'npc_plantlight': npc_plantlight,
    'npc_plantlight_stalker': npc_plantlight_stalker,
    'npc_puffballfungus': npc_puffballfungus,
    'npc_xentree': npc_xentree,
    'npc_protozoan': npc_protozoan,
    'npc_sentry_ceiling': npc_sentry_ceiling,
    'npc_sentry_ground': npc_sentry_ground,
    'npc_xenturret': npc_xenturret,
    'npc_alien_slave_dummy': npc_alien_slave_dummy,
    'npc_alien_slave': npc_alien_slave,
    'npc_xort': npc_xort,
    'npc_xortEB': npc_xortEB,
    'npc_headcrab': npc_headcrab,
    'npc_headcrab_fast': npc_headcrab_fast,
    'npc_headcrab_black': npc_headcrab_black,
    'npc_headcrab_baby': npc_headcrab_baby,
    'npc_barnacle': npc_barnacle,
    'npc_beneathticle': npc_beneathticle,
    'npc_bullsquid': npc_bullsquid,
    'npc_bullsquid_melee': npc_bullsquid_melee,
    'npc_houndeye': npc_houndeye,
    'npc_houndeye_suicide': npc_houndeye_suicide,
    'npc_houndeye_knockback': npc_houndeye_knockback,
    'npc_human_assassin': npc_human_assassin,
    'BaseMarine': BaseMarine,
    'npc_human_commander': npc_human_commander,
    'npc_human_grunt': npc_human_grunt,
    'npc_human_medic': npc_human_medic,
    'npc_human_grenadier': npc_human_grenadier,
    'npc_alien_controller': npc_alien_controller,
    'npc_xontroller': npc_xontroller,
    'npc_alien_grunt_unarmored': npc_alien_grunt_unarmored,
    'npc_alien_grunt_melee': npc_alien_grunt_melee,
    'npc_alien_grunt': npc_alien_grunt,
    'npc_alien_grunt_elite': npc_alien_grunt_elite,
    'npc_xen_grunt': npc_xen_grunt,
    'npc_cockroach': npc_cockroach,
    'npc_flyer_flock': npc_flyer_flock,
    'npc_gargantua': npc_gargantua,
    'info_bigmomma': info_bigmomma,
    'npc_gonarch': npc_gonarch,
    'env_gon_mortar_area': env_gon_mortar_area,
    'npc_generic': npc_generic,
    'npc_gman': npc_gman,
    'npc_ichthyosaur': npc_ichthyosaur,
    'npc_maintenance': npc_maintenance,
    'npc_nihilanth': npc_nihilanth,
    'prop_nihi_shield': prop_nihi_shield,
    'nihilanth_pylon': nihilanth_pylon,
    'BMBaseHelicopter': BMBaseHelicopter,
    'npc_manta': npc_manta,
    'prop_xen_grunt_pod': prop_xen_grunt_pod,
    'prop_xen_grunt_pod_dynamic': prop_xen_grunt_pod_dynamic,
    'prop_xen_int_barrel': prop_xen_int_barrel,
    'prop_barrel_cactus': prop_barrel_cactus,
    'prop_barrel_cactus_semilarge': prop_barrel_cactus_semilarge,
    'prop_barrel_cactus_adolescent': prop_barrel_cactus_adolescent,
    'prop_barrel_cactus_infant': prop_barrel_cactus_infant,
    'prop_barrel_cactus_exploder': prop_barrel_cactus_exploder,
    'prop_barrel_interloper': prop_barrel_interloper,
    'prop_barrel_interloper_small': prop_barrel_interloper_small,
    'npc_apache': npc_apache,
    'npc_osprey': npc_osprey,
    'npc_rat': npc_rat,
    'BaseColleague': BaseColleague,
    'npc_human_security': npc_human_security,
    'npc_human_scientist_kleiner': npc_human_scientist_kleiner,
    'npc_human_scientist_eli': npc_human_scientist_eli,
    'npc_human_scientist': npc_human_scientist,
    'npc_human_scientist_female': npc_human_scientist_female,
    'npc_xentacle': npc_xentacle,
    'npc_tentacle': npc_tentacle,
    'npc_snark': npc_snark,
    'npc_sniper': npc_sniper,
    'info_target_helicoptercrash': info_target_helicoptercrash,
    'info_dlightmap_update': info_dlightmap_update,
    'info_timescale_controller': info_timescale_controller,
    'info_stopallsounds': info_stopallsounds,
    'info_player_deathmatch': info_player_deathmatch,
    'info_player_marine': info_player_marine,
    'info_player_scientist': info_player_scientist,
    'material_timer': material_timer,
    'xen_portal': xen_portal,
    'env_introcredits': env_introcredits,
    'env_particle_beam': env_particle_beam,
    'env_particle_tesla': env_particle_tesla,
    'env_xen_portal': env_xen_portal,
    'env_xen_portal_template': env_xen_portal_template,
    'env_pinch': env_pinch,
    'env_dispenser': env_dispenser,
    'item_crate': item_crate,
    'func_50cal': func_50cal,
    'func_tow': func_tow,
    'func_tow_mp': func_tow_mp,
    'func_conveyor_bms': func_conveyor_bms,
    'item_tow_missile': item_tow_missile,
    'env_mortar_launcher': env_mortar_launcher,
    'env_mortar_controller': env_mortar_controller,
    'npc_abrams': npc_abrams,
    'npc_lav': npc_lav,
    'env_tram_screen': env_tram_screen,
    'prop_retinalscanner': prop_retinalscanner,
    'prop_physics_respawnable': prop_physics_respawnable,
    'prop_scalable': prop_scalable,
    'logic_parent': logic_parent,
    'env_xen_pushpad': env_xen_pushpad,
    'trigger_gargantua_shake': trigger_gargantua_shake,
    'trigger_lift': trigger_lift,
    'trigger_weaponfire': trigger_weaponfire,
    'func_minefield': func_minefield,
    'func_friction': func_friction,
    'prop_train_awesome': prop_train_awesome,
    'prop_train_apprehension': prop_train_apprehension,
    'BaseZombie': BaseZombie,
    'npc_zombie_scientist': npc_zombie_scientist,
    'npc_zombie_scientist_torso': npc_zombie_scientist_torso,
    'npc_zombie_security': npc_zombie_security,
    'npc_zombie_grunt': npc_zombie_grunt,
    'npc_zombie_grunt_torso': npc_zombie_grunt_torso,
    'npc_zombie_hev': npc_zombie_hev,
    'filter_damage_class': filter_damage_class,
    'filter_activator_flag': filter_activator_flag,
    'filter_activator_team': filter_activator_team,
    'prop_flare': prop_flare,
    'prop_surgerybot': prop_surgerybot,
    'env_xen_healpool': env_xen_healpool,
    'env_xen_healshower': env_xen_healshower,
    'prop_web_burnable': prop_web_burnable,
    'prop_charger_base': prop_charger_base,
    'prop_hev_charger': prop_hev_charger,
    'prop_radiation_charger': prop_radiation_charger,
    'camera_satellite': camera_satellite,
    'logic_achievement': logic_achievement,
    'ai_goal_throw_prop': ai_goal_throw_prop,
    'info_observer_menu': info_observer_menu,
    'game_round_win': game_round_win,
    'game_round_start': game_round_start,
    'game_mp_gamerules': game_mp_gamerules,
    'mp_round_time': mp_round_time,
    'env_gravity': env_gravity,
    'env_godrays_controller': env_godrays_controller,
    'misc_dead_hev': misc_dead_hev,
    'env_lensflare': env_lensflare,
    'env_fumer': env_fumer,
    'trigger_apply_impulse': trigger_apply_impulse,
    'info_nihilanth_summon': info_nihilanth_summon,
    'point_weaponstrip': point_weaponstrip,
    'misc_marionettist': misc_marionettist,
    'misc_xen_healing_pylon': misc_xen_healing_pylon,
    'misc_xen_shield': misc_xen_shield,
    'prop_physics_psychokinesis': prop_physics_psychokinesis,
    'nihiportalsbase': nihiportalsbase,
    'nihiportals_teleprops': nihiportals_teleprops,
    'music_track': music_track,
}