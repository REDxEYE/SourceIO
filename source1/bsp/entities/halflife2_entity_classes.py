
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


class Breakable(DamageFilter, Targetname, Shadow):
    def __init__(self):
        super(DamageFilter).__init__()
        super(Targetname).__init__()
        super(Shadow).__init__()
        self.ExplodeDamage = None  # Type: float
        self.ExplodeRadius = None  # Type: float
        self.PerformanceMode = None  # Type: choices
        self.BreakModelMessage = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        DamageFilter.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
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


class BaseNPC(Angles, DamageFilter, Targetname, ResponseContext, Shadow, RenderFields):
    def __init__(self):
        super(RenderFields).__init__()
        super(Angles).__init__()
        super(DamageFilter).__init__()
        super(Targetname).__init__()
        super(ResponseContext).__init__()
        super(Shadow).__init__()
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
        Angles.from_dict(instance, entity_data)
        DamageFilter.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        ResponseContext.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
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


class info_npc_spawn_destination(Parentname, Targetname, Angles):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.ReuseDelay = 1  # Type: float
        self.RenameNPC = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.ReuseDelay = float(entity_data.get('reusedelay', 1))  # Type: float
        instance.RenameNPC = entity_data.get('renamenpc', None)  # Type: string


class BaseNPCMaker(Angles, EnableDisable, Targetname):
    icon_sprite = "editor/npc_maker.vmt"
    def __init__(self):
        super(Angles).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        self.MaxNPCCount = 1  # Type: integer
        self.SpawnFrequency = "5"  # Type: string
        self.MaxLiveChildren = 5  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class TriggerOnce(Parentname, EnableDisable, Origin, Targetname, Global):
    def __init__(self):
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
        self.filtername = None  # Type: filterclass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        instance.filtername = entity_data.get('filtername', None)  # Type: filterclass


class Trigger(TriggerOnce):
    def __init__(self):
        super(TriggerOnce).__init__()
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        TriggerOnce.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)


class worldbase(Base):
    def __init__(self):
        super().__init__()
        self.message = None  # Type: string
        self.skyname = "sky_day01_01"  # Type: string
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

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.message = entity_data.get('message', None)  # Type: string
        instance.skyname = entity_data.get('skyname', "sky_day01_01")  # Type: string
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


class worldspawn(ResponseContext, Targetname, worldbase):
    def __init__(self):
        super(ResponseContext).__init__()
        super(Targetname).__init__()
        super(worldbase).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        ResponseContext.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        worldbase.from_dict(instance, entity_data)


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


class env_lightglow(Parentname, Targetname, Angles):
    model = "models/editor/axis_helper_thick.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
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
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


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


class func_ladderendpoint(Parentname, Targetname, Angles):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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


class func_wall(RenderFields, Global, Targetname, Shadow):
    def __init__(self):
        super(RenderFields).__init__()
        super(Global).__init__()
        super(Targetname).__init__()
        super(Shadow).__init__()
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        RenderFields.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
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


class func_brush(Parentname, Inputfilter, EnableDisable, Origin, Targetname, Shadow, RenderFields, Global):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Inputfilter).__init__()
        super(EnableDisable).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        super(Shadow).__init__()
        super(Global).__init__()
        self._minlight = None  # Type: string
        self.Solidity = None  # Type: choices
        self.excludednpc = None  # Type: string
        self.invert_exclusion = None  # Type: choices
        self.solidbsp = None  # Type: choices
        self.vrad_brush_cast_shadows = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Inputfilter.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.Solidity = entity_data.get('solidity', None)  # Type: choices
        instance.excludednpc = entity_data.get('excludednpc', None)  # Type: string
        instance.invert_exclusion = entity_data.get('invert_exclusion', None)  # Type: choices
        instance.solidbsp = entity_data.get('solidbsp', None)  # Type: choices
        instance.vrad_brush_cast_shadows = entity_data.get('vrad_brush_cast_shadows', None)  # Type: choices


class vgui_screen_base(Parentname, Targetname, Angles):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.panelname = None  # Type: string
        self.overlaymaterial = None  # Type: string
        self.width = 32  # Type: integer
        self.height = 32  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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


class vgui_slideshow_display(Angles, Parentname, Targetname):
    model = "models/editor/axis_helper_thick.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
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
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.displaytext = entity_data.get('displaytext', None)  # Type: string
        instance.directory = entity_data.get('directory', "slideshow")  # Type: string
        instance.minslidetime = float(entity_data.get('minslidetime', 0.5))  # Type: float
        instance.maxslidetime = float(entity_data.get('maxslidetime', 0.5))  # Type: float
        instance.cycletype = entity_data.get('cycletype', None)  # Type: choices
        instance.nolistrepeat = entity_data.get('nolistrepeat', None)  # Type: choices
        instance.width = parse_source_value(entity_data.get('width', 256))  # Type: integer
        instance.height = parse_source_value(entity_data.get('height', 128))  # Type: integer


class cycler(Parentname, Angles, Targetname, RenderFields):
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
        Angles.from_dict(instance, entity_data)
        RenderFxChoices.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
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


class env_embers(Parentname, Targetname, Angles):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.particletype = None  # Type: choices
        self.density = 50  # Type: integer
        self.lifetime = 4  # Type: integer
        self.speed = 32  # Type: integer
        self.rendercolor = [255, 255, 255]  # Type: color255

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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


class env_fire(Parentname, EnableDisable, Targetname):
    icon_sprite = "editor/env_fire"
    def __init__(self):
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
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
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class env_fog_controller(Angles, DXLevelChoice, Targetname):
    icon_sprite = "editor/fog_controller.vmt"
    def __init__(self):
        super(Angles).__init__()
        super(DXLevelChoice).__init__()
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
        Angles.from_dict(instance, entity_data)
        DXLevelChoice.from_dict(instance, entity_data)
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


class env_steam(Parentname, Targetname, Angles):
    viewport_model = "models/editor/spot_cone.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
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
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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


class env_soundscape(Parentname, EnableDisable, Targetname):
    icon_sprite = "editor/env_soundscape.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
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
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class env_spark(Parentname, Targetname, Angles):
    icon_sprite = "editor/env_spark.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.MaxDelay = "0"  # Type: string
        self.Magnitude = "CHOICES NOT SUPPORTED"  # Type: choices
        self.TrailLength = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.MaxDelay = entity_data.get('maxdelay', "0")  # Type: string
        instance.Magnitude = entity_data.get('magnitude', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.TrailLength = entity_data.get('traillength', "CHOICES NOT SUPPORTED")  # Type: choices


class env_sprite(Parentname, RenderFields, DXLevelChoice, Targetname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(DXLevelChoice).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.framerate = "10.0"  # Type: string
        self.model = "sprites/glow01.spr"  # Type: sprite
        self.scale = None  # Type: string
        self.GlowProxySize = 2.0  # Type: float
        self.HDRColorScale = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        DXLevelChoice.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.framerate = entity_data.get('framerate', "10.0")  # Type: string
        instance.model = entity_data.get('model', "sprites/glow01.spr")  # Type: sprite
        instance.scale = entity_data.get('scale', None)  # Type: string
        instance.GlowProxySize = float(entity_data.get('glowproxysize', 2.0))  # Type: float
        instance.HDRColorScale = float(entity_data.get('hdrcolorscale', 1.0))  # Type: float


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
        self.rulescript = None  # Type: string
        self.concept = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        ResponseContext.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class point_spotlight(Parentname, Angles, DXLevelChoice, Targetname, RenderFields):
    model = "models/editor/cone_helper.mdl"
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(DXLevelChoice).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.spotlightlength = 500  # Type: integer
        self.spotlightwidth = 50  # Type: integer
        self.HDRColorScale = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        DXLevelChoice.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
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
    model = "models/editor/axis_helper_thick.mdl"
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


class info_target(Parentname, Targetname, Angles):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_particle_system(Parentname, Targetname, Angles):
    model = "models/editor/cone_helper.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.effect_name = None  # Type: string
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
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.effect_name = entity_data.get('effect_name', None)  # Type: string
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


class phys_ragdollmagnet(EnableDisable, Parentname, Targetname, Angles):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(EnableDisable).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.axis = None  # Type: vecline
        self.radius = 512  # Type: float
        self.force = 5000  # Type: float
        self.target = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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


class info_teleport_destination(Parentname, PlayerClass, Targetname, Angles):
    model = "models/editor/playerstart.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(PlayerClass).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        PlayerClass.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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


class info_node_hint(Angles, HintNode, Targetname):
    model = "models/editor/ground_node_hint.mdl"
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


class info_node_air_hint(Angles, HintNode, Targetname):
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
        HintNode.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.nodeheight = parse_source_value(entity_data.get('nodeheight', 0))  # Type: integer


class info_hint(Angles, HintNode, Targetname):
    model = "models/editor/node_hint.mdl"
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
    model = "models/editor/climb_node.mdl"
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


class light_spot(Angles, Targetname, Light):
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
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
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Light.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance._inner_cone = parse_source_value(entity_data.get('_inner_cone', 30))  # Type: integer
        instance._cone = parse_source_value(entity_data.get('_cone', 45))  # Type: integer
        instance._exponent = parse_source_value(entity_data.get('_exponent', 1))  # Type: integer
        instance._distance = parse_source_value(entity_data.get('_distance', 0))  # Type: integer
        instance.pitch = float(entity_data.get('pitch', -90))  # Type: angle_negative_pitch


class light_dynamic(Parentname, Targetname, Angles):
    icon_sprite = "editor/light.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
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
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.angles = entity_data.get('angles', "80 30 0")  # Type: string
        instance.color = parse_int_vector(entity_data.get('color', "128 128 128"))  # Type: color255
        instance.distance = float(entity_data.get('distance', 75))  # Type: float
        instance.disableallshadows = entity_data.get('disableallshadows', None)  # Type: choices


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


class func_movelinear(Parentname, RenderFields, Origin, Targetname):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
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
        RenderFields.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.movedir = parse_float_vector(entity_data.get('movedir', "0 0 0"))  # Type: angle
        instance.startposition = float(entity_data.get('startposition', 0))  # Type: float
        instance.speed = parse_source_value(entity_data.get('speed', 100))  # Type: integer
        instance.movedistance = float(entity_data.get('movedistance', 100))  # Type: float
        instance.blockdamage = float(entity_data.get('blockdamage', 0))  # Type: float
        instance.startsound = entity_data.get('startsound', None)  # Type: sound
        instance.stopsound = entity_data.get('stopsound', None)  # Type: sound


class func_water_analog(Parentname, Origin, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        self.movedir = [0.0, 0.0, 0.0]  # Type: angle
        self.startposition = None  # Type: float
        self.speed = 100  # Type: integer
        self.movedistance = 100  # Type: float
        self.startsound = None  # Type: sound
        self.stopsound = None  # Type: sound
        self.WaveHeight = "3.0"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.movedir = parse_float_vector(entity_data.get('movedir', "0 0 0"))  # Type: angle
        instance.startposition = float(entity_data.get('startposition', 0))  # Type: float
        instance.speed = parse_source_value(entity_data.get('speed', 100))  # Type: integer
        instance.movedistance = float(entity_data.get('movedistance', 100))  # Type: float
        instance.startsound = entity_data.get('startsound', None)  # Type: sound
        instance.stopsound = entity_data.get('stopsound', None)  # Type: sound
        instance.WaveHeight = entity_data.get('waveheight', "3.0")  # Type: string


class func_rotating(Parentname, Angles, Origin, Targetname, Shadow, RenderFields):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        super(Shadow).__init__()
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
        Angles.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance.maxspeed = parse_source_value(entity_data.get('maxspeed', 100))  # Type: integer
        instance.fanfriction = parse_source_value(entity_data.get('fanfriction', 20))  # Type: integer
        instance.message = entity_data.get('message', None)  # Type: sound
        instance.volume = parse_source_value(entity_data.get('volume', 10))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.dmg = parse_source_value(entity_data.get('dmg', 0))  # Type: integer
        instance.solidbsp = entity_data.get('solidbsp', None)  # Type: choices


class func_platrot(Parentname, Angles, BasePlat, Origin, Targetname, Shadow, RenderFields):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(BasePlat).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        super(Shadow).__init__()
        self.noise1 = None  # Type: sound
        self.noise2 = None  # Type: sound
        self.speed = 50  # Type: integer
        self.height = None  # Type: integer
        self.rotation = None  # Type: integer
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        BasePlat.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance.noise1 = entity_data.get('noise1', None)  # Type: sound
        instance.noise2 = entity_data.get('noise2', None)  # Type: sound
        instance.speed = parse_source_value(entity_data.get('speed', 50))  # Type: integer
        instance.height = parse_source_value(entity_data.get('height', 0))  # Type: integer
        instance.rotation = parse_source_value(entity_data.get('rotation', 0))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class keyframe_track(Parentname, Targetname, Angles, KeyFrame):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        super(KeyFrame).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        KeyFrame.from_dict(instance, entity_data)


class move_keyframed(Parentname, Mover, Targetname, KeyFrame):
    def __init__(self):
        super(Parentname).__init__()
        super(Mover).__init__()
        super(Targetname).__init__()
        super(KeyFrame).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Mover.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        KeyFrame.from_dict(instance, entity_data)


class move_track(Parentname, Mover, Targetname, KeyFrame):
    def __init__(self):
        super(Parentname).__init__()
        super(Mover).__init__()
        super(Targetname).__init__()
        super(KeyFrame).__init__()
        self.WheelBaseLength = 50  # Type: integer
        self.Damage = None  # Type: integer
        self.NoRotate = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Mover.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        KeyFrame.from_dict(instance, entity_data)
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


class keyframe_rope(Parentname, Targetname, RopeKeyFrame, KeyFrame):
    model = "models/editor/axis_helper_thick.mdl"
    def __init__(self):
        super(RopeKeyFrame).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(KeyFrame).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RopeKeyFrame.from_dict(instance, entity_data)
        KeyFrame.from_dict(instance, entity_data)


class move_rope(Parentname, Targetname, RopeKeyFrame, KeyFrame):
    model = "models/editor/axis_helper.mdl"
    def __init__(self):
        super(RopeKeyFrame).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(KeyFrame).__init__()
        self.PositionInterpolator = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RopeKeyFrame.from_dict(instance, entity_data)
        KeyFrame.from_dict(instance, entity_data)
        instance.PositionInterpolator = entity_data.get('positioninterpolator', "CHOICES NOT SUPPORTED")  # Type: choices


class Button(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class func_button(Parentname, DamageFilter, Origin, Targetname, Button, RenderFields):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(DamageFilter).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        super(Button).__init__()
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
        DamageFilter.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Button.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
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


class func_rot_button(Parentname, Angles, EnableDisable, Origin, Targetname, Button, Global):
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(EnableDisable).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        super(Button).__init__()
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
        EnableDisable.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Button.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        instance.master = entity_data.get('master', None)  # Type: string
        instance.speed = parse_source_value(entity_data.get('speed', 50))  # Type: integer
        instance.health = parse_source_value(entity_data.get('health', 0))  # Type: integer
        instance.sounds = entity_data.get('sounds', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.wait = parse_source_value(entity_data.get('wait', 3))  # Type: integer
        instance.distance = parse_source_value(entity_data.get('distance', 90))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class momentary_rot_button(Parentname, Angles, Origin, Targetname, RenderFields):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
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
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance.speed = parse_source_value(entity_data.get('speed', 50))  # Type: integer
        instance.master = entity_data.get('master', None)  # Type: string
        instance.sounds = entity_data.get('sounds', None)  # Type: choices
        instance.distance = parse_source_value(entity_data.get('distance', 90))  # Type: integer
        instance.returnspeed = parse_source_value(entity_data.get('returnspeed', 0))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.startposition = float(entity_data.get('startposition', 0))  # Type: float
        instance.startdirection = entity_data.get('startdirection', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.solidbsp = entity_data.get('solidbsp', None)  # Type: choices


class Door(Parentname, Targetname, Shadow, RenderFields, Global):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Shadow).__init__()
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
        Targetname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
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


class func_door_rotating(Door, Origin, Angles):
    def __init__(self):
        super(Door).__init__()
        super(Origin).__init__()
        super(Angles).__init__()
        self.distance = 90  # Type: integer
        self.solidbsp = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Door.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.distance = parse_source_value(entity_data.get('distance', 90))  # Type: integer
        instance.solidbsp = entity_data.get('solidbsp', None)  # Type: choices


class prop_door_rotating(Parentname, Angles, Targetname, Global, Studiomodel):
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
        super(Studiomodel).__init__()
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
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
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


class env_dustpuff(Parentname, Targetname, Angles):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.scale = 8  # Type: float
        self.speed = 16  # Type: float
        self.color = [128, 128, 128]  # Type: color255

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.scale = float(entity_data.get('scale', 8))  # Type: float
        instance.speed = float(entity_data.get('speed', 16))  # Type: float
        instance.color = parse_int_vector(entity_data.get('color', "128 128 128"))  # Type: color255


class env_particlescript(Parentname, Targetname, Angles):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.model = "models/Ambient_citadel_paths.mdl"  # Type: studio

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', "models/Ambient_citadel_paths.mdl")  # Type: studio


class env_effectscript(Parentname, Targetname, Angles):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.model = "models/Effects/teleporttrail.mdl"  # Type: studio
        self.scriptfile = "scripts/effects/testeffect.txt"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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


class point_viewcontrol(Angles, Parentname, Targetname):
    viewport_model = "models/editor/camera.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
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
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class env_microphone(Parentname, EnableDisable, Targetname):
    icon_sprite = "editor/env_microphone.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
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
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class env_entity_maker(Parentname, Targetname, Angles):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.EntityTemplate = None  # Type: target_destination
        self.PostSpawnSpeed = 0  # Type: float
        self.PostSpawnDirection = [0.0, 0.0, 0.0]  # Type: angle
        self.PostSpawnDirectionVariance = 0.15  # Type: float
        self.PostSpawnInheritAngles = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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


class point_anglesensor(Parentname, EnableDisable, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.lookatname = None  # Type: target_destination
        self.duration = None  # Type: float
        self.tolerance = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class point_proximity_sensor(Parentname, EnableDisable, Targetname, Angles):
    def __init__(self):
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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


class func_physbox(BreakableBrush, Origin, RenderFields):
    def __init__(self):
        super(BreakableBrush).__init__()
        super(RenderFields).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        super(Shadow).__init__()
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
        BreakableBrush.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
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


class phys_magnet(Parentname, Studiomodel, Targetname, Angles):
    def __init__(self):
        super(Parentname).__init__()
        super(Studiomodel).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
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
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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


class prop_dynamic_base(Parentname, Angles, DXLevelChoice, BreakableProp, BaseFadeProp, RenderFields, Global, Studiomodel):
    def __init__(self):
        super(BreakableProp).__init__()
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(DXLevelChoice).__init__()
        super(BaseFadeProp).__init__()
        super(Global).__init__()
        super(Studiomodel).__init__()
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
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        DXLevelChoice.from_dict(instance, entity_data)
        BreakableProp.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
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

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_dynamic_base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.health = parse_source_value(entity_data.get('health', 0))  # Type: integer


class BasePropPhysics(Angles, DXLevelChoice, BreakableProp, BaseFadeProp, Global, Studiomodel):
    def __init__(self):
        super(BreakableProp).__init__()
        super(Angles).__init__()
        super(DXLevelChoice).__init__()
        super(BaseFadeProp).__init__()
        super(Targetname).__init__()
        super(Global).__init__()
        super(Studiomodel).__init__()
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
        DXLevelChoice.from_dict(instance, entity_data)
        BreakableProp.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
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

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePropPhysics.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


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


class prop_ragdoll(Angles, DXLevelChoice, EnableDisable, BaseFadeProp, Targetname, Studiomodel):
    def __init__(self):
        super(Angles).__init__()
        super(DXLevelChoice).__init__()
        super(EnableDisable).__init__()
        super(BaseFadeProp).__init__()
        super(Targetname).__init__()
        super(Studiomodel).__init__()
        self.origin = [0, 0, 0]
        self.angleOverride = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        DXLevelChoice.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
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
        super(Origin).__init__()
        super(Shadow).__init__()
        self.minhealthdmg = None  # Type: integer
        self._minlight = None  # Type: string
        self.physdamagescale = 1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BreakableBrush.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        instance.minhealthdmg = parse_source_value(entity_data.get('minhealthdmg', 0))  # Type: integer
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.physdamagescale = float(entity_data.get('physdamagescale', 1.0))  # Type: float


class func_breakable_surf(BreakableBrush, RenderFields):
    def __init__(self):
        super(BreakableBrush).__init__()
        super(RenderFields).__init__()
        super(Shadow).__init__()
        self.fragility = 100  # Type: integer
        self.surfacetype = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BreakableBrush.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        instance.fragility = parse_source_value(entity_data.get('fragility', 100))  # Type: integer
        instance.surfacetype = entity_data.get('surfacetype', None)  # Type: choices


class func_conveyor(Parentname, RenderFields, Targetname, Shadow):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Shadow).__init__()
        self.movedir = [0.0, 0.0, 0.0]  # Type: angle
        self.speed = "100"  # Type: string
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
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


class func_illusionary(Parentname, Origin, Targetname, Shadow, RenderFields):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        super(Shadow).__init__()
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
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


class func_guntarget(Parentname, RenderFields, Global, Targetname):
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
        RenderFields.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class BaseTrain(Parentname, Origin, Targetname, Shadow, RenderFields, Global):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        super(Shadow).__init__()
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
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
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


class path_track(Parentname, Targetname, Angles):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.altpath = None  # Type: target_destination
        self.speed = None  # Type: float
        self.radius = None  # Type: float
        self.orientationtype = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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
        Targetname.from_dict(instance, entity_data)
        Trigger.from_dict(instance, entity_data)
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
        Targetname.from_dict(instance, entity_data)
        Trigger.from_dict(instance, entity_data)


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


class trigger_wind(Angles, Trigger):
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
        Angles.from_dict(instance, entity_data)
        Trigger.from_dict(instance, entity_data)
        instance.Speed = parse_source_value(entity_data.get('speed', 200))  # Type: integer
        instance.SpeedNoise = parse_source_value(entity_data.get('speednoise', 0))  # Type: integer
        instance.DirectionNoise = parse_source_value(entity_data.get('directionnoise', 10))  # Type: integer
        instance.HoldTime = parse_source_value(entity_data.get('holdtime', 0))  # Type: integer
        instance.HoldNoise = parse_source_value(entity_data.get('holdnoise', 0))  # Type: integer


class trigger_impact(Origin, Targetname, Angles):
    def __init__(self):
        super(Origin).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.Magnitude = 200  # Type: float
        self.noise = 0.1  # Type: float
        self.viewkick = 0.05  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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


class ai_speechfilter(ResponseContext, EnableDisable, Targetname):
    def __init__(self):
        super(ResponseContext).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.subject = None  # Type: target_destination
        self.IdleModifier = 1.0  # Type: float
        self.NeverSayHello = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        ResponseContext.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class logic_playerproxy(DamageFilter, Targetname):
    def __init__(self):
        super(DamageFilter).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        DamageFilter.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class env_projectedtexture(Parentname, Targetname, Angles):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
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
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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
        Targetname.from_dict(instance, entity_data)
        BaseNPC.from_dict(instance, entity_data)
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


class AlyxInteractable(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class CombineBallSpawners(Origin, Global, Targetname, Angles):
    def __init__(self):
        super(Origin).__init__()
        super(Global).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.ballcount = 3  # Type: integer
        self.minspeed = 300.0  # Type: float
        self.maxspeed = 600.0  # Type: float
        self.ballradius = 20.0  # Type: float
        self.balltype = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ballrespawntime = 4.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Origin.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
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


class npc_blob(BaseNPC):
    model = "models/combine_soldier.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_grenade_frag(BaseNPC):
    model = "models/Weapons/w_grenade.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_combine_cannon(BaseNPC):
    model = "models/combine_soldier.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.sightdist = 1024  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.sightdist = float(entity_data.get('sightdist', 1024))  # Type: float


class npc_combine_camera(BaseNPC):
    model = "models/combine_camera/combine_camera.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.innerradius = 300  # Type: integer
        self.outerradius = 450  # Type: integer
        self.minhealthdmg = None  # Type: integer
        self.defaulttarget = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.innerradius = parse_source_value(entity_data.get('innerradius', 300))  # Type: integer
        instance.outerradius = parse_source_value(entity_data.get('outerradius', 450))  # Type: integer
        instance.minhealthdmg = parse_source_value(entity_data.get('minhealthdmg', 0))  # Type: integer
        instance.defaulttarget = entity_data.get('defaulttarget', None)  # Type: target_destination


class npc_turret_ground(Parentname, BaseNPC, AlyxInteractable):
    model = "models/combine_turrets/ground_turret.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        super(Parentname).__init__()
        super(AlyxInteractable).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        BaseNPC.from_dict(instance, entity_data)
        AlyxInteractable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class npc_turret_ceiling(Angles, Studiomodel, Targetname):
    model = "models/combine_turrets/ceiling_turret.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Studiomodel).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.minhealthdmg = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.minhealthdmg = parse_source_value(entity_data.get('minhealthdmg', 0))  # Type: integer


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


class VehicleDriverNPC(BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        self.vehicle = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.vehicle = entity_data.get('vehicle', None)  # Type: target_destination


class npc_vehicledriver(VehicleDriverNPC):
    model = "models/roller.mdl"
    def __init__(self):
        super(VehicleDriverNPC).__init__()
        self.drivermaxspeed = 1  # Type: float
        self.driverminspeed = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        VehicleDriverNPC.from_dict(instance, entity_data)
        instance.drivermaxspeed = float(entity_data.get('drivermaxspeed', 1))  # Type: float
        instance.driverminspeed = float(entity_data.get('driverminspeed', 0))  # Type: float


class npc_cranedriver(VehicleDriverNPC):
    model = "models/roller.mdl"
    def __init__(self):
        super(VehicleDriverNPC).__init__()
        self.releasepause = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        VehicleDriverNPC.from_dict(instance, entity_data)
        instance.releasepause = float(entity_data.get('releasepause', 0))  # Type: float


class npc_apcdriver(VehicleDriverNPC):
    model = "models/roller.mdl"
    def __init__(self):
        super(VehicleDriverNPC).__init__()
        self.drivermaxspeed = 1  # Type: float
        self.driverminspeed = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        VehicleDriverNPC.from_dict(instance, entity_data)
        instance.drivermaxspeed = float(entity_data.get('drivermaxspeed', 1))  # Type: float
        instance.driverminspeed = float(entity_data.get('driverminspeed', 0))  # Type: float


class npc_rollermine(BaseNPC, AlyxInteractable):
    model = "models/roller.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        super(AlyxInteractable).__init__()
        self.startburied = "CHOICES NOT SUPPORTED"  # Type: choices
        self.uniformsightdist = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        AlyxInteractable.from_dict(instance, entity_data)
        instance.startburied = entity_data.get('startburied', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.uniformsightdist = entity_data.get('uniformsightdist', None)  # Type: choices


class npc_missiledefense(BaseNPC):
    model = "models/missile_defense.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_sniper(BaseNPC):
    model = "models/combine_soldier.mdl"
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


class info_radar_target(Parentname, EnableDisable, Targetname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.radius = 6000  # Type: float
        self.type = None  # Type: choices
        self.mode = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.radius = float(entity_data.get('radius', 6000))  # Type: float
        instance.type = entity_data.get('type', None)  # Type: choices
        instance.mode = entity_data.get('mode', None)  # Type: choices


class info_target_vehicle_transition(Angles, EnableDisable, Targetname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Angles).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_snipertarget(Parentname, Targetname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.speed = 2  # Type: integer
        self.groupname = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.speed = parse_source_value(entity_data.get('speed', 2))  # Type: integer
        instance.groupname = entity_data.get('groupname', None)  # Type: string


class prop_thumper(Angles, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.model = "models/props_combine/CombineThumper002.mdl"  # Type: studio
        self.dustscale = "CHOICES NOT SUPPORTED"  # Type: choices
        self.EffectRadius = 1000  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', "models/props_combine/CombineThumper002.mdl")  # Type: studio
        instance.dustscale = entity_data.get('dustscale', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.EffectRadius = parse_source_value(entity_data.get('effectradius', 1000))  # Type: integer


class npc_antlion(BaseNPC):
    model = "models/antlion.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.startburrowed = "CHOICES NOT SUPPORTED"  # Type: choices
        self.radius = 256  # Type: integer
        self.eludedist = 1024  # Type: integer
        self.ignorebugbait = "CHOICES NOT SUPPORTED"  # Type: choices
        self.unburroweffects = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.startburrowed = entity_data.get('startburrowed', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.radius = parse_source_value(entity_data.get('radius', 256))  # Type: integer
        instance.eludedist = parse_source_value(entity_data.get('eludedist', 1024))  # Type: integer
        instance.ignorebugbait = entity_data.get('ignorebugbait', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.unburroweffects = entity_data.get('unburroweffects', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_antlionguard(BaseNPC):
    model = "models/antlion_guard.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.startburrowed = "CHOICES NOT SUPPORTED"  # Type: choices
        self.allowbark = "CHOICES NOT SUPPORTED"  # Type: choices
        self.cavernbreed = "CHOICES NOT SUPPORTED"  # Type: choices
        self.incavern = "CHOICES NOT SUPPORTED"  # Type: choices
        self.shovetargets = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.startburrowed = entity_data.get('startburrowed', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.allowbark = entity_data.get('allowbark', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.cavernbreed = entity_data.get('cavernbreed', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.incavern = entity_data.get('incavern', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.shovetargets = entity_data.get('shovetargets', None)  # Type: string


class npc_crow(BaseNPC):
    model = "models/crow.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.deaf = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.deaf = entity_data.get('deaf', None)  # Type: choices


class npc_seagull(BaseNPC):
    model = "models/seagull.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.deaf = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.deaf = entity_data.get('deaf', None)  # Type: choices


class npc_pigeon(BaseNPC):
    model = "models/pigeon.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.deaf = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.deaf = entity_data.get('deaf', None)  # Type: choices


class npc_ichthyosaur(BaseNPC):
    model = "models/ichthyosaur.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class BaseHeadcrab(BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        self.startburrowed = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.startburrowed = entity_data.get('startburrowed', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_headcrab(BaseHeadcrab, Parentname):
    model = "models/Headcrabclassic.mdl"
    def __init__(self):
        super(BaseHeadcrab).__init__()
        super(Parentname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseHeadcrab.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)


class npc_headcrab_fast(BaseHeadcrab):
    model = "models/Headcrab.mdl"
    def __init__(self):
        super(BaseHeadcrab).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseHeadcrab.from_dict(instance, entity_data)


class npc_headcrab_black(BaseHeadcrab):
    model = "models/Headcrabblack.mdl"
    def __init__(self):
        super(BaseHeadcrab).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseHeadcrab.from_dict(instance, entity_data)


class npc_stalker(BaseNPC):
    model = "models/Stalker.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.BeamPower = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.BeamPower = entity_data.get('beampower', "CHOICES NOT SUPPORTED")  # Type: choices


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


class npc_enemyfinder_combinecannon(Parentname, BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        super(Parentname).__init__()
        self.FieldOfView = "0.2"  # Type: string
        self.MinSearchDist = None  # Type: integer
        self.MaxSearchDist = 2048  # Type: integer
        self.SnapToEnt = None  # Type: target_destination
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
        instance.SnapToEnt = entity_data.get('snaptoent', None)  # Type: target_destination
        instance.freepass_timetotrigger = float(entity_data.get('freepass_timetotrigger', 0))  # Type: float
        instance.freepass_duration = float(entity_data.get('freepass_duration', 0))  # Type: float
        instance.freepass_movetolerance = float(entity_data.get('freepass_movetolerance', 120))  # Type: float
        instance.freepass_refillrate = float(entity_data.get('freepass_refillrate', 0.5))  # Type: float
        instance.freepass_peektime = float(entity_data.get('freepass_peektime', 0))  # Type: float
        instance.StartOn = entity_data.get('starton', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_citizen(Parentname, TalkNPC, PlayerCompanion):
    def __init__(self):
        super(BaseNPC).__init__()
        super(TalkNPC).__init__()
        super(PlayerCompanion).__init__()
        super(Parentname).__init__()
        self.additionalequipment = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ammosupply = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ammoamount = 1  # Type: integer
        self.citizentype = "CHOICES NOT SUPPORTED"  # Type: choices
        self.expressiontype = "CHOICES NOT SUPPORTED"  # Type: choices
        self.model = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ExpressionOverride = None  # Type: string
        self.notifynavfailblocked = None  # Type: choices
        self.neverleaveplayersquad = None  # Type: choices
        self.denycommandconcept = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        TalkNPC.from_dict(instance, entity_data)
        PlayerCompanion.from_dict(instance, entity_data)
        BaseNPC.from_dict(instance, entity_data)
        instance.additionalequipment = entity_data.get('additionalequipment', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ammosupply = entity_data.get('ammosupply', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ammoamount = parse_source_value(entity_data.get('ammoamount', 1))  # Type: integer
        instance.citizentype = entity_data.get('citizentype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.expressiontype = entity_data.get('expressiontype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.model = entity_data.get('model', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ExpressionOverride = entity_data.get('expressionoverride', None)  # Type: string
        instance.notifynavfailblocked = entity_data.get('notifynavfailblocked', None)  # Type: choices
        instance.neverleaveplayersquad = entity_data.get('neverleaveplayersquad', None)  # Type: choices
        instance.denycommandconcept = entity_data.get('denycommandconcept', None)  # Type: string


class npc_fisherman(BaseNPC):
    model = "models/Barney.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.ExpressionOverride = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.ExpressionOverride = entity_data.get('expressionoverride', None)  # Type: string


class npc_barney(TalkNPC, PlayerCompanion):
    model = "models/Barney.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        super(TalkNPC).__init__()
        super(PlayerCompanion).__init__()
        self.additionalequipment = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ExpressionOverride = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TalkNPC.from_dict(instance, entity_data)
        PlayerCompanion.from_dict(instance, entity_data)
        BaseNPC.from_dict(instance, entity_data)
        instance.additionalequipment = entity_data.get('additionalequipment', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ExpressionOverride = entity_data.get('expressionoverride', None)  # Type: string


class BaseCombine(RappelNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        super(RappelNPC).__init__()
        self.additionalequipment = "CHOICES NOT SUPPORTED"  # Type: choices
        self.NumGrenades = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        RappelNPC.from_dict(instance, entity_data)
        BaseNPC.from_dict(instance, entity_data)
        instance.additionalequipment = entity_data.get('additionalequipment', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.NumGrenades = entity_data.get('numgrenades', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_combine_s(BaseCombine):
    model = "models/Combine_Soldier.mdl"
    def __init__(self):
        super(BaseCombine).__init__()
        self.model = "CHOICES NOT SUPPORTED"  # Type: choices
        self.tacticalvariant = "CHOICES NOT SUPPORTED"  # Type: choices
        self.usemarch = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseCombine.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.tacticalvariant = entity_data.get('tacticalvariant', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.usemarch = entity_data.get('usemarch', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_launcher(Parentname, BaseNPC):
    model = "models/junk/w_traffcone.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.StartOn = None  # Type: choices
        self.MissileModel = "models/Weapons/wscanner_grenade.mdl"  # Type: studio
        self.LaunchSound = "npc/waste_scanner/grenade_fire.wav"  # Type: sound
        self.FlySound = "ambient/objects/machine2.wav"  # Type: sound
        self.SmokeTrail = "CHOICES NOT SUPPORTED"  # Type: choices
        self.LaunchSmoke = "CHOICES NOT SUPPORTED"  # Type: choices
        self.LaunchDelay = 8  # Type: integer
        self.LaunchSpeed = "200"  # Type: string
        self.PathCornerName = None  # Type: target_destination
        self.HomingSpeed = None  # Type: string
        self.HomingStrength = 10  # Type: integer
        self.HomingDelay = None  # Type: string
        self.HomingRampUp = "0.5"  # Type: string
        self.HomingDuration = "5"  # Type: string
        self.HomingRampDown = "1.0"  # Type: string
        self.Gravity = "1.0"  # Type: string
        self.MinRange = 100  # Type: integer
        self.MaxRange = 2048  # Type: integer
        self.SpinMagnitude = None  # Type: string
        self.SpinSpeed = None  # Type: string
        self.Damage = "50"  # Type: string
        self.DamageRadius = "200"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        BaseNPC.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.StartOn = entity_data.get('starton', None)  # Type: choices
        instance.MissileModel = entity_data.get('missilemodel', "models/Weapons/wscanner_grenade.mdl")  # Type: studio
        instance.LaunchSound = entity_data.get('launchsound', "npc/waste_scanner/grenade_fire.wav")  # Type: sound
        instance.FlySound = entity_data.get('flysound', "ambient/objects/machine2.wav")  # Type: sound
        instance.SmokeTrail = entity_data.get('smoketrail', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.LaunchSmoke = entity_data.get('launchsmoke', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.LaunchDelay = parse_source_value(entity_data.get('launchdelay', 8))  # Type: integer
        instance.LaunchSpeed = entity_data.get('launchspeed', "200")  # Type: string
        instance.PathCornerName = entity_data.get('pathcornername', None)  # Type: target_destination
        instance.HomingSpeed = entity_data.get('homingspeed', None)  # Type: string
        instance.HomingStrength = parse_source_value(entity_data.get('homingstrength', 10))  # Type: integer
        instance.HomingDelay = entity_data.get('homingdelay', None)  # Type: string
        instance.HomingRampUp = entity_data.get('homingrampup', "0.5")  # Type: string
        instance.HomingDuration = entity_data.get('homingduration', "5")  # Type: string
        instance.HomingRampDown = entity_data.get('homingrampdown', "1.0")  # Type: string
        instance.Gravity = entity_data.get('gravity', "1.0")  # Type: string
        instance.MinRange = parse_source_value(entity_data.get('minrange', 100))  # Type: integer
        instance.MaxRange = parse_source_value(entity_data.get('maxrange', 2048))  # Type: integer
        instance.SpinMagnitude = entity_data.get('spinmagnitude', None)  # Type: string
        instance.SpinSpeed = entity_data.get('spinspeed', None)  # Type: string
        instance.Damage = entity_data.get('damage', "50")  # Type: string
        instance.DamageRadius = entity_data.get('damageradius', "200")  # Type: string


class npc_hunter(BaseNPC):
    model = "models/hunter.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.FollowTarget = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.FollowTarget = entity_data.get('followtarget', None)  # Type: target_destination


class npc_hunter_maker(npc_template_maker):
    icon_sprite = "editor/npc_maker.vmt"
    def __init__(self):
        super(npc_template_maker).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        npc_template_maker.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class npc_advisor(BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        self.model = "models/advisor.mdl"  # Type: studio
        self.levitationarea = None  # Type: string
        self.levitategoal_bottom = None  # Type: target_destination
        self.levitategoal_top = None  # Type: target_destination
        self.staging_ent_names = None  # Type: string
        self.priority_grab_name = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', "models/advisor.mdl")  # Type: studio
        instance.levitationarea = entity_data.get('levitationarea', None)  # Type: string
        instance.levitategoal_bottom = entity_data.get('levitategoal_bottom', None)  # Type: target_destination
        instance.levitategoal_top = entity_data.get('levitategoal_top', None)  # Type: target_destination
        instance.staging_ent_names = entity_data.get('staging_ent_names', None)  # Type: string
        instance.priority_grab_name = entity_data.get('priority_grab_name', None)  # Type: string


class env_sporeexplosion(Parentname, EnableDisable, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.spawnrate = 25  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.spawnrate = float(entity_data.get('spawnrate', 25))  # Type: float


class env_gunfire(Parentname, EnableDisable, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
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
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
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


class env_headcrabcanister(Parentname, Targetname, Angles):
    model = "models/props_combine/headcrabcannister01b.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.HeadcrabType = None  # Type: choices
        self.HeadcrabCount = 6  # Type: integer
        self.FlightSpeed = 3000  # Type: float
        self.FlightTime = 5  # Type: float
        self.StartingHeight = None  # Type: float
        self.MinSkyboxRefireTime = None  # Type: float
        self.MaxSkyboxRefireTime = None  # Type: float
        self.SkyboxCannisterCount = 1  # Type: integer
        self.Damage = 150  # Type: float
        self.DamageRadius = 750  # Type: float
        self.SmokeLifetime = 30  # Type: float
        self.LaunchPositionName = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.HeadcrabType = entity_data.get('headcrabtype', None)  # Type: choices
        instance.HeadcrabCount = parse_source_value(entity_data.get('headcrabcount', 6))  # Type: integer
        instance.FlightSpeed = float(entity_data.get('flightspeed', 3000))  # Type: float
        instance.FlightTime = float(entity_data.get('flighttime', 5))  # Type: float
        instance.StartingHeight = float(entity_data.get('startingheight', 0))  # Type: float
        instance.MinSkyboxRefireTime = float(entity_data.get('minskyboxrefiretime', 0))  # Type: float
        instance.MaxSkyboxRefireTime = float(entity_data.get('maxskyboxrefiretime', 0))  # Type: float
        instance.SkyboxCannisterCount = parse_source_value(entity_data.get('skyboxcannistercount', 1))  # Type: integer
        instance.Damage = float(entity_data.get('damage', 150))  # Type: float
        instance.DamageRadius = float(entity_data.get('damageradius', 750))  # Type: float
        instance.SmokeLifetime = float(entity_data.get('smokelifetime', 30))  # Type: float
        instance.LaunchPositionName = entity_data.get('launchpositionname', None)  # Type: target_destination


class npc_vortigaunt(TalkNPC, PlayerCompanion):
    def __init__(self):
        super(BaseNPC).__init__()
        super(TalkNPC).__init__()
        super(PlayerCompanion).__init__()
        self.model = "models/vortigaunt.mdl"  # Type: studio
        self.ArmorRechargeEnabled = "CHOICES NOT SUPPORTED"  # Type: choices
        self.HealthRegenerateEnabled = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TalkNPC.from_dict(instance, entity_data)
        PlayerCompanion.from_dict(instance, entity_data)
        BaseNPC.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', "models/vortigaunt.mdl")  # Type: studio
        instance.ArmorRechargeEnabled = entity_data.get('armorrechargeenabled', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.HealthRegenerateEnabled = entity_data.get('healthregenerateenabled', None)  # Type: choices


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


class npc_strider(BaseNPC):
    def __init__(self):
        super(BaseNPC).__init__()
        self.model = "CHOICES NOT SUPPORTED"  # Type: choices
        self.disablephysics = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.disablephysics = entity_data.get('disablephysics', None)  # Type: choices


class npc_barnacle(BaseNPC, BaseFadeProp):
    model = "models/Barnacle.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        super(BaseFadeProp).__init__()
        self.RestDist = 16  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        instance.RestDist = float(entity_data.get('restdist', 16))  # Type: float


class npc_combinegunship(BaseHelicopter):
    model = "models/gunship.mdl"
    def __init__(self):
        super(BaseHelicopter).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseHelicopter.from_dict(instance, entity_data)


class info_target_helicopter_crash(Parentname, Targetname):
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


class info_target_gunshipcrash(Parentname, Targetname):
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


class npc_combinedropship(BaseHelicopter):
    model = "models/combine_dropship.mdl"
    def __init__(self):
        super(BaseHelicopter).__init__()
        self.LandTarget = None  # Type: target_destination
        self.GunRange = 2048  # Type: float
        self.RollermineTemplate = None  # Type: target_destination
        self.NPCTemplate = None  # Type: target_destination
        self.NPCTemplate2 = None  # Type: target_destination
        self.NPCTemplate3 = None  # Type: target_destination
        self.NPCTemplate4 = None  # Type: target_destination
        self.NPCTemplate5 = None  # Type: target_destination
        self.NPCTemplate6 = None  # Type: target_destination
        self.Dustoff1 = None  # Type: target_destination
        self.Dustoff2 = None  # Type: target_destination
        self.Dustoff3 = None  # Type: target_destination
        self.Dustoff4 = None  # Type: target_destination
        self.Dustoff5 = None  # Type: target_destination
        self.Dustoff6 = None  # Type: target_destination
        self.APCVehicleName = None  # Type: target_destination
        self.Invulnerable = None  # Type: choices
        self.CrateType = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseHelicopter.from_dict(instance, entity_data)
        instance.LandTarget = entity_data.get('landtarget', None)  # Type: target_destination
        instance.GunRange = float(entity_data.get('gunrange', 2048))  # Type: float
        instance.RollermineTemplate = entity_data.get('rollerminetemplate', None)  # Type: target_destination
        instance.NPCTemplate = entity_data.get('npctemplate', None)  # Type: target_destination
        instance.NPCTemplate2 = entity_data.get('npctemplate2', None)  # Type: target_destination
        instance.NPCTemplate3 = entity_data.get('npctemplate3', None)  # Type: target_destination
        instance.NPCTemplate4 = entity_data.get('npctemplate4', None)  # Type: target_destination
        instance.NPCTemplate5 = entity_data.get('npctemplate5', None)  # Type: target_destination
        instance.NPCTemplate6 = entity_data.get('npctemplate6', None)  # Type: target_destination
        instance.Dustoff1 = entity_data.get('dustoff1', None)  # Type: target_destination
        instance.Dustoff2 = entity_data.get('dustoff2', None)  # Type: target_destination
        instance.Dustoff3 = entity_data.get('dustoff3', None)  # Type: target_destination
        instance.Dustoff4 = entity_data.get('dustoff4', None)  # Type: target_destination
        instance.Dustoff5 = entity_data.get('dustoff5', None)  # Type: target_destination
        instance.Dustoff6 = entity_data.get('dustoff6', None)  # Type: target_destination
        instance.APCVehicleName = entity_data.get('apcvehiclename', None)  # Type: target_destination
        instance.Invulnerable = entity_data.get('invulnerable', None)  # Type: choices
        instance.CrateType = entity_data.get('cratetype', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_helicopter(BaseHelicopter):
    model = "models/combine_helicopter.mdl"
    def __init__(self):
        super(BaseHelicopter).__init__()
        self.GracePeriod = 2.0  # Type: float
        self.PatrolSpeed = 0  # Type: float
        self.noncombat = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseHelicopter.from_dict(instance, entity_data)
        instance.GracePeriod = float(entity_data.get('graceperiod', 2.0))  # Type: float
        instance.PatrolSpeed = float(entity_data.get('patrolspeed', 0))  # Type: float
        instance.noncombat = entity_data.get('noncombat', "CHOICES NOT SUPPORTED")  # Type: choices


class grenade_helicopter(Parentname, Targetname):
    model = "models/combine_helicopter/helicopter_bomb01.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class npc_heli_avoidsphere(Parentname, Targetname):
    icon_sprite = "editor/env_firesource"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.radius = 128  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.radius = float(entity_data.get('radius', 128))  # Type: float


class npc_heli_avoidbox(Origin, Parentname, Angles):
    def __init__(self):
        super(Origin).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Origin.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)


class npc_heli_nobomb(Origin, Parentname, Angles):
    def __init__(self):
        super(Origin).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Origin.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)


class npc_fastzombie(BaseNPC):
    model = "models/Zombie/fast.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_fastzombie_torso(BaseNPC):
    model = "models/Zombie/Fast_torso.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_zombie(BaseNPC):
    model = "models/Zombie/Classic.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_zombie_torso(BaseNPC):
    model = "models/Zombie/Classic_torso.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_zombine(BaseNPC):
    model = "models/Zombie/zombie_soldier.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_poisonzombie(BaseNPC):
    model = "models/Zombie/Poison.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.crabcount = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.crabcount = entity_data.get('crabcount', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_cscanner(BaseNPC):
    model = "models/combine_scanner.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.spotlightlength = 500  # Type: integer
        self.spotlightwidth = 50  # Type: integer
        self.spotlightdisabled = None  # Type: choices
        self.ShouldInspect = "CHOICES NOT SUPPORTED"  # Type: choices
        self.OnlyInspectPlayers = None  # Type: choices
        self.NeverInspectPlayers = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.spotlightlength = parse_source_value(entity_data.get('spotlightlength', 500))  # Type: integer
        instance.spotlightwidth = parse_source_value(entity_data.get('spotlightwidth', 50))  # Type: integer
        instance.spotlightdisabled = entity_data.get('spotlightdisabled', None)  # Type: choices
        instance.ShouldInspect = entity_data.get('shouldinspect', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.OnlyInspectPlayers = entity_data.get('onlyinspectplayers', None)  # Type: choices
        instance.NeverInspectPlayers = entity_data.get('neverinspectplayers', None)  # Type: choices


class npc_clawscanner(BaseNPC):
    model = "models/shield_scanner.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        self.spotlightlength = 500  # Type: integer
        self.spotlightwidth = 50  # Type: integer
        self.spotlightdisabled = None  # Type: choices
        self.ShouldInspect = "CHOICES NOT SUPPORTED"  # Type: choices
        self.OnlyInspectPlayers = None  # Type: choices
        self.NeverInspectPlayers = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        instance.spotlightlength = parse_source_value(entity_data.get('spotlightlength', 500))  # Type: integer
        instance.spotlightwidth = parse_source_value(entity_data.get('spotlightwidth', 50))  # Type: integer
        instance.spotlightdisabled = entity_data.get('spotlightdisabled', None)  # Type: choices
        instance.ShouldInspect = entity_data.get('shouldinspect', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.OnlyInspectPlayers = entity_data.get('onlyinspectplayers', None)  # Type: choices
        instance.NeverInspectPlayers = entity_data.get('neverinspectplayers', None)  # Type: choices


class npc_manhack(BaseNPC, AlyxInteractable):
    model = "models/manhack.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        super(AlyxInteractable).__init__()
        self.ignoreclipbrushes = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        AlyxInteractable.from_dict(instance, entity_data)
        instance.ignoreclipbrushes = entity_data.get('ignoreclipbrushes', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_mortarsynth(BaseNPC):
    model = "models/mortarsynth.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_metropolice(RappelNPC):
    model = "models/Police.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        super(RappelNPC).__init__()
        self.additionalequipment = "CHOICES NOT SUPPORTED"  # Type: choices
        self.manhacks = None  # Type: choices
        self.weapondrawn = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        RappelNPC.from_dict(instance, entity_data)
        BaseNPC.from_dict(instance, entity_data)
        instance.additionalequipment = entity_data.get('additionalequipment', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.manhacks = entity_data.get('manhacks', None)  # Type: choices
        instance.weapondrawn = entity_data.get('weapondrawn', None)  # Type: choices


class npc_crabsynth(BaseNPC):
    model = "models/Synth.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


class npc_monk(TalkNPC):
    model = "models/Monk.mdl"
    def __init__(self):
        super(TalkNPC).__init__()
        self.additionalequipment = "CHOICES NOT SUPPORTED"  # Type: choices
        self.HasGun = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TalkNPC.from_dict(instance, entity_data)
        instance.additionalequipment = entity_data.get('additionalequipment', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.HasGun = entity_data.get('hasgun', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_alyx(TalkNPC, PlayerCompanion, Parentname):
    def __init__(self):
        super(TalkNPC).__init__()
        super(PlayerCompanion).__init__()
        super(Parentname).__init__()
        self.model = "models/alyx.mdl"  # Type: studio
        self.additionalequipment = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ShouldHaveEMP = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TalkNPC.from_dict(instance, entity_data)
        PlayerCompanion.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', "models/alyx.mdl")  # Type: studio
        instance.additionalequipment = entity_data.get('additionalequipment', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ShouldHaveEMP = entity_data.get('shouldhaveemp', "CHOICES NOT SUPPORTED")  # Type: choices


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


class npc_kleiner(TalkNPC):
    def __init__(self):
        super(TalkNPC).__init__()
        self.model = "models/kleiner.mdl"  # Type: studio

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TalkNPC.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', "models/kleiner.mdl")  # Type: studio


class npc_eli(TalkNPC, Parentname):
    def __init__(self):
        super(TalkNPC).__init__()
        super(Parentname).__init__()
        self.model = "models/eli.mdl"  # Type: studio

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TalkNPC.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', "models/eli.mdl")  # Type: studio


class npc_magnusson(TalkNPC):
    def __init__(self):
        super(TalkNPC).__init__()
        self.model = "models/magnusson.mdl"  # Type: studio

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TalkNPC.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', "models/magnusson.mdl")  # Type: studio


class npc_breen(TalkNPC):
    def __init__(self):
        super(TalkNPC).__init__()
        self.model = "models/breen.mdl"  # Type: studio

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TalkNPC.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', "models/breen.mdl")  # Type: studio


class npc_mossman(TalkNPC):
    model = "models/mossman.mdl"
    def __init__(self):
        super(TalkNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TalkNPC.from_dict(instance, entity_data)


class npc_gman(TalkNPC):
    model = "models/gman.mdl"
    def __init__(self):
        super(TalkNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TalkNPC.from_dict(instance, entity_data)


class npc_dog(BaseNPC):
    model = "models/dog.mdl"
    def __init__(self):
        super(BaseNPC).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)


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
        RenderFields.from_dict(instance, entity_data)
        BaseNPC.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', None)  # Type: studio
        instance.body = parse_source_value(entity_data.get('body', 0))  # Type: integer


class generic_actor(Parentname, BaseNPC):
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
        Parentname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        BaseNPC.from_dict(instance, entity_data)
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
        RenderFields.from_dict(instance, entity_data)
        BaseNPC.from_dict(instance, entity_data)
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


class npc_antlion_template_maker(BaseNPCMaker):
    def __init__(self):
        super(BaseNPCMaker).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.TemplateName = None  # Type: target_destination
        self.spawngroup = None  # Type: string
        self.spawnradius = 512  # Type: float
        self.spawntarget = None  # Type: string
        self.fighttarget = None  # Type: string
        self.followtarget = None  # Type: string
        self.vehicledistance = 1  # Type: float
        self.workerspawnrate = 0  # Type: float
        self.ignorebugbait = "CHOICES NOT SUPPORTED"  # Type: choices
        self.pool_start = None  # Type: integer
        self.pool_max = None  # Type: integer
        self.pool_regen_amount = None  # Type: integer
        self.pool_regen_time = None  # Type: float
        self.createspores = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        BaseNPCMaker.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.TemplateName = entity_data.get('templatename', None)  # Type: target_destination
        instance.spawngroup = entity_data.get('spawngroup', None)  # Type: string
        instance.spawnradius = float(entity_data.get('spawnradius', 512))  # Type: float
        instance.spawntarget = entity_data.get('spawntarget', None)  # Type: string
        instance.fighttarget = entity_data.get('fighttarget', None)  # Type: string
        instance.followtarget = entity_data.get('followtarget', None)  # Type: string
        instance.vehicledistance = float(entity_data.get('vehicledistance', 1))  # Type: float
        instance.workerspawnrate = float(entity_data.get('workerspawnrate', 0))  # Type: float
        instance.ignorebugbait = entity_data.get('ignorebugbait', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.pool_start = parse_source_value(entity_data.get('pool_start', 0))  # Type: integer
        instance.pool_max = parse_source_value(entity_data.get('pool_max', 0))  # Type: integer
        instance.pool_regen_amount = parse_source_value(entity_data.get('pool_regen_amount', 0))  # Type: integer
        instance.pool_regen_time = float(entity_data.get('pool_regen_time', 0))  # Type: float
        instance.createspores = entity_data.get('createspores', "CHOICES NOT SUPPORTED")  # Type: choices


class point_antlion_repellant(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.repelradius = 512  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.repelradius = float(entity_data.get('repelradius', 512))  # Type: float


class player_control(Targetname):
    def __init__(self):
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)


class BaseScripted(Parentname, Targetname, Angles):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.m_iszEntity = None  # Type: target_destination
        self.m_iszIdle = None  # Type: string
        self.m_iszEntry = None  # Type: string
        self.m_iszPlay = None  # Type: string
        self.m_iszPostIdle = None  # Type: string
        self.m_iszCustomMove = None  # Type: string
        self.m_bLoopActionSequence = None  # Type: choices
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
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.m_iszEntity = entity_data.get('m_iszentity', None)  # Type: target_destination
        instance.m_iszIdle = entity_data.get('m_iszidle', None)  # Type: string
        instance.m_iszEntry = entity_data.get('m_iszentry', None)  # Type: string
        instance.m_iszPlay = entity_data.get('m_iszplay', None)  # Type: string
        instance.m_iszPostIdle = entity_data.get('m_iszpostidle', None)  # Type: string
        instance.m_iszCustomMove = entity_data.get('m_iszcustommove', None)  # Type: string
        instance.m_bLoopActionSequence = entity_data.get('m_bloopactionsequence', None)  # Type: choices
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


class ai_battle_line(Angles, Parentname, Targetname):
    model = "models/pigeon.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.actor = None  # Type: target_name_or_class
        self.Active = None  # Type: choices
        self.Strict = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.actor = entity_data.get('actor', None)  # Type: target_name_or_class
        instance.Active = entity_data.get('active', None)  # Type: choices
        instance.Strict = entity_data.get('strict', "CHOICES NOT SUPPORTED")  # Type: choices


class ai_goal_standoff(Targetname):
    icon_sprite = "editor/ai_goal_standoff.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.actor = None  # Type: target_name_or_class
        self.SearchType = None  # Type: choices
        self.StartActive = None  # Type: choices
        self.HintGroupChangeReaction = "CHOICES NOT SUPPORTED"  # Type: choices
        self.Aggressiveness = "CHOICES NOT SUPPORTED"  # Type: choices
        self.PlayerBattleline = "CHOICES NOT SUPPORTED"  # Type: choices
        self.StayAtCover = None  # Type: choices
        self.AbandonIfEnemyHides = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.actor = entity_data.get('actor', None)  # Type: target_name_or_class
        instance.SearchType = entity_data.get('searchtype', None)  # Type: choices
        instance.StartActive = entity_data.get('startactive', None)  # Type: choices
        instance.HintGroupChangeReaction = entity_data.get('hintgroupchangereaction', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.Aggressiveness = entity_data.get('aggressiveness', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.PlayerBattleline = entity_data.get('playerbattleline', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.StayAtCover = entity_data.get('stayatcover', None)  # Type: choices
        instance.AbandonIfEnemyHides = entity_data.get('abandonifenemyhides', None)  # Type: choices


class ai_goal_police(Parentname, Targetname, Angles):
    icon_sprite = "editor/ai_goal_police.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.policeradius = 512  # Type: float
        self.policetarget = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.policeradius = float(entity_data.get('policeradius', 512))  # Type: float
        instance.policetarget = entity_data.get('policetarget', None)  # Type: string


class assault_rallypoint(Parentname, Targetname, Angles):
    icon_sprite = "editor/assault_rally.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.assaultpoint = None  # Type: target_destination
        self.assaultdelay = None  # Type: float
        self.rallysequence = None  # Type: string
        self.priority = 1  # Type: integer
        self.forcecrouch = None  # Type: choices
        self.urgent = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.assaultpoint = entity_data.get('assaultpoint', None)  # Type: target_destination
        instance.assaultdelay = float(entity_data.get('assaultdelay', 0))  # Type: float
        instance.rallysequence = entity_data.get('rallysequence', None)  # Type: string
        instance.priority = parse_source_value(entity_data.get('priority', 1))  # Type: integer
        instance.forcecrouch = entity_data.get('forcecrouch', None)  # Type: choices
        instance.urgent = entity_data.get('urgent', None)  # Type: choices


class assault_assaultpoint(Parentname, Targetname, Angles):
    icon_sprite = "editor/assault_point.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.assaultgroup = None  # Type: string
        self.nextassaultpoint = None  # Type: target_destination
        self.assaulttimeout = 3.0  # Type: float
        self.clearoncontact = None  # Type: choices
        self.allowdiversion = None  # Type: choices
        self.allowdiversionradius = None  # Type: float
        self.nevertimeout = None  # Type: choices
        self.strict = None  # Type: choices
        self.forcecrouch = None  # Type: choices
        self.urgent = None  # Type: choices
        self.assaulttolerance = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.assaultgroup = entity_data.get('assaultgroup', None)  # Type: string
        instance.nextassaultpoint = entity_data.get('nextassaultpoint', None)  # Type: target_destination
        instance.assaulttimeout = float(entity_data.get('assaulttimeout', 3.0))  # Type: float
        instance.clearoncontact = entity_data.get('clearoncontact', None)  # Type: choices
        instance.allowdiversion = entity_data.get('allowdiversion', None)  # Type: choices
        instance.allowdiversionradius = float(entity_data.get('allowdiversionradius', 0))  # Type: float
        instance.nevertimeout = entity_data.get('nevertimeout', None)  # Type: choices
        instance.strict = entity_data.get('strict', None)  # Type: choices
        instance.forcecrouch = entity_data.get('forcecrouch', None)  # Type: choices
        instance.urgent = entity_data.get('urgent', None)  # Type: choices
        instance.assaulttolerance = entity_data.get('assaulttolerance', "CHOICES NOT SUPPORTED")  # Type: choices


class ai_goal_assault(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.actor = None  # Type: target_name_or_class
        self.rallypoint = None  # Type: target_destination
        self.SearchType = None  # Type: choices
        self.StartActive = None  # Type: choices
        self.AssaultCue = "CHOICES NOT SUPPORTED"  # Type: choices
        self.RallySelectMethod = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.actor = entity_data.get('actor', None)  # Type: target_name_or_class
        instance.rallypoint = entity_data.get('rallypoint', None)  # Type: target_destination
        instance.SearchType = entity_data.get('searchtype', None)  # Type: choices
        instance.StartActive = entity_data.get('startactive', None)  # Type: choices
        instance.AssaultCue = entity_data.get('assaultcue', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.RallySelectMethod = entity_data.get('rallyselectmethod', None)  # Type: choices


class BaseActBusy(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.actor = None  # Type: target_name_or_class
        self.StartActive = None  # Type: choices
        self.SearchType = None  # Type: choices
        self.busysearchrange = 2048  # Type: float
        self.visibleonly = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.actor = entity_data.get('actor', None)  # Type: target_name_or_class
        instance.StartActive = entity_data.get('startactive', None)  # Type: choices
        instance.SearchType = entity_data.get('searchtype', None)  # Type: choices
        instance.busysearchrange = float(entity_data.get('busysearchrange', 2048))  # Type: float
        instance.visibleonly = entity_data.get('visibleonly', None)  # Type: choices


class ai_goal_actbusy(BaseActBusy):
    def __init__(self):
        super(BaseActBusy).__init__()
        self.origin = [0, 0, 0]
        self.seeentity = None  # Type: target_name_or_class
        self.seeentitytimeout = "1"  # Type: string
        self.sightmethod = None  # Type: choices
        self.type = None  # Type: choices
        self.safezone = None  # Type: target_destination
        self.allowteleport = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseActBusy.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.seeentity = entity_data.get('seeentity', None)  # Type: target_name_or_class
        instance.seeentitytimeout = entity_data.get('seeentitytimeout', "1")  # Type: string
        instance.sightmethod = entity_data.get('sightmethod', None)  # Type: choices
        instance.type = entity_data.get('type', None)  # Type: choices
        instance.safezone = entity_data.get('safezone', None)  # Type: target_destination
        instance.allowteleport = entity_data.get('allowteleport', None)  # Type: choices


class ai_goal_actbusy_queue(BaseActBusy):
    def __init__(self):
        super(BaseActBusy).__init__()
        self.origin = [0, 0, 0]
        self.node_exit = None  # Type: target_destination
        self.node01 = None  # Type: target_destination
        self.node02 = None  # Type: target_destination
        self.node03 = None  # Type: target_destination
        self.node04 = None  # Type: target_destination
        self.node05 = None  # Type: target_destination
        self.node06 = None  # Type: target_destination
        self.node07 = None  # Type: target_destination
        self.node08 = None  # Type: target_destination
        self.node09 = None  # Type: target_destination
        self.node10 = None  # Type: target_destination
        self.node11 = None  # Type: target_destination
        self.node12 = None  # Type: target_destination
        self.node13 = None  # Type: target_destination
        self.node14 = None  # Type: target_destination
        self.node15 = None  # Type: target_destination
        self.node16 = None  # Type: target_destination
        self.node17 = None  # Type: target_destination
        self.node18 = None  # Type: target_destination
        self.node19 = None  # Type: target_destination
        self.node20 = None  # Type: target_destination
        self.mustreachfront = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseActBusy.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.node_exit = entity_data.get('node_exit', None)  # Type: target_destination
        instance.node01 = entity_data.get('node01', None)  # Type: target_destination
        instance.node02 = entity_data.get('node02', None)  # Type: target_destination
        instance.node03 = entity_data.get('node03', None)  # Type: target_destination
        instance.node04 = entity_data.get('node04', None)  # Type: target_destination
        instance.node05 = entity_data.get('node05', None)  # Type: target_destination
        instance.node06 = entity_data.get('node06', None)  # Type: target_destination
        instance.node07 = entity_data.get('node07', None)  # Type: target_destination
        instance.node08 = entity_data.get('node08', None)  # Type: target_destination
        instance.node09 = entity_data.get('node09', None)  # Type: target_destination
        instance.node10 = entity_data.get('node10', None)  # Type: target_destination
        instance.node11 = entity_data.get('node11', None)  # Type: target_destination
        instance.node12 = entity_data.get('node12', None)  # Type: target_destination
        instance.node13 = entity_data.get('node13', None)  # Type: target_destination
        instance.node14 = entity_data.get('node14', None)  # Type: target_destination
        instance.node15 = entity_data.get('node15', None)  # Type: target_destination
        instance.node16 = entity_data.get('node16', None)  # Type: target_destination
        instance.node17 = entity_data.get('node17', None)  # Type: target_destination
        instance.node18 = entity_data.get('node18', None)  # Type: target_destination
        instance.node19 = entity_data.get('node19', None)  # Type: target_destination
        instance.node20 = entity_data.get('node20', None)  # Type: target_destination
        instance.mustreachfront = entity_data.get('mustreachfront', None)  # Type: choices


class ai_changetarget(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.m_iszNewTarget = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.m_iszNewTarget = entity_data.get('m_isznewtarget', None)  # Type: string


class ai_npc_eventresponsesystem(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class ai_changehintgroup(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.SearchType = None  # Type: choices
        self.SearchName = None  # Type: string
        self.NewHintGroup = None  # Type: string
        self.Radius = "0.0"  # Type: string
        self.hintlimiting = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.SearchType = entity_data.get('searchtype', None)  # Type: choices
        instance.SearchName = entity_data.get('searchname', None)  # Type: string
        instance.NewHintGroup = entity_data.get('newhintgroup', None)  # Type: string
        instance.Radius = entity_data.get('radius', "0.0")  # Type: string
        instance.hintlimiting = entity_data.get('hintlimiting', None)  # Type: choices


class ai_script_conditions(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.Actor = None  # Type: target_destination
        self.StartDisabled = "CHOICES NOT SUPPORTED"  # Type: choices
        self.MinimumState = "CHOICES NOT SUPPORTED"  # Type: choices
        self.MaximumState = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ScriptStatus = "CHOICES NOT SUPPORTED"  # Type: choices
        self.RequiredTime = None  # Type: float
        self.MinTimeout = None  # Type: float
        self.MaxTimeout = None  # Type: float
        self.ActorSeePlayer = "CHOICES NOT SUPPORTED"  # Type: choices
        self.PlayerActorProximity = None  # Type: float
        self.PlayerActorFOV = 360  # Type: float
        self.PlayerActorFOVTrueCone = None  # Type: choices
        self.PlayerActorLOS = "CHOICES NOT SUPPORTED"  # Type: choices
        self.target = None  # Type: target_destination
        self.ActorSeeTarget = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ActorTargetProximity = None  # Type: float
        self.PlayerTargetProximity = None  # Type: float
        self.PlayerTargetFOV = 360  # Type: float
        self.PlayerTargetFOVTrueCone = None  # Type: choices
        self.PlayerTargetLOS = "CHOICES NOT SUPPORTED"  # Type: choices
        self.PlayerBlockingActor = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ActorInPVS = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ActorInVehicle = "CHOICES NOT SUPPORTED"  # Type: choices
        self.PlayerInVehicle = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Actor = entity_data.get('actor', None)  # Type: target_destination
        instance.StartDisabled = entity_data.get('startdisabled', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.MinimumState = entity_data.get('minimumstate', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.MaximumState = entity_data.get('maximumstate', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ScriptStatus = entity_data.get('scriptstatus', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.RequiredTime = float(entity_data.get('requiredtime', 0))  # Type: float
        instance.MinTimeout = float(entity_data.get('mintimeout', 0))  # Type: float
        instance.MaxTimeout = float(entity_data.get('maxtimeout', 0))  # Type: float
        instance.ActorSeePlayer = entity_data.get('actorseeplayer', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.PlayerActorProximity = float(entity_data.get('playeractorproximity', 0))  # Type: float
        instance.PlayerActorFOV = float(entity_data.get('playeractorfov', 360))  # Type: float
        instance.PlayerActorFOVTrueCone = entity_data.get('playeractorfovtruecone', None)  # Type: choices
        instance.PlayerActorLOS = entity_data.get('playeractorlos', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.ActorSeeTarget = entity_data.get('actorseetarget', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ActorTargetProximity = float(entity_data.get('actortargetproximity', 0))  # Type: float
        instance.PlayerTargetProximity = float(entity_data.get('playertargetproximity', 0))  # Type: float
        instance.PlayerTargetFOV = float(entity_data.get('playertargetfov', 360))  # Type: float
        instance.PlayerTargetFOVTrueCone = entity_data.get('playertargetfovtruecone', None)  # Type: choices
        instance.PlayerTargetLOS = entity_data.get('playertargetlos', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.PlayerBlockingActor = entity_data.get('playerblockingactor', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ActorInPVS = entity_data.get('actorinpvs', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ActorInVehicle = entity_data.get('actorinvehicle', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.PlayerInVehicle = entity_data.get('playerinvehicle', "CHOICES NOT SUPPORTED")  # Type: choices


class scripted_sequence(BaseScripted, DXLevelChoice):
    model = "models/editor/scriptedsequence.mdl"
    def __init__(self):
        super(BaseScripted).__init__()
        super(DXLevelChoice).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.onplayerdeath = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseScripted.from_dict(instance, entity_data)
        DXLevelChoice.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.onplayerdeath = entity_data.get('onplayerdeath', None)  # Type: choices


class aiscripted_schedule(Targetname):
    icon_sprite = "editor/aiscripted_schedule.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.m_iszEntity = None  # Type: target_destination
        self.m_flRadius = None  # Type: integer
        self.graball = None  # Type: choices
        self.forcestate = None  # Type: choices
        self.schedule = "CHOICES NOT SUPPORTED"  # Type: choices
        self.interruptability = None  # Type: choices
        self.goalent = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.m_iszEntity = entity_data.get('m_iszentity', None)  # Type: target_destination
        instance.m_flRadius = parse_source_value(entity_data.get('m_flradius', 0))  # Type: integer
        instance.graball = entity_data.get('graball', None)  # Type: choices
        instance.forcestate = entity_data.get('forcestate', None)  # Type: choices
        instance.schedule = entity_data.get('schedule', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.interruptability = entity_data.get('interruptability', None)  # Type: choices
        instance.goalent = entity_data.get('goalent', None)  # Type: target_destination


class ai_citizen_response_system(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class func_healthcharger(EnableDisable, Global, Origin, Parentname):
    def __init__(self):
        super(EnableDisable).__init__()
        super(Global).__init__()
        super(Origin).__init__()
        super(Parentname).__init__()
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class func_recharge(Parentname, Origin, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class func_vehicleclip(Parentname, Global, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Global).__init__()
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)


class func_lookdoor(func_movelinear):
    def __init__(self):
        super(func_movelinear).__init__()
        self.ProximityDistance = "0.0"  # Type: string
        self.ProximityOffset = "0.0"  # Type: string
        self.FieldOfView = "0.0"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        func_movelinear.from_dict(instance, entity_data)
        instance.ProximityDistance = entity_data.get('proximitydistance', "0.0")  # Type: string
        instance.ProximityOffset = entity_data.get('proximityoffset', "0.0")  # Type: string
        instance.FieldOfView = entity_data.get('fieldofview', "0.0")  # Type: string


class trigger_waterydeath(Trigger):
    def __init__(self):
        super(Trigger).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)


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


class BaseTank(Parentname, Angles, Origin, Targetname, Shadow, RenderFields, Global):
    def __init__(self):
        super(RenderFields).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        super(Shadow).__init__()
        super(Global).__init__()
        self.control_volume = None  # Type: target_destination
        self.master = None  # Type: string
        self.yawrate = "30"  # Type: string
        self.yawrange = "180"  # Type: string
        self.yawtolerance = "15"  # Type: string
        self.pitchrate = "0"  # Type: string
        self.pitchrange = "0"  # Type: string
        self.pitchtolerance = "5"  # Type: string
        self.barrel = "0"  # Type: string
        self.barrely = "0"  # Type: string
        self.barrelz = "0"  # Type: string
        self.spritesmoke = None  # Type: sprite
        self.spriteflash = None  # Type: sprite
        self.spritescale = "1"  # Type: string
        self.rotatestartsound = None  # Type: sound
        self.rotatesound = None  # Type: sound
        self.rotatestopsound = None  # Type: sound
        self.firerate = "1"  # Type: string
        self.bullet_damage = "0"  # Type: string
        self.bullet_damage_vs_player = "0"  # Type: string
        self.persistence = "1"  # Type: string
        self.persistence2 = "0"  # Type: string
        self.firespread = None  # Type: choices
        self.minRange = "0"  # Type: string
        self.maxRange = "0"  # Type: string
        self._minlight = None  # Type: string
        self.gun_base_attach = None  # Type: string
        self.gun_barrel_attach = None  # Type: string
        self.gun_yaw_pose_param = None  # Type: string
        self.gun_yaw_pose_center = 0  # Type: float
        self.gun_pitch_pose_param = None  # Type: string
        self.gun_pitch_pose_center = 0  # Type: float
        self.ammo_count = -1  # Type: integer
        self.LeadTarget = "CHOICES NOT SUPPORTED"  # Type: choices
        self.npc_man_point = None  # Type: target_destination
        self.playergraceperiod = 0  # Type: float
        self.ignoregraceupto = 768  # Type: float
        self.playerlocktimebeforefire = 0  # Type: float
        self.effecthandling = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        instance.control_volume = entity_data.get('control_volume', None)  # Type: target_destination
        instance.master = entity_data.get('master', None)  # Type: string
        instance.yawrate = entity_data.get('yawrate', "30")  # Type: string
        instance.yawrange = entity_data.get('yawrange', "180")  # Type: string
        instance.yawtolerance = entity_data.get('yawtolerance', "15")  # Type: string
        instance.pitchrate = entity_data.get('pitchrate', "0")  # Type: string
        instance.pitchrange = entity_data.get('pitchrange', "0")  # Type: string
        instance.pitchtolerance = entity_data.get('pitchtolerance', "5")  # Type: string
        instance.barrel = entity_data.get('barrel', "0")  # Type: string
        instance.barrely = entity_data.get('barrely', "0")  # Type: string
        instance.barrelz = entity_data.get('barrelz', "0")  # Type: string
        instance.spritesmoke = entity_data.get('spritesmoke', None)  # Type: sprite
        instance.spriteflash = entity_data.get('spriteflash', None)  # Type: sprite
        instance.spritescale = entity_data.get('spritescale', "1")  # Type: string
        instance.rotatestartsound = entity_data.get('rotatestartsound', None)  # Type: sound
        instance.rotatesound = entity_data.get('rotatesound', None)  # Type: sound
        instance.rotatestopsound = entity_data.get('rotatestopsound', None)  # Type: sound
        instance.firerate = entity_data.get('firerate', "1")  # Type: string
        instance.bullet_damage = entity_data.get('bullet_damage', "0")  # Type: string
        instance.bullet_damage_vs_player = entity_data.get('bullet_damage_vs_player', "0")  # Type: string
        instance.persistence = entity_data.get('persistence', "1")  # Type: string
        instance.persistence2 = entity_data.get('persistence2', "0")  # Type: string
        instance.firespread = entity_data.get('firespread', None)  # Type: choices
        instance.minRange = entity_data.get('minrange', "0")  # Type: string
        instance.maxRange = entity_data.get('maxrange', "0")  # Type: string
        instance._minlight = entity_data.get('_minlight', None)  # Type: string
        instance.gun_base_attach = entity_data.get('gun_base_attach', None)  # Type: string
        instance.gun_barrel_attach = entity_data.get('gun_barrel_attach', None)  # Type: string
        instance.gun_yaw_pose_param = entity_data.get('gun_yaw_pose_param', None)  # Type: string
        instance.gun_yaw_pose_center = float(entity_data.get('gun_yaw_pose_center', 0))  # Type: float
        instance.gun_pitch_pose_param = entity_data.get('gun_pitch_pose_param', None)  # Type: string
        instance.gun_pitch_pose_center = float(entity_data.get('gun_pitch_pose_center', 0))  # Type: float
        instance.ammo_count = parse_source_value(entity_data.get('ammo_count', -1))  # Type: integer
        instance.LeadTarget = entity_data.get('leadtarget', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.npc_man_point = entity_data.get('npc_man_point', None)  # Type: target_destination
        instance.playergraceperiod = float(entity_data.get('playergraceperiod', 0))  # Type: float
        instance.ignoregraceupto = float(entity_data.get('ignoregraceupto', 768))  # Type: float
        instance.playerlocktimebeforefire = float(entity_data.get('playerlocktimebeforefire', 0))  # Type: float
        instance.effecthandling = entity_data.get('effecthandling', None)  # Type: choices


class func_tank(BaseTank):
    def __init__(self):
        super(BaseTank).__init__()
        self.ammotype = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseTank.from_dict(instance, entity_data)
        instance.ammotype = entity_data.get('ammotype', None)  # Type: choices


class func_tankpulselaser(BaseTank):
    def __init__(self):
        super(BaseTank).__init__()
        self.PulseSpeed = 1000  # Type: float
        self.PulseColor = [255, 0, 0]  # Type: color255
        self.PulseWidth = 20  # Type: float
        self.PulseLife = 2  # Type: float
        self.PulseLag = 0.05  # Type: float
        self.PulseFireSound = None  # Type: sound

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseTank.from_dict(instance, entity_data)
        instance.PulseSpeed = float(entity_data.get('pulsespeed', 1000))  # Type: float
        instance.PulseColor = parse_int_vector(entity_data.get('pulsecolor', "255 0 0"))  # Type: color255
        instance.PulseWidth = float(entity_data.get('pulsewidth', 20))  # Type: float
        instance.PulseLife = float(entity_data.get('pulselife', 2))  # Type: float
        instance.PulseLag = float(entity_data.get('pulselag', 0.05))  # Type: float
        instance.PulseFireSound = entity_data.get('pulsefiresound', None)  # Type: sound


class func_tanklaser(BaseTank):
    def __init__(self):
        super(BaseTank).__init__()
        self.laserentity = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseTank.from_dict(instance, entity_data)
        instance.laserentity = entity_data.get('laserentity', None)  # Type: target_destination


class func_tankrocket(BaseTank):
    def __init__(self):
        super(BaseTank).__init__()
        self.rocketspeed = 800  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseTank.from_dict(instance, entity_data)
        instance.rocketspeed = float(entity_data.get('rocketspeed', 800))  # Type: float


class func_tankairboatgun(BaseTank):
    def __init__(self):
        super(BaseTank).__init__()
        self.airboat_gun_model = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseTank.from_dict(instance, entity_data)
        instance.airboat_gun_model = entity_data.get('airboat_gun_model', None)  # Type: target_destination


class func_tankapcrocket(BaseTank):
    def __init__(self):
        super(BaseTank).__init__()
        self.rocketspeed = 800  # Type: float
        self.burstcount = 10  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseTank.from_dict(instance, entity_data)
        instance.rocketspeed = float(entity_data.get('rocketspeed', 800))  # Type: float
        instance.burstcount = parse_source_value(entity_data.get('burstcount', 10))  # Type: integer


class func_tankmortar(BaseTank):
    def __init__(self):
        super(BaseTank).__init__()
        self.iMagnitude = 100  # Type: integer
        self.firedelay = "2"  # Type: string
        self.firestartsound = None  # Type: sound
        self.fireendsound = None  # Type: sound
        self.incomingsound = None  # Type: sound
        self.warningtime = 1  # Type: float
        self.firevariance = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseTank.from_dict(instance, entity_data)
        instance.iMagnitude = parse_source_value(entity_data.get('imagnitude', 100))  # Type: integer
        instance.firedelay = entity_data.get('firedelay', "2")  # Type: string
        instance.firestartsound = entity_data.get('firestartsound', None)  # Type: sound
        instance.fireendsound = entity_data.get('fireendsound', None)  # Type: sound
        instance.incomingsound = entity_data.get('incomingsound', None)  # Type: sound
        instance.warningtime = float(entity_data.get('warningtime', 1))  # Type: float
        instance.firevariance = float(entity_data.get('firevariance', 0))  # Type: float


class func_tankphyscannister(BaseTank):
    def __init__(self):
        super(BaseTank).__init__()
        self.barrel_volume = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseTank.from_dict(instance, entity_data)
        instance.barrel_volume = entity_data.get('barrel_volume', None)  # Type: target_destination


class func_tank_combine_cannon(BaseTank):
    def __init__(self):
        super(BaseTank).__init__()
        self.ammotype = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseTank.from_dict(instance, entity_data)
        instance.ammotype = entity_data.get('ammotype', None)  # Type: choices


class Item(Angles, Targetname, Shadow):
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        super(Shadow).__init__()
        self.fademindist = -1  # Type: float
        self.fademaxdist = None  # Type: float
        self.fadescale = 1  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        instance.fademindist = float(entity_data.get('fademindist', -1))  # Type: float
        instance.fademaxdist = float(entity_data.get('fademaxdist', 0))  # Type: float
        instance.fadescale = float(entity_data.get('fadescale', 1))  # Type: float


class item_dynamic_resupply(Item):
    model = "models/items/healthkit.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]
        self.DesiredHealth = 1  # Type: float
        self.DesiredArmor = 0.3  # Type: float
        self.DesiredAmmoPistol = 0.5  # Type: float
        self.DesiredAmmoSMG1 = 0.5  # Type: float
        self.DesiredAmmoSMG1_Grenade = 0.1  # Type: float
        self.DesiredAmmoAR2 = 0.4  # Type: float
        self.DesiredAmmoBuckshot = 0.5  # Type: float
        self.DesiredAmmoRPG_Round = 0  # Type: float
        self.DesiredAmmoGrenade = 0.1  # Type: float
        self.DesiredAmmo357 = 0  # Type: float
        self.DesiredAmmoCrossbow = 0  # Type: float
        self.DesiredAmmoAR2_AltFire = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.DesiredHealth = float(entity_data.get('desiredhealth', 1))  # Type: float
        instance.DesiredArmor = float(entity_data.get('desiredarmor', 0.3))  # Type: float
        instance.DesiredAmmoPistol = float(entity_data.get('desiredammopistol', 0.5))  # Type: float
        instance.DesiredAmmoSMG1 = float(entity_data.get('desiredammosmg1', 0.5))  # Type: float
        instance.DesiredAmmoSMG1_Grenade = float(entity_data.get('desiredammosmg1_grenade', 0.1))  # Type: float
        instance.DesiredAmmoAR2 = float(entity_data.get('desiredammoar2', 0.4))  # Type: float
        instance.DesiredAmmoBuckshot = float(entity_data.get('desiredammobuckshot', 0.5))  # Type: float
        instance.DesiredAmmoRPG_Round = float(entity_data.get('desiredammorpg_round', 0))  # Type: float
        instance.DesiredAmmoGrenade = float(entity_data.get('desiredammogrenade', 0.1))  # Type: float
        instance.DesiredAmmo357 = float(entity_data.get('desiredammo357', 0))  # Type: float
        instance.DesiredAmmoCrossbow = float(entity_data.get('desiredammocrossbow', 0))  # Type: float
        instance.DesiredAmmoAR2_AltFire = float(entity_data.get('desiredammoar2_altfire', 0))  # Type: float


class item_ammo_pistol(Item):
    model = "models/items/boxsrounds.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammo_pistol_large(Item):
    model = "models/items/largeBoxSRounds.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammo_smg1(Item):
    model = "models/items/BoxMRounds.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammo_smg1_large(Item):
    model = "models/items/LargeBoxMRounds.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammo_ar2(Item):
    model = "models/items/BoxBRounds.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammo_ar2_large(Item):
    model = "models/items/LargeBoxBRounds.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammo_357(Item):
    model = "models/items/357ammo.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammo_357_large(Item):
    model = "models/items/357ammobox.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammo_crossbow(Item):
    model = "models/items/CrossbowRounds.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_box_buckshot(Item):
    model = "models/items/BoxBuckshot.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_rpg_round(Item):
    model = "models/weapons/w_missile_closed.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammo_smg1_grenade(Item):
    model = "models/items/AR2_Grenade.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_battery(Item):
    model = "models/items/battery.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_healthkit(Item):
    model = "models/items/healthkit.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_healthvial(Item):
    model = "models/healthvial.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammo_ar2_altfire(Item):
    model = "models/items/combine_rifle_ammo01.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_suit(Item):
    model = "models/items/hevsuit.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammo_crate(Angles, Targetname, BaseFadeProp):
    model = "models/items/ammocrate_rockets.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        super(BaseFadeProp).__init__()
        self.origin = [0, 0, 0]
        self.AmmoType = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.AmmoType = entity_data.get('ammotype', None)  # Type: choices


class item_item_crate(BasePropPhysics):
    model = "models/items/item_item_crate.mdl"
    def __init__(self):
        super(BasePropPhysics).__init__()
        super(Angles).__init__()
        super(DamageFilter).__init__()
        super(BaseFadeProp).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.CrateType = None  # Type: choices
        self.CrateAppearance = None  # Type: choices
        self.ItemClass = item_dynamic_resupply  # Type: pointentityclass
        self.ItemCount = 1  # Type: integer
        self.SpecificResupply = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        BasePropPhysics.from_dict(instance, entity_data)
        DamageFilter.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.CrateType = entity_data.get('cratetype', None)  # Type: choices
        instance.CrateAppearance = entity_data.get('crateappearance', None)  # Type: choices
        instance.ItemClass = entity_data.get('itemclass', None)  # Type: pointentityclass
        instance.ItemCount = parse_source_value(entity_data.get('itemcount', 1))  # Type: integer
        instance.SpecificResupply = entity_data.get('specificresupply', None)  # Type: target_destination


class item_healthcharger(Angles, Targetname, BaseFadeProp):
    model = "models/props_combine/health_charger001.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        super(BaseFadeProp).__init__()
        self.origin = [0, 0, 0]
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class item_suitcharger(Angles, Targetname, BaseFadeProp):
    model = "models/props_combine/suit_charger001.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        super(BaseFadeProp).__init__()
        self.origin = [0, 0, 0]
        self._minlight = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance._minlight = entity_data.get('_minlight', None)  # Type: string


class Weapon(Angles, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        self.fademindist = -1  # Type: float
        self.fademaxdist = None  # Type: float
        self.fadescale = 1  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.fademindist = float(entity_data.get('fademindist', -1))  # Type: float
        instance.fademaxdist = float(entity_data.get('fademaxdist', 0))  # Type: float
        instance.fadescale = float(entity_data.get('fadescale', 1))  # Type: float


class weapon_crowbar(Weapon):
    model = "models/weapons/w_crowbar.mdl"
    def __init__(self):
        super(Weapon).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Weapon.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_stunstick(Weapon):
    model = "models/weapons/w_stunbaton.mdl"
    def __init__(self):
        super(Weapon).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Weapon.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_pistol(Weapon):
    model = "models/weapons/w_pistol.mdl"
    def __init__(self):
        super(Weapon).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Weapon.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_ar2(Weapon):
    model = "models/weapons/w_irifle.mdl"
    def __init__(self):
        super(Weapon).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Weapon.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_rpg(Weapon):
    model = "models/weapons/w_rocket_launcher.mdl"
    def __init__(self):
        super(Weapon).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Weapon.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_smg1(Weapon):
    model = "models/weapons/w_smg1.mdl"
    def __init__(self):
        super(Weapon).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Weapon.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_357(Weapon):
    model = "models/weapons/w_357.mdl"
    def __init__(self):
        super(Weapon).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Weapon.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_crossbow(Weapon):
    model = "models/weapons/w_crossbow.mdl"
    def __init__(self):
        super(Weapon).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Weapon.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_shotgun(Weapon):
    model = "models/weapons/w_shotgun.mdl"
    def __init__(self):
        super(Weapon).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Weapon.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_frag(Weapon):
    model = "models/weapons/w_grenade.mdl"
    def __init__(self):
        super(Weapon).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Weapon.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_physcannon(Weapon):
    model = "models/weapons/w_physics.mdl"
    def __init__(self):
        super(Weapon).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Weapon.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_bugbait(Weapon):
    model = "models/spore.mdl"
    def __init__(self):
        super(Weapon).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Weapon.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_alyxgun(Weapon):
    model = "models/weapons/W_Alyx_Gun.mdl"
    def __init__(self):
        super(Weapon).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Weapon.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class weapon_annabelle(Weapon):
    model = "models/weapons/W_annabelle.mdl"
    def __init__(self):
        super(Weapon).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Weapon.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class trigger_rpgfire(Trigger):
    def __init__(self):
        super(Trigger).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)


class trigger_vphysics_motion(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        self.SetGravityScale = 1.0  # Type: float
        self.SetAdditionalAirDensity = 0  # Type: float
        self.SetVelocityLimit = 0.0  # Type: float
        self.SetVelocityLimitDelta = 0.0  # Type: float
        self.SetVelocityScale = 1.0  # Type: float
        self.SetAngVelocityLimit = 0.0  # Type: float
        self.SetAngVelocityScale = 1.0  # Type: float
        self.SetLinearForce = 0.0  # Type: float
        self.SetLinearForceAngles = [0.0, 0.0, 0.0]  # Type: angle
        self.ParticleTrailMaterial = None  # Type: string
        self.ParticleTrailLifetime = 4  # Type: float
        self.ParticleTrailStartSize = 2  # Type: float
        self.ParticleTrailEndSize = 3  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.SetGravityScale = float(entity_data.get('setgravityscale', 1.0))  # Type: float
        instance.SetAdditionalAirDensity = float(entity_data.get('setadditionalairdensity', 0))  # Type: float
        instance.SetVelocityLimit = float(entity_data.get('setvelocitylimit', 0.0))  # Type: float
        instance.SetVelocityLimitDelta = float(entity_data.get('setvelocitylimitdelta', 0.0))  # Type: float
        instance.SetVelocityScale = float(entity_data.get('setvelocityscale', 1.0))  # Type: float
        instance.SetAngVelocityLimit = float(entity_data.get('setangvelocitylimit', 0.0))  # Type: float
        instance.SetAngVelocityScale = float(entity_data.get('setangvelocityscale', 1.0))  # Type: float
        instance.SetLinearForce = float(entity_data.get('setlinearforce', 0.0))  # Type: float
        instance.SetLinearForceAngles = parse_float_vector(entity_data.get('setlinearforceangles', "0 0 0"))  # Type: angle
        instance.ParticleTrailMaterial = entity_data.get('particletrailmaterial', None)  # Type: string
        instance.ParticleTrailLifetime = float(entity_data.get('particletraillifetime', 4))  # Type: float
        instance.ParticleTrailStartSize = float(entity_data.get('particletrailstartsize', 2))  # Type: float
        instance.ParticleTrailEndSize = float(entity_data.get('particletrailendsize', 3))  # Type: float


class point_bugbait(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.Enabled = "CHOICES NOT SUPPORTED"  # Type: choices
        self.radius = 512  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Enabled = entity_data.get('enabled', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.radius = parse_source_value(entity_data.get('radius', 512))  # Type: integer


class weapon_brickbat(Weapon):
    def __init__(self):
        super(Weapon).__init__()
        self.origin = [0, 0, 0]
        self.BrickbatType = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Weapon.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.BrickbatType = entity_data.get('brickbattype', "CHOICES NOT SUPPORTED")  # Type: choices


class path_corner(Angles, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.wait = None  # Type: integer
        self.speed = None  # Type: integer
        self.yaw_speed = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.wait = parse_source_value(entity_data.get('wait', 0))  # Type: integer
        instance.speed = parse_source_value(entity_data.get('speed', 0))  # Type: integer
        instance.yaw_speed = parse_source_value(entity_data.get('yaw_speed', 0))  # Type: integer


class path_corner_crash(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination


class player_loadsaved(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.duration = "2"  # Type: string
        self.holdtime = "0"  # Type: string
        self.renderamt = 255  # Type: integer
        self.rendercolor = [0, 0, 0]  # Type: color255
        self.loadtime = "0"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.duration = entity_data.get('duration', "2")  # Type: string
        instance.holdtime = entity_data.get('holdtime', "0")  # Type: string
        instance.renderamt = parse_source_value(entity_data.get('renderamt', 255))  # Type: integer
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "0 0 0"))  # Type: color255
        instance.loadtime = entity_data.get('loadtime', "0")  # Type: string


class player_weaponstrip(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class player_speedmod(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


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


class ai_sound(Parentname, Targetname):
    icon_sprite = "editor/ai_sound.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.volume = 120  # Type: integer
        self.duration = 0.5  # Type: float
        self.soundtype = None  # Type: choices
        self.soundcontext = None  # Type: choices
        self.locationproxy = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.volume = parse_source_value(entity_data.get('volume', 120))  # Type: integer
        instance.duration = float(entity_data.get('duration', 0.5))  # Type: float
        instance.soundtype = entity_data.get('soundtype', None)  # Type: choices
        instance.soundcontext = entity_data.get('soundcontext', None)  # Type: choices
        instance.locationproxy = entity_data.get('locationproxy', None)  # Type: target_destination


class env_rotorwash(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class combine_mine(Parentname, Targetname):
    model = "models/props_combine/combine_mine01.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.bounce = "CHOICES NOT SUPPORTED"  # Type: choices
        self.LockSilently = "CHOICES NOT SUPPORTED"  # Type: choices
        self.StartDisarmed = None  # Type: choices
        self.Modification = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.bounce = entity_data.get('bounce', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.LockSilently = entity_data.get('locksilently', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.StartDisarmed = entity_data.get('startdisarmed', None)  # Type: choices
        instance.Modification = entity_data.get('modification', None)  # Type: choices


class env_ar2explosion(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.material = "particle/particle_noisesphere"  # Type: material

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.material = entity_data.get('material', "particle/particle_noisesphere")  # Type: material


class env_starfield(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_flare(Parentname, Targetname, Angles):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.scale = 1  # Type: float
        self.duration = 30  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.scale = float(entity_data.get('scale', 1))  # Type: float
        instance.duration = float(entity_data.get('duration', 30))  # Type: float


class env_muzzleflash(Parentname, Targetname, Angles):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.parentattachment = None  # Type: string
        self.scale = 1  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.parentattachment = entity_data.get('parentattachment', None)  # Type: string
        instance.scale = float(entity_data.get('scale', 1))  # Type: float


class env_terrainmorph(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.startradius = 500  # Type: integer
        self.goalradius = 100  # Type: integer
        self.duration = 3  # Type: integer
        self.fraction = 1  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.startradius = parse_source_value(entity_data.get('startradius', 500))  # Type: integer
        instance.goalradius = parse_source_value(entity_data.get('goalradius', 100))  # Type: integer
        instance.duration = parse_source_value(entity_data.get('duration', 3))  # Type: integer
        instance.fraction = parse_source_value(entity_data.get('fraction', 1))  # Type: integer


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


class point_camera(Parentname, Angles):
    viewport_model = "models/editor/camera.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
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
        instance.targetname = entity_data.get('targetname', None)  # Type: target_source
        instance.FOV = float(entity_data.get('fov', 90))  # Type: float
        instance.UseScreenAspectRatio = entity_data.get('usescreenaspectratio', None)  # Type: choices
        instance.fogEnable = entity_data.get('fogenable', None)  # Type: choices
        instance.fogColor = parse_int_vector(entity_data.get('fogcolor', "0 0 0"))  # Type: color255
        instance.fogStart = float(entity_data.get('fogstart', 2048))  # Type: float
        instance.fogEnd = float(entity_data.get('fogend', 4096))  # Type: float
        instance.fogMaxDensity = float(entity_data.get('fogmaxdensity', 1))  # Type: float


class func_monitor(func_brush):
    def __init__(self):
        super(func_brush).__init__()
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        func_brush.from_dict(instance, entity_data)
        instance.target = entity_data.get('target', None)  # Type: target_destination


class func_bulletshield(func_brush):
    def __init__(self):
        super(func_brush).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        func_brush.from_dict(instance, entity_data)


class BaseVehicle(Origin, Global, Targetname, prop_static_base):
    def __init__(self):
        super(prop_static_base).__init__()
        super(Origin).__init__()
        super(Global).__init__()
        super(Targetname).__init__()
        self.vehiclescript = "scripts/vehicles/jeep_test.txt"  # Type: string
        self.actionScale = 1  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Origin.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        prop_static_base.from_dict(instance, entity_data)
        instance.vehiclescript = entity_data.get('vehiclescript', "scripts/vehicles/jeep_test.txt")  # Type: string
        instance.actionScale = float(entity_data.get('actionscale', 1))  # Type: float


class BaseDriveableVehicle(BaseVehicle):
    def __init__(self):
        super(BaseVehicle).__init__()
        self.VehicleLocked = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseVehicle.from_dict(instance, entity_data)
        instance.VehicleLocked = entity_data.get('vehiclelocked', None)  # Type: choices


class prop_vehicle(BaseVehicle):
    def __init__(self):
        super(BaseVehicle).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseVehicle.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class prop_vehicle_driveable(BaseDriveableVehicle):
    def __init__(self):
        super(BaseDriveableVehicle).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseDriveableVehicle.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class point_apc_controller(Angles, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.yawrate = "30"  # Type: string
        self.yawtolerance = "15"  # Type: string
        self.pitchrate = "0"  # Type: string
        self.pitchtolerance = "20"  # Type: string
        self.rotatestartsound = None  # Type: sound
        self.rotatesound = None  # Type: sound
        self.rotatestopsound = None  # Type: sound
        self.minRange = "0"  # Type: string
        self.maxRange = "0"  # Type: string
        self.targetentityname = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.yawrate = entity_data.get('yawrate', "30")  # Type: string
        instance.yawtolerance = entity_data.get('yawtolerance', "15")  # Type: string
        instance.pitchrate = entity_data.get('pitchrate', "0")  # Type: string
        instance.pitchtolerance = entity_data.get('pitchtolerance', "20")  # Type: string
        instance.rotatestartsound = entity_data.get('rotatestartsound', None)  # Type: sound
        instance.rotatesound = entity_data.get('rotatesound', None)  # Type: sound
        instance.rotatestopsound = entity_data.get('rotatestopsound', None)  # Type: sound
        instance.minRange = entity_data.get('minrange', "0")  # Type: string
        instance.maxRange = entity_data.get('maxrange', "0")  # Type: string
        instance.targetentityname = entity_data.get('targetentityname', None)  # Type: string


class prop_vehicle_apc(BaseDriveableVehicle):
    def __init__(self):
        super(BaseDriveableVehicle).__init__()
        self.origin = [0, 0, 0]
        self.missilehint = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseDriveableVehicle.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.missilehint = entity_data.get('missilehint', None)  # Type: target_destination


class info_apc_missile_hint(Origin, EnableDisable, Targetname, Angles):
    def __init__(self):
        super(Origin).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.target = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Origin.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.target = entity_data.get('target', None)  # Type: target_destination


class prop_vehicle_jeep(BaseDriveableVehicle):
    def __init__(self):
        super(BaseDriveableVehicle).__init__()
        self.origin = [0, 0, 0]
        self.CargoVisible = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseDriveableVehicle.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.CargoVisible = entity_data.get('cargovisible', None)  # Type: choices


class vehicle_viewcontroller(BaseDriveableVehicle):
    def __init__(self):
        super(BaseDriveableVehicle).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseDriveableVehicle.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class prop_vehicle_airboat(BaseDriveableVehicle):
    def __init__(self):
        super(BaseDriveableVehicle).__init__()
        self.origin = [0, 0, 0]
        self.EnableGun = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseDriveableVehicle.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.EnableGun = entity_data.get('enablegun', None)  # Type: choices


class prop_vehicle_cannon(BaseDriveableVehicle):
    def __init__(self):
        super(BaseDriveableVehicle).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseDriveableVehicle.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class prop_vehicle_crane(BaseDriveableVehicle):
    def __init__(self):
        super(BaseDriveableVehicle).__init__()
        self.origin = [0, 0, 0]
        self.magnetname = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseDriveableVehicle.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.magnetname = entity_data.get('magnetname', None)  # Type: target_destination


class prop_vehicle_prisoner_pod(BaseDriveableVehicle, Parentname):
    def __init__(self):
        super(BaseDriveableVehicle).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseDriveableVehicle.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class env_speaker(BaseSpeaker):
    icon_sprite = "editor/ambient_generic.vmt"
    def __init__(self):
        super(BaseSpeaker).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseSpeaker.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class script_tauremoval(Angles, Parentname, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.vortigaunt = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.vortigaunt = entity_data.get('vortigaunt', None)  # Type: target_destination


class script_intro(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.alternatefovchange = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.alternatefovchange = entity_data.get('alternatefovchange', None)  # Type: choices


class env_citadel_energy_core(Angles, Parentname, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.scale = 1  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.scale = float(entity_data.get('scale', 1))  # Type: float


class env_alyxemp(Angles, Parentname, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.Type = None  # Type: choices
        self.EndTargetName = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.Type = entity_data.get('type', None)  # Type: choices
        instance.EndTargetName = entity_data.get('endtargetname', None)  # Type: target_destination


class test_sidelist(Base):
    def __init__(self):
        super().__init__()
        self.origin = [0, 0, 0]
        self.sides = None  # Type: sidelist

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.sides = entity_data.get('sides', None)  # Type: sidelist


class info_teleporter_countdown(Targetname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class prop_vehicle_choreo_generic(BaseDriveableVehicle, Parentname):
    def __init__(self):
        super(BaseDriveableVehicle).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.ignoremoveparent = None  # Type: choices
        self.ignoreplayer = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseDriveableVehicle.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.ignoremoveparent = entity_data.get('ignoremoveparent', None)  # Type: choices
        instance.ignoreplayer = entity_data.get('ignoreplayer', None)  # Type: choices


class filter_combineball_type(BaseFilter):
    icon_sprite = "editor/filter_class.vmt"
    def __init__(self):
        super(BaseFilter).__init__()
        self.balltype = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.balltype = entity_data.get('balltype', "CHOICES NOT SUPPORTED")  # Type: choices


class env_entity_dissolver(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self.magnitude = 250  # Type: integer
        self.dissolvetype = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.magnitude = parse_source_value(entity_data.get('magnitude', 250))  # Type: integer
        instance.dissolvetype = entity_data.get('dissolvetype', "CHOICES NOT SUPPORTED")  # Type: choices


class prop_coreball(Angles, Targetname):
    model = "models/props_combine/coreball.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class prop_scalable(Studiomodel, RenderFields, Targetname, Angles):
    def __init__(self):
        super(RenderFields).__init__()
        super(Studiomodel).__init__()
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Studiomodel.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class point_push(Angles, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.enabled = "CHOICES NOT SUPPORTED"  # Type: choices
        self.magnitude = 100  # Type: float
        self.radius = 128  # Type: float
        self.inner_radius = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.enabled = entity_data.get('enabled', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.magnitude = float(entity_data.get('magnitude', 100))  # Type: float
        instance.radius = float(entity_data.get('radius', 128))  # Type: float
        instance.inner_radius = float(entity_data.get('inner_radius', 0))  # Type: float


class npc_antlion_grub(Angles, Global, Targetname, BaseFadeProp):
    model = "models/antlion_grub.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Global).__init__()
        super(Targetname).__init__()
        super(BaseFadeProp).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        BaseFadeProp.from_dict(instance, entity_data)


class weapon_striderbuster(BasePropPhysics):
    model = "models/magnusson_device.mdl"
    def __init__(self):
        super(BasePropPhysics).__init__()
        self.origin = [0, 0, 0]
        self.dud = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePropPhysics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.dud = entity_data.get('dud', None)  # Type: choices


class point_flesh_effect_target(Parentname, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.radius = 8  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.radius = float(entity_data.get('radius', 8))  # Type: float



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
    'light': light,
    'light_environment': light_environment,
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
    'TalkNPC': TalkNPC,
    'PlayerCompanion': PlayerCompanion,
    'RappelNPC': RappelNPC,
    'AlyxInteractable': AlyxInteractable,
    'CombineBallSpawners': CombineBallSpawners,
    'prop_combine_ball': prop_combine_ball,
    'trigger_physics_trap': trigger_physics_trap,
    'trigger_weapon_dissolve': trigger_weapon_dissolve,
    'trigger_weapon_strip': trigger_weapon_strip,
    'func_combine_ball_spawner': func_combine_ball_spawner,
    'point_combine_ball_launcher': point_combine_ball_launcher,
    'npc_blob': npc_blob,
    'npc_grenade_frag': npc_grenade_frag,
    'npc_combine_cannon': npc_combine_cannon,
    'npc_combine_camera': npc_combine_camera,
    'npc_turret_ground': npc_turret_ground,
    'npc_turret_ceiling': npc_turret_ceiling,
    'npc_turret_floor': npc_turret_floor,
    'VehicleDriverNPC': VehicleDriverNPC,
    'npc_vehicledriver': npc_vehicledriver,
    'npc_cranedriver': npc_cranedriver,
    'npc_apcdriver': npc_apcdriver,
    'npc_rollermine': npc_rollermine,
    'npc_missiledefense': npc_missiledefense,
    'npc_sniper': npc_sniper,
    'info_radar_target': info_radar_target,
    'info_target_vehicle_transition': info_target_vehicle_transition,
    'info_snipertarget': info_snipertarget,
    'prop_thumper': prop_thumper,
    'npc_antlion': npc_antlion,
    'npc_antlionguard': npc_antlionguard,
    'npc_crow': npc_crow,
    'npc_seagull': npc_seagull,
    'npc_pigeon': npc_pigeon,
    'npc_ichthyosaur': npc_ichthyosaur,
    'BaseHeadcrab': BaseHeadcrab,
    'npc_headcrab': npc_headcrab,
    'npc_headcrab_fast': npc_headcrab_fast,
    'npc_headcrab_black': npc_headcrab_black,
    'npc_stalker': npc_stalker,
    'npc_bullseye': npc_bullseye,
    'npc_enemyfinder': npc_enemyfinder,
    'npc_enemyfinder_combinecannon': npc_enemyfinder_combinecannon,
    'npc_citizen': npc_citizen,
    'npc_fisherman': npc_fisherman,
    'npc_barney': npc_barney,
    'BaseCombine': BaseCombine,
    'npc_combine_s': npc_combine_s,
    'npc_launcher': npc_launcher,
    'npc_hunter': npc_hunter,
    'npc_hunter_maker': npc_hunter_maker,
    'npc_advisor': npc_advisor,
    'env_sporeexplosion': env_sporeexplosion,
    'env_gunfire': env_gunfire,
    'env_headcrabcanister': env_headcrabcanister,
    'npc_vortigaunt': npc_vortigaunt,
    'npc_spotlight': npc_spotlight,
    'npc_strider': npc_strider,
    'npc_barnacle': npc_barnacle,
    'npc_combinegunship': npc_combinegunship,
    'info_target_helicopter_crash': info_target_helicopter_crash,
    'info_target_gunshipcrash': info_target_gunshipcrash,
    'npc_combinedropship': npc_combinedropship,
    'npc_helicopter': npc_helicopter,
    'grenade_helicopter': grenade_helicopter,
    'npc_heli_avoidsphere': npc_heli_avoidsphere,
    'npc_heli_avoidbox': npc_heli_avoidbox,
    'npc_heli_nobomb': npc_heli_nobomb,
    'npc_fastzombie': npc_fastzombie,
    'npc_fastzombie_torso': npc_fastzombie_torso,
    'npc_zombie': npc_zombie,
    'npc_zombie_torso': npc_zombie_torso,
    'npc_zombine': npc_zombine,
    'npc_poisonzombie': npc_poisonzombie,
    'npc_cscanner': npc_cscanner,
    'npc_clawscanner': npc_clawscanner,
    'npc_manhack': npc_manhack,
    'npc_mortarsynth': npc_mortarsynth,
    'npc_metropolice': npc_metropolice,
    'npc_crabsynth': npc_crabsynth,
    'npc_monk': npc_monk,
    'npc_alyx': npc_alyx,
    'ai_goal_operator': ai_goal_operator,
    'info_darknessmode_lightsource': info_darknessmode_lightsource,
    'npc_kleiner': npc_kleiner,
    'npc_eli': npc_eli,
    'npc_magnusson': npc_magnusson,
    'npc_breen': npc_breen,
    'npc_mossman': npc_mossman,
    'npc_gman': npc_gman,
    'npc_dog': npc_dog,
    'monster_generic': monster_generic,
    'generic_actor': generic_actor,
    'cycler_actor': cycler_actor,
    'npc_maker': npc_maker,
    'npc_antlion_template_maker': npc_antlion_template_maker,
    'point_antlion_repellant': point_antlion_repellant,
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
    'ai_battle_line': ai_battle_line,
    'ai_goal_standoff': ai_goal_standoff,
    'ai_goal_police': ai_goal_police,
    'assault_rallypoint': assault_rallypoint,
    'assault_assaultpoint': assault_assaultpoint,
    'ai_goal_assault': ai_goal_assault,
    'BaseActBusy': BaseActBusy,
    'ai_goal_actbusy': ai_goal_actbusy,
    'ai_goal_actbusy_queue': ai_goal_actbusy_queue,
    'ai_changetarget': ai_changetarget,
    'ai_npc_eventresponsesystem': ai_npc_eventresponsesystem,
    'ai_changehintgroup': ai_changehintgroup,
    'ai_script_conditions': ai_script_conditions,
    'scripted_sequence': scripted_sequence,
    'aiscripted_schedule': aiscripted_schedule,
    'ai_citizen_response_system': ai_citizen_response_system,
    'func_healthcharger': func_healthcharger,
    'func_recharge': func_recharge,
    'func_vehicleclip': func_vehicleclip,
    'func_lookdoor': func_lookdoor,
    'trigger_waterydeath': trigger_waterydeath,
    'env_global': env_global,
    'BaseTank': BaseTank,
    'func_tank': func_tank,
    'func_tankpulselaser': func_tankpulselaser,
    'func_tanklaser': func_tanklaser,
    'func_tankrocket': func_tankrocket,
    'func_tankairboatgun': func_tankairboatgun,
    'func_tankapcrocket': func_tankapcrocket,
    'func_tankmortar': func_tankmortar,
    'func_tankphyscannister': func_tankphyscannister,
    'func_tank_combine_cannon': func_tank_combine_cannon,
    'Item': Item,
    'item_dynamic_resupply': item_dynamic_resupply,
    'item_ammo_pistol': item_ammo_pistol,
    'item_ammo_pistol_large': item_ammo_pistol_large,
    'item_ammo_smg1': item_ammo_smg1,
    'item_ammo_smg1_large': item_ammo_smg1_large,
    'item_ammo_ar2': item_ammo_ar2,
    'item_ammo_ar2_large': item_ammo_ar2_large,
    'item_ammo_357': item_ammo_357,
    'item_ammo_357_large': item_ammo_357_large,
    'item_ammo_crossbow': item_ammo_crossbow,
    'item_box_buckshot': item_box_buckshot,
    'item_rpg_round': item_rpg_round,
    'item_ammo_smg1_grenade': item_ammo_smg1_grenade,
    'item_battery': item_battery,
    'item_healthkit': item_healthkit,
    'item_healthvial': item_healthvial,
    'item_ammo_ar2_altfire': item_ammo_ar2_altfire,
    'item_suit': item_suit,
    'item_ammo_crate': item_ammo_crate,
    'item_item_crate': item_item_crate,
    'item_healthcharger': item_healthcharger,
    'item_suitcharger': item_suitcharger,
    'Weapon': Weapon,
    'weapon_crowbar': weapon_crowbar,
    'weapon_stunstick': weapon_stunstick,
    'weapon_pistol': weapon_pistol,
    'weapon_ar2': weapon_ar2,
    'weapon_rpg': weapon_rpg,
    'weapon_smg1': weapon_smg1,
    'weapon_357': weapon_357,
    'weapon_crossbow': weapon_crossbow,
    'weapon_shotgun': weapon_shotgun,
    'weapon_frag': weapon_frag,
    'weapon_physcannon': weapon_physcannon,
    'weapon_bugbait': weapon_bugbait,
    'weapon_alyxgun': weapon_alyxgun,
    'weapon_annabelle': weapon_annabelle,
    'trigger_rpgfire': trigger_rpgfire,
    'trigger_vphysics_motion': trigger_vphysics_motion,
    'point_bugbait': point_bugbait,
    'weapon_brickbat': weapon_brickbat,
    'path_corner': path_corner,
    'path_corner_crash': path_corner_crash,
    'player_loadsaved': player_loadsaved,
    'player_weaponstrip': player_weaponstrip,
    'player_speedmod': player_speedmod,
    'logic_choreographed_scene': logic_choreographed_scene,
    'logic_scene_list_manager': logic_scene_list_manager,
    'ai_sound': ai_sound,
    'env_rotorwash': env_rotorwash,
    'combine_mine': combine_mine,
    'env_ar2explosion': env_ar2explosion,
    'env_starfield': env_starfield,
    'env_flare': env_flare,
    'env_muzzleflash': env_muzzleflash,
    'env_terrainmorph': env_terrainmorph,
    'logic_achievement': logic_achievement,
    'point_camera': point_camera,
    'func_monitor': func_monitor,
    'func_bulletshield': func_bulletshield,
    'BaseVehicle': BaseVehicle,
    'BaseDriveableVehicle': BaseDriveableVehicle,
    'prop_vehicle': prop_vehicle,
    'prop_vehicle_driveable': prop_vehicle_driveable,
    'point_apc_controller': point_apc_controller,
    'prop_vehicle_apc': prop_vehicle_apc,
    'info_apc_missile_hint': info_apc_missile_hint,
    'prop_vehicle_jeep': prop_vehicle_jeep,
    'vehicle_viewcontroller': vehicle_viewcontroller,
    'prop_vehicle_airboat': prop_vehicle_airboat,
    'prop_vehicle_cannon': prop_vehicle_cannon,
    'prop_vehicle_crane': prop_vehicle_crane,
    'prop_vehicle_prisoner_pod': prop_vehicle_prisoner_pod,
    'env_speaker': env_speaker,
    'script_tauremoval': script_tauremoval,
    'script_intro': script_intro,
    'env_citadel_energy_core': env_citadel_energy_core,
    'env_alyxemp': env_alyxemp,
    'test_sidelist': test_sidelist,
    'info_teleporter_countdown': info_teleporter_countdown,
    'prop_vehicle_choreo_generic': prop_vehicle_choreo_generic,
    'filter_combineball_type': filter_combineball_type,
    'env_entity_dissolver': env_entity_dissolver,
    'prop_coreball': prop_coreball,
    'prop_scalable': prop_scalable,
    'point_push': point_push,
    'npc_antlion_grub': npc_antlion_grub,
    'weapon_striderbuster': weapon_striderbuster,
    'point_flesh_effect_target': point_flesh_effect_target,
}