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
    if string is None:
        return [0.0, 0.0, 0.0]
    return [float(val) for val in string.replace('  ', ' ').split(' ')]


class Base:
    hammer_id_counter = 0

    def __init__(self, entity_data: dict):
        self._hammer_id = -1
        self._raw_data = entity_data

    @classmethod
    def new_hammer_id(cls):
        new_id = cls.hammer_id_counter
        cls.hammer_id_counter += 1
        return new_id

    @property
    def class_name(self):
        return self._raw_data.get('classname')

    @property
    def hammer_id(self):
        if self._hammer_id == -1:
            if 'hammerid' in self._raw_data:
                self._hammer_id = int(self._raw_data.get('hammerid'))
            else:  # Titanfall
                self._hammer_id = Base.new_hammer_id()
        return self._hammer_id


class Angles(Base):

    @property
    def angles(self):
        return parse_float_vector(self._raw_data.get('angles', "0 0 0"))


class Origin(Base):

    @property
    def origin(self):
        return parse_float_vector(self._raw_data.get('origin', None))


class Studiomodel(Base):

    @property
    def model(self):
        return self._raw_data.get('model', None)

    @property
    def skin(self):
        return parse_source_value(self._raw_data.get('skin', 0))

    @property
    def modelscale(self):
        return parse_source_value(self._raw_data.get('modelscale', 1.0))

    @property
    def disableshadows(self):
        return self._raw_data.get('disableshadows', "0")


class BasePlat(Base):
    pass


class Targetname(Base):

    @property
    def targetname(self):
        return self._raw_data.get('targetname', None)


class Parentname(Base):

    @property
    def parentname(self):
        return self._raw_data.get('parentname', None)


class BaseBrush(Base):
    pass


class EnableDisable(Base):

    @property
    def StartDisabled(self):
        return self._raw_data.get('startdisabled', "0")


class RenderFxChoices(Base):

    @property
    def renderfx(self):
        return self._raw_data.get('renderfx', "0")


class Shadow(Base):

    @property
    def disableshadows(self):
        return self._raw_data.get('disableshadows', "0")


class RenderFields(RenderFxChoices):

    @property
    def rendermode(self):
        return self._raw_data.get('rendermode', "0")

    @property
    def renderamt(self):
        return parse_source_value(self._raw_data.get('renderamt', 255))

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "255 255 255"))

    @property
    def disablereceiveshadows(self):
        return self._raw_data.get('disablereceiveshadows', "0")


class DXLevelChoice(Base):

    @property
    def mindxlevel(self):
        return self._raw_data.get('mindxlevel', "0")

    @property
    def maxdxlevel(self):
        return self._raw_data.get('maxdxlevel', "0")


class Inputfilter(Base):

    @property
    def InputFilter(self):
        return self._raw_data.get('inputfilter', "0")


class Global(Base):

    @property
    def globalname(self):
        return self._raw_data.get('globalname', "")


class EnvGlobal(Targetname):

    @property
    def initialstate(self):
        return self._raw_data.get('initialstate', "0")

    @property
    def counter(self):
        return parse_source_value(self._raw_data.get('counter', 0))


class DamageFilter(Base):

    @property
    def damagefilter(self):
        return self._raw_data.get('damagefilter', "")


class ResponseContext(Base):

    @property
    def ResponseContext(self):
        return self._raw_data.get('responsecontext', "")


class Breakable(DamageFilter, Targetname, Shadow):

    @property
    def ExplodeDamage(self):
        return parse_source_value(self._raw_data.get('explodedamage', 0))

    @property
    def ExplodeRadius(self):
        return parse_source_value(self._raw_data.get('exploderadius', 0))

    @property
    def PerformanceMode(self):
        return self._raw_data.get('performancemode', "0")

    @property
    def BreakModelMessage(self):
        return self._raw_data.get('breakmodelmessage', "")


class BreakableBrush(Parentname, Breakable, Global):

    @property
    def propdata(self):
        return self._raw_data.get('propdata', "0")

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 1))

    @property
    def material(self):
        return self._raw_data.get('material', "0")

    @property
    def explosion(self):
        return self._raw_data.get('explosion', "0")

    @property
    def gibdir(self):
        return parse_float_vector(self._raw_data.get('gibdir', "0 0 0"))

    @property
    def nodamageforces(self):
        return self._raw_data.get('nodamageforces', "0")

    @property
    def gibmodel(self):
        return self._raw_data.get('gibmodel', "")

    @property
    def spawnobject(self):
        return self._raw_data.get('spawnobject', "0")

    @property
    def explodemagnitude(self):
        return parse_source_value(self._raw_data.get('explodemagnitude', 0))

    @property
    def pressuredelay(self):
        return parse_source_value(self._raw_data.get('pressuredelay', 0))


class BreakableProp(Breakable):

    @property
    def pressuredelay(self):
        return parse_source_value(self._raw_data.get('pressuredelay', 0))


class BaseNPC(Angles, ResponseContext, Shadow, Targetname, DamageFilter, RenderFields):

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def squadname(self):
        return self._raw_data.get('squadname', None)

    @property
    def hintgroup(self):
        return self._raw_data.get('hintgroup', "")

    @property
    def hintlimiting(self):
        return self._raw_data.get('hintlimiting', "0")

    @property
    def sleepstate(self):
        return self._raw_data.get('sleepstate', "0")

    @property
    def wakeradius(self):
        return parse_source_value(self._raw_data.get('wakeradius', 0))

    @property
    def wakesquad(self):
        return self._raw_data.get('wakesquad', "0")

    @property
    def enemyfilter(self):
        return self._raw_data.get('enemyfilter', "")

    @property
    def ignoreunseenenemies(self):
        return self._raw_data.get('ignoreunseenenemies', "0")

    @property
    def physdamagescale(self):
        return parse_source_value(self._raw_data.get('physdamagescale', 1.0))


class info_npc_spawn_destination(Parentname, Angles, Targetname):
    icon_sprite = "editor/info_target.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def ReuseDelay(self):
        return parse_source_value(self._raw_data.get('reusedelay', 1))

    @property
    def RenameNPC(self):
        return self._raw_data.get('renamenpc', "")


class BaseNPCMaker(Angles, Targetname, EnableDisable):
    icon_sprite = "editor/npc_maker.vmt"

    @property
    def MaxNPCCount(self):
        return parse_source_value(self._raw_data.get('maxnpccount', 1))

    @property
    def SpawnFrequency(self):
        return self._raw_data.get('spawnfrequency', "5")

    @property
    def MaxLiveChildren(self):
        return parse_source_value(self._raw_data.get('maxlivechildren', 5))


class npc_template_maker(BaseNPCMaker):
    icon_sprite = "editor/npc_maker.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def TemplateName(self):
        return self._raw_data.get('templatename', "")

    @property
    def Radius(self):
        return parse_source_value(self._raw_data.get('radius', 256))

    @property
    def DestinationGroup(self):
        return self._raw_data.get('destinationgroup', None)

    @property
    def CriterionVisibility(self):
        return self._raw_data.get('criterionvisibility', "2")

    @property
    def CriterionDistance(self):
        return self._raw_data.get('criteriondistance', "2")

    @property
    def MinSpawnDistance(self):
        return parse_source_value(self._raw_data.get('minspawndistance', 0))


class BaseHelicopter(BaseNPC):

    @property
    def InitialSpeed(self):
        return self._raw_data.get('initialspeed', "0")


class PlayerClass(Base):
    pass


class Light(Base):

    @property
    def _light(self):
        return parse_int_vector(self._raw_data.get('_light', "255 255 255 200"))

    @property
    def _lightHDR(self):
        return parse_int_vector(self._raw_data.get('_lighthdr', "-1 -1 -1 1"))

    @property
    def _lightscaleHDR(self):
        return parse_source_value(self._raw_data.get('_lightscalehdr', 1))

    @property
    def style(self):
        return self._raw_data.get('style', "0")

    @property
    def pattern(self):
        return self._raw_data.get('pattern', "")

    @property
    def _constant_attn(self):
        return self._raw_data.get('_constant_attn', "0")

    @property
    def _linear_attn(self):
        return self._raw_data.get('_linear_attn', "0")

    @property
    def _quadratic_attn(self):
        return self._raw_data.get('_quadratic_attn', "1")

    @property
    def _fifty_percent_distance(self):
        return self._raw_data.get('_fifty_percent_distance', "0")

    @property
    def _zero_percent_distance(self):
        return self._raw_data.get('_zero_percent_distance', "0")

    @property
    def _hardfalloff(self):
        return parse_source_value(self._raw_data.get('_hardfalloff', 0))


class Node(Base):

    @property
    def nodeid(self):
        return parse_source_value(self._raw_data.get('nodeid', None))


class HintNode(Node):

    @property
    def hinttype(self):
        return self._raw_data.get('hinttype', "0")

    @property
    def hintactivity(self):
        return self._raw_data.get('hintactivity', "")

    @property
    def nodeFOV(self):
        return self._raw_data.get('nodefov', "180")

    @property
    def StartHintDisabled(self):
        return self._raw_data.get('starthintdisabled', "0")

    @property
    def Group(self):
        return self._raw_data.get('group', "")

    @property
    def TargetNode(self):
        return parse_source_value(self._raw_data.get('targetnode', -1))

    @property
    def IgnoreFacing(self):
        return self._raw_data.get('ignorefacing', "2")

    @property
    def MinimumState(self):
        return self._raw_data.get('minimumstate', "1")

    @property
    def MaximumState(self):
        return self._raw_data.get('maximumstate', "3")


class TriggerOnce(Parentname, Global, EnableDisable, Targetname, Origin):

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)


class Trigger(TriggerOnce):
    pass


class worldbase(Base):

    @property
    def message(self):
        return self._raw_data.get('message', None)

    @property
    def skyname(self):
        return self._raw_data.get('skyname', "sky_day01_01")

    @property
    def chaptertitle(self):
        return self._raw_data.get('chaptertitle', "")

    @property
    def startdark(self):
        return self._raw_data.get('startdark', "0")

    @property
    def gametitle(self):
        return self._raw_data.get('gametitle', "0")

    @property
    def newunit(self):
        return self._raw_data.get('newunit', "0")

    @property
    def maxoccludeearea(self):
        return parse_source_value(self._raw_data.get('maxoccludeearea', 0))

    @property
    def minoccluderarea(self):
        return parse_source_value(self._raw_data.get('minoccluderarea', 0))

    @property
    def maxoccludeearea_x360(self):
        return parse_source_value(self._raw_data.get('maxoccludeearea_x360', 0))

    @property
    def minoccluderarea_x360(self):
        return parse_source_value(self._raw_data.get('minoccluderarea_x360', 0))

    @property
    def maxpropscreenwidth(self):
        return parse_source_value(self._raw_data.get('maxpropscreenwidth', -1))

    @property
    def minpropscreenwidth(self):
        return parse_source_value(self._raw_data.get('minpropscreenwidth', 0))

    @property
    def detailvbsp(self):
        return self._raw_data.get('detailvbsp', "detail.vbsp")

    @property
    def detailmaterial(self):
        return self._raw_data.get('detailmaterial', "detail/detailsprites")

    @property
    def coldworld(self):
        return self._raw_data.get('coldworld', "0")


class worldspawn(ResponseContext, worldbase, Targetname):
    pass


class ambient_generic(Targetname):
    icon_sprite = "editor/ambient_generic.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def message(self):
        return self._raw_data.get('message', "")

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 10))

    @property
    def preset(self):
        return self._raw_data.get('preset', "0")

    @property
    def volstart(self):
        return parse_source_value(self._raw_data.get('volstart', 0))

    @property
    def fadeinsecs(self):
        return parse_source_value(self._raw_data.get('fadeinsecs', 0))

    @property
    def fadeoutsecs(self):
        return parse_source_value(self._raw_data.get('fadeoutsecs', 0))

    @property
    def pitch(self):
        return parse_source_value(self._raw_data.get('pitch', 100))

    @property
    def pitchstart(self):
        return parse_source_value(self._raw_data.get('pitchstart', 100))

    @property
    def spinup(self):
        return parse_source_value(self._raw_data.get('spinup', 0))

    @property
    def spindown(self):
        return parse_source_value(self._raw_data.get('spindown', 0))

    @property
    def lfotype(self):
        return parse_source_value(self._raw_data.get('lfotype', 0))

    @property
    def lforate(self):
        return parse_source_value(self._raw_data.get('lforate', 0))

    @property
    def lfomodpitch(self):
        return parse_source_value(self._raw_data.get('lfomodpitch', 0))

    @property
    def lfomodvol(self):
        return parse_source_value(self._raw_data.get('lfomodvol', 0))

    @property
    def cspinup(self):
        return parse_source_value(self._raw_data.get('cspinup', 0))

    @property
    def radius(self):
        return self._raw_data.get('radius', "1250")

    @property
    def SourceEntityName(self):
        return self._raw_data.get('sourceentityname', None)


class func_lod(Targetname):

    @property
    def DisappearDist(self):
        return parse_source_value(self._raw_data.get('disappeardist', 2000))

    @property
    def Solid(self):
        return self._raw_data.get('solid', "0")


class env_zoom(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def Rate(self):
        return parse_source_value(self._raw_data.get('rate', 1.0))

    @property
    def FOV(self):
        return parse_source_value(self._raw_data.get('fov', 75))


class env_screenoverlay(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def OverlayName1(self):
        return self._raw_data.get('overlayname1', "")

    @property
    def OverlayTime1(self):
        return parse_source_value(self._raw_data.get('overlaytime1', 1.0))

    @property
    def OverlayName2(self):
        return self._raw_data.get('overlayname2', "")

    @property
    def OverlayTime2(self):
        return parse_source_value(self._raw_data.get('overlaytime2', 1.0))

    @property
    def OverlayName3(self):
        return self._raw_data.get('overlayname3', "")

    @property
    def OverlayTime3(self):
        return parse_source_value(self._raw_data.get('overlaytime3', 1.0))

    @property
    def OverlayName4(self):
        return self._raw_data.get('overlayname4', "")

    @property
    def OverlayTime4(self):
        return parse_source_value(self._raw_data.get('overlaytime4', 1.0))

    @property
    def OverlayName5(self):
        return self._raw_data.get('overlayname5', "")

    @property
    def OverlayTime5(self):
        return parse_source_value(self._raw_data.get('overlaytime5', 1.0))

    @property
    def OverlayName6(self):
        return self._raw_data.get('overlayname6', "")

    @property
    def OverlayTime6(self):
        return parse_source_value(self._raw_data.get('overlaytime6', 1.0))

    @property
    def OverlayName7(self):
        return self._raw_data.get('overlayname7', "")

    @property
    def OverlayTime7(self):
        return parse_source_value(self._raw_data.get('overlaytime7', 1.0))

    @property
    def OverlayName8(self):
        return self._raw_data.get('overlayname8', "")

    @property
    def OverlayTime8(self):
        return parse_source_value(self._raw_data.get('overlaytime8', 1.0))

    @property
    def OverlayName9(self):
        return self._raw_data.get('overlayname9', "")

    @property
    def OverlayTime9(self):
        return parse_source_value(self._raw_data.get('overlaytime9', 1.0))

    @property
    def OverlayName10(self):
        return self._raw_data.get('overlayname10', "")

    @property
    def OverlayTime10(self):
        return parse_source_value(self._raw_data.get('overlaytime10', 1.0))


class env_screeneffect(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def type(self):
        return self._raw_data.get('type', "0")


class env_texturetoggle(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)


class env_splash(Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def scale(self):
        return parse_source_value(self._raw_data.get('scale', 8.0))


class env_particlelight(Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def Color(self):
        return parse_int_vector(self._raw_data.get('color', "255 0 0"))

    @property
    def Intensity(self):
        return parse_source_value(self._raw_data.get('intensity', 5000))

    @property
    def directional(self):
        return self._raw_data.get('directional', "0")

    @property
    def PSName(self):
        return self._raw_data.get('psname', "")


class env_sun(Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def use_angles(self):
        return self._raw_data.get('use_angles', "0")

    @property
    def pitch(self):
        return parse_source_value(self._raw_data.get('pitch', 0))

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "100 80 80"))

    @property
    def overlaycolor(self):
        return parse_int_vector(self._raw_data.get('overlaycolor', "0 0 0"))

    @property
    def size(self):
        return parse_source_value(self._raw_data.get('size', 16))

    @property
    def overlaysize(self):
        return parse_source_value(self._raw_data.get('overlaysize', -1))

    @property
    def material(self):
        return self._raw_data.get('material', "sprites/light_glow02_add_noz")

    @property
    def overlaymaterial(self):
        return self._raw_data.get('overlaymaterial', "sprites/light_glow02_add_noz")

    @property
    def HDRColorScale(self):
        return parse_source_value(self._raw_data.get('hdrcolorscale', 1.0))


class game_ragdoll_manager(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def MaxRagdollCount(self):
        return parse_source_value(self._raw_data.get('maxragdollcount', -1))

    @property
    def MaxRagdollCountDX8(self):
        return parse_source_value(self._raw_data.get('maxragdollcountdx8', -1))

    @property
    def SaveImportant(self):
        return self._raw_data.get('saveimportant', "0")


class game_gib_manager(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def maxpieces(self):
        return parse_source_value(self._raw_data.get('maxpieces', -1))

    @property
    def maxpiecesdx8(self):
        return parse_source_value(self._raw_data.get('maxpiecesdx8', -1))

    @property
    def allownewgibs(self):
        return self._raw_data.get('allownewgibs', "0")


class env_lightglow(Parentname, Angles, Targetname):
    model_ = "models/editor/axis_helper_thick.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "255 255 255"))

    @property
    def VerticalGlowSize(self):
        return parse_source_value(self._raw_data.get('verticalglowsize', 30))

    @property
    def HorizontalGlowSize(self):
        return parse_source_value(self._raw_data.get('horizontalglowsize', 30))

    @property
    def MinDist(self):
        return parse_source_value(self._raw_data.get('mindist', 500))

    @property
    def MaxDist(self):
        return parse_source_value(self._raw_data.get('maxdist', 2000))

    @property
    def OuterMaxDist(self):
        return parse_source_value(self._raw_data.get('outermaxdist', 0))

    @property
    def GlowProxySize(self):
        return parse_source_value(self._raw_data.get('glowproxysize', 2.0))

    @property
    def HDRColorScale(self):
        return parse_source_value(self._raw_data.get('hdrcolorscale', 1.0))


class env_smokestack(Parentname, Angles):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def targetname(self):
        return self._raw_data.get('targetname', None)

    @property
    def InitialState(self):
        return self._raw_data.get('initialstate', "0")

    @property
    def BaseSpread(self):
        return parse_source_value(self._raw_data.get('basespread', 20))

    @property
    def SpreadSpeed(self):
        return parse_source_value(self._raw_data.get('spreadspeed', 15))

    @property
    def Speed(self):
        return parse_source_value(self._raw_data.get('speed', 30))

    @property
    def StartSize(self):
        return parse_source_value(self._raw_data.get('startsize', 20))

    @property
    def EndSize(self):
        return parse_source_value(self._raw_data.get('endsize', 30))

    @property
    def Rate(self):
        return parse_source_value(self._raw_data.get('rate', 20))

    @property
    def JetLength(self):
        return parse_source_value(self._raw_data.get('jetlength', 180))

    @property
    def WindAngle(self):
        return parse_source_value(self._raw_data.get('windangle', 0))

    @property
    def WindSpeed(self):
        return parse_source_value(self._raw_data.get('windspeed', 0))

    @property
    def SmokeMaterial(self):
        return self._raw_data.get('smokematerial', "particle/SmokeStack.vmt")

    @property
    def twist(self):
        return parse_source_value(self._raw_data.get('twist', 0))

    @property
    def roll(self):
        return parse_source_value(self._raw_data.get('roll', 0))

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "255 255 255"))

    @property
    def renderamt(self):
        return parse_source_value(self._raw_data.get('renderamt', 255))


class env_fade(Targetname):
    icon_sprite = "editor/env_fade"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def duration(self):
        return self._raw_data.get('duration', "2")

    @property
    def holdtime(self):
        return self._raw_data.get('holdtime', "0")

    @property
    def renderamt(self):
        return parse_source_value(self._raw_data.get('renderamt', 255))

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "0 0 0"))


class env_player_surface_trigger(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def gamematerial(self):
        return self._raw_data.get('gamematerial', "0")


class env_tonemap_controller(Targetname):
    icon_sprite = "editor/env_tonemap_controller.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class func_useableladder(Parentname, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def point0(self):
        return parse_float_vector(self._raw_data.get('point0', None))

    @property
    def point1(self):
        return parse_float_vector(self._raw_data.get('point1', None))

    @property
    def StartDisabled(self):
        return self._raw_data.get('startdisabled', "0")

    @property
    def ladderSurfaceProperties(self):
        return self._raw_data.get('laddersurfaceproperties', None)


class func_ladderendpoint(Parentname, Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)


class info_ladder_dismount(Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)


class func_areaportalwindow(Targetname):

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def FadeStartDist(self):
        return parse_source_value(self._raw_data.get('fadestartdist', 128))

    @property
    def FadeDist(self):
        return parse_source_value(self._raw_data.get('fadedist', 512))

    @property
    def TranslucencyLimit(self):
        return self._raw_data.get('translucencylimit', "0.2")

    @property
    def BackgroundBModel(self):
        return self._raw_data.get('backgroundbmodel', "")

    @property
    def PortalVersion(self):
        return parse_source_value(self._raw_data.get('portalversion', 1))


class func_wall(Shadow, Targetname, Global, RenderFields):

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)


class func_clip_vphysics(Targetname, EnableDisable):

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)


class func_brush(Parentname, Global, EnableDisable, Shadow, Inputfilter, Targetname, Origin, RenderFields):

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)

    @property
    def Solidity(self):
        return self._raw_data.get('solidity', "0")

    @property
    def excludednpc(self):
        return self._raw_data.get('excludednpc', "")

    @property
    def invert_exclusion(self):
        return self._raw_data.get('invert_exclusion', "0")

    @property
    def solidbsp(self):
        return self._raw_data.get('solidbsp', "0")

    @property
    def vrad_brush_cast_shadows(self):
        return self._raw_data.get('vrad_brush_cast_shadows', "0")


class vgui_screen_base(Parentname, Angles, Targetname):

    @property
    def panelname(self):
        return self._raw_data.get('panelname', None)

    @property
    def overlaymaterial(self):
        return self._raw_data.get('overlaymaterial', "")

    @property
    def width(self):
        return parse_source_value(self._raw_data.get('width', 32))

    @property
    def height(self):
        return parse_source_value(self._raw_data.get('height', 32))


class vgui_screen(vgui_screen_base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class vgui_slideshow_display(Parentname, Angles, Targetname):
    model_ = "models/editor/axis_helper_thick.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def displaytext(self):
        return self._raw_data.get('displaytext', "")

    @property
    def directory(self):
        return self._raw_data.get('directory', "slideshow")

    @property
    def minslidetime(self):
        return parse_source_value(self._raw_data.get('minslidetime', 0.5))

    @property
    def maxslidetime(self):
        return parse_source_value(self._raw_data.get('maxslidetime', 0.5))

    @property
    def cycletype(self):
        return self._raw_data.get('cycletype', "0")

    @property
    def nolistrepeat(self):
        return self._raw_data.get('nolistrepeat', "0")

    @property
    def width(self):
        return parse_source_value(self._raw_data.get('width', 256))

    @property
    def height(self):
        return parse_source_value(self._raw_data.get('height', 128))


class cycler(Angles, Parentname, Targetname, RenderFields):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', None)

    @property
    def skin(self):
        return parse_source_value(self._raw_data.get('skin', 0))

    @property
    def sequence(self):
        return parse_source_value(self._raw_data.get('sequence', 0))


class gibshooterbase(Parentname, Targetname):

    @property
    def angles(self):
        return self._raw_data.get('angles', "0 0 0")

    @property
    def m_iGibs(self):
        return parse_source_value(self._raw_data.get('m_igibs', 3))

    @property
    def delay(self):
        return self._raw_data.get('delay', "0")

    @property
    def gibangles(self):
        return self._raw_data.get('gibangles', "0 0 0")

    @property
    def gibanglevelocity(self):
        return self._raw_data.get('gibanglevelocity', "0")

    @property
    def m_flVelocity(self):
        return parse_source_value(self._raw_data.get('m_flvelocity', 200))

    @property
    def m_flVariance(self):
        return self._raw_data.get('m_flvariance', "0.15")

    @property
    def m_flGibLife(self):
        return self._raw_data.get('m_flgiblife', "4")

    @property
    def lightingorigin(self):
        return self._raw_data.get('lightingorigin', "")


class env_beam(Parentname, RenderFxChoices, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def renderamt(self):
        return parse_source_value(self._raw_data.get('renderamt', 100))

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "255 255 255"))

    @property
    def Radius(self):
        return parse_source_value(self._raw_data.get('radius', 256))

    @property
    def life(self):
        return self._raw_data.get('life', "1")

    @property
    def BoltWidth(self):
        return parse_source_value(self._raw_data.get('boltwidth', 2))

    @property
    def NoiseAmplitude(self):
        return parse_source_value(self._raw_data.get('noiseamplitude', 0))

    @property
    def texture(self):
        return self._raw_data.get('texture', "sprites/laserbeam.spr")

    @property
    def TextureScroll(self):
        return parse_source_value(self._raw_data.get('texturescroll', 35))

    @property
    def framerate(self):
        return parse_source_value(self._raw_data.get('framerate', 0))

    @property
    def framestart(self):
        return parse_source_value(self._raw_data.get('framestart', 0))

    @property
    def StrikeTime(self):
        return self._raw_data.get('striketime', "1")

    @property
    def damage(self):
        return self._raw_data.get('damage', "0")

    @property
    def LightningStart(self):
        return self._raw_data.get('lightningstart', "")

    @property
    def LightningEnd(self):
        return self._raw_data.get('lightningend', "")

    @property
    def decalname(self):
        return self._raw_data.get('decalname', "Bigshot")

    @property
    def HDRColorScale(self):
        return parse_source_value(self._raw_data.get('hdrcolorscale', 1.0))

    @property
    def TouchType(self):
        return self._raw_data.get('touchtype', "0")

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)


class env_beverage(Parentname, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 10))

    @property
    def beveragetype(self):
        return self._raw_data.get('beveragetype', "0")


class env_embers(Parentname, Angles, Targetname):

    @property
    def particletype(self):
        return self._raw_data.get('particletype', "0")

    @property
    def density(self):
        return parse_source_value(self._raw_data.get('density', 50))

    @property
    def lifetime(self):
        return parse_source_value(self._raw_data.get('lifetime', 4))

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 32))

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "255 255 255"))


class env_funnel(Parentname, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class env_blood(Parentname, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def spraydir(self):
        return parse_float_vector(self._raw_data.get('spraydir', "0 0 0"))

    @property
    def color(self):
        return self._raw_data.get('color', "0")

    @property
    def amount(self):
        return self._raw_data.get('amount', "100")


class env_bubbles(Parentname, Targetname):

    @property
    def density(self):
        return parse_source_value(self._raw_data.get('density', 2))

    @property
    def frequency(self):
        return parse_source_value(self._raw_data.get('frequency', 2))

    @property
    def current(self):
        return parse_source_value(self._raw_data.get('current', 0))


class env_explosion(Parentname, Targetname):
    icon_sprite = "editor/env_explosion.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def iMagnitude(self):
        return parse_source_value(self._raw_data.get('imagnitude', 100))

    @property
    def iRadiusOverride(self):
        return parse_source_value(self._raw_data.get('iradiusoverride', 0))

    @property
    def fireballsprite(self):
        return self._raw_data.get('fireballsprite', "sprites/zerogxplode.spr")

    @property
    def rendermode(self):
        return self._raw_data.get('rendermode', "5")

    @property
    def ignoredEntity(self):
        return self._raw_data.get('ignoredentity', None)


class env_smoketrail(Parentname, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def opacity(self):
        return parse_source_value(self._raw_data.get('opacity', 0.75))

    @property
    def spawnrate(self):
        return parse_source_value(self._raw_data.get('spawnrate', 20))

    @property
    def lifetime(self):
        return parse_source_value(self._raw_data.get('lifetime', 5.0))

    @property
    def startcolor(self):
        return parse_int_vector(self._raw_data.get('startcolor', "192 192 192"))

    @property
    def endcolor(self):
        return parse_int_vector(self._raw_data.get('endcolor', "160 160 160"))

    @property
    def emittime(self):
        return parse_source_value(self._raw_data.get('emittime', 0))

    @property
    def minspeed(self):
        return parse_source_value(self._raw_data.get('minspeed', 10))

    @property
    def maxspeed(self):
        return parse_source_value(self._raw_data.get('maxspeed', 20))

    @property
    def mindirectedspeed(self):
        return parse_source_value(self._raw_data.get('mindirectedspeed', 0))

    @property
    def maxdirectedspeed(self):
        return parse_source_value(self._raw_data.get('maxdirectedspeed', 0))

    @property
    def startsize(self):
        return parse_source_value(self._raw_data.get('startsize', 15))

    @property
    def endsize(self):
        return parse_source_value(self._raw_data.get('endsize', 50))

    @property
    def spawnradius(self):
        return parse_source_value(self._raw_data.get('spawnradius', 15))

    @property
    def firesprite(self):
        return self._raw_data.get('firesprite', "sprites/firetrail.spr")

    @property
    def smokesprite(self):
        return self._raw_data.get('smokesprite', "sprites/whitepuff.spr")


class env_physexplosion(Parentname, Targetname):
    icon_sprite = "editor/env_physexplosion.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def magnitude(self):
        return self._raw_data.get('magnitude', "100")

    @property
    def radius(self):
        return self._raw_data.get('radius', "0")

    @property
    def targetentityname(self):
        return self._raw_data.get('targetentityname', "")

    @property
    def inner_radius(self):
        return parse_source_value(self._raw_data.get('inner_radius', 0))


class env_physimpact(Parentname, Targetname):
    icon_sprite = "editor/env_physexplosion.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def angles(self):
        return self._raw_data.get('angles', "0 0 0")

    @property
    def magnitude(self):
        return parse_source_value(self._raw_data.get('magnitude', 100))

    @property
    def distance(self):
        return parse_source_value(self._raw_data.get('distance', 0))

    @property
    def directionentityname(self):
        return self._raw_data.get('directionentityname', "")


class env_fire(Parentname, Targetname, EnableDisable):
    icon_sprite = "editor/env_fire"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 30))

    @property
    def firesize(self):
        return parse_source_value(self._raw_data.get('firesize', 64))

    @property
    def fireattack(self):
        return parse_source_value(self._raw_data.get('fireattack', 4))

    @property
    def firetype(self):
        return self._raw_data.get('firetype', "0")

    @property
    def ignitionpoint(self):
        return parse_source_value(self._raw_data.get('ignitionpoint', 32))

    @property
    def damagescale(self):
        return parse_source_value(self._raw_data.get('damagescale', 1.0))


class env_firesource(Parentname, Targetname):
    icon_sprite = "editor/env_firesource"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def fireradius(self):
        return parse_source_value(self._raw_data.get('fireradius', 128))

    @property
    def firedamage(self):
        return parse_source_value(self._raw_data.get('firedamage', 10))


class env_firesensor(Parentname, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def fireradius(self):
        return parse_source_value(self._raw_data.get('fireradius', 128))

    @property
    def heatlevel(self):
        return parse_source_value(self._raw_data.get('heatlevel', 32))

    @property
    def heattime(self):
        return parse_source_value(self._raw_data.get('heattime', 0))


class env_entity_igniter(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def lifetime(self):
        return parse_source_value(self._raw_data.get('lifetime', 10))


class env_fog_controller(DXLevelChoice, Angles, Targetname):
    icon_sprite = "editor/fog_controller.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def fogenable(self):
        return self._raw_data.get('fogenable', "0")

    @property
    def fogblend(self):
        return self._raw_data.get('fogblend', "0")

    @property
    def use_angles(self):
        return self._raw_data.get('use_angles', "0")

    @property
    def fogcolor(self):
        return parse_int_vector(self._raw_data.get('fogcolor', "255 255 255"))

    @property
    def fogcolor2(self):
        return parse_int_vector(self._raw_data.get('fogcolor2', "255 255 255"))

    @property
    def fogdir(self):
        return self._raw_data.get('fogdir', "1 0 0")

    @property
    def fogstart(self):
        return self._raw_data.get('fogstart', "500.0")

    @property
    def fogend(self):
        return self._raw_data.get('fogend', "2000.0")

    @property
    def fogmaxdensity(self):
        return parse_source_value(self._raw_data.get('fogmaxdensity', 1))

    @property
    def foglerptime(self):
        return parse_source_value(self._raw_data.get('foglerptime', 0))

    @property
    def farz(self):
        return self._raw_data.get('farz', "-1")


class env_steam(Parentname, Angles, Targetname):
    viewport_model = "models/editor/spot_cone.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def InitialState(self):
        return self._raw_data.get('initialstate', "0")

    @property
    def type(self):
        return self._raw_data.get('type', "0")

    @property
    def SpreadSpeed(self):
        return parse_source_value(self._raw_data.get('spreadspeed', 15))

    @property
    def Speed(self):
        return parse_source_value(self._raw_data.get('speed', 120))

    @property
    def StartSize(self):
        return parse_source_value(self._raw_data.get('startsize', 10))

    @property
    def EndSize(self):
        return parse_source_value(self._raw_data.get('endsize', 25))

    @property
    def Rate(self):
        return parse_source_value(self._raw_data.get('rate', 26))

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "255 255 255"))

    @property
    def JetLength(self):
        return parse_source_value(self._raw_data.get('jetlength', 80))

    @property
    def renderamt(self):
        return parse_source_value(self._raw_data.get('renderamt', 255))

    @property
    def rollspeed(self):
        return parse_source_value(self._raw_data.get('rollspeed', 8))


class env_laser(Parentname, RenderFxChoices, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def LaserTarget(self):
        return self._raw_data.get('lasertarget', None)

    @property
    def renderamt(self):
        return parse_source_value(self._raw_data.get('renderamt', 100))

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "255 255 255"))

    @property
    def width(self):
        return parse_source_value(self._raw_data.get('width', 2))

    @property
    def NoiseAmplitude(self):
        return parse_source_value(self._raw_data.get('noiseamplitude', 0))

    @property
    def texture(self):
        return self._raw_data.get('texture', "sprites/laserbeam.spr")

    @property
    def EndSprite(self):
        return self._raw_data.get('endsprite', "")

    @property
    def TextureScroll(self):
        return parse_source_value(self._raw_data.get('texturescroll', 35))

    @property
    def framestart(self):
        return parse_source_value(self._raw_data.get('framestart', 0))

    @property
    def damage(self):
        return self._raw_data.get('damage', "100")

    @property
    def dissolvetype(self):
        return self._raw_data.get('dissolvetype', "None")


class env_message(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def message(self):
        return self._raw_data.get('message', None)

    @property
    def messagesound(self):
        return self._raw_data.get('messagesound', "")

    @property
    def messagevolume(self):
        return self._raw_data.get('messagevolume', "10")

    @property
    def messageattenuation(self):
        return self._raw_data.get('messageattenuation', "0")


class env_hudhint(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def message(self):
        return self._raw_data.get('message', "")


class env_shake(Parentname, Targetname):
    icon_sprite = "editor/env_shake.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def amplitude(self):
        return parse_source_value(self._raw_data.get('amplitude', 4))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 500))

    @property
    def duration(self):
        return parse_source_value(self._raw_data.get('duration', 1))

    @property
    def frequency(self):
        return parse_source_value(self._raw_data.get('frequency', 2.5))


class env_viewpunch(Parentname, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def punchangle(self):
        return parse_float_vector(self._raw_data.get('punchangle', "0 0 90"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 500))


class env_rotorwash_emitter(Parentname, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def altitude(self):
        return parse_source_value(self._raw_data.get('altitude', 1024))


class gibshooter(gibshooterbase):
    icon_sprite = "editor/gibshooter.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class env_shooter(gibshooterbase, RenderFields):
    icon_sprite = "editor/env_shooter.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def shootmodel(self):
        return self._raw_data.get('shootmodel', "")

    @property
    def shootsounds(self):
        return self._raw_data.get('shootsounds', "-1")

    @property
    def simulation(self):
        return self._raw_data.get('simulation', "0")

    @property
    def skin(self):
        return parse_source_value(self._raw_data.get('skin', 0))

    @property
    def nogibshadows(self):
        return self._raw_data.get('nogibshadows', "0")

    @property
    def gibgravityscale(self):
        return parse_source_value(self._raw_data.get('gibgravityscale', 1))

    @property
    def massoverride(self):
        return parse_source_value(self._raw_data.get('massoverride', 0))


class env_rotorshooter(gibshooterbase, RenderFields):
    icon_sprite = "editor/env_shooter.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def shootmodel(self):
        return self._raw_data.get('shootmodel', "")

    @property
    def shootsounds(self):
        return self._raw_data.get('shootsounds', "-1")

    @property
    def simulation(self):
        return self._raw_data.get('simulation', "0")

    @property
    def skin(self):
        return parse_source_value(self._raw_data.get('skin', 0))

    @property
    def rotortime(self):
        return parse_source_value(self._raw_data.get('rotortime', 1))

    @property
    def rotortimevariance(self):
        return parse_source_value(self._raw_data.get('rotortimevariance', 0.3))


class env_soundscape_proxy(Parentname, Targetname):
    icon_sprite = "editor/env_soundscape.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def MainSoundscapeName(self):
        return self._raw_data.get('mainsoundscapename', "")

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 128))


class env_soundscape(Parentname, Targetname, EnableDisable):
    icon_sprite = "editor/env_soundscape.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 128))

    @property
    def soundscape(self):
        return self._raw_data.get('soundscape', "Nothing")

    @property
    def position0(self):
        return self._raw_data.get('position0', "")

    @property
    def position1(self):
        return self._raw_data.get('position1', "")

    @property
    def position2(self):
        return self._raw_data.get('position2', "")

    @property
    def position3(self):
        return self._raw_data.get('position3', "")

    @property
    def position4(self):
        return self._raw_data.get('position4', "")

    @property
    def position5(self):
        return self._raw_data.get('position5', "")

    @property
    def position6(self):
        return self._raw_data.get('position6', "")

    @property
    def position7(self):
        return self._raw_data.get('position7', "")


class env_soundscape_triggerable(env_soundscape):
    icon_sprite = "editor/env_soundscape.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class env_spark(Parentname, Angles, Targetname):
    icon_sprite = "editor/env_spark.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def MaxDelay(self):
        return self._raw_data.get('maxdelay', "0")

    @property
    def Magnitude(self):
        return self._raw_data.get('magnitude', "1")

    @property
    def TrailLength(self):
        return self._raw_data.get('traillength', "1")


class env_sprite(Parentname, Targetname, DXLevelChoice, RenderFields):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def framerate(self):
        return self._raw_data.get('framerate', "10.0")

    @property
    def model(self):
        return self._raw_data.get('model', "sprites/glow01.spr")

    @property
    def scale(self):
        return self._raw_data.get('scale', "")

    @property
    def GlowProxySize(self):
        return parse_source_value(self._raw_data.get('glowproxysize', 2.0))

    @property
    def HDRColorScale(self):
        return parse_source_value(self._raw_data.get('hdrcolorscale', 1.0))


class env_sprite_oriented(Angles, env_sprite):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class env_wind(Angles, Targetname):
    icon_sprite = "editor/env_wind.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def minwind(self):
        return parse_source_value(self._raw_data.get('minwind', 20))

    @property
    def maxwind(self):
        return parse_source_value(self._raw_data.get('maxwind', 50))

    @property
    def mingust(self):
        return parse_source_value(self._raw_data.get('mingust', 100))

    @property
    def maxgust(self):
        return parse_source_value(self._raw_data.get('maxgust', 250))

    @property
    def mingustdelay(self):
        return parse_source_value(self._raw_data.get('mingustdelay', 10))

    @property
    def maxgustdelay(self):
        return parse_source_value(self._raw_data.get('maxgustdelay', 20))

    @property
    def gustduration(self):
        return parse_source_value(self._raw_data.get('gustduration', 5))

    @property
    def gustdirchange(self):
        return parse_source_value(self._raw_data.get('gustdirchange', 20))


class sky_camera(Angles):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def scale(self):
        return parse_source_value(self._raw_data.get('scale', 16))

    @property
    def fogenable(self):
        return self._raw_data.get('fogenable', "0")

    @property
    def fogblend(self):
        return self._raw_data.get('fogblend', "0")

    @property
    def use_angles(self):
        return self._raw_data.get('use_angles', "0")

    @property
    def fogcolor(self):
        return parse_int_vector(self._raw_data.get('fogcolor', "255 255 255"))

    @property
    def fogcolor2(self):
        return parse_int_vector(self._raw_data.get('fogcolor2', "255 255 255"))

    @property
    def fogdir(self):
        return self._raw_data.get('fogdir', "1 0 0")

    @property
    def fogstart(self):
        return self._raw_data.get('fogstart', "500.0")

    @property
    def fogend(self):
        return self._raw_data.get('fogend', "2000.0")


class BaseSpeaker(ResponseContext, Targetname):

    @property
    def delaymin(self):
        return self._raw_data.get('delaymin', "15")

    @property
    def delaymax(self):
        return self._raw_data.get('delaymax', "135")

    @property
    def rulescript(self):
        return self._raw_data.get('rulescript', "")

    @property
    def concept(self):
        return self._raw_data.get('concept', "")


class game_weapon_manager(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def weaponname(self):
        return self._raw_data.get('weaponname', "")

    @property
    def maxpieces(self):
        return parse_source_value(self._raw_data.get('maxpieces', 0))

    @property
    def ammomod(self):
        return parse_source_value(self._raw_data.get('ammomod', 1))


class game_end(Targetname):
    icon_sprite = "editor/game_end.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def master(self):
        return self._raw_data.get('master', None)


class game_player_equip(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def master(self):
        return self._raw_data.get('master', None)


class game_player_team(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def master(self):
        return self._raw_data.get('master', None)


class game_score(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def points(self):
        return parse_source_value(self._raw_data.get('points', 1))

    @property
    def master(self):
        return self._raw_data.get('master', None)


class game_text(Targetname):
    icon_sprite = "editor/game_text.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def message(self):
        return self._raw_data.get('message', "")

    @property
    def x(self):
        return self._raw_data.get('x', "-1")

    @property
    def y(self):
        return self._raw_data.get('y', "-1")

    @property
    def effect(self):
        return self._raw_data.get('effect', "0")

    @property
    def color(self):
        return parse_int_vector(self._raw_data.get('color', "100 100 100"))

    @property
    def color2(self):
        return parse_int_vector(self._raw_data.get('color2', "240 110 0"))

    @property
    def fadein(self):
        return self._raw_data.get('fadein', "1.5")

    @property
    def fadeout(self):
        return self._raw_data.get('fadeout', "0.5")

    @property
    def holdtime(self):
        return self._raw_data.get('holdtime', "1.2")

    @property
    def fxtime(self):
        return self._raw_data.get('fxtime', "0.25")

    @property
    def channel(self):
        return self._raw_data.get('channel', "1")

    @property
    def master(self):
        return self._raw_data.get('master', None)


class point_enable_motion_fixup(Parentname, Angles):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class point_message(Parentname, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def message(self):
        return self._raw_data.get('message', None)

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 128))

    @property
    def developeronly(self):
        return self._raw_data.get('developeronly', "0")


class point_spotlight(Angles, Parentname, Targetname, DXLevelChoice, RenderFields):
    model_ = "models/editor/cone_helper.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def spotlightlength(self):
        return parse_source_value(self._raw_data.get('spotlightlength', 500))

    @property
    def spotlightwidth(self):
        return parse_source_value(self._raw_data.get('spotlightwidth', 50))

    @property
    def HDRColorScale(self):
        return parse_source_value(self._raw_data.get('hdrcolorscale', 1.0))


class point_tesla(Parentname, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def m_SourceEntityName(self):
        return self._raw_data.get('m_sourceentityname', "")

    @property
    def m_SoundName(self):
        return self._raw_data.get('m_soundname', "DoSpark")

    @property
    def texture(self):
        return self._raw_data.get('texture', "sprites/physbeam.vmt")

    @property
    def m_Color(self):
        return parse_int_vector(self._raw_data.get('m_color', "255 255 255"))

    @property
    def m_flRadius(self):
        return parse_source_value(self._raw_data.get('m_flradius', 200))

    @property
    def beamcount_min(self):
        return parse_source_value(self._raw_data.get('beamcount_min', 6))

    @property
    def beamcount_max(self):
        return parse_source_value(self._raw_data.get('beamcount_max', 8))

    @property
    def thick_min(self):
        return self._raw_data.get('thick_min', "4")

    @property
    def thick_max(self):
        return self._raw_data.get('thick_max', "5")

    @property
    def lifetime_min(self):
        return self._raw_data.get('lifetime_min', "0.3")

    @property
    def lifetime_max(self):
        return self._raw_data.get('lifetime_max', "0.3")

    @property
    def interval_min(self):
        return self._raw_data.get('interval_min', "0.5")

    @property
    def interval_max(self):
        return self._raw_data.get('interval_max', "2")


class point_clientcommand(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class point_servercommand(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class point_bonusmaps_accessor(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def filename(self):
        return self._raw_data.get('filename', "")

    @property
    def mapname(self):
        return self._raw_data.get('mapname', "")


class game_ui(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def FieldOfView(self):
        return parse_source_value(self._raw_data.get('fieldofview', -1.0))


class game_zone_player(Parentname, Targetname):
    pass


class infodecal(Targetname):
    model_ = "models/editor/axis_helper_thick.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def texture(self):
        return self._raw_data.get('texture', None)

    @property
    def LowPriority(self):
        return self._raw_data.get('lowpriority', "0")


class info_projecteddecal(Angles, Targetname):
    model_ = "models/editor/axis_helper_thick.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def texture(self):
        return self._raw_data.get('texture', None)

    @property
    def Distance(self):
        return parse_source_value(self._raw_data.get('distance', 64))


class info_no_dynamic_shadow(Base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def sides(self):
        return self._raw_data.get('sides', None)


class info_player_start(Angles, PlayerClass):
    model_ = "models/editor/playerstart.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class info_overlay(Targetname):
    model_ = "models/editor/overlay_helper.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def material(self):
        return self._raw_data.get('material', None)

    @property
    def sides(self):
        return self._raw_data.get('sides', None)

    @property
    def RenderOrder(self):
        return parse_source_value(self._raw_data.get('renderorder', 0))

    @property
    def StartU(self):
        return parse_source_value(self._raw_data.get('startu', 0.0))

    @property
    def EndU(self):
        return parse_source_value(self._raw_data.get('endu', 1.0))

    @property
    def StartV(self):
        return parse_source_value(self._raw_data.get('startv', 0.0))

    @property
    def EndV(self):
        return parse_source_value(self._raw_data.get('endv', 1.0))

    @property
    def BasisOrigin(self):
        return parse_float_vector(self._raw_data.get('basisorigin', None))

    @property
    def BasisU(self):
        return parse_float_vector(self._raw_data.get('basisu', None))

    @property
    def BasisV(self):
        return parse_float_vector(self._raw_data.get('basisv', None))

    @property
    def BasisNormal(self):
        return parse_float_vector(self._raw_data.get('basisnormal', None))

    @property
    def uv0(self):
        return parse_float_vector(self._raw_data.get('uv0', None))

    @property
    def uv1(self):
        return parse_float_vector(self._raw_data.get('uv1', None))

    @property
    def uv2(self):
        return parse_float_vector(self._raw_data.get('uv2', None))

    @property
    def uv3(self):
        return parse_float_vector(self._raw_data.get('uv3', None))

    @property
    def fademindist(self):
        return parse_source_value(self._raw_data.get('fademindist', -1))

    @property
    def fademaxdist(self):
        return parse_source_value(self._raw_data.get('fademaxdist', 0))


class info_overlay_transition(Base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def material(self):
        return self._raw_data.get('material', None)

    @property
    def sides(self):
        return self._raw_data.get('sides', None)

    @property
    def sides2(self):
        return self._raw_data.get('sides2', None)

    @property
    def LengthTexcoordStart(self):
        return parse_source_value(self._raw_data.get('lengthtexcoordstart', 0.0))

    @property
    def LengthTexcoordEnd(self):
        return parse_source_value(self._raw_data.get('lengthtexcoordend', 1.0))

    @property
    def WidthTexcoordStart(self):
        return parse_source_value(self._raw_data.get('widthtexcoordstart', 0.0))

    @property
    def WidthTexcoordEnd(self):
        return parse_source_value(self._raw_data.get('widthtexcoordend', 1.0))

    @property
    def Width1(self):
        return parse_source_value(self._raw_data.get('width1', 25.0))

    @property
    def Width2(self):
        return parse_source_value(self._raw_data.get('width2', 25.0))

    @property
    def DebugDraw(self):
        return parse_source_value(self._raw_data.get('debugdraw', 0))


class info_intermission(Base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)


class info_landmark(Targetname):
    icon_sprite = "editor/info_landmark"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class info_null(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class info_target(Parentname, Angles, Targetname):
    icon_sprite = "editor/info_target.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class info_particle_system(Parentname, Angles, Targetname):
    model_ = "models/editor/cone_helper.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def effect_name(self):
        return self._raw_data.get('effect_name', None)

    @property
    def start_active(self):
        return self._raw_data.get('start_active', "0")

    @property
    def flag_as_weather(self):
        return self._raw_data.get('flag_as_weather', "0")

    @property
    def cpoint1(self):
        return self._raw_data.get('cpoint1', None)

    @property
    def cpoint2(self):
        return self._raw_data.get('cpoint2', None)

    @property
    def cpoint3(self):
        return self._raw_data.get('cpoint3', None)

    @property
    def cpoint4(self):
        return self._raw_data.get('cpoint4', None)

    @property
    def cpoint5(self):
        return self._raw_data.get('cpoint5', None)

    @property
    def cpoint6(self):
        return self._raw_data.get('cpoint6', None)

    @property
    def cpoint7(self):
        return self._raw_data.get('cpoint7', None)

    @property
    def cpoint8(self):
        return self._raw_data.get('cpoint8', None)

    @property
    def cpoint9(self):
        return self._raw_data.get('cpoint9', None)

    @property
    def cpoint10(self):
        return self._raw_data.get('cpoint10', None)

    @property
    def cpoint11(self):
        return self._raw_data.get('cpoint11', None)

    @property
    def cpoint12(self):
        return self._raw_data.get('cpoint12', None)

    @property
    def cpoint13(self):
        return self._raw_data.get('cpoint13', None)

    @property
    def cpoint14(self):
        return self._raw_data.get('cpoint14', None)

    @property
    def cpoint15(self):
        return self._raw_data.get('cpoint15', None)

    @property
    def cpoint16(self):
        return self._raw_data.get('cpoint16', None)

    @property
    def cpoint17(self):
        return self._raw_data.get('cpoint17', None)

    @property
    def cpoint18(self):
        return self._raw_data.get('cpoint18', None)

    @property
    def cpoint19(self):
        return self._raw_data.get('cpoint19', None)

    @property
    def cpoint20(self):
        return self._raw_data.get('cpoint20', None)

    @property
    def cpoint21(self):
        return self._raw_data.get('cpoint21', None)

    @property
    def cpoint22(self):
        return self._raw_data.get('cpoint22', None)

    @property
    def cpoint23(self):
        return self._raw_data.get('cpoint23', None)

    @property
    def cpoint24(self):
        return self._raw_data.get('cpoint24', None)

    @property
    def cpoint25(self):
        return self._raw_data.get('cpoint25', None)

    @property
    def cpoint26(self):
        return self._raw_data.get('cpoint26', None)

    @property
    def cpoint27(self):
        return self._raw_data.get('cpoint27', None)

    @property
    def cpoint28(self):
        return self._raw_data.get('cpoint28', None)

    @property
    def cpoint29(self):
        return self._raw_data.get('cpoint29', None)

    @property
    def cpoint30(self):
        return self._raw_data.get('cpoint30', None)

    @property
    def cpoint31(self):
        return self._raw_data.get('cpoint31', None)

    @property
    def cpoint32(self):
        return self._raw_data.get('cpoint32', None)

    @property
    def cpoint33(self):
        return self._raw_data.get('cpoint33', None)

    @property
    def cpoint34(self):
        return self._raw_data.get('cpoint34', None)

    @property
    def cpoint35(self):
        return self._raw_data.get('cpoint35', None)

    @property
    def cpoint36(self):
        return self._raw_data.get('cpoint36', None)

    @property
    def cpoint37(self):
        return self._raw_data.get('cpoint37', None)

    @property
    def cpoint38(self):
        return self._raw_data.get('cpoint38', None)

    @property
    def cpoint39(self):
        return self._raw_data.get('cpoint39', None)

    @property
    def cpoint40(self):
        return self._raw_data.get('cpoint40', None)

    @property
    def cpoint41(self):
        return self._raw_data.get('cpoint41', None)

    @property
    def cpoint42(self):
        return self._raw_data.get('cpoint42', None)

    @property
    def cpoint43(self):
        return self._raw_data.get('cpoint43', None)

    @property
    def cpoint44(self):
        return self._raw_data.get('cpoint44', None)

    @property
    def cpoint45(self):
        return self._raw_data.get('cpoint45', None)

    @property
    def cpoint46(self):
        return self._raw_data.get('cpoint46', None)

    @property
    def cpoint47(self):
        return self._raw_data.get('cpoint47', None)

    @property
    def cpoint48(self):
        return self._raw_data.get('cpoint48', None)

    @property
    def cpoint49(self):
        return self._raw_data.get('cpoint49', None)

    @property
    def cpoint50(self):
        return self._raw_data.get('cpoint50', None)

    @property
    def cpoint51(self):
        return self._raw_data.get('cpoint51', None)

    @property
    def cpoint52(self):
        return self._raw_data.get('cpoint52', None)

    @property
    def cpoint53(self):
        return self._raw_data.get('cpoint53', None)

    @property
    def cpoint54(self):
        return self._raw_data.get('cpoint54', None)

    @property
    def cpoint55(self):
        return self._raw_data.get('cpoint55', None)

    @property
    def cpoint56(self):
        return self._raw_data.get('cpoint56', None)

    @property
    def cpoint57(self):
        return self._raw_data.get('cpoint57', None)

    @property
    def cpoint58(self):
        return self._raw_data.get('cpoint58', None)

    @property
    def cpoint59(self):
        return self._raw_data.get('cpoint59', None)

    @property
    def cpoint60(self):
        return self._raw_data.get('cpoint60', None)

    @property
    def cpoint61(self):
        return self._raw_data.get('cpoint61', None)

    @property
    def cpoint62(self):
        return self._raw_data.get('cpoint62', None)

    @property
    def cpoint63(self):
        return self._raw_data.get('cpoint63', None)

    @property
    def cpoint1_parent(self):
        return parse_source_value(self._raw_data.get('cpoint1_parent', 0))

    @property
    def cpoint2_parent(self):
        return parse_source_value(self._raw_data.get('cpoint2_parent', 0))

    @property
    def cpoint3_parent(self):
        return parse_source_value(self._raw_data.get('cpoint3_parent', 0))

    @property
    def cpoint4_parent(self):
        return parse_source_value(self._raw_data.get('cpoint4_parent', 0))

    @property
    def cpoint5_parent(self):
        return parse_source_value(self._raw_data.get('cpoint5_parent', 0))

    @property
    def cpoint6_parent(self):
        return parse_source_value(self._raw_data.get('cpoint6_parent', 0))

    @property
    def cpoint7_parent(self):
        return parse_source_value(self._raw_data.get('cpoint7_parent', 0))


class phys_ragdollmagnet(Parentname, Angles, Targetname, EnableDisable):
    icon_sprite = "editor/info_target.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def axis(self):
        return self._raw_data.get('axis', None)

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 512))

    @property
    def force(self):
        return parse_source_value(self._raw_data.get('force', 5000))

    @property
    def target(self):
        return self._raw_data.get('target', "")


class info_lighting(Targetname):
    icon_sprite = "editor/info_lighting.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class info_teleport_destination(Parentname, Angles, Targetname, PlayerClass):
    model_ = "models/editor/playerstart.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class info_node(Node):
    model_ = "models/editor/ground_node.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class info_node_hint(Angles, HintNode, Targetname):
    model_ = "models/editor/ground_node_hint.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class info_node_air(Node):
    model_ = "models/editor/air_node.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def nodeheight(self):
        return parse_source_value(self._raw_data.get('nodeheight', 0))


class info_node_air_hint(Angles, HintNode, Targetname):
    model_ = "models/editor/air_node_hint.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def nodeheight(self):
        return parse_source_value(self._raw_data.get('nodeheight', 0))


class info_hint(Angles, HintNode, Targetname):
    model_ = "models/editor/node_hint.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class info_node_link(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def StartNode(self):
        return parse_source_value(self._raw_data.get('startnode', None))

    @property
    def EndNode(self):
        return parse_source_value(self._raw_data.get('endnode', None))

    @property
    def initialstate(self):
        return self._raw_data.get('initialstate', "1")

    @property
    def linktype(self):
        return self._raw_data.get('linktype', "1")

    @property
    def AllowUse(self):
        return self._raw_data.get('allowuse', None)

    @property
    def InvertAllow(self):
        return self._raw_data.get('invertallow', "0")


class info_node_link_controller(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def mins(self):
        return parse_float_vector(self._raw_data.get('mins', "-8 -32 -36"))

    @property
    def maxs(self):
        return parse_float_vector(self._raw_data.get('maxs', "8 32 36"))

    @property
    def initialstate(self):
        return self._raw_data.get('initialstate', "1")

    @property
    def useairlinkradius(self):
        return self._raw_data.get('useairlinkradius', "0")

    @property
    def AllowUse(self):
        return self._raw_data.get('allowuse', None)

    @property
    def InvertAllow(self):
        return self._raw_data.get('invertallow', "0")


class info_radial_link_controller(Parentname, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 120))


class info_node_climb(Angles, HintNode, Targetname):
    model_ = "models/editor/climb_node.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class light(Targetname, Light):
    icon_sprite = "editor/light.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def _distance(self):
        return parse_source_value(self._raw_data.get('_distance', 0))


class light_environment(Angles):
    icon_sprite = "editor/light_env.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def pitch(self):
        return parse_source_value(self._raw_data.get('pitch', 0))

    @property
    def _light(self):
        return parse_int_vector(self._raw_data.get('_light', "255 255 255 200"))

    @property
    def _ambient(self):
        return parse_int_vector(self._raw_data.get('_ambient', "255 255 255 20"))

    @property
    def _lightHDR(self):
        return parse_int_vector(self._raw_data.get('_lighthdr', "-1 -1 -1 1"))

    @property
    def _lightscaleHDR(self):
        return parse_source_value(self._raw_data.get('_lightscalehdr', 1))

    @property
    def _ambientHDR(self):
        return parse_int_vector(self._raw_data.get('_ambienthdr', "-1 -1 -1 1"))

    @property
    def _AmbientScaleHDR(self):
        return parse_source_value(self._raw_data.get('_ambientscalehdr', 1))

    @property
    def SunSpreadAngle(self):
        return parse_source_value(self._raw_data.get('sunspreadangle', 0))


class light_spot(Angles, Targetname, Light):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def _inner_cone(self):
        return parse_source_value(self._raw_data.get('_inner_cone', 30))

    @property
    def _cone(self):
        return parse_source_value(self._raw_data.get('_cone', 45))

    @property
    def _exponent(self):
        return parse_source_value(self._raw_data.get('_exponent', 1))

    @property
    def _distance(self):
        return parse_source_value(self._raw_data.get('_distance', 0))

    @property
    def pitch(self):
        return parse_source_value(self._raw_data.get('pitch', -90))


class light_dynamic(Parentname, Angles, Targetname):
    icon_sprite = "editor/light.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def _light(self):
        return parse_int_vector(self._raw_data.get('_light', "255 255 255 200"))

    @property
    def brightness(self):
        return parse_source_value(self._raw_data.get('brightness', 0))

    @property
    def _inner_cone(self):
        return parse_source_value(self._raw_data.get('_inner_cone', 30))

    @property
    def _cone(self):
        return parse_source_value(self._raw_data.get('_cone', 45))

    @property
    def pitch(self):
        return parse_source_value(self._raw_data.get('pitch', -90))

    @property
    def distance(self):
        return parse_source_value(self._raw_data.get('distance', 120))

    @property
    def spotlight_radius(self):
        return parse_source_value(self._raw_data.get('spotlight_radius', 80))

    @property
    def style(self):
        return self._raw_data.get('style', "0")


class shadow_control(Targetname):
    icon_sprite = "editor/shadow_control.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def angles(self):
        return self._raw_data.get('angles', "80 30 0")

    @property
    def color(self):
        return parse_int_vector(self._raw_data.get('color', "128 128 128"))

    @property
    def distance(self):
        return parse_source_value(self._raw_data.get('distance', 75))

    @property
    def disableallshadows(self):
        return self._raw_data.get('disableallshadows', "0")


class color_correction(Targetname, EnableDisable):
    icon_sprite = "editor/color_correction.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def minfalloff(self):
        return parse_source_value(self._raw_data.get('minfalloff', 0.0))

    @property
    def maxfalloff(self):
        return parse_source_value(self._raw_data.get('maxfalloff', 200.0))

    @property
    def maxweight(self):
        return parse_source_value(self._raw_data.get('maxweight', 1.0))

    @property
    def filename(self):
        return self._raw_data.get('filename', "")

    @property
    def fadeInDuration(self):
        return parse_source_value(self._raw_data.get('fadeinduration', 0.0))

    @property
    def fadeOutDuration(self):
        return parse_source_value(self._raw_data.get('fadeoutduration', 0.0))


class color_correction_volume(Targetname, EnableDisable):

    @property
    def fadeDuration(self):
        return parse_source_value(self._raw_data.get('fadeduration', 10.0))

    @property
    def maxweight(self):
        return parse_source_value(self._raw_data.get('maxweight', 1.0))

    @property
    def filename(self):
        return self._raw_data.get('filename', "")


class KeyFrame(Base):

    @property
    def NextKey(self):
        return self._raw_data.get('nextkey', None)

    @property
    def MoveSpeed(self):
        return parse_source_value(self._raw_data.get('movespeed', 64))


class Mover(Base):

    @property
    def PositionInterpolator(self):
        return self._raw_data.get('positioninterpolator', "0")


class func_movelinear(Parentname, Targetname, Origin, RenderFields):

    @property
    def movedir(self):
        return parse_float_vector(self._raw_data.get('movedir', "0 0 0"))

    @property
    def startposition(self):
        return parse_source_value(self._raw_data.get('startposition', 0))

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 100))

    @property
    def movedistance(self):
        return parse_source_value(self._raw_data.get('movedistance', 100))

    @property
    def blockdamage(self):
        return parse_source_value(self._raw_data.get('blockdamage', 0))

    @property
    def startsound(self):
        return self._raw_data.get('startsound', None)

    @property
    def stopsound(self):
        return self._raw_data.get('stopsound', None)


class func_water_analog(Parentname, Targetname, Origin):

    @property
    def movedir(self):
        return parse_float_vector(self._raw_data.get('movedir', "0 0 0"))

    @property
    def startposition(self):
        return parse_source_value(self._raw_data.get('startposition', 0))

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 100))

    @property
    def movedistance(self):
        return parse_source_value(self._raw_data.get('movedistance', 100))

    @property
    def startsound(self):
        return self._raw_data.get('startsound', None)

    @property
    def stopsound(self):
        return self._raw_data.get('stopsound', None)

    @property
    def WaveHeight(self):
        return self._raw_data.get('waveheight', "3.0")


class func_rotating(Angles, Parentname, Shadow, Targetname, Origin, RenderFields):

    @property
    def maxspeed(self):
        return parse_source_value(self._raw_data.get('maxspeed', 100))

    @property
    def fanfriction(self):
        return parse_source_value(self._raw_data.get('fanfriction', 20))

    @property
    def message(self):
        return self._raw_data.get('message', None)

    @property
    def volume(self):
        return parse_source_value(self._raw_data.get('volume', 10))

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)

    @property
    def dmg(self):
        return parse_source_value(self._raw_data.get('dmg', 0))

    @property
    def solidbsp(self):
        return self._raw_data.get('solidbsp', "0")


class func_platrot(Angles, BasePlat, Parentname, Shadow, Targetname, Origin, RenderFields):

    @property
    def noise1(self):
        return self._raw_data.get('noise1', None)

    @property
    def noise2(self):
        return self._raw_data.get('noise2', None)

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 50))

    @property
    def height(self):
        return parse_source_value(self._raw_data.get('height', 0))

    @property
    def rotation(self):
        return parse_source_value(self._raw_data.get('rotation', 0))

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)


class keyframe_track(Parentname, Angles, KeyFrame, Targetname):
    pass


class move_keyframed(Parentname, KeyFrame, Mover, Targetname):
    pass


class move_track(Parentname, KeyFrame, Mover, Targetname):

    @property
    def WheelBaseLength(self):
        return parse_source_value(self._raw_data.get('wheelbaselength', 50))

    @property
    def Damage(self):
        return parse_source_value(self._raw_data.get('damage', 0))

    @property
    def NoRotate(self):
        return self._raw_data.get('norotate', "0")


class RopeKeyFrame(DXLevelChoice):

    @property
    def Slack(self):
        return parse_source_value(self._raw_data.get('slack', 25))

    @property
    def Type(self):
        return self._raw_data.get('type', "0")

    @property
    def Subdiv(self):
        return parse_source_value(self._raw_data.get('subdiv', 2))

    @property
    def Barbed(self):
        return self._raw_data.get('barbed', "0")

    @property
    def Width(self):
        return self._raw_data.get('width', "2")

    @property
    def TextureScale(self):
        return self._raw_data.get('texturescale', "1")

    @property
    def Collide(self):
        return self._raw_data.get('collide', "0")

    @property
    def Dangling(self):
        return self._raw_data.get('dangling', "0")

    @property
    def Breakable(self):
        return self._raw_data.get('breakable', "0")

    @property
    def RopeMaterial(self):
        return self._raw_data.get('ropematerial', "cable/cable.vmt")

    @property
    def NoWind(self):
        return self._raw_data.get('nowind', "0")


class keyframe_rope(Parentname, RopeKeyFrame, KeyFrame, Targetname):
    model_ = "models/editor/axis_helper_thick.mdl"
    pass


class move_rope(Parentname, RopeKeyFrame, KeyFrame, Targetname):
    model_ = "models/editor/axis_helper.mdl"

    @property
    def PositionInterpolator(self):
        return self._raw_data.get('positioninterpolator', "2")


class Button(Base):
    pass


class func_button(Parentname, Button, Targetname, DamageFilter, Origin, RenderFields):

    @property
    def movedir(self):
        return parse_float_vector(self._raw_data.get('movedir', "0 0 0"))

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 5))

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 0))

    @property
    def lip(self):
        return parse_source_value(self._raw_data.get('lip', 0))

    @property
    def master(self):
        return self._raw_data.get('master', None)

    @property
    def sounds(self):
        return self._raw_data.get('sounds', "0")

    @property
    def wait(self):
        return parse_source_value(self._raw_data.get('wait', 3))

    @property
    def locked_sound(self):
        return self._raw_data.get('locked_sound', "0")

    @property
    def unlocked_sound(self):
        return self._raw_data.get('unlocked_sound', "0")

    @property
    def locked_sentence(self):
        return self._raw_data.get('locked_sentence', "0")

    @property
    def unlocked_sentence(self):
        return self._raw_data.get('unlocked_sentence', "0")

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)


class func_rot_button(Angles, Parentname, Global, EnableDisable, Button, Targetname, Origin):

    @property
    def master(self):
        return self._raw_data.get('master', None)

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 50))

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 0))

    @property
    def sounds(self):
        return self._raw_data.get('sounds', "21")

    @property
    def wait(self):
        return parse_source_value(self._raw_data.get('wait', 3))

    @property
    def distance(self):
        return parse_source_value(self._raw_data.get('distance', 90))

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)


class momentary_rot_button(Angles, Parentname, Targetname, Origin, RenderFields):

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 50))

    @property
    def master(self):
        return self._raw_data.get('master', None)

    @property
    def sounds(self):
        return self._raw_data.get('sounds', "0")

    @property
    def distance(self):
        return parse_source_value(self._raw_data.get('distance', 90))

    @property
    def returnspeed(self):
        return parse_source_value(self._raw_data.get('returnspeed', 0))

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)

    @property
    def startposition(self):
        return parse_source_value(self._raw_data.get('startposition', 0))

    @property
    def startdirection(self):
        return self._raw_data.get('startdirection', "Forward")

    @property
    def solidbsp(self):
        return self._raw_data.get('solidbsp', "0")


class Door(Parentname, Shadow, Global, Targetname, RenderFields):

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 100))

    @property
    def master(self):
        return self._raw_data.get('master', None)

    @property
    def noise1(self):
        return self._raw_data.get('noise1', None)

    @property
    def noise2(self):
        return self._raw_data.get('noise2', None)

    @property
    def startclosesound(self):
        return self._raw_data.get('startclosesound', None)

    @property
    def closesound(self):
        return self._raw_data.get('closesound', None)

    @property
    def wait(self):
        return parse_source_value(self._raw_data.get('wait', 4))

    @property
    def lip(self):
        return parse_source_value(self._raw_data.get('lip', 0))

    @property
    def dmg(self):
        return parse_source_value(self._raw_data.get('dmg', 0))

    @property
    def forceclosed(self):
        return self._raw_data.get('forceclosed', "0")

    @property
    def ignoredebris(self):
        return self._raw_data.get('ignoredebris', "0")

    @property
    def message(self):
        return self._raw_data.get('message', None)

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 0))

    @property
    def locked_sound(self):
        return self._raw_data.get('locked_sound', None)

    @property
    def unlocked_sound(self):
        return self._raw_data.get('unlocked_sound', None)

    @property
    def spawnpos(self):
        return self._raw_data.get('spawnpos', "0")

    @property
    def locked_sentence(self):
        return self._raw_data.get('locked_sentence', "0")

    @property
    def unlocked_sentence(self):
        return self._raw_data.get('unlocked_sentence', "0")

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)

    @property
    def loopmovesound(self):
        return self._raw_data.get('loopmovesound', "0")


class func_door(Door, Origin):

    @property
    def movedir(self):
        return parse_float_vector(self._raw_data.get('movedir', "0 0 0"))

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)


class func_door_rotating(Angles, Door, Origin):

    @property
    def distance(self):
        return parse_source_value(self._raw_data.get('distance', 90))

    @property
    def solidbsp(self):
        return self._raw_data.get('solidbsp', "0")


class prop_door_rotating(Angles, Parentname, Studiomodel, Global, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def slavename(self):
        return self._raw_data.get('slavename', None)

    @property
    def hardware(self):
        return self._raw_data.get('hardware', "1")

    @property
    def ajarangles(self):
        return parse_float_vector(self._raw_data.get('ajarangles', "0 0 0"))

    @property
    def spawnpos(self):
        return self._raw_data.get('spawnpos', "0")

    @property
    def axis(self):
        return self._raw_data.get('axis', None)

    @property
    def distance(self):
        return parse_source_value(self._raw_data.get('distance', 90))

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 100))

    @property
    def soundopenoverride(self):
        return self._raw_data.get('soundopenoverride', None)

    @property
    def soundcloseoverride(self):
        return self._raw_data.get('soundcloseoverride', None)

    @property
    def soundmoveoverride(self):
        return self._raw_data.get('soundmoveoverride', None)

    @property
    def returndelay(self):
        return parse_source_value(self._raw_data.get('returndelay', -1))

    @property
    def dmg(self):
        return parse_source_value(self._raw_data.get('dmg', 0))

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 0))

    @property
    def soundlockedoverride(self):
        return self._raw_data.get('soundlockedoverride', None)

    @property
    def soundunlockedoverride(self):
        return self._raw_data.get('soundunlockedoverride', None)

    @property
    def forceclosed(self):
        return self._raw_data.get('forceclosed', "0")

    @property
    def opendir(self):
        return self._raw_data.get('opendir', "0")


class env_cubemap(Base):
    icon_sprite = "editor/env_cubemap.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def cubemapsize(self):
        return self._raw_data.get('cubemapsize', "0")

    @property
    def sides(self):
        return self._raw_data.get('sides', None)


class BModelParticleSpawner(Base):

    @property
    def StartDisabled(self):
        return self._raw_data.get('startdisabled', "0")

    @property
    def Color(self):
        return parse_int_vector(self._raw_data.get('color', "255 255 255"))

    @property
    def SpawnRate(self):
        return parse_source_value(self._raw_data.get('spawnrate', 40))

    @property
    def SpeedMax(self):
        return self._raw_data.get('speedmax', "13")

    @property
    def LifetimeMin(self):
        return self._raw_data.get('lifetimemin', "3")

    @property
    def LifetimeMax(self):
        return self._raw_data.get('lifetimemax', "5")

    @property
    def DistMax(self):
        return parse_source_value(self._raw_data.get('distmax', 1024))

    @property
    def Frozen(self):
        return self._raw_data.get('frozen', "0")


class func_dustmotes(Targetname, BModelParticleSpawner):

    @property
    def SizeMin(self):
        return self._raw_data.get('sizemin', "10")

    @property
    def SizeMax(self):
        return self._raw_data.get('sizemax', "20")

    @property
    def Alpha(self):
        return parse_source_value(self._raw_data.get('alpha', 255))


class func_smokevolume(Targetname):

    @property
    def Color1(self):
        return parse_int_vector(self._raw_data.get('color1', "255 255 255"))

    @property
    def Color2(self):
        return parse_int_vector(self._raw_data.get('color2', "255 255 255"))

    @property
    def material(self):
        return self._raw_data.get('material', "particle/particle_smokegrenade")

    @property
    def ParticleDrawWidth(self):
        return parse_source_value(self._raw_data.get('particledrawwidth', 120))

    @property
    def ParticleSpacingDistance(self):
        return parse_source_value(self._raw_data.get('particlespacingdistance', 80))

    @property
    def DensityRampSpeed(self):
        return parse_source_value(self._raw_data.get('densityrampspeed', 1))

    @property
    def RotationSpeed(self):
        return parse_source_value(self._raw_data.get('rotationspeed', 10))

    @property
    def MovementSpeed(self):
        return parse_source_value(self._raw_data.get('movementspeed', 10))

    @property
    def Density(self):
        return parse_source_value(self._raw_data.get('density', 1))


class func_dustcloud(Targetname, BModelParticleSpawner):

    @property
    def Alpha(self):
        return parse_source_value(self._raw_data.get('alpha', 30))

    @property
    def SizeMin(self):
        return self._raw_data.get('sizemin', "100")

    @property
    def SizeMax(self):
        return self._raw_data.get('sizemax', "200")


class env_dustpuff(Parentname, Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def scale(self):
        return parse_source_value(self._raw_data.get('scale', 8))

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 16))

    @property
    def color(self):
        return parse_int_vector(self._raw_data.get('color', "128 128 128"))


class env_particlescript(Parentname, Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/Ambient_citadel_paths.mdl")


class env_effectscript(Parentname, Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/Effects/teleporttrail.mdl")

    @property
    def scriptfile(self):
        return self._raw_data.get('scriptfile', "scripts/effects/testeffect.txt")


class logic_auto(Base):
    icon_sprite = "editor/logic_auto.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def globalstate(self):
        return self._raw_data.get('globalstate', None)


class point_viewcontrol(Parentname, Angles, Targetname):
    viewport_model = "models/editor/camera.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def targetattachment(self):
        return self._raw_data.get('targetattachment', None)

    @property
    def wait(self):
        return parse_source_value(self._raw_data.get('wait', 10))

    @property
    def moveto(self):
        return self._raw_data.get('moveto', None)

    @property
    def interpolatepositiontoplayer(self):
        return self._raw_data.get('interpolatepositiontoplayer', "0")

    @property
    def speed(self):
        return self._raw_data.get('speed', "0")

    @property
    def acceleration(self):
        return self._raw_data.get('acceleration', "500")

    @property
    def deceleration(self):
        return self._raw_data.get('deceleration', "500")


class point_posecontroller(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def PropName(self):
        return self._raw_data.get('propname', None)

    @property
    def PoseParameterName(self):
        return self._raw_data.get('poseparametername', None)

    @property
    def PoseValue(self):
        return parse_source_value(self._raw_data.get('posevalue', 0.0))

    @property
    def InterpolationTime(self):
        return parse_source_value(self._raw_data.get('interpolationtime', 0.0))

    @property
    def InterpolationWrap(self):
        return self._raw_data.get('interpolationwrap', "0")

    @property
    def CycleFrequency(self):
        return parse_source_value(self._raw_data.get('cyclefrequency', 0.0))

    @property
    def FModulationType(self):
        return self._raw_data.get('fmodulationtype', "0")

    @property
    def FModTimeOffset(self):
        return parse_source_value(self._raw_data.get('fmodtimeoffset', 0.0))

    @property
    def FModRate(self):
        return parse_source_value(self._raw_data.get('fmodrate', 0.0))

    @property
    def FModAmplitude(self):
        return parse_source_value(self._raw_data.get('fmodamplitude', 0.0))


class logic_compare(Targetname):
    icon_sprite = "editor/logic_compare.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def InitialValue(self):
        return parse_source_value(self._raw_data.get('initialvalue', None))

    @property
    def CompareValue(self):
        return parse_source_value(self._raw_data.get('comparevalue', None))


class logic_branch(Targetname):
    icon_sprite = "editor/logic_branch.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def InitialValue(self):
        return parse_source_value(self._raw_data.get('initialvalue', None))


class logic_branch_listener(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def Branch01(self):
        return self._raw_data.get('branch01', None)

    @property
    def Branch02(self):
        return self._raw_data.get('branch02', None)

    @property
    def Branch03(self):
        return self._raw_data.get('branch03', None)

    @property
    def Branch04(self):
        return self._raw_data.get('branch04', None)

    @property
    def Branch05(self):
        return self._raw_data.get('branch05', None)

    @property
    def Branch06(self):
        return self._raw_data.get('branch06', None)

    @property
    def Branch07(self):
        return self._raw_data.get('branch07', None)

    @property
    def Branch08(self):
        return self._raw_data.get('branch08', None)

    @property
    def Branch09(self):
        return self._raw_data.get('branch09', None)

    @property
    def Branch10(self):
        return self._raw_data.get('branch10', None)

    @property
    def Branch11(self):
        return self._raw_data.get('branch11', None)

    @property
    def Branch12(self):
        return self._raw_data.get('branch12', None)

    @property
    def Branch13(self):
        return self._raw_data.get('branch13', None)

    @property
    def Branch14(self):
        return self._raw_data.get('branch14', None)

    @property
    def Branch15(self):
        return self._raw_data.get('branch15', None)

    @property
    def Branch16(self):
        return self._raw_data.get('branch16', None)


class logic_case(Targetname):
    icon_sprite = "editor/logic_case.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def Case01(self):
        return self._raw_data.get('case01', None)

    @property
    def Case02(self):
        return self._raw_data.get('case02', None)

    @property
    def Case03(self):
        return self._raw_data.get('case03', None)

    @property
    def Case04(self):
        return self._raw_data.get('case04', None)

    @property
    def Case05(self):
        return self._raw_data.get('case05', None)

    @property
    def Case06(self):
        return self._raw_data.get('case06', None)

    @property
    def Case07(self):
        return self._raw_data.get('case07', None)

    @property
    def Case08(self):
        return self._raw_data.get('case08', None)

    @property
    def Case09(self):
        return self._raw_data.get('case09', None)

    @property
    def Case10(self):
        return self._raw_data.get('case10', None)

    @property
    def Case11(self):
        return self._raw_data.get('case11', None)

    @property
    def Case12(self):
        return self._raw_data.get('case12', None)

    @property
    def Case13(self):
        return self._raw_data.get('case13', None)

    @property
    def Case14(self):
        return self._raw_data.get('case14', None)

    @property
    def Case15(self):
        return self._raw_data.get('case15', None)

    @property
    def Case16(self):
        return self._raw_data.get('case16', None)


class logic_multicompare(Targetname):
    icon_sprite = "editor/logic_multicompare.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def IntegerValue(self):
        return parse_source_value(self._raw_data.get('integervalue', None))

    @property
    def ShouldComparetoValue(self):
        return self._raw_data.get('shouldcomparetovalue', "0")


class logic_relay(Targetname, EnableDisable):
    icon_sprite = "editor/logic_relay.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class logic_timer(Targetname, EnableDisable):
    icon_sprite = "editor/logic_timer.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def UseRandomTime(self):
        return self._raw_data.get('userandomtime', "0")

    @property
    def LowerRandomBound(self):
        return self._raw_data.get('lowerrandombound', None)

    @property
    def UpperRandomBound(self):
        return self._raw_data.get('upperrandombound', None)

    @property
    def RefireTime(self):
        return self._raw_data.get('refiretime', None)


class hammer_updateignorelist(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def IgnoredName01(self):
        return self._raw_data.get('ignoredname01', "")

    @property
    def IgnoredName02(self):
        return self._raw_data.get('ignoredname02', "")

    @property
    def IgnoredName03(self):
        return self._raw_data.get('ignoredname03', "")

    @property
    def IgnoredName04(self):
        return self._raw_data.get('ignoredname04', "")

    @property
    def IgnoredName05(self):
        return self._raw_data.get('ignoredname05', "")

    @property
    def IgnoredName06(self):
        return self._raw_data.get('ignoredname06', "")

    @property
    def IgnoredName07(self):
        return self._raw_data.get('ignoredname07', "")

    @property
    def IgnoredName08(self):
        return self._raw_data.get('ignoredname08', "")

    @property
    def IgnoredName09(self):
        return self._raw_data.get('ignoredname09', "")

    @property
    def IgnoredName10(self):
        return self._raw_data.get('ignoredname10', "")

    @property
    def IgnoredName11(self):
        return self._raw_data.get('ignoredname11', "")

    @property
    def IgnoredName12(self):
        return self._raw_data.get('ignoredname12', "")

    @property
    def IgnoredName13(self):
        return self._raw_data.get('ignoredname13', "")

    @property
    def IgnoredName14(self):
        return self._raw_data.get('ignoredname14', "")

    @property
    def IgnoredName15(self):
        return self._raw_data.get('ignoredname15', "")

    @property
    def IgnoredName16(self):
        return self._raw_data.get('ignoredname16', "")


class logic_collision_pair(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def attach1(self):
        return self._raw_data.get('attach1', "")

    @property
    def attach2(self):
        return self._raw_data.get('attach2', "")

    @property
    def startdisabled(self):
        return self._raw_data.get('startdisabled', "1")


class env_microphone(Parentname, Targetname, EnableDisable):
    icon_sprite = "editor/env_microphone.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def SpeakerName(self):
        return self._raw_data.get('speakername', "")

    @property
    def ListenFilter(self):
        return self._raw_data.get('listenfilter', "")

    @property
    def speaker_dsp_preset(self):
        return self._raw_data.get('speaker_dsp_preset', "0")

    @property
    def Sensitivity(self):
        return parse_source_value(self._raw_data.get('sensitivity', 1))

    @property
    def SmoothFactor(self):
        return parse_source_value(self._raw_data.get('smoothfactor', 0))

    @property
    def MaxRange(self):
        return parse_source_value(self._raw_data.get('maxrange', 240))


class math_remap(Targetname, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def in1(self):
        return parse_source_value(self._raw_data.get('in1', 0))

    @property
    def in2(self):
        return parse_source_value(self._raw_data.get('in2', 1))

    @property
    def out1(self):
        return parse_source_value(self._raw_data.get('out1', None))

    @property
    def out2(self):
        return parse_source_value(self._raw_data.get('out2', None))


class math_colorblend(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def inmin(self):
        return parse_source_value(self._raw_data.get('inmin', 0))

    @property
    def inmax(self):
        return parse_source_value(self._raw_data.get('inmax', 1))

    @property
    def colormin(self):
        return parse_int_vector(self._raw_data.get('colormin', "0 0 0"))

    @property
    def colormax(self):
        return parse_int_vector(self._raw_data.get('colormax', "255 255 255"))


class math_counter(Targetname, EnableDisable):
    icon_sprite = "editor/math_counter.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def startvalue(self):
        return parse_source_value(self._raw_data.get('startvalue', 0))

    @property
    def min(self):
        return parse_source_value(self._raw_data.get('min', 0))

    @property
    def max(self):
        return parse_source_value(self._raw_data.get('max', 0))


class logic_lineto(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def source(self):
        return self._raw_data.get('source', None)

    @property
    def target(self):
        return self._raw_data.get('target', None)


class logic_navigation(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)  # Set to none due to bug in BlackMesa base.fgd file

    @property
    def navprop(self):
        return self._raw_data.get('navprop', "Ignore")


class logic_autosave(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def NewLevelUnit(self):
        return self._raw_data.get('newlevelunit', "0")

    @property
    def MinimumHitPoints(self):
        return parse_source_value(self._raw_data.get('minimumhitpoints', 0))

    @property
    def MinHitPointsToCommit(self):
        return parse_source_value(self._raw_data.get('minhitpointstocommit', 0))


class logic_active_autosave(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def MinimumHitPoints(self):
        return parse_source_value(self._raw_data.get('minimumhitpoints', 30))

    @property
    def TriggerHitPoints(self):
        return parse_source_value(self._raw_data.get('triggerhitpoints', 75))

    @property
    def TimeToTrigget(self):
        return parse_source_value(self._raw_data.get('timetotrigget', 0))

    @property
    def DangerousTime(self):
        return parse_source_value(self._raw_data.get('dangeroustime', 10))


class point_template(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def Template01(self):
        return self._raw_data.get('template01', None)

    @property
    def Template02(self):
        return self._raw_data.get('template02', None)

    @property
    def Template03(self):
        return self._raw_data.get('template03', None)

    @property
    def Template04(self):
        return self._raw_data.get('template04', None)

    @property
    def Template05(self):
        return self._raw_data.get('template05', None)

    @property
    def Template06(self):
        return self._raw_data.get('template06', None)

    @property
    def Template07(self):
        return self._raw_data.get('template07', None)

    @property
    def Template08(self):
        return self._raw_data.get('template08', None)

    @property
    def Template09(self):
        return self._raw_data.get('template09', None)

    @property
    def Template10(self):
        return self._raw_data.get('template10', None)

    @property
    def Template11(self):
        return self._raw_data.get('template11', None)

    @property
    def Template12(self):
        return self._raw_data.get('template12', None)

    @property
    def Template13(self):
        return self._raw_data.get('template13', None)

    @property
    def Template14(self):
        return self._raw_data.get('template14', None)

    @property
    def Template15(self):
        return self._raw_data.get('template15', None)

    @property
    def Template16(self):
        return self._raw_data.get('template16', None)


class env_entity_maker(Parentname, Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def EntityTemplate(self):
        return self._raw_data.get('entitytemplate', "")

    @property
    def PostSpawnSpeed(self):
        return parse_source_value(self._raw_data.get('postspawnspeed', 0))

    @property
    def PostSpawnDirection(self):
        return parse_float_vector(self._raw_data.get('postspawndirection', "0 0 0"))

    @property
    def PostSpawnDirectionVariance(self):
        return parse_source_value(self._raw_data.get('postspawndirectionvariance', 0.15))

    @property
    def PostSpawnInheritAngles(self):
        return self._raw_data.get('postspawninheritangles', "0")


class BaseFilter(Targetname):

    @property
    def Negated(self):
        return self._raw_data.get('negated', "Allow entities that match criteria")


class filter_multi(BaseFilter):
    icon_sprite = "editor/filter_multiple.vmt"

    @property
    def filtertype(self):
        return self._raw_data.get('filtertype', "0")

    @property
    def Filter01(self):
        return self._raw_data.get('filter01', None)

    @property
    def Filter02(self):
        return self._raw_data.get('filter02', None)

    @property
    def Filter03(self):
        return self._raw_data.get('filter03', None)

    @property
    def Filter04(self):
        return self._raw_data.get('filter04', None)

    @property
    def Filter05(self):
        return self._raw_data.get('filter05', None)


class filter_activator_name(BaseFilter):
    icon_sprite = "editor/filter_name.vmt"

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)


class filter_activator_class(BaseFilter):
    icon_sprite = "editor/filter_class.vmt"

    @property
    def filterclass(self):
        return self._raw_data.get('filterclass', None)


class filter_activator_mass_greater(BaseFilter):
    icon_sprite = "editor/filter_class.vmt"

    @property
    def filtermass(self):
        return parse_source_value(self._raw_data.get('filtermass', None))


class filter_damage_type(BaseFilter):

    @property
    def damagetype(self):
        return self._raw_data.get('damagetype', "64")


class filter_enemy(BaseFilter):
    icon_sprite = "editor/filter_class.vmt"

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)

    @property
    def filter_radius(self):
        return parse_source_value(self._raw_data.get('filter_radius', 0))

    @property
    def filter_outer_radius(self):
        return parse_source_value(self._raw_data.get('filter_outer_radius', 0))

    @property
    def filter_max_per_enemy(self):
        return parse_source_value(self._raw_data.get('filter_max_per_enemy', 0))


class point_anglesensor(Parentname, Targetname, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def lookatname(self):
        return self._raw_data.get('lookatname', None)

    @property
    def duration(self):
        return parse_source_value(self._raw_data.get('duration', None))

    @property
    def tolerance(self):
        return parse_source_value(self._raw_data.get('tolerance', None))


class point_angularvelocitysensor(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def threshold(self):
        return parse_source_value(self._raw_data.get('threshold', 0))

    @property
    def fireinterval(self):
        return parse_source_value(self._raw_data.get('fireinterval', 0.2))

    @property
    def axis(self):
        return self._raw_data.get('axis', None)

    @property
    def usehelper(self):
        return self._raw_data.get('usehelper', "0")


class point_velocitysensor(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def axis(self):
        return self._raw_data.get('axis', None)

    @property
    def enabled(self):
        return self._raw_data.get('enabled', "1")


class point_proximity_sensor(Parentname, Angles, Targetname, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)


class point_teleport(Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)


class point_hurt(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def DamageTarget(self):
        return self._raw_data.get('damagetarget', "")

    @property
    def DamageRadius(self):
        return parse_source_value(self._raw_data.get('damageradius', 256))

    @property
    def Damage(self):
        return parse_source_value(self._raw_data.get('damage', 5))

    @property
    def DamageDelay(self):
        return parse_source_value(self._raw_data.get('damagedelay', 1))

    @property
    def DamageType(self):
        return self._raw_data.get('damagetype', "0")


class point_playermoveconstraint(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 256))

    @property
    def width(self):
        return parse_source_value(self._raw_data.get('width', 75.0))

    @property
    def speedfactor(self):
        return parse_source_value(self._raw_data.get('speedfactor', 0.15))


class func_physbox(BreakableBrush, Origin, RenderFields):

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)

    @property
    def Damagetype(self):
        return self._raw_data.get('damagetype', "0")

    @property
    def massScale(self):
        return parse_source_value(self._raw_data.get('massscale', 0))

    @property
    def overridescript(self):
        return self._raw_data.get('overridescript', "")

    @property
    def damagetoenablemotion(self):
        return parse_source_value(self._raw_data.get('damagetoenablemotion', 0))

    @property
    def forcetoenablemotion(self):
        return parse_source_value(self._raw_data.get('forcetoenablemotion', 0))

    @property
    def preferredcarryangles(self):
        return parse_float_vector(self._raw_data.get('preferredcarryangles', "0 0 0"))

    @property
    def notsolid(self):
        return self._raw_data.get('notsolid', "0")


class TwoObjectPhysics(Targetname):

    @property
    def attach1(self):
        return self._raw_data.get('attach1', "")

    @property
    def attach2(self):
        return self._raw_data.get('attach2', "")

    @property
    def constraintsystem(self):
        return self._raw_data.get('constraintsystem', "")

    @property
    def forcelimit(self):
        return parse_source_value(self._raw_data.get('forcelimit', 0))

    @property
    def torquelimit(self):
        return parse_source_value(self._raw_data.get('torquelimit', 0))

    @property
    def breaksound(self):
        return self._raw_data.get('breaksound', "")

    @property
    def teleportfollowdistance(self):
        return parse_source_value(self._raw_data.get('teleportfollowdistance', 0))


class phys_constraintsystem(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def additionaliterations(self):
        return parse_source_value(self._raw_data.get('additionaliterations', 0))


class phys_keepupright(Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def attach1(self):
        return self._raw_data.get('attach1', "")

    @property
    def angularlimit(self):
        return parse_source_value(self._raw_data.get('angularlimit', 15))


class physics_cannister(Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/fire_equipment/w_weldtank.mdl")

    @property
    def expdamage(self):
        return self._raw_data.get('expdamage', "200.0")

    @property
    def expradius(self):
        return self._raw_data.get('expradius', "250.0")

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 25))

    @property
    def thrust(self):
        return self._raw_data.get('thrust', "3000.0")

    @property
    def fuel(self):
        return self._raw_data.get('fuel', "12.0")

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "255 255 255"))

    @property
    def renderamt(self):
        return parse_source_value(self._raw_data.get('renderamt', 128))

    @property
    def gassound(self):
        return self._raw_data.get('gassound', "ambient/objects/cannister_loop.wav")


class info_constraint_anchor(Parentname, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def massScale(self):
        return parse_source_value(self._raw_data.get('massscale', 1))


class info_mass_center(Base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', "")


class phys_spring(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def attach1(self):
        return self._raw_data.get('attach1', "")

    @property
    def attach2(self):
        return self._raw_data.get('attach2', "")

    @property
    def springaxis(self):
        return self._raw_data.get('springaxis', "")

    @property
    def length(self):
        return self._raw_data.get('length', "0")

    @property
    def constant(self):
        return self._raw_data.get('constant', "50")

    @property
    def damping(self):
        return self._raw_data.get('damping', "2.0")

    @property
    def relativedamping(self):
        return self._raw_data.get('relativedamping', "0.1")

    @property
    def breaklength(self):
        return self._raw_data.get('breaklength', "0")


class phys_hinge(TwoObjectPhysics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def hingefriction(self):
        return parse_source_value(self._raw_data.get('hingefriction', 0))

    @property
    def hingeaxis(self):
        return self._raw_data.get('hingeaxis', None)

    @property
    def SystemLoadScale(self):
        return parse_source_value(self._raw_data.get('systemloadscale', 1))

    @property
    def minSoundThreshold(self):
        return parse_source_value(self._raw_data.get('minsoundthreshold', 6))

    @property
    def maxSoundThreshold(self):
        return parse_source_value(self._raw_data.get('maxsoundthreshold', 80))

    @property
    def slidesoundfwd(self):
        return self._raw_data.get('slidesoundfwd', "")

    @property
    def slidesoundback(self):
        return self._raw_data.get('slidesoundback', "")

    @property
    def reversalsoundthresholdSmall(self):
        return parse_source_value(self._raw_data.get('reversalsoundthresholdsmall', 0))

    @property
    def reversalsoundthresholdMedium(self):
        return parse_source_value(self._raw_data.get('reversalsoundthresholdmedium', 0))

    @property
    def reversalsoundthresholdLarge(self):
        return parse_source_value(self._raw_data.get('reversalsoundthresholdlarge', 0))

    @property
    def reversalsoundSmall(self):
        return self._raw_data.get('reversalsoundsmall', "")

    @property
    def reversalsoundMedium(self):
        return self._raw_data.get('reversalsoundmedium', "")

    @property
    def reversalsoundLarge(self):
        return self._raw_data.get('reversalsoundlarge', "")


class phys_ballsocket(TwoObjectPhysics):
    icon_sprite = "editor/phys_ballsocket.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class phys_constraint(TwoObjectPhysics):
    model_ = "models/editor/axis_helper.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class phys_pulleyconstraint(TwoObjectPhysics):
    model_ = "models/editor/axis_helper.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def addlength(self):
        return parse_source_value(self._raw_data.get('addlength', 0))

    @property
    def gearratio(self):
        return parse_source_value(self._raw_data.get('gearratio', 1))

    @property
    def position2(self):
        return self._raw_data.get('position2', None)


class phys_slideconstraint(TwoObjectPhysics):
    model_ = "models/editor/axis_helper.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def slideaxis(self):
        return self._raw_data.get('slideaxis', None)

    @property
    def slidefriction(self):
        return parse_source_value(self._raw_data.get('slidefriction', 0))

    @property
    def SystemLoadScale(self):
        return parse_source_value(self._raw_data.get('systemloadscale', 1))

    @property
    def minSoundThreshold(self):
        return parse_source_value(self._raw_data.get('minsoundthreshold', 6))

    @property
    def maxSoundThreshold(self):
        return parse_source_value(self._raw_data.get('maxsoundthreshold', 80))

    @property
    def slidesoundfwd(self):
        return self._raw_data.get('slidesoundfwd', "")

    @property
    def slidesoundback(self):
        return self._raw_data.get('slidesoundback', "")

    @property
    def reversalsoundthresholdSmall(self):
        return parse_source_value(self._raw_data.get('reversalsoundthresholdsmall', 0))

    @property
    def reversalsoundthresholdMedium(self):
        return parse_source_value(self._raw_data.get('reversalsoundthresholdmedium', 0))

    @property
    def reversalsoundthresholdLarge(self):
        return parse_source_value(self._raw_data.get('reversalsoundthresholdlarge', 0))

    @property
    def reversalsoundSmall(self):
        return self._raw_data.get('reversalsoundsmall', "")

    @property
    def reversalsoundMedium(self):
        return self._raw_data.get('reversalsoundmedium', "")

    @property
    def reversalsoundLarge(self):
        return self._raw_data.get('reversalsoundlarge', "")


class phys_lengthconstraint(TwoObjectPhysics):
    model_ = "models/editor/axis_helper.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def addlength(self):
        return parse_source_value(self._raw_data.get('addlength', 0))

    @property
    def minlength(self):
        return parse_source_value(self._raw_data.get('minlength', 0))

    @property
    def attachpoint(self):
        return self._raw_data.get('attachpoint', None)  # Set to none due to bug in BlackMesa base.fgd file


class phys_ragdollconstraint(TwoObjectPhysics):
    model_ = "models/editor/axis_helper.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def xmin(self):
        return parse_source_value(self._raw_data.get('xmin', -90))

    @property
    def xmax(self):
        return parse_source_value(self._raw_data.get('xmax', 90))

    @property
    def ymin(self):
        return parse_source_value(self._raw_data.get('ymin', 0))

    @property
    def ymax(self):
        return parse_source_value(self._raw_data.get('ymax', 0))

    @property
    def zmin(self):
        return parse_source_value(self._raw_data.get('zmin', 0))

    @property
    def zmax(self):
        return parse_source_value(self._raw_data.get('zmax', 0))

    @property
    def xfriction(self):
        return parse_source_value(self._raw_data.get('xfriction', 0))

    @property
    def yfriction(self):
        return parse_source_value(self._raw_data.get('yfriction', 0))

    @property
    def zfriction(self):
        return parse_source_value(self._raw_data.get('zfriction', 0))


class phys_convert(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def swapmodel(self):
        return self._raw_data.get('swapmodel', None)

    @property
    def massoverride(self):
        return parse_source_value(self._raw_data.get('massoverride', 0))


class ForceController(Targetname):

    @property
    def attach1(self):
        return self._raw_data.get('attach1', "")

    @property
    def forcetime(self):
        return self._raw_data.get('forcetime', "0")


class phys_thruster(Angles, ForceController):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def force(self):
        return self._raw_data.get('force', "0")


class phys_torque(ForceController):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def force(self):
        return self._raw_data.get('force', "0")

    @property
    def axis(self):
        return self._raw_data.get('axis', "")


class phys_motor(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def speed(self):
        return self._raw_data.get('speed', "0")

    @property
    def spinup(self):
        return self._raw_data.get('spinup', "1")

    @property
    def inertiafactor(self):
        return parse_source_value(self._raw_data.get('inertiafactor', 1.0))

    @property
    def axis(self):
        return self._raw_data.get('axis', "")

    @property
    def attach1(self):
        return self._raw_data.get('attach1', "")


class phys_magnet(Parentname, Angles, Studiomodel, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def forcelimit(self):
        return parse_source_value(self._raw_data.get('forcelimit', 0))

    @property
    def torquelimit(self):
        return parse_source_value(self._raw_data.get('torquelimit', 0))

    @property
    def massScale(self):
        return parse_source_value(self._raw_data.get('massscale', 0))

    @property
    def overridescript(self):
        return self._raw_data.get('overridescript', "")

    @property
    def maxobjects(self):
        return parse_source_value(self._raw_data.get('maxobjects', 0))


class prop_detail_base(Base):

    @property
    def model(self):
        return self._raw_data.get('model', None)


class prop_static_base(DXLevelChoice, Angles):

    @property
    def model(self):
        return self._raw_data.get('model', None)

    @property
    def skin(self):
        return parse_source_value(self._raw_data.get('skin', 0))

    @property
    def solid(self):
        return self._raw_data.get('solid', "6")

    @property
    def disableshadows(self):
        return self._raw_data.get('disableshadows', "0")

    @property
    def screenspacefade(self):
        return self._raw_data.get('screenspacefade', "0")

    @property
    def fademindist(self):
        return parse_source_value(self._raw_data.get('fademindist', -1))

    @property
    def fademaxdist(self):
        return parse_source_value(self._raw_data.get('fademaxdist', 0))

    @property
    def fadescale(self):
        return parse_source_value(self._raw_data.get('fadescale', 1))

    @property
    def lightingorigin(self):
        return self._raw_data.get('lightingorigin', "")

    @property
    def disablevertexlighting(self):
        return self._raw_data.get('disablevertexlighting', "0")

    @property
    def disableselfshadowing(self):
        return self._raw_data.get('disableselfshadowing', "0")

    @property
    def ignorenormals(self):
        return self._raw_data.get('ignorenormals', "0")


class BaseFadeProp(Base):

    @property
    def fademindist(self):
        return parse_source_value(self._raw_data.get('fademindist', -1))

    @property
    def fademaxdist(self):
        return parse_source_value(self._raw_data.get('fademaxdist', 0))

    @property
    def fadescale(self):
        return parse_source_value(self._raw_data.get('fadescale', 1))


class prop_dynamic_base(Angles, BaseFadeProp, Parentname, BreakableProp, Studiomodel, Global, DXLevelChoice,
                        RenderFields):

    @property
    def solid(self):
        return self._raw_data.get('solid', "6")

    @property
    def DefaultAnim(self):
        return self._raw_data.get('defaultanim', "")

    @property
    def RandomAnimation(self):
        return self._raw_data.get('randomanimation', "0")

    @property
    def MinAnimTime(self):
        return parse_source_value(self._raw_data.get('minanimtime', 5))

    @property
    def MaxAnimTime(self):
        return parse_source_value(self._raw_data.get('maxanimtime', 10))

    @property
    def SetBodyGroup(self):
        return parse_source_value(self._raw_data.get('setbodygroup', 0))

    @property
    def DisableBoneFollowers(self):
        return self._raw_data.get('disablebonefollowers', "0")

    @property
    def lightingorigin(self):
        return self._raw_data.get('lightingorigin', "")


class prop_detail(prop_detail_base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class prop_static(prop_static_base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class prop_dynamic(prop_dynamic_base, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class prop_dynamic_override(prop_dynamic_base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 0))


class BasePropPhysics(Angles, BaseFadeProp, Studiomodel, BreakableProp, Global, DXLevelChoice):

    @property
    def minhealthdmg(self):
        return parse_source_value(self._raw_data.get('minhealthdmg', 0))

    @property
    def shadowcastdist(self):
        return parse_source_value(self._raw_data.get('shadowcastdist', 0))

    @property
    def physdamagescale(self):
        return parse_source_value(self._raw_data.get('physdamagescale', 0.1))

    @property
    def Damagetype(self):
        return self._raw_data.get('damagetype', "0")

    @property
    def nodamageforces(self):
        return self._raw_data.get('nodamageforces', "0")

    @property
    def inertiaScale(self):
        return parse_source_value(self._raw_data.get('inertiascale', 1.0))

    @property
    def massScale(self):
        return parse_source_value(self._raw_data.get('massscale', 0))

    @property
    def overridescript(self):
        return self._raw_data.get('overridescript', "")

    @property
    def damagetoenablemotion(self):
        return parse_source_value(self._raw_data.get('damagetoenablemotion', 0))

    @property
    def forcetoenablemotion(self):
        return parse_source_value(self._raw_data.get('forcetoenablemotion', 0))

    @property
    def puntsound(self):
        return self._raw_data.get('puntsound', None)


class prop_physics_override(BasePropPhysics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 0))


class prop_physics(BasePropPhysics, RenderFields):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class prop_physics_multiplayer(prop_physics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def physicsmode(self):
        return self._raw_data.get('physicsmode', "0")


class prop_ragdoll(Angles, BaseFadeProp, Studiomodel, Targetname, DXLevelChoice, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def angleOverride(self):
        return self._raw_data.get('angleoverride', "")


class prop_dynamic_ornament(prop_dynamic_base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def InitialOwner(self):
        return self._raw_data.get('initialowner', None)


class func_areaportal(Targetname):

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def StartOpen(self):
        return self._raw_data.get('startopen', "1")

    @property
    def PortalVersion(self):
        return parse_source_value(self._raw_data.get('portalversion', 1))


class func_occluder(Targetname):

    @property
    def StartActive(self):
        return self._raw_data.get('startactive', "1")


class func_breakable(BreakableBrush, Origin, RenderFields):

    @property
    def minhealthdmg(self):
        return parse_source_value(self._raw_data.get('minhealthdmg', 0))

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)

    @property
    def physdamagescale(self):
        return parse_source_value(self._raw_data.get('physdamagescale', 1.0))


class func_breakable_surf(BreakableBrush, RenderFields):

    @property
    def fragility(self):
        return parse_source_value(self._raw_data.get('fragility', 100))

    @property
    def surfacetype(self):
        return self._raw_data.get('surfacetype', "0")


class func_conveyor(Parentname, Targetname, Shadow, RenderFields):

    @property
    def movedir(self):
        return parse_float_vector(self._raw_data.get('movedir', "0 0 0"))

    @property
    def speed(self):
        return self._raw_data.get('speed', "100")

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)


class func_detail(DXLevelChoice):
    pass


class func_viscluster(Base):
    pass


class func_illusionary(Parentname, Shadow, Targetname, Origin, RenderFields):

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)


class func_precipitation(Parentname, Targetname):

    @property
    def renderamt(self):
        return parse_source_value(self._raw_data.get('renderamt', 5))

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "100 100 100"))

    @property
    def preciptype(self):
        return self._raw_data.get('preciptype', "0")


class func_wall_toggle(func_wall):
    pass


class func_guntarget(Parentname, Targetname, Global, RenderFields):

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 100))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 0))

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)


class func_fish_pool(Base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/Junkola.mdl")

    @property
    def fish_count(self):
        return parse_source_value(self._raw_data.get('fish_count', 10))

    @property
    def max_range(self):
        return parse_source_value(self._raw_data.get('max_range', 150))


class PlatSounds(Base):

    @property
    def movesnd(self):
        return self._raw_data.get('movesnd', "0")

    @property
    def stopsnd(self):
        return self._raw_data.get('stopsnd', "0")

    @property
    def volume(self):
        return self._raw_data.get('volume', "0.85")


class Trackchange(Parentname, PlatSounds, Global, Targetname, RenderFields):

    @property
    def height(self):
        return parse_source_value(self._raw_data.get('height', 0))

    @property
    def rotation(self):
        return parse_source_value(self._raw_data.get('rotation', 0))

    @property
    def train(self):
        return self._raw_data.get('train', None)

    @property
    def toptrack(self):
        return self._raw_data.get('toptrack', None)

    @property
    def bottomtrack(self):
        return self._raw_data.get('bottomtrack', None)

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 0))


class BaseTrain(Parentname, Global, Shadow, Targetname, Origin, RenderFields):

    @property
    def target(self):
        return self._raw_data.get('target', "")

    @property
    def startspeed(self):
        return parse_source_value(self._raw_data.get('startspeed', 100))

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 0))

    @property
    def velocitytype(self):
        return self._raw_data.get('velocitytype', "0")

    @property
    def orientationtype(self):
        return self._raw_data.get('orientationtype', "1")

    @property
    def wheels(self):
        return parse_source_value(self._raw_data.get('wheels', 50))

    @property
    def height(self):
        return parse_source_value(self._raw_data.get('height', 4))

    @property
    def bank(self):
        return self._raw_data.get('bank', "0")

    @property
    def dmg(self):
        return parse_source_value(self._raw_data.get('dmg', 0))

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)

    @property
    def MoveSound(self):
        return self._raw_data.get('movesound', "")

    @property
    def MovePingSound(self):
        return self._raw_data.get('movepingsound', "")

    @property
    def StartSound(self):
        return self._raw_data.get('startsound', "")

    @property
    def StopSound(self):
        return self._raw_data.get('stopsound', "")

    @property
    def volume(self):
        return parse_source_value(self._raw_data.get('volume', 10))

    @property
    def MoveSoundMinPitch(self):
        return parse_source_value(self._raw_data.get('movesoundminpitch', 60))

    @property
    def MoveSoundMaxPitch(self):
        return parse_source_value(self._raw_data.get('movesoundmaxpitch', 200))

    @property
    def MoveSoundMinTime(self):
        return parse_source_value(self._raw_data.get('movesoundmintime', 0))

    @property
    def MoveSoundMaxTime(self):
        return parse_source_value(self._raw_data.get('movesoundmaxtime', 0))


class func_trackautochange(Trackchange):

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)


class func_trackchange(Trackchange):

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)


class func_tracktrain(BaseTrain):

    @property
    def ManualSpeedChanges(self):
        return self._raw_data.get('manualspeedchanges', "0")

    @property
    def ManualAccelSpeed(self):
        return parse_source_value(self._raw_data.get('manualaccelspeed', 0))

    @property
    def ManualDecelSpeed(self):
        return parse_source_value(self._raw_data.get('manualdecelspeed', 0))


class func_tanktrain(BaseTrain):

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 100))


class func_traincontrols(Parentname, Global):

    @property
    def target(self):
        return self._raw_data.get('target', None)


class tanktrain_aitarget(Targetname):
    icon_sprite = "editor/tanktrain_aitarget.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def newtarget(self):
        return self._raw_data.get('newtarget', None)


class tanktrain_ai(Targetname):
    icon_sprite = "editor/tanktrain_ai.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def startsound(self):
        return self._raw_data.get('startsound', "vehicles/diesel_start1.wav")

    @property
    def enginesound(self):
        return self._raw_data.get('enginesound', "vehicles/diesel_turbo_loop1.wav")

    @property
    def movementsound(self):
        return self._raw_data.get('movementsound', "vehicles/tank_treads_loop1.wav")


class path_track(Parentname, Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def altpath(self):
        return self._raw_data.get('altpath', None)

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 0))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 0))

    @property
    def orientationtype(self):
        return self._raw_data.get('orientationtype', "1")


class test_traceline(Angles):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class trigger_autosave(Targetname):

    @property
    def master(self):
        return self._raw_data.get('master', None)

    @property
    def NewLevelUnit(self):
        return self._raw_data.get('newlevelunit', "0")

    @property
    def DangerousTimer(self):
        return parse_source_value(self._raw_data.get('dangeroustimer', 0))

    @property
    def MinimumHitPoints(self):
        return parse_source_value(self._raw_data.get('minimumhitpoints', 0))


class trigger_changelevel(EnableDisable):

    @property
    def targetname(self):
        return self._raw_data.get('targetname', None)

    @property
    def map(self):
        return self._raw_data.get('map', None)

    @property
    def landmark(self):
        return self._raw_data.get('landmark', None)


class trigger_gravity(Trigger):

    @property
    def gravity(self):
        return parse_source_value(self._raw_data.get('gravity', 1))


class trigger_playermovement(Trigger):
    pass


class trigger_soundscape(Trigger):

    @property
    def soundscape(self):
        return self._raw_data.get('soundscape', None)


class trigger_hurt(Trigger):

    @property
    def master(self):
        return self._raw_data.get('master', None)

    @property
    def damage(self):
        return parse_source_value(self._raw_data.get('damage', 10))

    @property
    def damagecap(self):
        return parse_source_value(self._raw_data.get('damagecap', 20))

    @property
    def damagetype(self):
        return self._raw_data.get('damagetype', "0")

    @property
    def damagemodel(self):
        return self._raw_data.get('damagemodel', "0")

    @property
    def nodmgforce(self):
        return self._raw_data.get('nodmgforce', "0")


class trigger_remove(Trigger):
    pass


class trigger_multiple(Trigger):

    @property
    def wait(self):
        return parse_source_value(self._raw_data.get('wait', 1))


class trigger_once(TriggerOnce):
    pass


class trigger_look(Trigger):

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def LookTime(self):
        return self._raw_data.get('looktime', "0.5")

    @property
    def FieldOfView(self):
        return self._raw_data.get('fieldofview', "0.9")

    @property
    def Timeout(self):
        return parse_source_value(self._raw_data.get('timeout', 0))


class trigger_push(Trigger):

    @property
    def pushdir(self):
        return parse_float_vector(self._raw_data.get('pushdir', "0 0 0"))

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 40))

    @property
    def alternateticksfix(self):
        return parse_source_value(self._raw_data.get('alternateticksfix', 0))


class trigger_wind(Angles, Trigger):

    @property
    def Speed(self):
        return parse_source_value(self._raw_data.get('speed', 200))

    @property
    def SpeedNoise(self):
        return parse_source_value(self._raw_data.get('speednoise', 0))

    @property
    def DirectionNoise(self):
        return parse_source_value(self._raw_data.get('directionnoise', 10))

    @property
    def HoldTime(self):
        return parse_source_value(self._raw_data.get('holdtime', 0))

    @property
    def HoldNoise(self):
        return parse_source_value(self._raw_data.get('holdnoise', 0))


class trigger_impact(Angles, Targetname, Origin):

    @property
    def Magnitude(self):
        return parse_source_value(self._raw_data.get('magnitude', 200))

    @property
    def noise(self):
        return parse_source_value(self._raw_data.get('noise', 0.1))

    @property
    def viewkick(self):
        return parse_source_value(self._raw_data.get('viewkick', 0.05))


class trigger_proximity(Trigger):

    @property
    def measuretarget(self):
        return self._raw_data.get('measuretarget', None)

    @property
    def radius(self):
        return self._raw_data.get('radius', "256")


class trigger_teleport(Trigger):

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def landmark(self):
        return self._raw_data.get('landmark', None)


class trigger_transition(Targetname):
    pass


class trigger_serverragdoll(Targetname):
    pass


class ai_speechfilter(ResponseContext, Targetname, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def subject(self):
        return self._raw_data.get('subject', "")

    @property
    def IdleModifier(self):
        return parse_source_value(self._raw_data.get('idlemodifier', 1.0))

    @property
    def NeverSayHello(self):
        return self._raw_data.get('neversayhello', "0")


class water_lod_control(Targetname):
    icon_sprite = "editor/waterlodcontrol.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def cheapwaterstartdistance(self):
        return parse_source_value(self._raw_data.get('cheapwaterstartdistance', 1000))

    @property
    def cheapwaterenddistance(self):
        return parse_source_value(self._raw_data.get('cheapwaterenddistance', 2000))


class info_camera_link(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def PointCamera(self):
        return self._raw_data.get('pointcamera', None)


class logic_measure_movement(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def MeasureTarget(self):
        return self._raw_data.get('measuretarget', "")

    @property
    def MeasureReference(self):
        return self._raw_data.get('measurereference', "")

    @property
    def Target(self):
        return self._raw_data.get('target', "")

    @property
    def TargetReference(self):
        return self._raw_data.get('targetreference', "")

    @property
    def TargetScale(self):
        return parse_source_value(self._raw_data.get('targetscale', 1))

    @property
    def MeasureType(self):
        return self._raw_data.get('measuretype', "0")


class npc_furniture(Parentname, BaseNPC):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', None)


class env_credits(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class material_modify_control(Parentname, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def materialName(self):
        return self._raw_data.get('materialname', None)

    @property
    def materialVar(self):
        return self._raw_data.get('materialvar', None)


class point_devshot_camera(Angles):
    viewport_model = "models/editor/camera.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def cameraname(self):
        return self._raw_data.get('cameraname', "")

    @property
    def FOV(self):
        return parse_source_value(self._raw_data.get('fov', 75))


class logic_playerproxy(DamageFilter, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class env_spritetrail(Parentname, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def lifetime(self):
        return parse_source_value(self._raw_data.get('lifetime', 0.5))

    @property
    def startwidth(self):
        return parse_source_value(self._raw_data.get('startwidth', 8.0))

    @property
    def endwidth(self):
        return parse_source_value(self._raw_data.get('endwidth', 1.0))

    @property
    def spritename(self):
        return self._raw_data.get('spritename', "sprites/bluelaser1.vmt")

    @property
    def renderamt(self):
        return parse_source_value(self._raw_data.get('renderamt', 255))

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "255 255 255"))

    @property
    def rendermode(self):
        return self._raw_data.get('rendermode', "5")


class env_projectedtexture(Parentname, Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def lightfov(self):
        return parse_source_value(self._raw_data.get('lightfov', 90.0))

    @property
    def nearz(self):
        return parse_source_value(self._raw_data.get('nearz', 4.0))

    @property
    def farz(self):
        return parse_source_value(self._raw_data.get('farz', 750.0))

    @property
    def enableshadows(self):
        return self._raw_data.get('enableshadows', "0")

    @property
    def shadowquality(self):
        return self._raw_data.get('shadowquality', "1")

    @property
    def lightonlytarget(self):
        return self._raw_data.get('lightonlytarget', "0")

    @property
    def lightworld(self):
        return self._raw_data.get('lightworld', "1")

    @property
    def lightcolor(self):
        return parse_int_vector(self._raw_data.get('lightcolor', "255 255 255 200"))

    @property
    def cameraspace(self):
        return parse_source_value(self._raw_data.get('cameraspace', 0))


class func_reflective_glass(func_brush):
    pass


class env_particle_performance_monitor(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class npc_puppet(Parentname, Studiomodel, BaseNPC):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def animationtarget(self):
        return self._raw_data.get('animationtarget', "")

    @property
    def attachmentname(self):
        return self._raw_data.get('attachmentname', "")


class point_gamestats_counter(Targetname, Origin, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def Name(self):
        return self._raw_data.get('name', None)


class func_instance(Angles):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def targetname(self):
        return self._raw_data.get('targetname', None)

    @property
    def file(self):
        return self._raw_data.get('file', None)

    @property
    def fixup_style(self):
        return self._raw_data.get('fixup_style', "0")

    @property
    def replace01(self):
        return self._raw_data.get('replace01', None)

    @property
    def replace02(self):
        return self._raw_data.get('replace02', None)

    @property
    def replace03(self):
        return self._raw_data.get('replace03', None)

    @property
    def replace04(self):
        return self._raw_data.get('replace04', None)

    @property
    def replace05(self):
        return self._raw_data.get('replace05', None)

    @property
    def replace06(self):
        return self._raw_data.get('replace06', None)

    @property
    def replace07(self):
        return self._raw_data.get('replace07', None)

    @property
    def replace08(self):
        return self._raw_data.get('replace08', None)

    @property
    def replace09(self):
        return self._raw_data.get('replace09', None)

    @property
    def replace10(self):
        return self._raw_data.get('replace10', None)


class func_instance_parms(Base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def parm1(self):
        return self._raw_data.get('parm1', None)

    @property
    def parm2(self):
        return self._raw_data.get('parm2', None)

    @property
    def parm3(self):
        return self._raw_data.get('parm3', None)

    @property
    def parm4(self):
        return self._raw_data.get('parm4', None)

    @property
    def parm5(self):
        return self._raw_data.get('parm5', None)

    @property
    def parm6(self):
        return self._raw_data.get('parm6', None)

    @property
    def parm7(self):
        return self._raw_data.get('parm7', None)

    @property
    def parm8(self):
        return self._raw_data.get('parm8', None)

    @property
    def parm9(self):
        return self._raw_data.get('parm9', None)

    @property
    def parm10(self):
        return self._raw_data.get('parm10', None)


class TalkNPC(BaseNPC):

    @property
    def UseSentence(self):
        return self._raw_data.get('usesentence', None)

    @property
    def UnUseSentence(self):
        return self._raw_data.get('unusesentence', None)

    @property
    def DontUseSpeechSemaphore(self):
        return self._raw_data.get('dontusespeechsemaphore', "0")


class PlayerCompanion(BaseNPC):

    @property
    def AlwaysTransition(self):
        return self._raw_data.get('alwaystransition', "No")

    @property
    def DontPickupWeapons(self):
        return self._raw_data.get('dontpickupweapons', "No")

    @property
    def GameEndAlly(self):
        return self._raw_data.get('gameendally', "No")


class RappelNPC(BaseNPC):

    @property
    def waitingtorappel(self):
        return self._raw_data.get('waitingtorappel', "No")


class AlyxInteractable(Base):
    pass


class CombineBallSpawners(Angles, Targetname, Origin, Global):

    @property
    def ballcount(self):
        return parse_source_value(self._raw_data.get('ballcount', 3))

    @property
    def minspeed(self):
        return parse_source_value(self._raw_data.get('minspeed', 300.0))

    @property
    def maxspeed(self):
        return parse_source_value(self._raw_data.get('maxspeed', 600.0))

    @property
    def ballradius(self):
        return parse_source_value(self._raw_data.get('ballradius', 20.0))

    @property
    def balltype(self):
        return self._raw_data.get('balltype', "Combine Energy Ball 1")

    @property
    def ballrespawntime(self):
        return parse_source_value(self._raw_data.get('ballrespawntime', 4.0))


class prop_combine_ball(BasePropPhysics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class trigger_physics_trap(Angles, Trigger):

    @property
    def dissolvetype(self):
        return self._raw_data.get('dissolvetype', "Energy")


class trigger_weapon_dissolve(Trigger):

    @property
    def emittername(self):
        return self._raw_data.get('emittername', "")


class trigger_weapon_strip(Trigger):

    @property
    def KillWeapons(self):
        return self._raw_data.get('killweapons', "No")


class func_combine_ball_spawner(CombineBallSpawners):
    pass


class point_combine_ball_launcher(CombineBallSpawners):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def launchconenoise(self):
        return parse_source_value(self._raw_data.get('launchconenoise', 0.0))

    @property
    def bullseyename(self):
        return self._raw_data.get('bullseyename', "")

    @property
    def maxballbounces(self):
        return parse_source_value(self._raw_data.get('maxballbounces', 8))


class npc_blob(BaseNPC):
    model_ = "models/combine_soldier.mdl"
    pass


class npc_grenade_frag(BaseNPC):
    model_ = "models/Weapons/w_grenade.mdl"
    pass


class npc_combine_cannon(BaseNPC):
    model_ = "models/combine_soldier.mdl"

    @property
    def sightdist(self):
        return parse_source_value(self._raw_data.get('sightdist', 1024))


class npc_combine_camera(BaseNPC):
    model_ = "models/combine_camera/combine_camera.mdl"

    @property
    def innerradius(self):
        return parse_source_value(self._raw_data.get('innerradius', 300))

    @property
    def outerradius(self):
        return parse_source_value(self._raw_data.get('outerradius', 450))

    @property
    def minhealthdmg(self):
        return parse_source_value(self._raw_data.get('minhealthdmg', 0))

    @property
    def defaulttarget(self):
        return self._raw_data.get('defaulttarget', "")


class npc_turret_ground(Parentname, BaseNPC, AlyxInteractable):
    model_ = "models/combine_turrets/ground_turret.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class npc_turret_ceiling(Angles, Studiomodel, Targetname):
    model_ = "models/combine_turrets/ceiling_turret.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def minhealthdmg(self):
        return parse_source_value(self._raw_data.get('minhealthdmg', 0))


class npc_turret_floor(Angles, Targetname):
    model_ = "models/combine_turrets/floor_turret.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def SkinNumber(self):
        return parse_source_value(self._raw_data.get('skinnumber', 0))


class VehicleDriverNPC(BaseNPC):

    @property
    def vehicle(self):
        return self._raw_data.get('vehicle', None)


class npc_vehicledriver(VehicleDriverNPC):
    model_ = "models/roller.mdl"

    @property
    def drivermaxspeed(self):
        return parse_source_value(self._raw_data.get('drivermaxspeed', 1))

    @property
    def driverminspeed(self):
        return parse_source_value(self._raw_data.get('driverminspeed', 0))


class npc_cranedriver(VehicleDriverNPC):
    model_ = "models/roller.mdl"

    @property
    def releasepause(self):
        return parse_source_value(self._raw_data.get('releasepause', 0))


class npc_apcdriver(VehicleDriverNPC):
    model_ = "models/roller.mdl"

    @property
    def drivermaxspeed(self):
        return parse_source_value(self._raw_data.get('drivermaxspeed', 1))

    @property
    def driverminspeed(self):
        return parse_source_value(self._raw_data.get('driverminspeed', 0))


class npc_rollermine(BaseNPC, AlyxInteractable):
    model_ = "models/roller.mdl"

    @property
    def startburied(self):
        return self._raw_data.get('startburied', "No")

    @property
    def uniformsightdist(self):
        return self._raw_data.get('uniformsightdist', "0")


class npc_missiledefense(BaseNPC):
    model_ = "models/missile_defense.mdl"
    pass


class npc_sniper(BaseNPC):
    model_ = "models/combine_soldier.mdl"

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 0))

    @property
    def misses(self):
        return parse_source_value(self._raw_data.get('misses', 0))

    @property
    def beambrightness(self):
        return parse_source_value(self._raw_data.get('beambrightness', 100))

    @property
    def shootZombiesInChest(self):
        return self._raw_data.get('shootzombiesinchest', "0")

    @property
    def shielddistance(self):
        return parse_source_value(self._raw_data.get('shielddistance', 64))

    @property
    def shieldradius(self):
        return parse_source_value(self._raw_data.get('shieldradius', 48))

    @property
    def PaintInterval(self):
        return parse_source_value(self._raw_data.get('paintinterval', 1))

    @property
    def PaintIntervalVariance(self):
        return parse_source_value(self._raw_data.get('paintintervalvariance', 0.75))


class info_radar_target(Parentname, Targetname, EnableDisable):
    icon_sprite = "editor/info_target.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 6000))

    @property
    def type(self):
        return self._raw_data.get('type', "0")

    @property
    def mode(self):
        return self._raw_data.get('mode', "0")


class info_target_vehicle_transition(Angles, Targetname, EnableDisable):
    icon_sprite = "editor/info_target.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class info_snipertarget(Parentname, Targetname):
    icon_sprite = "editor/info_target.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 2))

    @property
    def groupname(self):
        return self._raw_data.get('groupname', None)


class prop_thumper(Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/props_combine/CombineThumper002.mdl")

    @property
    def dustscale(self):
        return self._raw_data.get('dustscale', "Small Thumper")

    @property
    def EffectRadius(self):
        return parse_source_value(self._raw_data.get('effectradius', 1000))


class npc_antlion(BaseNPC):
    model_ = "models/antlion.mdl"

    @property
    def startburrowed(self):
        return self._raw_data.get('startburrowed', "No")

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 256))

    @property
    def eludedist(self):
        return parse_source_value(self._raw_data.get('eludedist', 1024))

    @property
    def ignorebugbait(self):
        return self._raw_data.get('ignorebugbait', "No")

    @property
    def unburroweffects(self):
        return self._raw_data.get('unburroweffects', "No")


class npc_antlionguard(BaseNPC):
    model_ = "models/antlion_guard.mdl"

    @property
    def startburrowed(self):
        return self._raw_data.get('startburrowed', "No")

    @property
    def allowbark(self):
        return self._raw_data.get('allowbark', "No")

    @property
    def cavernbreed(self):
        return self._raw_data.get('cavernbreed', "No")

    @property
    def incavern(self):
        return self._raw_data.get('incavern', "No")

    @property
    def shovetargets(self):
        return self._raw_data.get('shovetargets', "")


class npc_crow(BaseNPC):
    model_ = "models/crow.mdl"

    @property
    def deaf(self):
        return self._raw_data.get('deaf', "0")


class npc_seagull(BaseNPC):
    model_ = "models/seagull.mdl"

    @property
    def deaf(self):
        return self._raw_data.get('deaf', "0")


class npc_pigeon(BaseNPC):
    model_ = "models/pigeon.mdl"

    @property
    def deaf(self):
        return self._raw_data.get('deaf', "0")


class npc_ichthyosaur(BaseNPC):
    model_ = "models/ichthyosaur.mdl"
    pass


class BaseHeadcrab(BaseNPC):

    @property
    def startburrowed(self):
        return self._raw_data.get('startburrowed', "No")


class npc_headcrab(Parentname, BaseHeadcrab):
    model_ = "models/Headcrabclassic.mdl"
    pass


class npc_headcrab_fast(BaseHeadcrab):
    model_ = "models/Headcrab.mdl"
    pass


class npc_headcrab_black(BaseHeadcrab):
    model_ = "models/Headcrabblack.mdl"
    pass


class npc_stalker(BaseNPC):
    model_ = "models/Stalker.mdl"

    @property
    def BeamPower(self):
        return self._raw_data.get('beampower', "Low")


class npc_bullseye(Parentname, BaseNPC):
    icon_sprite = "editor/bullseye.vmt"

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 35))

    @property
    def minangle(self):
        return self._raw_data.get('minangle', "360")

    @property
    def mindist(self):
        return self._raw_data.get('mindist', "0")

    @property
    def autoaimradius(self):
        return parse_source_value(self._raw_data.get('autoaimradius', 0))


class npc_enemyfinder(Parentname, BaseNPC):

    @property
    def FieldOfView(self):
        return self._raw_data.get('fieldofview', "0.2")

    @property
    def MinSearchDist(self):
        return parse_source_value(self._raw_data.get('minsearchdist', 0))

    @property
    def MaxSearchDist(self):
        return parse_source_value(self._raw_data.get('maxsearchdist', 2048))

    @property
    def freepass_timetotrigger(self):
        return parse_source_value(self._raw_data.get('freepass_timetotrigger', 0))

    @property
    def freepass_duration(self):
        return parse_source_value(self._raw_data.get('freepass_duration', 0))

    @property
    def freepass_movetolerance(self):
        return parse_source_value(self._raw_data.get('freepass_movetolerance', 120))

    @property
    def freepass_refillrate(self):
        return parse_source_value(self._raw_data.get('freepass_refillrate', 0.5))

    @property
    def freepass_peektime(self):
        return parse_source_value(self._raw_data.get('freepass_peektime', 0))

    @property
    def StartOn(self):
        return self._raw_data.get('starton', "1")


class npc_enemyfinder_combinecannon(Parentname, BaseNPC):

    @property
    def FieldOfView(self):
        return self._raw_data.get('fieldofview', "0.2")

    @property
    def MinSearchDist(self):
        return parse_source_value(self._raw_data.get('minsearchdist', 0))

    @property
    def MaxSearchDist(self):
        return parse_source_value(self._raw_data.get('maxsearchdist', 2048))

    @property
    def SnapToEnt(self):
        return self._raw_data.get('snaptoent', "")

    @property
    def freepass_timetotrigger(self):
        return parse_source_value(self._raw_data.get('freepass_timetotrigger', 0))

    @property
    def freepass_duration(self):
        return parse_source_value(self._raw_data.get('freepass_duration', 0))

    @property
    def freepass_movetolerance(self):
        return parse_source_value(self._raw_data.get('freepass_movetolerance', 120))

    @property
    def freepass_refillrate(self):
        return parse_source_value(self._raw_data.get('freepass_refillrate', 0.5))

    @property
    def freepass_peektime(self):
        return parse_source_value(self._raw_data.get('freepass_peektime', 0))

    @property
    def StartOn(self):
        return self._raw_data.get('starton', "1")


class npc_citizen(Parentname, TalkNPC, PlayerCompanion):

    @property
    def additionalequipment(self):
        return self._raw_data.get('additionalequipment', "0")

    @property
    def ammosupply(self):
        return self._raw_data.get('ammosupply', "SMG1")

    @property
    def ammoamount(self):
        return parse_source_value(self._raw_data.get('ammoamount', 1))

    @property
    def citizentype(self):
        return self._raw_data.get('citizentype', "Default")

    @property
    def expressiontype(self):
        return self._raw_data.get('expressiontype', "Random")

    @property
    def model(self):
        return self._raw_data.get('model', "models/humans/group01/male_01.mdl")

    @property
    def ExpressionOverride(self):
        return self._raw_data.get('expressionoverride', None)

    @property
    def notifynavfailblocked(self):
        return self._raw_data.get('notifynavfailblocked', "0")

    @property
    def neverleaveplayersquad(self):
        return self._raw_data.get('neverleaveplayersquad', "0")

    @property
    def denycommandconcept(self):
        return self._raw_data.get('denycommandconcept', "")


class npc_fisherman(BaseNPC):
    model_ = "models/Barney.mdl"

    @property
    def ExpressionOverride(self):
        return self._raw_data.get('expressionoverride', None)


class npc_barney(TalkNPC, PlayerCompanion):
    model_ = "models/Barney.mdl"

    @property
    def additionalequipment(self):
        return self._raw_data.get('additionalequipment', "weapon_pistol")

    @property
    def ExpressionOverride(self):
        return self._raw_data.get('expressionoverride', None)


class BaseCombine(RappelNPC):

    @property
    def additionalequipment(self):
        return self._raw_data.get('additionalequipment', "weapon_smg1")

    @property
    def NumGrenades(self):
        return self._raw_data.get('numgrenades', "5")


class npc_combine_s(BaseCombine):
    model_ = "models/Combine_Soldier.mdl"

    @property
    def model(self):
        return self._raw_data.get('model', "models/combine_soldier.mdl")

    @property
    def tacticalvariant(self):
        return self._raw_data.get('tacticalvariant', "0")

    @property
    def usemarch(self):
        return self._raw_data.get('usemarch', "0")


class npc_launcher(Parentname, BaseNPC):
    model_ = "models/junk/w_traffcone.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def StartOn(self):
        return self._raw_data.get('starton', "0")

    @property
    def MissileModel(self):
        return self._raw_data.get('missilemodel', "models/Weapons/wscanner_grenade.mdl")

    @property
    def LaunchSound(self):
        return self._raw_data.get('launchsound', "npc/waste_scanner/grenade_fire.wav")

    @property
    def FlySound(self):
        return self._raw_data.get('flysound', "ambient/objects/machine2.wav")

    @property
    def SmokeTrail(self):
        return self._raw_data.get('smoketrail', "1")

    @property
    def LaunchSmoke(self):
        return self._raw_data.get('launchsmoke', "1")

    @property
    def LaunchDelay(self):
        return parse_source_value(self._raw_data.get('launchdelay', 8))

    @property
    def LaunchSpeed(self):
        return self._raw_data.get('launchspeed', "200")

    @property
    def PathCornerName(self):
        return self._raw_data.get('pathcornername', "")

    @property
    def HomingSpeed(self):
        return self._raw_data.get('homingspeed', "0")

    @property
    def HomingStrength(self):
        return parse_source_value(self._raw_data.get('homingstrength', 10))

    @property
    def HomingDelay(self):
        return self._raw_data.get('homingdelay', "0")

    @property
    def HomingRampUp(self):
        return self._raw_data.get('homingrampup', "0.5")

    @property
    def HomingDuration(self):
        return self._raw_data.get('homingduration', "5")

    @property
    def HomingRampDown(self):
        return self._raw_data.get('homingrampdown', "1.0")

    @property
    def Gravity(self):
        return self._raw_data.get('gravity', "1.0")

    @property
    def MinRange(self):
        return parse_source_value(self._raw_data.get('minrange', 100))

    @property
    def MaxRange(self):
        return parse_source_value(self._raw_data.get('maxrange', 2048))

    @property
    def SpinMagnitude(self):
        return self._raw_data.get('spinmagnitude', "0")

    @property
    def SpinSpeed(self):
        return self._raw_data.get('spinspeed', "0")

    @property
    def Damage(self):
        return self._raw_data.get('damage', "50")

    @property
    def DamageRadius(self):
        return self._raw_data.get('damageradius', "200")


class npc_hunter(BaseNPC):
    model_ = "models/hunter.mdl"

    @property
    def FollowTarget(self):
        return self._raw_data.get('followtarget', "")


class npc_hunter_maker(npc_template_maker):
    icon_sprite = "editor/npc_maker.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class npc_advisor(BaseNPC):

    @property
    def model(self):
        return self._raw_data.get('model', "models/advisor.mdl")

    @property
    def levitationarea(self):
        return self._raw_data.get('levitationarea', "")

    @property
    def levitategoal_bottom(self):
        return self._raw_data.get('levitategoal_bottom', "")

    @property
    def levitategoal_top(self):
        return self._raw_data.get('levitategoal_top', "")

    @property
    def staging_ent_names(self):
        return self._raw_data.get('staging_ent_names', "")

    @property
    def priority_grab_name(self):
        return self._raw_data.get('priority_grab_name', "")


class env_sporeexplosion(Parentname, Targetname, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def spawnrate(self):
        return parse_source_value(self._raw_data.get('spawnrate', 25))


class env_gunfire(Parentname, Targetname, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', "")

    @property
    def minburstsize(self):
        return parse_source_value(self._raw_data.get('minburstsize', 2))

    @property
    def maxburstsize(self):
        return parse_source_value(self._raw_data.get('maxburstsize', 7))

    @property
    def minburstdelay(self):
        return parse_source_value(self._raw_data.get('minburstdelay', 2))

    @property
    def maxburstdelay(self):
        return parse_source_value(self._raw_data.get('maxburstdelay', 5))

    @property
    def rateoffire(self):
        return parse_source_value(self._raw_data.get('rateoffire', 10))

    @property
    def spread(self):
        return self._raw_data.get('spread', "5")

    @property
    def bias(self):
        return self._raw_data.get('bias', "1")

    @property
    def collisions(self):
        return self._raw_data.get('collisions', "0")

    @property
    def shootsound(self):
        return self._raw_data.get('shootsound', "Weapon_AR2.NPC_Single")

    @property
    def tracertype(self):
        return self._raw_data.get('tracertype', "AR2TRACER")


class env_headcrabcanister(Parentname, Angles, Targetname):
    model_ = "models/props_combine/headcrabcannister01b.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def HeadcrabType(self):
        return self._raw_data.get('headcrabtype', "0")

    @property
    def HeadcrabCount(self):
        return parse_source_value(self._raw_data.get('headcrabcount', 6))

    @property
    def FlightSpeed(self):
        return parse_source_value(self._raw_data.get('flightspeed', 3000))

    @property
    def FlightTime(self):
        return parse_source_value(self._raw_data.get('flighttime', 5))

    @property
    def StartingHeight(self):
        return parse_source_value(self._raw_data.get('startingheight', 0))

    @property
    def MinSkyboxRefireTime(self):
        return parse_source_value(self._raw_data.get('minskyboxrefiretime', 0))

    @property
    def MaxSkyboxRefireTime(self):
        return parse_source_value(self._raw_data.get('maxskyboxrefiretime', 0))

    @property
    def SkyboxCannisterCount(self):
        return parse_source_value(self._raw_data.get('skyboxcannistercount', 1))

    @property
    def Damage(self):
        return parse_source_value(self._raw_data.get('damage', 150))

    @property
    def DamageRadius(self):
        return parse_source_value(self._raw_data.get('damageradius', 750))

    @property
    def SmokeLifetime(self):
        return parse_source_value(self._raw_data.get('smokelifetime', 30))

    @property
    def LaunchPositionName(self):
        return self._raw_data.get('launchpositionname', "")


class npc_vortigaunt(TalkNPC, PlayerCompanion):

    @property
    def model(self):
        return self._raw_data.get('model', "models/vortigaunt.mdl")

    @property
    def ArmorRechargeEnabled(self):
        return self._raw_data.get('armorrechargeenabled', "1")

    @property
    def HealthRegenerateEnabled(self):
        return self._raw_data.get('healthregenerateenabled', "0")


class npc_spotlight(BaseNPC):

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 100))

    @property
    def YawRange(self):
        return parse_source_value(self._raw_data.get('yawrange', 90))

    @property
    def PitchMin(self):
        return parse_source_value(self._raw_data.get('pitchmin', 35))

    @property
    def PitchMax(self):
        return parse_source_value(self._raw_data.get('pitchmax', 50))

    @property
    def IdleSpeed(self):
        return parse_source_value(self._raw_data.get('idlespeed', 2))

    @property
    def AlertSpeed(self):
        return parse_source_value(self._raw_data.get('alertspeed', 5))

    @property
    def spotlightlength(self):
        return parse_source_value(self._raw_data.get('spotlightlength', 500))

    @property
    def spotlightwidth(self):
        return parse_source_value(self._raw_data.get('spotlightwidth', 50))


class npc_strider(BaseNPC):

    @property
    def model(self):
        return self._raw_data.get('model', "models/combine_strider.mdl")

    @property
    def disablephysics(self):
        return self._raw_data.get('disablephysics', "0")


class npc_barnacle(BaseNPC, BaseFadeProp):
    model_ = "models/Barnacle.mdl"

    @property
    def RestDist(self):
        return parse_source_value(self._raw_data.get('restdist', 16))


class npc_combinegunship(BaseHelicopter):
    model_ = "models/gunship.mdl"
    pass


class info_target_helicopter_crash(Parentname, Targetname):
    icon_sprite = "editor/info_target.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class info_target_gunshipcrash(Parentname, Targetname):
    icon_sprite = "editor/info_target.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class npc_combinedropship(BaseHelicopter):
    model_ = "models/combine_dropship.mdl"

    @property
    def LandTarget(self):
        return self._raw_data.get('landtarget', None)

    @property
    def GunRange(self):
        return parse_source_value(self._raw_data.get('gunrange', 2048))

    @property
    def RollermineTemplate(self):
        return self._raw_data.get('rollerminetemplate', "")

    @property
    def NPCTemplate(self):
        return self._raw_data.get('npctemplate', None)

    @property
    def NPCTemplate2(self):
        return self._raw_data.get('npctemplate2', None)

    @property
    def NPCTemplate3(self):
        return self._raw_data.get('npctemplate3', None)

    @property
    def NPCTemplate4(self):
        return self._raw_data.get('npctemplate4', None)

    @property
    def NPCTemplate5(self):
        return self._raw_data.get('npctemplate5', None)

    @property
    def NPCTemplate6(self):
        return self._raw_data.get('npctemplate6', None)

    @property
    def Dustoff1(self):
        return self._raw_data.get('dustoff1', None)

    @property
    def Dustoff2(self):
        return self._raw_data.get('dustoff2', None)

    @property
    def Dustoff3(self):
        return self._raw_data.get('dustoff3', None)

    @property
    def Dustoff4(self):
        return self._raw_data.get('dustoff4', None)

    @property
    def Dustoff5(self):
        return self._raw_data.get('dustoff5', None)

    @property
    def Dustoff6(self):
        return self._raw_data.get('dustoff6', None)

    @property
    def APCVehicleName(self):
        return self._raw_data.get('apcvehiclename', None)

    @property
    def Invulnerable(self):
        return self._raw_data.get('invulnerable', "0")

    @property
    def CrateType(self):
        return self._raw_data.get('cratetype', "2")


class npc_helicopter(BaseHelicopter):
    model_ = "models/combine_helicopter.mdl"

    @property
    def GracePeriod(self):
        return parse_source_value(self._raw_data.get('graceperiod', 2.0))

    @property
    def PatrolSpeed(self):
        return parse_source_value(self._raw_data.get('patrolspeed', 0))

    @property
    def noncombat(self):
        return self._raw_data.get('noncombat', "No")


class grenade_helicopter(Parentname, Targetname):
    model_ = "models/combine_helicopter/helicopter_bomb01.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class npc_heli_avoidsphere(Parentname, Targetname):
    icon_sprite = "editor/env_firesource"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 128))


class npc_heli_avoidbox(Parentname, Angles, Origin):
    pass


class npc_heli_nobomb(Parentname, Angles, Origin):
    pass


class npc_fastzombie(BaseNPC):
    model_ = "models/Zombie/fast.mdl"
    pass


class npc_fastzombie_torso(BaseNPC):
    model_ = "models/Zombie/Fast_torso.mdl"
    pass


class npc_zombie(BaseNPC):
    model_ = "models/Zombie/Classic.mdl"
    pass


class npc_zombie_torso(BaseNPC):
    model_ = "models/Zombie/Classic_torso.mdl"
    pass


class npc_zombine(BaseNPC):
    model_ = "models/Zombie/zombie_soldier.mdl"
    pass


class npc_poisonzombie(BaseNPC):
    model_ = "models/Zombie/Poison.mdl"

    @property
    def crabcount(self):
        return self._raw_data.get('crabcount', "3")


class npc_cscanner(BaseNPC):
    model_ = "models/combine_scanner.mdl"

    @property
    def spotlightlength(self):
        return parse_source_value(self._raw_data.get('spotlightlength', 500))

    @property
    def spotlightwidth(self):
        return parse_source_value(self._raw_data.get('spotlightwidth', 50))

    @property
    def spotlightdisabled(self):
        return self._raw_data.get('spotlightdisabled', "0")

    @property
    def ShouldInspect(self):
        return self._raw_data.get('shouldinspect', "1")

    @property
    def OnlyInspectPlayers(self):
        return self._raw_data.get('onlyinspectplayers', "0")

    @property
    def NeverInspectPlayers(self):
        return self._raw_data.get('neverinspectplayers', "0")


class npc_clawscanner(BaseNPC):
    model_ = "models/shield_scanner.mdl"

    @property
    def spotlightlength(self):
        return parse_source_value(self._raw_data.get('spotlightlength', 500))

    @property
    def spotlightwidth(self):
        return parse_source_value(self._raw_data.get('spotlightwidth', 50))

    @property
    def spotlightdisabled(self):
        return self._raw_data.get('spotlightdisabled', "0")

    @property
    def ShouldInspect(self):
        return self._raw_data.get('shouldinspect', "1")

    @property
    def OnlyInspectPlayers(self):
        return self._raw_data.get('onlyinspectplayers', "0")

    @property
    def NeverInspectPlayers(self):
        return self._raw_data.get('neverinspectplayers', "0")


class npc_manhack(BaseNPC, AlyxInteractable):
    model_ = "models/manhack.mdl"

    @property
    def ignoreclipbrushes(self):
        return self._raw_data.get('ignoreclipbrushes', "0")


class npc_mortarsynth(BaseNPC):
    model_ = "models/mortarsynth.mdl"
    pass


class npc_metropolice(RappelNPC):
    model_ = "models/Police.mdl"

    @property
    def additionalequipment(self):
        return self._raw_data.get('additionalequipment', "weapon_pistol")

    @property
    def manhacks(self):
        return self._raw_data.get('manhacks', "0")

    @property
    def weapondrawn(self):
        return self._raw_data.get('weapondrawn', "0")


class npc_crabsynth(BaseNPC):
    model_ = "models/Synth.mdl"
    pass


class npc_monk(TalkNPC):
    model_ = "models/Monk.mdl"

    @property
    def additionalequipment(self):
        return self._raw_data.get('additionalequipment', "weapon_annabelle")

    @property
    def HasGun(self):
        return self._raw_data.get('hasgun', "1")


class npc_alyx(Parentname, TalkNPC, PlayerCompanion):

    @property
    def model(self):
        return self._raw_data.get('model', "models/alyx.mdl")

    @property
    def additionalequipment(self):
        return self._raw_data.get('additionalequipment', "weapon_alyxgun")

    @property
    def ShouldHaveEMP(self):
        return self._raw_data.get('shouldhaveemp', "1")


class ai_goal_operator(Targetname, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def actor(self):
        return self._raw_data.get('actor', "")

    @property
    def target(self):
        return self._raw_data.get('target', "")

    @property
    def contexttarget(self):
        return self._raw_data.get('contexttarget', "")

    @property
    def state(self):
        return self._raw_data.get('state', "0")

    @property
    def moveto(self):
        return self._raw_data.get('moveto', "1")


class info_darknessmode_lightsource(Targetname, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def LightRadius(self):
        return parse_source_value(self._raw_data.get('lightradius', 256.0))


class npc_kleiner(TalkNPC):

    @property
    def model(self):
        return self._raw_data.get('model', "models/kleiner.mdl")


class npc_eli(Parentname, TalkNPC):

    @property
    def model(self):
        return self._raw_data.get('model', "models/eli.mdl")


class npc_magnusson(TalkNPC):

    @property
    def model(self):
        return self._raw_data.get('model', "models/magnusson.mdl")


class npc_breen(TalkNPC):

    @property
    def model(self):
        return self._raw_data.get('model', "models/breen.mdl")


class npc_mossman(TalkNPC):
    model_ = "models/mossman.mdl"
    pass


class npc_gman(TalkNPC):
    model_ = "models/gman.mdl"
    pass


class npc_dog(BaseNPC):
    model_ = "models/dog.mdl"
    pass


class monster_generic(BaseNPC):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', None)

    @property
    def body(self):
        return parse_source_value(self._raw_data.get('body', 0))


class generic_actor(Parentname, BaseNPC):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', None)

    @property
    def hull_name(self):
        return self._raw_data.get('hull_name', "Human")


class cycler_actor(BaseNPC):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', None)

    @property
    def Sentence(self):
        return self._raw_data.get('sentence', "")


class npc_maker(BaseNPCMaker):
    icon_sprite = "editor/npc_maker.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def NPCType(self):
        return self._raw_data.get('npctype', None)

    @property
    def NPCTargetname(self):
        return self._raw_data.get('npctargetname', None)

    @property
    def NPCSquadname(self):
        return self._raw_data.get('npcsquadname', None)

    @property
    def NPCHintGroup(self):
        return self._raw_data.get('npchintgroup', None)

    @property
    def additionalequipment(self):
        return self._raw_data.get('additionalequipment', "0")


class npc_antlion_template_maker(BaseNPCMaker):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def TemplateName(self):
        return self._raw_data.get('templatename', None)

    @property
    def spawngroup(self):
        return self._raw_data.get('spawngroup', "")

    @property
    def spawnradius(self):
        return parse_source_value(self._raw_data.get('spawnradius', 512))

    @property
    def spawntarget(self):
        return self._raw_data.get('spawntarget', "")

    @property
    def fighttarget(self):
        return self._raw_data.get('fighttarget', "")

    @property
    def followtarget(self):
        return self._raw_data.get('followtarget', "")

    @property
    def vehicledistance(self):
        return parse_source_value(self._raw_data.get('vehicledistance', 1))

    @property
    def workerspawnrate(self):
        return parse_source_value(self._raw_data.get('workerspawnrate', 0))

    @property
    def ignorebugbait(self):
        return self._raw_data.get('ignorebugbait', "0")

    @property
    def pool_start(self):
        return parse_source_value(self._raw_data.get('pool_start', 0))

    @property
    def pool_max(self):
        return parse_source_value(self._raw_data.get('pool_max', 0))

    @property
    def pool_regen_amount(self):
        return parse_source_value(self._raw_data.get('pool_regen_amount', 0))

    @property
    def pool_regen_time(self):
        return parse_source_value(self._raw_data.get('pool_regen_time', 0))

    @property
    def createspores(self):
        return self._raw_data.get('createspores', "0")


class point_antlion_repellant(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def repelradius(self):
        return parse_source_value(self._raw_data.get('repelradius', 512))


class player_control(Targetname):
    pass


class BaseScripted(Parentname, Angles, Targetname):

    @property
    def m_iszEntity(self):
        return self._raw_data.get('m_iszentity', None)

    @property
    def m_iszIdle(self):
        return self._raw_data.get('m_iszidle', "")

    @property
    def m_iszEntry(self):
        return self._raw_data.get('m_iszentry', "")

    @property
    def m_iszPlay(self):
        return self._raw_data.get('m_iszplay', "")

    @property
    def m_iszPostIdle(self):
        return self._raw_data.get('m_iszpostidle', "")

    @property
    def m_iszCustomMove(self):
        return self._raw_data.get('m_iszcustommove', "")

    @property
    def m_bLoopActionSequence(self):
        return self._raw_data.get('m_bloopactionsequence', "0")

    @property
    def m_bSynchPostIdles(self):
        return self._raw_data.get('m_bsynchpostidles', "0")

    @property
    def m_flRadius(self):
        return parse_source_value(self._raw_data.get('m_flradius', 0))

    @property
    def m_flRepeat(self):
        return parse_source_value(self._raw_data.get('m_flrepeat', 0))

    @property
    def m_fMoveTo(self):
        return self._raw_data.get('m_fmoveto', "1")

    @property
    def m_iszNextScript(self):
        return self._raw_data.get('m_isznextscript', None)

    @property
    def m_bIgnoreGravity(self):
        return self._raw_data.get('m_bignoregravity', "0")

    @property
    def m_bDisableNPCCollisions(self):
        return self._raw_data.get('m_bdisablenpccollisions', "0")


class scripted_sentence(Targetname):
    icon_sprite = "editor/scripted_sentence.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def sentence(self):
        return self._raw_data.get('sentence', "")

    @property
    def entity(self):
        return self._raw_data.get('entity', None)

    @property
    def delay(self):
        return self._raw_data.get('delay', "0")

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 512))

    @property
    def refire(self):
        return self._raw_data.get('refire', "3")

    @property
    def listener(self):
        return self._raw_data.get('listener', None)

    @property
    def volume(self):
        return self._raw_data.get('volume', "10")

    @property
    def attenuation(self):
        return self._raw_data.get('attenuation', "0")


class scripted_target(Parentname, Targetname):
    icon_sprite = "editor/info_target.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def StartDisabled(self):
        return self._raw_data.get('startdisabled', "1")

    @property
    def m_iszEntity(self):
        return self._raw_data.get('m_iszentity', None)

    @property
    def m_flRadius(self):
        return parse_source_value(self._raw_data.get('m_flradius', 0))

    @property
    def MoveSpeed(self):
        return parse_source_value(self._raw_data.get('movespeed', 5))

    @property
    def PauseDuration(self):
        return parse_source_value(self._raw_data.get('pauseduration', 0))

    @property
    def EffectDuration(self):
        return parse_source_value(self._raw_data.get('effectduration', 2))

    @property
    def target(self):
        return self._raw_data.get('target', None)


class ai_relationship(Targetname):
    icon_sprite = "editor/ai_relationship.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def subject(self):
        return self._raw_data.get('subject', "")

    @property
    def target(self):
        return self._raw_data.get('target', "")

    @property
    def disposition(self):
        return self._raw_data.get('disposition', "3")

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 0))

    @property
    def rank(self):
        return parse_source_value(self._raw_data.get('rank', 0))

    @property
    def StartActive(self):
        return self._raw_data.get('startactive', "0")

    @property
    def Reciprocal(self):
        return self._raw_data.get('reciprocal', "0")


class ai_ally_manager(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def maxallies(self):
        return parse_source_value(self._raw_data.get('maxallies', 5))

    @property
    def maxmedics(self):
        return parse_source_value(self._raw_data.get('maxmedics', 1))


class LeadGoalBase(Targetname):

    @property
    def actor(self):
        return self._raw_data.get('actor', None)

    @property
    def goal(self):
        return self._raw_data.get('goal', None)

    @property
    def WaitPointName(self):
        return self._raw_data.get('waitpointname', None)

    @property
    def WaitDistance(self):
        return parse_source_value(self._raw_data.get('waitdistance', None))

    @property
    def LeadDistance(self):
        return parse_source_value(self._raw_data.get('leaddistance', 64))

    @property
    def RetrieveDistance(self):
        return parse_source_value(self._raw_data.get('retrievedistance', 96))

    @property
    def SuccessDistance(self):
        return parse_source_value(self._raw_data.get('successdistance', 0))

    @property
    def Run(self):
        return self._raw_data.get('run', "0")

    @property
    def Retrieve(self):
        return self._raw_data.get('retrieve', "1")

    @property
    def ComingBackWaitForSpeak(self):
        return self._raw_data.get('comingbackwaitforspeak', "1")

    @property
    def RetrieveWaitForSpeak(self):
        return self._raw_data.get('retrievewaitforspeak', "1")

    @property
    def DontSpeakStart(self):
        return self._raw_data.get('dontspeakstart', "0")

    @property
    def LeadDuringCombat(self):
        return self._raw_data.get('leadduringcombat', "0")

    @property
    def GagLeader(self):
        return self._raw_data.get('gagleader', "0")

    @property
    def AttractPlayerConceptModifier(self):
        return self._raw_data.get('attractplayerconceptmodifier', "")

    @property
    def WaitOverConceptModifier(self):
        return self._raw_data.get('waitoverconceptmodifier', "")

    @property
    def ArrivalConceptModifier(self):
        return self._raw_data.get('arrivalconceptmodifier', "")

    @property
    def PostArrivalConceptModifier(self):
        return self._raw_data.get('postarrivalconceptmodifier', None)

    @property
    def SuccessConceptModifier(self):
        return self._raw_data.get('successconceptmodifier', "")

    @property
    def FailureConceptModifier(self):
        return self._raw_data.get('failureconceptmodifier', "")

    @property
    def ComingBackConceptModifier(self):
        return self._raw_data.get('comingbackconceptmodifier', "")

    @property
    def RetrieveConceptModifier(self):
        return self._raw_data.get('retrieveconceptmodifier', "")


class ai_goal_lead(LeadGoalBase):
    icon_sprite = "editor/ai_goal_lead.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def SearchType(self):
        return self._raw_data.get('searchtype', "0")


class ai_goal_lead_weapon(LeadGoalBase):
    icon_sprite = "editor/ai_goal_lead.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def WeaponName(self):
        return self._raw_data.get('weaponname', "weapon_bugbait")

    @property
    def MissingWeaponConceptModifier(self):
        return self._raw_data.get('missingweaponconceptmodifier', None)

    @property
    def SearchType(self):
        return self._raw_data.get('searchtype', "0")


class FollowGoal(Targetname):

    @property
    def actor(self):
        return self._raw_data.get('actor', None)

    @property
    def goal(self):
        return self._raw_data.get('goal', None)

    @property
    def SearchType(self):
        return self._raw_data.get('searchtype', "0")

    @property
    def StartActive(self):
        return self._raw_data.get('startactive', "0")

    @property
    def MaximumState(self):
        return self._raw_data.get('maximumstate', "1")

    @property
    def Formation(self):
        return self._raw_data.get('formation', "0")


class ai_goal_follow(FollowGoal):
    icon_sprite = "editor/ai_goal_follow.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class ai_goal_injured_follow(FollowGoal):
    icon_sprite = "editor/ai_goal_follow.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class ai_battle_line(Parentname, Angles, Targetname):
    model_ = "models/pigeon.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def actor(self):
        return self._raw_data.get('actor', None)

    @property
    def Active(self):
        return self._raw_data.get('active', "0")

    @property
    def Strict(self):
        return self._raw_data.get('strict', "1")


class ai_goal_standoff(Targetname):
    icon_sprite = "editor/ai_goal_standoff.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def actor(self):
        return self._raw_data.get('actor', None)

    @property
    def SearchType(self):
        return self._raw_data.get('searchtype', "0")

    @property
    def StartActive(self):
        return self._raw_data.get('startactive', "0")

    @property
    def HintGroupChangeReaction(self):
        return self._raw_data.get('hintgroupchangereaction', "1")

    @property
    def Aggressiveness(self):
        return self._raw_data.get('aggressiveness', "2")

    @property
    def PlayerBattleline(self):
        return self._raw_data.get('playerbattleline', "1")

    @property
    def StayAtCover(self):
        return self._raw_data.get('stayatcover', "0")

    @property
    def AbandonIfEnemyHides(self):
        return self._raw_data.get('abandonifenemyhides', "0")


class ai_goal_police(Parentname, Angles, Targetname):
    icon_sprite = "editor/ai_goal_police.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def policeradius(self):
        return parse_source_value(self._raw_data.get('policeradius', 512))

    @property
    def policetarget(self):
        return self._raw_data.get('policetarget', "")


class assault_rallypoint(Parentname, Angles, Targetname):
    icon_sprite = "editor/assault_rally.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def assaultpoint(self):
        return self._raw_data.get('assaultpoint', "")

    @property
    def assaultdelay(self):
        return parse_source_value(self._raw_data.get('assaultdelay', 0))

    @property
    def rallysequence(self):
        return self._raw_data.get('rallysequence', "")

    @property
    def priority(self):
        return parse_source_value(self._raw_data.get('priority', 1))

    @property
    def forcecrouch(self):
        return self._raw_data.get('forcecrouch', "0")

    @property
    def urgent(self):
        return self._raw_data.get('urgent', "0")


class assault_assaultpoint(Parentname, Angles, Targetname):
    icon_sprite = "editor/assault_point.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def assaultgroup(self):
        return self._raw_data.get('assaultgroup', "")

    @property
    def nextassaultpoint(self):
        return self._raw_data.get('nextassaultpoint', None)

    @property
    def assaulttimeout(self):
        return parse_source_value(self._raw_data.get('assaulttimeout', 3.0))

    @property
    def clearoncontact(self):
        return self._raw_data.get('clearoncontact', "0")

    @property
    def allowdiversion(self):
        return self._raw_data.get('allowdiversion', "0")

    @property
    def allowdiversionradius(self):
        return parse_source_value(self._raw_data.get('allowdiversionradius', 0))

    @property
    def nevertimeout(self):
        return self._raw_data.get('nevertimeout', "0")

    @property
    def strict(self):
        return self._raw_data.get('strict', "0")

    @property
    def forcecrouch(self):
        return self._raw_data.get('forcecrouch', "0")

    @property
    def urgent(self):
        return self._raw_data.get('urgent', "0")

    @property
    def assaulttolerance(self):
        return self._raw_data.get('assaulttolerance', "36")


class ai_goal_assault(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def actor(self):
        return self._raw_data.get('actor', "")

    @property
    def rallypoint(self):
        return self._raw_data.get('rallypoint', "")

    @property
    def SearchType(self):
        return self._raw_data.get('searchtype', "0")

    @property
    def StartActive(self):
        return self._raw_data.get('startactive', "0")

    @property
    def AssaultCue(self):
        return self._raw_data.get('assaultcue', "1")

    @property
    def RallySelectMethod(self):
        return self._raw_data.get('rallyselectmethod', "0")


class BaseActBusy(Targetname):

    @property
    def actor(self):
        return self._raw_data.get('actor', "")

    @property
    def StartActive(self):
        return self._raw_data.get('startactive', "0")

    @property
    def SearchType(self):
        return self._raw_data.get('searchtype', "0")

    @property
    def busysearchrange(self):
        return parse_source_value(self._raw_data.get('busysearchrange', 2048))

    @property
    def visibleonly(self):
        return self._raw_data.get('visibleonly', "0")


class ai_goal_actbusy(BaseActBusy):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def seeentity(self):
        return self._raw_data.get('seeentity', "")

    @property
    def seeentitytimeout(self):
        return self._raw_data.get('seeentitytimeout', "1")

    @property
    def sightmethod(self):
        return self._raw_data.get('sightmethod', "0")

    @property
    def type(self):
        return self._raw_data.get('type', "0")

    @property
    def safezone(self):
        return self._raw_data.get('safezone', "")

    @property
    def allowteleport(self):
        return self._raw_data.get('allowteleport', "0")


class ai_goal_actbusy_queue(BaseActBusy):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def node_exit(self):
        return self._raw_data.get('node_exit', "")

    @property
    def node01(self):
        return self._raw_data.get('node01', "")

    @property
    def node02(self):
        return self._raw_data.get('node02', "")

    @property
    def node03(self):
        return self._raw_data.get('node03', "")

    @property
    def node04(self):
        return self._raw_data.get('node04', "")

    @property
    def node05(self):
        return self._raw_data.get('node05', "")

    @property
    def node06(self):
        return self._raw_data.get('node06', "")

    @property
    def node07(self):
        return self._raw_data.get('node07', "")

    @property
    def node08(self):
        return self._raw_data.get('node08', "")

    @property
    def node09(self):
        return self._raw_data.get('node09', "")

    @property
    def node10(self):
        return self._raw_data.get('node10', "")

    @property
    def node11(self):
        return self._raw_data.get('node11', "")

    @property
    def node12(self):
        return self._raw_data.get('node12', "")

    @property
    def node13(self):
        return self._raw_data.get('node13', "")

    @property
    def node14(self):
        return self._raw_data.get('node14', "")

    @property
    def node15(self):
        return self._raw_data.get('node15', "")

    @property
    def node16(self):
        return self._raw_data.get('node16', "")

    @property
    def node17(self):
        return self._raw_data.get('node17', "")

    @property
    def node18(self):
        return self._raw_data.get('node18', "")

    @property
    def node19(self):
        return self._raw_data.get('node19', "")

    @property
    def node20(self):
        return self._raw_data.get('node20', "")

    @property
    def mustreachfront(self):
        return self._raw_data.get('mustreachfront', "0")


class ai_changetarget(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def m_iszNewTarget(self):
        return self._raw_data.get('m_isznewtarget', None)


class ai_npc_eventresponsesystem(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class ai_changehintgroup(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def SearchType(self):
        return self._raw_data.get('searchtype', "0")

    @property
    def SearchName(self):
        return self._raw_data.get('searchname', None)

    @property
    def NewHintGroup(self):
        return self._raw_data.get('newhintgroup', None)

    @property
    def Radius(self):
        return self._raw_data.get('radius', "0.0")

    @property
    def hintlimiting(self):
        return self._raw_data.get('hintlimiting', "0")


class ai_script_conditions(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def Actor(self):
        return self._raw_data.get('actor', None)

    @property
    def StartDisabled(self):
        return self._raw_data.get('startdisabled', "1")

    @property
    def MinimumState(self):
        return self._raw_data.get('minimumstate', "1")

    @property
    def MaximumState(self):
        return self._raw_data.get('maximumstate', "3")

    @property
    def ScriptStatus(self):
        return self._raw_data.get('scriptstatus', "2")

    @property
    def RequiredTime(self):
        return parse_source_value(self._raw_data.get('requiredtime', 0))

    @property
    def MinTimeout(self):
        return parse_source_value(self._raw_data.get('mintimeout', 0))

    @property
    def MaxTimeout(self):
        return parse_source_value(self._raw_data.get('maxtimeout', 0))

    @property
    def ActorSeePlayer(self):
        return self._raw_data.get('actorseeplayer', "2")

    @property
    def PlayerActorProximity(self):
        return parse_source_value(self._raw_data.get('playeractorproximity', 0))

    @property
    def PlayerActorFOV(self):
        return parse_source_value(self._raw_data.get('playeractorfov', 360))

    @property
    def PlayerActorFOVTrueCone(self):
        return self._raw_data.get('playeractorfovtruecone', "0")

    @property
    def PlayerActorLOS(self):
        return self._raw_data.get('playeractorlos', "2")

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def ActorSeeTarget(self):
        return self._raw_data.get('actorseetarget', "2")

    @property
    def ActorTargetProximity(self):
        return parse_source_value(self._raw_data.get('actortargetproximity', 0))

    @property
    def PlayerTargetProximity(self):
        return parse_source_value(self._raw_data.get('playertargetproximity', 0))

    @property
    def PlayerTargetFOV(self):
        return parse_source_value(self._raw_data.get('playertargetfov', 360))

    @property
    def PlayerTargetFOVTrueCone(self):
        return self._raw_data.get('playertargetfovtruecone', "0")

    @property
    def PlayerTargetLOS(self):
        return self._raw_data.get('playertargetlos', "2")

    @property
    def PlayerBlockingActor(self):
        return self._raw_data.get('playerblockingactor', "2")

    @property
    def ActorInPVS(self):
        return self._raw_data.get('actorinpvs', "2")

    @property
    def ActorInVehicle(self):
        return self._raw_data.get('actorinvehicle', "2")

    @property
    def PlayerInVehicle(self):
        return self._raw_data.get('playerinvehicle', "2")


class scripted_sequence(DXLevelChoice, BaseScripted):
    model_ = "models/editor/scriptedsequence.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def onplayerdeath(self):
        return self._raw_data.get('onplayerdeath', "0")


class aiscripted_schedule(Targetname):
    icon_sprite = "editor/aiscripted_schedule"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def m_iszEntity(self):
        return self._raw_data.get('m_iszentity', None)

    @property
    def m_flRadius(self):
        return parse_source_value(self._raw_data.get('m_flradius', 0))

    @property
    def graball(self):
        return self._raw_data.get('graball', "0")

    @property
    def forcestate(self):
        return self._raw_data.get('forcestate', "0")

    @property
    def schedule(self):
        return self._raw_data.get('schedule', "1")

    @property
    def interruptability(self):
        return self._raw_data.get('interruptability', "0")

    @property
    def goalent(self):
        return self._raw_data.get('goalent', None)


class ai_citizen_response_system(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class func_healthcharger(Parentname, Origin, EnableDisable, Global):

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)


class func_recharge(Parentname, Targetname, Origin):

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)


class func_vehicleclip(Parentname, Targetname, Global):
    pass


class func_lookdoor(func_movelinear):

    @property
    def ProximityDistance(self):
        return self._raw_data.get('proximitydistance', "0.0")

    @property
    def ProximityOffset(self):
        return self._raw_data.get('proximityoffset', "0.0")

    @property
    def FieldOfView(self):
        return self._raw_data.get('fieldofview', "0.0")


class trigger_waterydeath(Trigger):
    pass


class env_global(EnvGlobal):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def globalstate(self):
        return self._raw_data.get('globalstate', None)


class BaseTank(Angles, Parentname, Global, Shadow, Targetname, Origin, RenderFields):

    @property
    def control_volume(self):
        return self._raw_data.get('control_volume', "")

    @property
    def master(self):
        return self._raw_data.get('master', None)

    @property
    def yawrate(self):
        return self._raw_data.get('yawrate', "30")

    @property
    def yawrange(self):
        return self._raw_data.get('yawrange', "180")

    @property
    def yawtolerance(self):
        return self._raw_data.get('yawtolerance', "15")

    @property
    def pitchrate(self):
        return self._raw_data.get('pitchrate', "0")

    @property
    def pitchrange(self):
        return self._raw_data.get('pitchrange', "0")

    @property
    def pitchtolerance(self):
        return self._raw_data.get('pitchtolerance', "5")

    @property
    def barrel(self):
        return self._raw_data.get('barrel', "0")

    @property
    def barrely(self):
        return self._raw_data.get('barrely', "0")

    @property
    def barrelz(self):
        return self._raw_data.get('barrelz', "0")

    @property
    def spritesmoke(self):
        return self._raw_data.get('spritesmoke', "")

    @property
    def spriteflash(self):
        return self._raw_data.get('spriteflash', "")

    @property
    def spritescale(self):
        return self._raw_data.get('spritescale', "1")

    @property
    def rotatestartsound(self):
        return self._raw_data.get('rotatestartsound', "")

    @property
    def rotatesound(self):
        return self._raw_data.get('rotatesound', "")

    @property
    def rotatestopsound(self):
        return self._raw_data.get('rotatestopsound', "")

    @property
    def firerate(self):
        return self._raw_data.get('firerate', "1")

    @property
    def bullet_damage(self):
        return self._raw_data.get('bullet_damage', "0")

    @property
    def bullet_damage_vs_player(self):
        return self._raw_data.get('bullet_damage_vs_player', "0")

    @property
    def persistence(self):
        return self._raw_data.get('persistence', "1")

    @property
    def persistence2(self):
        return self._raw_data.get('persistence2', "0")

    @property
    def firespread(self):
        return self._raw_data.get('firespread', "0")

    @property
    def minRange(self):
        return self._raw_data.get('minrange', "0")

    @property
    def maxRange(self):
        return self._raw_data.get('maxrange', "0")

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)

    @property
    def gun_base_attach(self):
        return self._raw_data.get('gun_base_attach', "")

    @property
    def gun_barrel_attach(self):
        return self._raw_data.get('gun_barrel_attach', "")

    @property
    def gun_yaw_pose_param(self):
        return self._raw_data.get('gun_yaw_pose_param', "")

    @property
    def gun_yaw_pose_center(self):
        return parse_source_value(self._raw_data.get('gun_yaw_pose_center', 0))

    @property
    def gun_pitch_pose_param(self):
        return self._raw_data.get('gun_pitch_pose_param', "")

    @property
    def gun_pitch_pose_center(self):
        return parse_source_value(self._raw_data.get('gun_pitch_pose_center', 0))

    @property
    def ammo_count(self):
        return parse_source_value(self._raw_data.get('ammo_count', -1))

    @property
    def LeadTarget(self):
        return self._raw_data.get('leadtarget', "No")

    @property
    def npc_man_point(self):
        return self._raw_data.get('npc_man_point', "")

    @property
    def playergraceperiod(self):
        return parse_source_value(self._raw_data.get('playergraceperiod', 0))

    @property
    def ignoregraceupto(self):
        return parse_source_value(self._raw_data.get('ignoregraceupto', 768))

    @property
    def playerlocktimebeforefire(self):
        return parse_source_value(self._raw_data.get('playerlocktimebeforefire', 0))

    @property
    def effecthandling(self):
        return self._raw_data.get('effecthandling', "0")


class func_tank(BaseTank):

    @property
    def ammotype(self):
        return self._raw_data.get('ammotype', "")


class func_tankpulselaser(BaseTank):

    @property
    def PulseSpeed(self):
        return parse_source_value(self._raw_data.get('pulsespeed', 1000))

    @property
    def PulseColor(self):
        return parse_int_vector(self._raw_data.get('pulsecolor', "255 0 0"))

    @property
    def PulseWidth(self):
        return parse_source_value(self._raw_data.get('pulsewidth', 20))

    @property
    def PulseLife(self):
        return parse_source_value(self._raw_data.get('pulselife', 2))

    @property
    def PulseLag(self):
        return parse_source_value(self._raw_data.get('pulselag', 0.05))

    @property
    def PulseFireSound(self):
        return self._raw_data.get('pulsefiresound', "")


class func_tanklaser(BaseTank):

    @property
    def laserentity(self):
        return self._raw_data.get('laserentity', None)


class func_tankrocket(BaseTank):

    @property
    def rocketspeed(self):
        return parse_source_value(self._raw_data.get('rocketspeed', 800))


class func_tankairboatgun(BaseTank):

    @property
    def airboat_gun_model(self):
        return self._raw_data.get('airboat_gun_model', None)


class func_tankapcrocket(BaseTank):

    @property
    def rocketspeed(self):
        return parse_source_value(self._raw_data.get('rocketspeed', 800))

    @property
    def burstcount(self):
        return parse_source_value(self._raw_data.get('burstcount', 10))


class func_tankmortar(BaseTank):

    @property
    def iMagnitude(self):
        return parse_source_value(self._raw_data.get('imagnitude', 100))

    @property
    def firedelay(self):
        return self._raw_data.get('firedelay', "2")

    @property
    def firestartsound(self):
        return self._raw_data.get('firestartsound', "")

    @property
    def fireendsound(self):
        return self._raw_data.get('fireendsound', "")

    @property
    def incomingsound(self):
        return self._raw_data.get('incomingsound', "")

    @property
    def warningtime(self):
        return parse_source_value(self._raw_data.get('warningtime', 1))

    @property
    def firevariance(self):
        return parse_source_value(self._raw_data.get('firevariance', 0))


class func_tankphyscannister(BaseTank):

    @property
    def barrel_volume(self):
        return self._raw_data.get('barrel_volume', "")


class func_tank_combine_cannon(BaseTank):

    @property
    def ammotype(self):
        return self._raw_data.get('ammotype', "")


class Item(Angles, Targetname, Shadow):

    @property
    def fademindist(self):
        return parse_source_value(self._raw_data.get('fademindist', -1))

    @property
    def fademaxdist(self):
        return parse_source_value(self._raw_data.get('fademaxdist', 0))

    @property
    def fadescale(self):
        return parse_source_value(self._raw_data.get('fadescale', 1))


class item_dynamic_resupply(Item):
    model_ = "models/items/healthkit.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def DesiredHealth(self):
        return parse_source_value(self._raw_data.get('desiredhealth', 1))

    @property
    def DesiredArmor(self):
        return parse_source_value(self._raw_data.get('desiredarmor', 0.3))

    @property
    def DesiredAmmoPistol(self):
        return parse_source_value(self._raw_data.get('desiredammopistol', 0.5))

    @property
    def DesiredAmmoSMG1(self):
        return parse_source_value(self._raw_data.get('desiredammosmg1', 0.5))

    @property
    def DesiredAmmoSMG1_Grenade(self):
        return parse_source_value(self._raw_data.get('desiredammosmg1_grenade', 0.1))

    @property
    def DesiredAmmoAR2(self):
        return parse_source_value(self._raw_data.get('desiredammoar2', 0.4))

    @property
    def DesiredAmmoBuckshot(self):
        return parse_source_value(self._raw_data.get('desiredammobuckshot', 0.5))

    @property
    def DesiredAmmoRPG_Round(self):
        return parse_source_value(self._raw_data.get('desiredammorpg_round', 0))

    @property
    def DesiredAmmoGrenade(self):
        return parse_source_value(self._raw_data.get('desiredammogrenade', 0.1))

    @property
    def DesiredAmmo357(self):
        return parse_source_value(self._raw_data.get('desiredammo357', 0))

    @property
    def DesiredAmmoCrossbow(self):
        return parse_source_value(self._raw_data.get('desiredammocrossbow', 0))

    @property
    def DesiredAmmoAR2_AltFire(self):
        return parse_source_value(self._raw_data.get('desiredammoar2_altfire', 0))


class item_ammo_pistol(Item):
    model_ = "models/items/boxsrounds.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class item_ammo_pistol_large(Item):
    model_ = "models/items/largeBoxSRounds.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class item_ammo_smg1(Item):
    model_ = "models/items/BoxMRounds.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class item_ammo_smg1_large(Item):
    model_ = "models/items/LargeBoxMRounds.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class item_ammo_ar2(Item):
    model_ = "models/items/BoxBRounds.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class item_ammo_ar2_large(Item):
    model_ = "models/items/LargeBoxBRounds.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class item_ammo_357(Item):
    model_ = "models/items/357ammo.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class item_ammo_357_large(Item):
    model_ = "models/items/357ammobox.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class item_ammo_crossbow(Item):
    model_ = "models/items/CrossbowRounds.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class item_box_buckshot(Item):
    model_ = "models/items/BoxBuckshot.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class item_rpg_round(Item):
    model_ = "models/weapons/w_missile_closed.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class item_ammo_smg1_grenade(Item):
    model_ = "models/items/AR2_Grenade.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class item_battery(Item):
    model_ = "models/items/battery.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class item_healthkit(Item):
    model_ = "models/items/healthkit.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class item_healthvial(Item):
    model_ = "models/healthvial.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class item_ammo_ar2_altfire(Item):
    model_ = "models/items/combine_rifle_ammo01.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class item_suit(Item):
    model_ = "models/items/hevsuit.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class item_ammo_crate(Angles, BaseFadeProp, Targetname):
    model_ = "models/items/ammocrate_rockets.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def AmmoType(self):
        return self._raw_data.get('ammotype', "0")


class item_item_crate(BasePropPhysics):
    model_ = "models/items/item_item_crate.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def CrateType(self):
        return self._raw_data.get('cratetype', "0")

    @property
    def CrateAppearance(self):
        return self._raw_data.get('crateappearance', "0")

    @property
    def ItemClass(self):
        return self._raw_data.get('itemclass', "item_dynamic_resupply")

    @property
    def ItemCount(self):
        return parse_source_value(self._raw_data.get('itemcount', 1))

    @property
    def SpecificResupply(self):
        return self._raw_data.get('specificresupply', "")


class item_healthcharger(Angles, BaseFadeProp, Targetname):
    model_ = "models/props_combine/health_charger001.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)


class item_suitcharger(Angles, BaseFadeProp, Targetname):
    model_ = "models/props_combine/suit_charger001.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)


class Weapon(Angles, Targetname):

    @property
    def fademindist(self):
        return parse_source_value(self._raw_data.get('fademindist', -1))

    @property
    def fademaxdist(self):
        return parse_source_value(self._raw_data.get('fademaxdist', 0))

    @property
    def fadescale(self):
        return parse_source_value(self._raw_data.get('fadescale', 1))


class weapon_crowbar(Weapon):
    model_ = "models/weapons/w_crowbar.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class weapon_stunstick(Weapon):
    model_ = "models/weapons/w_stunbaton.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class weapon_pistol(Weapon):
    model_ = "models/weapons/w_pistol.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class weapon_ar2(Weapon):
    model_ = "models/weapons/w_irifle.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class weapon_rpg(Weapon):
    model_ = "models/weapons/w_rocket_launcher.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class weapon_smg1(Weapon):
    model_ = "models/weapons/w_smg1.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class weapon_357(Weapon):
    model_ = "models/weapons/w_357.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class weapon_crossbow(Weapon):
    model_ = "models/weapons/w_crossbow.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class weapon_shotgun(Weapon):
    model_ = "models/weapons/w_shotgun.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class weapon_frag(Weapon):
    model_ = "models/weapons/w_grenade.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class weapon_physcannon(Weapon):
    model_ = "models/weapons/w_physics.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class weapon_bugbait(Weapon):
    model_ = "models/spore.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class weapon_alyxgun(Weapon):
    model_ = "models/weapons/W_Alyx_Gun.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class weapon_annabelle(Weapon):
    model_ = "models/weapons/W_annabelle.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class trigger_rpgfire(Trigger):
    pass


class trigger_vphysics_motion(Trigger):

    @property
    def SetGravityScale(self):
        return parse_source_value(self._raw_data.get('setgravityscale', 1.0))

    @property
    def SetAdditionalAirDensity(self):
        return parse_source_value(self._raw_data.get('setadditionalairdensity', 0))

    @property
    def SetVelocityLimit(self):
        return parse_source_value(self._raw_data.get('setvelocitylimit', 0.0))

    @property
    def SetVelocityLimitDelta(self):
        return parse_source_value(self._raw_data.get('setvelocitylimitdelta', 0.0))

    @property
    def SetVelocityScale(self):
        return parse_source_value(self._raw_data.get('setvelocityscale', 1.0))

    @property
    def SetAngVelocityLimit(self):
        return parse_source_value(self._raw_data.get('setangvelocitylimit', 0.0))

    @property
    def SetAngVelocityScale(self):
        return parse_source_value(self._raw_data.get('setangvelocityscale', 1.0))

    @property
    def SetLinearForce(self):
        return parse_source_value(self._raw_data.get('setlinearforce', 0.0))

    @property
    def SetLinearForceAngles(self):
        return parse_float_vector(self._raw_data.get('setlinearforceangles', "0 0 0"))

    @property
    def ParticleTrailMaterial(self):
        return self._raw_data.get('particletrailmaterial', None)

    @property
    def ParticleTrailLifetime(self):
        return parse_source_value(self._raw_data.get('particletraillifetime', 4))

    @property
    def ParticleTrailStartSize(self):
        return parse_source_value(self._raw_data.get('particletrailstartsize', 2))

    @property
    def ParticleTrailEndSize(self):
        return parse_source_value(self._raw_data.get('particletrailendsize', 3))


class point_bugbait(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def Enabled(self):
        return self._raw_data.get('enabled', "1")

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 512))


class weapon_brickbat(Weapon):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def BrickbatType(self):
        return self._raw_data.get('brickbattype', "Rock")


class path_corner(Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def wait(self):
        return parse_source_value(self._raw_data.get('wait', 0))

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 0))

    @property
    def yaw_speed(self):
        return parse_source_value(self._raw_data.get('yaw_speed', 0))


class path_corner_crash(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)


class player_loadsaved(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def duration(self):
        return self._raw_data.get('duration', "2")

    @property
    def holdtime(self):
        return self._raw_data.get('holdtime', "0")

    @property
    def renderamt(self):
        return parse_source_value(self._raw_data.get('renderamt', 255))

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "0 0 0"))

    @property
    def loadtime(self):
        return self._raw_data.get('loadtime', "0")


class player_weaponstrip(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class player_speedmod(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class logic_choreographed_scene(Targetname):
    icon_sprite = "editor/choreo_scene.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def SceneFile(self):
        return self._raw_data.get('scenefile', None)

    @property
    def target1(self):
        return self._raw_data.get('target1', None)

    @property
    def target2(self):
        return self._raw_data.get('target2', None)

    @property
    def target3(self):
        return self._raw_data.get('target3', None)

    @property
    def target4(self):
        return self._raw_data.get('target4', None)

    @property
    def target5(self):
        return self._raw_data.get('target5', None)

    @property
    def target6(self):
        return self._raw_data.get('target6', None)

    @property
    def target7(self):
        return self._raw_data.get('target7', None)

    @property
    def target8(self):
        return self._raw_data.get('target8', None)

    @property
    def busyactor(self):
        return self._raw_data.get('busyactor', "1")

    @property
    def onplayerdeath(self):
        return self._raw_data.get('onplayerdeath', "0")


class logic_scene_list_manager(Targetname):
    icon_sprite = "editor/choreo_manager.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def scene0(self):
        return self._raw_data.get('scene0', "")

    @property
    def scene1(self):
        return self._raw_data.get('scene1', "")

    @property
    def scene2(self):
        return self._raw_data.get('scene2', "")

    @property
    def scene3(self):
        return self._raw_data.get('scene3', "")

    @property
    def scene4(self):
        return self._raw_data.get('scene4', "")

    @property
    def scene5(self):
        return self._raw_data.get('scene5', "")

    @property
    def scene6(self):
        return self._raw_data.get('scene6', "")

    @property
    def scene7(self):
        return self._raw_data.get('scene7', "")

    @property
    def scene8(self):
        return self._raw_data.get('scene8', "")

    @property
    def scene9(self):
        return self._raw_data.get('scene9', "")

    @property
    def scene10(self):
        return self._raw_data.get('scene10', "")

    @property
    def scene11(self):
        return self._raw_data.get('scene11', "")

    @property
    def scene12(self):
        return self._raw_data.get('scene12', "")

    @property
    def scene13(self):
        return self._raw_data.get('scene13', "")

    @property
    def scene14(self):
        return self._raw_data.get('scene14', "")

    @property
    def scene15(self):
        return self._raw_data.get('scene15', "")


class ai_sound(Parentname, Targetname):
    icon_sprite = "editor/ai_sound.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def volume(self):
        return parse_source_value(self._raw_data.get('volume', 120))

    @property
    def duration(self):
        return parse_source_value(self._raw_data.get('duration', 0.5))

    @property
    def soundtype(self):
        return self._raw_data.get('soundtype', "0")

    @property
    def soundcontext(self):
        return self._raw_data.get('soundcontext', "0")

    @property
    def locationproxy(self):
        return self._raw_data.get('locationproxy', "")


class env_rotorwash(Parentname, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class combine_mine(Parentname, Targetname):
    model_ = "models/props_combine/combine_mine01.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def bounce(self):
        return self._raw_data.get('bounce', "1")

    @property
    def LockSilently(self):
        return self._raw_data.get('locksilently', "1")

    @property
    def StartDisarmed(self):
        return self._raw_data.get('startdisarmed', "0")

    @property
    def Modification(self):
        return self._raw_data.get('modification', "0")


class env_ar2explosion(Parentname, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def material(self):
        return self._raw_data.get('material', "particle/particle_noisesphere")


class env_starfield(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class env_flare(Parentname, Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def scale(self):
        return parse_source_value(self._raw_data.get('scale', 1))

    @property
    def duration(self):
        return parse_source_value(self._raw_data.get('duration', 30))


class env_muzzleflash(Parentname, Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def parentattachment(self):
        return self._raw_data.get('parentattachment', "")

    @property
    def scale(self):
        return parse_source_value(self._raw_data.get('scale', 1))


class env_terrainmorph(Parentname, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def startradius(self):
        return parse_source_value(self._raw_data.get('startradius', 500))

    @property
    def goalradius(self):
        return parse_source_value(self._raw_data.get('goalradius', 100))

    @property
    def duration(self):
        return parse_source_value(self._raw_data.get('duration', 3))

    @property
    def fraction(self):
        return parse_source_value(self._raw_data.get('fraction', 1))


class logic_achievement(Targetname, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def AchievementEvent(self):
        return self._raw_data.get('achievementevent', "0")


class point_camera(Parentname, Angles):
    viewport_model = "models/editor/camera.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def targetname(self):
        return self._raw_data.get('targetname', None)

    @property
    def FOV(self):
        return parse_source_value(self._raw_data.get('fov', 90))

    @property
    def UseScreenAspectRatio(self):
        return self._raw_data.get('usescreenaspectratio', "0")

    @property
    def fogEnable(self):
        return self._raw_data.get('fogenable', "0")

    @property
    def fogColor(self):
        return parse_int_vector(self._raw_data.get('fogcolor', "0 0 0"))

    @property
    def fogStart(self):
        return parse_source_value(self._raw_data.get('fogstart', 2048))

    @property
    def fogEnd(self):
        return parse_source_value(self._raw_data.get('fogend', 4096))

    @property
    def fogMaxDensity(self):
        return parse_source_value(self._raw_data.get('fogmaxdensity', 1))


class func_monitor(func_brush):

    @property
    def target(self):
        return self._raw_data.get('target', None)


class func_bulletshield(func_brush):
    pass


class BaseVehicle(Targetname, Origin, Global, prop_static_base):

    @property
    def vehiclescript(self):
        return self._raw_data.get('vehiclescript', "scripts/vehicles/jeep_test.txt")

    @property
    def actionScale(self):
        return parse_source_value(self._raw_data.get('actionscale', 1))


class BaseDriveableVehicle(BaseVehicle):

    @property
    def VehicleLocked(self):
        return self._raw_data.get('vehiclelocked', "0")


class prop_vehicle(BaseVehicle):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class prop_vehicle_driveable(BaseDriveableVehicle):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class point_apc_controller(Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def yawrate(self):
        return self._raw_data.get('yawrate', "30")

    @property
    def yawtolerance(self):
        return self._raw_data.get('yawtolerance', "15")

    @property
    def pitchrate(self):
        return self._raw_data.get('pitchrate', "0")

    @property
    def pitchtolerance(self):
        return self._raw_data.get('pitchtolerance', "20")

    @property
    def rotatestartsound(self):
        return self._raw_data.get('rotatestartsound', "")

    @property
    def rotatesound(self):
        return self._raw_data.get('rotatesound', "")

    @property
    def rotatestopsound(self):
        return self._raw_data.get('rotatestopsound', "")

    @property
    def minRange(self):
        return self._raw_data.get('minrange', "0")

    @property
    def maxRange(self):
        return self._raw_data.get('maxrange', "0")

    @property
    def targetentityname(self):
        return self._raw_data.get('targetentityname', "")


class prop_vehicle_apc(BaseDriveableVehicle):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def missilehint(self):
        return self._raw_data.get('missilehint', "")


class info_apc_missile_hint(Angles, Targetname, Origin, EnableDisable):

    @property
    def target(self):
        return self._raw_data.get('target', "")


class prop_vehicle_jeep(BaseDriveableVehicle):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def CargoVisible(self):
        return self._raw_data.get('cargovisible', "0")


class vehicle_viewcontroller(BaseDriveableVehicle):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class prop_vehicle_airboat(BaseDriveableVehicle):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def EnableGun(self):
        return self._raw_data.get('enablegun', "0")


class prop_vehicle_cannon(BaseDriveableVehicle):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class prop_vehicle_crane(BaseDriveableVehicle):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def magnetname(self):
        return self._raw_data.get('magnetname', "")


class prop_vehicle_prisoner_pod(Parentname, BaseDriveableVehicle):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class env_speaker(BaseSpeaker):
    icon_sprite = "editor/ambient_generic.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class script_tauremoval(Parentname, Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def vortigaunt(self):
        return self._raw_data.get('vortigaunt', None)


class script_intro(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def alternatefovchange(self):
        return self._raw_data.get('alternatefovchange', "0")


class env_citadel_energy_core(Parentname, Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def scale(self):
        return parse_source_value(self._raw_data.get('scale', 1))


class env_alyxemp(Parentname, Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def Type(self):
        return self._raw_data.get('type', "0")

    @property
    def EndTargetName(self):
        return self._raw_data.get('endtargetname', "")


class test_sidelist(Base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def sides(self):
        return self._raw_data.get('sides', None)


class info_teleporter_countdown(Targetname):
    icon_sprite = "editor/info_target.vmt"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class prop_vehicle_choreo_generic(Parentname, BaseDriveableVehicle):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def ignoremoveparent(self):
        return self._raw_data.get('ignoremoveparent', "0")

    @property
    def ignoreplayer(self):
        return self._raw_data.get('ignoreplayer', "0")


class filter_combineball_type(BaseFilter):
    icon_sprite = "editor/filter_class.vmt"

    @property
    def balltype(self):
        return self._raw_data.get('balltype', "1")


class env_entity_dissolver(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', "")

    @property
    def magnitude(self):
        return parse_source_value(self._raw_data.get('magnitude', 250))

    @property
    def dissolvetype(self):
        return self._raw_data.get('dissolvetype', "Energy")


class prop_coreball(Angles, Targetname):
    model_ = "models/props_combine/coreball.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class prop_scalable(Angles, Studiomodel, Targetname, RenderFields):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    pass


class point_push(Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def enabled(self):
        return self._raw_data.get('enabled', "1")

    @property
    def magnitude(self):
        return parse_source_value(self._raw_data.get('magnitude', 100))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 128))

    @property
    def inner_radius(self):
        return parse_source_value(self._raw_data.get('inner_radius', 0))


class npc_antlion_grub(Angles, BaseFadeProp, Targetname, Global):
    model_ = "models/antlion_grub.mdl"
    pass


class weapon_striderbuster(BasePropPhysics):
    model_ = "models/magnusson_device.mdl"

    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def dud(self):
        return self._raw_data.get('dud', "0")


class point_flesh_effect_target(Parentname, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 8))


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
