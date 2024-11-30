
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



class Breakable(Targetname, Shadow, DamageFilter):

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



class BreakableBrush(Global, Breakable, Parentname):

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



class BaseNPC(DamageFilter, Targetname, ResponseContext, Angles, Shadow, RenderFields):

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



class info_npc_spawn_destination(Targetname, Angles, Parentname):
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



class BaseNPCMaker(Targetname, Angles, EnableDisable):
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



class TriggerOnce(Origin, Parentname, Targetname, Global, EnableDisable):

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



class worldspawn(Targetname, worldbase, ResponseContext):
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



class env_splash(Targetname, Angles):
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



class env_sun(Targetname, Angles):
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



class env_lightglow(Targetname, Angles, Parentname):
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



class env_smokestack(Angles, Parentname):
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


class func_useableladder(Targetname, Parentname):
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



class func_ladderendpoint(Targetname, Angles, Parentname):
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



class func_wall(Targetname, Global, Shadow, RenderFields):

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)



class func_clip_vphysics(Targetname, EnableDisable):

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)



class func_brush(Origin, Parentname, Targetname, Global, Inputfilter, EnableDisable, Shadow, RenderFields):

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



class vgui_screen_base(Targetname, Angles, Parentname):

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


class vgui_slideshow_display(Targetname, Angles, Parentname):
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



class cycler(Parentname, Targetname, Angles, RenderFields):
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



class gibshooterbase(Targetname, Parentname):

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



class env_beam(Targetname, RenderFxChoices, Parentname):
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



class env_beverage(Targetname, Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 10))

    @property
    def beveragetype(self):
        return self._raw_data.get('beveragetype', "0")



class env_embers(Targetname, Angles, Parentname):

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



class env_funnel(Targetname, Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class env_blood(Targetname, Parentname):
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



class env_bubbles(Targetname, Parentname):

    @property
    def density(self):
        return parse_source_value(self._raw_data.get('density', 2))

    @property
    def frequency(self):
        return parse_source_value(self._raw_data.get('frequency', 2))

    @property
    def current(self):
        return parse_source_value(self._raw_data.get('current', 0))



class env_explosion(Targetname, Parentname):
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



class env_smoketrail(Targetname, Parentname):
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



class env_physexplosion(Targetname, Parentname):
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



class env_physimpact(Targetname, Parentname):
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



class env_fire(Targetname, EnableDisable, Parentname):
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



class env_firesource(Targetname, Parentname):
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



class env_firesensor(Targetname, Parentname):
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



class env_fog_controller(Targetname, Angles, DXLevelChoice):
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



class env_steam(Targetname, Angles, Parentname):
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



class env_laser(Targetname, RenderFxChoices, Parentname):
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



class env_shake(Targetname, Parentname):
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



class env_viewpunch(Targetname, Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def punchangle(self):
        return parse_float_vector(self._raw_data.get('punchangle', "0 0 90"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 500))



class env_rotorwash_emitter(Targetname, Parentname):
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



class env_soundscape_proxy(Targetname, Parentname):
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



class env_soundscape(Targetname, EnableDisable, Parentname):
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


class env_spark(Targetname, Angles, Parentname):
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



class env_sprite(Targetname, RenderFields, DXLevelChoice, Parentname):
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


class env_wind(Targetname, Angles):
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



class BaseSpeaker(Targetname, ResponseContext):

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



class point_enable_motion_fixup(Angles, Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class point_message(Targetname, Parentname):
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



class point_spotlight(Parentname, Targetname, DXLevelChoice, Angles, RenderFields):
    model_ = "models/editor/cone_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def IgnoreSolid(self):
        return self._raw_data.get('ignoresolid', "0")

    @property
    def spotlightlength(self):
        return parse_source_value(self._raw_data.get('spotlightlength', 500))

    @property
    def spotlightwidth(self):
        return parse_source_value(self._raw_data.get('spotlightwidth', 50))

    @property
    def HDRColorScale(self):
        return parse_source_value(self._raw_data.get('hdrcolorscale', 1.0))



class point_tesla(Targetname, Parentname):
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



class game_zone_player(Targetname, Parentname):
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



class info_projecteddecal(Targetname, Angles):
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



class info_player_start(PlayerClass, Angles):
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


class info_target(Targetname, Angles, Parentname):
    icon_sprite = "editor/info_target.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class info_particle_system(Targetname, Angles, Parentname):
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



class phys_ragdollmagnet(Targetname, Angles, EnableDisable, Parentname):
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


class info_teleport_destination(Targetname, PlayerClass, Angles, Parentname):
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


class info_node_hint(Targetname, HintNode, Angles):
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



class info_node_air_hint(Targetname, HintNode, Angles):
    model_ = "models/editor/air_node_hint.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def nodeheight(self):
        return parse_source_value(self._raw_data.get('nodeheight', 0))



class info_hint(Targetname, HintNode, Angles):
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



class info_radial_link_controller(Targetname, Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 120))



class info_node_climb(Targetname, HintNode, Angles):
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



class light_spot(Targetname, Angles, Light):
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



class light_dynamic(Targetname, Angles, Parentname):
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



class func_movelinear(Targetname, RenderFields, Origin, Parentname):

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



class func_water_analog(Targetname, Origin, Parentname):

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



class func_rotating(Origin, Parentname, Targetname, Angles, Shadow, RenderFields):

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



class func_platrot(Origin, BasePlat, Parentname, Targetname, Angles, Shadow, RenderFields):

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



class keyframe_track(Targetname, Angles, KeyFrame, Parentname):
    pass


class move_keyframed(Targetname, KeyFrame, Mover, Parentname):
    pass


class move_track(Targetname, KeyFrame, Mover, Parentname):

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



class keyframe_rope(Targetname, KeyFrame, RopeKeyFrame, Parentname):
    model_ = "models/editor/axis_helper_thick.mdl"
    pass


class move_rope(Targetname, KeyFrame, RopeKeyFrame, Parentname):
    model_ = "models/editor/axis_helper.mdl"

    @property
    def PositionInterpolator(self):
        return self._raw_data.get('positioninterpolator', "2")



class Button(Base):
    pass


class func_button(Origin, DamageFilter, Targetname, Parentname, Button, RenderFields):

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



class func_rot_button(Origin, Parentname, Targetname, Global, Button, Angles, EnableDisable):

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



class momentary_rot_button(Origin, Parentname, Targetname, Angles, RenderFields):

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



class Door(Parentname, Targetname, Global, Shadow, RenderFields):

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



class func_door_rotating(Door, Angles, Origin):

    @property
    def distance(self):
        return parse_source_value(self._raw_data.get('distance', 90))

    @property
    def solidbsp(self):
        return self._raw_data.get('solidbsp', "0")



class prop_door_rotating(Parentname, Targetname, Global, Angles, Studiomodel):
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



class env_dustpuff(Targetname, Angles, Parentname):
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



class env_particlescript(Targetname, Angles, Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/Ambient_citadel_paths.mdl")



class env_effectscript(Targetname, Angles, Parentname):
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



class point_viewcontrol(Targetname, Angles, Parentname):
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



class env_microphone(Targetname, EnableDisable, Parentname):
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



class env_entity_maker(Targetname, Angles, Parentname):
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



class point_anglesensor(Targetname, EnableDisable, Parentname):
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



class point_proximity_sensor(Targetname, Angles, EnableDisable, Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)



class point_teleport(Targetname, Angles):
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



class func_physbox(Origin, BreakableBrush, RenderFields):

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



class phys_keepupright(Targetname, Angles):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def attach1(self):
        return self._raw_data.get('attach1', "")

    @property
    def angularlimit(self):
        return parse_source_value(self._raw_data.get('angularlimit', 15))



class physics_cannister(Targetname, Angles):
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



class info_constraint_anchor(Targetname, Parentname):
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



class phys_thruster(ForceController, Angles):
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



class phys_magnet(Targetname, Angles, Studiomodel, Parentname):
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

    @property
    def generatelightmaps(self):
        return self._raw_data.get('generatelightmaps', "0")

    @property
    def lightmapresolutionx(self):
        return parse_source_value(self._raw_data.get('lightmapresolutionx', 32))

    @property
    def lightmapresolutiony(self):
        return parse_source_value(self._raw_data.get('lightmapresolutiony', 32))



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



class prop_dynamic_base(Parentname, Global, BaseFadeProp, DXLevelChoice, Angles, BreakableProp, RenderFields, Studiomodel):

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



class BasePropPhysics(Global, BaseFadeProp, DXLevelChoice, Angles, BreakableProp, Studiomodel):

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



class prop_ragdoll(Studiomodel, Targetname, DXLevelChoice, Angles, EnableDisable, BaseFadeProp):
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



class func_conveyor(Targetname, RenderFields, Shadow, Parentname):

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


class func_illusionary(Origin, Parentname, Targetname, Shadow, RenderFields):

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)



class func_precipitation(Targetname, Parentname):

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


class func_guntarget(Targetname, Global, RenderFields, Parentname):

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



class Trackchange(Parentname, Targetname, Global, PlatSounds, RenderFields):

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



class BaseTrain(Origin, Parentname, Targetname, Global, Shadow, RenderFields):

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



class func_traincontrols(Global, Parentname):

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



class path_track(Targetname, Angles, Parentname):
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



class trigger_impact(Targetname, Angles, Origin):

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



class trigger_teleport_relative(Trigger):

    @property
    def teleportoffset(self):
        return parse_float_vector(self._raw_data.get('teleportoffset', "0 0 0"))



class trigger_transition(Targetname):
    pass


class trigger_serverragdoll(Targetname):
    pass


class ai_speechfilter(Targetname, EnableDisable, ResponseContext):
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



class npc_furniture(BaseNPC, Parentname):
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


class material_modify_control(Targetname, Parentname):
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



class logic_playerproxy(Targetname, DamageFilter):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class env_spritetrail(Targetname, Parentname):
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



class env_projectedtexture(Targetname, Angles, Parentname):
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


class npc_puppet(BaseNPC, Studiomodel, Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def animationtarget(self):
        return self._raw_data.get('animationtarget', "")

    @property
    def attachmentname(self):
        return self._raw_data.get('attachmentname', "")



class point_gamestats_counter(Targetname, EnableDisable, Origin):
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



class trigger_apply_impulse(Trigger):

    @property
    def impulse_dir(self):
        return parse_float_vector(self._raw_data.get('impulse_dir', "0 0 0"))

    @property
    def force(self):
        return parse_source_value(self._raw_data.get('force', 300))



class TeamNum(Base):

    @property
    def TeamNum(self):
        return self._raw_data.get('teamnum', "0")



class MatchSummary(Base):

    @property
    def MatchSummary(self):
        return self._raw_data.get('matchsummary', "0")



class FadeDistance(Base):

    @property
    def fademindist(self):
        return parse_source_value(self._raw_data.get('fademindist', -1))

    @property
    def fademaxdist(self):
        return parse_source_value(self._raw_data.get('fademaxdist', 0))



class GameType(Base):

    @property
    def GameType(self):
        return self._raw_data.get('gametype', "0")



class Condition(Base):

    @property
    def condition(self):
        return self._raw_data.get('condition', "-1")



class PlayerTouch(Base):
    pass


class Toggle(Base):
    pass


class tf_gamerules(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def hud_type(self):
        return self._raw_data.get('hud_type', "0")

    @property
    def ctf_overtime(self):
        return self._raw_data.get('ctf_overtime', "1")



class info_player_teamspawn(Targetname, TeamNum, Angles, EnableDisable, MatchSummary):
    model_ = "models/editor/playerstart.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def controlpoint(self):
        return self._raw_data.get('controlpoint', None)

    @property
    def SpawnMode(self):
        return self._raw_data.get('spawnmode', "0")

    @property
    def round_bluespawn(self):
        return self._raw_data.get('round_bluespawn', None)

    @property
    def round_redspawn(self):
        return self._raw_data.get('round_redspawn', None)



class game_forcerespawn(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_teamflag(Parentname, Targetname, TeamNum, GameType, Angles, EnableDisable):
    model_ = "models/flag/briefcase.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def ReturnTime(self):
        return parse_source_value(self._raw_data.get('returntime', 60))

    @property
    def NeutralType(self):
        return self._raw_data.get('neutraltype', "1")

    @property
    def ScoringType(self):
        return self._raw_data.get('scoringtype', "0")

    @property
    def flag_model(self):
        return self._raw_data.get('flag_model', "models/flag/briefcase.mdl")

    @property
    def flag_icon(self):
        return self._raw_data.get('flag_icon', "../hud/objectives_flagpanel_carried")

    @property
    def flag_paper(self):
        return self._raw_data.get('flag_paper', "player_intel_papertrail")

    @property
    def flag_trail(self):
        return self._raw_data.get('flag_trail', "flagtrail")

    @property
    def trail_effect(self):
        return self._raw_data.get('trail_effect', "1")

    @property
    def VisibleWhenDisabled(self):
        return self._raw_data.get('visiblewhendisabled', "0")

    @property
    def ShotClockMode(self):
        return self._raw_data.get('shotclockmode', "0")

    @property
    def PointValue(self):
        return parse_source_value(self._raw_data.get('pointvalue', 0))

    @property
    def ReturnBetweenWaves(self):
        return self._raw_data.get('returnbetweenwaves', "1")

    @property
    def tags(self):
        return self._raw_data.get('tags', "")



class point_intermission(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class info_observer_point(Parentname, Targetname, TeamNum, Angles, EnableDisable):
    viewport_model = "models/editor/camera.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def associated_team_entity(self):
        return self._raw_data.get('associated_team_entity', "")

    @property
    def defaultwelcome(self):
        return self._raw_data.get('defaultwelcome', "0")

    @property
    def fov(self):
        return parse_source_value(self._raw_data.get('fov', 0))

    @property
    def match_summary(self):
        return self._raw_data.get('match_summary', "0")



class game_round_win(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def TeamNum(self):
        return self._raw_data.get('teamnum', "0")

    @property
    def force_map_reset(self):
        return self._raw_data.get('force_map_reset', "1")

    @property
    def switch_teams(self):
        return self._raw_data.get('switch_teams', "0")



class team_round_timer(Targetname, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def timer_length(self):
        return parse_source_value(self._raw_data.get('timer_length', 600))

    @property
    def max_length(self):
        return parse_source_value(self._raw_data.get('max_length', 0))

    @property
    def start_paused(self):
        return self._raw_data.get('start_paused', "1")

    @property
    def show_time_remaining(self):
        return self._raw_data.get('show_time_remaining', "1")

    @property
    def setup_length(self):
        return parse_source_value(self._raw_data.get('setup_length', 0))

    @property
    def reset_time(self):
        return self._raw_data.get('reset_time', "0")

    @property
    def auto_countdown(self):
        return self._raw_data.get('auto_countdown', "1")

    @property
    def show_in_hud(self):
        return self._raw_data.get('show_in_hud', "1")



class Item(FadeDistance, PlayerTouch, Targetname, TeamNum, Angles, Toggle, EnableDisable):

    @property
    def powerup_model(self):
        return self._raw_data.get('powerup_model', "")

    @property
    def AutoMaterialize(self):
        return self._raw_data.get('automaterialize', "1")



class item_healthkit_full(Item):
    model_ = "models/items/medkit_large.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_healthkit_small(Item):
    model_ = "models/items/medkit_small.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_healthkit_medium(Item):
    model_ = "models/items/medkit_medium.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_ammopack_full(Item):
    model_ = "models/items/ammopack_large.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_ammopack_small(Item):
    model_ = "models/items/ammopack_small.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_ammopack_medium(Item):
    model_ = "models/items/ammopack_medium.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class tf_spell_pickup(Item):
    model_ = "models/props_halloween/hwn_spellbook_flying.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def tier(self):
        return self._raw_data.get('tier', "0")



class item_bonuspack(FadeDistance, PlayerTouch, Targetname, TeamNum, Angles, Toggle, EnableDisable):
    model_ = "models/crafting/moustachium.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def powerup_model(self):
        return self._raw_data.get('powerup_model', "")



class tf_halloween_pickup(Item, Parentname):
    model_ = "models/items/target_duck.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def pickup_sound(self):
        return self._raw_data.get('pickup_sound', "")

    @property
    def pickup_particle(self):
        return self._raw_data.get('pickup_particle', "")



class info_powerup_spawn(Targetname, EnableDisable):
    model_ = "models/pickups/pickup_powerup_regen.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def team(self):
        return self._raw_data.get('team', "-2")



class item_powerup_crit(Targetname, EnableDisable):
    model_ = "models/pickups/pickup_powerup_crit.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_powerup_uber(Targetname, EnableDisable):
    model_ = "models/pickups/pickup_powerup_uber.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class filter_activator_tfteam(TeamNum, BaseFilter):
    icon_sprite = "editor/filter_team.vmt"

    @property
    def controlpoint(self):
        return self._raw_data.get('controlpoint', "")



class filter_tf_player_can_cap(TeamNum, BaseFilter):
    icon_sprite = "editor/filter_team.vmt"
    pass


class filter_tf_damaged_by_weapon_in_slot(BaseFilter):

    @property
    def weaponSlot(self):
        return self._raw_data.get('weaponslot', "0")



class filter_tf_condition(BaseFilter, Condition):
    pass


class filter_tf_class(BaseFilter):
    icon_sprite = "editor/filter_class.vmt"

    @property
    def tfclass(self):
        return self._raw_data.get('tfclass', "0")



class func_capturezone(Targetname, TeamNum, EnableDisable):

    @property
    def capturepoint(self):
        return parse_source_value(self._raw_data.get('capturepoint', 1))

    @property
    def capture_delay(self):
        return parse_source_value(self._raw_data.get('capture_delay', 1.1))

    @property
    def capture_delay_offset(self):
        return parse_source_value(self._raw_data.get('capture_delay_offset', 0.025))

    @property
    def shouldBlock(self):
        return self._raw_data.get('shouldblock', "1")



class func_flagdetectionzone(Targetname, TeamNum, EnableDisable, Parentname):

    @property
    def alarm(self):
        return self._raw_data.get('alarm', "0")



class func_nogrenades(Targetname, TeamNum, EnableDisable, Toggle):
    pass


class func_achievement(Targetname, TeamNum, EnableDisable, Toggle):

    @property
    def zone_id(self):
        return parse_source_value(self._raw_data.get('zone_id', 0))



class func_nobuild(Targetname, TeamNum, EnableDisable, Toggle):

    @property
    def AllowSentry(self):
        return self._raw_data.get('allowsentry', "0")

    @property
    def AllowDispenser(self):
        return self._raw_data.get('allowdispenser', "0")

    @property
    def AllowTeleporters(self):
        return self._raw_data.get('allowteleporters', "0")



class func_croc(Targetname, TeamNum, EnableDisable, Toggle):

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)



class func_suggested_build(Origin, Targetname, TeamNum, Toggle, EnableDisable):

    @property
    def object_type(self):
        return self._raw_data.get('object_type', "0")

    @property
    def face_entity(self):
        return self._raw_data.get('face_entity', None)

    @property
    def face_entity_fov(self):
        return parse_source_value(self._raw_data.get('face_entity_fov', 90))



class func_regenerate(Targetname, TeamNum, EnableDisable, Toggle):

    @property
    def associatedmodel(self):
        return self._raw_data.get('associatedmodel', "")



class func_powerupvolume(Trigger, TeamNum, Toggle):
    pass


class func_respawnflag(Targetname, EnableDisable, Toggle):
    pass


class func_respawnroom(Targetname, TeamNum, EnableDisable, Toggle):
    pass


class func_flag_alert(Targetname, TeamNum, EnableDisable, Toggle):

    @property
    def playsound(self):
        return self._raw_data.get('playsound', "1")

    @property
    def alert_delay(self):
        return parse_source_value(self._raw_data.get('alert_delay', 10))



class func_respawnroomvisualizer(Origin, Parentname, Targetname, Global, Inputfilter, EnableDisable, Shadow, RenderFields):

    @property
    def respawnroomname(self):
        return self._raw_data.get('respawnroomname', "")

    @property
    def Solidity(self):
        return self._raw_data.get('solidity', "1")

    @property
    def vrad_brush_cast_shadows(self):
        return self._raw_data.get('vrad_brush_cast_shadows', "0")

    @property
    def solid_to_enemies(self):
        return self._raw_data.get('solid_to_enemies', "1")



class func_forcefield(Origin, Parentname, Targetname, TeamNum, EnableDisable, RenderFields):
    pass


class func_changeclass(Targetname, TeamNum, EnableDisable, Toggle):
    pass


class game_intro_viewpoint(Angles):
    viewport_model = "models/editor/camera.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def TeamNum(self):
        return self._raw_data.get('teamnum', "0")

    @property
    def step_number(self):
        return parse_source_value(self._raw_data.get('step_number', 1))

    @property
    def time_delay(self):
        return parse_source_value(self._raw_data.get('time_delay', 12))

    @property
    def hint_message(self):
        return self._raw_data.get('hint_message', None)

    @property
    def event_to_fire(self):
        return self._raw_data.get('event_to_fire', None)

    @property
    def event_delay(self):
        return parse_source_value(self._raw_data.get('event_delay', 3))

    @property
    def event_data_int(self):
        return parse_source_value(self._raw_data.get('event_data_int', 0))

    @property
    def fov(self):
        return parse_source_value(self._raw_data.get('fov', 0))



class func_proprrespawnzone(Targetname):
    pass


class team_control_point(Targetname, Angles, EnableDisable, Parentname):
    model_ = "models/effects/cappoint_hologram.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def point_start_locked(self):
        return self._raw_data.get('point_start_locked', "0")

    @property
    def point_printname(self):
        return self._raw_data.get('point_printname', "TODO: Set Name")

    @property
    def point_group(self):
        return parse_source_value(self._raw_data.get('point_group', 0))

    @property
    def point_default_owner(self):
        return self._raw_data.get('point_default_owner', "0")

    @property
    def point_index(self):
        return parse_source_value(self._raw_data.get('point_index', 0))

    @property
    def point_warn_on_cap(self):
        return self._raw_data.get('point_warn_on_cap', "0")

    @property
    def point_warn_sound(self):
        return self._raw_data.get('point_warn_sound', "ControlPoint.CaptureWarn")

    @property
    def random_owner_on_restart(self):
        return self._raw_data.get('random_owner_on_restart', "0")

    @property
    def team_timedpoints_2(self):
        return parse_source_value(self._raw_data.get('team_timedpoints_2', 0))

    @property
    def team_timedpoints_3(self):
        return parse_source_value(self._raw_data.get('team_timedpoints_3', 0))

    @property
    def team_capsound_0(self):
        return self._raw_data.get('team_capsound_0', "")

    @property
    def team_capsound_2(self):
        return self._raw_data.get('team_capsound_2', "")

    @property
    def team_capsound_3(self):
        return self._raw_data.get('team_capsound_3', "")

    @property
    def team_model_0(self):
        return self._raw_data.get('team_model_0', "models/effects/cappoint_hologram.mdl")

    @property
    def team_model_2(self):
        return self._raw_data.get('team_model_2', "models/effects/cappoint_hologram.mdl")

    @property
    def team_model_3(self):
        return self._raw_data.get('team_model_3', "models/effects/cappoint_hologram.mdl")

    @property
    def team_bodygroup_0(self):
        return parse_source_value(self._raw_data.get('team_bodygroup_0', 3))

    @property
    def team_bodygroup_2(self):
        return parse_source_value(self._raw_data.get('team_bodygroup_2', 1))

    @property
    def team_bodygroup_3(self):
        return parse_source_value(self._raw_data.get('team_bodygroup_3', 1))

    @property
    def team_icon_0(self):
        return self._raw_data.get('team_icon_0', "sprites/obj_icons/icon_obj_neutral")

    @property
    def team_icon_2(self):
        return self._raw_data.get('team_icon_2', "sprites/obj_icons/icon_obj_red")

    @property
    def team_icon_3(self):
        return self._raw_data.get('team_icon_3', "sprites/obj_icons/icon_obj_blu")

    @property
    def team_overlay_0(self):
        return self._raw_data.get('team_overlay_0', "")

    @property
    def team_overlay_2(self):
        return self._raw_data.get('team_overlay_2', "")

    @property
    def team_overlay_3(self):
        return self._raw_data.get('team_overlay_3', "")

    @property
    def team_previouspoint_2_0(self):
        return self._raw_data.get('team_previouspoint_2_0', "")

    @property
    def team_previouspoint_2_1(self):
        return self._raw_data.get('team_previouspoint_2_1', "")

    @property
    def team_previouspoint_2_2(self):
        return self._raw_data.get('team_previouspoint_2_2', "")

    @property
    def team_previouspoint_3_0(self):
        return self._raw_data.get('team_previouspoint_3_0', "")

    @property
    def team_previouspoint_3_1(self):
        return self._raw_data.get('team_previouspoint_3_1', "")

    @property
    def team_previouspoint_3_2(self):
        return self._raw_data.get('team_previouspoint_3_2', "")



class team_control_point_round(Targetname, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def cpr_printname(self):
        return self._raw_data.get('cpr_printname', "")

    @property
    def cpr_priority(self):
        return parse_source_value(self._raw_data.get('cpr_priority', 0))

    @property
    def cpr_cp_names(self):
        return self._raw_data.get('cpr_cp_names', "")

    @property
    def cpr_restrict_team_cap_win(self):
        return self._raw_data.get('cpr_restrict_team_cap_win', "0")



class team_control_point_master(Targetname, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def team_base_icon_2(self):
        return self._raw_data.get('team_base_icon_2', "sprites/obj_icons/icon_base_red")

    @property
    def team_base_icon_3(self):
        return self._raw_data.get('team_base_icon_3', "sprites/obj_icons/icon_base_blu")

    @property
    def caplayout(self):
        return self._raw_data.get('caplayout', "")

    @property
    def custom_position_x(self):
        return parse_source_value(self._raw_data.get('custom_position_x', -1))

    @property
    def custom_position_y(self):
        return parse_source_value(self._raw_data.get('custom_position_y', -1))

    @property
    def cpm_restrict_team_cap_win(self):
        return self._raw_data.get('cpm_restrict_team_cap_win', "0")

    @property
    def switch_teams(self):
        return self._raw_data.get('switch_teams', "0")

    @property
    def score_style(self):
        return self._raw_data.get('score_style', "0")

    @property
    def play_all_rounds(self):
        return self._raw_data.get('play_all_rounds', "0")

    @property
    def partial_cap_points_rate(self):
        return parse_source_value(self._raw_data.get('partial_cap_points_rate', 0))



class trigger_capture_area(Targetname, EnableDisable, Parentname):

    @property
    def area_cap_point(self):
        return self._raw_data.get('area_cap_point', "")

    @property
    def team_cancap_2(self):
        return self._raw_data.get('team_cancap_2', "1")

    @property
    def team_cancap_3(self):
        return self._raw_data.get('team_cancap_3', "1")

    @property
    def team_startcap_2(self):
        return parse_source_value(self._raw_data.get('team_startcap_2', 1))

    @property
    def team_startcap_3(self):
        return parse_source_value(self._raw_data.get('team_startcap_3', 1))

    @property
    def team_numcap_2(self):
        return parse_source_value(self._raw_data.get('team_numcap_2', 1))

    @property
    def team_numcap_3(self):
        return parse_source_value(self._raw_data.get('team_numcap_3', 1))

    @property
    def team_spawn_2(self):
        return parse_source_value(self._raw_data.get('team_spawn_2', 0))

    @property
    def team_spawn_3(self):
        return parse_source_value(self._raw_data.get('team_spawn_3', 0))

    @property
    def area_time_to_cap(self):
        return parse_source_value(self._raw_data.get('area_time_to_cap', 5))



class team_train_watcher(Targetname, TeamNum, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def train_can_recede(self):
        return self._raw_data.get('train_can_recede', "1")

    @property
    def train_recede_time(self):
        return parse_source_value(self._raw_data.get('train_recede_time', 0))

    @property
    def handle_train_movement(self):
        return self._raw_data.get('handle_train_movement', "0")

    @property
    def speed_forward_modifier(self):
        return parse_source_value(self._raw_data.get('speed_forward_modifier', 1))

    @property
    def env_spark_name(self):
        return self._raw_data.get('env_spark_name', "")

    @property
    def train(self):
        return self._raw_data.get('train', None)

    @property
    def start_node(self):
        return self._raw_data.get('start_node', None)

    @property
    def goal_node(self):
        return self._raw_data.get('goal_node', None)

    @property
    def linked_pathtrack_1(self):
        return self._raw_data.get('linked_pathtrack_1', None)

    @property
    def linked_cp_1(self):
        return self._raw_data.get('linked_cp_1', None)

    @property
    def linked_pathtrack_2(self):
        return self._raw_data.get('linked_pathtrack_2', None)

    @property
    def linked_cp_2(self):
        return self._raw_data.get('linked_cp_2', None)

    @property
    def linked_pathtrack_3(self):
        return self._raw_data.get('linked_pathtrack_3', None)

    @property
    def linked_cp_3(self):
        return self._raw_data.get('linked_cp_3', None)

    @property
    def linked_pathtrack_4(self):
        return self._raw_data.get('linked_pathtrack_4', None)

    @property
    def linked_cp_4(self):
        return self._raw_data.get('linked_cp_4', None)

    @property
    def linked_pathtrack_5(self):
        return self._raw_data.get('linked_pathtrack_5', None)

    @property
    def linked_cp_5(self):
        return self._raw_data.get('linked_cp_5', None)

    @property
    def linked_pathtrack_6(self):
        return self._raw_data.get('linked_pathtrack_6', None)

    @property
    def linked_cp_6(self):
        return self._raw_data.get('linked_cp_6', None)

    @property
    def linked_pathtrack_7(self):
        return self._raw_data.get('linked_pathtrack_7', None)

    @property
    def linked_cp_7(self):
        return self._raw_data.get('linked_cp_7', None)

    @property
    def linked_pathtrack_8(self):
        return self._raw_data.get('linked_pathtrack_8', None)

    @property
    def linked_cp_8(self):
        return self._raw_data.get('linked_cp_8', None)

    @property
    def hud_min_speed_level_1(self):
        return parse_source_value(self._raw_data.get('hud_min_speed_level_1', 30))

    @property
    def hud_min_speed_level_2(self):
        return parse_source_value(self._raw_data.get('hud_min_speed_level_2', 60))

    @property
    def hud_min_speed_level_3(self):
        return parse_source_value(self._raw_data.get('hud_min_speed_level_3', 90))



class game_text_tf(Targetname):
    icon_sprite = "editor/game_text.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def message(self):
        return self._raw_data.get('message', "")

    @property
    def icon(self):
        return self._raw_data.get('icon', "")

    @property
    def display_to_team(self):
        return self._raw_data.get('display_to_team', "0")

    @property
    def background(self):
        return self._raw_data.get('background', "0")



class BaseObject(Base):

    @property
    def TeamNum(self):
        return self._raw_data.get('teamnum', "2")



class obj_dispenser(Targetname, Angles, BaseObject, Parentname):
    model_ = "models/buildables/dispenser_light.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def defaultupgrade(self):
        return self._raw_data.get('defaultupgrade', "0")



class obj_sentrygun(Targetname, Angles, BaseObject, Parentname):
    model_ = "models/buildables/sentry3.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def defaultupgrade(self):
        return self._raw_data.get('defaultupgrade', "0")



class obj_teleporter(Targetname, Angles, BaseObject, Parentname):
    model_ = "models/buildables/teleporter_light.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def defaultupgrade(self):
        return self._raw_data.get('defaultupgrade', "0")

    @property
    def teleporterType(self):
        return self._raw_data.get('teleportertype', "1")

    @property
    def matchingTeleporter(self):
        return self._raw_data.get('matchingteleporter', None)



class bot_hint_sentrygun(Parentname, Targetname, Angles, EnableDisable, BaseObject):
    model_ = "models/buildables/sentry3.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def sequence(self):
        return parse_source_value(self._raw_data.get('sequence', 5))

    @property
    def sticky(self):
        return self._raw_data.get('sticky', "0")



class bot_hint_teleporter_exit(Parentname, Targetname, Angles, EnableDisable, BaseObject):
    model_ = "models/buildables/teleporter_blueprint_exit.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class bot_hint_engineer_nest(Parentname, Targetname, Angles, EnableDisable, BaseObject):
    model_ = "models/bots/engineer/bot_engineer.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class bot_hint_sniper_spot(Targetname, Angles, BaseObject, Parentname):
    model_ = "models/player/sniper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 100))



class mapobj_cart_dispenser(Targetname, BaseObject, Parentname):
    icon_sprite = "editor/bullseye.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def touch_trigger(self):
        return self._raw_data.get('touch_trigger', None)



class dispenser_touch_trigger(Trigger):
    pass


class tf_logic_arena(Targetname, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def CapEnableDelay(self):
        return parse_source_value(self._raw_data.get('capenabledelay', 0))



class tf_logic_competitive(Targetname, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class tf_logic_mannpower(Targetname, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class bot_controller(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def TeamNum(self):
        return self._raw_data.get('teamnum', "2")

    @property
    def bot_class(self):
        return self._raw_data.get('bot_class', "0")

    @property
    def bot_name(self):
        return self._raw_data.get('bot_name', None)



class tf_logic_training_mode(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def nextMap(self):
        return self._raw_data.get('nextmap', "")



class boss_alpha(Targetname, Angles):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class base_boss(Targetname, TeamNum, Angles, Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 1000))

    @property
    def model(self):
        return self._raw_data.get('model', "models/bots/boss_bot/boss_tank.mdl")

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 75))

    @property
    def start_disabled(self):
        return self._raw_data.get('start_disabled', "0")



class tank_boss(base_boss):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class tf_logic_multiple_escort(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class tf_logic_koth(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def timer_length(self):
        return parse_source_value(self._raw_data.get('timer_length', 180))

    @property
    def unlock_point(self):
        return parse_source_value(self._raw_data.get('unlock_point', 30))



class tf_robot_destruction_robot_spawn(Targetname, Angles, Parentname):
    model_ = "models/bots/bot_worker/bot_worker_a.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 500))

    @property
    def gibs(self):
        return parse_source_value(self._raw_data.get('gibs', 0))

    @property
    def type(self):
        return self._raw_data.get('type', "0")

    @property
    def spawngroup(self):
        return self._raw_data.get('spawngroup', None)

    @property
    def startpath(self):
        return self._raw_data.get('startpath', None)



class tf_robot_destruction_spawn_group(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def respawn_time(self):
        return parse_source_value(self._raw_data.get('respawn_time', 0))

    @property
    def group_number(self):
        return parse_source_value(self._raw_data.get('group_number', 0))

    @property
    def team_number(self):
        return self._raw_data.get('team_number', "2")

    @property
    def hud_icon(self):
        return self._raw_data.get('hud_icon', "../HUD/hud_bot_worker_outline_blue")

    @property
    def respawn_reduction_scale(self):
        return parse_source_value(self._raw_data.get('respawn_reduction_scale', 0))



class RobotDestruction(Base):
    pass


class tf_logic_robot_destruction(Targetname, RobotDestruction):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def score_interval(self):
        return parse_source_value(self._raw_data.get('score_interval', 1))

    @property
    def loser_respawn_bonus_per_bot(self):
        return parse_source_value(self._raw_data.get('loser_respawn_bonus_per_bot', 0))

    @property
    def red_respawn_time(self):
        return parse_source_value(self._raw_data.get('red_respawn_time', 10))

    @property
    def blue_respawn_time(self):
        return parse_source_value(self._raw_data.get('blue_respawn_time', 10))

    @property
    def max_points(self):
        return parse_source_value(self._raw_data.get('max_points', 200))

    @property
    def finale_length(self):
        return parse_source_value(self._raw_data.get('finale_length', 30))

    @property
    def res_file(self):
        return self._raw_data.get('res_file', "resource/UI/HudObjectiveRobotDestruction.res")



class tf_logic_player_destruction(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def prop_model_name(self):
        return self._raw_data.get('prop_model_name', "models/flag/flag.mdl")

    @property
    def prop_drop_sound(self):
        return self._raw_data.get('prop_drop_sound', "")

    @property
    def prop_pickup_sound(self):
        return self._raw_data.get('prop_pickup_sound', "")

    @property
    def red_respawn_time(self):
        return parse_source_value(self._raw_data.get('red_respawn_time', 10))

    @property
    def blue_respawn_time(self):
        return parse_source_value(self._raw_data.get('blue_respawn_time', 10))

    @property
    def min_points(self):
        return parse_source_value(self._raw_data.get('min_points', 10))

    @property
    def points_per_player(self):
        return parse_source_value(self._raw_data.get('points_per_player', 5))

    @property
    def finale_length(self):
        return parse_source_value(self._raw_data.get('finale_length', 30))

    @property
    def res_file(self):
        return self._raw_data.get('res_file', "resource/UI/HudObjectivePlayerDestruction.res")

    @property
    def flag_reset_delay(self):
        return parse_source_value(self._raw_data.get('flag_reset_delay', 60))

    @property
    def heal_distance(self):
        return parse_source_value(self._raw_data.get('heal_distance', 450))



class trigger_rd_vault_trigger(TeamNum, Trigger):
    pass


class tf_logic_medieval(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class tf_logic_cp_timer(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def controlpoint(self):
        return self._raw_data.get('controlpoint', None)

    @property
    def timer_length(self):
        return parse_source_value(self._raw_data.get('timer_length', 60))



class tf_logic_hybrid_ctf_cp(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class tf_logic_raid(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class tf_logic_boss_battle(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class tf_logic_mann_vs_machine(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class tf_logic_holiday(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def holiday_type(self):
        return self._raw_data.get('holiday_type', "1")

    @property
    def tauntInHell(self):
        return self._raw_data.get('tauntinhell', "0")

    @property
    def allowHaunting(self):
        return self._raw_data.get('allowhaunting', "0")



class func_upgradestation(Targetname, EnableDisable):
    pass


class bot_generator(Targetname, Angles):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def team(self):
        return self._raw_data.get('team', "auto")

    @property
    def class_(self):
        return self._raw_data.get('class', "auto")

    @property
    def count(self):
        return parse_source_value(self._raw_data.get('count', 1))

    @property
    def maxActive(self):
        return parse_source_value(self._raw_data.get('maxactive', 1))

    @property
    def interval(self):
        return parse_source_value(self._raw_data.get('interval', 0))

    @property
    def action_point(self):
        return self._raw_data.get('action_point', None)

    @property
    def initial_command(self):
        return self._raw_data.get('initial_command', "")

    @property
    def suppressFire(self):
        return self._raw_data.get('suppressfire', "0")

    @property
    def disableDodge(self):
        return self._raw_data.get('disabledodge', "0")

    @property
    def actionOnDeath(self):
        return self._raw_data.get('actionondeath', "1")

    @property
    def spectateOnDeath(self):
        return self._raw_data.get('spectateondeath', "0")

    @property
    def useTeamSpawnPoint(self):
        return self._raw_data.get('useteamspawnpoint', "0")

    @property
    def retainBuildings(self):
        return self._raw_data.get('retainbuildings', "0")

    @property
    def difficulty(self):
        return self._raw_data.get('difficulty', "0")

    @property
    def spawnOnlyWhenTriggered(self):
        return self._raw_data.get('spawnonlywhentriggered', "0")



class bot_roster(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def team(self):
        return self._raw_data.get('team', "auto")

    @property
    def allowClassChanges(self):
        return self._raw_data.get('allowclasschanges', "1")

    @property
    def allowScout(self):
        return self._raw_data.get('allowscout', "1")

    @property
    def allowSniper(self):
        return self._raw_data.get('allowsniper', "1")

    @property
    def allowSoldier(self):
        return self._raw_data.get('allowsoldier', "1")

    @property
    def allowDemoman(self):
        return self._raw_data.get('allowdemoman', "1")

    @property
    def allowMedic(self):
        return self._raw_data.get('allowmedic', "1")

    @property
    def allowHeavy(self):
        return self._raw_data.get('allowheavy', "1")

    @property
    def allowPyro(self):
        return self._raw_data.get('allowpyro', "1")

    @property
    def allowSpy(self):
        return self._raw_data.get('allowspy', "1")

    @property
    def allowEngineer(self):
        return self._raw_data.get('allowengineer', "1")



class bot_action_point(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def next_action_point(self):
        return self._raw_data.get('next_action_point', None)

    @property
    def desired_distance(self):
        return parse_source_value(self._raw_data.get('desired_distance', 5))

    @property
    def stay_time(self):
        return parse_source_value(self._raw_data.get('stay_time', 0))

    @property
    def command(self):
        return self._raw_data.get('command', "")



class bot_proxy(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def bot_name(self):
        return self._raw_data.get('bot_name', "TFBot")

    @property
    def team(self):
        return self._raw_data.get('team', "auto")

    @property
    def class_(self):
        return self._raw_data.get('class', "auto")

    @property
    def spawn_on_start(self):
        return self._raw_data.get('spawn_on_start', "0")

    @property
    def respawn_interval(self):
        return parse_source_value(self._raw_data.get('respawn_interval', 0))

    @property
    def action_point(self):
        return self._raw_data.get('action_point', None)



class tf_spawner(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def count(self):
        return parse_source_value(self._raw_data.get('count', 1))

    @property
    def maxActive(self):
        return parse_source_value(self._raw_data.get('maxactive', 1))

    @property
    def interval(self):
        return parse_source_value(self._raw_data.get('interval', 0))

    @property
    def template(self):
        return self._raw_data.get('template', None)



class tf_template_stun_drone(Targetname, Angles, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class func_nav_blocker(Targetname):

    @property
    def teamToBlock(self):
        return self._raw_data.get('teamtoblock', "-1")

    @property
    def affectsFlow(self):
        return self._raw_data.get('affectsflow', "0")



class func_nav_avoid(Targetname):

    @property
    def tags(self):
        return self._raw_data.get('tags', "")

    @property
    def team(self):
        return self._raw_data.get('team', "-2")

    @property
    def start_disabled(self):
        return self._raw_data.get('start_disabled', "0")



class func_nav_prefer(Targetname):

    @property
    def tags(self):
        return self._raw_data.get('tags', "")

    @property
    def team(self):
        return self._raw_data.get('team', "-2")

    @property
    def start_disabled(self):
        return self._raw_data.get('start_disabled', "0")



class func_nav_prerequisite(Trigger):

    @property
    def Task(self):
        return self._raw_data.get('task', "1")

    @property
    def Entity(self):
        return self._raw_data.get('entity', None)

    @property
    def Value(self):
        return parse_source_value(self._raw_data.get('value', 0))

    @property
    def start_disabled(self):
        return self._raw_data.get('start_disabled', "0")



class func_tfbot_hint(Targetname, EnableDisable, Origin):

    @property
    def team(self):
        return self._raw_data.get('team', "-1")

    @property
    def hint(self):
        return self._raw_data.get('hint', "0")



class trigger_stun(Trigger):

    @property
    def trigger_delay(self):
        return parse_source_value(self._raw_data.get('trigger_delay', 0))

    @property
    def stun_duration(self):
        return parse_source_value(self._raw_data.get('stun_duration', 0))

    @property
    def move_speed_reduction(self):
        return parse_source_value(self._raw_data.get('move_speed_reduction', 0))

    @property
    def stun_type(self):
        return self._raw_data.get('stun_type', "0")

    @property
    def stun_effects(self):
        return self._raw_data.get('stun_effects', "0")



class entity_spawn_point(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def spawn_manager_name(self):
        return self._raw_data.get('spawn_manager_name', "")



class entity_spawn_manager(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def entity_name(self):
        return self._raw_data.get('entity_name', "")

    @property
    def entity_count(self):
        return parse_source_value(self._raw_data.get('entity_count', 0))

    @property
    def respawn_time(self):
        return parse_source_value(self._raw_data.get('respawn_time', 0))

    @property
    def drop_to_ground(self):
        return self._raw_data.get('drop_to_ground', "0")

    @property
    def random_rotation(self):
        return self._raw_data.get('random_rotation', "0")



class tf_pumpkin_bomb(Item):
    model_ = "models/props_halloween/pumpkin_explode.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class tf_generic_bomb(Targetname, Angles, Origin, Studiomodel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def damage(self):
        return parse_source_value(self._raw_data.get('damage', 50))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 100))

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 1))

    @property
    def explode_particle(self):
        return self._raw_data.get('explode_particle', "")

    @property
    def sound(self):
        return self._raw_data.get('sound', "")

    @property
    def friendlyfire(self):
        return self._raw_data.get('friendlyfire', "0")



class training_annotation(Targetname):
    model_ = "models/extras/info_speech.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def display_text(self):
        return self._raw_data.get('display_text', "<Add Text Here>")

    @property
    def lifetime(self):
        return parse_source_value(self._raw_data.get('lifetime', 0))

    @property
    def offset(self):
        return parse_source_value(self._raw_data.get('offset', 0))



class training_prop_dynamic(prop_dynamic):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class trigger_ignite_arrows(Trigger):
    pass


class wheel_of_doom(Targetname, Angles, Parentname):
    model_ = "models/editor/cone_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def effect_duration(self):
        return parse_source_value(self._raw_data.get('effect_duration', 30))

    @property
    def has_spiral(self):
        return self._raw_data.get('has_spiral', "0")



class point_populator_interface(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class tf_point_weapon_mimic(Targetname, Angles, Parentname):
    model_ = "models/editor/cone_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def WeaponType(self):
        return self._raw_data.get('weapontype', "0")

    @property
    def FireSound(self):
        return self._raw_data.get('firesound', "")

    @property
    def ParticleEffect(self):
        return self._raw_data.get('particleeffect', "")

    @property
    def ModelOverride(self):
        return self._raw_data.get('modeloverride', "")

    @property
    def ModelScale(self):
        return parse_source_value(self._raw_data.get('modelscale', 1))

    @property
    def SpeedMin(self):
        return parse_source_value(self._raw_data.get('speedmin', 1000))

    @property
    def SpeedMax(self):
        return parse_source_value(self._raw_data.get('speedmax', 1000))

    @property
    def Damage(self):
        return parse_source_value(self._raw_data.get('damage', 75))

    @property
    def SplashRadius(self):
        return parse_source_value(self._raw_data.get('splashradius', 50))

    @property
    def SpreadAngle(self):
        return parse_source_value(self._raw_data.get('spreadangle', 0))

    @property
    def Crits(self):
        return self._raw_data.get('crits', "0")



class tf_point_nav_interface(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class trigger_timer_door(Trigger):

    @property
    def door_name(self):
        return self._raw_data.get('door_name', "")

    @property
    def area_cap_point(self):
        return self._raw_data.get('area_cap_point', "")

    @property
    def team_cancap_2(self):
        return self._raw_data.get('team_cancap_2', "1")

    @property
    def team_cancap_3(self):
        return self._raw_data.get('team_cancap_3', "1")

    @property
    def team_startcap_2(self):
        return parse_source_value(self._raw_data.get('team_startcap_2', 1))

    @property
    def team_startcap_3(self):
        return parse_source_value(self._raw_data.get('team_startcap_3', 1))

    @property
    def team_numcap_2(self):
        return parse_source_value(self._raw_data.get('team_numcap_2', 1))

    @property
    def team_numcap_3(self):
        return parse_source_value(self._raw_data.get('team_numcap_3', 1))

    @property
    def team_spawn_2(self):
        return parse_source_value(self._raw_data.get('team_spawn_2', 0))

    @property
    def team_spawn_3(self):
        return parse_source_value(self._raw_data.get('team_spawn_3', 0))

    @property
    def area_time_to_cap(self):
        return parse_source_value(self._raw_data.get('area_time_to_cap', 5))



class trigger_bot_tag(Trigger):

    @property
    def tags(self):
        return self._raw_data.get('tags', "")

    @property
    def add(self):
        return self._raw_data.get('add', "1")



class filter_tf_bot_has_tag(BaseFilter):

    @property
    def tags(self):
        return self._raw_data.get('tags', "")

    @property
    def require_all_tags(self):
        return self._raw_data.get('require_all_tags', "1")



class trigger_add_tf_player_condition(Condition, Trigger):

    @property
    def duration(self):
        return parse_source_value(self._raw_data.get('duration', 0))



class trigger_remove_tf_player_condition(Condition, Trigger):
    pass


class hightower_teleport_vortex(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target_base_name(self):
        return self._raw_data.get('target_base_name', None)

    @property
    def lifetime(self):
        return parse_source_value(self._raw_data.get('lifetime', 0))



class tf_zombie_spawner(Targetname):
    model_ = "models/bots/skeleton_sniper/skeleton_sniper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def zombie_lifetime(self):
        return parse_source_value(self._raw_data.get('zombie_lifetime', 0))

    @property
    def max_zombies(self):
        return parse_source_value(self._raw_data.get('max_zombies', 1))

    @property
    def infinite_zombies(self):
        return self._raw_data.get('infinite_zombies', "0")

    @property
    def zombie_type(self):
        return self._raw_data.get('zombie_type', "0")



class halloween_zapper(Targetname, Parentname):
    icon_sprite = "editor/bullseye.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def touch_trigger(self):
        return self._raw_data.get('touch_trigger', None)

    @property
    def ParticleEffect(self):
        return self._raw_data.get('particleeffect', "")

    @property
    def ZapperType(self):
        return self._raw_data.get('zappertype', "0")



class trigger_player_respawn_override(Trigger, Toggle):

    @property
    def RespawnTime(self):
        return parse_source_value(self._raw_data.get('respawntime', -1))

    @property
    def RespawnName(self):
        return self._raw_data.get('respawnname', "")



class prop_soccer_ball(Targetname, Studiomodel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def trigger_name(self):
        return self._raw_data.get('trigger_name', "")



class MiniGame(Base):
    pass


class tf_logic_minigames(Targetname, MiniGame):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class tf_base_minigame(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def RedSpawn(self):
        return self._raw_data.get('redspawn', "")

    @property
    def BlueSpawn(self):
        return self._raw_data.get('bluespawn', "")

    @property
    def InRandomPool(self):
        return self._raw_data.get('inrandompool', "1")

    @property
    def MaxScore(self):
        return parse_source_value(self._raw_data.get('maxscore', 5))

    @property
    def hud_res_file(self):
        return self._raw_data.get('hud_res_file', "")

    @property
    def your_team_score_sound(self):
        return self._raw_data.get('your_team_score_sound', "")

    @property
    def enemy_team_score_sound(self):
        return self._raw_data.get('enemy_team_score_sound', "")

    @property
    def ScoreType(self):
        return self._raw_data.get('scoretype', "0")

    @property
    def SuddenDeathTime(self):
        return parse_source_value(self._raw_data.get('suddendeathtime', -1))



class tf_halloween_minigame(tf_base_minigame):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def MinigameType(self):
        return self._raw_data.get('minigametype', "1")



class tf_halloween_minigame_falling_platforms(tf_halloween_minigame):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class halloween_fortune_teller(Targetname, Angles, Origin):
    model_ = "models/bots/merasmus/merasmas_misfortune_teller.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def red_teleport(self):
        return self._raw_data.get('red_teleport', "")

    @property
    def blue_teleport(self):
        return self._raw_data.get('blue_teleport', "")



class tf_teleport_location(Targetname, Angles, Origin):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class func_passtime_goal(Targetname, TeamNum, EnableDisable, Origin):

    @property
    def points(self):
        return parse_source_value(self._raw_data.get('points', 1))



class info_passtime_ball_spawn(Targetname, TeamNum, EnableDisable):
    icon_sprite = "editor/passtime_ball_spawner.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class passtime_logic(Targetname):
    icon_sprite = "editor/passtime_master.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def num_sections(self):
        return parse_source_value(self._raw_data.get('num_sections', 0))

    @property
    def ball_spawn_countdown(self):
        return parse_source_value(self._raw_data.get('ball_spawn_countdown', 15))

    @property
    def max_pass_range(self):
        return parse_source_value(self._raw_data.get('max_pass_range', 0))



class func_passtime_goalie_zone(Targetname, TeamNum, EnableDisable):
    pass


class func_passtime_no_ball_zone(Targetname):
    pass


class trigger_passtime_ball(Targetname):
    pass


class trigger_catapult(Trigger):

    @property
    def playerSpeed(self):
        return parse_source_value(self._raw_data.get('playerspeed', 450))

    @property
    def physicsSpeed(self):
        return parse_source_value(self._raw_data.get('physicsspeed', 450))

    @property
    def useThresholdCheck(self):
        return parse_source_value(self._raw_data.get('usethresholdcheck', 0))

    @property
    def entryAngleTolerance(self):
        return parse_source_value(self._raw_data.get('entryangletolerance', 0.0))

    @property
    def useExactVelocity(self):
        return parse_source_value(self._raw_data.get('useexactvelocity', 0))

    @property
    def exactVelocityChoiceType(self):
        return self._raw_data.get('exactvelocitychoicetype', "0")

    @property
    def lowerThreshold(self):
        return parse_source_value(self._raw_data.get('lowerthreshold', 0.15))

    @property
    def upperThreshold(self):
        return parse_source_value(self._raw_data.get('upperthreshold', 0.30))

    @property
    def launchDirection(self):
        return parse_float_vector(self._raw_data.get('launchdirection', "0 0 0"))

    @property
    def launchTarget(self):
        return self._raw_data.get('launchtarget', "")

    @property
    def onlyVelocityCheck(self):
        return parse_source_value(self._raw_data.get('onlyvelocitycheck', 0))

    @property
    def applyAngularImpulse(self):
        return parse_source_value(self._raw_data.get('applyangularimpulse', 1))

    @property
    def AirCtrlSupressionTime(self):
        return parse_source_value(self._raw_data.get('airctrlsupressiontime', -1.0))



class trigger_ignite(Trigger):

    @property
    def burn_duration(self):
        return parse_source_value(self._raw_data.get('burn_duration', 5))

    @property
    def damage_percent_per_second(self):
        return parse_source_value(self._raw_data.get('damage_percent_per_second', 10))

    @property
    def ignite_particle_name(self):
        return self._raw_data.get('ignite_particle_name', "")

    @property
    def ignite_sound_name(self):
        return self._raw_data.get('ignite_sound_name', "")



class tf_halloween_gift_spawn_location(Targetname, Angles, Origin):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class tf_logic_on_holiday(Targetname):
    icon_sprite = "editor/logic_auto.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class tf_glow(EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def targetname(self):
        return self._raw_data.get('targetname', None)

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def GlowColor(self):
        return parse_int_vector(self._raw_data.get('glowcolor', "255 0 0 255"))

    @property
    def Mode(self):
        return self._raw_data.get('mode', "0")




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
    'trigger_teleport_relative': trigger_teleport_relative,
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
    'trigger_apply_impulse': trigger_apply_impulse,
    'TeamNum': TeamNum,
    'MatchSummary': MatchSummary,
    'FadeDistance': FadeDistance,
    'GameType': GameType,
    'Condition': Condition,
    'PlayerTouch': PlayerTouch,
    'Toggle': Toggle,
    'tf_gamerules': tf_gamerules,
    'info_player_teamspawn': info_player_teamspawn,
    'game_forcerespawn': game_forcerespawn,
    'item_teamflag': item_teamflag,
    'point_intermission': point_intermission,
    'info_observer_point': info_observer_point,
    'game_round_win': game_round_win,
    'team_round_timer': team_round_timer,
    'Item': Item,
    'item_healthkit_full': item_healthkit_full,
    'item_healthkit_small': item_healthkit_small,
    'item_healthkit_medium': item_healthkit_medium,
    'item_ammopack_full': item_ammopack_full,
    'item_ammopack_small': item_ammopack_small,
    'item_ammopack_medium': item_ammopack_medium,
    'tf_spell_pickup': tf_spell_pickup,
    'item_bonuspack': item_bonuspack,
    'tf_halloween_pickup': tf_halloween_pickup,
    'info_powerup_spawn': info_powerup_spawn,
    'item_powerup_crit': item_powerup_crit,
    'item_powerup_uber': item_powerup_uber,
    'filter_activator_tfteam': filter_activator_tfteam,
    'filter_tf_player_can_cap': filter_tf_player_can_cap,
    'filter_tf_damaged_by_weapon_in_slot': filter_tf_damaged_by_weapon_in_slot,
    'filter_tf_condition': filter_tf_condition,
    'filter_tf_class': filter_tf_class,
    'func_capturezone': func_capturezone,
    'func_flagdetectionzone': func_flagdetectionzone,
    'func_nogrenades': func_nogrenades,
    'func_achievement': func_achievement,
    'func_nobuild': func_nobuild,
    'func_croc': func_croc,
    'func_suggested_build': func_suggested_build,
    'func_regenerate': func_regenerate,
    'func_powerupvolume': func_powerupvolume,
    'func_respawnflag': func_respawnflag,
    'func_respawnroom': func_respawnroom,
    'func_flag_alert': func_flag_alert,
    'func_respawnroomvisualizer': func_respawnroomvisualizer,
    'func_forcefield': func_forcefield,
    'func_changeclass': func_changeclass,
    'game_intro_viewpoint': game_intro_viewpoint,
    'func_proprrespawnzone': func_proprrespawnzone,
    'team_control_point': team_control_point,
    'team_control_point_round': team_control_point_round,
    'team_control_point_master': team_control_point_master,
    'trigger_capture_area': trigger_capture_area,
    'team_train_watcher': team_train_watcher,
    'game_text_tf': game_text_tf,
    'BaseObject': BaseObject,
    'obj_dispenser': obj_dispenser,
    'obj_sentrygun': obj_sentrygun,
    'obj_teleporter': obj_teleporter,
    'bot_hint_sentrygun': bot_hint_sentrygun,
    'bot_hint_teleporter_exit': bot_hint_teleporter_exit,
    'bot_hint_engineer_nest': bot_hint_engineer_nest,
    'bot_hint_sniper_spot': bot_hint_sniper_spot,
    'mapobj_cart_dispenser': mapobj_cart_dispenser,
    'dispenser_touch_trigger': dispenser_touch_trigger,
    'tf_logic_arena': tf_logic_arena,
    'tf_logic_competitive': tf_logic_competitive,
    'tf_logic_mannpower': tf_logic_mannpower,
    'bot_controller': bot_controller,
    'tf_logic_training_mode': tf_logic_training_mode,
    'boss_alpha': boss_alpha,
    'base_boss': base_boss,
    'tank_boss': tank_boss,
    'tf_logic_multiple_escort': tf_logic_multiple_escort,
    'tf_logic_koth': tf_logic_koth,
    'tf_robot_destruction_robot_spawn': tf_robot_destruction_robot_spawn,
    'tf_robot_destruction_spawn_group': tf_robot_destruction_spawn_group,
    'RobotDestruction': RobotDestruction,
    'tf_logic_robot_destruction': tf_logic_robot_destruction,
    'tf_logic_player_destruction': tf_logic_player_destruction,
    'trigger_rd_vault_trigger': trigger_rd_vault_trigger,
    'tf_logic_medieval': tf_logic_medieval,
    'tf_logic_cp_timer': tf_logic_cp_timer,
    'tf_logic_hybrid_ctf_cp': tf_logic_hybrid_ctf_cp,
    'tf_logic_raid': tf_logic_raid,
    'tf_logic_boss_battle': tf_logic_boss_battle,
    'tf_logic_mann_vs_machine': tf_logic_mann_vs_machine,
    'tf_logic_holiday': tf_logic_holiday,
    'func_upgradestation': func_upgradestation,
    'bot_generator': bot_generator,
    'bot_roster': bot_roster,
    'bot_action_point': bot_action_point,
    'bot_proxy': bot_proxy,
    'tf_spawner': tf_spawner,
    'tf_template_stun_drone': tf_template_stun_drone,
    'func_nav_blocker': func_nav_blocker,
    'func_nav_avoid': func_nav_avoid,
    'func_nav_prefer': func_nav_prefer,
    'func_nav_prerequisite': func_nav_prerequisite,
    'func_tfbot_hint': func_tfbot_hint,
    'trigger_stun': trigger_stun,
    'entity_spawn_point': entity_spawn_point,
    'entity_spawn_manager': entity_spawn_manager,
    'tf_pumpkin_bomb': tf_pumpkin_bomb,
    'tf_generic_bomb': tf_generic_bomb,
    'training_annotation': training_annotation,
    'training_prop_dynamic': training_prop_dynamic,
    'trigger_ignite_arrows': trigger_ignite_arrows,
    'wheel_of_doom': wheel_of_doom,
    'point_populator_interface': point_populator_interface,
    'tf_point_weapon_mimic': tf_point_weapon_mimic,
    'tf_point_nav_interface': tf_point_nav_interface,
    'trigger_timer_door': trigger_timer_door,
    'trigger_bot_tag': trigger_bot_tag,
    'filter_tf_bot_has_tag': filter_tf_bot_has_tag,
    'trigger_add_tf_player_condition': trigger_add_tf_player_condition,
    'trigger_remove_tf_player_condition': trigger_remove_tf_player_condition,
    'hightower_teleport_vortex': hightower_teleport_vortex,
    'tf_zombie_spawner': tf_zombie_spawner,
    'halloween_zapper': halloween_zapper,
    'trigger_player_respawn_override': trigger_player_respawn_override,
    'prop_soccer_ball': prop_soccer_ball,
    'MiniGame': MiniGame,
    'tf_logic_minigames': tf_logic_minigames,
    'tf_base_minigame': tf_base_minigame,
    'tf_halloween_minigame': tf_halloween_minigame,
    'tf_halloween_minigame_falling_platforms': tf_halloween_minigame_falling_platforms,
    'halloween_fortune_teller': halloween_fortune_teller,
    'tf_teleport_location': tf_teleport_location,
    'func_passtime_goal': func_passtime_goal,
    'info_passtime_ball_spawn': info_passtime_ball_spawn,
    'passtime_logic': passtime_logic,
    'func_passtime_goalie_zone': func_passtime_goalie_zone,
    'func_passtime_no_ball_zone': func_passtime_no_ball_zone,
    'trigger_passtime_ball': trigger_passtime_ball,
    'trigger_catapult': trigger_catapult,
    'trigger_ignite': trigger_ignite,
    'tf_halloween_gift_spawn_location': tf_halloween_gift_spawn_location,
    'tf_logic_on_holiday': tf_logic_on_holiday,
    'tf_glow': tf_glow,
}