
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


class Empty(Base):
    pass


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



class Breakable(Targetname, DamageFilter, Shadow):

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



class BaseNPC(Targetname, ResponseContext, Angles, DamageFilter, RenderFields, Shadow):

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



class BaseNPCAssault(BaseNPC):
    pass


class info_npc_spawn_destination(Angles, Targetname, Parentname):
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

    @property
    def ForceScheduleOnSpawn(self):
        return self._raw_data.get('forcescheduleonspawn', "")



class npc_template_maker(BaseNPCMaker):
    icon_sprite = "editor/npc_maker.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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



class TriggerOnce(EnableDisable, Global, Targetname, Parentname, Origin):

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
    def underwaterparticle(self):
        return self._raw_data.get('underwaterparticle', "underwater_default")

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



class worldspawn(Targetname, ResponseContext, worldbase):
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

    @property
    def m_bDontModifyPitchVolOnSpawn(self):
        return self._raw_data.get('m_bdontmodifypitchvolonspawn', "0")



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



class env_lightglow(Angles, Targetname, Parentname):
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

    @property
    def bUseNextGenBloom(self):
        return self._raw_data.get('busenextgenbloom', "0")

    @property
    def bUseCusBloomNG_Threshold(self):
        return self._raw_data.get('busecusbloomng_threshold', "0")

    @property
    def fCusBloomNG_Threshold(self):
        return parse_source_value(self._raw_data.get('fcusbloomng_threshold', 0))

    @property
    def bUseCusBloomNG_tintExponent(self):
        return self._raw_data.get('busecusbloomng_tintexponent', "0")

    @property
    def m_fCustomBloomNextGen_r(self):
        return parse_source_value(self._raw_data.get('m_fcustombloomnextgen_r', 1.0))

    @property
    def m_fCustomBloomNextGen_g(self):
        return parse_source_value(self._raw_data.get('m_fcustombloomnextgen_g', 1.0))

    @property
    def m_fCustomBloomNextGen_b(self):
        return parse_source_value(self._raw_data.get('m_fcustombloomnextgen_b', 1.0))

    @property
    def m_fCusBloomNG_exponent(self):
        return parse_source_value(self._raw_data.get('m_fcusbloomng_exponent', 1.0))



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



class func_ladderendpoint(Angles, Targetname, Parentname):
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



class func_wall(Shadow, Global, Targetname, RenderFields):

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)



class func_clip_vphysics(EnableDisable, Targetname):

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)



class func_brush(EnableDisable, Global, Targetname, Parentname, Origin, RenderFields, Shadow, Inputfilter):

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



class vgui_screen_base(Angles, Targetname, Parentname):

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


class vgui_slideshow_display(Angles, Targetname, Parentname):
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



class cycler(Targetname, Parentname, Angles, RenderFields):
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



class env_beam(RenderFxChoices, Targetname, Parentname):
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
    def dissolvetype(self):
        return self._raw_data.get('dissolvetype', "None")

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



class env_embers(Angles, Targetname, Parentname):

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



class env_fire(EnableDisable, Targetname, Parentname):
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



class env_fog_controller(Angles, Targetname, DXLevelChoice):
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



class env_steam(Angles, Targetname, Parentname):
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



class env_laser(RenderFxChoices, Targetname, Parentname):
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
    def decalname(self):
        return self._raw_data.get('decalname', "FadingScorch")

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

    @property
    def bloodcolor(self):
        return self._raw_data.get('bloodcolor', "-1")

    @property
    def touchkill(self):
        return self._raw_data.get('touchkill', "0")

    @property
    def gibdamage(self):
        return parse_source_value(self._raw_data.get('gibdamage', 0))

    @property
    def gibsound(self):
        return self._raw_data.get('gibsound', None)



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



class env_soundscape(EnableDisable, Targetname, Parentname):
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


class env_spark(Angles, Targetname, Parentname):
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



class env_sprite(Targetname, DXLevelChoice, Parentname, RenderFields):
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



class env_sprite_oriented(env_sprite, Angles):
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



class BaseSpeaker(Targetname, ResponseContext):

    @property
    def delaymin(self):
        return self._raw_data.get('delaymin', "15")

    @property
    def delaymax(self):
        return self._raw_data.get('delaymax', "135")

    @property
    def rulescript(self):
        return self._raw_data.get('rulescript', "scripts/talker/announcements.txt")

    @property
    def concept(self):
        return self._raw_data.get('concept', "announcement")



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



class point_spotlight(Targetname, Parentname, Angles, DXLevelChoice, RenderFields):
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


class info_target(Angles, Targetname, Parentname):
    icon_sprite = "editor/info_target.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class info_particle_system(Angles, Targetname, Parentname):
    model_ = "models/editor/cone_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def orientation_follows_viewer(self):
        return self._raw_data.get('orientation_follows_viewer', "0")

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



class phys_ragdollmagnet(EnableDisable, Targetname, Angles, Parentname):
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


class info_teleport_destination(Angles, Targetname, PlayerClass, Parentname):
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



class info_radial_link_controller(Targetname, Parentname):
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


class env_cascade_light(EnableDisable, Targetname):
    icon_sprite = "editor/shadow_control.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def angles(self):
        return self._raw_data.get('angles', "50 40 0")

    @property
    def color(self):
        return parse_int_vector(self._raw_data.get('color', "255 255 255 1"))

    @property
    def maxshadowdistance(self):
        return parse_source_value(self._raw_data.get('maxshadowdistance', 400))

    @property
    def uselightenvangles(self):
        return parse_source_value(self._raw_data.get('uselightenvangles', 1))

    @property
    def LightRadius1(self):
        return parse_source_value(self._raw_data.get('lightradius1', 0.001))

    @property
    def LightRadius2(self):
        return parse_source_value(self._raw_data.get('lightradius2', 0.001))

    @property
    def LightRadius3(self):
        return parse_source_value(self._raw_data.get('lightradius3', 0.001))

    @property
    def Depthbias1(self):
        return parse_source_value(self._raw_data.get('depthbias1', 0.00025))

    @property
    def Depthbias2(self):
        return parse_source_value(self._raw_data.get('depthbias2', 0.00005))

    @property
    def Depthbias3(self):
        return parse_source_value(self._raw_data.get('depthbias3', 0.00005))

    @property
    def Slopescaledepthbias1(self):
        return parse_source_value(self._raw_data.get('slopescaledepthbias1', 2.0))

    @property
    def Slopescaledepthbias2(self):
        return parse_source_value(self._raw_data.get('slopescaledepthbias2', 2.0))

    @property
    def Slopescaledepthbias3(self):
        return parse_source_value(self._raw_data.get('slopescaledepthbias3', 2.0))

    @property
    def ViewModelDepthbias(self):
        return parse_source_value(self._raw_data.get('viewmodeldepthbias', 0.000009))

    @property
    def ViewModelSlopescaledepthbias(self):
        return parse_source_value(self._raw_data.get('viewmodelslopescaledepthbias', 0.9))

    @property
    def CSMVolumeMode(self):
        return self._raw_data.get('csmvolumemode', "0")



class newLight_Dir(Angles, Targetname, Parentname):
    icon_sprite = "editor/light_new.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def LightEnvEnabled(self):
        return self._raw_data.get('lightenvenabled', "1")

    @property
    def PushbackDist(self):
        return parse_source_value(self._raw_data.get('pushbackdist', 9999999))

    @property
    def EnableGodRays(self):
        return self._raw_data.get('enablegodrays', "1")

    @property
    def Density(self):
        return parse_source_value(self._raw_data.get('density', 1.0))

    @property
    def Weight(self):
        return parse_source_value(self._raw_data.get('weight', 1.0))

    @property
    def Decay(self):
        return parse_source_value(self._raw_data.get('decay', 1.0))

    @property
    def Exposure(self):
        return parse_source_value(self._raw_data.get('exposure', 2.5))

    @property
    def DistFactor(self):
        return parse_source_value(self._raw_data.get('distfactor', 1.0))

    @property
    def DiskRadius(self):
        return parse_source_value(self._raw_data.get('diskradius', 0.02))

    @property
    def DiskInnerSizePercent(self):
        return parse_source_value(self._raw_data.get('diskinnersizepercent', 0.75))

    @property
    def ColorInner(self):
        return parse_int_vector(self._raw_data.get('colorinner', "128 200 255 255"))

    @property
    def ColorOuter(self):
        return parse_int_vector(self._raw_data.get('colorouter', "255 255 164 255"))

    @property
    def ColorRays(self):
        return parse_int_vector(self._raw_data.get('colorrays', "200 200 255 255"))

    @property
    def m_bUseToneMapRays(self):
        return self._raw_data.get('m_busetonemaprays', "1")



class newLight_Point(Targetname, Parentname):
    icon_sprite = "editor/light_new.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def Enabled(self):
        return self._raw_data.get('enabled', "1")

    @property
    def style(self):
        return self._raw_data.get('style', "0")

    @property
    def LightColorAmbient(self):
        return parse_int_vector(self._raw_data.get('lightcolorambient', "0 0 0 0"))

    @property
    def LightColor(self):
        return parse_int_vector(self._raw_data.get('lightcolor', "255 255 255 1"))

    @property
    def Intensity(self):
        return parse_source_value(self._raw_data.get('intensity', 8000))

    @property
    def SpecMultiplier(self):
        return parse_source_value(self._raw_data.get('specmultiplier', 1))

    @property
    def Range(self):
        return parse_source_value(self._raw_data.get('range', 1000))

    @property
    def LightType(self):
        return self._raw_data.get('lighttype', "0")

    @property
    def HasShadow(self):
        return self._raw_data.get('hasshadow', "0")

    @property
    def ShadowLod(self):
        return self._raw_data.get('shadowlod', "0")

    @property
    def ShadowFaceX(self):
        return self._raw_data.get('shadowfacex', "1")

    @property
    def ShadowFaceX_Minus(self):
        return self._raw_data.get('shadowfacex_minus', "1")

    @property
    def ShadowFaceY(self):
        return self._raw_data.get('shadowfacey', "1")

    @property
    def ShadowFaceY_Minus(self):
        return self._raw_data.get('shadowfacey_minus', "1")

    @property
    def ShadowFaceZ(self):
        return self._raw_data.get('shadowfacez', "1")

    @property
    def ShadowFaceZ_Minus(self):
        return self._raw_data.get('shadowfacez_minus', "1")

    @property
    def NearZ(self):
        return parse_source_value(self._raw_data.get('nearz', 2))

    @property
    def DepthBias(self):
        return parse_source_value(self._raw_data.get('depthbias', 0.0002))

    @property
    def SlopeDepthBias(self):
        return parse_source_value(self._raw_data.get('slopedepthbias', 0.2))

    @property
    def NormalBias(self):
        return parse_source_value(self._raw_data.get('normalbias', 1.0))

    @property
    def ShadowRadius(self):
        return parse_source_value(self._raw_data.get('shadowradius', -1.0))

    @property
    def bTexLight(self):
        return self._raw_data.get('btexlight', "0")

    @property
    def texName(self):
        return self._raw_data.get('texname', "")

    @property
    def bNegLight(self):
        return self._raw_data.get('bneglight', "0")

    @property
    def LightnGodRayMode(self):
        return self._raw_data.get('lightngodraymode', "0")

    @property
    def EnableGodRays(self):
        return self._raw_data.get('enablegodrays', "0")

    @property
    def Density(self):
        return parse_source_value(self._raw_data.get('density', 1.0))

    @property
    def Weight(self):
        return parse_source_value(self._raw_data.get('weight', 1.0))

    @property
    def Decay(self):
        return parse_source_value(self._raw_data.get('decay', 1.0))

    @property
    def Exposure(self):
        return parse_source_value(self._raw_data.get('exposure', 2.5))

    @property
    def DistFactor(self):
        return parse_source_value(self._raw_data.get('distfactor', 1.0))

    @property
    def DiskRadius(self):
        return parse_source_value(self._raw_data.get('diskradius', 0.02))

    @property
    def ColorInner(self):
        return parse_int_vector(self._raw_data.get('colorinner', "128 200 255 255"))

    @property
    def ColorRays(self):
        return parse_int_vector(self._raw_data.get('colorrays', "200 200 255 255"))

    @property
    def GodRaysType(self):
        return self._raw_data.get('godraystype', "0")

    @property
    def DiskInnerSizePercent(self):
        return parse_source_value(self._raw_data.get('diskinnersizepercent', 0.75))

    @property
    def ColorOuter(self):
        return parse_int_vector(self._raw_data.get('colorouter', "255 255 164 1"))

    @property
    def Ell_FR_ConstA(self):
        return parse_source_value(self._raw_data.get('ell_fr_consta', 0.9))

    @property
    def Ell_FR_ConstB(self):
        return parse_source_value(self._raw_data.get('ell_fr_constb', 0.1))

    @property
    def Ell_SR_ConstA(self):
        return parse_source_value(self._raw_data.get('ell_sr_consta', 0.9))

    @property
    def Ell_SR_ConstB(self):
        return parse_source_value(self._raw_data.get('ell_sr_constb', 0.1))

    @property
    def Ell_RRF_ConstA(self):
        return parse_source_value(self._raw_data.get('ell_rrf_consta', 0.9))

    @property
    def Ell_RRF_ConstB(self):
        return parse_source_value(self._raw_data.get('ell_rrf_constb', 0.1))

    @property
    def RotSpeed(self):
        return parse_source_value(self._raw_data.get('rotspeed', 3.14))

    @property
    def RotPatternFreq(self):
        return parse_source_value(self._raw_data.get('rotpatternfreq', 10.0))

    @property
    def m_bEnableWorldSpace(self):
        return self._raw_data.get('m_benableworldspace', "0")

    @property
    def m_fAlphaDiskInner(self):
        return parse_source_value(self._raw_data.get('m_falphadiskinner', 1))

    @property
    def m_fAlphaDiskOuter(self):
        return parse_source_value(self._raw_data.get('m_falphadiskouter', 1))

    @property
    def m_bUseToneMapRays(self):
        return self._raw_data.get('m_busetonemaprays', "1")

    @property
    def m_bUseToneMapDisk(self):
        return self._raw_data.get('m_busetonemapdisk', "1")

    @property
    def m_bSRO_Brush(self):
        return self._raw_data.get('m_bsro_brush', "1")

    @property
    def m_bSRO_StaticProp(self):
        return self._raw_data.get('m_bsro_staticprop', "1")

    @property
    def m_bSRO_DynProps(self):
        return self._raw_data.get('m_bsro_dynprops', "1")

    @property
    def m_bSRO_Trans(self):
        return self._raw_data.get('m_bsro_trans', "1")



class newLight_Spot(Angles, Targetname, Parentname):
    icon_sprite = "editor/light_new.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def ReverseDir(self):
        return self._raw_data.get('reversedir', "0")

    @property
    def Enabled(self):
        return self._raw_data.get('enabled', "1")

    @property
    def style(self):
        return self._raw_data.get('style', "0")

    @property
    def LightColorAmbient(self):
        return parse_int_vector(self._raw_data.get('lightcolorambient', "0 0 0 0"))

    @property
    def LightColor(self):
        return parse_int_vector(self._raw_data.get('lightcolor', "255 255 255 1"))

    @property
    def Intensity(self):
        return parse_source_value(self._raw_data.get('intensity', 8000))

    @property
    def SpecMultiplier(self):
        return parse_source_value(self._raw_data.get('specmultiplier', 1))

    @property
    def Range(self):
        return parse_source_value(self._raw_data.get('range', 1000))

    @property
    def phi(self):
        return parse_source_value(self._raw_data.get('phi', 60))

    @property
    def theta(self):
        return parse_source_value(self._raw_data.get('theta', 30))

    @property
    def angularFallOff(self):
        return parse_source_value(self._raw_data.get('angularfalloff', 2))

    @property
    def LightType(self):
        return self._raw_data.get('lighttype', "0")

    @property
    def HasShadow(self):
        return self._raw_data.get('hasshadow', "0")

    @property
    def ShadowLod(self):
        return self._raw_data.get('shadowlod', "0")

    @property
    def NearZ(self):
        return parse_source_value(self._raw_data.get('nearz', 2))

    @property
    def DepthBias(self):
        return parse_source_value(self._raw_data.get('depthbias', 0.0002))

    @property
    def SlopeDepthBias(self):
        return parse_source_value(self._raw_data.get('slopedepthbias', 0.2))

    @property
    def NormalBias(self):
        return parse_source_value(self._raw_data.get('normalbias', 1.0))

    @property
    def ShadowFOV(self):
        return parse_source_value(self._raw_data.get('shadowfov', 0))

    @property
    def ShadowRadius(self):
        return parse_source_value(self._raw_data.get('shadowradius', -1.0))

    @property
    def bTexLight(self):
        return self._raw_data.get('btexlight', "0")

    @property
    def texName(self):
        return self._raw_data.get('texname', "")

    @property
    def TexCookieFramesX(self):
        return parse_source_value(self._raw_data.get('texcookieframesx', 1))

    @property
    def TexCookieFramesY(self):
        return parse_source_value(self._raw_data.get('texcookieframesy', 1))

    @property
    def TexCookieFps(self):
        return parse_source_value(self._raw_data.get('texcookiefps', 0))

    @property
    def bTexCookieScrollMode(self):
        return self._raw_data.get('btexcookiescrollmode', "0")

    @property
    def fScrollSpeedU(self):
        return parse_source_value(self._raw_data.get('fscrollspeedu', 0))

    @property
    def fScrollSpeedV(self):
        return parse_source_value(self._raw_data.get('fscrollspeedv', 0))

    @property
    def bNegLight(self):
        return self._raw_data.get('bneglight', "0")

    @property
    def m_bSRO_Brush(self):
        return self._raw_data.get('m_bsro_brush', "1")

    @property
    def m_bSRO_StaticProp(self):
        return self._raw_data.get('m_bsro_staticprop', "1")

    @property
    def m_bSRO_DynProps(self):
        return self._raw_data.get('m_bsro_dynprops', "1")

    @property
    def m_bSRO_Trans(self):
        return self._raw_data.get('m_bsro_trans', "1")



class godrays_settings(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def TargetGodRays(self):
        return self._raw_data.get('targetgodrays', "")

    @property
    def TransitionTime(self):
        return parse_source_value(self._raw_data.get('transitiontime', 0))

    @property
    def LightType(self):
        return self._raw_data.get('lighttype', "1")

    @property
    def EnableGodRays(self):
        return parse_source_value(self._raw_data.get('enablegodrays', 1))

    @property
    def Density(self):
        return parse_source_value(self._raw_data.get('density', 1.0))

    @property
    def Weight(self):
        return parse_source_value(self._raw_data.get('weight', 1.0))

    @property
    def Decay(self):
        return parse_source_value(self._raw_data.get('decay', 1.0))

    @property
    def Exposure(self):
        return parse_source_value(self._raw_data.get('exposure', 2.5))

    @property
    def DistFactor(self):
        return parse_source_value(self._raw_data.get('distfactor', 1.0))

    @property
    def DiskRadius(self):
        return parse_source_value(self._raw_data.get('diskradius', 0.02))

    @property
    def DiskInnerSizePercent(self):
        return parse_source_value(self._raw_data.get('diskinnersizepercent', 0.75))

    @property
    def ColorInner(self):
        return parse_int_vector(self._raw_data.get('colorinner', "128 200 255 1"))

    @property
    def ColorOuter(self):
        return parse_int_vector(self._raw_data.get('colorouter', "255 255 164 1"))

    @property
    def ColorRays(self):
        return parse_int_vector(self._raw_data.get('colorrays', "200 200 255 1"))



class newLights_settings(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def TargetNewLightEntity(self):
        return self._raw_data.get('targetnewlightentity', "")

    @property
    def TransitionTime(self):
        return parse_source_value(self._raw_data.get('transitiontime', 0))

    @property
    def LightType(self):
        return self._raw_data.get('lighttype', "0")

    @property
    def Enabled(self):
        return parse_source_value(self._raw_data.get('enabled', 1))

    @property
    def LightColorAmbient(self):
        return parse_int_vector(self._raw_data.get('lightcolorambient', "0 0 0 0"))

    @property
    def LightColor(self):
        return parse_int_vector(self._raw_data.get('lightcolor', "255 255 255 1"))

    @property
    def style(self):
        return self._raw_data.get('style', "0")

    @property
    def Intensity(self):
        return parse_source_value(self._raw_data.get('intensity', 8000))

    @property
    def SpecMultiplier(self):
        return parse_source_value(self._raw_data.get('specmultiplier', 1))

    @property
    def Range(self):
        return parse_source_value(self._raw_data.get('range', 1000))

    @property
    def falloffQuadratic(self):
        return parse_source_value(self._raw_data.get('falloffquadratic', 0))

    @property
    def falloffLinear(self):
        return parse_source_value(self._raw_data.get('fallofflinear', 0))

    @property
    def falloffConstant(self):
        return parse_source_value(self._raw_data.get('falloffconstant', 1))

    @property
    def phi(self):
        return parse_source_value(self._raw_data.get('phi', 60))

    @property
    def theta(self):
        return parse_source_value(self._raw_data.get('theta', 30))

    @property
    def angularFallOff(self):
        return parse_source_value(self._raw_data.get('angularfalloff', 2))



class newlights_gbuffersettings(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def Enable4WaysFastPath(self):
        return self._raw_data.get('enable4waysfastpath', "1")

    @property
    def DisableGbufferOnSecondaryCams(self):
        return self._raw_data.get('disablegbufferonsecondarycams', "0")

    @property
    def DisableGbufferOnRefractions(self):
        return self._raw_data.get('disablegbufferonrefractions', "0")

    @property
    def DisableGbufferOnReflections(self):
        return self._raw_data.get('disablegbufferonreflections', "0")



class newLights_Spawner(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def LightType(self):
        return self._raw_data.get('lighttype', "0")

    @property
    def NumLights(self):
        return parse_source_value(self._raw_data.get('numlights', 100))

    @property
    def NumLightsInRow(self):
        return parse_source_value(self._raw_data.get('numlightsinrow', 10))

    @property
    def LightRange(self):
        return parse_source_value(self._raw_data.get('lightrange', 250))

    @property
    def LightIntensity(self):
        return parse_source_value(self._raw_data.get('lightintensity', 4000))

    @property
    def RowSpaceing(self):
        return parse_source_value(self._raw_data.get('rowspaceing', 100))

    @property
    def ColSpacing(self):
        return parse_source_value(self._raw_data.get('colspacing', 100))

    @property
    def RandomColor(self):
        return self._raw_data.get('randomcolor', "1")

    @property
    def LightColor(self):
        return parse_int_vector(self._raw_data.get('lightcolor', "255 255 255 1"))

    @property
    def SpawnDir_Forward(self):
        return parse_float_vector(self._raw_data.get('spawndir_forward', "1.0 0.0 0.0"))

    @property
    def SpawnDir_Right(self):
        return parse_float_vector(self._raw_data.get('spawndir_right', "0.0 1.0 0.0"))



class newxog_global(Targetname, Parentname):
    icon_sprite = "editor/xog_global.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def Enabled(self):
        return self._raw_data.get('enabled', "1")

    @property
    def XogType(self):
        return self._raw_data.get('xogtype', "0")

    @property
    def skyColor(self):
        return parse_int_vector(self._raw_data.get('skycolor', "0 0 255 255"))

    @property
    def skyBlendFactor(self):
        return parse_source_value(self._raw_data.get('skyblendfactor', 0.4))

    @property
    def colorMode(self):
        return self._raw_data.get('colormode', "0")

    @property
    def texName(self):
        return self._raw_data.get('texname', "")

    @property
    def colorTop(self):
        return parse_int_vector(self._raw_data.get('colortop', "255 0 0 255"))

    @property
    def colorBottom(self):
        return parse_int_vector(self._raw_data.get('colorbottom', "0 255 0 255"))

    @property
    def distStart(self):
        return parse_source_value(self._raw_data.get('diststart', 50))

    @property
    def distEnd(self):
        return parse_source_value(self._raw_data.get('distend', 2000))

    @property
    def distDensity(self):
        return parse_source_value(self._raw_data.get('distdensity', 1.0))

    @property
    def opacityOffsetTop(self):
        return parse_source_value(self._raw_data.get('opacityoffsettop', 0))

    @property
    def opacityOffsetBottom(self):
        return parse_source_value(self._raw_data.get('opacityoffsetbottom', 0))

    @property
    def htZStart(self):
        return parse_source_value(self._raw_data.get('htzstart', 0))

    @property
    def htZEnd(self):
        return parse_source_value(self._raw_data.get('htzend', 400))

    @property
    def htZColStart(self):
        return parse_source_value(self._raw_data.get('htzcolstart', 0))

    @property
    def htZColEnd(self):
        return parse_source_value(self._raw_data.get('htzcolend', 400))

    @property
    def noise1ScrollSpeed(self):
        return parse_float_vector(self._raw_data.get('noise1scrollspeed', "0.007 0.006, -0.01 0.0"))

    @property
    def noise1Tiling(self):
        return parse_float_vector(self._raw_data.get('noise1tiling', "0.34 0.34 0.34 0.0"))

    @property
    def noise2ScrollSpeed(self):
        return parse_float_vector(self._raw_data.get('noise2scrollspeed', "0.0035, 0.003, -0.005 0.0"))

    @property
    def noise2Tiling(self):
        return parse_float_vector(self._raw_data.get('noise2tiling', "0.24 0.24 0.24 0.0"))

    @property
    def noiseContrast(self):
        return parse_source_value(self._raw_data.get('noisecontrast', 1.0))

    @property
    def noiseMultiplier(self):
        return parse_source_value(self._raw_data.get('noisemultiplier', 1.0))

    @property
    def RadiusX(self):
        return parse_source_value(self._raw_data.get('radiusx', 0))

    @property
    def RadiusY(self):
        return parse_source_value(self._raw_data.get('radiusy', 0))

    @property
    def RadiusZ(self):
        return parse_source_value(self._raw_data.get('radiusz', 0))



class newxog_settings(Targetname, Parentname):
    icon_sprite = "editor/xog_settings.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def TargetNewLightEntity(self):
        return self._raw_data.get('targetnewlightentity', "")

    @property
    def TransitionTime(self):
        return parse_source_value(self._raw_data.get('transitiontime', 0))

    @property
    def Enabled(self):
        return parse_source_value(self._raw_data.get('enabled', 1))

    @property
    def skyColorTop(self):
        return parse_int_vector(self._raw_data.get('skycolortop', "112 104 255 255"))

    @property
    def skyBlendType(self):
        return parse_source_value(self._raw_data.get('skyblendtype', 0))

    @property
    def skyBlendFactor(self):
        return parse_source_value(self._raw_data.get('skyblendfactor', 0.25))

    @property
    def colorTop(self):
        return parse_int_vector(self._raw_data.get('colortop', "61 255 235 255"))

    @property
    def colorBottom(self):
        return parse_int_vector(self._raw_data.get('colorbottom', "255 62 235 255"))

    @property
    def distStart(self):
        return parse_source_value(self._raw_data.get('diststart', 50))

    @property
    def distEnd(self):
        return parse_source_value(self._raw_data.get('distend', 2000))

    @property
    def distDensity(self):
        return parse_source_value(self._raw_data.get('distdensity', 1.0))

    @property
    def opacityOffsetTop(self):
        return parse_source_value(self._raw_data.get('opacityoffsettop', 0))

    @property
    def opacityOffsetBottom(self):
        return parse_source_value(self._raw_data.get('opacityoffsetbottom', 0))

    @property
    def htZStart(self):
        return parse_source_value(self._raw_data.get('htzstart', 0))

    @property
    def htZEnd(self):
        return parse_source_value(self._raw_data.get('htzend', 2000))

    @property
    def htZColStart(self):
        return parse_source_value(self._raw_data.get('htzcolstart', 0))

    @property
    def htZColEnd(self):
        return parse_source_value(self._raw_data.get('htzcolend', 400))

    @property
    def noise1ScrollSpeed(self):
        return parse_float_vector(self._raw_data.get('noise1scrollspeed', "0.007 0.006 -0.01 0.0"))

    @property
    def noise1Tiling(self):
        return parse_float_vector(self._raw_data.get('noise1tiling', "0.34 0.34 0.34 0.0"))

    @property
    def noise2ScrollSpeed(self):
        return parse_float_vector(self._raw_data.get('noise2scrollspeed', "0.0035 0.003 -0.005 0.0"))

    @property
    def noise2Tiling(self):
        return parse_float_vector(self._raw_data.get('noise2tiling', "0.24 0.24 0.24 0.0"))

    @property
    def noiseContrast(self):
        return parse_source_value(self._raw_data.get('noisecontrast', 1.0))

    @property
    def noiseMultiplier(self):
        return parse_source_value(self._raw_data.get('noisemultiplier', 1.0))



class newxog_volume(TriggerOnce):

    @property
    def Enabled(self):
        return self._raw_data.get('enabled', "1")

    @property
    def XogType(self):
        return self._raw_data.get('xogtype', "0")

    @property
    def colorMode(self):
        return self._raw_data.get('colormode', "0")

    @property
    def texName(self):
        return self._raw_data.get('texname', "")

    @property
    def colorTop(self):
        return parse_int_vector(self._raw_data.get('colortop', "255 0 0 255"))

    @property
    def colorBottom(self):
        return parse_int_vector(self._raw_data.get('colorbottom', "0 255 0 255"))

    @property
    def distStart(self):
        return parse_source_value(self._raw_data.get('diststart', 50))

    @property
    def distEnd(self):
        return parse_source_value(self._raw_data.get('distend', 2000))

    @property
    def distDensity(self):
        return parse_source_value(self._raw_data.get('distdensity', 1.0))

    @property
    def opacityOffsetTop(self):
        return parse_source_value(self._raw_data.get('opacityoffsettop', 0))

    @property
    def opacityOffsetBottom(self):
        return parse_source_value(self._raw_data.get('opacityoffsetbottom', 0))

    @property
    def htZStart(self):
        return parse_source_value(self._raw_data.get('htzstart', 0))

    @property
    def htZEnd(self):
        return parse_source_value(self._raw_data.get('htzend', 400))

    @property
    def htZColStart(self):
        return parse_source_value(self._raw_data.get('htzcolstart', 0))

    @property
    def htZColEnd(self):
        return parse_source_value(self._raw_data.get('htzcolend', 400))

    @property
    def noise1ScrollSpeed(self):
        return parse_float_vector(self._raw_data.get('noise1scrollspeed', "0.01095 0.00855 -0.02265 0.0"))

    @property
    def noise1Tiling(self):
        return parse_float_vector(self._raw_data.get('noise1tiling', "1.32 1.32 1.32 0.0"))

    @property
    def noise2ScrollSpeed(self):
        return parse_float_vector(self._raw_data.get('noise2scrollspeed', "0.00525 0.00495 -0.0075 0.0"))

    @property
    def noise2Tiling(self):
        return parse_float_vector(self._raw_data.get('noise2tiling', "0.96 0.96 0.96 0.0"))

    @property
    def noiseContrast(self):
        return parse_source_value(self._raw_data.get('noisecontrast', 1.0))

    @property
    def noiseMultiplier(self):
        return parse_source_value(self._raw_data.get('noisemultiplier', 1.0))

    @property
    def EnableVol_Height(self):
        return self._raw_data.get('enablevol_height', "0")



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



class light_dynamic(Angles, Targetname, Parentname):
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
        return self._raw_data.get('disableallshadows', "1")

    @property
    def ForceBlobShadows(self):
        return self._raw_data.get('forceblobshadows', "1")



class color_correction(EnableDisable, Targetname):
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

    @property
    def exclusive(self):
        return self._raw_data.get('exclusive', "0")



class fog_volume(EnableDisable, Targetname):

    @property
    def FogName(self):
        return self._raw_data.get('fogname', None)

    @property
    def ColorCorrectionName(self):
        return self._raw_data.get('colorcorrectionname', None)



class color_correction_volume(EnableDisable, Targetname):

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



class func_movelinear(Origin, Targetname, Parentname, RenderFields):

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



class func_water_analog(Origin, Targetname, Parentname):

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
    def WaterMaterial(self):
        return self._raw_data.get('watermaterial', "liquids/c4a1_water_green")

    @property
    def WaveHeight(self):
        return self._raw_data.get('waveheight', "3.0")



class func_rotating(Targetname, Parentname, Origin, Angles, RenderFields, Shadow):

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



class func_platrot(Targetname, Parentname, Origin, Angles, BasePlat, RenderFields, Shadow):

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



class keyframe_track(Angles, KeyFrame, Targetname, Parentname):
    pass


class move_keyframed(KeyFrame, Targetname, Mover, Parentname):
    pass


class move_track(Mover, KeyFrame, Targetname, Parentname):

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



class keyframe_rope(KeyFrame, Targetname, RopeKeyFrame, Parentname):
    model_ = "models/editor/axis_helper_thick.mdl"
    pass


class move_rope(KeyFrame, Targetname, RopeKeyFrame, Parentname):
    model_ = "models/editor/axis_helper.mdl"

    @property
    def PositionInterpolator(self):
        return self._raw_data.get('positioninterpolator', "2")



class Button(Base):
    pass


class func_button(Targetname, Parentname, Button, Origin, DamageFilter, RenderFields):

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



class func_detail_blocker(Empty):
    pass


class func_rot_button(EnableDisable, Global, Targetname, Parentname, Button, Origin, Angles):

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



class momentary_rot_button(Targetname, Parentname, Origin, Angles, RenderFields):

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



class Door(Global, Targetname, Parentname, Shadow, RenderFields):

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



class func_door(Origin, Door):

    @property
    def movedir(self):
        return parse_float_vector(self._raw_data.get('movedir', "0 0 0"))

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)



class func_door_rotating(Angles, Origin, Door):

    @property
    def distance(self):
        return parse_source_value(self._raw_data.get('distance', 90))

    @property
    def solidbsp(self):
        return self._raw_data.get('solidbsp', "0")



class prop_door_rotating(Global, Targetname, Parentname, Angles, Studiomodel):
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



class func_dustmotes(BModelParticleSpawner, Targetname):

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



class func_dustcloud(BModelParticleSpawner, Targetname):

    @property
    def Alpha(self):
        return parse_source_value(self._raw_data.get('alpha', 30))

    @property
    def SizeMin(self):
        return self._raw_data.get('sizemin', "100")

    @property
    def SizeMax(self):
        return self._raw_data.get('sizemax', "200")



class env_dustpuff(Angles, Targetname, Parentname):
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



class env_particlescript(Angles, Targetname, Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/Ambient_citadel_paths.mdl")



class env_effectscript(Angles, Targetname, Parentname):
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



class point_viewcontrol(Angles, Targetname, Parentname):
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



class logic_relay(EnableDisable, Targetname):
    icon_sprite = "editor/logic_relay.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class logic_timer(EnableDisable, Targetname):
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



class env_microphone(EnableDisable, Targetname, Parentname):
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



class math_remap(EnableDisable, Targetname):
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



class math_counter(EnableDisable, Targetname):
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



class env_entity_maker(Angles, Targetname, Parentname):
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



class filter_activator_model(BaseFilter):
    icon_sprite = "editor/filter_name.vmt"

    @property
    def model(self):
        return self._raw_data.get('model', None)



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



class point_anglesensor(EnableDisable, Targetname, Parentname):
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



class point_proximity_sensor(EnableDisable, Targetname, Angles, Parentname):
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



class phys_magnet(Angles, Targetname, Studiomodel, Parentname):
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



class prop_static_base(Angles, DXLevelChoice):

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



class prop_dynamic_base(Global, Parentname, BreakableProp, Angles, BaseFadeProp, DXLevelChoice, RenderFields, Studiomodel):

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


class prop_dynamic(EnableDisable, prop_dynamic_base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def IgnoreNPCCollisions(self):
        return self._raw_data.get('ignorenpccollisions', "0")



class prop_dynamic_playertouch(EnableDisable, prop_dynamic_base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def IgnoreNPCCollisions(self):
        return self._raw_data.get('ignorenpccollisions', "0")

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 0))

    @property
    def m_szParticlesOnBreak(self):
        return self._raw_data.get('m_szparticlesonbreak', "")

    @property
    def m_szSoundOnBreak(self):
        return self._raw_data.get('m_szsoundonbreak', "")

    @property
    def m_FDamageToPlayerOnTouch(self):
        return parse_source_value(self._raw_data.get('m_fdamagetoplayerontouch', 20))

    @property
    def fGluonDmgMultiplier(self):
        return parse_source_value(self._raw_data.get('fgluondmgmultiplier', 1.0))



class prop_dynamic_override(prop_dynamic_base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 0))



class BasePropPhysics(Global, BreakableProp, Angles, BaseFadeProp, DXLevelChoice, Studiomodel):

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

    @property
    def physicsdamagescale(self):
        return parse_source_value(self._raw_data.get('physicsdamagescale', 1.0))

    @property
    def physicsdamagelimit(self):
        return parse_source_value(self._raw_data.get('physicsdamagelimit', 0))



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


class prop_physics_teleprop(BasePropPhysics, RenderFields):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def m_bHammerEntity(self):
        return self._raw_data.get('m_bhammerentity', "1")

    @property
    def m_szOwnnerPortalName(self):
        return self._raw_data.get('m_szownnerportalname', "")



class prop_physics_multiplayer(prop_physics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def physicsmode(self):
        return self._raw_data.get('physicsmode', "0")



class prop_ragdoll(EnableDisable, Targetname, Angles, BaseFadeProp, DXLevelChoice, Studiomodel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def MimicName(self):
        return self._raw_data.get('mimicname', "")

    @property
    def angleOverride(self):
        return self._raw_data.get('angleoverride', "")

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 100))



class prop_ragdoll_original(EnableDisable, Targetname, Angles, BaseFadeProp, DXLevelChoice, Studiomodel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def MimicName(self):
        return self._raw_data.get('mimicname', "")

    @property
    def angleOverride(self):
        return self._raw_data.get('angleoverride', "")

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 100))



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



class func_conveyor(Shadow, Targetname, Parentname, RenderFields):

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


class func_illusionary(Targetname, Parentname, Origin, Shadow, RenderFields):

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


class func_guntarget(Global, Targetname, Parentname, RenderFields):

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



class Trackchange(Global, Targetname, Parentname, PlatSounds, RenderFields):

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



class BaseTrain(Global, Targetname, Parentname, Origin, RenderFields, Shadow):

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



class path_track(Angles, Targetname, Parentname):
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



class trigger_csm_volume(Trigger):

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

    @property
    def NotLookingFrequency(self):
        return parse_source_value(self._raw_data.get('notlookingfrequency', 0.5))



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



class trigger_impact(Angles, Origin, Targetname):

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


class ai_speechfilter(EnableDisable, Targetname, ResponseContext):
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
    icon_sprite = "editor/playerproxy.vmt"
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



class func_reflective_glass(func_brush):
    pass


class env_particle_performance_monitor(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class npc_puppet(Studiomodel, BaseNPC, Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def animationtarget(self):
        return self._raw_data.get('animationtarget', "")

    @property
    def attachmentname(self):
        return self._raw_data.get('attachmentname', "")



class point_gamestats_counter(EnableDisable, Origin, Targetname):
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
        return self._raw_data.get('fixup_style', "2")

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



class func_instance_io_proxy(Base):
    icon_sprite = "editor/func_instance_parms.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def targetname(self):
        return self._raw_data.get('targetname', None)



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



class npc_bullseye(BaseNPC, Parentname):
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



class npc_enemyfinder(BaseNPC, Parentname):

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



class env_gunfire(Targetname, Parentname, EnableDisable):
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



class generic_actor(BaseNPC, Parentname):
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



class player_control(Targetname):
    pass


class BaseScripted(Targetname, Parentname, Angles):

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
    def m_bNoBlendedMovement(self):
        return self._raw_data.get('m_bnoblendedmovement', "0")

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



class scripted_target(Targetname, Parentname):
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


class env_detail_controller(Angles):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def fademindist(self):
        return parse_source_value(self._raw_data.get('fademindist', 400))

    @property
    def fademaxdist(self):
        return parse_source_value(self._raw_data.get('fademaxdist', 1200))



class env_global(EnvGlobal):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def globalstate(self):
        return self._raw_data.get('globalstate', None)



class BaseCharger(Angles, Targetname, BaseFadeProp):
    pass


class item_healthcharger(BaseCharger):
    model_ = "models/props_blackmesa/health_charger.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def charge(self):
        return parse_source_value(self._raw_data.get('charge', 50))

    @property
    def skintype(self):
        return self._raw_data.get('skintype', "0")



class item_suitcharger(BaseCharger):
    model_ = "models/props_blackmesa/hev_charger.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def charge(self):
        return parse_source_value(self._raw_data.get('charge', 75))

    @property
    def skintype(self):
        return self._raw_data.get('skintype', "0")



class BasePickup(Angles, Targetname, BaseFadeProp, Shadow):

    @property
    def respawntime(self):
        return parse_source_value(self._raw_data.get('respawntime', 15))



class item_weapon_357(BasePickup):
    model_ = "models/weapons/w_357.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_weapon_crowbar(BasePickup):
    model_ = "models/weapons/w_crowbar.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_weapon_crossbow(BasePickup):
    model_ = "models/weapons/w_crossbow.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_weapon_frag(BasePickup):
    model_ = "models/weapons/w_grenade.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_weapon_glock(BasePickup):
    model_ = "models/weapons/w_glock.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_weapon_gluon(BasePickup):
    model_ = "models/weapons/w_egon_pickup.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_weapon_hivehand(BasePickup):
    model_ = "models/weapons/w_hgun.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_weapon_mp5(BasePickup):
    model_ = "models/weapons/w_mp5.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_weapon_shotgun(BasePickup):
    model_ = "models/weapons/w_shotgun.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_weapon_rpg(BasePickup):
    model_ = "models/weapons/w_rpg.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_weapon_satchel(BasePickup):
    model_ = "models/weapons/w_satchel.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_weapon_snark(BasePickup):
    model_ = "models/xenians/snarknest.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_weapon_tau(BasePickup):
    model_ = "models/weapons/w_gauss.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_weapon_tripmine(BasePickup):
    model_ = "models/weapons/w_tripmine.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_ammo_357(BasePickup):
    model_ = "models/weapons/w_357ammobox.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_ammo_crossbow(BasePickup):
    model_ = "models/weapons/w_crossbow_clip.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_ammo_glock(BasePickup):
    model_ = "models/weapons/w_9mmclip.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_ammo_energy(BasePickup):
    model_ = "models/weapons/w_gaussammo.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_ammo_mp5(BasePickup):
    model_ = "models/weapons/w_9mmARclip.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_ammo_shotgun(BasePickup):
    model_ = "models/weapons/w_shotbox.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_grenade_mp5(BasePickup):
    model_ = "models/weapons/w_argrenade.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_grenade_rpg(BasePickup):
    model_ = "models/weapons/w_rpgammo.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_ammo_canister(BasePickup):
    model_ = "models/weapons/w_weaponbox.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def AmmoGlock(self):
        return parse_source_value(self._raw_data.get('ammoglock', 0))

    @property
    def AmmoMp5(self):
        return parse_source_value(self._raw_data.get('ammomp5', 0))

    @property
    def Ammo357(self):
        return parse_source_value(self._raw_data.get('ammo357', 0))

    @property
    def AmmoBolt(self):
        return parse_source_value(self._raw_data.get('ammobolt', 0))

    @property
    def AmmoBuckshot(self):
        return parse_source_value(self._raw_data.get('ammobuckshot', 0))

    @property
    def AmmoEnergy(self):
        return parse_source_value(self._raw_data.get('ammoenergy', 0))

    @property
    def AmmoMp5Grenade(self):
        return parse_source_value(self._raw_data.get('ammomp5grenade', 0))

    @property
    def AmmoRPG(self):
        return parse_source_value(self._raw_data.get('ammorpg', 0))

    @property
    def AmmoSatchel(self):
        return parse_source_value(self._raw_data.get('ammosatchel', 0))

    @property
    def AmmoSnark(self):
        return parse_source_value(self._raw_data.get('ammosnark', 0))

    @property
    def AmmoTripmine(self):
        return parse_source_value(self._raw_data.get('ammotripmine', 0))

    @property
    def AmmoFrag(self):
        return parse_source_value(self._raw_data.get('ammofrag', 0))



class item_ammo_crate(Angles, Targetname, BaseFadeProp):
    model_ = "models/items/ammocrate_rockets.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/items/ammocrate_rockets.mdl")

    @property
    def AmmoType(self):
        return self._raw_data.get('ammotype', "grenade_rpg")

    @property
    def isDynamicMoving(self):
        return parse_source_value(self._raw_data.get('isdynamicmoving', 0))

    @property
    def AmmoCount(self):
        return parse_source_value(self._raw_data.get('ammocount', 1))



class item_suit(BasePickup):
    model_ = "models/props_am/hev_suit.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_battery(BasePickup):
    model_ = "models/weapons/w_battery.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_healthkit(BasePickup):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/weapons/w_medkit.mdl")



class item_longjump(BasePickup):
    model_ = "models/weapons/w_longjump.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class BaseGrenade(Angles, Targetname, Shadow):
    pass


class grenade_satchel(BaseGrenade):
    model_ = "models/weapons/w_satchel.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class grenade_tripmine(BaseGrenade):
    model_ = "models/weapons/w_tripmine.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class BaseSentry(BaseNPC):
    pass


class npc_plantlight(BaseNPC):
    model_ = "models/props_xen/xen_protractinglight.mdl"

    @property
    def planttype(self):
        return self._raw_data.get('planttype', "0")

    @property
    def ICanTakeDamage(self):
        return self._raw_data.get('icantakedamage', "1")

    @property
    def LightColor(self):
        return parse_int_vector(self._raw_data.get('lightcolor', "255 223 43 255"))

    @property
    def Intensity(self):
        return parse_source_value(self._raw_data.get('intensity', 1000))

    @property
    def Range(self):
        return parse_source_value(self._raw_data.get('range', 500))



class npc_plantlight_stalker(Base):
    model_ = "models/props_xen/xen_plantlightstalker.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def planttype(self):
        return self._raw_data.get('planttype', "1")

    @property
    def ICanTakeDamage(self):
        return self._raw_data.get('icantakedamage', "0")

    @property
    def LightColor(self):
        return parse_int_vector(self._raw_data.get('lightcolor', "255 223 43 255"))

    @property
    def Intensity(self):
        return parse_source_value(self._raw_data.get('intensity', 1000))

    @property
    def Range(self):
        return parse_source_value(self._raw_data.get('range', 500))



class npc_puffballfungus(BaseNPC):
    model_ = "models/props_xen/xen_puffballfungus.mdl"

    @property
    def RangeOuter(self):
        return parse_source_value(self._raw_data.get('rangeouter', 500))

    @property
    def RangeInner(self):
        return parse_source_value(self._raw_data.get('rangeinner', 250))



class npc_xentree(BaseNPC):
    model_ = "models/props_xen/foliage/hacker_tree.mdl"
    pass


class npc_protozoan(BaseNPC):
    model_ = "models/xenians/protozoan.mdl"
    pass


class npc_sentry_ceiling(BaseSentry):
    model_ = "models/npcs/sentry_ceiling.mdl"
    pass


class npc_sentry_ground(BaseSentry):
    model_ = "models/npcs/sentry_ground.mdl"
    pass


class npc_xenturret(BaseNPC):
    model_ = "models/props_xen/xen_turret.mdl"

    @property
    def RangeOuter(self):
        return parse_source_value(self._raw_data.get('rangeouter', 999999))

    @property
    def RangeInner(self):
        return parse_source_value(self._raw_data.get('rangeinner', 99999))

    @property
    def m_Color(self):
        return parse_int_vector(self._raw_data.get('m_color', "255 255 255"))

    @property
    def nShield(self):
        return parse_source_value(self._raw_data.get('nshield', 0))



class npc_alien_slave_dummy(BaseNPC):
    model_ = "models/vortigaunt_slave.mdl"

    @property
    def passive(self):
        return self._raw_data.get('passive', "No")

    @property
    def bPlayTeleportAnimOnSpawn(self):
        return self._raw_data.get('bplayteleportanimonspawn', "0")



class npc_alien_slave(BaseNPC):
    model_ = "models/vortigaunt_slave.mdl"

    @property
    def passive(self):
        return self._raw_data.get('passive', "No")

    @property
    def bPlayTeleportAnimOnSpawn(self):
        return self._raw_data.get('bplayteleportanimonspawn', "0")



class npc_xort(BaseNPC):
    model_ = "models/vortigaunt_slave.mdl"

    @property
    def XortState(self):
        return self._raw_data.get('xortstate', "1")

    @property
    def m_nFearLevel(self):
        return self._raw_data.get('m_nfearlevel', "2")

    @property
    def m_nDamageCallEveryone(self):
        return parse_source_value(self._raw_data.get('m_ndamagecalleveryone', -1))

    @property
    def m_fDamageCallRadius(self):
        return parse_source_value(self._raw_data.get('m_fdamagecallradius', -1))

    @property
    def m_nAlertCallEveryone(self):
        return parse_source_value(self._raw_data.get('m_nalertcalleveryone', -1))

    @property
    def m_fAlertCallRadius(self):
        return parse_source_value(self._raw_data.get('m_falertcallradius', -1))

    @property
    def FearNodesGroupName(self):
        return self._raw_data.get('fearnodesgroupname', "")

    @property
    def HealNodesGroupName(self):
        return self._raw_data.get('healnodesgroupname', "")

    @property
    def bPlayTeleportAnimOnSpawn(self):
        return self._raw_data.get('bplayteleportanimonspawn', "0")

    @property
    def bMakeThemStationary(self):
        return self._raw_data.get('bmakethemstationary', "0")

    @property
    def bDisableSpells(self):
        return self._raw_data.get('bdisablespells', "0")

    @property
    def CanUseHealingNodes(self):
        return self._raw_data.get('canusehealingnodes', "1")

    @property
    def CanUseFearNodes(self):
        return self._raw_data.get('canusefearnodes', "1")

    @property
    def PossesBreakCooldownOVerride(self):
        return parse_source_value(self._raw_data.get('possesbreakcooldownoverride', 0))



class npc_xortEB(BaseNPC):
    model_ = "models/vortigaunt_slave.mdl"

    @property
    def XortState(self):
        return self._raw_data.get('xortstate', "1")

    @property
    def m_nFearLevel(self):
        return self._raw_data.get('m_nfearlevel', "0")

    @property
    def m_nDamageCallEveryone(self):
        return parse_source_value(self._raw_data.get('m_ndamagecalleveryone', -1))

    @property
    def m_fDamageCallRadius(self):
        return parse_source_value(self._raw_data.get('m_fdamagecallradius', -1))

    @property
    def m_nAlertCallEveryone(self):
        return parse_source_value(self._raw_data.get('m_nalertcalleveryone', -1))

    @property
    def m_fAlertCallRadius(self):
        return parse_source_value(self._raw_data.get('m_falertcallradius', -1))

    @property
    def FearNodesGroupName(self):
        return self._raw_data.get('fearnodesgroupname', "")

    @property
    def HealNodesGroupName(self):
        return self._raw_data.get('healnodesgroupname', "")

    @property
    def bPlayTeleportAnimOnSpawn(self):
        return self._raw_data.get('bplayteleportanimonspawn', "0")

    @property
    def bDisableSpells(self):
        return self._raw_data.get('bdisablespells', "0")

    @property
    def CanUseHealingNodes(self):
        return self._raw_data.get('canusehealingnodes', "0")

    @property
    def CanUseFearNodes(self):
        return self._raw_data.get('canusefearnodes', "0")



class npc_headcrab(BaseNPC, Parentname):
    model_ = "models/headcrabclassic.mdl"

    @property
    def startburrowed(self):
        return self._raw_data.get('startburrowed', "No")



class npc_headcrab_fast(BaseNPC):
    model_ = "models/Headcrab.mdl"
    pass


class npc_headcrab_black(BaseNPC):
    model_ = "models/Headcrabblack.mdl"
    pass


class npc_headcrab_baby(BaseNPC):
    model_ = "models/xenians/bebcrab.mdl"
    pass


class npc_barnacle(BaseNPC):
    model_ = "models/barnacle.mdl"

    @property
    def RestDist(self):
        return parse_source_value(self._raw_data.get('restdist', 16))



class npc_beneathticle(BaseNPC):
    model_ = "models/xenians/barnacle_underwater.mdl"

    @property
    def TongueLength(self):
        return parse_source_value(self._raw_data.get('tonguelength', 0))

    @property
    def TonguePullSpeed(self):
        return parse_source_value(self._raw_data.get('tonguepullspeed', 0))



class npc_bullsquid(BaseNPCAssault):
    model_ = "models/xenians/bullsquid.mdl"
    pass


class npc_bullsquid_melee(BaseNPCAssault):
    model_ = "models/xenians/bullsquid.mdl"
    pass


class npc_houndeye(BaseNPCAssault):
    model_ = "models/xenians/houndeye.mdl"

    @property
    def m_bEnableMemoryUpdateEveryFrame(self):
        return self._raw_data.get('m_benablememoryupdateeveryframe', "0")



class npc_houndeye_suicide(npc_houndeye):
    model_ = "models/xenians/houndeye_suicide.mdl"
    pass


class npc_houndeye_knockback(npc_houndeye):
    model_ = "models/xenians/houndeye_knockback.mdl"
    pass


class npc_human_assassin(BaseNPC):
    model_ = "models/humans/hassassin.mdl"
    pass


class BaseMarine(RappelNPC):

    @property
    def NumGrenades(self):
        return self._raw_data.get('numgrenades', "5")

    @property
    def additionalequipment(self):
        return self._raw_data.get('additionalequipment', "weapon_mp5")



class npc_human_commander(BaseMarine):
    model_ = "models/humans/marine.mdl"
    pass


class npc_human_grunt(BaseMarine):
    model_ = "models/humans/marine.mdl"
    pass


class npc_human_medic(BaseMarine):
    model_ = "models/humans/marine.mdl"
    pass


class npc_human_grenadier(BaseMarine):
    model_ = "models/humans/marine.mdl"
    pass


class npc_alien_controller(BaseNPC):
    model_ = "models/xenians/controller.mdl"
    pass


class npc_xontroller(BaseNPC):
    model_ = "models/xenians/controller.mdl"

    @property
    def mainbehaviortreename(self):
        return self._raw_data.get('mainbehaviortreename', "")

    @property
    def assaultbehaviortreename(self):
        return self._raw_data.get('assaultbehaviortreename', "")

    @property
    def preferred_pawns_group_01(self):
        return self._raw_data.get('preferred_pawns_group_01', "")

    @property
    def preferred_pawns_group_02(self):
        return self._raw_data.get('preferred_pawns_group_02', "")

    @property
    def preferred_pawns_group_03(self):
        return self._raw_data.get('preferred_pawns_group_03', "")

    @property
    def preferred_pawns_group_04(self):
        return self._raw_data.get('preferred_pawns_group_04', "")

    @property
    def preferred_pawns_group_05(self):
        return self._raw_data.get('preferred_pawns_group_05', "")

    @property
    def preferred_pawns_group_06(self):
        return self._raw_data.get('preferred_pawns_group_06', "")

    @property
    def preferred_pawns_group_07(self):
        return self._raw_data.get('preferred_pawns_group_07', "")

    @property
    def preferred_pawns_group_08(self):
        return self._raw_data.get('preferred_pawns_group_08', "")

    @property
    def preferred_pawns_group_09(self):
        return self._raw_data.get('preferred_pawns_group_09', "")

    @property
    def preferred_pawns_group_10(self):
        return self._raw_data.get('preferred_pawns_group_10', "")

    @property
    def preferred_pawns_group_11(self):
        return self._raw_data.get('preferred_pawns_group_11', "")

    @property
    def preferred_pawns_group_12(self):
        return self._raw_data.get('preferred_pawns_group_12', "")

    @property
    def preferred_pawns_group_13(self):
        return self._raw_data.get('preferred_pawns_group_13', "")

    @property
    def preferred_pawns_group_14(self):
        return self._raw_data.get('preferred_pawns_group_14', "")

    @property
    def preferred_pawns_group_15(self):
        return self._raw_data.get('preferred_pawns_group_15', "")

    @property
    def preferred_pawns_group_16(self):
        return self._raw_data.get('preferred_pawns_group_16', "")

    @property
    def select_preferred_pawns_only(self):
        return self._raw_data.get('select_preferred_pawns_only', "0")

    @property
    def mindcontrol_attacks_disabled(self):
        return self._raw_data.get('mindcontrol_attacks_disabled', "0")

    @property
    def attack_mode_energy_enabled(self):
        return self._raw_data.get('attack_mode_energy_enabled', "1")

    @property
    def attack_mode_cluster_enabled(self):
        return self._raw_data.get('attack_mode_cluster_enabled', "1")

    @property
    def attack_mode_brainwash_enabled(self):
        return self._raw_data.get('attack_mode_brainwash_enabled', "1")

    @property
    def attack_mode_throw_enabled(self):
        return self._raw_data.get('attack_mode_throw_enabled', "1")

    @property
    def attack_mode_smash_enabled(self):
        return self._raw_data.get('attack_mode_smash_enabled', "1")



class npc_alien_grunt_unarmored(BaseNPC):
    model_ = "models/xenians/agrunt_unarmored.mdl"
    pass


class npc_alien_grunt_melee(BaseNPC):
    model_ = "models/xenians/agrunt_unarmored.mdl"
    pass


class npc_alien_grunt(BaseNPC):
    model_ = "models/xenians/agrunt.mdl"
    pass


class npc_alien_grunt_elite(BaseNPC):
    model_ = "models/xenians/agrunt.mdl"
    pass


class npc_xen_grunt(BaseNPC):
    model_ = "models/xenians/agrunt.mdl"

    @property
    def mainbehaviortreename(self):
        return self._raw_data.get('mainbehaviortreename', "")

    @property
    def assaultbehaviortreename(self):
        return self._raw_data.get('assaultbehaviortreename', "")



class npc_cockroach(BaseNPC):
    model_ = "models/fauna/roach.mdl"
    pass


class npc_flyer_flock(BaseNPC):
    model_ = "models/xenians/flock.mdl"
    pass


class npc_gargantua(BaseNPCAssault):
    model_ = "models/xenians/garg.mdl"
    pass


class info_bigmomma(Node):
    model_ = "models/editor/ground_node.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def m_flRadius(self):
        return parse_source_value(self._raw_data.get('m_flradius', 1))

    @property
    def m_flDelay(self):
        return parse_source_value(self._raw_data.get('m_fldelay', 1))

    @property
    def reachtarget(self):
        return self._raw_data.get('reachtarget', None)

    @property
    def reachsequence(self):
        return self._raw_data.get('reachsequence', "0")

    @property
    def presequence(self):
        return self._raw_data.get('presequence', "0")



class npc_gonarch(Studiomodel, BaseNPC):
    model_ = "models/xenians/gonarch.mdl"

    @property
    def cavernbreed(self):
        return self._raw_data.get('cavernbreed', "No")

    @property
    def shovetargets(self):
        return self._raw_data.get('shovetargets', "")

    @property
    def taunttargets(self):
        return self._raw_data.get('taunttargets', "")

    @property
    def covertargetsHI(self):
        return self._raw_data.get('covertargetshi', "")

    @property
    def attacktargetsHI(self):
        return self._raw_data.get('attacktargetshi', "")

    @property
    def m_tGSState(self):
        return self._raw_data.get('m_tgsstate', "6")

    @property
    def bTouchKillActive(self):
        return self._raw_data.get('btouchkillactive', "No")



class env_gon_mortar_area(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def Radius(self):
        return parse_source_value(self._raw_data.get('radius', 100))



class npc_generic(BaseNPC):
    model_ = "models/gman.mdl"
    pass


class npc_gman(BaseNPC):
    model_ = "models/gman.mdl"
    pass


class npc_ichthyosaur(BaseNPC):
    model_ = "models/ichthyosaur.mdl"
    pass


class npc_maintenance(BaseNPC):
    model_ = "models/humans/maintenance/maintenance_1.mdl"
    pass


class npc_nihilanth(BaseNPC):
    model_ = "models/xenians/nihilanth.mdl"

    @property
    def m_tNHState(self):
        return parse_source_value(self._raw_data.get('m_tnhstate', 0))



class prop_nihi_shield(Angles, Targetname, Studiomodel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def m_fStartHP(self):
        return parse_source_value(self._raw_data.get('m_fstarthp', 1500))

    @property
    def m_szGaurdedEntityName(self):
        return self._raw_data.get('m_szgaurdedentityname', "")



class nihilanth_pylon(prop_dynamic_base):
    viewport_model = "models/props_xen/nil_pylon.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def bHealNihi(self):
        return parse_source_value(self._raw_data.get('bhealnihi', 1))

    @property
    def bHealShield(self):
        return parse_source_value(self._raw_data.get('bhealshield', 1))



class BMBaseHelicopter(BaseNPC):

    @property
    def InitialSpeed(self):
        return self._raw_data.get('initialspeed', "0")



class npc_manta(BMBaseHelicopter):
    model_ = "models/xenians/manta_jet.mdl"
    pass


class prop_xen_grunt_pod(Global, Parentname, BreakableProp, Angles, BaseFadeProp, DXLevelChoice, RenderFields, Studiomodel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def solid(self):
        return self._raw_data.get('solid', "6")

    @property
    def MyCustomMass(self):
        return parse_source_value(self._raw_data.get('mycustommass', 0))

    @property
    def SetBodyGroup(self):
        return parse_source_value(self._raw_data.get('setbodygroup', 0))

    @property
    def lightingorigin(self):
        return self._raw_data.get('lightingorigin', "")

    @property
    def SpawnEntityName(self):
        return self._raw_data.get('spawnentityname', "npc_alien_grunt_melee")

    @property
    def EnablePodLight(self):
        return self._raw_data.get('enablepodlight', "1")

    @property
    def EnablePodShadows(self):
        return self._raw_data.get('enablepodshadows', "0")

    @property
    def PodLightColor(self):
        return parse_int_vector(self._raw_data.get('podlightcolor', "232 251 0"))

    @property
    def ShouldKeepUpright(self):
        return self._raw_data.get('shouldkeepupright', "0")

    @property
    def TargetEntityToIgnore01(self):
        return self._raw_data.get('targetentitytoignore01', "")

    @property
    def TargetEntityToIgnore02(self):
        return self._raw_data.get('targetentitytoignore02', "")

    @property
    def TargetEntityToIgnore03(self):
        return self._raw_data.get('targetentitytoignore03', "")

    @property
    def TargetEntityToIgnore04(self):
        return self._raw_data.get('targetentitytoignore04', "")

    @property
    def TargetEntityToIgnore05(self):
        return self._raw_data.get('targetentitytoignore05', "")

    @property
    def TargetEntityToIgnore06(self):
        return self._raw_data.get('targetentitytoignore06', "")

    @property
    def TargetEntityToIgnore07(self):
        return self._raw_data.get('targetentitytoignore07', "")

    @property
    def TargetEntityToIgnore08(self):
        return self._raw_data.get('targetentitytoignore08', "")

    @property
    def TargetEntityToIgnore09(self):
        return self._raw_data.get('targetentitytoignore09', "")

    @property
    def TargetEntityToIgnore10(self):
        return self._raw_data.get('targetentitytoignore10', "")

    @property
    def TargetEntityToIgnore11(self):
        return self._raw_data.get('targetentitytoignore11', "")

    @property
    def TargetEntityToIgnore12(self):
        return self._raw_data.get('targetentitytoignore12', "")

    @property
    def TargetEntityToIgnore13(self):
        return self._raw_data.get('targetentitytoignore13', "")

    @property
    def TargetEntityToIgnore14(self):
        return self._raw_data.get('targetentitytoignore14', "")

    @property
    def TargetEntityToIgnore15(self):
        return self._raw_data.get('targetentitytoignore15', "")

    @property
    def TargetEntityToIgnore16(self):
        return self._raw_data.get('targetentitytoignore16', "")



class prop_xen_grunt_pod_dynamic(Global, Parentname, BreakableProp, Angles, BaseFadeProp, DXLevelChoice, RenderFields, Studiomodel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def solid(self):
        return self._raw_data.get('solid', "6")

    @property
    def MyCustomMass(self):
        return parse_source_value(self._raw_data.get('mycustommass', 0))

    @property
    def SetBodyGroup(self):
        return parse_source_value(self._raw_data.get('setbodygroup', 0))

    @property
    def lightingorigin(self):
        return self._raw_data.get('lightingorigin', "")

    @property
    def SpawnEntityName(self):
        return self._raw_data.get('spawnentityname', "npc_alien_grunt_melee")

    @property
    def PodLightColor(self):
        return parse_int_vector(self._raw_data.get('podlightcolor', "232 251 0"))

    @property
    def EnablePodLight(self):
        return self._raw_data.get('enablepodlight', "1")

    @property
    def EnablePodShadows(self):
        return self._raw_data.get('enablepodshadows', "0")



class prop_xen_int_barrel(Global, Parentname, BreakableProp, Angles, BaseFadeProp, DXLevelChoice, RenderFields, Studiomodel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def solid(self):
        return self._raw_data.get('solid', "6")

    @property
    def SetBodyGroup(self):
        return parse_source_value(self._raw_data.get('setbodygroup', 0))

    @property
    def lightingorigin(self):
        return self._raw_data.get('lightingorigin', "")

    @property
    def EnablePodLight(self):
        return self._raw_data.get('enablepodlight', "1")

    @property
    def EnablePodShadows(self):
        return self._raw_data.get('enablepodshadows', "0")

    @property
    def PodLightColor(self):
        return parse_int_vector(self._raw_data.get('podlightcolor', "232 251 0"))



class prop_barrel_cactus(prop_dynamic_base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def cactustype(self):
        return self._raw_data.get('cactustype', "0")

    @property
    def m_bOverrideModel(self):
        return self._raw_data.get('m_boverridemodel', "0")

    @property
    def m_bDisableGibs(self):
        return self._raw_data.get('m_bdisablegibs', "0")

    @property
    def lightRadius(self):
        return parse_source_value(self._raw_data.get('lightradius', 400))



class prop_barrel_cactus_semilarge(prop_dynamic_base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def cactustype(self):
        return self._raw_data.get('cactustype', "1")

    @property
    def m_bOverrideModel(self):
        return self._raw_data.get('m_boverridemodel', "0")

    @property
    def m_bDisableGibs(self):
        return self._raw_data.get('m_bdisablegibs', "0")

    @property
    def lightRadius(self):
        return parse_source_value(self._raw_data.get('lightradius', 400))



class prop_barrel_cactus_adolescent(prop_dynamic_base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def cactustype(self):
        return self._raw_data.get('cactustype', "2")

    @property
    def m_bOverrideModel(self):
        return self._raw_data.get('m_boverridemodel', "0")

    @property
    def m_bDisableGibs(self):
        return self._raw_data.get('m_bdisablegibs', "0")



class prop_barrel_cactus_infant(prop_dynamic_base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def cactustype(self):
        return self._raw_data.get('cactustype', "3")

    @property
    def m_bOverrideModel(self):
        return self._raw_data.get('m_boverridemodel', "0")

    @property
    def m_bDisableGibs(self):
        return self._raw_data.get('m_bdisablegibs', "0")



class prop_barrel_cactus_exploder(prop_dynamic_base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def cactustype(self):
        return self._raw_data.get('cactustype', "4")

    @property
    def m_bOverrideModel(self):
        return self._raw_data.get('m_boverridemodel', "0")

    @property
    def m_bDisableGibs(self):
        return self._raw_data.get('m_bdisablegibs', "0")



class prop_barrel_interloper(prop_dynamic_base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def cactustype(self):
        return self._raw_data.get('cactustype', "6")

    @property
    def m_bOverrideModel(self):
        return self._raw_data.get('m_boverridemodel', "0")

    @property
    def m_bDisableGibs(self):
        return self._raw_data.get('m_bdisablegibs', "0")

    @property
    def lightRadius(self):
        return parse_source_value(self._raw_data.get('lightradius', 400))



class prop_barrel_interloper_small(prop_dynamic_base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def cactustype(self):
        return self._raw_data.get('cactustype', "7")

    @property
    def m_bOverrideModel(self):
        return self._raw_data.get('m_boverridemodel', "0")

    @property
    def m_bDisableGibs(self):
        return self._raw_data.get('m_bdisablegibs', "0")

    @property
    def lightRadius(self):
        return parse_source_value(self._raw_data.get('lightradius', 400))



class npc_apache(BMBaseHelicopter):
    model_ = "models/props_vehicles/apache.mdl"

    @property
    def bNerfedFireCone(self):
        return self._raw_data.get('bnerfedfirecone', "0")



class npc_osprey(BMBaseHelicopter):
    model_ = "models/props_vehicles/osprey.mdl"

    @property
    def NPCTemplate1(self):
        return self._raw_data.get('npctemplate1', None)

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
    def NPCTemplate7(self):
        return self._raw_data.get('npctemplate7', None)

    @property
    def NPCTemplate8(self):
        return self._raw_data.get('npctemplate8', None)



class npc_rat(BaseNPC):
    model_ = "models/fauna/rat.mdl"
    pass


class BaseColleague(BaseNPC):

    @property
    def expressiontype(self):
        return self._raw_data.get('expressiontype', "Random")

    @property
    def CanSpeakWhileScripting(self):
        return self._raw_data.get('canspeakwhilescripting', "No")

    @property
    def AlwaysTransition(self):
        return self._raw_data.get('alwaystransition', "No")

    @property
    def GameEndAlly(self):
        return self._raw_data.get('gameendally', "No")



class npc_human_security(BaseColleague, Parentname, TalkNPC):
    model_ = "models/humans/guard.mdl"

    @property
    def additionalequipment(self):
        return self._raw_data.get('additionalequipment', "Default")



class npc_human_scientist_kleiner(BaseColleague, Parentname, TalkNPC):
    model_ = "models/humans/scientist_kliener.mdl"
    pass


class npc_human_scientist_eli(BaseColleague, Parentname, TalkNPC):
    model_ = "models/humans/scientist_eli.mdl"
    pass


class npc_human_scientist(BaseColleague, Parentname, TalkNPC):
    model_ = "models/humans/scientist.mdl"
    pass


class npc_human_scientist_female(BaseColleague, Parentname, TalkNPC):
    model_ = "models/humans/scientist_female.mdl"
    pass


class npc_xentacle(BaseNPC):
    model_ = "models/xenians/xentacle.mdl"

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 320))

    @property
    def target01(self):
        return self._raw_data.get('target01', "")

    @property
    def target02(self):
        return self._raw_data.get('target02', "")

    @property
    def target03(self):
        return self._raw_data.get('target03', "")

    @property
    def target04(self):
        return self._raw_data.get('target04', "")

    @property
    def target05(self):
        return self._raw_data.get('target05', "")

    @property
    def target06(self):
        return self._raw_data.get('target06', "")

    @property
    def target07(self):
        return self._raw_data.get('target07', "")

    @property
    def target08(self):
        return self._raw_data.get('target08', "")

    @property
    def target09(self):
        return self._raw_data.get('target09', "")

    @property
    def target10(self):
        return self._raw_data.get('target10', "")

    @property
    def target11(self):
        return self._raw_data.get('target11', "")

    @property
    def target12(self):
        return self._raw_data.get('target12', "")

    @property
    def target13(self):
        return self._raw_data.get('target13', "")

    @property
    def target14(self):
        return self._raw_data.get('target14', "")

    @property
    def target15(self):
        return self._raw_data.get('target15', "")

    @property
    def target16(self):
        return self._raw_data.get('target16', "")



class npc_tentacle(BaseNPC):
    model_ = "models/xenians/tentacle.mdl"
    pass


class npc_snark(BaseNPC):
    model_ = "models/xenians/snark.mdl"

    @property
    def m_fLifeTime(self):
        return parse_source_value(self._raw_data.get('m_flifetime', 14))



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



class info_target_helicoptercrash(Targetname, Parentname):
    icon_sprite = "editor/info_target.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class info_dlightmap_update(Targetname, Parentname):
    icon_sprite = "editor/info_target.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class info_timescale_controller(Targetname, Parentname):
    icon_sprite = "editor/info_target.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class info_stopallsounds(Targetname, Parentname):
    icon_sprite = "editor/info_target.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class info_player_deathmatch(Angles, Targetname, PlayerClass):
    model_ = "models/editor/playerstart.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def itemstogive(self):
        return self._raw_data.get('itemstogive', None)



class info_player_marine(Angles, Targetname, PlayerClass):
    model_ = "models/Player/mp_marine.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def itemstogive(self):
        return self._raw_data.get('itemstogive', None)



class info_player_scientist(Angles, Targetname, PlayerClass):
    model_ = "models/Player/mp_scientist_hev.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def itemstogive(self):
        return self._raw_data.get('itemstogive', None)



class material_timer(Targetname, Parentname):
    icon_sprite = "editor/info_target.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def length(self):
        return parse_source_value(self._raw_data.get('length', 30))



class xen_portal(Base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def size(self):
        return self._raw_data.get('size', "Default")

    @property
    def sound(self):
        return self._raw_data.get('sound', "XenPortal.Sound")

    @property
    def jump_distance(self):
        return parse_source_value(self._raw_data.get('jump_distance', 0))

    @property
    def jump_hmaxspeed(self):
        return parse_source_value(self._raw_data.get('jump_hmaxspeed', 200))

    @property
    def min_delay(self):
        return parse_source_value(self._raw_data.get('min_delay', 0))

    @property
    def max_delay(self):
        return parse_source_value(self._raw_data.get('max_delay', 0))



class env_introcredits(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def startactive(self):
        return self._raw_data.get('startactive', "Off")



class env_particle_beam(Targetname, Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def particlebeam(self):
        return self._raw_data.get('particlebeam', "Lc Beam")

    @property
    def target(self):
        return self._raw_data.get('target', "")

    @property
    def damage(self):
        return parse_source_value(self._raw_data.get('damage', 1))

    @property
    def damagetick(self):
        return parse_source_value(self._raw_data.get('damagetick', 0.1))

    @property
    def burntrail(self):
        return self._raw_data.get('burntrail', "effects/gluon_burn_trail.vmt")

    @property
    def burntrail_life(self):
        return parse_source_value(self._raw_data.get('burntrail_life', 4))

    @property
    def burntrail_size(self):
        return parse_source_value(self._raw_data.get('burntrail_size', 16))

    @property
    def burntrail_text(self):
        return parse_source_value(self._raw_data.get('burntrail_text', 0.01))

    @property
    def burntrail_flags(self):
        return self._raw_data.get('burntrail_flags', "Shrink + Fade")



class env_particle_tesla(Targetname, Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def particletesla(self):
        return self._raw_data.get('particletesla', "tesla_lc_core")

    @property
    def frequency(self):
        return parse_source_value(self._raw_data.get('frequency', 0.1))

    @property
    def mincount(self):
        return parse_source_value(self._raw_data.get('mincount', 2))

    @property
    def maxcount(self):
        return parse_source_value(self._raw_data.get('maxcount', 4))

    @property
    def range(self):
        return parse_source_value(self._raw_data.get('range', 2048))

    @property
    def life(self):
        return parse_source_value(self._raw_data.get('life', -1))

    @property
    def min(self):
        return parse_float_vector(self._raw_data.get('min', "-1 -1 -1"))

    @property
    def max(self):
        return parse_float_vector(self._raw_data.get('max', "1 1 1"))

    @property
    def decalname(self):
        return self._raw_data.get('decalname', "ZapScorch")



class env_xen_portal(npc_maker, xen_portal):
    icon_sprite = "Editor/Xen_Portal"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class env_xen_portal_template(npc_template_maker, xen_portal):
    icon_sprite = "Editor/Xen_Portal"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class env_pinch(Targetname, Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def timer(self):
        return parse_source_value(self._raw_data.get('timer', 1.8))

    @property
    def startsize(self):
        return parse_source_value(self._raw_data.get('startsize', 10))

    @property
    def endsize(self):
        return parse_source_value(self._raw_data.get('endsize', 30))



class env_dispenser(Targetname, Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def spawnmodel(self):
        return self._raw_data.get('spawnmodel', "models/props_junk/popcan01a.mdl")

    @property
    def spawnangles(self):
        return parse_float_vector(self._raw_data.get('spawnangles', "Orientation of the model at spawn (Y Z X)"))

    @property
    def capacity(self):
        return parse_source_value(self._raw_data.get('capacity', 15))

    @property
    def skinmin(self):
        return parse_source_value(self._raw_data.get('skinmin', 0))

    @property
    def skinmax(self):
        return parse_source_value(self._raw_data.get('skinmax', 0))



class item_crate(BasePropPhysics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def scriptpreset(self):
        return self._raw_data.get('scriptpreset', "")

    @property
    def spawnonbreak(self):
        return self._raw_data.get('spawnonbreak', "")



class func_50cal(Base):
    pass


class func_tow(Base):
    pass


class func_tow_mp(Base):
    pass


class func_conveyor_bms(Shadow, Targetname, Parentname, RenderFields):

    @property
    def direction(self):
        return parse_float_vector(self._raw_data.get('direction', "0 0 0"))

    @property
    def speed(self):
        return self._raw_data.get('speed', "150")



class item_tow_missile(BasePropPhysics):
    model_ = "models/props_marines/tow_missile_projectile.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class env_mortar_launcher(Targetname):
    icon_sprite = "editor/info_target.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def firedelay(self):
        return parse_source_value(self._raw_data.get('firedelay', 1))

    @property
    def rateoffire(self):
        return parse_source_value(self._raw_data.get('rateoffire', 10))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 128))

    @property
    def target(self):
        return self._raw_data.get('target', "")

    @property
    def grenadeentityname(self):
        return self._raw_data.get('grenadeentityname', "Small")

    @property
    def apexheightratio(self):
        return parse_source_value(self._raw_data.get('apexheightratio', 1))

    @property
    def pathoption(self):
        return self._raw_data.get('pathoption', "Apex")

    @property
    def fireshellscount(self):
        return parse_source_value(self._raw_data.get('fireshellscount', 1))

    @property
    def override_damage(self):
        return parse_source_value(self._raw_data.get('override_damage', -1))

    @property
    def override_damageradius(self):
        return parse_source_value(self._raw_data.get('override_damageradius', -1))



class env_mortar_controller(Angles, Targetname):
    model_ = "models/props_st/airstrike_map.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def MortarLauncher(self):
        return self._raw_data.get('mortarlauncher', None)



class npc_abrams(BaseNPC):
    model_ = "models/props_vehicles/abrams.mdl"

    @property
    def enableminiguns(self):
        return self._raw_data.get('enableminiguns', "1")

    @property
    def enablebodyrotation(self):
        return self._raw_data.get('enablebodyrotation', "1")



class npc_lav(BaseNPC):
    model_ = "models/props_vehicles/lav.mdl"
    pass


class env_tram_screen(Angles, Origin, Targetname):
    icon_sprite = "editor/info_target.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def panelname(self):
        return self._raw_data.get('panelname', None)

    @property
    def functrainname(self):
        return self._raw_data.get('functrainname', None)

    @property
    def propname(self):
        return self._raw_data.get('propname', None)



class prop_retinalscanner(prop_dynamic_base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def locked(self):
        return self._raw_data.get('locked', "0")

    @property
    def nextlockeduse(self):
        return parse_source_value(self._raw_data.get('nextlockeduse', 4))

    @property
    def nextunlockeduse(self):
        return parse_source_value(self._raw_data.get('nextunlockeduse', 4))

    @property
    def lockedsound(self):
        return self._raw_data.get('lockedsound', "")

    @property
    def unlockedsound(self):
        return self._raw_data.get('unlockedsound', "")

    @property
    def lockedusesound(self):
        return self._raw_data.get('lockedusesound', "")

    @property
    def unlockedusesound(self):
        return self._raw_data.get('unlockedusesound', "")

    @property
    def lockedusevox(self):
        return self._raw_data.get('lockedusevox', "")

    @property
    def unlockedusevox(self):
        return self._raw_data.get('unlockedusevox', "")

    @property
    def delaylockedvox(self):
        return parse_source_value(self._raw_data.get('delaylockedvox', 0))

    @property
    def delayunlockedvox(self):
        return parse_source_value(self._raw_data.get('delayunlockedvox', 0))



class prop_physics_respawnable(prop_physics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def RespawnTime(self):
        return parse_source_value(self._raw_data.get('respawntime', 60))



class prop_scalable(EnableDisable, prop_dynamic_base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class logic_parent(Targetname):
    icon_sprite = "editor/logic_auto.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class env_xen_pushpad(Targetname):
    model_ = "models/xenians/jump_pad.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def nextjumpdelta(self):
        return parse_source_value(self._raw_data.get('nextjumpdelta', 0.5))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def height(self):
        return parse_source_value(self._raw_data.get('height', 512))

    @property
    def disableshadows(self):
        return self._raw_data.get('disableshadows', "0")

    @property
    def m_bMuteME(self):
        return self._raw_data.get('m_bmuteme', "0")



class trigger_gargantua_shake(Trigger):
    pass


class trigger_lift(Trigger):

    @property
    def liftaccel(self):
        return parse_source_value(self._raw_data.get('liftaccel', 100))

    @property
    def clampspeed(self):
        return parse_source_value(self._raw_data.get('clampspeed', 512))



class trigger_weaponfire(trigger_multiple):
    pass


class func_minefield(Trigger):

    @property
    def minecount(self):
        return parse_source_value(self._raw_data.get('minecount', 25))

    @property
    def ranx(self):
        return parse_source_value(self._raw_data.get('ranx', 35))

    @property
    def rany(self):
        return parse_source_value(self._raw_data.get('rany', 35))



class func_friction(Trigger):

    @property
    def modifier(self):
        return parse_source_value(self._raw_data.get('modifier', 100))



class prop_train_awesome(Global, Targetname, Parentname, Angles, BaseFadeProp, DXLevelChoice, Shadow, RenderFields):
    model_ = "models/props_vehicles/oar_awesome_tram.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def solid(self):
        return self._raw_data.get('solid', "6")

    @property
    def bTrainDisabled(self):
        return self._raw_data.get('btraindisabled', "0")

    @property
    def lightingorigin(self):
        return self._raw_data.get('lightingorigin', "")



class prop_train_apprehension(Global, Targetname, Parentname, Angles, BaseFadeProp, DXLevelChoice, Shadow, RenderFields):
    model_ = "models/props_vehicles/oar_tram.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def solid(self):
        return self._raw_data.get('solid', "6")

    @property
    def bTrainDisabled(self):
        return self._raw_data.get('btraindisabled', "0")

    @property
    def lightingorigin(self):
        return self._raw_data.get('lightingorigin', "")



class BaseZombie(BaseNPC):
    pass


class npc_zombie_scientist(BaseZombie):
    model_ = "models/zombies/zombie_sci.mdl"
    pass


class npc_zombie_scientist_torso(BaseZombie):
    model_ = "models/zombies/zombie_sci_torso.mdl"
    pass


class npc_zombie_security(BaseZombie):
    model_ = "models/zombies/zombie_guard.mdl"
    pass


class npc_zombie_grunt(BaseZombie):
    model_ = "models/zombies/zombie_grunt.mdl"
    pass


class npc_zombie_grunt_torso(BaseZombie):
    model_ = "models/zombies/zombie_grunt_torso.mdl"
    pass


class npc_zombie_hev(BaseZombie):
    model_ = "models/zombies/zombie_hev.mdl"

    @property
    def flashlight_status(self):
        return self._raw_data.get('flashlight_status', "1")

    @property
    def FlashLight_Shadows(self):
        return self._raw_data.get('flashlight_shadows', "0")



class filter_damage_class(BaseFilter):

    @property
    def filterclass(self):
        return self._raw_data.get('filterclass', None)



class filter_activator_flag(BaseFilter):

    @property
    def flag(self):
        return self._raw_data.get('flag', "0")



class filter_activator_team(BaseFilter):

    @property
    def filterteam(self):
        return self._raw_data.get('filterteam', "2")



class prop_flare(BasePropPhysics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class prop_surgerybot(Angles, Targetname):
    viewport_model = "models/props_questionableethics/qe_surgery_bot_main.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def startactive(self):
        return self._raw_data.get('startactive', "1")



class env_xen_healpool(Angles, Targetname, Studiomodel):
    model_ = "models/props_Xen/xen_healingpool_full.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def healRate(self):
        return parse_source_value(self._raw_data.get('healrate', 5))



class env_xen_healshower(env_xen_healpool):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class prop_web_burnable(Angles, Targetname, Studiomodel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def m_fBurnTime(self):
        return parse_source_value(self._raw_data.get('m_fburntime', 2))



class prop_charger_base(Angles, Targetname, Studiomodel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 128))

    @property
    def chargerate(self):
        return parse_source_value(self._raw_data.get('chargerate', 1))

    @property
    def chargeamount(self):
        return parse_source_value(self._raw_data.get('chargeamount', 10))

    @property
    def warmuptime(self):
        return parse_source_value(self._raw_data.get('warmuptime', 5))

    @property
    def cooldowntime(self):
        return parse_source_value(self._raw_data.get('cooldowntime', 5))

    @property
    def warmlightcolor(self):
        return parse_int_vector(self._raw_data.get('warmlightcolor', "245 154 52"))

    @property
    def coollightcolor(self):
        return parse_int_vector(self._raw_data.get('coollightcolor', "128 255 255"))

    @property
    def lightpos(self):
        return parse_float_vector(self._raw_data.get('lightpos', "0 0 0"))

    @property
    def lightintensity(self):
        return parse_source_value(self._raw_data.get('lightintensity', 16000))

    @property
    def lightrange(self):
        return parse_source_value(self._raw_data.get('lightrange', 512))

    @property
    def bPlayIdleSounds(self):
        return self._raw_data.get('bplayidlesounds', "1")



class prop_hev_charger(prop_charger_base, Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class prop_radiation_charger(prop_charger_base, Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class camera_satellite(Angles, Parentname):
    viewport_model = "models/editor/camera.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def rendertarget(self):
        return self._raw_data.get('rendertarget', "_rt_Satellite")

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



class logic_achievement(EnableDisable, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def AchievementEvent(self):
        return self._raw_data.get('achievementevent', "0")



class ai_goal_throw_prop(Targetname):
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
    def PropName(self):
        return self._raw_data.get('propname', "")



class info_observer_menu(Angles):
    viewport_model = "models/editor/camera.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def observerid(self):
        return parse_source_value(self._raw_data.get('observerid', 0))



class game_round_win(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def team(self):
        return self._raw_data.get('team', "0")

    @property
    def force_map_reset(self):
        return self._raw_data.get('force_map_reset', "1")

    @property
    def switch_teams(self):
        return self._raw_data.get('switch_teams', "0")



class game_round_start(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class game_mp_gamerules(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class mp_round_time(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def WarmupTime(self):
        return parse_source_value(self._raw_data.get('warmuptime', 0))

    @property
    def RoundTime(self):
        return parse_source_value(self._raw_data.get('roundtime', 0))



class env_gravity(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class env_godrays_controller(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def Density(self):
        return parse_source_value(self._raw_data.get('density', 0.5))

    @property
    def Decay(self):
        return parse_source_value(self._raw_data.get('decay', 0.5))

    @property
    def Weight(self):
        return parse_source_value(self._raw_data.get('weight', 1.0))

    @property
    def Exposure(self):
        return parse_source_value(self._raw_data.get('exposure', 0.20))

    @property
    def DensityUW(self):
        return parse_source_value(self._raw_data.get('densityuw', 0.5))

    @property
    def DecayUW(self):
        return parse_source_value(self._raw_data.get('decayuw', 0.5))

    @property
    def WeightUW(self):
        return parse_source_value(self._raw_data.get('weightuw', 1.0))

    @property
    def ExposureUW(self):
        return parse_source_value(self._raw_data.get('exposureuw', 0.20))



class misc_dead_hev(EnableDisable, Targetname, Angles, BaseFadeProp, DXLevelChoice, Studiomodel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def spritecolor(self):
        return parse_int_vector(self._raw_data.get('spritecolor', "255 0 0 200"))

    @property
    def lightcolor(self):
        return parse_int_vector(self._raw_data.get('lightcolor', "255 0 0 4"))

    @property
    def lightradius(self):
        return parse_source_value(self._raw_data.get('lightradius', 64))

    @property
    def attachmentname(self):
        return self._raw_data.get('attachmentname', "eyes")

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 100))



class env_lensflare(Angles, Targetname, Parentname):
    icon_sprite = "editor/lensflare.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def FlareFile(self):
        return self._raw_data.get('flarefile', "")

    @property
    def FlareAttenuation(self):
        return parse_source_value(self._raw_data.get('flareattenuation', 0.0))

    @property
    def FlareType(self):
        return self._raw_data.get('flaretype', "0")

    @property
    def FlareStyle(self):
        return self._raw_data.get('flarestyle', "0")

    @property
    def GlowProxySize(self):
        return parse_source_value(self._raw_data.get('glowproxysize', 2.0))

    @property
    def Flare01_texture(self):
        return self._raw_data.get('flare01_texture', "")

    @property
    def Flare01_params(self):
        return self._raw_data.get('flare01_params', "")

    @property
    def Flare01_intensity(self):
        return parse_float_vector(self._raw_data.get('flare01_intensity', ""))

    @property
    def Flare01_sizes(self):
        return parse_float_vector(self._raw_data.get('flare01_sizes', ""))

    @property
    def Flare01_color(self):
        return parse_int_vector(self._raw_data.get('flare01_color', ""))

    @property
    def Flare02_texture(self):
        return self._raw_data.get('flare02_texture', "")

    @property
    def Flare02_params(self):
        return self._raw_data.get('flare02_params', "")

    @property
    def Flare02_intensity(self):
        return parse_float_vector(self._raw_data.get('flare02_intensity', ""))

    @property
    def Flare02_sizes(self):
        return parse_float_vector(self._raw_data.get('flare02_sizes', ""))

    @property
    def Flare02_color(self):
        return parse_int_vector(self._raw_data.get('flare02_color', ""))

    @property
    def Flare03_texture(self):
        return self._raw_data.get('flare03_texture', "")

    @property
    def Flare03_params(self):
        return self._raw_data.get('flare03_params', "")

    @property
    def Flare03_intensity(self):
        return parse_float_vector(self._raw_data.get('flare03_intensity', ""))

    @property
    def Flare03_sizes(self):
        return parse_float_vector(self._raw_data.get('flare03_sizes', ""))

    @property
    def Flare03_color(self):
        return parse_int_vector(self._raw_data.get('flare03_color', ""))

    @property
    def Flare04_texture(self):
        return self._raw_data.get('flare04_texture', "")

    @property
    def Flare04_params(self):
        return self._raw_data.get('flare04_params', "")

    @property
    def Flare04_intensity(self):
        return parse_float_vector(self._raw_data.get('flare04_intensity', ""))

    @property
    def Flare04_sizes(self):
        return parse_float_vector(self._raw_data.get('flare04_sizes', ""))

    @property
    def Flare04_color(self):
        return parse_int_vector(self._raw_data.get('flare04_color', ""))

    @property
    def Flare05_texture(self):
        return self._raw_data.get('flare05_texture', "")

    @property
    def Flare05_params(self):
        return self._raw_data.get('flare05_params', "")

    @property
    def Flare05_intensity(self):
        return parse_float_vector(self._raw_data.get('flare05_intensity', ""))

    @property
    def Flare05_sizes(self):
        return parse_float_vector(self._raw_data.get('flare05_sizes', ""))

    @property
    def Flare05_color(self):
        return parse_int_vector(self._raw_data.get('flare05_color', ""))

    @property
    def Flare06_texture(self):
        return self._raw_data.get('flare06_texture', "")

    @property
    def Flare06_params(self):
        return self._raw_data.get('flare06_params', "")

    @property
    def Flare06_intensity(self):
        return parse_float_vector(self._raw_data.get('flare06_intensity', ""))

    @property
    def Flare06_sizes(self):
        return parse_float_vector(self._raw_data.get('flare06_sizes', ""))

    @property
    def Flare06_color(self):
        return parse_int_vector(self._raw_data.get('flare06_color', ""))

    @property
    def Flare07_texture(self):
        return self._raw_data.get('flare07_texture', "")

    @property
    def Flare07_params(self):
        return self._raw_data.get('flare07_params', "")

    @property
    def Flare07_intensity(self):
        return parse_float_vector(self._raw_data.get('flare07_intensity', ""))

    @property
    def Flare07_sizes(self):
        return parse_float_vector(self._raw_data.get('flare07_sizes', ""))

    @property
    def Flare07_color(self):
        return parse_int_vector(self._raw_data.get('flare07_color', ""))

    @property
    def Flare08_texture(self):
        return self._raw_data.get('flare08_texture', "")

    @property
    def Flare08_params(self):
        return self._raw_data.get('flare08_params', "")

    @property
    def Flare08_intensity(self):
        return parse_float_vector(self._raw_data.get('flare08_intensity', ""))

    @property
    def Flare08_sizes(self):
        return parse_float_vector(self._raw_data.get('flare08_sizes', ""))

    @property
    def Flare08_color(self):
        return parse_int_vector(self._raw_data.get('flare08_color', ""))

    @property
    def Flare09_texture(self):
        return self._raw_data.get('flare09_texture', "")

    @property
    def Flare09_params(self):
        return self._raw_data.get('flare09_params', "")

    @property
    def Flare09_intensity(self):
        return parse_float_vector(self._raw_data.get('flare09_intensity', ""))

    @property
    def Flare09_sizes(self):
        return parse_float_vector(self._raw_data.get('flare09_sizes', ""))

    @property
    def Flare09_color(self):
        return parse_int_vector(self._raw_data.get('flare09_color', ""))

    @property
    def Flare10_texture(self):
        return self._raw_data.get('flare10_texture', "")

    @property
    def Flare10_params(self):
        return self._raw_data.get('flare10_params', "")

    @property
    def Flare10_intensity(self):
        return parse_float_vector(self._raw_data.get('flare10_intensity', ""))

    @property
    def Flare10_sizes(self):
        return parse_float_vector(self._raw_data.get('flare10_sizes', ""))

    @property
    def Flare10_color(self):
        return parse_int_vector(self._raw_data.get('flare10_color', ""))

    @property
    def Flare11_texture(self):
        return self._raw_data.get('flare11_texture', "")

    @property
    def Flare11_params(self):
        return self._raw_data.get('flare11_params', "")

    @property
    def Flare11_intensity(self):
        return parse_float_vector(self._raw_data.get('flare11_intensity', ""))

    @property
    def Flare11_sizes(self):
        return parse_float_vector(self._raw_data.get('flare11_sizes', ""))

    @property
    def Flare11_color(self):
        return parse_int_vector(self._raw_data.get('flare11_color', ""))

    @property
    def Flare12_texture(self):
        return self._raw_data.get('flare12_texture', "")

    @property
    def Flare12_params(self):
        return self._raw_data.get('flare12_params', "")

    @property
    def Flare12_intensity(self):
        return parse_float_vector(self._raw_data.get('flare12_intensity', ""))

    @property
    def Flare12_sizes(self):
        return parse_float_vector(self._raw_data.get('flare12_sizes', ""))

    @property
    def Flare12_color(self):
        return parse_int_vector(self._raw_data.get('flare12_color', ""))

    @property
    def Flare13_texture(self):
        return self._raw_data.get('flare13_texture', "")

    @property
    def Flare13_params(self):
        return self._raw_data.get('flare13_params', "")

    @property
    def Flare13_intensity(self):
        return parse_float_vector(self._raw_data.get('flare13_intensity', ""))

    @property
    def Flare13_sizes(self):
        return parse_float_vector(self._raw_data.get('flare13_sizes', ""))

    @property
    def Flare13_color(self):
        return parse_int_vector(self._raw_data.get('flare13_color', ""))

    @property
    def Flare14_texture(self):
        return self._raw_data.get('flare14_texture', "")

    @property
    def Flare14_params(self):
        return self._raw_data.get('flare14_params', "")

    @property
    def Flare14_intensity(self):
        return parse_float_vector(self._raw_data.get('flare14_intensity', ""))

    @property
    def Flare14_sizes(self):
        return parse_float_vector(self._raw_data.get('flare14_sizes', ""))

    @property
    def Flare14_color(self):
        return parse_int_vector(self._raw_data.get('flare14_color', ""))

    @property
    def Flare15_texture(self):
        return self._raw_data.get('flare15_texture', "")

    @property
    def Flare15_params(self):
        return self._raw_data.get('flare15_params', "")

    @property
    def Flare15_intensity(self):
        return parse_float_vector(self._raw_data.get('flare15_intensity', ""))

    @property
    def Flare15_sizes(self):
        return parse_float_vector(self._raw_data.get('flare15_sizes', ""))

    @property
    def Flare15_color(self):
        return parse_int_vector(self._raw_data.get('flare15_color', ""))

    @property
    def Flare16_texture(self):
        return self._raw_data.get('flare16_texture', "")

    @property
    def Flare16_params(self):
        return self._raw_data.get('flare16_params', "")

    @property
    def Flare16_intensity(self):
        return parse_float_vector(self._raw_data.get('flare16_intensity', ""))

    @property
    def Flare16_sizes(self):
        return parse_float_vector(self._raw_data.get('flare16_sizes', ""))

    @property
    def Flare16_color(self):
        return parse_int_vector(self._raw_data.get('flare16_color', ""))

    @property
    def Flare17_texture(self):
        return self._raw_data.get('flare17_texture', "")

    @property
    def Flare17_params(self):
        return self._raw_data.get('flare17_params', "")

    @property
    def Flare17_intensity(self):
        return parse_float_vector(self._raw_data.get('flare17_intensity', ""))

    @property
    def Flare17_sizes(self):
        return parse_float_vector(self._raw_data.get('flare17_sizes', ""))

    @property
    def Flare17_color(self):
        return parse_int_vector(self._raw_data.get('flare17_color', ""))

    @property
    def Flare18_texture(self):
        return self._raw_data.get('flare18_texture', "")

    @property
    def Flare18_params(self):
        return self._raw_data.get('flare18_params', "")

    @property
    def Flare18_intensity(self):
        return parse_float_vector(self._raw_data.get('flare18_intensity', ""))

    @property
    def Flare18_sizes(self):
        return parse_float_vector(self._raw_data.get('flare18_sizes', ""))

    @property
    def Flare18_color(self):
        return parse_int_vector(self._raw_data.get('flare18_color', ""))

    @property
    def Flare19_texture(self):
        return self._raw_data.get('flare19_texture', "")

    @property
    def Flare19_params(self):
        return self._raw_data.get('flare19_params', "")

    @property
    def Flare19_intensity(self):
        return parse_float_vector(self._raw_data.get('flare19_intensity', ""))

    @property
    def Flare19_sizes(self):
        return parse_float_vector(self._raw_data.get('flare19_sizes', ""))

    @property
    def Flare19_color(self):
        return parse_int_vector(self._raw_data.get('flare19_color', ""))

    @property
    def Flare20_texture(self):
        return self._raw_data.get('flare20_texture', "")

    @property
    def Flare20_params(self):
        return self._raw_data.get('flare20_params', "")

    @property
    def Flare20_intensity(self):
        return parse_float_vector(self._raw_data.get('flare20_intensity', ""))

    @property
    def Flare20_sizes(self):
        return parse_float_vector(self._raw_data.get('flare20_sizes', ""))

    @property
    def Flare20_color(self):
        return parse_int_vector(self._raw_data.get('flare20_color', ""))



class env_fumer(EnableDisable, Targetname, Parentname, Angles, BaseFadeProp, Studiomodel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def DetectionRadius(self):
        return parse_source_value(self._raw_data.get('detectionradius', 128))

    @property
    def ExplodeRaius(self):
        return parse_source_value(self._raw_data.get('exploderaius', 128))

    @property
    def ExplodeDmg(self):
        return parse_source_value(self._raw_data.get('explodedmg', 30))

    @property
    def ExplodeForce(self):
        return parse_source_value(self._raw_data.get('explodeforce', 1))

    @property
    def FlameTime(self):
        return parse_source_value(self._raw_data.get('flametime', 10))



class trigger_apply_impulse(Trigger):

    @property
    def impulse_dir(self):
        return parse_float_vector(self._raw_data.get('impulse_dir', "0 0 0"))

    @property
    def force(self):
        return parse_source_value(self._raw_data.get('force', 300))



class info_nihilanth_summon(Angles, Targetname, Parentname):
    icon_sprite = "editor/info_target.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class point_weaponstrip(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def Weapon(self):
        return self._raw_data.get('weapon', "2")



class misc_marionettist(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def innerdestinationradius(self):
        return parse_source_value(self._raw_data.get('innerdestinationradius', 32))

    @property
    def innerpullspeed(self):
        return parse_source_value(self._raw_data.get('innerpullspeed', 448))

    @property
    def outerdestinationradius(self):
        return parse_source_value(self._raw_data.get('outerdestinationradius', 128))

    @property
    def outerpullspeed(self):
        return parse_source_value(self._raw_data.get('outerpullspeed', 512))

    @property
    def ignorecollisions(self):
        return self._raw_data.get('ignorecollisions', "0")

    @property
    def target01(self):
        return self._raw_data.get('target01', "")

    @property
    def target02(self):
        return self._raw_data.get('target02', "")

    @property
    def target03(self):
        return self._raw_data.get('target03', "")

    @property
    def target04(self):
        return self._raw_data.get('target04', "")

    @property
    def target05(self):
        return self._raw_data.get('target05', "")

    @property
    def target06(self):
        return self._raw_data.get('target06', "")

    @property
    def target07(self):
        return self._raw_data.get('target07', "")

    @property
    def target08(self):
        return self._raw_data.get('target08', "")

    @property
    def target09(self):
        return self._raw_data.get('target09', "")

    @property
    def target10(self):
        return self._raw_data.get('target10', "")

    @property
    def target11(self):
        return self._raw_data.get('target11', "")

    @property
    def target12(self):
        return self._raw_data.get('target12', "")

    @property
    def target13(self):
        return self._raw_data.get('target13', "")

    @property
    def target14(self):
        return self._raw_data.get('target14', "")

    @property
    def target15(self):
        return self._raw_data.get('target15', "")

    @property
    def target16(self):
        return self._raw_data.get('target16', "")

    @property
    def soundscriptstart(self):
        return self._raw_data.get('soundscriptstart', "")

    @property
    def soundscriptloop(self):
        return self._raw_data.get('soundscriptloop', "")

    @property
    def soundscriptend(self):
        return self._raw_data.get('soundscriptend', "")



class misc_xen_healing_pylon(Angles, Targetname, Studiomodel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def max_health(self):
        return parse_source_value(self._raw_data.get('max_health', 200))

    @property
    def can_be_damaged_only_healing(self):
        return self._raw_data.get('can_be_damaged_only_healing', "0")

    @property
    def danger_recovering_duration(self):
        return parse_source_value(self._raw_data.get('danger_recovering_duration', 5.0))

    @property
    def healing_request_duration(self):
        return parse_source_value(self._raw_data.get('healing_request_duration', 5.0))

    @property
    def healing_request_hp_per_tick(self):
        return parse_source_value(self._raw_data.get('healing_request_hp_per_tick', 16))

    @property
    def healing_request_tick_delta(self):
        return parse_source_value(self._raw_data.get('healing_request_tick_delta', 0.125))

    @property
    def healing_beam_attachment_name(self):
        return self._raw_data.get('healing_beam_attachment_name', "")

    @property
    def healing_beam_spread_radius(self):
        return parse_source_value(self._raw_data.get('healing_beam_spread_radius', 16.0))

    @property
    def healing_beam_sprite_model(self):
        return self._raw_data.get('healing_beam_sprite_model', "sprites/rollermine_shock.vmt")

    @property
    def healing_beam_noise_amplitude(self):
        return parse_source_value(self._raw_data.get('healing_beam_noise_amplitude', 4.0))

    @property
    def healing_beam_starting_width(self):
        return parse_source_value(self._raw_data.get('healing_beam_starting_width', 8.0))

    @property
    def healing_beam_ending_width(self):
        return parse_source_value(self._raw_data.get('healing_beam_ending_width', 32.0))

    @property
    def healing_beam_color(self):
        return parse_int_vector(self._raw_data.get('healing_beam_color', "255 255 255 255"))

    @property
    def healing_beam_starting_pfx(self):
        return self._raw_data.get('healing_beam_starting_pfx', "gloun_zap")

    @property
    def healing_beam_ending_pfx(self):
        return self._raw_data.get('healing_beam_ending_pfx', "gloun_zap")

    @property
    def pylon_sequence_opening(self):
        return self._raw_data.get('pylon_sequence_opening', "deploy")

    @property
    def pylon_sequence_opened_idle(self):
        return self._raw_data.get('pylon_sequence_opened_idle', "idle_deploy")

    @property
    def pylon_sequence_closing(self):
        return self._raw_data.get('pylon_sequence_closing', "retract")

    @property
    def pylon_sequence_closed_idle(self):
        return self._raw_data.get('pylon_sequence_closed_idle', "idle_retract")

    @property
    def pylon_sequence_dying(self):
        return self._raw_data.get('pylon_sequence_dying', "explode")

    @property
    def pylon_sequence_died_idle(self):
        return self._raw_data.get('pylon_sequence_died_idle', "idle_explode")

    @property
    def trace_targetname_filter(self):
        return self._raw_data.get('trace_targetname_filter', "")



class misc_xen_shield(Angles, Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def panel_modelname_template(self):
        return self._raw_data.get('panel_modelname_template', "models/xenians/shield/pentagonal.hexecontahedron/nihilanth/panel.%03d.mdl")

    @property
    def panels_amount(self):
        return parse_source_value(self._raw_data.get('panels_amount', 60))

    @property
    def max_health_for_panel(self):
        return parse_source_value(self._raw_data.get('max_health_for_panel', 75))

    @property
    def max_health(self):
        return parse_source_value(self._raw_data.get('max_health', 3000))

    @property
    def healing_per_tick_for_panel(self):
        return parse_source_value(self._raw_data.get('healing_per_tick_for_panel', 2))

    @property
    def healing_tick_delta_for_panel(self):
        return parse_source_value(self._raw_data.get('healing_tick_delta_for_panel', 0.125))

    @property
    def healing_request_cooldown(self):
        return parse_source_value(self._raw_data.get('healing_request_cooldown', 7.5))

    @property
    def hp_amount_to_request_heal(self):
        return parse_source_value(self._raw_data.get('hp_amount_to_request_heal', 0.85))

    @property
    def pylon01(self):
        return self._raw_data.get('pylon01', "")

    @property
    def pylon02(self):
        return self._raw_data.get('pylon02', "")

    @property
    def pylon03(self):
        return self._raw_data.get('pylon03', "")

    @property
    def pylon04(self):
        return self._raw_data.get('pylon04', "")

    @property
    def pylon05(self):
        return self._raw_data.get('pylon05', "")

    @property
    def pylon06(self):
        return self._raw_data.get('pylon06', "")

    @property
    def pylon07(self):
        return self._raw_data.get('pylon07', "")

    @property
    def pylon08(self):
        return self._raw_data.get('pylon08', "")

    @property
    def pylon09(self):
        return self._raw_data.get('pylon09', "")

    @property
    def pylon10(self):
        return self._raw_data.get('pylon10', "")

    @property
    def pylon11(self):
        return self._raw_data.get('pylon11', "")

    @property
    def pylon12(self):
        return self._raw_data.get('pylon12', "")

    @property
    def pylon13(self):
        return self._raw_data.get('pylon13', "")

    @property
    def pylon14(self):
        return self._raw_data.get('pylon14', "")

    @property
    def pylon15(self):
        return self._raw_data.get('pylon15', "")

    @property
    def pylon16(self):
        return self._raw_data.get('pylon16', "")

    @property
    def angular_velocity_value01(self):
        return parse_float_vector(self._raw_data.get('angular_velocity_value01', "5.0 30.0 15.0"))

    @property
    def angular_velocity_value02(self):
        return parse_float_vector(self._raw_data.get('angular_velocity_value02', "-25.0 45.0 -5.0"))

    @property
    def angular_velocity_value03(self):
        return parse_float_vector(self._raw_data.get('angular_velocity_value03', "5.0 60.0 15.0"))

    @property
    def angular_velocity_value04(self):
        return parse_float_vector(self._raw_data.get('angular_velocity_value04', "25.0 45.0 0.0"))

    @property
    def angular_velocity_value05(self):
        return parse_float_vector(self._raw_data.get('angular_velocity_value05', "-5.0 15.0 -15.0"))

    @property
    def angular_velocity_value06(self):
        return parse_float_vector(self._raw_data.get('angular_velocity_value06', ""))

    @property
    def angular_velocity_value07(self):
        return parse_float_vector(self._raw_data.get('angular_velocity_value07', ""))

    @property
    def angular_velocity_value08(self):
        return parse_float_vector(self._raw_data.get('angular_velocity_value08', ""))

    @property
    def angular_velocity_value09(self):
        return parse_float_vector(self._raw_data.get('angular_velocity_value09', ""))

    @property
    def angular_velocity_value10(self):
        return parse_float_vector(self._raw_data.get('angular_velocity_value10', ""))

    @property
    def angular_velocity_value11(self):
        return parse_float_vector(self._raw_data.get('angular_velocity_value11', ""))

    @property
    def angular_velocity_value12(self):
        return parse_float_vector(self._raw_data.get('angular_velocity_value12', ""))

    @property
    def angular_velocity_value13(self):
        return parse_float_vector(self._raw_data.get('angular_velocity_value13', ""))

    @property
    def angular_velocity_value14(self):
        return parse_float_vector(self._raw_data.get('angular_velocity_value14', ""))

    @property
    def angular_velocity_value15(self):
        return parse_float_vector(self._raw_data.get('angular_velocity_value15', ""))

    @property
    def angular_velocity_value16(self):
        return parse_float_vector(self._raw_data.get('angular_velocity_value16', ""))

    @property
    def angular_velocity_values_used(self):
        return parse_source_value(self._raw_data.get('angular_velocity_values_used', 5))

    @property
    def health_color01(self):
        return parse_float_vector(self._raw_data.get('health_color01', "1.0 0.0 0.0"))

    @property
    def health_color02(self):
        return parse_float_vector(self._raw_data.get('health_color02', "1.0 1.0 0.0"))

    @property
    def health_color03(self):
        return parse_float_vector(self._raw_data.get('health_color03', "0.0 1.0 0.0"))

    @property
    def health_color04(self):
        return parse_float_vector(self._raw_data.get('health_color04', "0.0 1.0 1.0"))

    @property
    def health_color05(self):
        return parse_float_vector(self._raw_data.get('health_color05', "0.0 0.0 1.0"))

    @property
    def health_color06(self):
        return parse_float_vector(self._raw_data.get('health_color06', ""))

    @property
    def health_color07(self):
        return parse_float_vector(self._raw_data.get('health_color07', ""))

    @property
    def health_color08(self):
        return parse_float_vector(self._raw_data.get('health_color08', ""))

    @property
    def health_color09(self):
        return parse_float_vector(self._raw_data.get('health_color09', ""))

    @property
    def health_color10(self):
        return parse_float_vector(self._raw_data.get('health_color10', ""))

    @property
    def health_color11(self):
        return parse_float_vector(self._raw_data.get('health_color11', ""))

    @property
    def health_color12(self):
        return parse_float_vector(self._raw_data.get('health_color12', ""))

    @property
    def health_color13(self):
        return parse_float_vector(self._raw_data.get('health_color13', ""))

    @property
    def health_color14(self):
        return parse_float_vector(self._raw_data.get('health_color14', ""))

    @property
    def health_color15(self):
        return parse_float_vector(self._raw_data.get('health_color15', ""))

    @property
    def health_color16(self):
        return parse_float_vector(self._raw_data.get('health_color16', ""))

    @property
    def health_colors_used(self):
        return parse_source_value(self._raw_data.get('health_colors_used', 5))

    @property
    def intro_for_panel_minimum(self):
        return parse_source_value(self._raw_data.get('intro_for_panel_minimum', 2.5))

    @property
    def intro_for_panel_maximum(self):
        return parse_source_value(self._raw_data.get('intro_for_panel_maximum', 5.0))

    @property
    def pause_for_panel_minimum(self):
        return parse_source_value(self._raw_data.get('pause_for_panel_minimum', 0.5))

    @property
    def pause_for_panel_maximum(self):
        return parse_source_value(self._raw_data.get('pause_for_panel_maximum', 2.5))

    @property
    def death_for_panel(self):
        return parse_source_value(self._raw_data.get('death_for_panel', 1.5))

    @property
    def per_panel_color_scheme(self):
        return self._raw_data.get('per_panel_color_scheme', "0")

    @property
    def no_impact_on_alive_pylons(self):
        return self._raw_data.get('no_impact_on_alive_pylons', "1")



class prop_physics_psychokinesis(BasePropPhysics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class nihiportalsbase(Angles, Targetname, Studiomodel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def m_fTimeToActivate(self):
        return parse_source_value(self._raw_data.get('m_ftimetoactivate', 0))

    @property
    def m_fTimeToDie(self):
        return parse_source_value(self._raw_data.get('m_ftimetodie', 9000))

    @property
    def m_bManualAwake(self):
        return self._raw_data.get('m_bmanualawake', "0")

    @property
    def m_bLightNeeded(self):
        return self._raw_data.get('m_blightneeded', "0")



class nihiportals_teleprops(Angles, Targetname, Studiomodel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def m_fTimeToActivate(self):
        return parse_source_value(self._raw_data.get('m_ftimetoactivate', 0))

    @property
    def m_fTimeToDie(self):
        return parse_source_value(self._raw_data.get('m_ftimetodie', 9000))

    @property
    def m_bManualAwake(self):
        return self._raw_data.get('m_bmanualawake', "0")

    @property
    def m_bLightNeeded(self):
        return self._raw_data.get('m_blightneeded', "0")



class music_track(Targetname):
    icon_sprite = "editor/ambient_generic.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def track_script_sound(self):
        return self._raw_data.get('track_script_sound', "")

    @property
    def next_track_entity(self):
        return self._raw_data.get('next_track_entity', "")




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