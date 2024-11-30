
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


class PaintableBrush(Base):
    pass


class Angles(Base):

    @property
    def angles(self):
        return parse_float_vector(self._raw_data.get('angles', "0 0 0"))



class Origin(Base):

    @property
    def origin(self):
        return parse_float_vector(self._raw_data.get('origin', None))



class Reflection(Base):

    @property
    def drawinfastreflection(self):
        return self._raw_data.get('drawinfastreflection', "0")



class ToggleDraw(Base):
    pass


class Shadow(Base):

    @property
    def disableshadows(self):
        return self._raw_data.get('disableshadows', "0")

    @property
    def disableshadowdepth(self):
        return self._raw_data.get('disableshadowdepth', "0")

    @property
    def shadowdepthnocache(self):
        return self._raw_data.get('shadowdepthnocache', "0")

    @property
    def disableflashlight(self):
        return self._raw_data.get('disableflashlight', "0")



class Studiomodel(Reflection, Shadow, ToggleDraw):

    @property
    def model(self):
        return self._raw_data.get('model', None)

    @property
    def skin(self):
        return parse_source_value(self._raw_data.get('skin', 0))

    @property
    def body(self):
        return parse_source_value(self._raw_data.get('body', 0))



class BasePlat(Base):
    pass


class Targetname(Base):

    @property
    def targetname(self):
        return self._raw_data.get('targetname', None)

    @property
    def vscripts(self):
        return self._raw_data.get('vscripts', "")

    @property
    def thinkfunction(self):
        return self._raw_data.get('thinkfunction', "")



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



class SpatialEntity(Base):

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
    def fadeInDuration(self):
        return parse_source_value(self._raw_data.get('fadeinduration', 0.0))

    @property
    def fadeOutDuration(self):
        return parse_source_value(self._raw_data.get('fadeoutduration', 0.0))



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

    @property
    def fademindist(self):
        return parse_source_value(self._raw_data.get('fademindist', -1))

    @property
    def fademaxdist(self):
        return parse_source_value(self._raw_data.get('fademaxdist', 0))

    @property
    def fadescale(self):
        return parse_source_value(self._raw_data.get('fadescale', 1))



class SystemLevelChoice(Base):

    @property
    def mincpulevel(self):
        return self._raw_data.get('mincpulevel', "0")

    @property
    def maxcpulevel(self):
        return self._raw_data.get('maxcpulevel', "0")

    @property
    def mingpulevel(self):
        return self._raw_data.get('mingpulevel', "0")

    @property
    def maxgpulevel(self):
        return self._raw_data.get('maxgpulevel', "0")

    @property
    def disableX360(self):
        return self._raw_data.get('disablex360', "0")



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



class Breakable(DamageFilter, Targetname, Reflection, Shadow):

    @property
    def ExplodeDamage(self):
        return parse_source_value(self._raw_data.get('explodedamage', 0))

    @property
    def ExplodeRadius(self):
        return parse_source_value(self._raw_data.get('exploderadius', 0))

    @property
    def PerformanceMode(self):
        return self._raw_data.get('performancemode', "0")



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



class BaseNPC(ResponseContext, RenderFields, Shadow, DamageFilter, ToggleDraw, Targetname, Angles):

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



class info_npc_spawn_destination(Targetname, Parentname, Angles):
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



class BaseNPCMaker(Targetname, EnableDisable, Angles):
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
    def HullCheckMode(self):
        return self._raw_data.get('hullcheckmode', "0")



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
        return self._raw_data.get('nodeid', None)



class HintNode(Node):

    @property
    def hinttype(self):
        return self._raw_data.get('hinttype', "0")

    @property
    def generictype(self):
        return self._raw_data.get('generictype', "")

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
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 0))

    @property
    def IgnoreFacing(self):
        return self._raw_data.get('ignorefacing', "2")

    @property
    def MinimumState(self):
        return self._raw_data.get('minimumstate', "1")

    @property
    def MaximumState(self):
        return self._raw_data.get('maximumstate', "3")



class TriggerOnce(EnableDisable, Parentname, Global, Origin, Targetname):

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
        return self._raw_data.get('skyname', "sky_dust")

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



class worldspawn(ResponseContext, Targetname, worldbase):
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
    icon_sprite = "editor/env_texturetoggle.vmt"
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
    def glowDistanceScale(self):
        return parse_source_value(self._raw_data.get('glowdistancescale', 0.99))

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



class env_dof_controller(Targetname):
    icon_sprite = "editor/env_dof_controller.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def enabled(self):
        return self._raw_data.get('enabled', "0")

    @property
    def near_blur(self):
        return parse_source_value(self._raw_data.get('near_blur', 20))

    @property
    def near_focus(self):
        return parse_source_value(self._raw_data.get('near_focus', 100))

    @property
    def near_radius(self):
        return parse_source_value(self._raw_data.get('near_radius', 8))

    @property
    def far_blur(self):
        return parse_source_value(self._raw_data.get('far_blur', 1000))

    @property
    def far_focus(self):
        return parse_source_value(self._raw_data.get('far_focus', 500))

    @property
    def far_radius(self):
        return parse_source_value(self._raw_data.get('far_radius', 8))

    @property
    def focus_target(self):
        return self._raw_data.get('focus_target', "")

    @property
    def focus_range(self):
        return parse_source_value(self._raw_data.get('focus_range', 200))



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

    @property
    def ReverseFadeDuration(self):
        return parse_source_value(self._raw_data.get('reversefadeduration', 2))



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



class func_ladderendpoint(Targetname, Parentname, Angles):
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



class func_wall(Shadow, Targetname, RenderFields, Global):

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)



class func_clip_vphysics(Targetname, EnableDisable):

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)



class func_brush(RenderFields, Shadow, Inputfilter, Reflection, EnableDisable, Parentname, Global, Origin, Targetname, PaintableBrush):

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



class vgui_screen_base(Targetname, Parentname, Angles):

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

    @property
    def IsTransparent(self):
        return self._raw_data.get('istransparent', "0")



class vgui_screen(vgui_screen_base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class vgui_slideshow_display(Targetname, Parentname, Angles):
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



class vgui_movie_display(Targetname, Parentname, Angles):
    model_ = "models/editor/axis_helper_thick.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def displaytext(self):
        return self._raw_data.get('displaytext', "")

    @property
    def moviefilename(self):
        return self._raw_data.get('moviefilename', "media/")

    @property
    def groupname(self):
        return self._raw_data.get('groupname', "")

    @property
    def looping(self):
        return self._raw_data.get('looping', "0")

    @property
    def stretch(self):
        return self._raw_data.get('stretch', "0")

    @property
    def forcedslave(self):
        return self._raw_data.get('forcedslave', "0")

    @property
    def width(self):
        return parse_source_value(self._raw_data.get('width', 256))

    @property
    def height(self):
        return parse_source_value(self._raw_data.get('height', 128))



class cycler(RenderFields, Parentname, Targetname, Angles):
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
    def targetpoint(self):
        return self._raw_data.get('targetpoint', "0 0 0")

    @property
    def TouchType(self):
        return self._raw_data.get('touchtype', "0")

    @property
    def ClipStyle(self):
        return self._raw_data.get('clipstyle', "0")

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



class env_embers(Targetname, Parentname, Angles):

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



class env_fire(Targetname, Parentname, EnableDisable):
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



class env_fog_controller(Targetname, SystemLevelChoice, Angles):
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

    @property
    def ZoomFogScale(self):
        return parse_source_value(self._raw_data.get('zoomfogscale', 1))



class env_steam(Targetname, Parentname, Angles):
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



class env_tilt(Targetname, Parentname, Angles):
    model_ = "models/editor/axis_helper_thick.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 500))

    @property
    def duration(self):
        return parse_source_value(self._raw_data.get('duration', 1))

    @property
    def tilttime(self):
        return parse_source_value(self._raw_data.get('tilttime', 2.5))



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


class env_shooter(RenderFields, gibshooterbase):
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



class env_rotorshooter(RenderFields, gibshooterbase):
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



class env_soundscape(Targetname, Parentname, EnableDisable):
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


class env_spark(Targetname, Parentname, Angles):
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



class env_sprite(Targetname, RenderFields, Parentname, SystemLevelChoice):
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


class env_sprite_clientside(env_sprite, Angles):
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

    @property
    def fogmaxdensity(self):
        return parse_source_value(self._raw_data.get('fogmaxdensity', 1))

    @property
    def HDRColorScale(self):
        return parse_source_value(self._raw_data.get('hdrcolorscale', 1.0))



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



class point_worldtext(Targetname, Parentname, Angles):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def message(self):
        return self._raw_data.get('message', None)

    @property
    def textsize(self):
        return parse_source_value(self._raw_data.get('textsize', 10))

    @property
    def color(self):
        return parse_int_vector(self._raw_data.get('color', "255 255 255"))



class point_spotlight(RenderFields, SystemLevelChoice, Parentname, Targetname, Angles):
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



class point_entity_finder(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)

    @property
    def referencename(self):
        return self._raw_data.get('referencename', "")

    @property
    def Method(self):
        return self._raw_data.get('method', "0")



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


class info_target(Targetname, Parentname, Angles):
    icon_sprite = "editor/info_target.vmt"
    model_ = "models/editor/axis_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class info_particle_system(Targetname, Reflection, Parentname, Angles):
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



class phys_ragdollmagnet(Targetname, Parentname, Angles, EnableDisable):
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


class info_teleport_destination(Targetname, PlayerClass, Parentname, Angles):
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


class info_node_hint(HintNode, Targetname, Angles):
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



class info_node_air_hint(HintNode, Targetname, Angles):
    model_ = "models/editor/air_node_hint.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def nodeheight(self):
        return parse_source_value(self._raw_data.get('nodeheight', 0))



class info_hint(HintNode, Targetname, Angles):
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

    @property
    def preciseMovement(self):
        return self._raw_data.get('precisemovement', "0")

    @property
    def priority(self):
        return self._raw_data.get('priority', "0")



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

    @property
    def priority(self):
        return self._raw_data.get('priority', "0")



class info_radial_link_controller(Targetname, Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 120))



class info_node_climb(HintNode, Targetname, Angles):
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



class light_spot(Targetname, Light, Angles):
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



class light_dynamic(Targetname, Parentname, Angles):
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



class sunlight_shadow_control(Targetname, EnableDisable):
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
    def colortransitiontime(self):
        return parse_source_value(self._raw_data.get('colortransitiontime', 0.5))

    @property
    def distance(self):
        return parse_source_value(self._raw_data.get('distance', 10000))

    @property
    def fov(self):
        return parse_source_value(self._raw_data.get('fov', 5))

    @property
    def nearz(self):
        return parse_source_value(self._raw_data.get('nearz', 512))

    @property
    def northoffset(self):
        return parse_source_value(self._raw_data.get('northoffset', 200))

    @property
    def texturename(self):
        return self._raw_data.get('texturename', "effects/flashlight001")

    @property
    def enableshadows(self):
        return self._raw_data.get('enableshadows', "0")



class env_cascade_light(Targetname, EnableDisable):
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
        return self._raw_data.get('uselightenvangles', "1")



class env_ambient_light(Targetname, SpatialEntity, EnableDisable):
    icon_sprite = "editor/color_correction.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def color(self):
        return parse_int_vector(self._raw_data.get('color', "255 255 255"))



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



class func_movelinear(Origin, Targetname, RenderFields, Parentname):

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
    def WaveHeight(self):
        return self._raw_data.get('waveheight', "3.0")



class func_rotating(RenderFields, Shadow, Reflection, Parentname, Origin, Targetname, Angles):

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



class func_platrot(RenderFields, Shadow, BasePlat, Reflection, Parentname, Origin, Targetname, Angles):

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



class keyframe_track(Targetname, KeyFrame, Parentname, Angles):
    pass


class move_keyframed(Mover, Targetname, Parentname, KeyFrame):
    pass


class move_track(Mover, Targetname, Parentname, KeyFrame):

    @property
    def WheelBaseLength(self):
        return parse_source_value(self._raw_data.get('wheelbaselength', 50))

    @property
    def Damage(self):
        return parse_source_value(self._raw_data.get('damage', 0))

    @property
    def NoRotate(self):
        return self._raw_data.get('norotate', "0")



class RopeKeyFrame(SystemLevelChoice):

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



class keyframe_rope(Targetname, Parentname, RopeKeyFrame, KeyFrame):
    model_ = "models/editor/axis_helper_thick.mdl"
    pass


class move_rope(Targetname, Parentname, RopeKeyFrame, KeyFrame):
    model_ = "models/editor/axis_helper.mdl"

    @property
    def PositionInterpolator(self):
        return self._raw_data.get('positioninterpolator', "2")



class Button(Base):
    pass


class func_button(RenderFields, DamageFilter, Parentname, Origin, Targetname, Button):

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

    @property
    def min_use_angle(self):
        return parse_source_value(self._raw_data.get('min_use_angle', 0.8))



class func_rot_button(Button, EnableDisable, Parentname, Global, Origin, Targetname, Angles):

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



class momentary_rot_button(RenderFields, Parentname, Origin, Targetname, Angles):

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



class Door(RenderFields, Shadow, Reflection, Parentname, Global, Targetname):

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



class func_door_rotating(Origin, Door, Angles):

    @property
    def distance(self):
        return parse_source_value(self._raw_data.get('distance', 90))

    @property
    def solidbsp(self):
        return self._raw_data.get('solidbsp', "0")



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



class PropDoorRotating(Parentname, Global, Angles, BaseFadeProp, Targetname, Studiomodel):

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

    @property
    def glowdist(self):
        return parse_source_value(self._raw_data.get('glowdist', 1024))

    @property
    def glowenabled(self):
        return self._raw_data.get('glowenabled', "0")

    @property
    def glowcolor(self):
        return parse_int_vector(self._raw_data.get('glowcolor', "255 255 255"))



class prop_door_rotating(PropDoorRotating):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


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

    @property
    def AffectedByWind(self):
        return self._raw_data.get('affectedbywind', "1")



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



class env_dustpuff(Targetname, Parentname, Angles):
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



class env_particlescript(Targetname, Parentname, Angles):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/Ambient_citadel_paths.mdl")



class env_effectscript(Targetname, Parentname, Angles):
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



class point_viewcontrol(Targetname, Parentname, Angles):
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
    def trackspeed(self):
        return parse_source_value(self._raw_data.get('trackspeed', 40))

    @property
    def fov(self):
        return parse_source_value(self._raw_data.get('fov', 90))

    @property
    def fov_rate(self):
        return parse_source_value(self._raw_data.get('fov_rate', 1))

    @property
    def speed(self):
        return self._raw_data.get('speed', "0")

    @property
    def acceleration(self):
        return self._raw_data.get('acceleration', "500")

    @property
    def deceleration(self):
        return self._raw_data.get('deceleration', "500")



class point_viewcontrol_multiplayer(Targetname, Parentname, Angles):
    viewport_model = "models/editor/camera.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def fov(self):
        return parse_source_value(self._raw_data.get('fov', 90))

    @property
    def fov_rate(self):
        return parse_source_value(self._raw_data.get('fov_rate', 1.0))

    @property
    def target_entity(self):
        return self._raw_data.get('target_entity', "")

    @property
    def interp_time(self):
        return parse_source_value(self._raw_data.get('interp_time', 1.0))

    @property
    def target_team(self):
        return self._raw_data.get('target_team', "-1")



class point_viewproxy(Targetname, Parentname, Angles):
    viewport_model = "models/editor/camera.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def offsettype(self):
        return self._raw_data.get('offsettype', "0")

    @property
    def proxy(self):
        return self._raw_data.get('proxy', None)

    @property
    def proxyattachment(self):
        return self._raw_data.get('proxyattachment', None)

    @property
    def tiltfraction(self):
        return parse_source_value(self._raw_data.get('tiltfraction', 0.5))

    @property
    def usefakeacceleration(self):
        return self._raw_data.get('usefakeacceleration', "0")

    @property
    def skewaccelerationforward(self):
        return self._raw_data.get('skewaccelerationforward', "1")

    @property
    def accelerationscalar(self):
        return parse_source_value(self._raw_data.get('accelerationscalar', 1.0))

    @property
    def easeanglestocamera(self):
        return self._raw_data.get('easeanglestocamera', "0")



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


class logic_register_activator(Targetname, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class logic_random_outputs(Targetname, EnableDisable):
    icon_sprite = "editor/logic_random_outputs.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def OnTriggerChance1(self):
        return parse_source_value(self._raw_data.get('ontriggerchance1', 1.0))

    @property
    def OnTriggerChance2(self):
        return parse_source_value(self._raw_data.get('ontriggerchance2', 1.0))

    @property
    def OnTriggerChance3(self):
        return parse_source_value(self._raw_data.get('ontriggerchance3', 1.0))

    @property
    def OnTriggerChance4(self):
        return parse_source_value(self._raw_data.get('ontriggerchance4', 1.0))

    @property
    def OnTriggerChance5(self):
        return parse_source_value(self._raw_data.get('ontriggerchance5', 1.0))

    @property
    def OnTriggerChance6(self):
        return parse_source_value(self._raw_data.get('ontriggerchance6', 1.0))

    @property
    def OnTriggerChance7(self):
        return parse_source_value(self._raw_data.get('ontriggerchance7', 1.0))

    @property
    def OnTriggerChance8(self):
        return parse_source_value(self._raw_data.get('ontriggerchance8', 1.0))



class logic_script(Targetname):
    icon_sprite = "editor/logic_script.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def Group00(self):
        return self._raw_data.get('group00', None)

    @property
    def Group01(self):
        return self._raw_data.get('group01', None)

    @property
    def Group02(self):
        return self._raw_data.get('group02', None)

    @property
    def Group03(self):
        return self._raw_data.get('group03', None)

    @property
    def Group04(self):
        return self._raw_data.get('group04', None)

    @property
    def Group05(self):
        return self._raw_data.get('group05', None)

    @property
    def Group06(self):
        return self._raw_data.get('group06', None)

    @property
    def Group07(self):
        return self._raw_data.get('group07', None)

    @property
    def Group08(self):
        return self._raw_data.get('group08', None)

    @property
    def Group09(self):
        return self._raw_data.get('group09', None)

    @property
    def Group10(self):
        return self._raw_data.get('group10', None)

    @property
    def Group11(self):
        return self._raw_data.get('group11', None)

    @property
    def Group12(self):
        return self._raw_data.get('group12', None)

    @property
    def Group13(self):
        return self._raw_data.get('group13', None)

    @property
    def Group14(self):
        return self._raw_data.get('group14', None)

    @property
    def Group15(self):
        return self._raw_data.get('group15', None)



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



class env_microphone(Targetname, Parentname, EnableDisable):
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
    icon_sprite = "editor/logic_autosave.vmt"
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



class logic_playmovie(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def MovieFilename(self):
        return self._raw_data.get('moviefilename', "")

    @property
    def allowskip(self):
        return self._raw_data.get('allowskip', "0")

    @property
    def loopvideo(self):
        return self._raw_data.get('loopvideo', "0")

    @property
    def fadeintime(self):
        return parse_source_value(self._raw_data.get('fadeintime', 0))



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



class env_entity_maker(Targetname, Parentname, Angles):
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

    @property
    def Filter06(self):
        return self._raw_data.get('filter06', None)

    @property
    def Filter07(self):
        return self._raw_data.get('filter07', None)

    @property
    def Filter08(self):
        return self._raw_data.get('filter08', None)

    @property
    def Filter09(self):
        return self._raw_data.get('filter09', None)

    @property
    def Filter10(self):
        return self._raw_data.get('filter10', None)



class filter_activator_name(BaseFilter):
    icon_sprite = "editor/filter_name.vmt"

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)



class filter_activator_model(BaseFilter):
    icon_sprite = "editor/filter_name.vmt"

    @property
    def model(self):
        return self._raw_data.get('model', None)



class filter_activator_context(BaseFilter):
    icon_sprite = "editor/filter_name.vmt"

    @property
    def ResponseContext(self):
        return self._raw_data.get('responsecontext', None)



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



class point_anglesensor(Targetname, Parentname, EnableDisable):
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



class point_proximity_sensor(Targetname, Parentname, Angles, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)



class point_teleport(Targetname, Angles):
    model_ = "models/editor/axis_helper_thick.mdl"
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



class func_physbox(RenderFields, Origin, BreakableBrush):

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

    @property
    def ExploitableByPlayer(self):
        return self._raw_data.get('exploitablebyplayer', "0")



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



class phys_magnet(Targetname, Studiomodel, Parentname, Angles):
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



class prop_static_base(Shadow, Reflection, SystemLevelChoice, Angles):

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
    def renderamt(self):
        return parse_source_value(self._raw_data.get('renderamt', 255))

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "255 255 255"))

    @property
    def enablelightbounce(self):
        return self._raw_data.get('enablelightbounce', "0")

    @property
    def preventpropcombine(self):
        return self._raw_data.get('preventpropcombine', "0")

    @property
    def uniformscale(self):
        return parse_source_value(self._raw_data.get('uniformscale', 1))



class prop_dynamic_base(RenderFields, SystemLevelChoice, Parentname, Global, Angles, BaseFadeProp, BreakableProp, Studiomodel):

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
    def DisableBoneFollowers(self):
        return self._raw_data.get('disablebonefollowers', "0")

    @property
    def is_autoaim_target(self):
        return self._raw_data.get('is_autoaim_target', "0")

    @property
    def HoldAnimation(self):
        return self._raw_data.get('holdanimation', "0")

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
    pass


class prop_dynamic_override(prop_dynamic_base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 0))

    @property
    def AnimateEveryFrame(self):
        return self._raw_data.get('animateeveryframe', "0")



class BasePropPhysics(SystemLevelChoice, Global, Angles, BaseFadeProp, BreakableProp, Studiomodel):

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
    def addon(self):
        return self._raw_data.get('addon', "")



class prop_physics_override(BasePropPhysics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 0))



class prop_physics(RenderFields, BasePropPhysics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def ExploitableByPlayer(self):
        return self._raw_data.get('exploitablebyplayer', "0")



class prop_physics_multiplayer(prop_physics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def physicsmode(self):
        return self._raw_data.get('physicsmode', "0")



class prop_ragdoll(SystemLevelChoice, EnableDisable, Angles, BaseFadeProp, Targetname, Studiomodel):
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



class func_breakable(RenderFields, Origin, BreakableBrush):

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



class func_conveyor(Targetname, RenderFields, Parentname, Shadow):

    @property
    def movedir(self):
        return parse_float_vector(self._raw_data.get('movedir', "0 0 0"))

    @property
    def speed(self):
        return self._raw_data.get('speed', "100")

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)



class func_detail(SystemLevelChoice):
    pass


class func_viscluster(Base):
    pass


class func_illusionary(RenderFields, Shadow, Parentname, Origin, Targetname):

    @property
    def _minlight(self):
        return self._raw_data.get('_minlight', None)



class func_precipitation(Targetname, Parentname):

    @property
    def renderamt(self):
        return parse_source_value(self._raw_data.get('renderamt', 100))

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "100 100 100"))

    @property
    def preciptype(self):
        return self._raw_data.get('preciptype', "0")

    @property
    def minSpeed(self):
        return parse_source_value(self._raw_data.get('minspeed', 25))

    @property
    def maxSpeed(self):
        return parse_source_value(self._raw_data.get('maxspeed', 35))



class func_precipitation_blocker(Targetname, Parentname):
    pass


class func_detail_blocker(Targetname, Parentname):
    pass


class func_wall_toggle(func_wall):
    pass


class func_guntarget(Targetname, RenderFields, Parentname, Global):

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



class Trackchange(RenderFields, PlatSounds, Parentname, Global, Targetname):

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



class BaseTrain(RenderFields, Shadow, Reflection, Parentname, Global, Origin, Targetname):

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



class func_tracktrain(BaseTrain, PaintableBrush):
    pass


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



class path_track(Targetname, Parentname, Angles):
    model_ = "models/editor/cone_helper.mdl"
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


class trigger_soundoperator(Trigger):

    @property
    def sosvar(self):
        return self._raw_data.get('sosvar', "0")



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

    @property
    def OnlyFallingPlayers(self):
        return self._raw_data.get('onlyfallingplayers', "0")

    @property
    def FallingSpeedThreshold(self):
        return parse_source_value(self._raw_data.get('fallingspeedthreshold', -150))



class trigger_wind(Trigger, Angles):

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



class trigger_hierarchy(trigger_multiple):

    @property
    def childfiltername(self):
        return self._raw_data.get('childfiltername', None)



class trigger_impact(Origin, Targetname, Angles):

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

    @property
    def UseLandmarkAngles(self):
        return self._raw_data.get('uselandmarkangles', "0")

    @property
    def CheckDestIfClearForPlayer(self):
        return self._raw_data.get('checkdestifclearforplayer', "0")



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



class ai_addon(Targetname):

    @property
    def target(self):
        return self._raw_data.get('target', None)



class ai_addon_builder(Targetname, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def NPCName(self):
        return self._raw_data.get('npcname', "")

    @property
    def AddOnName(self):
        return self._raw_data.get('addonname', "")

    @property
    def NpcPoints(self):
        return parse_source_value(self._raw_data.get('npcpoints', 10))

    @property
    def AddonPoints(self):
        return parse_source_value(self._raw_data.get('addonpoints', 10))



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



class logic_eventlistener(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def EventName(self):
        return self._raw_data.get('eventname', "")

    @property
    def FetchEventData(self):
        return self._raw_data.get('fetcheventdata', "0")

    @property
    def IsEnabled(self):
        return self._raw_data.get('isenabled', "1")

    @property
    def TeamNum(self):
        return self._raw_data.get('teamnum', "-1")



class logic_eventlistener_itemequip(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def WeaponType(self):
        return self._raw_data.get('weapontype', "-1")

    @property
    def WeaponClassname(self):
        return self._raw_data.get('weaponclassname', "")

    @property
    def IsEnabled(self):
        return self._raw_data.get('isenabled', "1")

    @property
    def TeamNum(self):
        return self._raw_data.get('teamnum', "-1")



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



class env_projectedtexture(Targetname, Parentname, Angles):
    model_ = "models/editor/cone_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def style(self):
        return self._raw_data.get('style', "0")

    @property
    def pattern(self):
        return self._raw_data.get('pattern', "")

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
    def simpleprojection(self):
        return self._raw_data.get('simpleprojection', "0")

    @property
    def brightnessscale(self):
        return parse_source_value(self._raw_data.get('brightnessscale', 1.0))

    @property
    def lightcolor(self):
        return parse_int_vector(self._raw_data.get('lightcolor', "255 255 255 200"))

    @property
    def colortransitiontime(self):
        return parse_source_value(self._raw_data.get('colortransitiontime', 0.5))

    @property
    def cameraspace(self):
        return parse_source_value(self._raw_data.get('cameraspace', 0))

    @property
    def texturename(self):
        return self._raw_data.get('texturename', "effects/flashlight001")



class func_reflective_glass(func_brush):
    pass


class env_particle_performance_monitor(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class npc_puppet(Parentname, BaseNPC, Studiomodel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def animationtarget(self):
        return self._raw_data.get('animationtarget', "")

    @property
    def attachmentname(self):
        return self._raw_data.get('attachmentname', "")



class point_gamestats_counter(Origin, Targetname, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def Name(self):
        return self._raw_data.get('name', None)



class beam_spotlight(RenderFields, SystemLevelChoice, Parentname, Targetname, Angles):
    model_ = "models/editor/cone_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def maxspeed(self):
        return parse_source_value(self._raw_data.get('maxspeed', 100))

    @property
    def spotlightlength(self):
        return parse_source_value(self._raw_data.get('spotlightlength', 500))

    @property
    def spotlightwidth(self):
        return parse_source_value(self._raw_data.get('spotlightwidth', 50))

    @property
    def HDRColorScale(self):
        return parse_source_value(self._raw_data.get('hdrcolorscale', 0.7))



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
    icon_sprite = "editor/func_instance_parms.vmt"
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



class func_instance_origin(Base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class env_instructor_hint(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def hint_replace_key(self):
        return self._raw_data.get('hint_replace_key', None)

    @property
    def hint_target(self):
        return self._raw_data.get('hint_target', None)

    @property
    def hint_static(self):
        return self._raw_data.get('hint_static', "0")

    @property
    def hint_allow_nodraw_target(self):
        return self._raw_data.get('hint_allow_nodraw_target', "1")

    @property
    def hint_caption(self):
        return self._raw_data.get('hint_caption', None)

    @property
    def hint_activator_caption(self):
        return self._raw_data.get('hint_activator_caption', None)

    @property
    def hint_color(self):
        return parse_int_vector(self._raw_data.get('hint_color', "255 255 255"))

    @property
    def hint_forcecaption(self):
        return self._raw_data.get('hint_forcecaption', "0")

    @property
    def hint_icon_onscreen(self):
        return self._raw_data.get('hint_icon_onscreen', "icon_tip")

    @property
    def hint_icon_offscreen(self):
        return self._raw_data.get('hint_icon_offscreen', "icon_tip")

    @property
    def hint_nooffscreen(self):
        return self._raw_data.get('hint_nooffscreen', "0")

    @property
    def hint_binding(self):
        return self._raw_data.get('hint_binding', None)

    @property
    def hint_gamepad_binding(self):
        return self._raw_data.get('hint_gamepad_binding', None)

    @property
    def hint_icon_offset(self):
        return parse_source_value(self._raw_data.get('hint_icon_offset', 0))

    @property
    def hint_pulseoption(self):
        return self._raw_data.get('hint_pulseoption', "0")

    @property
    def hint_alphaoption(self):
        return self._raw_data.get('hint_alphaoption', "0")

    @property
    def hint_shakeoption(self):
        return self._raw_data.get('hint_shakeoption', "0")

    @property
    def hint_local_player_only(self):
        return self._raw_data.get('hint_local_player_only', "No")

    @property
    def hint_timeout(self):
        return parse_source_value(self._raw_data.get('hint_timeout', 0))

    @property
    def hint_range(self):
        return parse_source_value(self._raw_data.get('hint_range', 0))



class info_target_instructor_hint(Targetname, Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class info_game_event_proxy(Targetname, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def event_name(self):
        return self._raw_data.get('event_name', None)

    @property
    def range(self):
        return parse_source_value(self._raw_data.get('range', 512))



class prop_hallucination(Targetname, Parentname, Angles, Studiomodel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def EnabledChance(self):
        return parse_source_value(self._raw_data.get('enabledchance', 100.0))

    @property
    def VisibleTime(self):
        return parse_source_value(self._raw_data.get('visibletime', 0.215))

    @property
    def RechargeTime(self):
        return parse_source_value(self._raw_data.get('rechargetime', 0.0))



class light_directional(Angles):
    icon_sprite = "editor/light_directional.vmt"
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
    def _lightHDR(self):
        return parse_int_vector(self._raw_data.get('_lighthdr', "-1 -1 -1 1"))

    @property
    def _lightscaleHDR(self):
        return parse_source_value(self._raw_data.get('_lightscalehdr', 0.7))

    @property
    def SunSpreadAngle(self):
        return parse_source_value(self._raw_data.get('sunspreadangle', 0))



class postprocess_controller(Targetname):
    icon_sprite = "editor/postprocess_controller.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def fadetime(self):
        return parse_source_value(self._raw_data.get('fadetime', 2))

    @property
    def localcontraststrength(self):
        return parse_source_value(self._raw_data.get('localcontraststrength', 0))

    @property
    def localcontrastedgestrength(self):
        return parse_source_value(self._raw_data.get('localcontrastedgestrength', 0))

    @property
    def vignettestart(self):
        return parse_source_value(self._raw_data.get('vignettestart', 0.8))

    @property
    def vignetteend(self):
        return parse_source_value(self._raw_data.get('vignetteend', 1.1))

    @property
    def vignetteblurstrength(self):
        return parse_source_value(self._raw_data.get('vignetteblurstrength', 0))

    @property
    def fadetoblackstrength(self):
        return parse_source_value(self._raw_data.get('fadetoblackstrength', 0))

    @property
    def depthblurfocaldistance(self):
        return parse_source_value(self._raw_data.get('depthblurfocaldistance', 0))

    @property
    def depthblurstrength(self):
        return parse_source_value(self._raw_data.get('depthblurstrength', 0))

    @property
    def screenblurstrength(self):
        return parse_source_value(self._raw_data.get('screenblurstrength', 0))

    @property
    def filmgrainstrength(self):
        return parse_source_value(self._raw_data.get('filmgrainstrength', 0))



class fog_volume(Targetname, EnableDisable):

    @property
    def FogName(self):
        return self._raw_data.get('fogname', None)

    @property
    def PostProcessName(self):
        return self._raw_data.get('postprocessname', None)

    @property
    def ColorCorrectionName(self):
        return self._raw_data.get('colorcorrectionname', None)



class BaseTank(RenderFields, Shadow, Parentname, Global, Origin, Targetname, Angles):

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



class Target(Base):

    @property
    def target(self):
        return self._raw_data.get('target', None)



class TeamNum(Base):

    @property
    def TeamNum(self):
        return self._raw_data.get('teamnum', "0")



class Weapon(Targetname, RenderFields, Angles):

    @property
    def CanBePickedUp(self):
        return self._raw_data.get('canbepickedup', "1")



class filter_activator_team(BaseFilter):

    @property
    def filterteam(self):
        return self._raw_data.get('filterteam', "2")



class prop_dynamic_glow(prop_dynamic):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def glowdist(self):
        return parse_source_value(self._raw_data.get('glowdist', 1024))

    @property
    def glowenabled(self):
        return self._raw_data.get('glowenabled', "1")

    @property
    def glowcolor(self):
        return parse_int_vector(self._raw_data.get('glowcolor', "255 255 255"))

    @property
    def glowstyle(self):
        return self._raw_data.get('glowstyle', "0")



class info_player_counterterrorist(Targetname, Angles):
    model_ = "models/player/ctm_st6.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def priority(self):
        return parse_source_value(self._raw_data.get('priority', 0))

    @property
    def enabled(self):
        return self._raw_data.get('enabled', "1")



class info_player_terrorist(Targetname, Angles):
    model_ = "models/player/tm_phoenix.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def priority(self):
        return parse_source_value(self._raw_data.get('priority', 0))

    @property
    def enabled(self):
        return self._raw_data.get('enabled', "1")



class info_deathmatch_spawn(Targetname, Angles):
    model_ = "models/player/tm_pirate.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def enabled(self):
        return self._raw_data.get('enabled', "1")



class info_armsrace_counterterrorist(Targetname, Angles):
    model_ = "models/player/ctm_idf_variantb.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def priority(self):
        return parse_source_value(self._raw_data.get('priority', 0))

    @property
    def enabled(self):
        return self._raw_data.get('enabled', "1")



class info_armsrace_terrorist(Targetname, Angles):
    model_ = "models/player/tm_leet_varianta.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def priority(self):
        return parse_source_value(self._raw_data.get('priority', 0))

    @property
    def enabled(self):
        return self._raw_data.get('enabled', "1")



class info_enemy_terrorist_spawn(Targetname, Angles):
    model_ = "models/player/tm_phoenix.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def priority(self):
        return parse_source_value(self._raw_data.get('priority', 0))

    @property
    def enabled(self):
        return self._raw_data.get('enabled', "1")

    @property
    def model_to_use(self):
        return self._raw_data.get('model_to_use', "")

    @property
    def weapons_to_give(self):
        return self._raw_data.get('weapons_to_give', "glock")

    @property
    def armor_to_give(self):
        return self._raw_data.get('armor_to_give', "1")

    @property
    def start_asleep(self):
        return self._raw_data.get('start_asleep', "1")

    @property
    def default_behavior(self):
        return self._raw_data.get('default_behavior', "0")

    @property
    def bot_difficulty(self):
        return parse_source_value(self._raw_data.get('bot_difficulty', 0))

    @property
    def is_agressive(self):
        return self._raw_data.get('is_agressive', "0")

    @property
    def hide_radius(self):
        return parse_source_value(self._raw_data.get('hide_radius', 1000))

    @property
    def behavior_tree_file(self):
        return self._raw_data.get('behavior_tree_file', "")



class info_survival_crate_spawn(Targetname, Angles):
    model_ = "models/props_junk/wood_crate001a.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def bonuscrate(self):
        return self._raw_data.get('bonuscrate', "0")



class info_player_logo(Angles):
    model_ = "models/editor/playerstart.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class hostage_entity(Targetname, Angles):
    model_ = "models/Characters/hostage_04.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def HostageType(self):
        return self._raw_data.get('hostagetype', "0")

    @property
    def HostageSpawnRandomFactor(self):
        return parse_source_value(self._raw_data.get('hostagespawnrandomfactor', 1))

    @property
    def HostageSpawnExclusionGroup1(self):
        return self._raw_data.get('hostagespawnexclusiongroup1', "0")

    @property
    def HostageSpawnExclusionGroup2(self):
        return self._raw_data.get('hostagespawnexclusiongroup2', "0")

    @property
    def HostageSpawnExclusionGroup3(self):
        return self._raw_data.get('hostagespawnexclusiongroup3', "0")

    @property
    def HostageSpawnExclusionGroup4(self):
        return self._raw_data.get('hostagespawnexclusiongroup4', "0")

    @property
    def HostageSpawnExclusionGroup5(self):
        return self._raw_data.get('hostagespawnexclusiongroup5', "0")

    @property
    def HostageSpawnExclusionGroup6(self):
        return self._raw_data.get('hostagespawnexclusiongroup6', "0")

    @property
    def HostageSpawnExclusionGroup7(self):
        return self._raw_data.get('hostagespawnexclusiongroup7', "0")

    @property
    def HostageSpawnExclusionGroup8(self):
        return self._raw_data.get('hostagespawnexclusiongroup8', "0")

    @property
    def HostageSpawnExclusionGroup9(self):
        return self._raw_data.get('hostagespawnexclusiongroup9', "0")

    @property
    def HostageSpawnExclusionGroup10(self):
        return self._raw_data.get('hostagespawnexclusiongroup10', "0")

    @property
    def HostageSpawnExclusionGroup11(self):
        return self._raw_data.get('hostagespawnexclusiongroup11', "0")

    @property
    def HostageSpawnExclusionGroup12(self):
        return self._raw_data.get('hostagespawnexclusiongroup12', "0")

    @property
    def HostageSpawnExclusionGroup13(self):
        return self._raw_data.get('hostagespawnexclusiongroup13', "0")

    @property
    def HostageSpawnExclusionGroup14(self):
        return self._raw_data.get('hostagespawnexclusiongroup14', "0")

    @property
    def HostageSpawnExclusionGroup15(self):
        return self._raw_data.get('hostagespawnexclusiongroup15', "0")

    @property
    def HostageSpawnExclusionGroup16(self):
        return self._raw_data.get('hostagespawnexclusiongroup16', "0")

    @property
    def HostageSpawnExclusionGroup17(self):
        return self._raw_data.get('hostagespawnexclusiongroup17', "0")

    @property
    def HostageSpawnExclusionGroup18(self):
        return self._raw_data.get('hostagespawnexclusiongroup18', "0")

    @property
    def HostageSpawnExclusionGroup19(self):
        return self._raw_data.get('hostagespawnexclusiongroup19', "0")

    @property
    def HostageSpawnExclusionGroup20(self):
        return self._raw_data.get('hostagespawnexclusiongroup20', "0")

    @property
    def HostageSpawnExclusionGroup21(self):
        return self._raw_data.get('hostagespawnexclusiongroup21', "0")

    @property
    def HostageSpawnExclusionGroup22(self):
        return self._raw_data.get('hostagespawnexclusiongroup22', "0")

    @property
    def HostageSpawnExclusionGroup23(self):
        return self._raw_data.get('hostagespawnexclusiongroup23', "0")

    @property
    def HostageSpawnExclusionGroup24(self):
        return self._raw_data.get('hostagespawnexclusiongroup24', "0")

    @property
    def HostageSpawnExclusionGroup25(self):
        return self._raw_data.get('hostagespawnexclusiongroup25', "0")

    @property
    def HostageSpawnExclusionGroup26(self):
        return self._raw_data.get('hostagespawnexclusiongroup26', "0")

    @property
    def HostageSpawnExclusionGroup27(self):
        return self._raw_data.get('hostagespawnexclusiongroup27', "0")

    @property
    def HostageSpawnExclusionGroup28(self):
        return self._raw_data.get('hostagespawnexclusiongroup28', "0")

    @property
    def HostageSpawnExclusionGroup29(self):
        return self._raw_data.get('hostagespawnexclusiongroup29', "0")

    @property
    def HostageSpawnExclusionGroup30(self):
        return self._raw_data.get('hostagespawnexclusiongroup30', "0")



class info_hostage_spawn(hostage_entity):
    model_ = "models/Characters/hostage_04.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class pet_entity(Angles):
    model_ = "models/chicken/chicken.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class info_view_parameters(Angles):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def ViewMode(self):
        return self._raw_data.get('viewmode', "0")



class info_map_parameters(Targetname, Angles):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def buying(self):
        return self._raw_data.get('buying', "0")

    @property
    def bombradius(self):
        return parse_source_value(self._raw_data.get('bombradius', 500))

    @property
    def petpopulation(self):
        return parse_source_value(self._raw_data.get('petpopulation', 0))

    @property
    def usenormalspawnsfordm(self):
        return self._raw_data.get('usenormalspawnsfordm', "0")

    @property
    def disableautogenerateddmspawns(self):
        return self._raw_data.get('disableautogenerateddmspawns', "0")

    @property
    def botmaxvisiondistance(self):
        return parse_source_value(self._raw_data.get('botmaxvisiondistance', -1))



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



class weapon_ak47(Weapon):
    model_ = "models/weapons/w_rif_ak47_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_aug(Weapon):
    model_ = "models/weapons/w_rif_aug_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_awp(Weapon):
    model_ = "models/weapons/w_snip_awp_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_bizon(Weapon):
    model_ = "models/weapons/w_smg_bizon_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_cz75a(Weapon):
    model_ = "models/weapons/w_pist_cz_75_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_deagle(Weapon):
    model_ = "models/weapons/w_pist_deagle_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_decoy(Weapon):
    model_ = "models/weapons/w_eq_decoy_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_elite(Weapon):
    model_ = "models/weapons/w_pist_elite_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_famas(Weapon):
    model_ = "models/weapons/w_rif_famas_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_fiveseven(Weapon):
    model_ = "models/weapons/w_pist_fiveseven_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_flashbang(Weapon):
    model_ = "models/weapons/w_eq_flashbang_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_g3sg1(Weapon):
    model_ = "models/weapons/w_snip_g3sg1_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_galilar(Weapon):
    model_ = "models/weapons/w_rif_galilar_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_glock(Weapon):
    model_ = "models/weapons/w_pist_glock18_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_hegrenade(Weapon):
    model_ = "models/weapons/w_eq_fraggrenade_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_hkp2000(Weapon):
    model_ = "models/weapons/w_pist_hkp2000_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_incgrenade(Weapon):
    model_ = "models/weapons/w_eq_incendiarygrenade_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_knife(Weapon):
    model_ = "models/weapons/w_knife.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_m4a1(Weapon):
    model_ = "models/weapons/w_rif_m4a1_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_m4a1_silencer(Weapon):
    model_ = "models/weapons/w_rif_m4a1_s_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_m249(Weapon):
    model_ = "models/weapons/w_mach_m249_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_mac10(Weapon):
    model_ = "models/weapons/w_smg_mac10_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_mag7(Weapon):
    model_ = "models/weapons/w_shot_mag7_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_molotov(Weapon):
    model_ = "models/weapons/w_eq_molotov_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_mp7(Weapon):
    model_ = "models/weapons/w_smg_mp7_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_mp9(Weapon):
    model_ = "models/weapons/w_smg_mp9_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_negev(Weapon):
    model_ = "models/weapons/w_mach_negev_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_nova(Weapon):
    model_ = "models/weapons/w_shot_nova_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_p90(Weapon):
    model_ = "models/weapons/w_smg_p90_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_p250(Weapon):
    model_ = "models/weapons/w_pist_p250_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_revolver(Weapon):
    model_ = "models/weapons/w_pist_revolver_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_sawedoff(Weapon):
    model_ = "models/weapons/w_shot_sawedoff_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_sg556(Weapon):
    model_ = "models/weapons/w_rif_sg556_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_tagrenade(Weapon):
    model_ = "models/weapons/w_eq_sensorgrenade_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_smokegrenade(Weapon):
    model_ = "models/weapons/w_eq_smokegrenade_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_ssg08(Weapon):
    model_ = "models/weapons/w_snip_ssg08_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_tec9(Weapon):
    model_ = "models/weapons/w_pist_tec9_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_ump45(Weapon):
    model_ = "models/weapons/w_smg_ump45_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_usp_silencer(Weapon):
    model_ = "models/weapons/w_pist_223_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_xm1014(Weapon):
    model_ = "models/weapons/w_shot_xm1014_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_taser(Weapon):
    model_ = "models/weapons/w_eq_taser.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_defuser(Weapon):
    model_ = "models/weapons/w_defuser.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_c4(Weapon):
    model_ = "models/weapons/w_c4.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_healthshot(Weapon):
    model_ = "models/weapons/w_eq_healthshot_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_scar20(Weapon):
    model_ = "models/weapons/w_snip_scar20_dropped.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class weapon_zone_repulsor(Weapon):
    model_ = "models/props_survival/repulsor/repulsor.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class flashbang_projectile(Targetname, Angles):
    model_ = "models/weapons/w_eq_flashbang_thrown.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class hegrenade_projectile(TeamNum, Weapon):
    model_ = "models/weapons/w_eq_fraggrenade.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class molotov_projectile(TeamNum, Weapon):
    model_ = "models/weapons/w_eq_molotov.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class planted_c4_training(Targetname, Angles):
    model_ = "models/weapons/w_c4_planted.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


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



class npc_heardanger(Targetname, RenderFields, Parentname, Shadow):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', None)



class func_ladder(Base):
    pass


class func_physbox_multiplayer(func_physbox):
    pass


class func_buyzone(TeamNum, Targetname):
    pass


class func_bomb_target(Targetname, Parentname):

    @property
    def heistbomb(self):
        return self._raw_data.get('heistbomb', "0")

    @property
    def bomb_mount_target(self):
        return self._raw_data.get('bomb_mount_target', None)



class func_no_defuse(Targetname, Parentname):
    pass


class func_hostage_rescue(Trigger):
    pass


class point_surroundtest(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class Territory(Targetname, Angles):
    model_ = "models/props/cs_assault/consolepanelloadingbay.mdl"

    @property
    def respawn_area(self):
        return self._raw_data.get('respawn_area', None)

    @property
    def buyzone(self):
        return self._raw_data.get('buyzone', None)



class info_territory_control_hospital(Territory):
    model_ = "models/props/cs_assault/consolepanelloadingbay.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class info_territory_control_terrorist(Territory):
    model_ = "models/props/cs_assault/consolepanelloadingbay.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class info_territory_control_counterterrorist(Territory):
    model_ = "models/props/cs_assault/consolepanelloadingbay.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class info_territory_control_barracks(Territory):
    model_ = "models/props/cs_assault/consolepanelloadingbay.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class info_territory_control_bank(Territory):
    model_ = "models/props/cs_assault/consolepanelloadingbay.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class info_territory_control_encryption(Territory):
    model_ = "models/props/cs_assault/consolepanelloadingbay.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class info_territory_control_surveillance(Territory):
    model_ = "models/props/cs_assault/consolepanelloadingbay.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class info_gascanister_launchpoint(Targetname):
    icon_sprite = "editor/gibshooter.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class point_territory_control_rules(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def respawntime(self):
        return parse_source_value(self._raw_data.get('respawntime', 30))



class func_territory_respawn_area(Targetname):
    pass


class game_round_end(Targetname):
    icon_sprite = "editor/game_end.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class game_money(Targetname):
    icon_sprite = "editor/game_money.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def Money(self):
        return parse_source_value(self._raw_data.get('money', 1000))

    @property
    def AwardText(self):
        return self._raw_data.get('awardtext', None)



class BaseVehicle(Origin, Targetname, Global, prop_static_base):

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


class trigger_bomb_reset(Trigger):
    pass


class game_survival_logic(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class game_coopmission_manager(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


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
    def spread(self):
        return self._raw_data.get('spread', "5")

    @property
    def bias(self):
        return self._raw_data.get('bias', "1")

    @property
    def weaponname(self):
        return self._raw_data.get('weaponname', "weapon_ak47")



class point_hiding_spot(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class item_coop_coin(Targetname, Angles):
    model_ = "models/coop/challenge_coin.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class prop_exploding_barrel(Targetname, Angles):
    model_ = "models/props/coop_cementplant/exloding_barrel/exploding_barrel.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class trigger_softbarrier(Trigger):

    @property
    def pushdir(self):
        return parse_float_vector(self._raw_data.get('pushdir', "0 0 0"))



class func_nav_blocker(Origin, Targetname):

    @property
    def teamToBlock(self):
        return self._raw_data.get('teamtoblock', "-1")

    @property
    def affectsFlow(self):
        return self._raw_data.get('affectsflow', "0")



class point_give_ammo(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class skybox_swapper(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def SkyboxName(self):
        return self._raw_data.get('skyboxname', "sky_dust")



class point_dz_itemspawn(Targetname):
    pass


class point_dz_weaponspawn(point_dz_itemspawn, Angles):
    model_ = "models/props_survival/cases/case_heavy_weapon.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def groupID(self):
        return parse_source_value(self._raw_data.get('groupid', 0))

    @property
    def default_weight(self):
        return parse_source_value(self._raw_data.get('default_weight', 100))

    @property
    def door_name(self):
        return self._raw_data.get('door_name', "")



class point_dz_itemspawn_group(Targetname):

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 512))



class point_dz_weaponspawn_group(point_dz_itemspawn_group):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class dangerzone_controller(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def dz_enabled(self):
        return self._raw_data.get('dz_enabled', "1")

    @property
    def dz_suppressbombalert(self):
        return self._raw_data.get('dz_suppressbombalert', "1")

    @property
    def dz_missioncontrolled(self):
        return self._raw_data.get('dz_missioncontrolled', "1")



class trigger_survival_playarea(Trigger):
    pass


class trigger_survival_maproom(Trigger):
    pass


class trigger_survival_maproom_table(Trigger):
    pass


class point_dz_dronegun(Targetname, Angles):
    model_ = "models/props_survival/dronegun/dronegun.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def spawnAutomatically(self):
        return self._raw_data.get('spawnautomatically', "1")



class point_dz_parachute(Targetname, Angles):
    model_ = "models/props_survival/upgrades/parachutepack.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class dz_door(DamageFilter, PropDoorRotating):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def is_security_door(self):
        return self._raw_data.get('is_security_door', "0")



class prop_counter(Targetname, Parentname, Angles):
    model_ = "models/props_survival/counter/counter_a.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def initialValue(self):
        return parse_source_value(self._raw_data.get('initialvalue', 0))



class vgui_world_text_panel(Targetname, Parentname, Angles):
    model_ = "models/editor/axis_helper_thick.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def enabled(self):
        return self._raw_data.get('enabled', "1")

    @property
    def displaytext(self):
        return self._raw_data.get('displaytext', "")

    @property
    def displaytextoption(self):
        return self._raw_data.get('displaytextoption', "")

    @property
    def font(self):
        return self._raw_data.get('font', "DefaultLarge")

    @property
    def width(self):
        return parse_source_value(self._raw_data.get('width', 256))

    @property
    def height(self):
        return parse_source_value(self._raw_data.get('height', 128))

    @property
    def textpanelwidth(self):
        return parse_source_value(self._raw_data.get('textpanelwidth', 256))

    @property
    def textcolor(self):
        return parse_int_vector(self._raw_data.get('textcolor', "255 255 255"))



class func_survival_c4_target(Targetname, Angles):
    model_ = "models/props_survival/safe/safe.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 350))



class func_tablet_blocker(Targetname, Parentname):
    pass


class info_map_region(Base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 1200))

    @property
    def token(self):
        return self._raw_data.get('token', "")



class info_paradrop_denial(Base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 128))



class game_dz_coop_manager(Targetname, Parentname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def configFile(self):
        return self._raw_data.get('configfile', "")



class prop_ammo_box_generic(Targetname, Parentname):
    model_ = "models/props_survival/crates/crate_ammobox.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class prop_weapon_upgrade_armor(Targetname, Parentname):
    model_ = "models/props_survival/upgrades/upgrade_dz_armor.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class prop_weapon_refill_heavyarmor(Targetname, Parentname):
    model_ = "models/props_survival/upgrades/upgrade_heavy_armor.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def skin(self):
        return parse_source_value(self._raw_data.get('skin', 0))

    @property
    def body(self):
        return parse_source_value(self._raw_data.get('body', 0))



class prop_mapplaced_long_use_entity(Targetname, Parentname, Angles):
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
    def body(self):
        return parse_source_value(self._raw_data.get('body', 0))

    @property
    def sequence(self):
        return parse_source_value(self._raw_data.get('sequence', 0))

    @property
    def pingtype(self):
        return self._raw_data.get('pingtype', "0")

    @property
    def longuseactiontype(self):
        return self._raw_data.get('longuseactiontype', "0")

    @property
    def longuseduration(self):
        return parse_source_value(self._raw_data.get('longuseduration', 0))

    @property
    def soundstartuse(self):
        return self._raw_data.get('soundstartuse', "")



class weapon_mp5sd(Weapon):
    model_ = "models/weapons/w_smg_mp5sd.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class chicken(RenderFields, SystemLevelChoice, Parentname, Global, BaseFadeProp, BreakableProp, Angles):
    model_ = "models/chicken/chicken.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def Health(self):
        return parse_source_value(self._raw_data.get('health', 1))

    @property
    def minhealthdmg(self):
        return parse_source_value(self._raw_data.get('minhealthdmg', 1))

    @property
    def glowdist(self):
        return parse_source_value(self._raw_data.get('glowdist', 1024))

    @property
    def glowenabled(self):
        return self._raw_data.get('glowenabled', "0")

    @property
    def glowcolor(self):
        return parse_int_vector(self._raw_data.get('glowcolor', "255 255 255"))

    @property
    def glowstyle(self):
        return self._raw_data.get('glowstyle', "0")

    @property
    def skin(self):
        return parse_source_value(self._raw_data.get('skin', 0))

    @property
    def SetBodyGroup(self):
        return self._raw_data.get('setbodygroup', "0")



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



class mapvetopick_controller(Angles):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass



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
    'vgui_movie_display': vgui_movie_display,
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
    'env_tilt': env_tilt,
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
    'env_sprite_clientside': env_sprite_clientside,
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
    'point_worldtext': point_worldtext,
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
    'env_cascade_light': env_cascade_light,
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
    'BaseFadeProp': BaseFadeProp,
    'PropDoorRotating': PropDoorRotating,
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
    'trigger_soundoperator': trigger_soundoperator,
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
    'logic_eventlistener_itemequip': logic_eventlistener_itemequip,
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
    'prop_hallucination': prop_hallucination,
    'light_directional': light_directional,
    'postprocess_controller': postprocess_controller,
    'fog_volume': fog_volume,
    'BaseTank': BaseTank,
    'func_tank': func_tank,
    'Target': Target,
    'TeamNum': TeamNum,
    'Weapon': Weapon,
    'filter_activator_team': filter_activator_team,
    'prop_dynamic_glow': prop_dynamic_glow,
    'info_player_counterterrorist': info_player_counterterrorist,
    'info_player_terrorist': info_player_terrorist,
    'info_deathmatch_spawn': info_deathmatch_spawn,
    'info_armsrace_counterterrorist': info_armsrace_counterterrorist,
    'info_armsrace_terrorist': info_armsrace_terrorist,
    'info_enemy_terrorist_spawn': info_enemy_terrorist_spawn,
    'info_survival_crate_spawn': info_survival_crate_spawn,
    'info_player_logo': info_player_logo,
    'hostage_entity': hostage_entity,
    'info_hostage_spawn': info_hostage_spawn,
    'pet_entity': pet_entity,
    'info_view_parameters': info_view_parameters,
    'info_map_parameters': info_map_parameters,
    'env_detail_controller': env_detail_controller,
    'weapon_ak47': weapon_ak47,
    'weapon_aug': weapon_aug,
    'weapon_awp': weapon_awp,
    'weapon_bizon': weapon_bizon,
    'weapon_cz75a': weapon_cz75a,
    'weapon_deagle': weapon_deagle,
    'weapon_decoy': weapon_decoy,
    'weapon_elite': weapon_elite,
    'weapon_famas': weapon_famas,
    'weapon_fiveseven': weapon_fiveseven,
    'weapon_flashbang': weapon_flashbang,
    'weapon_g3sg1': weapon_g3sg1,
    'weapon_galilar': weapon_galilar,
    'weapon_glock': weapon_glock,
    'weapon_hegrenade': weapon_hegrenade,
    'weapon_hkp2000': weapon_hkp2000,
    'weapon_incgrenade': weapon_incgrenade,
    'weapon_knife': weapon_knife,
    'weapon_m4a1': weapon_m4a1,
    'weapon_m4a1_silencer': weapon_m4a1_silencer,
    'weapon_m249': weapon_m249,
    'weapon_mac10': weapon_mac10,
    'weapon_mag7': weapon_mag7,
    'weapon_molotov': weapon_molotov,
    'weapon_mp7': weapon_mp7,
    'weapon_mp9': weapon_mp9,
    'weapon_negev': weapon_negev,
    'weapon_nova': weapon_nova,
    'weapon_p90': weapon_p90,
    'weapon_p250': weapon_p250,
    'weapon_revolver': weapon_revolver,
    'weapon_sawedoff': weapon_sawedoff,
    'weapon_sg556': weapon_sg556,
    'weapon_tagrenade': weapon_tagrenade,
    'weapon_smokegrenade': weapon_smokegrenade,
    'weapon_ssg08': weapon_ssg08,
    'weapon_tec9': weapon_tec9,
    'weapon_ump45': weapon_ump45,
    'weapon_usp_silencer': weapon_usp_silencer,
    'weapon_xm1014': weapon_xm1014,
    'weapon_taser': weapon_taser,
    'item_defuser': item_defuser,
    'weapon_c4': weapon_c4,
    'weapon_healthshot': weapon_healthshot,
    'weapon_scar20': weapon_scar20,
    'weapon_zone_repulsor': weapon_zone_repulsor,
    'flashbang_projectile': flashbang_projectile,
    'hegrenade_projectile': hegrenade_projectile,
    'molotov_projectile': molotov_projectile,
    'planted_c4_training': planted_c4_training,
    'generic_actor': generic_actor,
    'npc_heardanger': npc_heardanger,
    'func_ladder': func_ladder,
    'func_physbox_multiplayer': func_physbox_multiplayer,
    'func_buyzone': func_buyzone,
    'func_bomb_target': func_bomb_target,
    'func_no_defuse': func_no_defuse,
    'func_hostage_rescue': func_hostage_rescue,
    'point_surroundtest': point_surroundtest,
    'Territory': Territory,
    'info_territory_control_hospital': info_territory_control_hospital,
    'info_territory_control_terrorist': info_territory_control_terrorist,
    'info_territory_control_counterterrorist': info_territory_control_counterterrorist,
    'info_territory_control_barracks': info_territory_control_barracks,
    'info_territory_control_bank': info_territory_control_bank,
    'info_territory_control_encryption': info_territory_control_encryption,
    'info_territory_control_surveillance': info_territory_control_surveillance,
    'info_gascanister_launchpoint': info_gascanister_launchpoint,
    'point_territory_control_rules': point_territory_control_rules,
    'func_territory_respawn_area': func_territory_respawn_area,
    'game_round_end': game_round_end,
    'game_money': game_money,
    'BaseVehicle': BaseVehicle,
    'BaseDriveableVehicle': BaseDriveableVehicle,
    'prop_vehicle': prop_vehicle,
    'prop_vehicle_driveable': prop_vehicle_driveable,
    'trigger_bomb_reset': trigger_bomb_reset,
    'game_survival_logic': game_survival_logic,
    'game_coopmission_manager': game_coopmission_manager,
    'env_gunfire': env_gunfire,
    'point_hiding_spot': point_hiding_spot,
    'item_coop_coin': item_coop_coin,
    'prop_exploding_barrel': prop_exploding_barrel,
    'trigger_softbarrier': trigger_softbarrier,
    'func_nav_blocker': func_nav_blocker,
    'point_give_ammo': point_give_ammo,
    'skybox_swapper': skybox_swapper,
    'point_dz_itemspawn': point_dz_itemspawn,
    'point_dz_weaponspawn': point_dz_weaponspawn,
    'point_dz_itemspawn_group': point_dz_itemspawn_group,
    'point_dz_weaponspawn_group': point_dz_weaponspawn_group,
    'dangerzone_controller': dangerzone_controller,
    'trigger_survival_playarea': trigger_survival_playarea,
    'trigger_survival_maproom': trigger_survival_maproom,
    'trigger_survival_maproom_table': trigger_survival_maproom_table,
    'point_dz_dronegun': point_dz_dronegun,
    'point_dz_parachute': point_dz_parachute,
    'dz_door': dz_door,
    'prop_counter': prop_counter,
    'vgui_world_text_panel': vgui_world_text_panel,
    'func_survival_c4_target': func_survival_c4_target,
    'func_tablet_blocker': func_tablet_blocker,
    'info_map_region': info_map_region,
    'info_paradrop_denial': info_paradrop_denial,
    'game_dz_coop_manager': game_dz_coop_manager,
    'prop_ammo_box_generic': prop_ammo_box_generic,
    'prop_weapon_upgrade_armor': prop_weapon_upgrade_armor,
    'prop_weapon_refill_heavyarmor': prop_weapon_refill_heavyarmor,
    'prop_mapplaced_long_use_entity': prop_mapplaced_long_use_entity,
    'weapon_mp5sd': weapon_mp5sd,
    'chicken': chicken,
    'point_camera': point_camera,
    'func_monitor': func_monitor,
    'mapvetopick_controller': mapvetopick_controller,
}