
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


class AlyxInteractable(Base):
    pass


class Angles(Base):

    @property
    def angles(self):
        return parse_float_vector(self._raw_data.get('angles', "0 0 0"))



class BaseClusteredLight(Base):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def _specularmode(self):
        return self._raw_data.get('_specularmode', "0")

    @property
    def _directmode(self):
        return self._raw_data.get('_directmode', "1")

    @property
    def _indirectmode(self):
        return self._raw_data.get('_indirectmode', "1")

    @property
    def _initialshadowsize(self):
        return self._raw_data.get('_initialshadowsize', "3")

    @property
    def nearz(self):
        return parse_source_value(self._raw_data.get('nearz', 4.0))



class BaseEffectBrush(Base):

    @property
    def targetname(self):
        return self._raw_data.get('targetname', None)

    @property
    def globalname(self):
        return self._raw_data.get('globalname', None)

    @property
    def parentname(self):
        return self._raw_data.get('parentname', None)

    @property
    def vscripts(self):
        return self._raw_data.get('vscripts', None)

    @property
    def thinkfunction(self):
        return self._raw_data.get('thinkfunction', None)

    @property
    def linedivider_base(self):
        return self._raw_data.get('linedivider_base', None)



class BaseEntityInputs(Base):
    pass


class BaseEntityOutputs(Base):
    pass


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



class BaseLight(Base):

    @property
    def _light(self):
        return parse_int_vector(self._raw_data.get('_light', "255 255 255 200"))

    @property
    def _lighthdr(self):
        return parse_int_vector(self._raw_data.get('_lighthdr', "-1 -1 -1 1"))

    @property
    def _lightscalehdr(self):
        return parse_source_value(self._raw_data.get('_lightscalehdr', 1))

    @property
    def style(self):
        return self._raw_data.get('style', "0")

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def pattern(self):
        return self._raw_data.get('pattern', None)

    @property
    def fadetickinterval(self):
        return parse_source_value(self._raw_data.get('fadetickinterval', 0.1))

    @property
    def _castentityshadow(self):
        return self._raw_data.get('_castentityshadow', "1")

    @property
    def _shadoworiginoffset(self):
        return parse_float_vector(self._raw_data.get('_shadoworiginoffset', "0 0 0"))

    @property
    def _nocubemapsprite(self):
        return self._raw_data.get('_nocubemapsprite', "1")



class BaseLightFalloff(Base):

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

    @property
    def _distance(self):
        return parse_source_value(self._raw_data.get('_distance', 0))

    @property
    def _hard_radius_threshold(self):
        return parse_source_value(self._raw_data.get('_hard_radius_threshold', 32))

    @property
    def _hard_radius_override(self):
        return parse_source_value(self._raw_data.get('_hard_radius_override', 0))



class BasePaintType(Base):

    @property
    def painttype(self):
        return self._raw_data.get('painttype', "0")

    @property
    def skin(self):
        return self._raw_data.get('skin', "0")



class CombineScanner(Base):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

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
    def shouldinspect(self):
        return self._raw_data.get('shouldinspect', "1")

    @property
    def onlyinspectplayers(self):
        return self._raw_data.get('onlyinspectplayers', "0")

    @property
    def neverinspectplayers(self):
        return self._raw_data.get('neverinspectplayers', "0")



class ControlEnables(Base):

    @property
    def ctrl_type(self):
        return self._raw_data.get('ctrl_type', "0")

    @property
    def ctrl_value(self):
        return self._raw_data.get('ctrl_value', "1")



class DamageFilter(Base):

    @property
    def damagefilter(self):
        return self._raw_data.get('damagefilter', None)



class DamageType(Base):

    @property
    def damagetype(self):
        return self._raw_data.get('damagetype', "0")

    @property
    def damageor1(self):
        return self._raw_data.get('damageor1', "0")

    @property
    def damageor2(self):
        return self._raw_data.get('damageor2', "0")

    @property
    def damageor3(self):
        return self._raw_data.get('damageor3', "0")

    @property
    def damageor4(self):
        return self._raw_data.get('damageor4', "0")



class DetailPropBase(Base):

    @property
    def detailorientation(self):
        return self._raw_data.get('detailorientation', "0")



class EnableDisable(Base):

    @property
    def startdisabled(self):
        return self._raw_data.get('startdisabled', "0")



class FadeDistance(Base):

    @property
    def fademindist(self):
        return parse_source_value(self._raw_data.get('fademindist', -1))

    @property
    def fademaxdist(self):
        return parse_source_value(self._raw_data.get('fademaxdist', 0))

    @property
    def fadescale(self):
        return parse_source_value(self._raw_data.get('fadescale', 1))



class GrenadeUser(Base):

    @property
    def numgrenades(self):
        return self._raw_data.get('numgrenades', "0")



class KeyFrame(Base):

    @property
    def nextkey(self):
        return self._raw_data.get('nextkey', None)

    @property
    def movespeed(self):
        return parse_source_value(self._raw_data.get('movespeed', 64))



class LinkedPortalDoor(Base):
    pass


class MasterEnt(Base):

    @property
    def master(self):
        return self._raw_data.get('master', None)



class Mover(Base):

    @property
    def positioninterpolator(self):
        return self._raw_data.get('positioninterpolator', "0")



class Node(Base):

    @property
    def nodeid(self):
        return self._raw_data.get('nodeid', None)



class Origin(Base):

    @property
    def origin(self):
        return parse_float_vector(self._raw_data.get('origin', None))



class PaintableProp(Base):
    pass


class PortalBase(Base):

    @property
    def activated(self):
        return self._raw_data.get('activated', "0")

    @property
    def portaltwo(self):
        return self._raw_data.get('portaltwo', "0")

    @property
    def halfwidth(self):
        return parse_source_value(self._raw_data.get('halfwidth', 0))

    @property
    def halfheight(self):
        return parse_source_value(self._raw_data.get('halfheight', 0))



class Reflection(Base):

    @property
    def drawinfastreflection(self):
        return self._raw_data.get('drawinfastreflection', "0")



class RenderFields(Base):

    @property
    def rendermode(self):
        return self._raw_data.get('rendermode', "0")

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "255 255 255"))

    @property
    def renderamt(self):
        return parse_source_value(self._raw_data.get('renderamt', 255))

    @property
    def renderfx(self):
        return self._raw_data.get('renderfx', "0")

    @property
    def disablereceiveshadows(self):
        return self._raw_data.get('disablereceiveshadows', "0")

    @property
    def viewhideflags(self):
        return self._raw_data.get('viewhideflags', "0")



class ResponseContext(Base):

    @property
    def responsecontext(self):
        return self._raw_data.get('responsecontext', None)



class SRCIndicator(Base):

    @property
    def indicatorname(self):
        return self._raw_data.get('indicatorname', None)



class SRCModel(Base):

    @property
    def comp_custom_model_type(self):
        return self._raw_data.get('comp_custom_model_type', "0")



class SetSkin(Base):

    @property
    def skin(self):
        return parse_source_value(self._raw_data.get('skin', 0))

    @property
    def skinset(self):
        return self._raw_data.get('skinset', None)



class StaticTargetName(Base):

    @property
    def targetname(self):
        return self._raw_data.get('targetname', None)



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



class TeamNum(Base):

    @property
    def teamnum(self):
        return self._raw_data.get('teamnum', "0")



class Toggle(Base):
    pass


class ToggleDraw(Base):
    pass


class _Breakable(Base):

    @property
    def explodedamage(self):
        return parse_source_value(self._raw_data.get('explodedamage', 0))

    @property
    def exploderadius(self):
        return parse_source_value(self._raw_data.get('exploderadius', 0))

    @property
    def explodemagnitude(self):
        return parse_source_value(self._raw_data.get('explodemagnitude', 0))

    @property
    def performancemode(self):
        return self._raw_data.get('performancemode', "0")

    @property
    def pressuredelay(self):
        return parse_source_value(self._raw_data.get('pressuredelay', 0))

    @property
    def minhealthdmg(self):
        return parse_source_value(self._raw_data.get('minhealthdmg', 0))

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 0))

    @property
    def physdamagescale(self):
        return parse_source_value(self._raw_data.get('physdamagescale', 1.0))



class comp_entity_finder(Base):
    icon_sprite = "editor/comp_entity_finder"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def targetname(self):
        return self._raw_data.get('targetname', "<pack_rename>")

    @property
    def targetcls(self):
        return self._raw_data.get('targetcls', None)

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 64))

    @property
    def searchfov(self):
        return parse_source_value(self._raw_data.get('searchfov', 180))

    @property
    def angles(self):
        return parse_float_vector(self._raw_data.get('angles', "0 0 0"))

    @property
    def targetref(self):
        return self._raw_data.get('targetref', None)

    @property
    def blacklist(self):
        return self._raw_data.get('blacklist', None)

    @property
    def teleporttarget(self):
        return self._raw_data.get('teleporttarget', "0")

    @property
    def rotatetarget(self):
        return self._raw_data.get('rotatetarget', "0")

    @property
    def makeunique(self):
        return self._raw_data.get('makeunique', "0")

    @property
    def sep1(self):
        return self._raw_data.get('sep1', None)

    @property
    def kv1_mode(self):
        return self._raw_data.get('kv1_mode', None)

    @property
    def kv1_known(self):
        return self._raw_data.get('kv1_known', None)

    @property
    def kv1_src(self):
        return self._raw_data.get('kv1_src', None)

    @property
    def kv1_dest(self):
        return self._raw_data.get('kv1_dest', None)

    @property
    def sep2(self):
        return self._raw_data.get('sep2', None)

    @property
    def kv2_mode(self):
        return self._raw_data.get('kv2_mode', None)

    @property
    def kv2_known(self):
        return self._raw_data.get('kv2_known', None)

    @property
    def kv2_src(self):
        return self._raw_data.get('kv2_src', None)

    @property
    def kv2_dest(self):
        return self._raw_data.get('kv2_dest', None)

    @property
    def sep3(self):
        return self._raw_data.get('sep3', None)

    @property
    def kv3_mode(self):
        return self._raw_data.get('kv3_mode', None)

    @property
    def kv3_known(self):
        return self._raw_data.get('kv3_known', None)

    @property
    def kv3_src(self):
        return self._raw_data.get('kv3_src', None)

    @property
    def kv3_dest(self):
        return self._raw_data.get('kv3_dest', None)

    @property
    def sep4(self):
        return self._raw_data.get('sep4', None)

    @property
    def kv4_mode(self):
        return self._raw_data.get('kv4_mode', None)

    @property
    def kv4_known(self):
        return self._raw_data.get('kv4_known', None)

    @property
    def kv4_src(self):
        return self._raw_data.get('kv4_src', None)

    @property
    def kv4_dest(self):
        return self._raw_data.get('kv4_dest', None)

    @property
    def sep5(self):
        return self._raw_data.get('sep5', None)

    @property
    def kv5_mode(self):
        return self._raw_data.get('kv5_mode', None)

    @property
    def kv5_known(self):
        return self._raw_data.get('kv5_known', None)

    @property
    def kv5_src(self):
        return self._raw_data.get('kv5_src', None)

    @property
    def kv5_dest(self):
        return self._raw_data.get('kv5_dest', None)



class comp_entity_mover(Base):
    icon_sprite = "editor/comp_entity_mover"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def reference(self):
        return self._raw_data.get('reference', None)

    @property
    def direction(self):
        return parse_float_vector(self._raw_data.get('direction', "0 0 0"))

    @property
    def distance(self):
        return parse_source_value(self._raw_data.get('distance', 1))



class comp_player_input_helper(Base):
    icon_sprite = "editor/comp_player_input_helper.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def targetname(self):
        return self._raw_data.get('targetname', "!player")



class comp_propcombine_volume(Base):

    @property
    def name(self):
        return self._raw_data.get('name', None)

    @property
    def prop(self):
        return self._raw_data.get('prop', None)

    @property
    def skin(self):
        return parse_source_value(self._raw_data.get('skin', 0))



class comp_vactube_object(Base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', None)

    @property
    def skin(self):
        return parse_source_value(self._raw_data.get('skin', None))

    @property
    def offset(self):
        return self._raw_data.get('offset', None)

    @property
    def weight(self):
        return parse_source_value(self._raw_data.get('weight', 1))

    @property
    def group(self):
        return self._raw_data.get('group', None)

    @property
    def tv_skin(self):
        return self._raw_data.get('tv_skin', "0")

    @property
    def cube_model(self):
        return self._raw_data.get('cube_model', None)

    @property
    def cube_skin(self):
        return parse_source_value(self._raw_data.get('cube_skin', 0))



class env_cubemap(Base):
    icon_sprite = "editor/env_cubemap.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def cubemapsize(self):
        return self._raw_data.get('cubemapsize', "0")

    @property
    def sides(self):
        return self._raw_data.get('sides', None)

    @property
    def parallaxobb(self):
        return self._raw_data.get('parallaxobb', None)



class func_detail(Base):
    pass


class func_detail_blocker(Base):
    pass


class func_fish_pool(Base):
    icon_sprite = "editor/ficool2/func_fish_pool"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/Junkola.mdl")

    @property
    def fish_count(self):
        return parse_source_value(self._raw_data.get('fish_count', 10))

    @property
    def max_range(self):
        return parse_source_value(self._raw_data.get('max_range', 150))



class func_instance_io_proxy(Base):
    icon_sprite = "editor/func_instance_io_proxy.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def targetname(self):
        return self._raw_data.get('targetname', "proxy")



class func_instance_origin(Base):
    icon_sprite = "editor/func_instance_origin.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class func_instance_parms(Base):
    icon_sprite = "editor/func_instance_parms.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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

    @property
    def parm11(self):
        return self._raw_data.get('parm11', None)

    @property
    def parm12(self):
        return self._raw_data.get('parm12', None)

    @property
    def parm13(self):
        return self._raw_data.get('parm13', None)

    @property
    def parm14(self):
        return self._raw_data.get('parm14', None)

    @property
    def parm15(self):
        return self._raw_data.get('parm15', None)

    @property
    def parm16(self):
        return self._raw_data.get('parm16', None)

    @property
    def parm17(self):
        return self._raw_data.get('parm17', None)

    @property
    def parm18(self):
        return self._raw_data.get('parm18', None)

    @property
    def parm19(self):
        return self._raw_data.get('parm19', None)

    @property
    def parm20(self):
        return self._raw_data.get('parm20', None)

    @property
    def parm21(self):
        return self._raw_data.get('parm21', None)

    @property
    def parm22(self):
        return self._raw_data.get('parm22', None)

    @property
    def parm23(self):
        return self._raw_data.get('parm23', None)

    @property
    def parm24(self):
        return self._raw_data.get('parm24', None)

    @property
    def parm25(self):
        return self._raw_data.get('parm25', None)

    @property
    def parm26(self):
        return self._raw_data.get('parm26', None)

    @property
    def parm27(self):
        return self._raw_data.get('parm27', None)

    @property
    def parm28(self):
        return self._raw_data.get('parm28', None)

    @property
    def parm29(self):
        return self._raw_data.get('parm29', None)

    @property
    def parm30(self):
        return self._raw_data.get('parm30', None)



class func_ladder(Base):
    pass


class func_viscluster(Base):
    pass


class hammer_notes(Base):
    icon_sprite = "editor/ts_book.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def message(self):
        return self._raw_data.get('message', None)

    @property
    def textsize(self):
        return parse_source_value(self._raw_data.get('textsize', 10))

    @property
    def color(self):
        return parse_int_vector(self._raw_data.get('color', "255 255 255"))

    @property
    def linename1(self):
        return self._raw_data.get('linename1', None)

    @property
    def linename2(self):
        return self._raw_data.get('linename2', None)

    @property
    def linename3(self):
        return self._raw_data.get('linename3', None)

    @property
    def linename4(self):
        return self._raw_data.get('linename4', None)

    @property
    def mat(self):
        return self._raw_data.get('mat', None)

    @property
    def part(self):
        return self._raw_data.get('part', None)

    @property
    def model(self):
        return self._raw_data.get('model', None)

    @property
    def sound(self):
        return self._raw_data.get('sound', None)



class info_intermission(Base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)



class info_mass_center(Base):
    icon_sprite = "editor/ficool2/info_mass_center.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)



class info_no_dynamic_shadow(Base):
    icon_sprite = "editor/info_no_dynamic_shadow.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def sides(self):
        return self._raw_data.get('sides', None)



class info_overlay_transition(Base):
    model = "models/editor/overlay_helper_box.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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
    def lengthtexcoordstart(self):
        return parse_source_value(self._raw_data.get('lengthtexcoordstart', 0.0))

    @property
    def lengthtexcoordend(self):
        return parse_source_value(self._raw_data.get('lengthtexcoordend', 1.0))

    @property
    def widthtexcoordstart(self):
        return parse_source_value(self._raw_data.get('widthtexcoordstart', 0.0))

    @property
    def widthtexcoordend(self):
        return parse_source_value(self._raw_data.get('widthtexcoordend', 1.0))

    @property
    def width1(self):
        return parse_source_value(self._raw_data.get('width1', 25.0))

    @property
    def width2(self):
        return parse_source_value(self._raw_data.get('width2', 25.0))

    @property
    def debugdraw(self):
        return self._raw_data.get('debugdraw', "0")



class parallax_obb(Base):

    @property
    def targetname(self):
        return self._raw_data.get('targetname', None)



class BaseClusteredDynLight(BaseClusteredLight):

    @property
    def _volumetricmode(self):
        return self._raw_data.get('_volumetricmode', "2")

    @property
    def slopescale(self):
        return parse_source_value(self._raw_data.get('slopescale', 1.0))

    @property
    def texturename(self):
        return self._raw_data.get('texturename', None)

    @property
    def textureframe(self):
        return parse_source_value(self._raw_data.get('textureframe', 0))

    @property
    def volumetric_lightscale(self):
        return parse_source_value(self._raw_data.get('volumetric_lightscale', 1.0))

    @property
    def volumetric_density(self):
        return parse_source_value(self._raw_data.get('volumetric_density', 0.0))



class BaseDustParticleSpawner(BaseEffectBrush):

    @property
    def startdisabled(self):
        return self._raw_data.get('startdisabled', "0")

    @property
    def color(self):
        return parse_int_vector(self._raw_data.get('color', "255 255 255"))

    @property
    def color2(self):
        return parse_int_vector(self._raw_data.get('color2', "255 255 255"))

    @property
    def spawnrate(self):
        return parse_source_value(self._raw_data.get('spawnrate', 40))

    @property
    def speedmax(self):
        return parse_source_value(self._raw_data.get('speedmax', 13))

    @property
    def fallspeed(self):
        return parse_source_value(self._raw_data.get('fallspeed', 0))

    @property
    def falldirection(self):
        return parse_float_vector(self._raw_data.get('falldirection', "0 0 0"))

    @property
    def lifetimemin(self):
        return parse_source_value(self._raw_data.get('lifetimemin', 3))

    @property
    def lifetimemax(self):
        return parse_source_value(self._raw_data.get('lifetimemax', 5))

    @property
    def distmax(self):
        return parse_source_value(self._raw_data.get('distmax', 1024))

    @property
    def frozen(self):
        return self._raw_data.get('frozen', "0")

    @property
    def affectedbywind(self):
        return self._raw_data.get('affectedbywind', "1")

    @property
    def origin(self):
        return self._raw_data.get('origin', "0 0 0")



class BaseEntityIO(BaseEntityOutputs, BaseEntityInputs):
    pass


class BreakableProp(DamageFilter, _Breakable):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class Button(Angles):

    @property
    def sounds(self):
        return self._raw_data.get('sounds', "0")

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



class HintNode(Node):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def hinttype(self):
        return self._raw_data.get('hinttype', "0")

    @property
    def hintactivity(self):
        return self._raw_data.get('hintactivity', None)

    @property
    def nodefov(self):
        return self._raw_data.get('nodefov', "180")

    @property
    def starthintdisabled(self):
        return self._raw_data.get('starthintdisabled', "0")

    @property
    def group(self):
        return self._raw_data.get('group', None)

    @property
    def targetnode(self):
        return parse_source_value(self._raw_data.get('targetnode', -1))

    @property
    def ignorefacing(self):
        return self._raw_data.get('ignorefacing', "2")

    @property
    def minimumstate(self):
        return self._raw_data.get('minimumstate', "1")

    @property
    def maximumstate(self):
        return self._raw_data.get('maximumstate', "3")

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 0))



class RopeKeyFrame(SystemLevelChoice):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def nextkey(self):
        return self._raw_data.get('nextkey', None)

    @property
    def slack(self):
        return parse_source_value(self._raw_data.get('slack', 25))

    @property
    def type(self):
        return self._raw_data.get('type', "0")

    @property
    def subdiv(self):
        return parse_source_value(self._raw_data.get('subdiv', 2))

    @property
    def barbed(self):
        return self._raw_data.get('barbed', "0")

    @property
    def width(self):
        return parse_source_value(self._raw_data.get('width', 2))

    @property
    def texturescale(self):
        return parse_source_value(self._raw_data.get('texturescale', 1))

    @property
    def collide(self):
        return self._raw_data.get('collide', "0")

    @property
    def dangling(self):
        return self._raw_data.get('dangling', "0")

    @property
    def breakable(self):
        return self._raw_data.get('breakable', "0")

    @property
    def ropematerial(self):
        return self._raw_data.get('ropematerial', "cable/cable.vmt")

    @property
    def usewind(self):
        return self._raw_data.get('usewind', "0")

    @property
    def movespeed(self):
        return parse_source_value(self._raw_data.get('movespeed', 64))

    @property
    def positioninterpolator(self):
        return parse_source_value(self._raw_data.get('positioninterpolator', 2))



class SetModel(SetSkin):

    @property
    def model(self):
        return self._raw_data.get('model', None)



class comp_adv_output(ControlEnables):
    icon_sprite = "editor/comp_adv_output"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def out_ent(self):
        return self._raw_data.get('out_ent', None)

    @property
    def out_name(self):
        return self._raw_data.get('out_name', None)

    @property
    def target_global(self):
        return self._raw_data.get('target_global', None)

    @property
    def target_local(self):
        return self._raw_data.get('target_local', None)

    @property
    def target_instname(self):
        return self._raw_data.get('target_instname', None)

    @property
    def inp_name(self):
        return self._raw_data.get('inp_name', None)

    @property
    def delay(self):
        return parse_source_value(self._raw_data.get('delay', 0.0))

    @property
    def delay_max(self):
        return parse_source_value(self._raw_data.get('delay_max', None))

    @property
    def delay2(self):
        return parse_source_value(self._raw_data.get('delay2', 0.0))

    @property
    def times(self):
        return parse_source_value(self._raw_data.get('times', -1))

    @property
    def params_fmt(self):
        return self._raw_data.get('params_fmt', "{1}")

    @property
    def params_mode1(self):
        return self._raw_data.get('params_mode1', "legacy")

    @property
    def params_global1(self):
        return self._raw_data.get('params_global1', None)

    @property
    def params_local1(self):
        return self._raw_data.get('params_local1', None)

    @property
    def params_pos1(self):
        return parse_float_vector(self._raw_data.get('params_pos1', None))

    @property
    def params_mode2(self):
        return self._raw_data.get('params_mode2', "legacy")

    @property
    def params_global2(self):
        return self._raw_data.get('params_global2', None)

    @property
    def params_local2(self):
        return self._raw_data.get('params_local2', None)

    @property
    def params_pos2(self):
        return parse_float_vector(self._raw_data.get('params_pos2', None))

    @property
    def params_mode3(self):
        return self._raw_data.get('params_mode3', "legacy")

    @property
    def params_global3(self):
        return self._raw_data.get('params_global3', None)

    @property
    def params_local3(self):
        return self._raw_data.get('params_local3', None)

    @property
    def params_pos3(self):
        return parse_float_vector(self._raw_data.get('params_pos3', None))

    @property
    def params_mode4(self):
        return self._raw_data.get('params_mode4', "legacy")

    @property
    def params_global4(self):
        return self._raw_data.get('params_global4', None)

    @property
    def params_local4(self):
        return self._raw_data.get('params_local4', None)

    @property
    def params_pos4(self):
        return parse_float_vector(self._raw_data.get('params_pos4', None))

    @property
    def params_mode5(self):
        return self._raw_data.get('params_mode5', "legacy")

    @property
    def params_global5(self):
        return self._raw_data.get('params_global5', None)

    @property
    def params_local5(self):
        return self._raw_data.get('params_local5', None)

    @property
    def params_pos5(self):
        return parse_float_vector(self._raw_data.get('params_pos5', None))



class comp_case(ControlEnables, StaticTargetName):
    icon_sprite = "editor/comp_case"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def multiplecasesallowed(self):
        return self._raw_data.get('multiplecasesallowed', "0")

    @property
    def value(self):
        return self._raw_data.get('value', None)

    @property
    def mode(self):
        return self._raw_data.get('mode', "casefold")

    @property
    def seed(self):
        return self._raw_data.get('seed', None)

    @property
    def misschance(self):
        return parse_source_value(self._raw_data.get('misschance', 0))

    @property
    def case01(self):
        return self._raw_data.get('case01', None)

    @property
    def case02(self):
        return self._raw_data.get('case02', None)

    @property
    def case03(self):
        return self._raw_data.get('case03', None)

    @property
    def case04(self):
        return self._raw_data.get('case04', None)

    @property
    def case05(self):
        return self._raw_data.get('case05', None)

    @property
    def case06(self):
        return self._raw_data.get('case06', None)

    @property
    def case07(self):
        return self._raw_data.get('case07', None)

    @property
    def case08(self):
        return self._raw_data.get('case08', None)

    @property
    def case09(self):
        return self._raw_data.get('case09', None)

    @property
    def case10(self):
        return self._raw_data.get('case10', None)

    @property
    def case11(self):
        return self._raw_data.get('case11', None)

    @property
    def case12(self):
        return self._raw_data.get('case12', None)

    @property
    def case13(self):
        return self._raw_data.get('case13', None)

    @property
    def case14(self):
        return self._raw_data.get('case14', None)

    @property
    def case15(self):
        return self._raw_data.get('case15', None)

    @property
    def case16(self):
        return self._raw_data.get('case16', None)



class comp_kv_setter(ControlEnables, Angles):
    icon_sprite = "editor/comp_kv_setter"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def mode(self):
        return self._raw_data.get('mode', "kv")

    @property
    def kv_name(self):
        return self._raw_data.get('kv_name', None)

    @property
    def kv_value_mode(self):
        return self._raw_data.get('kv_value_mode', "legacy")

    @property
    def kv_value_global(self):
        return self._raw_data.get('kv_value_global', None)

    @property
    def kv_value_local(self):
        return self._raw_data.get('kv_value_local', None)

    @property
    def kv_value_pos(self):
        return parse_float_vector(self._raw_data.get('kv_value_pos', None))

    @property
    def invert(self):
        return self._raw_data.get('invert', "0")

    @property
    def rotate(self):
        return self._raw_data.get('rotate', "0")

    @property
    def conv_ang(self):
        return self._raw_data.get('conv_ang', "0")



class comp_pack(ControlEnables):
    icon_sprite = "editor/comp_pack"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def generic1(self):
        return self._raw_data.get('generic1', None)

    @property
    def generic2(self):
        return self._raw_data.get('generic2', None)

    @property
    def generic3(self):
        return self._raw_data.get('generic3', None)

    @property
    def generic4(self):
        return self._raw_data.get('generic4', None)

    @property
    def generic5(self):
        return self._raw_data.get('generic5', None)

    @property
    def sound1(self):
        return self._raw_data.get('sound1', None)

    @property
    def sound2(self):
        return self._raw_data.get('sound2', None)

    @property
    def sound3(self):
        return self._raw_data.get('sound3', None)

    @property
    def sound4(self):
        return self._raw_data.get('sound4', None)

    @property
    def sound5(self):
        return self._raw_data.get('sound5', None)

    @property
    def model1(self):
        return self._raw_data.get('model1', None)

    @property
    def model2(self):
        return self._raw_data.get('model2', None)

    @property
    def model3(self):
        return self._raw_data.get('model3', None)

    @property
    def model4(self):
        return self._raw_data.get('model4', None)

    @property
    def model5(self):
        return self._raw_data.get('model5', None)

    @property
    def material1(self):
        return self._raw_data.get('material1', None)

    @property
    def material2(self):
        return self._raw_data.get('material2', None)

    @property
    def material3(self):
        return self._raw_data.get('material3', None)

    @property
    def material4(self):
        return self._raw_data.get('material4', None)

    @property
    def material5(self):
        return self._raw_data.get('material5', None)

    @property
    def particle1(self):
        return self._raw_data.get('particle1', None)

    @property
    def particle2(self):
        return self._raw_data.get('particle2', None)

    @property
    def particle3(self):
        return self._raw_data.get('particle3', None)

    @property
    def particle4(self):
        return self._raw_data.get('particle4', None)

    @property
    def particle5(self):
        return self._raw_data.get('particle5', None)

    @property
    def soundscript1(self):
        return self._raw_data.get('soundscript1', None)

    @property
    def soundscript2(self):
        return self._raw_data.get('soundscript2', None)

    @property
    def soundscript3(self):
        return self._raw_data.get('soundscript3', None)

    @property
    def soundscript4(self):
        return self._raw_data.get('soundscript4', None)

    @property
    def soundscript5(self):
        return self._raw_data.get('soundscript5', None)



class comp_pack_rename(ControlEnables):
    icon_sprite = "editor/comp_pack_rename"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def filesrc(self):
        return self._raw_data.get('filesrc', None)

    @property
    def filedest(self):
        return self._raw_data.get('filedest', None)

    @property
    def filetype(self):
        return self._raw_data.get('filetype', "GENERIC")



class comp_pack_replace_soundscript(ControlEnables):
    icon_sprite = "editor/comp_pack_replace_soundscript"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def original(self):
        return self._raw_data.get('original', None)

    @property
    def replacement(self):
        return self._raw_data.get('replacement', None)



class comp_precache_model(ControlEnables, Angles):
    icon_sprite = "editor/comp_precache_model"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', None)

    @property
    def skin(self):
        return parse_source_value(self._raw_data.get('skin', None))

    @property
    def skinset(self):
        return self._raw_data.get('skinset', None)

    @property
    def lineent(self):
        return self._raw_data.get('lineent', None)



class comp_precache_sound(ControlEnables):
    icon_sprite = "editor/comp_precache_sound"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def sound1(self):
        return self._raw_data.get('sound1', None)

    @property
    def sound2(self):
        return self._raw_data.get('sound2', None)

    @property
    def sound3(self):
        return self._raw_data.get('sound3', None)

    @property
    def sound4(self):
        return self._raw_data.get('sound4', None)

    @property
    def sound5(self):
        return self._raw_data.get('sound5', None)

    @property
    def sound6(self):
        return self._raw_data.get('sound6', None)

    @property
    def sound7(self):
        return self._raw_data.get('sound7', None)

    @property
    def sound8(self):
        return self._raw_data.get('sound8', None)

    @property
    def sound9(self):
        return self._raw_data.get('sound9', None)

    @property
    def sound10(self):
        return self._raw_data.get('sound10', None)



class comp_prop_cable(StaticTargetName):
    icon_sprite = "editor/comp_prop_cable"

    @property
    def group(self):
        return self._raw_data.get('group', None)

    @property
    def nextkey(self):
        return self._raw_data.get('nextkey', None)

    @property
    def slack(self):
        return parse_source_value(self._raw_data.get('slack', 25))

    @property
    def positioninterpolator(self):
        return self._raw_data.get('positioninterpolator', "2")

    @property
    def segments(self):
        return parse_source_value(self._raw_data.get('segments', 2))

    @property
    def sides(self):
        return parse_source_value(self._raw_data.get('sides', 8))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 1.0))

    @property
    def coll_segments(self):
        return self._raw_data.get('coll_segments', "-1")

    @property
    def coll_sides(self):
        return self._raw_data.get('coll_sides', "0")

    @property
    def material(self):
        return self._raw_data.get('material', "models/cables/generic_black")

    @property
    def mat_scale(self):
        return parse_source_value(self._raw_data.get('mat_scale', 1))

    @property
    def mat_rotate(self):
        return self._raw_data.get('mat_rotate', "0")

    @property
    def u_min(self):
        return parse_source_value(self._raw_data.get('u_min', 0.0))

    @property
    def u_max(self):
        return parse_source_value(self._raw_data.get('u_max', 1.0))

    @property
    def bunting(self):
        return self._raw_data.get('bunting', None)

    @property
    def linedivider_staticprop(self):
        return self._raw_data.get('linedivider_staticprop', None)

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
    def disableshadows(self):
        return self._raw_data.get('disableshadows', "0")

    @property
    def disableselfshadowing(self):
        return self._raw_data.get('disableselfshadowing', "0")

    @property
    def disablevertexlighting(self):
        return self._raw_data.get('disablevertexlighting', "0")

    @property
    def movespeed(self):
        return parse_source_value(self._raw_data.get('movespeed', 1))



class comp_prop_rope(StaticTargetName):
    icon_sprite = "editor/comp_prop_rope"

    @property
    def group(self):
        return self._raw_data.get('group', None)

    @property
    def nextkey(self):
        return self._raw_data.get('nextkey', None)

    @property
    def slack(self):
        return parse_source_value(self._raw_data.get('slack', 25))

    @property
    def positioninterpolator(self):
        return self._raw_data.get('positioninterpolator', "2")

    @property
    def segments(self):
        return parse_source_value(self._raw_data.get('segments', 2))

    @property
    def sides(self):
        return parse_source_value(self._raw_data.get('sides', 8))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 1.0))

    @property
    def coll_segments(self):
        return self._raw_data.get('coll_segments', "-1")

    @property
    def coll_sides(self):
        return self._raw_data.get('coll_sides', "0")

    @property
    def material(self):
        return self._raw_data.get('material', "models/cables/generic_black")

    @property
    def mat_scale(self):
        return parse_source_value(self._raw_data.get('mat_scale', 1))

    @property
    def mat_rotate(self):
        return self._raw_data.get('mat_rotate', "0")

    @property
    def u_min(self):
        return parse_source_value(self._raw_data.get('u_min', 0.0))

    @property
    def u_max(self):
        return parse_source_value(self._raw_data.get('u_max', 1.0))

    @property
    def bunting(self):
        return self._raw_data.get('bunting', None)

    @property
    def linedivider_staticprop(self):
        return self._raw_data.get('linedivider_staticprop', None)

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
    def disableshadows(self):
        return self._raw_data.get('disableshadows', "0")

    @property
    def disableselfshadowing(self):
        return self._raw_data.get('disableselfshadowing', "0")

    @property
    def disablevertexlighting(self):
        return self._raw_data.get('disablevertexlighting', "0")

    @property
    def movespeed(self):
        return parse_source_value(self._raw_data.get('movespeed', 1))



class comp_prop_rope_bunting(StaticTargetName):
    icon_sprite = "editor/comp_prop_rope_bunting"

    @property
    def weight(self):
        return parse_source_value(self._raw_data.get('weight', 1))

    @property
    def placement_interval(self):
        return parse_source_value(self._raw_data.get('placement_interval', 1))

    @property
    def distance(self):
        return parse_source_value(self._raw_data.get('distance', 0))

    @property
    def model(self):
        return self._raw_data.get('model', None)

    @property
    def angles(self):
        return parse_float_vector(self._raw_data.get('angles', "0 0 0"))

    @property
    def orient(self):
        return self._raw_data.get('orient', "follow")



class comp_propcombine_set(Angles):
    icon_sprite = "editor/comp_propcombine_set"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def mins(self):
        return parse_float_vector(self._raw_data.get('mins', "-32 -32 -32"))

    @property
    def maxs(self):
        return parse_float_vector(self._raw_data.get('maxs', "32 32 32"))

    @property
    def name(self):
        return self._raw_data.get('name', None)

    @property
    def prop(self):
        return self._raw_data.get('prop', None)

    @property
    def skin(self):
        return parse_source_value(self._raw_data.get('skin', 0))



class comp_relay(ControlEnables, StaticTargetName):
    icon_sprite = "editor/comp_relay"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def delay(self):
        return parse_source_value(self._raw_data.get('delay', 0.0))

    @property
    def delay_max(self):
        return parse_source_value(self._raw_data.get('delay_max', None))



class comp_scriptvar_setter(Angles, ControlEnables, Origin):
    icon_sprite = "editor/comp_scriptvar_setter"
    model = "models/editor/cone_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def variable(self):
        return self._raw_data.get('variable', None)

    @property
    def ref(self):
        return self._raw_data.get('ref', None)

    @property
    def mode(self):
        return self._raw_data.get('mode', "pos")

    @property
    def const(self):
        return self._raw_data.get('const', None)



class comp_vactube_junction(Angles):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def targetname(self):
        return self._raw_data.get('targetname', None)

    @property
    def model(self):
        return self._raw_data.get('model', "models/editor/vactubes/straight.mdl")

    @property
    def skin(self):
        return self._raw_data.get('skin', "0")

    @property
    def persist_tv(self):
        return self._raw_data.get('persist_tv', "0")

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def target_sec(self):
        return self._raw_data.get('target_sec', None)

    @property
    def target_ter(self):
        return self._raw_data.get('target_ter', None)



class comp_vactube_sensor(Angles):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 32))

    @property
    def obj_model(self):
        return self._raw_data.get('obj_model', None)



class comp_vactube_spline(StaticTargetName, Angles):
    icon_sprite = "editor/comp_prop_rope"

    @property
    def nextkey(self):
        return self._raw_data.get('nextkey', None)

    @property
    def opaque(self):
        return self._raw_data.get('opaque', "0")

    @property
    def skin(self):
        return self._raw_data.get('skin', "0")

    @property
    def segments(self):
        return parse_source_value(self._raw_data.get('segments', 2))

    @property
    def collisions(self):
        return self._raw_data.get('collisions', "1")

    @property
    def positioninterpolator(self):
        return self._raw_data.get('positioninterpolator', "1")

    @property
    def vac_separateglass(self):
        return self._raw_data.get('vac_separateglass', "0")

    @property
    def linedivider_staticprop(self):
        return self._raw_data.get('linedivider_staticprop', None)

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
    def disableshadows(self):
        return self._raw_data.get('disableshadows', "0")

    @property
    def disableselfshadowing(self):
        return self._raw_data.get('disableselfshadowing', "0")

    @property
    def disablevertexlighting(self):
        return self._raw_data.get('disablevertexlighting', "0")

    @property
    def movespeed(self):
        return parse_source_value(self._raw_data.get('movespeed', 1))



class env_bubbles(BaseEffectBrush):

    @property
    def density(self):
        return parse_source_value(self._raw_data.get('density', 2))

    @property
    def frequency(self):
        return parse_source_value(self._raw_data.get('frequency', 2))

    @property
    def current(self):
        return parse_source_value(self._raw_data.get('current', 0))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class env_embers(BaseEffectBrush):

    @property
    def angles(self):
        return parse_float_vector(self._raw_data.get('angles', "0 0 0"))

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

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class func_instance(Angles):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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

    @property
    def replace11(self):
        return self._raw_data.get('replace11', None)

    @property
    def replace12(self):
        return self._raw_data.get('replace12', None)

    @property
    def replace13(self):
        return self._raw_data.get('replace13', None)

    @property
    def replace14(self):
        return self._raw_data.get('replace14', None)

    @property
    def replace15(self):
        return self._raw_data.get('replace15', None)

    @property
    def replace16(self):
        return self._raw_data.get('replace16', None)

    @property
    def replace17(self):
        return self._raw_data.get('replace17', None)

    @property
    def replace18(self):
        return self._raw_data.get('replace18', None)

    @property
    def replace19(self):
        return self._raw_data.get('replace19', None)

    @property
    def replace20(self):
        return self._raw_data.get('replace20', None)

    @property
    def replace21(self):
        return self._raw_data.get('replace21', None)

    @property
    def replace22(self):
        return self._raw_data.get('replace22', None)

    @property
    def replace23(self):
        return self._raw_data.get('replace23', None)

    @property
    def replace24(self):
        return self._raw_data.get('replace24', None)

    @property
    def replace25(self):
        return self._raw_data.get('replace25', None)

    @property
    def replace26(self):
        return self._raw_data.get('replace26', None)

    @property
    def replace27(self):
        return self._raw_data.get('replace27', None)

    @property
    def replace28(self):
        return self._raw_data.get('replace28', None)

    @property
    def replace29(self):
        return self._raw_data.get('replace29', None)

    @property
    def replace30(self):
        return self._raw_data.get('replace30', None)



class func_precipitation(BaseEffectBrush):

    @property
    def renderamt(self):
        return parse_source_value(self._raw_data.get('renderamt', 5))

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "100 100 100"))

    @property
    def preciptype(self):
        return self._raw_data.get('preciptype', "0")

    @property
    def innerdistance(self):
        return parse_source_value(self._raw_data.get('innerdistance', -1))

    @property
    def innernearparticle(self):
        return self._raw_data.get('innernearparticle', None)

    @property
    def innerfarparticle(self):
        return self._raw_data.get('innerfarparticle', None)

    @property
    def outerparticle(self):
        return self._raw_data.get('outerparticle', None)



class func_smokevolume(BaseEffectBrush):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def color1(self):
        return parse_int_vector(self._raw_data.get('color1', "255 255 255"))

    @property
    def color2(self):
        return parse_int_vector(self._raw_data.get('color2', "255 255 255"))

    @property
    def material(self):
        return self._raw_data.get('material', "particle/particle_smokegrenade")

    @property
    def particledrawwidth(self):
        return parse_source_value(self._raw_data.get('particledrawwidth', 120))

    @property
    def particlespacingdistance(self):
        return parse_source_value(self._raw_data.get('particlespacingdistance', 80))

    @property
    def densityrampspeed(self):
        return parse_source_value(self._raw_data.get('densityrampspeed', 1))

    @property
    def rotationspeed(self):
        return parse_source_value(self._raw_data.get('rotationspeed', 10))

    @property
    def movementspeed(self):
        return parse_source_value(self._raw_data.get('movementspeed', 10))

    @property
    def density(self):
        return parse_source_value(self._raw_data.get('density', 1))

    @property
    def maxdrawdistance(self):
        return parse_source_value(self._raw_data.get('maxdrawdistance', 0))



class info_lighting(StaticTargetName):
    icon_sprite = "editor/info_lighting.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class info_node(Node):
    model = "models/editor/ground_node.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class info_node_air(Node):
    model = "models/editor/air_node.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def nodeheight(self):
        return parse_source_value(self._raw_data.get('nodeheight', 0))



class prop_detail(Angles, Origin, DetailPropBase):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', None)



class prop_detail_sprite(Angles, Origin, DetailPropBase):
    icon_sprite = "editor/ficool2/prop_detail_sprite"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def position_ul(self):
        return parse_float_vector(self._raw_data.get('position_ul', "-10 20"))

    @property
    def position_lr(self):
        return parse_float_vector(self._raw_data.get('position_lr', "10 0"))

    @property
    def tex_ul(self):
        return parse_float_vector(self._raw_data.get('tex_ul', "0 0"))

    @property
    def tex_size(self):
        return parse_float_vector(self._raw_data.get('tex_size', "64 64"))

    @property
    def tex_total_size(self):
        return parse_source_value(self._raw_data.get('tex_total_size', 512))



class prop_static(Angles):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', None)

    @property
    def skin(self):
        return parse_source_value(self._raw_data.get('skin', 0))

    @property
    def renderamt(self):
        return parse_source_value(self._raw_data.get('renderamt', 255))

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "255 255 255"))

    @property
    def uniformscale(self):
        return parse_source_value(self._raw_data.get('uniformscale', 1))

    @property
    def scale(self):
        return parse_float_vector(self._raw_data.get('scale', "1 1 1"))

    @property
    def solid(self):
        return self._raw_data.get('solid', "6")

    @property
    def preventpropcombine(self):
        return self._raw_data.get('preventpropcombine', "0")

    @property
    def linedivider_levels(self):
        return self._raw_data.get('linedivider_levels', None)

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
    def linedivider_light(self):
        return self._raw_data.get('linedivider_light', None)

    @property
    def disableshadows(self):
        return self._raw_data.get('disableshadows', "0")

    @property
    def disableshadowdepth(self):
        return self._raw_data.get('disableshadowdepth', "0")

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
    def enablelightbounce(self):
        return self._raw_data.get('enablelightbounce', "0")

    @property
    def drawinfastreflection(self):
        return self._raw_data.get('drawinfastreflection', "0")

    @property
    def lightingorigin(self):
        return self._raw_data.get('lightingorigin', None)

    @property
    def linedivider_fade(self):
        return self._raw_data.get('linedivider_fade', None)

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



class BaseEntity(BaseEntityIO):

    @property
    def targetname(self):
        return self._raw_data.get('targetname', None)

    @property
    def globalname(self):
        return self._raw_data.get('globalname', None)

    @property
    def vscripts(self):
        return self._raw_data.get('vscripts', None)

    @property
    def thinkfunction(self):
        return self._raw_data.get('thinkfunction', None)

    @property
    def linedivider_base(self):
        return self._raw_data.get('linedivider_base', None)

    @property
    def vscript_init_code(self):
        return self._raw_data.get('vscript_init_code', None)

    @property
    def vscript_init_code2(self):
        return self._raw_data.get('vscript_init_code2', None)



class BaseEntityBrush(BaseEntityIO):

    @property
    def targetname(self):
        return self._raw_data.get('targetname', None)

    @property
    def globalname(self):
        return self._raw_data.get('globalname', None)

    @property
    def origin(self):
        return parse_float_vector(self._raw_data.get('origin', None))

    @property
    def parentname(self):
        return self._raw_data.get('parentname', None)

    @property
    def linedivider_vscript(self):
        return self._raw_data.get('linedivider_vscript', None)

    @property
    def vscripts(self):
        return self._raw_data.get('vscripts', None)

    @property
    def thinkfunction(self):
        return self._raw_data.get('thinkfunction', None)

    @property
    def linedivider_base(self):
        return self._raw_data.get('linedivider_base', None)

    @property
    def mins(self):
        return parse_float_vector(self._raw_data.get('mins', None))

    @property
    def maxs(self):
        return parse_float_vector(self._raw_data.get('maxs', None))

    @property
    def solid(self):
        return self._raw_data.get('solid', "6")

    @property
    def linedivider_basebrush(self):
        return self._raw_data.get('linedivider_basebrush', None)

    @property
    def parent_attachment_point(self):
        return self._raw_data.get('parent_attachment_point', None)

    @property
    def vscript_init_code(self):
        return self._raw_data.get('vscript_init_code', None)

    @property
    def vscript_init_code2(self):
        return self._raw_data.get('vscript_init_code2', None)



class BaseEntityPoint(BaseEntityIO):

    @property
    def targetname(self):
        return self._raw_data.get('targetname', None)

    @property
    def globalname(self):
        return self._raw_data.get('globalname', None)

    @property
    def angles(self):
        return parse_float_vector(self._raw_data.get('angles', "0 0 0"))

    @property
    def parentname(self):
        return self._raw_data.get('parentname', None)

    @property
    def linedivider_vscript(self):
        return self._raw_data.get('linedivider_vscript', None)

    @property
    def vscripts(self):
        return self._raw_data.get('vscripts', None)

    @property
    def thinkfunction(self):
        return self._raw_data.get('thinkfunction', None)

    @property
    def linedivider_base(self):
        return self._raw_data.get('linedivider_base', None)

    @property
    def parent_attachment_point(self):
        return self._raw_data.get('parent_attachment_point', None)

    @property
    def vscript_init_code(self):
        return self._raw_data.get('vscript_init_code', None)

    @property
    def vscript_init_code2(self):
        return self._raw_data.get('vscript_init_code2', None)



class func_dustcloud(BaseDustParticleSpawner):

    @property
    def sizemin(self):
        return self._raw_data.get('sizemin', "100")

    @property
    def sizemax(self):
        return self._raw_data.get('sizemax', "200")

    @property
    def alpha(self):
        return parse_source_value(self._raw_data.get('alpha', 30))



class func_dustmotes(BaseDustParticleSpawner):

    @property
    def sizemin(self):
        return self._raw_data.get('sizemin', "10")

    @property
    def sizemax(self):
        return self._raw_data.get('sizemax', "20")

    @property
    def alpha(self):
        return parse_source_value(self._raw_data.get('alpha', 255))



class BaseActBusy(BaseEntityPoint):

    @property
    def actor(self):
        return self._raw_data.get('actor', None)

    @property
    def startactive(self):
        return self._raw_data.get('startactive', "0")

    @property
    def searchtype(self):
        return self._raw_data.get('searchtype', "0")

    @property
    def busysearchrange(self):
        return parse_source_value(self._raw_data.get('busysearchrange', 2048))

    @property
    def visibleonly(self):
        return self._raw_data.get('visibleonly', "0")



class BaseBeam(Reflection, RenderFields, BaseEntityPoint):

    @property
    def hdrcolorscale(self):
        return parse_source_value(self._raw_data.get('hdrcolorscale', 1.0))

    @property
    def noiseamplitude(self):
        return parse_source_value(self._raw_data.get('noiseamplitude', 0))

    @property
    def framerate(self):
        return parse_source_value(self._raw_data.get('framerate', 0))

    @property
    def framestart(self):
        return parse_source_value(self._raw_data.get('framestart', 0))

    @property
    def texture(self):
        return self._raw_data.get('texture', "sprites/laserbeam.spr")

    @property
    def texturescroll(self):
        return parse_source_value(self._raw_data.get('texturescroll', 35))

    @property
    def damage(self):
        return self._raw_data.get('damage', "0")

    @property
    def dissolvetype(self):
        return self._raw_data.get('dissolvetype', "-1")



class BaseEntityAnimating(BaseEntityPoint, DamageFilter, RenderFields, ToggleDraw, Reflection):

    @property
    def effects(self):
        return self._raw_data.get('effects', "0")

    @property
    def solid(self):
        return self._raw_data.get('solid', "6")

    @property
    def body(self):
        return parse_source_value(self._raw_data.get('body', 0))

    @property
    def setbodygroup(self):
        return parse_source_value(self._raw_data.get('setbodygroup', 0))

    @property
    def texframeindex(self):
        return parse_source_value(self._raw_data.get('texframeindex', None))

    @property
    def hitboxset(self):
        return self._raw_data.get('hitboxset', None)

    @property
    def modelscale(self):
        return parse_source_value(self._raw_data.get('modelscale', None))

    @property
    def allowsilentdissolve(self):
        return self._raw_data.get('allowsilentdissolve', "1")

    @property
    def linedivider_animbase(self):
        return self._raw_data.get('linedivider_animbase', None)

    @property
    def lightingorigin(self):
        return self._raw_data.get('lightingorigin', None)

    @property
    def lightingoriginhack(self):
        return self._raw_data.get('lightingoriginhack', None)

    @property
    def fademindist(self):
        return parse_source_value(self._raw_data.get('fademindist', None))

    @property
    def fademaxdist(self):
        return parse_source_value(self._raw_data.get('fademaxdist', None))

    @property
    def fadescale(self):
        return parse_source_value(self._raw_data.get('fadescale', 1))

    @property
    def shadowcastdist(self):
        return parse_source_value(self._raw_data.get('shadowcastdist', None))

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

    @property
    def linedivider_anim(self):
        return self._raw_data.get('linedivider_anim', None)



class BaseEntityPhysics(Reflection, DamageFilter, RenderFields, BaseEntityPoint):

    @property
    def solid(self):
        return self._raw_data.get('solid', "6")

    @property
    def body(self):
        return parse_source_value(self._raw_data.get('body', 0))

    @property
    def texframeindex(self):
        return parse_source_value(self._raw_data.get('texframeindex', None))

    @property
    def lightingorigin(self):
        return self._raw_data.get('lightingorigin', None)

    @property
    def lightingoriginhack(self):
        return self._raw_data.get('lightingoriginhack', None)

    @property
    def fademindist(self):
        return parse_source_value(self._raw_data.get('fademindist', None))

    @property
    def fademaxdist(self):
        return parse_source_value(self._raw_data.get('fademaxdist', None))

    @property
    def fadescale(self):
        return parse_source_value(self._raw_data.get('fadescale', 1))

    @property
    def shadowcastdist(self):
        return parse_source_value(self._raw_data.get('shadowcastdist', None))

    @property
    def disableshadows(self):
        return self._raw_data.get('disableshadows', "0")

    @property
    def modelscale(self):
        return parse_source_value(self._raw_data.get('modelscale', None))

    @property
    def allowsilentdissolve(self):
        return self._raw_data.get('allowsilentdissolve', "1")

    @property
    def linedivider_phys(self):
        return self._raw_data.get('linedivider_phys', None)



class BaseEntityVisBrush(ToggleDraw, RenderFields, BaseEntityBrush):

    @property
    def effects(self):
        return self._raw_data.get('effects', "0")

    @property
    def vrad_brush_cast_shadows(self):
        return self._raw_data.get('vrad_brush_cast_shadows', "0")

    @property
    def _minlight(self):
        return parse_source_value(self._raw_data.get('_minlight', 0))

    @property
    def disableshadowdepth(self):
        return self._raw_data.get('disableshadowdepth', "0")

    @property
    def shadowdepthnocache(self):
        return self._raw_data.get('shadowdepthnocache', "0")

    @property
    def disableflashlight(self):
        return self._raw_data.get('disableflashlight', "0")

    @property
    def linedivider_visbrush(self):
        return self._raw_data.get('linedivider_visbrush', None)



class BaseNPCMaker(EnableDisable, BaseEntityPoint):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def maxnpccount(self):
        return parse_source_value(self._raw_data.get('maxnpccount', 1))

    @property
    def spawnfrequency(self):
        return self._raw_data.get('spawnfrequency', "5")

    @property
    def maxlivechildren(self):
        return parse_source_value(self._raw_data.get('maxlivechildren', 5))

    @property
    def hullcheckmode(self):
        return self._raw_data.get('hullcheckmode', "0")



class BasePointLight(BaseLight, BaseClusteredLight, BaseEntityPoint):

    @property
    def _removeaftercompile(self):
        return self._raw_data.get('_removeaftercompile', "0")



class BaseSpotLight(BaseLight, BaseClusteredLight, BaseEntityPoint):

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
    def _removeaftercompile(self):
        return self._raw_data.get('_removeaftercompile', "0")

    @property
    def pitch(self):
        return parse_source_value(self._raw_data.get('pitch', -90))



class BaseTank(BaseEntityBrush):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def control_volume(self):
        return self._raw_data.get('control_volume', None)

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
        return self._raw_data.get('spritesmoke', None)

    @property
    def spriteflash(self):
        return self._raw_data.get('spriteflash', None)

    @property
    def spritescale(self):
        return self._raw_data.get('spritescale', "1")

    @property
    def rotatestartsound(self):
        return self._raw_data.get('rotatestartsound', None)

    @property
    def rotatesound(self):
        return self._raw_data.get('rotatesound', None)

    @property
    def rotatestopsound(self):
        return self._raw_data.get('rotatestopsound', None)

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
    def minrange(self):
        return self._raw_data.get('minrange', "0")

    @property
    def maxrange(self):
        return self._raw_data.get('maxrange', "0")

    @property
    def gun_base_attach(self):
        return self._raw_data.get('gun_base_attach', None)

    @property
    def gun_barrel_attach(self):
        return self._raw_data.get('gun_barrel_attach', None)

    @property
    def gun_yaw_pose_param(self):
        return self._raw_data.get('gun_yaw_pose_param', None)

    @property
    def gun_yaw_pose_center(self):
        return parse_source_value(self._raw_data.get('gun_yaw_pose_center', 0))

    @property
    def gun_pitch_pose_param(self):
        return self._raw_data.get('gun_pitch_pose_param', None)

    @property
    def gun_pitch_pose_center(self):
        return parse_source_value(self._raw_data.get('gun_pitch_pose_center', 0))

    @property
    def ammo_count(self):
        return parse_source_value(self._raw_data.get('ammo_count', -1))

    @property
    def leadtarget(self):
        return self._raw_data.get('leadtarget', "0")

    @property
    def npc_man_point(self):
        return self._raw_data.get('npc_man_point', None)

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
    def shouldfindnpcs(self):
        return self._raw_data.get('shouldfindnpcs', "1")

    @property
    def effecthandling(self):
        return self._raw_data.get('effecthandling', "0")



class CombineBallSpawners(BaseEntityPoint):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

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
        return self._raw_data.get('balltype', "0")

    @property
    def ballrespawntime(self):
        return parse_source_value(self._raw_data.get('ballrespawntime', 4.0))



class FollowGoal(BaseEntityPoint):

    @property
    def actor(self):
        return self._raw_data.get('actor', None)

    @property
    def goal(self):
        return self._raw_data.get('goal', None)

    @property
    def searchtype(self):
        return self._raw_data.get('searchtype', "0")

    @property
    def startactive(self):
        return self._raw_data.get('startactive', "0")

    @property
    def maximumstate(self):
        return self._raw_data.get('maximumstate', "1")

    @property
    def formation(self):
        return self._raw_data.get('formation', "0")



class ForceController(BaseEntityPoint):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def attach1(self):
        return self._raw_data.get('attach1', None)

    @property
    def forcetime(self):
        return parse_source_value(self._raw_data.get('forcetime', 0))



class LeadGoalBase(BaseEntityPoint):

    @property
    def actor(self):
        return self._raw_data.get('actor', None)

    @property
    def goal(self):
        return self._raw_data.get('goal', None)

    @property
    def waitpointname(self):
        return self._raw_data.get('waitpointname', None)

    @property
    def waitdistance(self):
        return parse_source_value(self._raw_data.get('waitdistance', None))

    @property
    def leaddistance(self):
        return parse_source_value(self._raw_data.get('leaddistance', 64))

    @property
    def retrievedistance(self):
        return parse_source_value(self._raw_data.get('retrievedistance', 96))

    @property
    def successdistance(self):
        return parse_source_value(self._raw_data.get('successdistance', 0))

    @property
    def run(self):
        return self._raw_data.get('run', "0")

    @property
    def retrieve(self):
        return self._raw_data.get('retrieve', "1")

    @property
    def comingbackwaitforspeak(self):
        return self._raw_data.get('comingbackwaitforspeak', "1")

    @property
    def retrievewaitforspeak(self):
        return self._raw_data.get('retrievewaitforspeak', "1")

    @property
    def dontspeakstart(self):
        return self._raw_data.get('dontspeakstart', "0")

    @property
    def leadduringcombat(self):
        return self._raw_data.get('leadduringcombat', "0")

    @property
    def gagleader(self):
        return self._raw_data.get('gagleader', "0")

    @property
    def attractplayerconceptmodifier(self):
        return self._raw_data.get('attractplayerconceptmodifier', None)

    @property
    def waitoverconceptmodifier(self):
        return self._raw_data.get('waitoverconceptmodifier', None)

    @property
    def arrivalconceptmodifier(self):
        return self._raw_data.get('arrivalconceptmodifier', None)

    @property
    def postarrivalconceptmodifier(self):
        return self._raw_data.get('postarrivalconceptmodifier', None)

    @property
    def successconceptmodifier(self):
        return self._raw_data.get('successconceptmodifier', None)

    @property
    def failureconceptmodifier(self):
        return self._raw_data.get('failureconceptmodifier', None)

    @property
    def comingbackconceptmodifier(self):
        return self._raw_data.get('comingbackconceptmodifier', None)

    @property
    def retrieveconceptmodifier(self):
        return self._raw_data.get('retrieveconceptmodifier', None)

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class NavCost(TeamNum, BaseEntityBrush):

    @property
    def start_disabled(self):
        return self._raw_data.get('start_disabled', "0")



class TriggerOnce(BaseEntityBrush):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def startdisabled(self):
        return self._raw_data.get('startdisabled', "0")

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)



class TwoObjectPhysics(BaseEntityPoint):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def attach1(self):
        return self._raw_data.get('attach1', None)

    @property
    def attach2(self):
        return self._raw_data.get('attach2', None)

    @property
    def constraintsystem(self):
        return self._raw_data.get('constraintsystem', None)

    @property
    def forcelimit(self):
        return parse_source_value(self._raw_data.get('forcelimit', 0))

    @property
    def torquelimit(self):
        return parse_source_value(self._raw_data.get('torquelimit', 0))

    @property
    def breaksound(self):
        return self._raw_data.get('breaksound', None)

    @property
    def teleportfollowdistance(self):
        return parse_source_value(self._raw_data.get('teleportfollowdistance', 0))



class Weapon(BaseFadeProp, BaseEntityPoint):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class ai_ally_manager(BaseEntityPoint):
    icon_sprite = "materials/editor/ficool2/ai_ally_manager.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def maxallies(self):
        return parse_source_value(self._raw_data.get('maxallies', 5))

    @property
    def maxmedics(self):
        return parse_source_value(self._raw_data.get('maxmedics', 1))



class ai_battle_line(BaseEntityPoint):
    icon_sprite = "editor/ficool2/ai_battle_line"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def actor(self):
        return self._raw_data.get('actor', None)

    @property
    def active(self):
        return self._raw_data.get('active', "0")

    @property
    def strict(self):
        return self._raw_data.get('strict', "1")



class ai_changehintgroup(BaseEntityPoint):
    icon_sprite = "editor/ficool2/ai_changehintgroup"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def searchtype(self):
        return self._raw_data.get('searchtype', "0")

    @property
    def searchname(self):
        return self._raw_data.get('searchname', None)

    @property
    def newhintgroup(self):
        return self._raw_data.get('newhintgroup', None)

    @property
    def radius(self):
        return self._raw_data.get('radius', "0.0")

    @property
    def hintlimiting(self):
        return self._raw_data.get('hintlimiting', "0")



class ai_changetarget(BaseEntityPoint):
    icon_sprite = "editor/ai_changetarget.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def m_isznewtarget(self):
        return self._raw_data.get('m_isznewtarget', None)



class ai_citizen_response_system(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class ai_goal_assault(BaseEntityPoint):
    icon_sprite = "editor/ficool2/ai_goal_assault"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def actor(self):
        return self._raw_data.get('actor', None)

    @property
    def rallypoint(self):
        return self._raw_data.get('rallypoint', None)

    @property
    def searchtype(self):
        return self._raw_data.get('searchtype', "0")

    @property
    def startactive(self):
        return self._raw_data.get('startactive', "0")

    @property
    def assaultcue(self):
        return self._raw_data.get('assaultcue', "1")

    @property
    def rallyselectmethod(self):
        return self._raw_data.get('rallyselectmethod', "0")

    @property
    def branchmethod(self):
        return self._raw_data.get('branchmethod', "0")



class ai_goal_fightfromcover(BaseEntityPoint):
    icon_sprite = "editor/ficool2/ai_goal_fightfromcover.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def actor(self):
        return self._raw_data.get('actor', None)

    @property
    def goal(self):
        return self._raw_data.get('goal', None)

    @property
    def directionalmarker(self):
        return self._raw_data.get('directionalmarker', None)

    @property
    def generichinttype(self):
        return self._raw_data.get('generichinttype', None)

    @property
    def width(self):
        return parse_source_value(self._raw_data.get('width', 600))

    @property
    def length(self):
        return parse_source_value(self._raw_data.get('length', 480))

    @property
    def height(self):
        return parse_source_value(self._raw_data.get('height', 2400))

    @property
    def bias(self):
        return parse_source_value(self._raw_data.get('bias', 60))

    @property
    def startactive(self):
        return self._raw_data.get('startactive', "0")



class ai_goal_operator(EnableDisable, BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def actor(self):
        return self._raw_data.get('actor', None)

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def contexttarget(self):
        return self._raw_data.get('contexttarget', None)

    @property
    def state(self):
        return self._raw_data.get('state', "0")

    @property
    def moveto(self):
        return self._raw_data.get('moveto', "1")



class ai_goal_police(BaseEntityPoint):
    icon_sprite = "editor/ai_goal_police.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def policeradius(self):
        return parse_source_value(self._raw_data.get('policeradius', 512))

    @property
    def policetarget(self):
        return self._raw_data.get('policetarget', None)



class ai_goal_standoff(BaseEntityPoint):
    icon_sprite = "editor/ai_goal_standoff.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def actor(self):
        return self._raw_data.get('actor', None)

    @property
    def searchtype(self):
        return self._raw_data.get('searchtype', "0")

    @property
    def startactive(self):
        return self._raw_data.get('startactive', "0")

    @property
    def hintgroupchangereaction(self):
        return self._raw_data.get('hintgroupchangereaction', "1")

    @property
    def aggressiveness(self):
        return self._raw_data.get('aggressiveness', "2")

    @property
    def playerbattleline(self):
        return self._raw_data.get('playerbattleline', "1")

    @property
    def stayatcover(self):
        return self._raw_data.get('stayatcover', "0")

    @property
    def abandonifenemyhides(self):
        return self._raw_data.get('abandonifenemyhides', "0")

    @property
    def customcoveronreload(self):
        return self._raw_data.get('customcoveronreload', "1")

    @property
    def custommintimeshots(self):
        return parse_source_value(self._raw_data.get('custommintimeshots', 2))

    @property
    def custommaxtimeshots(self):
        return parse_source_value(self._raw_data.get('custommaxtimeshots', 4))

    @property
    def customminshots(self):
        return parse_source_value(self._raw_data.get('customminshots', 1))

    @property
    def custommaxshots(self):
        return parse_source_value(self._raw_data.get('custommaxshots', 4))

    @property
    def customoddscover(self):
        return parse_source_value(self._raw_data.get('customoddscover', 25))



class ai_npc_eventresponsesystem(BaseEntityPoint):
    icon_sprite = "editor/ficool2/ai_npc_eventresponsesystem"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class ai_relationship(BaseEntityPoint):
    icon_sprite = "editor/ai_relationship.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def subject(self):
        return self._raw_data.get('subject', None)

    @property
    def target(self):
        return self._raw_data.get('target', None)

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
    def startactive(self):
        return self._raw_data.get('startactive', "0")

    @property
    def reciprocal(self):
        return self._raw_data.get('reciprocal', "0")

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class ai_script_conditions(BaseEntityPoint):
    icon_sprite = "editor/ai_script_conditions.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def actor(self):
        return self._raw_data.get('actor', None)

    @property
    def startdisabled(self):
        return self._raw_data.get('startdisabled', "1")

    @property
    def minimumstate(self):
        return self._raw_data.get('minimumstate', "1")

    @property
    def maximumstate(self):
        return self._raw_data.get('maximumstate', "3")

    @property
    def scriptstatus(self):
        return self._raw_data.get('scriptstatus', "2")

    @property
    def requiredtime(self):
        return parse_source_value(self._raw_data.get('requiredtime', 0))

    @property
    def mintimeout(self):
        return parse_source_value(self._raw_data.get('mintimeout', 0))

    @property
    def maxtimeout(self):
        return parse_source_value(self._raw_data.get('maxtimeout', 0))

    @property
    def actorseeplayer(self):
        return self._raw_data.get('actorseeplayer', "2")

    @property
    def playeractorproximity(self):
        return parse_source_value(self._raw_data.get('playeractorproximity', 0))

    @property
    def playeractorfov(self):
        return parse_source_value(self._raw_data.get('playeractorfov', 360))

    @property
    def playeractorfovtruecone(self):
        return self._raw_data.get('playeractorfovtruecone', "0")

    @property
    def playeractorlos(self):
        return self._raw_data.get('playeractorlos', "2")

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def actorseetarget(self):
        return self._raw_data.get('actorseetarget', "2")

    @property
    def actortargetproximity(self):
        return parse_source_value(self._raw_data.get('actortargetproximity', 0))

    @property
    def playertargetproximity(self):
        return parse_source_value(self._raw_data.get('playertargetproximity', 0))

    @property
    def playertargetfov(self):
        return parse_source_value(self._raw_data.get('playertargetfov', 360))

    @property
    def playertargetfovtruecone(self):
        return self._raw_data.get('playertargetfovtruecone', "0")

    @property
    def playertargetlos(self):
        return self._raw_data.get('playertargetlos', "2")

    @property
    def playerblockingactor(self):
        return self._raw_data.get('playerblockingactor', "2")

    @property
    def actorinpvs(self):
        return self._raw_data.get('actorinpvs', "2")

    @property
    def actorinvehicle(self):
        return self._raw_data.get('actorinvehicle', "2")

    @property
    def playerinvehicle(self):
        return self._raw_data.get('playerinvehicle', "2")

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class ai_sound(BaseEntityPoint):
    icon_sprite = "editor/ai_sound.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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
        return self._raw_data.get('locationproxy', None)



class ai_speechfilter(EnableDisable, ResponseContext, BaseEntityPoint):
    icon_sprite = "editor/ficool2/ai_speechfilter.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def subject(self):
        return self._raw_data.get('subject', None)

    @property
    def idlemodifier(self):
        return parse_source_value(self._raw_data.get('idlemodifier', 1.0))

    @property
    def neversayhello(self):
        return self._raw_data.get('neversayhello', "0")



class aiscripted_schedule(BaseEntityPoint):
    icon_sprite = "editor/aiscripted_schedule"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def m_iszentity(self):
        return self._raw_data.get('m_iszentity', None)

    @property
    def m_flradius(self):
        return parse_source_value(self._raw_data.get('m_flradius', 0))

    @property
    def graball(self):
        return self._raw_data.get('graball', "0")

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

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



class ambient_generic(BaseEntityPoint):
    icon_sprite = "editor/ambient_generic.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def message(self):
        return self._raw_data.get('message', None)

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 10))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 1250))

    @property
    def pitch(self):
        return parse_source_value(self._raw_data.get('pitch', 100))

    @property
    def sourceentityname(self):
        return self._raw_data.get('sourceentityname', None)

    @property
    def soundflags(self):
        return self._raw_data.get('soundflags', "0")

    @property
    def preset(self):
        return self._raw_data.get('preset', "0")

    @property
    def linedivider_snd(self):
        return self._raw_data.get('linedivider_snd', None)

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
        return self._raw_data.get('lfotype', "0")

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
    def haddons_enabled(self):
        return self._raw_data.get('haddons_enabled', "-1")

    @property
    def haddons_infrange(self):
        return self._raw_data.get('haddons_infrange', "-1")

    @property
    def haddons_mode(self):
        return self._raw_data.get('haddons_mode', "-1")



class assault_assaultpoint(BaseEntityPoint):
    icon_sprite = "editor/assault_point.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def assaultgroup(self):
        return self._raw_data.get('assaultgroup', None)

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
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def forcecrouch(self):
        return self._raw_data.get('forcecrouch', "0")

    @property
    def urgent(self):
        return self._raw_data.get('urgent', "0")

    @property
    def assaulttolerance(self):
        return self._raw_data.get('assaulttolerance', "36")



class assault_rallypoint(BaseEntityPoint):
    icon_sprite = "editor/assault_rally.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def assaultpoint(self):
        return self._raw_data.get('assaultpoint', None)

    @property
    def assaultdelay(self):
        return parse_source_value(self._raw_data.get('assaultdelay', 0))

    @property
    def rallysequence(self):
        return self._raw_data.get('rallysequence', None)

    @property
    def priority(self):
        return parse_source_value(self._raw_data.get('priority', 1))

    @property
    def forcecrouch(self):
        return self._raw_data.get('forcecrouch', "0")

    @property
    def urgent(self):
        return self._raw_data.get('urgent', "0")

    @property
    def lockpoint(self):
        return self._raw_data.get('lockpoint', "1")



class beam_spotlight(Angles, SystemLevelChoice, BaseEntityPoint):
    model = "models/editor/cone_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

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
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "255 255 255"))

    @property
    def hdrcolorscale(self):
        return parse_source_value(self._raw_data.get('hdrcolorscale', 0.7))



class color_correction(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/color_correction.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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
        return self._raw_data.get('filename', None)

    @property
    def fadeinduration(self):
        return parse_source_value(self._raw_data.get('fadeinduration', 0.0))

    @property
    def fadeoutduration(self):
        return parse_source_value(self._raw_data.get('fadeoutduration', 0.0))

    @property
    def exclusive(self):
        return self._raw_data.get('exclusive', "0")



class color_correction_volume(EnableDisable, BaseEntityBrush):

    @property
    def fadeduration(self):
        return parse_source_value(self._raw_data.get('fadeduration', 10.0))

    @property
    def maxweight(self):
        return parse_source_value(self._raw_data.get('maxweight', 1.0))

    @property
    def filename(self):
        return self._raw_data.get('filename', None)



class combine_mine(BaseEntityPoint):
    model = "models/props_combine/combine_mine01.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def bounce(self):
        return self._raw_data.get('bounce', "1")

    @property
    def explosiondelay(self):
        return parse_source_value(self._raw_data.get('explosiondelay', 0.5))

    @property
    def locksilently(self):
        return self._raw_data.get('locksilently', "1")

    @property
    def startdisarmed(self):
        return self._raw_data.get('startdisarmed', "0")

    @property
    def modification(self):
        return self._raw_data.get('modification', "0")



class commentary_auto(BaseEntityPoint):
    icon_sprite = "editor/ficool2/commentary_auto.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class comp_choreo_sceneset(BaseEntityPoint):
    icon_sprite = "editor/comp_choreo_sceneset.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def play_dings(self):
        return self._raw_data.get('play_dings', "1")

    @property
    def delay(self):
        return parse_source_value(self._raw_data.get('delay', 0.1))

    @property
    def only_once(self):
        return self._raw_data.get('only_once', "1")

    @property
    def busyactor(self):
        return self._raw_data.get('busyactor', "1")

    @property
    def onplayerdeath(self):
        return self._raw_data.get('onplayerdeath', "0")

    @property
    def scene01(self):
        return self._raw_data.get('scene01', None)

    @property
    def scene02(self):
        return self._raw_data.get('scene02', None)

    @property
    def scene03(self):
        return self._raw_data.get('scene03', None)

    @property
    def scene04(self):
        return self._raw_data.get('scene04', None)

    @property
    def scene05(self):
        return self._raw_data.get('scene05', None)

    @property
    def scene06(self):
        return self._raw_data.get('scene06', None)

    @property
    def scene07(self):
        return self._raw_data.get('scene07', None)

    @property
    def scene08(self):
        return self._raw_data.get('scene08', None)

    @property
    def scene09(self):
        return self._raw_data.get('scene09', None)

    @property
    def scene10(self):
        return self._raw_data.get('scene10', None)

    @property
    def scene11(self):
        return self._raw_data.get('scene11', None)

    @property
    def scene12(self):
        return self._raw_data.get('scene12', None)

    @property
    def scene13(self):
        return self._raw_data.get('scene13', None)

    @property
    def scene14(self):
        return self._raw_data.get('scene14', None)

    @property
    def scene15(self):
        return self._raw_data.get('scene15', None)

    @property
    def scene16(self):
        return self._raw_data.get('scene16', None)

    @property
    def scene17(self):
        return self._raw_data.get('scene17', None)

    @property
    def scene18(self):
        return self._raw_data.get('scene18', None)

    @property
    def scene19(self):
        return self._raw_data.get('scene19', None)

    @property
    def scene20(self):
        return self._raw_data.get('scene20', None)



class comp_flicker(BaseEntityPoint):
    icon_sprite = "editor/comp_flicker"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target_mdl(self):
        return self._raw_data.get('target_mdl', None)

    @property
    def mdl_skin_on(self):
        return parse_source_value(self._raw_data.get('mdl_skin_on', 0))

    @property
    def mdl_skin_off(self):
        return parse_source_value(self._raw_data.get('mdl_skin_off', 1))

    @property
    def total_time(self):
        return parse_source_value(self._raw_data.get('total_time', 1.5))

    @property
    def flicker_min(self):
        return parse_source_value(self._raw_data.get('flicker_min', 0.05))

    @property
    def flicker_max(self):
        return parse_source_value(self._raw_data.get('flicker_max', 0.2))

    @property
    def variance(self):
        return parse_source_value(self._raw_data.get('variance', 0.0))



class comp_vactube_end(BaseEntityPoint):
    viewport_model = "models/editor/vactubes/end_point.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 0))

    @property
    def autorespawn(self):
        return self._raw_data.get('autorespawn', "1")

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)

    @property
    def template(self):
        return self._raw_data.get('template', None)



class comp_vactube_start(BaseEntityPoint):
    viewport_model = "models/editor/vactubes/start_point.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def group(self):
        return self._raw_data.get('group', None)

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 800.0))

    @property
    def seed(self):
        return self._raw_data.get('seed', None)

    @property
    def timer(self):
        return self._raw_data.get('timer', "1")

    @property
    def time_min(self):
        return parse_source_value(self._raw_data.get('time_min', 0.15))

    @property
    def time_max(self):
        return parse_source_value(self._raw_data.get('time_max', 0.5))

    @property
    def linedivider_vacvisual(self):
        return self._raw_data.get('linedivider_vacvisual', None)

    @property
    def prop_fast_reflection(self):
        return self._raw_data.get('prop_fast_reflection', "0")

    @property
    def prop_disable_shadows(self):
        return self._raw_data.get('prop_disable_shadows', "1")

    @property
    def prop_disable_projtex(self):
        return self._raw_data.get('prop_disable_projtex', "0")



class env_alyxemp(Angles, BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def type(self):
        return parse_source_value(self._raw_data.get('type', 0))

    @property
    def endtargetname(self):
        return self._raw_data.get('endtargetname', None)



class env_ambient_light(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/color_correction.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def color(self):
        return parse_int_vector(self._raw_data.get('color', "255 255 255"))

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
    def fadeinduration(self):
        return parse_source_value(self._raw_data.get('fadeinduration', 0.0))

    @property
    def fadeoutduration(self):
        return parse_source_value(self._raw_data.get('fadeoutduration', 0.0))



class env_ar2explosion(BaseEntityPoint):
    icon_sprite = "editor/ts2do/env_ar2explosion.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def material(self):
        return self._raw_data.get('material', "particle/particle_noisesphere")



class env_blood(BaseEntityPoint):
    icon_sprite = "editor/env_blood.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spraydir(self):
        return parse_float_vector(self._raw_data.get('spraydir', "0 0 0"))

    @property
    def color(self):
        return self._raw_data.get('color', "0")

    @property
    def amount(self):
        return self._raw_data.get('amount', "100")

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class env_cascade_light(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/env_cascade_light.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def color(self):
        return parse_int_vector(self._raw_data.get('color', "255 255 255 1"))

    @property
    def maxshadowdistance(self):
        return parse_source_value(self._raw_data.get('maxshadowdistance', 400))

    @property
    def uselightenvangles(self):
        return self._raw_data.get('uselightenvangles', "1")



class env_citadel_energy_core(Angles, BaseEntityPoint):
    model = "models/editor/cone_helper.mdl"
    icon_sprite = "editor/env_citadel_energy_core.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def scale(self):
        return parse_source_value(self._raw_data.get('scale', 1))



class env_credits(BaseEntityPoint):
    icon_sprite = "editor/ts2do/env_credits.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class env_detail_controller(Angles, BaseEntityPoint):
    icon_sprite = "editor/ficool2/env_detail_controller.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def fademindist(self):
        return parse_source_value(self._raw_data.get('fademindist', 512))

    @property
    def fademaxdist(self):
        return parse_source_value(self._raw_data.get('fademaxdist', 1024))



class env_dof_controller(BaseEntityPoint):
    icon_sprite = "editor/env_dof_controller.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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
        return self._raw_data.get('focus_target', None)

    @property
    def focus_range(self):
        return parse_source_value(self._raw_data.get('focus_range', 200))



class env_dustpuff(Angles, BaseEntityPoint):
    model = "models/editor/env_dustpuff.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def scale(self):
        return parse_source_value(self._raw_data.get('scale', 8))

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 16))

    @property
    def color(self):
        return parse_int_vector(self._raw_data.get('color', "128 128 128"))



class env_entity_dissolver(BaseEntityPoint):
    icon_sprite = "editor/ficool2/env_entity_dissolver.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def magnitude(self):
        return parse_source_value(self._raw_data.get('magnitude', 250))

    @property
    def dissolvetype(self):
        return self._raw_data.get('dissolvetype', "0")



class env_entity_igniter(BaseEntityPoint):
    icon_sprite = "editor/ficool2/env_entity_igniter.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def lifetime(self):
        return parse_source_value(self._raw_data.get('lifetime', 10))



class env_entity_maker(Angles, BaseEntityPoint):
    icon_sprite = "editor/env_entity_maker.vmt"
    model = "models/editor/angle_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def entitytemplate(self):
        return self._raw_data.get('entitytemplate', None)

    @property
    def postspawnspeed(self):
        return parse_source_value(self._raw_data.get('postspawnspeed', 0))

    @property
    def postspawndirection(self):
        return parse_float_vector(self._raw_data.get('postspawndirection', "0 0 0"))

    @property
    def postspawndirectionvariance(self):
        return parse_source_value(self._raw_data.get('postspawndirectionvariance', 0.15))

    @property
    def postspawninheritangles(self):
        return self._raw_data.get('postspawninheritangles', "0")



class env_explosion(BaseEntityPoint):
    icon_sprite = "editor/env_explosion.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def imagnitude(self):
        return parse_source_value(self._raw_data.get('imagnitude', 100))

    @property
    def iradiusoverride(self):
        return parse_source_value(self._raw_data.get('iradiusoverride', 0))

    @property
    def fireballsprite(self):
        return self._raw_data.get('fireballsprite', "sprites/zerogxplode.spr")

    @property
    def rendermode(self):
        return self._raw_data.get('rendermode', "5")

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def ignoredentity(self):
        return self._raw_data.get('ignoredentity', None)

    @property
    def ignoredclass(self):
        return self._raw_data.get('ignoredclass', "0")



class env_fade(BaseEntityPoint):
    icon_sprite = "editor/env_fade"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def duration(self):
        return parse_source_value(self._raw_data.get('duration', 2))

    @property
    def holdtime(self):
        return parse_source_value(self._raw_data.get('holdtime', 0))

    @property
    def renderamt(self):
        return parse_source_value(self._raw_data.get('renderamt', 255))

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "0 0 0"))

    @property
    def reversefadeduration(self):
        return parse_source_value(self._raw_data.get('reversefadeduration', 2))



class env_fire(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/env_fire"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def ignitionpoint(self):
        return parse_source_value(self._raw_data.get('ignitionpoint', 32))

    @property
    def damagescale(self):
        return parse_source_value(self._raw_data.get('damagescale', 1.0))



class env_firesensor(BaseEntityPoint):
    icon_sprite = "editor/ficool2/env_firesensor.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def fireradius(self):
        return parse_source_value(self._raw_data.get('fireradius', 128))

    @property
    def heatlevel(self):
        return parse_source_value(self._raw_data.get('heatlevel', 32))

    @property
    def heattime(self):
        return parse_source_value(self._raw_data.get('heattime', 0))



class env_firesource(BaseEntityPoint):
    icon_sprite = "editor/env_firesource"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def fireradius(self):
        return parse_source_value(self._raw_data.get('fireradius', 128))

    @property
    def firedamage(self):
        return parse_source_value(self._raw_data.get('firedamage', 10))



class env_flare(Reflection, RenderFields, BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def scale(self):
        return parse_source_value(self._raw_data.get('scale', 1))

    @property
    def duration(self):
        return parse_source_value(self._raw_data.get('duration', 30))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class env_fog_controller(SystemLevelChoice, BaseEntityPoint):
    icon_sprite = "editor/fog_controller.vmt"
    model = "models/editor/cone_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def fogenable(self):
        return self._raw_data.get('fogenable', "1")

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
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def hdrcolorscale(self):
        return parse_source_value(self._raw_data.get('hdrcolorscale', 1))

    @property
    def zoomfogscale(self):
        return parse_source_value(self._raw_data.get('zoomfogscale', 1))



class env_funnel(BaseEntityPoint):
    icon_sprite = "editor/ficool2/env_funnel"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class env_global(BaseEntityPoint):
    icon_sprite = "editor/env_global.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def globalstate(self):
        return self._raw_data.get('globalstate', None)

    @property
    def initialstate(self):
        return self._raw_data.get('initialstate', "0")

    @property
    def counter(self):
        return parse_source_value(self._raw_data.get('counter', 0))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class env_gunfire(EnableDisable, BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

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



class env_hudhint(BaseEntityPoint):
    icon_sprite = "editor/env_hudhint.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def message(self):
        return self._raw_data.get('message', None)



class env_instructor_hint(BaseEntityPoint):
    icon_sprite = "editor/env_instructor_hint.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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
        return self._raw_data.get('hint_local_player_only', "0")

    @property
    def hint_timeout(self):
        return parse_source_value(self._raw_data.get('hint_timeout', 0))

    @property
    def hint_range(self):
        return parse_source_value(self._raw_data.get('hint_range', 0))

    @property
    def hint_gamepad_binding(self):
        return self._raw_data.get('hint_gamepad_binding', None)



class env_lightglow(RenderFields, BaseEntityPoint):
    model = "models/editor/axis_helper_thick.mdl"
    icon_sprite = "editor/ficool2/env_lightglow.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def verticalglowsize(self):
        return parse_source_value(self._raw_data.get('verticalglowsize', 30))

    @property
    def horizontalglowsize(self):
        return parse_source_value(self._raw_data.get('horizontalglowsize', 30))

    @property
    def mindist(self):
        return parse_source_value(self._raw_data.get('mindist', 500))

    @property
    def maxdist(self):
        return parse_source_value(self._raw_data.get('maxdist', 2000))

    @property
    def outermaxdist(self):
        return parse_source_value(self._raw_data.get('outermaxdist', 0))

    @property
    def glowproxysize(self):
        return parse_source_value(self._raw_data.get('glowproxysize', 2))

    @property
    def hdrcolorscale(self):
        return parse_source_value(self._raw_data.get('hdrcolorscale', 0.5))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class env_lightrail_endpoint(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def small_fx_scale(self):
        return parse_source_value(self._raw_data.get('small_fx_scale', 1))

    @property
    def large_fx_scale(self):
        return parse_source_value(self._raw_data.get('large_fx_scale', 1))



class env_message(BaseEntityPoint):
    icon_sprite = "editor/ts2do/env_message.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def message(self):
        return self._raw_data.get('message', None)

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def messagesound(self):
        return self._raw_data.get('messagesound', None)

    @property
    def messagevolume(self):
        return self._raw_data.get('messagevolume', "10")

    @property
    def messageattenuation(self):
        return self._raw_data.get('messageattenuation', "0")



class env_microphone(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/env_microphone.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def speakername(self):
        return self._raw_data.get('speakername', None)

    @property
    def listenfilter(self):
        return self._raw_data.get('listenfilter', None)

    @property
    def speaker_dsp_preset(self):
        return self._raw_data.get('speaker_dsp_preset', "0")

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def sensitivity(self):
        return parse_source_value(self._raw_data.get('sensitivity', 1))

    @property
    def smoothfactor(self):
        return parse_source_value(self._raw_data.get('smoothfactor', 0))

    @property
    def maxrange(self):
        return parse_source_value(self._raw_data.get('maxrange', 240))



class env_movieexplosion(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class env_muzzleflash(BaseEntityPoint):
    model = "models/editor/env_muzzleflash.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def parentattachment(self):
        return self._raw_data.get('parentattachment', None)

    @property
    def scale(self):
        return parse_source_value(self._raw_data.get('scale', 1))



class env_particle_performance_monitor(BaseEntityPoint):
    icon_sprite = "editor/ficool2/env_particle_performance_monitor"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class env_particlelight(BaseEntityPoint):
    icon_sprite = "editor/ficool2/env_particlelight.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def color(self):
        return parse_int_vector(self._raw_data.get('color', "255 0 0"))

    @property
    def intensity(self):
        return parse_source_value(self._raw_data.get('intensity', 5000))

    @property
    def directional(self):
        return self._raw_data.get('directional', "0")

    @property
    def psname(self):
        return self._raw_data.get('psname', None)



class env_particlescript(BaseEntityPoint):
    icon_sprite = "editor/ficool2/env_effectscript"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', None)



class env_physexplosion(BaseEntityPoint):
    icon_sprite = "editor/env_physexplosion.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def magnitude(self):
        return self._raw_data.get('magnitude', "100")

    @property
    def radius(self):
        return self._raw_data.get('radius', "0")

    @property
    def targetentityname(self):
        return self._raw_data.get('targetentityname', None)

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def inner_radius(self):
        return parse_source_value(self._raw_data.get('inner_radius', 0))



class env_physimpact(BaseEntityPoint):
    icon_sprite = "editor/ficool2/env_physimpact.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def magnitude(self):
        return parse_source_value(self._raw_data.get('magnitude', 100))

    @property
    def distance(self):
        return parse_source_value(self._raw_data.get('distance', 0))

    @property
    def directionentityname(self):
        return self._raw_data.get('directionentityname', None)

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class env_player_surface_trigger(BaseEntityPoint):
    icon_sprite = "editor/env_player_surface_trigger.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def gamematerial(self):
        return self._raw_data.get('gamematerial', "0")



class env_player_viewfinder(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class env_portal_credits(BaseEntityPoint):
    icon_sprite = "editor/env_portal_credits"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class env_projectedtexture(SystemLevelChoice, BaseEntityPoint):
    model = "models/editor/cone_helper.mdl"
    icon_sprite = "editor/env_projectedtexture"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

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
        return self._raw_data.get('enableshadows', "1")

    @property
    def shadowquality(self):
        return self._raw_data.get('shadowquality', "1")

    @property
    def _dynamicshadowallocation(self):
        return self._raw_data.get('_dynamicshadowallocation', "0")

    @property
    def _initialshadowsize(self):
        return parse_source_value(self._raw_data.get('_initialshadowsize', 7))

    @property
    def _shadowscale(self):
        return parse_source_value(self._raw_data.get('_shadowscale', 1.0))

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
    def lightcolor(self):
        return parse_int_vector(self._raw_data.get('lightcolor', "255 255 255 200"))

    @property
    def brightnessscale(self):
        return parse_source_value(self._raw_data.get('brightnessscale', 1.0))

    @property
    def cameraspace(self):
        return self._raw_data.get('cameraspace', "0")

    @property
    def colortransitiontime(self):
        return parse_source_value(self._raw_data.get('colortransitiontime', 0.5))

    @property
    def texturename(self):
        return self._raw_data.get('texturename', "effects/flashlight001")

    @property
    def moviename(self):
        return self._raw_data.get('moviename', None)

    @property
    def textureframe(self):
        return parse_source_value(self._raw_data.get('textureframe', 0))

    @property
    def volumetric(self):
        return self._raw_data.get('volumetric', "0")

    @property
    def volumetricintensity(self):
        return parse_source_value(self._raw_data.get('volumetricintensity', 1.0))

    @property
    def style(self):
        return self._raw_data.get('style', "0")

    @property
    def pattern(self):
        return self._raw_data.get('pattern', None)



class env_rockettrail(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class env_rotorwash_emitter(BaseEntityPoint):
    icon_sprite = "editor/env_rotorwash_emitter"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def altitude(self):
        return parse_source_value(self._raw_data.get('altitude', 1024))



class env_screeneffect(BaseEntityPoint):
    icon_sprite = "editor/ficool2/env_screeneffect"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def type(self):
        return self._raw_data.get('type', "0")



class env_screenoverlay(BaseEntityPoint):
    icon_sprite = "editor/ficool2/env_screenoverlay"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def overlayname1(self):
        return self._raw_data.get('overlayname1', None)

    @property
    def overlaytime1(self):
        return parse_source_value(self._raw_data.get('overlaytime1', 1.0))

    @property
    def overlayname2(self):
        return self._raw_data.get('overlayname2', None)

    @property
    def overlaytime2(self):
        return parse_source_value(self._raw_data.get('overlaytime2', 1.0))

    @property
    def overlayname3(self):
        return self._raw_data.get('overlayname3', None)

    @property
    def overlaytime3(self):
        return parse_source_value(self._raw_data.get('overlaytime3', 1.0))

    @property
    def overlayname4(self):
        return self._raw_data.get('overlayname4', None)

    @property
    def overlaytime4(self):
        return parse_source_value(self._raw_data.get('overlaytime4', 1.0))

    @property
    def overlayname5(self):
        return self._raw_data.get('overlayname5', None)

    @property
    def overlaytime5(self):
        return parse_source_value(self._raw_data.get('overlaytime5', 1.0))

    @property
    def overlayname6(self):
        return self._raw_data.get('overlayname6', None)

    @property
    def overlaytime6(self):
        return parse_source_value(self._raw_data.get('overlaytime6', 1.0))

    @property
    def overlayname7(self):
        return self._raw_data.get('overlayname7', None)

    @property
    def overlaytime7(self):
        return parse_source_value(self._raw_data.get('overlaytime7', 1.0))

    @property
    def overlayname8(self):
        return self._raw_data.get('overlayname8', None)

    @property
    def overlaytime8(self):
        return parse_source_value(self._raw_data.get('overlaytime8', 1.0))

    @property
    def overlayname9(self):
        return self._raw_data.get('overlayname9', None)

    @property
    def overlaytime9(self):
        return parse_source_value(self._raw_data.get('overlaytime9', 1.0))

    @property
    def overlayname10(self):
        return self._raw_data.get('overlayname10', None)

    @property
    def overlaytime10(self):
        return parse_source_value(self._raw_data.get('overlaytime10', 1.0))



class env_shake(BaseEntityPoint):
    icon_sprite = "editor/env_shake.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

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



class env_smokestack(BaseEntityPoint):
    icon_sprite = "editor/env_smokestack.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def initialstate(self):
        return self._raw_data.get('initialstate', "0")

    @property
    def basespread(self):
        return parse_source_value(self._raw_data.get('basespread', 20))

    @property
    def spreadspeed(self):
        return parse_source_value(self._raw_data.get('spreadspeed', 15))

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 30))

    @property
    def startsize(self):
        return parse_source_value(self._raw_data.get('startsize', 20))

    @property
    def endsize(self):
        return parse_source_value(self._raw_data.get('endsize', 30))

    @property
    def rate(self):
        return parse_source_value(self._raw_data.get('rate', 20))

    @property
    def jetlength(self):
        return parse_source_value(self._raw_data.get('jetlength', 180))

    @property
    def windangle(self):
        return parse_source_value(self._raw_data.get('windangle', 0))

    @property
    def windspeed(self):
        return parse_source_value(self._raw_data.get('windspeed', 0))

    @property
    def smokematerial(self):
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



class env_smoketrail(BaseEntityPoint):
    icon_sprite = "editor/ficool2/env_smoketrail"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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



class env_soundscape(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/env_soundscape.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 128))

    @property
    def soundscape(self):
        return self._raw_data.get('soundscape', "Nothing")

    @property
    def position0(self):
        return self._raw_data.get('position0', None)

    @property
    def position1(self):
        return self._raw_data.get('position1', None)

    @property
    def position2(self):
        return self._raw_data.get('position2', None)

    @property
    def position3(self):
        return self._raw_data.get('position3', None)

    @property
    def position4(self):
        return self._raw_data.get('position4', None)

    @property
    def position5(self):
        return self._raw_data.get('position5', None)

    @property
    def position6(self):
        return self._raw_data.get('position6', None)

    @property
    def position7(self):
        return self._raw_data.get('position7', None)



class env_soundscape_proxy(BaseEntityPoint):
    icon_sprite = "editor/env_soundscape_proxy.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def mainsoundscapename(self):
        return self._raw_data.get('mainsoundscapename', None)

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 128))



class env_spark(BaseEntityPoint):
    icon_sprite = "editor/env_spark.vmt"
    model = "models/editor/cone_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def maxdelay(self):
        return self._raw_data.get('maxdelay', "0")

    @property
    def magnitude(self):
        return self._raw_data.get('magnitude', "1")

    @property
    def traillength(self):
        return self._raw_data.get('traillength', "1")

    @property
    def sparktype(self):
        return self._raw_data.get('sparktype', "0")

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def sound(self):
        return self._raw_data.get('sound', "DoSpark")



class env_speaker(ResponseContext, BaseEntityPoint):
    icon_sprite = "editor/ambient_generic.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def delaymin(self):
        return self._raw_data.get('delaymin', "15")

    @property
    def delaymax(self):
        return self._raw_data.get('delaymax', "135")

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def rulescript(self):
        return self._raw_data.get('rulescript', None)

    @property
    def concept(self):
        return self._raw_data.get('concept', None)



class env_splash(BaseEntityPoint):
    model = "models/editor/env_splash.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def scale(self):
        return parse_source_value(self._raw_data.get('scale', 8.0))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class env_sporeexplosion(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/ficool2/env_sporeexplosion.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnrate(self):
        return parse_source_value(self._raw_data.get('spawnrate', 25))



class env_sprite(RenderFields, SystemLevelChoice, BaseEntityPoint):
    model = "models/editor/axis_helper_white.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def framerate(self):
        return parse_source_value(self._raw_data.get('framerate', 10.0))

    @property
    def model(self):
        return self._raw_data.get('model', "sprites/glow01.spr")

    @property
    def scale(self):
        return parse_source_value(self._raw_data.get('scale', 0.25))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def glowproxysize(self):
        return parse_source_value(self._raw_data.get('glowproxysize', 2.0))

    @property
    def hdrcolorscale(self):
        return parse_source_value(self._raw_data.get('hdrcolorscale', 0.7))



class env_spritetrail(RenderFields, BaseEntityPoint):
    icon_sprite = "editor/env_spritetrail.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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



class env_starfield(BaseEntityPoint):
    icon_sprite = "editor/ts2do/env_starfield.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class env_steam(RenderFields, BaseEntityPoint):
    viewport_model = "models/editor/spot_cone_fixed.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def initialstate(self):
        return self._raw_data.get('initialstate', "0")

    @property
    def type(self):
        return self._raw_data.get('type', "0")

    @property
    def spreadspeed(self):
        return parse_source_value(self._raw_data.get('spreadspeed', 15))

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 120))

    @property
    def startsize(self):
        return parse_source_value(self._raw_data.get('startsize', 10))

    @property
    def endsize(self):
        return parse_source_value(self._raw_data.get('endsize', 25))

    @property
    def rate(self):
        return parse_source_value(self._raw_data.get('rate', 26))

    @property
    def jetlength(self):
        return parse_source_value(self._raw_data.get('jetlength', 80))

    @property
    def rollspeed(self):
        return parse_source_value(self._raw_data.get('rollspeed', 8))



class env_sun(RenderFields, SystemLevelChoice, BaseEntityPoint):
    icon_sprite = "editor/env_sun.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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
    def hdrcolorscale(self):
        return parse_source_value(self._raw_data.get('hdrcolorscale', 0.5))

    @property
    def glowdistancescale(self):
        return parse_source_value(self._raw_data.get('glowdistancescale', 0.99))



class env_texturetoggle(BaseEntityPoint):
    icon_sprite = "editor/env_texturetoggle.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)



class env_tilt(BaseEntityPoint):
    model = "models/editor/axis_helper_thick.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 500))

    @property
    def duration(self):
        return parse_source_value(self._raw_data.get('duration', 1))

    @property
    def tilttime(self):
        return parse_source_value(self._raw_data.get('tilttime', 2.5))



class env_tonemap_controller(BaseEntityPoint):
    icon_sprite = "editor/env_tonemap_controller.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class env_viewpunch(BaseEntityPoint):
    icon_sprite = "editor/ficool2/env_viewpunch.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def punchangle(self):
        return parse_float_vector(self._raw_data.get('punchangle', "0 0 90"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 500))



class env_wind(BaseEntityPoint):
    icon_sprite = "editor/env_wind.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

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



class env_zoom(BaseEntityPoint):
    icon_sprite = "editor/ficool2/env_zoom.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def rate(self):
        return parse_source_value(self._raw_data.get('rate', 1.0))

    @property
    def fov(self):
        return parse_source_value(self._raw_data.get('fov', 75))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class filter_base(BaseEntityPoint):
    icon_sprite = "editor/ficool2/filter_base.vmt"

    @property
    def negated(self):
        return self._raw_data.get('negated', "0")

    @property
    def passcallerwhentested(self):
        return self._raw_data.get('passcallerwhentested', "0")



class fog_volume(EnableDisable, BaseEntityBrush):

    @property
    def fogname(self):
        return self._raw_data.get('fogname', None)

    @property
    def postprocessname(self):
        return self._raw_data.get('postprocessname', None)

    @property
    def colorcorrectionname(self):
        return self._raw_data.get('colorcorrectionname', None)



class func_areaportal(BaseEntity):

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def startopen(self):
        return self._raw_data.get('startopen', "1")

    @property
    def portalversion(self):
        return parse_source_value(self._raw_data.get('portalversion', 1))



class func_areaportalwindow(BaseEntity):

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def fadestartdist(self):
        return parse_source_value(self._raw_data.get('fadestartdist', 128))

    @property
    def fadedist(self):
        return parse_source_value(self._raw_data.get('fadedist', 512))

    @property
    def translucencylimit(self):
        return parse_source_value(self._raw_data.get('translucencylimit', 0))

    @property
    def backgroundbmodel(self):
        return self._raw_data.get('backgroundbmodel', None)

    @property
    def portalversion(self):
        return parse_source_value(self._raw_data.get('portalversion', 1))



class func_clip_vphysics(BaseEntityBrush):

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)

    @property
    def startdisabled(self):
        return self._raw_data.get('startdisabled', "0")



class func_ladderendpoint(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)



class func_nav_avoidance_obstacle(BaseEntityBrush):

    @property
    def startdisabled(self):
        return self._raw_data.get('startdisabled', "0")



class func_nav_blocker(BaseEntityBrush):

    @property
    def startdisabled(self):
        return self._raw_data.get('startdisabled', "0")



class func_noportal_volume(Origin, BaseEntityBrush):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class func_occluder(BaseEntity):

    @property
    def startactive(self):
        return self._raw_data.get('startactive', "1")



class func_portal_bumper(Origin, BaseEntityBrush):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class func_portal_detector(Origin, BaseEntityBrush):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def linkagegroupid(self):
        return parse_source_value(self._raw_data.get('linkagegroupid', 0))

    @property
    def checkallids(self):
        return self._raw_data.get('checkallids', "0")



class func_portal_orientation(EnableDisable, BaseEntityBrush):

    @property
    def anglestoface(self):
        return parse_float_vector(self._raw_data.get('anglestoface', "0 0 0"))

    @property
    def matchlinkedangles(self):
        return self._raw_data.get('matchlinkedangles', "0")



class func_precipitation_blocker(BaseEntityBrush):
    pass


class func_proprrespawnzone(BaseEntityBrush):
    pass


class func_useableladder(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def point0(self):
        return parse_float_vector(self._raw_data.get('point0', None))

    @property
    def point1(self):
        return parse_float_vector(self._raw_data.get('point1', None))

    @property
    def startdisabled(self):
        return self._raw_data.get('startdisabled', "0")

    @property
    def laddersurfaceproperties(self):
        return self._raw_data.get('laddersurfaceproperties', None)



class func_vehicleclip(BaseEntityBrush):
    pass


class func_wall(BaseEntityBrush):
    pass


class game_end(MasterEnt, BaseEntityPoint):
    icon_sprite = "editor/game_end.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class game_gib_manager(BaseEntityPoint):
    icon_sprite = "editor/ficool2/game_gib_manager"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def maxpieces(self):
        return parse_source_value(self._raw_data.get('maxpieces', -1))

    @property
    def allownewgibs(self):
        return self._raw_data.get('allownewgibs', "0")



class game_globalvars(EnableDisable, BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class game_player_equip(MasterEnt, BaseEntityPoint):
    icon_sprite = "editor/ficool2/game_player_equip"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class game_player_team(MasterEnt, BaseEntityPoint):
    icon_sprite = "editor/ficool2/game_player_team"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def target(self):
        return self._raw_data.get('target', None)



class game_ragdoll_manager(BaseEntityPoint):
    icon_sprite = "editor/ficool2/game_ragdoll_manager"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def maxragdollcount(self):
        return parse_source_value(self._raw_data.get('maxragdollcount', -1))

    @property
    def saveimportant(self):
        return self._raw_data.get('saveimportant', "0")



class game_score(MasterEnt, BaseEntityPoint):
    icon_sprite = "editor/ficool2/game_score"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def points(self):
        return parse_source_value(self._raw_data.get('points', 1))



class game_text(MasterEnt, BaseEntityPoint):
    icon_sprite = "editor/game_text.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def message(self):
        return self._raw_data.get('message', None)

    @property
    def x(self):
        return parse_source_value(self._raw_data.get('x', -1))

    @property
    def y(self):
        return parse_source_value(self._raw_data.get('y', 0.6))

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
        return parse_source_value(self._raw_data.get('fadein', 1.5))

    @property
    def fadeout(self):
        return parse_source_value(self._raw_data.get('fadeout', 0.5))

    @property
    def holdtime(self):
        return parse_source_value(self._raw_data.get('holdtime', 1.2))

    @property
    def fxtime(self):
        return parse_source_value(self._raw_data.get('fxtime', 0.25))

    @property
    def channel(self):
        return self._raw_data.get('channel', "1")



class game_ui(BaseEntityPoint):
    icon_sprite = "editor/ficool2/game_ui.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def fieldofview(self):
        return parse_source_value(self._raw_data.get('fieldofview', -1.0))



class game_weapon_manager(BaseEntityPoint):
    icon_sprite = "editor/ficool2/game_weapon_manager"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def weaponname(self):
        return self._raw_data.get('weaponname', None)

    @property
    def maxpieces(self):
        return parse_source_value(self._raw_data.get('maxpieces', 0))

    @property
    def ammomod(self):
        return parse_source_value(self._raw_data.get('ammomod', 1))



class game_zone_player(BaseEntityBrush):
    pass


class gibshooterbase(BaseEntityPoint):

    @property
    def m_igibs(self):
        return parse_source_value(self._raw_data.get('m_igibs', 3))

    @property
    def delay(self):
        return parse_source_value(self._raw_data.get('delay', 0))

    @property
    def gibangles(self):
        return parse_float_vector(self._raw_data.get('gibangles', "0 0 0"))

    @property
    def gibanglevelocity(self):
        return parse_source_value(self._raw_data.get('gibanglevelocity', 0))

    @property
    def m_flvelocity(self):
        return parse_source_value(self._raw_data.get('m_flvelocity', 200))

    @property
    def m_flvariance(self):
        return parse_source_value(self._raw_data.get('m_flvariance', 0.15))

    @property
    def m_flgiblife(self):
        return parse_source_value(self._raw_data.get('m_flgiblife', 4))

    @property
    def simulation(self):
        return self._raw_data.get('simulation', "0")

    @property
    def lightingorigin(self):
        return self._raw_data.get('lightingorigin', None)

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class hammer_updateignorelist(BaseEntityPoint):
    icon_sprite = "editor/ficool2/hammer_updateignorelist"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def ignoredname01(self):
        return self._raw_data.get('ignoredname01', None)

    @property
    def ignoredname02(self):
        return self._raw_data.get('ignoredname02', None)

    @property
    def ignoredname03(self):
        return self._raw_data.get('ignoredname03', None)

    @property
    def ignoredname04(self):
        return self._raw_data.get('ignoredname04', None)

    @property
    def ignoredname05(self):
        return self._raw_data.get('ignoredname05', None)

    @property
    def ignoredname06(self):
        return self._raw_data.get('ignoredname06', None)

    @property
    def ignoredname07(self):
        return self._raw_data.get('ignoredname07', None)

    @property
    def ignoredname08(self):
        return self._raw_data.get('ignoredname08', None)

    @property
    def ignoredname09(self):
        return self._raw_data.get('ignoredname09', None)

    @property
    def ignoredname10(self):
        return self._raw_data.get('ignoredname10', None)

    @property
    def ignoredname11(self):
        return self._raw_data.get('ignoredname11', None)

    @property
    def ignoredname12(self):
        return self._raw_data.get('ignoredname12', None)

    @property
    def ignoredname13(self):
        return self._raw_data.get('ignoredname13', None)

    @property
    def ignoredname14(self):
        return self._raw_data.get('ignoredname14', None)

    @property
    def ignoredname15(self):
        return self._raw_data.get('ignoredname15', None)

    @property
    def ignoredname16(self):
        return self._raw_data.get('ignoredname16', None)



class info_apc_missile_hint(EnableDisable, Angles, BaseEntityBrush):

    @property
    def target(self):
        return self._raw_data.get('target', None)



class info_camera_link(BaseEntityPoint):
    model = "models/editor/camera.mdl"
    icon_sprite = "editor/info_camera_link.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def pointcamera(self):
        return self._raw_data.get('pointcamera', None)



class info_constraint_anchor(BaseEntityPoint):
    icon_sprite = "editor/info_constraint_anchor.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def massscale(self):
        return parse_source_value(self._raw_data.get('massscale', 1))



class info_coop_spawn(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def enabled(self):
        return self._raw_data.get('enabled', "1")

    @property
    def startingteam(self):
        return self._raw_data.get('startingteam', "0")

    @property
    def model(self):
        return self._raw_data.get('model', "models/editor/playerstart.mdl")

    @property
    def forcegunonspawn(self):
        return self._raw_data.get('forcegunonspawn', "0")



class info_darknessmode_lightsource(EnableDisable, BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def lightradius(self):
        return parse_source_value(self._raw_data.get('lightradius', 256.0))



class info_game_event_proxy(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/info_game_event_proxy.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def event_name(self):
        return self._raw_data.get('event_name', None)

    @property
    def range(self):
        return parse_source_value(self._raw_data.get('range', 512))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class info_hint(HintNode, BaseEntityPoint):
    model = "models/editor/node_hint.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class info_ladder_dismount(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)



class info_landmark(BaseEntityPoint):
    icon_sprite = "editor/info_landmark"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class info_landmark_entry(BaseEntityPoint):
    icon_sprite = "editor/info_landmark"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class info_landmark_exit(BaseEntityPoint):
    icon_sprite = "editor/info_landmark"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class info_lighting_relative(BaseEntityPoint):
    icon_sprite = "editor/ficool2/info_lighting_relative.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def lightinglandmark(self):
        return self._raw_data.get('lightinglandmark', None)



class info_node_air_hint(HintNode, BaseEntityPoint):
    model = "models/editor/air_node_hint.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def nodeheight(self):
        return parse_source_value(self._raw_data.get('nodeheight', 0))



class info_node_climb(HintNode, BaseEntityPoint):
    model = "models/editor/climb_node.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class info_node_hint(HintNode, BaseEntityPoint):
    model = "models/editor/ground_node_hint.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class info_node_link(BaseEntityPoint):
    icon_sprite = "editor/info_node_link.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def startnode(self):
        return parse_source_value(self._raw_data.get('startnode', None))

    @property
    def endnode(self):
        return parse_source_value(self._raw_data.get('endnode', None))

    @property
    def initialstate(self):
        return self._raw_data.get('initialstate', "1")

    @property
    def linktype(self):
        return self._raw_data.get('linktype', "1")

    @property
    def allowuse(self):
        return self._raw_data.get('allowuse', None)

    @property
    def invertallow(self):
        return self._raw_data.get('invertallow', "0")

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def precisemovement(self):
        return self._raw_data.get('precisemovement', "0")

    @property
    def priority(self):
        return self._raw_data.get('priority', "0")



class info_node_link_controller(BaseEntityPoint):
    icon_sprite = "editor/info_node_link_controller.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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
    def allowuse(self):
        return self._raw_data.get('allowuse', None)

    @property
    def invertallow(self):
        return self._raw_data.get('invertallow', "0")

    @property
    def priority(self):
        return self._raw_data.get('priority', "0")



class info_npc_spawn_destination(BaseEntityPoint):
    icon_sprite = "editor/ficool2/info_npc_spawn_destination.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def reusedelay(self):
        return parse_source_value(self._raw_data.get('reusedelay', 1))

    @property
    def renamenpc(self):
        return self._raw_data.get('renamenpc', None)



class info_null(BaseEntityPoint):
    icon_sprite = "editor/ficool2/info_null.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class info_overlay(BaseEntityPoint):
    model = "models/editor/overlay_helper_box.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def material(self):
        return self._raw_data.get('material', None)

    @property
    def sides(self):
        return self._raw_data.get('sides', None)

    @property
    def renderorder(self):
        return self._raw_data.get('renderorder', "0")

    @property
    def tint(self):
        return parse_int_vector(self._raw_data.get('tint', "255 255 255 255"))

    @property
    def startu(self):
        return parse_source_value(self._raw_data.get('startu', 0.0))

    @property
    def endu(self):
        return parse_source_value(self._raw_data.get('endu', 1.0))

    @property
    def startv(self):
        return parse_source_value(self._raw_data.get('startv', 0.0))

    @property
    def endv(self):
        return parse_source_value(self._raw_data.get('endv', 1.0))

    @property
    def basisorigin(self):
        return parse_float_vector(self._raw_data.get('basisorigin', None))

    @property
    def basisu(self):
        return parse_float_vector(self._raw_data.get('basisu', None))

    @property
    def basisv(self):
        return parse_float_vector(self._raw_data.get('basisv', None))

    @property
    def basisnormal(self):
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



class info_paint_sprayer(BasePaintType, BaseEntityPoint):
    model = "models/editor/info_paint_sprayer.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def maxblobcount(self):
        return parse_source_value(self._raw_data.get('maxblobcount', 250))

    @property
    def light_position_name(self):
        return self._raw_data.get('light_position_name', None)

    @property
    def start_active(self):
        return self._raw_data.get('start_active', "0")

    @property
    def silent(self):
        return self._raw_data.get('silent', "0")

    @property
    def drawonly(self):
        return self._raw_data.get('drawonly', "0")

    @property
    def ambientsound(self):
        return self._raw_data.get('ambientsound', "0")

    @property
    def blobs_per_second(self):
        return parse_source_value(self._raw_data.get('blobs_per_second', 20))

    @property
    def min_speed(self):
        return parse_source_value(self._raw_data.get('min_speed', 100))

    @property
    def max_speed(self):
        return parse_source_value(self._raw_data.get('max_speed', 100))

    @property
    def blob_spread_radius(self):
        return parse_source_value(self._raw_data.get('blob_spread_radius', 0))

    @property
    def blob_spread_angle(self):
        return parse_source_value(self._raw_data.get('blob_spread_angle', 8))

    @property
    def blob_streak_percentage(self):
        return parse_source_value(self._raw_data.get('blob_streak_percentage', 0))

    @property
    def min_streak_time(self):
        return parse_source_value(self._raw_data.get('min_streak_time', 0.2))

    @property
    def max_streak_time(self):
        return parse_source_value(self._raw_data.get('max_streak_time', 0.5))

    @property
    def min_streak_speed_dampen(self):
        return parse_source_value(self._raw_data.get('min_streak_speed_dampen', 500))

    @property
    def max_streak_speed_dampen(self):
        return parse_source_value(self._raw_data.get('max_streak_speed_dampen', 1000))

    @property
    def start_radius_min(self):
        return parse_source_value(self._raw_data.get('start_radius_min', 0.5))

    @property
    def start_radius_max(self):
        return parse_source_value(self._raw_data.get('start_radius_max', 0.7))

    @property
    def end_radius_min(self):
        return parse_source_value(self._raw_data.get('end_radius_min', 0.5))

    @property
    def end_radius_max(self):
        return parse_source_value(self._raw_data.get('end_radius_max', 0.7))

    @property
    def radius_grow_time_min(self):
        return parse_source_value(self._raw_data.get('radius_grow_time_min', 0.5))

    @property
    def radius_grow_time_max(self):
        return parse_source_value(self._raw_data.get('radius_grow_time_max', 1))



class info_particle_system(Reflection, BaseEntityPoint):
    model = "models/editor/cone_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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



class info_placement_helper(EnableDisable, BaseEntityPoint):
    viewport_model = "models/editor/placement_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 16))

    @property
    def proxy_name(self):
        return self._raw_data.get('proxy_name', None)

    @property
    def attach_target_name(self):
        return self._raw_data.get('attach_target_name', None)

    @property
    def snap_to_helper_angles(self):
        return self._raw_data.get('snap_to_helper_angles', "0")

    @property
    def force_placement(self):
        return self._raw_data.get('force_placement', "0")



class info_player_deathmatch(BaseEntityPoint):
    model = "models/editor/playerstart.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class info_player_ping_detector(BaseEntityPoint):
    icon_sprite = "editor/info_player_ping_detector.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def functankname(self):
        return self._raw_data.get('functankname', None)

    @property
    def teamtolookat(self):
        return self._raw_data.get('teamtolookat', "2")

    @property
    def enabled(self):
        return self._raw_data.get('enabled', "1")



class info_player_start(BaseEntityPoint):
    model = "models/editor/playerstart.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class info_playtest_manager(BaseEntityPoint):
    icon_sprite = "editor/info_target.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class info_portal_gamerules(BaseEntityPoint):
    icon_sprite = "editor/info_portal_gamerules.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def enableregen(self):
        return self._raw_data.get('enableregen', "1")

    @property
    def equipboots(self):
        return self._raw_data.get('equipboots', "1")

    @property
    def equipportalgun(self):
        return self._raw_data.get('equipportalgun', "0")

    @property
    def equippaintgun(self):
        return self._raw_data.get('equippaintgun', "0")

    @property
    def showportalpaintprops(self):
        return self._raw_data.get('showportalpaintprops', "0")

    @property
    def maxhealth(self):
        return parse_source_value(self._raw_data.get('maxhealth', 100))



class info_projecteddecal(BaseEntityPoint):
    model = "models/editor/decal_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def texture(self):
        return self._raw_data.get('texture', None)

    @property
    def distance(self):
        return parse_source_value(self._raw_data.get('distance', 64))



class info_radar_target(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/info_target.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 6000))

    @property
    def type(self):
        return self._raw_data.get('type', "0")

    @property
    def mode(self):
        return self._raw_data.get('mode', "0")



class info_radial_link_controller(BaseEntityPoint):
    icon_sprite = "editor/info_radial_link_controller.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 120))



class info_snipertarget(BaseEntityPoint):
    icon_sprite = "editor/info_target.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 2))

    @property
    def groupname(self):
        return self._raw_data.get('groupname', None)

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class info_target(BaseEntityPoint):
    icon_sprite = "editor/info_target.vmt"
    model = "models/editor/axis_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class info_target_gunshipcrash(BaseEntityPoint):
    icon_sprite = "editor/info_target.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class info_target_helicopter_crash(BaseEntityPoint):
    icon_sprite = "editor/info_target.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class info_target_instructor_hint(BaseEntityPoint):
    model = "models/editor/axis_helper.mdl"
    icon_sprite = "editor/info_target_instructor_hint.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class info_target_personality_sphere(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def sphereline(self):
        return self._raw_data.get('sphereline', None)

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 16))



class info_target_vehicle_transition(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/info_target.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class info_teleport_destination(BaseEntityPoint):
    model = "models/editor/playerstart.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class info_teleporter_countdown(BaseEntityPoint):
    icon_sprite = "editor/info_target.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class infodecal(BaseEntityPoint):
    model = "models/editor/decal_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def texture(self):
        return self._raw_data.get('texture', None)

    @property
    def lowpriority(self):
        return self._raw_data.get('lowpriority', "1")



class keyframe_rope(RopeKeyFrame, BaseEntityPoint):
    model = "models/editor/axis_helper_thick.mdl"
    icon_sprite = "editor/keyframe_rope"
    pass


class keyframe_track(KeyFrame, BaseEntityPoint):
    pass


class light_directional(BaseLight, BaseEntityPoint):
    icon_sprite = "editor/light_directional.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def pitch(self):
        return parse_source_value(self._raw_data.get('pitch', 0))

    @property
    def sunspreadangle(self):
        return parse_source_value(self._raw_data.get('sunspreadangle', 0))



class light_dynamic(BaseEntityPoint):
    icon_sprite = "editor/ficool2/light_dynamic.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class light_environment(BaseLight, Angles, BaseEntityPoint):
    icon_sprite = "editor/ficool2/light_environment.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def pitch(self):
        return parse_source_value(self._raw_data.get('pitch', 0))

    @property
    def _ambient(self):
        return parse_int_vector(self._raw_data.get('_ambient', "255 255 255 20"))

    @property
    def _ambienthdr(self):
        return parse_int_vector(self._raw_data.get('_ambienthdr', "-1 -1 -1 1"))

    @property
    def _ambientscalehdr(self):
        return parse_source_value(self._raw_data.get('_ambientscalehdr', 1))

    @property
    def sunspreadangle(self):
        return parse_source_value(self._raw_data.get('sunspreadangle', 5))



class linked_portal_door(LinkedPortalDoor, ToggleDraw, BaseEntityPoint):
    model = "models/editor/angle_helper.mdl"
    icon_sprite = "editor/portal_dual.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def partnername(self):
        return self._raw_data.get('partnername', None)

    @property
    def width(self):
        return parse_source_value(self._raw_data.get('width', 128))

    @property
    def height(self):
        return parse_source_value(self._raw_data.get('height', 128))

    @property
    def isstatic(self):
        return self._raw_data.get('isstatic', "0")

    @property
    def startactive(self):
        return self._raw_data.get('startactive', "1")



class logic_achievement(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/logic_achievement"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def achievementname(self):
        return self._raw_data.get('achievementname', None)



class logic_active_autosave(BaseEntityPoint):
    icon_sprite = "editor/logic_active_autosave.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def minimumhitpoints(self):
        return parse_source_value(self._raw_data.get('minimumhitpoints', 30))

    @property
    def triggerhitpoints(self):
        return parse_source_value(self._raw_data.get('triggerhitpoints', 75))

    @property
    def timetotrigget(self):
        return parse_source_value(self._raw_data.get('timetotrigget', 0))

    @property
    def dangeroustime(self):
        return parse_source_value(self._raw_data.get('dangeroustime', 10))



class logic_auto(BaseEntityPoint):
    icon_sprite = "editor/logic_auto.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def globalstate(self):
        return self._raw_data.get('globalstate', None)



class logic_autosave(BaseEntityPoint):
    icon_sprite = "editor/logic_autosave.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def newlevelunit(self):
        return self._raw_data.get('newlevelunit', "0")

    @property
    def minimumhitpoints(self):
        return parse_source_value(self._raw_data.get('minimumhitpoints', 0))

    @property
    def minhitpointstocommit(self):
        return parse_source_value(self._raw_data.get('minhitpointstocommit', 0))



class logic_branch(BaseEntityPoint):
    icon_sprite = "editor/logic_branch.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def initialvalue(self):
        return self._raw_data.get('initialvalue', "0")



class logic_branch_listener(BaseEntityPoint):
    icon_sprite = "editor/logic_branch_listener.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def branch01(self):
        return self._raw_data.get('branch01', None)

    @property
    def branch02(self):
        return self._raw_data.get('branch02', None)

    @property
    def branch03(self):
        return self._raw_data.get('branch03', None)

    @property
    def branch04(self):
        return self._raw_data.get('branch04', None)

    @property
    def branch05(self):
        return self._raw_data.get('branch05', None)

    @property
    def branch06(self):
        return self._raw_data.get('branch06', None)

    @property
    def branch07(self):
        return self._raw_data.get('branch07', None)

    @property
    def branch08(self):
        return self._raw_data.get('branch08', None)

    @property
    def branch09(self):
        return self._raw_data.get('branch09', None)

    @property
    def branch10(self):
        return self._raw_data.get('branch10', None)

    @property
    def branch11(self):
        return self._raw_data.get('branch11', None)

    @property
    def branch12(self):
        return self._raw_data.get('branch12', None)

    @property
    def branch13(self):
        return self._raw_data.get('branch13', None)

    @property
    def branch14(self):
        return self._raw_data.get('branch14', None)

    @property
    def branch15(self):
        return self._raw_data.get('branch15', None)

    @property
    def branch16(self):
        return self._raw_data.get('branch16', None)



class logic_case(BaseEntityPoint):
    icon_sprite = "editor/logic_case.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def multiplecasesallowed(self):
        return self._raw_data.get('multiplecasesallowed', "0")

    @property
    def case01(self):
        return self._raw_data.get('case01', None)

    @property
    def case02(self):
        return self._raw_data.get('case02', None)

    @property
    def case03(self):
        return self._raw_data.get('case03', None)

    @property
    def case04(self):
        return self._raw_data.get('case04', None)

    @property
    def case05(self):
        return self._raw_data.get('case05', None)

    @property
    def case06(self):
        return self._raw_data.get('case06', None)

    @property
    def case07(self):
        return self._raw_data.get('case07', None)

    @property
    def case08(self):
        return self._raw_data.get('case08', None)

    @property
    def case09(self):
        return self._raw_data.get('case09', None)

    @property
    def case10(self):
        return self._raw_data.get('case10', None)

    @property
    def case11(self):
        return self._raw_data.get('case11', None)

    @property
    def case12(self):
        return self._raw_data.get('case12', None)

    @property
    def case13(self):
        return self._raw_data.get('case13', None)

    @property
    def case14(self):
        return self._raw_data.get('case14', None)

    @property
    def case15(self):
        return self._raw_data.get('case15', None)

    @property
    def case16(self):
        return self._raw_data.get('case16', None)



class logic_choreographed_scene(BaseEntityPoint):
    icon_sprite = "editor/choreo_scene.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def scenefile(self):
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



class logic_collision_pair(BaseEntityPoint):
    icon_sprite = "editor/ficool2/logic_collision_pair.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def attach1(self):
        return self._raw_data.get('attach1', None)

    @property
    def attach2(self):
        return self._raw_data.get('attach2', None)

    @property
    def startdisabled(self):
        return self._raw_data.get('startdisabled', "1")



class logic_compare(BaseEntityPoint):
    icon_sprite = "editor/logic_compare.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def initialvalue(self):
        return self._raw_data.get('initialvalue', None)

    @property
    def comparevalue(self):
        return self._raw_data.get('comparevalue', None)

    @property
    def strlenallowed(self):
        return self._raw_data.get('strlenallowed', "0")



class logic_console(BaseEntityPoint):
    icon_sprite = "editor/l2/logic_console.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def setdevlvl(self):
        return parse_source_value(self._raw_data.get('setdevlvl', 1))

    @property
    def setmsgcolor(self):
        return parse_int_vector(self._raw_data.get('setmsgcolor', "210 250 255 255"))

    @property
    def setwarningcolor(self):
        return parse_int_vector(self._raw_data.get('setwarningcolor', "255 210 210 255"))

    @property
    def setnewlinenotauto(self):
        return self._raw_data.get('setnewlinenotauto', "0")



class logic_context_accessor(BaseEntityPoint):
    icon_sprite = "editor/logic_context_accessor.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def context(self):
        return self._raw_data.get('context', None)



class logic_convar(BaseEntityPoint):
    icon_sprite = "editor/logic_convar.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def setconvar(self):
        return self._raw_data.get('setconvar', None)

    @property
    def settestvalue(self):
        return self._raw_data.get('settestvalue', None)



class logic_coop_manager(BaseEntityPoint):
    icon_sprite = "editor/logic_coop_manager.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def defaultplayerstatea(self):
        return self._raw_data.get('defaultplayerstatea', "0")

    @property
    def defaultplayerstateb(self):
        return self._raw_data.get('defaultplayerstateb', "0")



class logic_datadesc_accessor(BaseEntityPoint):
    icon_sprite = "editor/logic_datadesc_accessor.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def keyname(self):
        return self._raw_data.get('keyname', None)



class logic_entity_position(BaseEntityPoint):
    icon_sprite = "editor/logic_entity_position.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def positiontype(self):
        return self._raw_data.get('positiontype', "0")

    @property
    def positionparameter(self):
        return self._raw_data.get('positionparameter', None)



class logic_eventlistener(BaseEntityPoint):
    icon_sprite = "editor/logic_eventlistener.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def eventname(self):
        return self._raw_data.get('eventname', None)

    @property
    def isenabled(self):
        return self._raw_data.get('isenabled', "1")

    @property
    def teamnum(self):
        return parse_source_value(self._raw_data.get('teamnum', -1))

    @property
    def fetcheventdata(self):
        return self._raw_data.get('fetcheventdata', "0")



class logic_eventlistener_itemequip(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def weapontype(self):
        return self._raw_data.get('weapontype', "-1")

    @property
    def weaponclassname(self):
        return self._raw_data.get('weaponclassname', None)

    @property
    def isenabled(self):
        return self._raw_data.get('isenabled', "1")

    @property
    def teamnum(self):
        return self._raw_data.get('teamnum', "-1")



class logic_format(BaseEntityPoint):
    icon_sprite = "editor/logic_format.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def setinputvalue(self):
        return self._raw_data.get('setinputvalue', None)

    @property
    def setparameter0(self):
        return self._raw_data.get('setparameter0', None)

    @property
    def setparameter1(self):
        return self._raw_data.get('setparameter1', None)

    @property
    def setparameter2(self):
        return self._raw_data.get('setparameter2', None)

    @property
    def setparameter3(self):
        return self._raw_data.get('setparameter3', None)

    @property
    def setparameter4(self):
        return self._raw_data.get('setparameter4', None)

    @property
    def setparameter5(self):
        return self._raw_data.get('setparameter5', None)

    @property
    def setparameter6(self):
        return self._raw_data.get('setparameter6', None)

    @property
    def setparameter7(self):
        return self._raw_data.get('setparameter7', None)

    @property
    def setbackupparameter(self):
        return self._raw_data.get('setbackupparameter', None)



class logic_gate(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/logic_gate.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def mode(self):
        return self._raw_data.get('mode', "1")



class logic_keyfield(BaseEntityPoint):
    icon_sprite = "editor/logic_keyfield.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def keyname(self):
        return self._raw_data.get('keyname', None)



class logic_lineto(BaseEntityPoint):
    icon_sprite = "editor/ficool2/logic_lineto"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def source(self):
        return self._raw_data.get('source', None)

    @property
    def target(self):
        return self._raw_data.get('target', None)



class logic_measure_movement(BaseEntityPoint):
    icon_sprite = "editor/logic_measure_movement.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def measuretarget(self):
        return self._raw_data.get('measuretarget', None)

    @property
    def measurereference(self):
        return self._raw_data.get('measurereference', None)

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def targetreference(self):
        return self._raw_data.get('targetreference', None)

    @property
    def targetscale(self):
        return parse_source_value(self._raw_data.get('targetscale', 1))

    @property
    def shouldoutputposition(self):
        return self._raw_data.get('shouldoutputposition', "0")

    @property
    def measuretype(self):
        return self._raw_data.get('measuretype', "0")

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def measureattachment(self):
        return self._raw_data.get('measureattachment', None)



class logic_modelinfo(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def poseparametername(self):
        return self._raw_data.get('poseparametername', None)



class logic_multicompare(BaseEntityPoint):
    icon_sprite = "editor/logic_multicompare.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def integervalue(self):
        return parse_source_value(self._raw_data.get('integervalue', None))

    @property
    def shouldcomparetovalue(self):
        return self._raw_data.get('shouldcomparetovalue', "0")

    @property
    def strlenallowed(self):
        return self._raw_data.get('strlenallowed', "0")



class logic_navigation(BaseEntityPoint):
    icon_sprite = "editor/ficool2/logic_navigation"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def navprop(self):
        return self._raw_data.get('navprop', "Ignore")



class logic_player_slowtime(BaseEntityPoint):
    icon_sprite = "editor/logic_player_slowtime.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class logic_playerproxy(DamageFilter, BaseEntityPoint):
    icon_sprite = "editor/logic_playerproxy.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class logic_playmovie(BaseEntityPoint):
    icon_sprite = "editor/logic_playmovie"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def moviefilename(self):
        return self._raw_data.get('moviefilename', "aperture_logo.bik")

    @property
    def allowskip(self):
        return self._raw_data.get('allowskip', "0")

    @property
    def loopvideo(self):
        return self._raw_data.get('loopvideo', "0")

    @property
    def fadeintime(self):
        return parse_source_value(self._raw_data.get('fadeintime', 0))



class logic_random_outputs(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/logic_random_outputs.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def ontriggerchance1(self):
        return parse_source_value(self._raw_data.get('ontriggerchance1', 1.0))

    @property
    def ontriggerchance2(self):
        return parse_source_value(self._raw_data.get('ontriggerchance2', 1.0))

    @property
    def ontriggerchance3(self):
        return parse_source_value(self._raw_data.get('ontriggerchance3', 1.0))

    @property
    def ontriggerchance4(self):
        return parse_source_value(self._raw_data.get('ontriggerchance4', 1.0))

    @property
    def ontriggerchance5(self):
        return parse_source_value(self._raw_data.get('ontriggerchance5', 1.0))

    @property
    def ontriggerchance6(self):
        return parse_source_value(self._raw_data.get('ontriggerchance6', 1.0))

    @property
    def ontriggerchance7(self):
        return parse_source_value(self._raw_data.get('ontriggerchance7', 1.0))

    @property
    def ontriggerchance8(self):
        return parse_source_value(self._raw_data.get('ontriggerchance8', 1.0))



class logic_register_activator(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/logic_register_activator"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class logic_relay(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/logic_relay.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class logic_relay_queue(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/logic_relay_queue.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def setmaxqueueitems(self):
        return parse_source_value(self._raw_data.get('setmaxqueueitems', 3))

    @property
    def dontqueuewhendisabled(self):
        return self._raw_data.get('dontqueuewhendisabled', "0")

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class logic_scene_list_manager(BaseEntityPoint):
    icon_sprite = "editor/ficool2/logic_scene_list_manager.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def scene0(self):
        return self._raw_data.get('scene0', None)

    @property
    def scene1(self):
        return self._raw_data.get('scene1', None)

    @property
    def scene2(self):
        return self._raw_data.get('scene2', None)

    @property
    def scene3(self):
        return self._raw_data.get('scene3', None)

    @property
    def scene4(self):
        return self._raw_data.get('scene4', None)

    @property
    def scene5(self):
        return self._raw_data.get('scene5', None)

    @property
    def scene6(self):
        return self._raw_data.get('scene6', None)

    @property
    def scene7(self):
        return self._raw_data.get('scene7', None)

    @property
    def scene8(self):
        return self._raw_data.get('scene8', None)

    @property
    def scene9(self):
        return self._raw_data.get('scene9', None)

    @property
    def scene10(self):
        return self._raw_data.get('scene10', None)

    @property
    def scene11(self):
        return self._raw_data.get('scene11', None)

    @property
    def scene12(self):
        return self._raw_data.get('scene12', None)

    @property
    def scene13(self):
        return self._raw_data.get('scene13', None)

    @property
    def scene14(self):
        return self._raw_data.get('scene14', None)

    @property
    def scene15(self):
        return self._raw_data.get('scene15', None)



class logic_script(BaseEntityPoint):
    icon_sprite = "editor/logic_script.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def group00(self):
        return self._raw_data.get('group00', None)

    @property
    def group01(self):
        return self._raw_data.get('group01', None)

    @property
    def group02(self):
        return self._raw_data.get('group02', None)

    @property
    def group03(self):
        return self._raw_data.get('group03', None)

    @property
    def group04(self):
        return self._raw_data.get('group04', None)

    @property
    def group05(self):
        return self._raw_data.get('group05', None)

    @property
    def group06(self):
        return self._raw_data.get('group06', None)

    @property
    def group07(self):
        return self._raw_data.get('group07', None)

    @property
    def group08(self):
        return self._raw_data.get('group08', None)

    @property
    def group09(self):
        return self._raw_data.get('group09', None)

    @property
    def group10(self):
        return self._raw_data.get('group10', None)

    @property
    def group11(self):
        return self._raw_data.get('group11', None)

    @property
    def group12(self):
        return self._raw_data.get('group12', None)

    @property
    def group13(self):
        return self._raw_data.get('group13', None)

    @property
    def group14(self):
        return self._raw_data.get('group14', None)

    @property
    def group15(self):
        return self._raw_data.get('group15', None)



class logic_sequence(BaseEntityPoint):
    icon_sprite = "editor/logic_sequence.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def initialcase(self):
        return parse_source_value(self._raw_data.get('initialcase', 1))

    @property
    def startdisabled(self):
        return self._raw_data.get('startdisabled', "0")

    @property
    def case01(self):
        return self._raw_data.get('case01', None)

    @property
    def case02(self):
        return self._raw_data.get('case02', None)

    @property
    def case03(self):
        return self._raw_data.get('case03', None)

    @property
    def case04(self):
        return self._raw_data.get('case04', None)

    @property
    def case05(self):
        return self._raw_data.get('case05', None)

    @property
    def case06(self):
        return self._raw_data.get('case06', None)

    @property
    def case07(self):
        return self._raw_data.get('case07', None)

    @property
    def case08(self):
        return self._raw_data.get('case08', None)

    @property
    def case09(self):
        return self._raw_data.get('case09', None)

    @property
    def case10(self):
        return self._raw_data.get('case10', None)

    @property
    def case11(self):
        return self._raw_data.get('case11', None)

    @property
    def case12(self):
        return self._raw_data.get('case12', None)

    @property
    def case13(self):
        return self._raw_data.get('case13', None)

    @property
    def case14(self):
        return self._raw_data.get('case14', None)

    @property
    def case15(self):
        return self._raw_data.get('case15', None)

    @property
    def case16(self):
        return self._raw_data.get('case16', None)

    @property
    def dontincrementonpass(self):
        return self._raw_data.get('dontincrementonpass', "0")



class logic_timer(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/logic_timer.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def userandomtime(self):
        return self._raw_data.get('userandomtime', "0")

    @property
    def lowerrandombound(self):
        return self._raw_data.get('lowerrandombound', None)

    @property
    def upperrandombound(self):
        return self._raw_data.get('upperrandombound', None)

    @property
    def refiretime(self):
        return self._raw_data.get('refiretime', None)



class logic_timescale(BaseEntityPoint):
    icon_sprite = "editor/logic_timescale.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def blendtime(self):
        return parse_source_value(self._raw_data.get('blendtime', 0))



class material_modify_control(BaseEntityPoint):
    icon_sprite = "editor/ficool2/material_modify_control.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def materialname(self):
        return self._raw_data.get('materialname', None)

    @property
    def materialvar(self):
        return self._raw_data.get('materialvar', None)

    @property
    def srctools_search_parent(self):
        return self._raw_data.get('srctools_search_parent', "0")



class math_bits(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/math_bits.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def startvalue(self):
        return parse_source_value(self._raw_data.get('startvalue', 0))



class math_clamp(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/math_clamp.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def setmax(self):
        return self._raw_data.get('setmax', None)

    @property
    def setmin(self):
        return self._raw_data.get('setmin', None)



class math_colorblend(BaseEntityPoint):
    icon_sprite = "editor/math_colorblend.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

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



class math_counter(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/math_counter.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def startvalue(self):
        return parse_source_value(self._raw_data.get('startvalue', 0))

    @property
    def min(self):
        return parse_source_value(self._raw_data.get('min', 0))

    @property
    def max(self):
        return parse_source_value(self._raw_data.get('max', 0))



class math_generate(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/math_generate.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def startdisabled(self):
        return self._raw_data.get('startdisabled', "1")

    @property
    def generatetype(self):
        return self._raw_data.get('generatetype', "1")

    @property
    def initialvalue(self):
        return parse_source_value(self._raw_data.get('initialvalue', 0))

    @property
    def sethitmin(self):
        return parse_source_value(self._raw_data.get('sethitmin', 0))

    @property
    def sethitmax(self):
        return parse_source_value(self._raw_data.get('sethitmax', 1))

    @property
    def setparam1(self):
        return parse_source_value(self._raw_data.get('setparam1', 0))

    @property
    def setparam2(self):
        return parse_source_value(self._raw_data.get('setparam2', 0))



class math_lightpattern(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/math_lightpattern.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def style(self):
        return self._raw_data.get('style', None)

    @property
    def pattern(self):
        return self._raw_data.get('pattern', None)

    @property
    def patternspeed(self):
        return parse_source_value(self._raw_data.get('patternspeed', 0.1))



class math_mod(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/math_mod.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def startvalue(self):
        return self._raw_data.get('startvalue', "0")

    @property
    def setoperator(self):
        return self._raw_data.get('setoperator', "43")



class math_remap(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/math_remap.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

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



class math_vector(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/math_vector.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def startvalue(self):
        return parse_float_vector(self._raw_data.get('startvalue', "0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class move_keyframed(Mover, KeyFrame, BaseEntityPoint):
    pass


class move_rope(RopeKeyFrame, BaseEntityPoint):
    model = "models/editor/axis_helper_thick.mdl"
    icon_sprite = "editor/move_rope"
    pass


class move_track(KeyFrame, Mover, BaseEntityPoint):

    @property
    def wheelbaselength(self):
        return parse_source_value(self._raw_data.get('wheelbaselength', 50))

    @property
    def damage(self):
        return parse_source_value(self._raw_data.get('damage', 0))

    @property
    def norotate(self):
        return self._raw_data.get('norotate', "0")



class npc_heli_avoidbox(BaseEntityBrush):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class npc_heli_avoidsphere(BaseEntityPoint):
    icon_sprite = "editor/env_firesource"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 128))



class npc_heli_nobomb(BaseEntityBrush):
    pass


class obb_volumefog(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def width(self):
        return parse_source_value(self._raw_data.get('width', 256))

    @property
    def height(self):
        return parse_source_value(self._raw_data.get('height', 256))

    @property
    def depth(self):
        return parse_source_value(self._raw_data.get('depth', 256))

    @property
    def spheroid_volume(self):
        return self._raw_data.get('spheroid_volume', "0")

    @property
    def emissive_color(self):
        return parse_int_vector(self._raw_data.get('emissive_color', "0 0 0 255"))

    @property
    def density(self):
        return parse_source_value(self._raw_data.get('density', 0.1))

    @property
    def scattering_color(self):
        return parse_int_vector(self._raw_data.get('scattering_color', "255 255 255 255"))

    @property
    def phase(self):
        return parse_source_value(self._raw_data.get('phase', 0.0))

    @property
    def texture_name(self):
        return self._raw_data.get('texture_name', None)

    @property
    def texture_slices_x(self):
        return parse_source_value(self._raw_data.get('texture_slices_x', 16))

    @property
    def texture_slices_y(self):
        return parse_source_value(self._raw_data.get('texture_slices_y', 16))



class paint_sphere(BaseEntityPoint):
    icon_sprite = "editor/paint_sphere.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def paint_type(self):
        return self._raw_data.get('paint_type', "0")

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 60.0))

    @property
    def alpha_percent(self):
        return parse_source_value(self._raw_data.get('alpha_percent', 1.0))



class panorama_screen(BaseEntityPoint):
    model = "models/editor/angle_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def layout(self):
        return self._raw_data.get('layout', None)

    @property
    def width(self):
        return parse_source_value(self._raw_data.get('width', 128))

    @property
    def height(self):
        return parse_source_value(self._raw_data.get('height', 128))

    @property
    def panel_scale(self):
        return parse_source_value(self._raw_data.get('panel_scale', 32))

    @property
    def panel_class_name(self):
        return self._raw_data.get('panel_class_name', None)

    @property
    def panel_id(self):
        return self._raw_data.get('panel_id', None)

    @property
    def start_active(self):
        return self._raw_data.get('start_active', "1")

    @property
    def ignore_input(self):
        return self._raw_data.get('ignore_input', "0")

    @property
    def interact_distance(self):
        return parse_source_value(self._raw_data.get('interact_distance', 8))



class path_corner(BaseEntityPoint):
    icon_sprite = "editor/ficool2/path_corner"
    model = "models/editor/angle_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

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



class path_corner_crash(BaseEntityPoint):
    icon_sprite = "editor/ficool2/path_corner_crash"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)



class path_track(BaseEntityPoint):
    model = "models/editor/angle_helper.mdl"
    icon_sprite = "editor/ficool2/path_track"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

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



class path_vphysics(BaseEntityPoint):
    model = "models/editor/angle_helper.mdl"

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 100))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 32))

    @property
    def damping(self):
        return parse_source_value(self._raw_data.get('damping', 5))

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)



class phys_constraintsystem(BaseEntityPoint):
    icon_sprite = "editor/phys_constraintsystem.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def additionaliterations(self):
        return parse_source_value(self._raw_data.get('additionaliterations', 0))



class phys_convert(BaseEntityPoint):
    icon_sprite = "editor/phys_convert.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def swapmodel(self):
        return self._raw_data.get('swapmodel', None)

    @property
    def massoverride(self):
        return parse_source_value(self._raw_data.get('massoverride', 0))



class phys_keepupright(BaseEntityPoint):
    icon_sprite = "editor/ficool2/phys_keepupright.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def attach1(self):
        return self._raw_data.get('attach1', None)

    @property
    def angularlimit(self):
        return parse_source_value(self._raw_data.get('angularlimit', 15))



class phys_motor(BaseEntityPoint):
    icon_sprite = "editor/ficool2/phys_motor"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 0))

    @property
    def spinup(self):
        return parse_source_value(self._raw_data.get('spinup', 1))

    @property
    def inertiafactor(self):
        return parse_source_value(self._raw_data.get('inertiafactor', 1.0))

    @property
    def axis(self):
        return self._raw_data.get('axis', None)

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def attach1(self):
        return self._raw_data.get('attach1', None)



class phys_ragdollmagnet(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/ficool2/phys_ragdollmagnet"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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
        return self._raw_data.get('target', None)

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class phys_spring(BaseEntityPoint):
    icon_sprite = "editor/ficool2/phys_spring"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def attach1(self):
        return self._raw_data.get('attach1', None)

    @property
    def attach2(self):
        return self._raw_data.get('attach2', None)

    @property
    def springaxis(self):
        return self._raw_data.get('springaxis', None)

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



class player_loadsaved(BaseEntityPoint):
    icon_sprite = "editor/ficool2/player_loadsaved"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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



class player_speedmod(BaseEntityPoint):
    icon_sprite = "editor/player_speedmod.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def additionalbuttons(self):
        return parse_source_value(self._raw_data.get('additionalbuttons', 0))



class player_weaponstrip(BaseEntityPoint):
    icon_sprite = "editor/player_weaponstrip.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class point_anglesensor(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/ficool2/point_anglesensor"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class point_angularvelocitysensor(BaseEntityPoint):
    icon_sprite = "editor/ficool2/point_angularvelocitysensor"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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



class point_antlion_repellant(BaseEntityPoint):
    icon_sprite = "editor/point_antlion_repellant.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def repelradius(self):
        return parse_source_value(self._raw_data.get('repelradius', 512))



class point_apc_controller(BaseEntityPoint):
    icon_sprite = "editor/point_apc_controller.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

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
        return self._raw_data.get('rotatestartsound', None)

    @property
    def rotatesound(self):
        return self._raw_data.get('rotatesound', None)

    @property
    def rotatestopsound(self):
        return self._raw_data.get('rotatestopsound', None)

    @property
    def minrange(self):
        return self._raw_data.get('minrange', "0")

    @property
    def maxrange(self):
        return self._raw_data.get('maxrange', "0")

    @property
    def targetentityname(self):
        return self._raw_data.get('targetentityname', None)



class point_bonusmaps_accessor(BaseEntityPoint):
    icon_sprite = "editor/ficool2/point_bonusmaps_accessor"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def filename(self):
        return self._raw_data.get('filename', None)

    @property
    def mapname(self):
        return self._raw_data.get('mapname', None)



class point_broadcastclientcommand(BaseEntityPoint):
    icon_sprite = "editor/point_clientcommand.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class point_bugbait(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def enabled(self):
        return self._raw_data.get('enabled', "1")

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 512))



class point_camera(BaseEntityPoint):
    viewport_model = "models/editor/camera.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def fov(self):
        return parse_source_value(self._raw_data.get('fov', 90))

    @property
    def usescreenaspectratio(self):
        return self._raw_data.get('usescreenaspectratio', "0")

    @property
    def fogenable(self):
        return self._raw_data.get('fogenable', "0")

    @property
    def fogcolor(self):
        return parse_int_vector(self._raw_data.get('fogcolor', "0 0 0"))

    @property
    def fogstart(self):
        return parse_source_value(self._raw_data.get('fogstart', 2048))

    @property
    def fogend(self):
        return parse_source_value(self._raw_data.get('fogend', 4096))

    @property
    def fogmaxdensity(self):
        return parse_source_value(self._raw_data.get('fogmaxdensity', 1))



class point_changelevel(BaseEntityPoint):
    icon_sprite = "editor/point_changelevel.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class point_clientcommand(BaseEntityPoint):
    icon_sprite = "editor/point_clientcommand.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class point_devshot_camera(BaseEntityPoint):
    viewport_model = "models/editor/camera.mdl"
    icon_sprite = "editor/point_devshot_camera.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def cameraname(self):
        return self._raw_data.get('cameraname', None)

    @property
    def fov(self):
        return parse_source_value(self._raw_data.get('fov', 90))

    @property
    def _frustum_far(self):
        return parse_source_value(self._raw_data.get('_frustum_far', 1024))



class point_enable_motion_fixup(BaseEntityPoint):
    icon_sprite = "editor/ficool2/point_enable_motion_fixup"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class point_entity_finder(BaseEntityPoint):
    icon_sprite = "editor/point_entity_finder"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)

    @property
    def referencename(self):
        return self._raw_data.get('referencename', None)

    @property
    def method(self):
        return self._raw_data.get('method', "0")



class point_flesh_effect_target(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 8))



class point_futbol_shooter(BaseEntityPoint):
    model = "models/editor/angle_helper.mdl"
    icon_sprite = "editor/point_futbol_shooter"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def launchspeed(self):
        return parse_source_value(self._raw_data.get('launchspeed', 100))



class point_gamestats_counter(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/ficool2/point_gamestats_counter"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def name(self):
        return self._raw_data.get('name', None)



class point_hiding_spot(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class point_hurt(DamageType, BaseEntityPoint):
    icon_sprite = "editor/ficool2/point_hurt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def damagetarget(self):
        return self._raw_data.get('damagetarget', None)

    @property
    def damageradius(self):
        return parse_source_value(self._raw_data.get('damageradius', 256))

    @property
    def damage(self):
        return parse_source_value(self._raw_data.get('damage', 5))

    @property
    def damagedelay(self):
        return parse_source_value(self._raw_data.get('damagedelay', 1))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class point_laser_target(SRCIndicator, BaseEntityPoint):
    model = "models/editor/axis_helper.mdl"
    icon_sprite = "editor/point_laser_target.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def terminalpoint(self):
        return self._raw_data.get('terminalpoint', "1")

    @property
    def filtercolor(self):
        return parse_int_vector(self._raw_data.get('filtercolor', "255 255 255 255"))



class point_message(BaseEntityPoint):
    model = "models/editor/axis_helper.mdl"
    icon_sprite = "editor/ficool2/point_message.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def message(self):
        return self._raw_data.get('message', None)

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 128))

    @property
    def developeronly(self):
        return self._raw_data.get('developeronly', "0")



class point_paint_sensor(BaseEntityPoint):
    icon_sprite = "editor/point_gel_sensor.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 64))

    @property
    def sides(self):
        return self._raw_data.get('sides', None)

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)



class point_playermoveconstraint(BaseEntityPoint):
    icon_sprite = "editor/ficool2/point_playermoveconstraint.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 256))

    @property
    def width(self):
        return parse_source_value(self._raw_data.get('width', 75.0))

    @property
    def speedfactor(self):
        return parse_source_value(self._raw_data.get('speedfactor', 0.15))



class point_posecontroller(BaseEntityPoint):
    icon_sprite = "editor/ficool2/point_posecontroller"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def propname(self):
        return self._raw_data.get('propname', None)

    @property
    def poseparametername(self):
        return self._raw_data.get('poseparametername', None)

    @property
    def posevalue(self):
        return parse_source_value(self._raw_data.get('posevalue', 0))

    @property
    def interpolationtime(self):
        return parse_source_value(self._raw_data.get('interpolationtime', 0))

    @property
    def interpolationwrap(self):
        return self._raw_data.get('interpolationwrap', "0")

    @property
    def cyclefrequency(self):
        return parse_source_value(self._raw_data.get('cyclefrequency', 0))

    @property
    def fmodulationtype(self):
        return self._raw_data.get('fmodulationtype', "0")

    @property
    def fmodtimeoffset(self):
        return parse_source_value(self._raw_data.get('fmodtimeoffset', 0))

    @property
    def fmodrate(self):
        return parse_source_value(self._raw_data.get('fmodrate', 0))

    @property
    def fmodamplitude(self):
        return parse_source_value(self._raw_data.get('fmodamplitude', 0))



class point_proximity_sensor(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/point_proximity_sensor.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class point_push(BaseEntityPoint):
    model = "models/editor/cone_helper.mdl"
    icon_sprite = "editor/ficool2/point_push"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

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
        return parse_source_value(self._raw_data.get('inner_radius', 128))

    @property
    def influence_cone(self):
        return parse_source_value(self._raw_data.get('influence_cone', 0))



class point_servercommand(BaseEntityPoint):
    icon_sprite = "editor/point_servercommand.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class point_spotlight(SystemLevelChoice, BaseEntityPoint):
    model = "models/editor/cone_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def spotlightlength(self):
        return parse_source_value(self._raw_data.get('spotlightlength', 500))

    @property
    def spotlightwidth(self):
        return parse_source_value(self._raw_data.get('spotlightwidth', 50))

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "255 255 255"))

    @property
    def hdrcolorscale(self):
        return parse_source_value(self._raw_data.get('hdrcolorscale', 0.7))

    @property
    def ignoresolid(self):
        return self._raw_data.get('ignoresolid', "0")

    @property
    def brightness(self):
        return parse_source_value(self._raw_data.get('brightness', 64))



class point_survey(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def surveyname(self):
        return self._raw_data.get('surveyname', "end_puzzle_survey")



class point_teleport(BaseEntityPoint):
    icon_sprite = "editor/point_teleport.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def model(self):
        return self._raw_data.get('model', "models/editor/angle_helper.mdl")

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class point_template(BaseEntityPoint):
    icon_sprite = "editor/point_template.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def template01(self):
        return self._raw_data.get('template01', None)

    @property
    def template02(self):
        return self._raw_data.get('template02', None)

    @property
    def template03(self):
        return self._raw_data.get('template03', None)

    @property
    def template04(self):
        return self._raw_data.get('template04', None)

    @property
    def template05(self):
        return self._raw_data.get('template05', None)

    @property
    def template06(self):
        return self._raw_data.get('template06', None)

    @property
    def template07(self):
        return self._raw_data.get('template07', None)

    @property
    def template08(self):
        return self._raw_data.get('template08', None)

    @property
    def template09(self):
        return self._raw_data.get('template09', None)

    @property
    def template10(self):
        return self._raw_data.get('template10', None)

    @property
    def template11(self):
        return self._raw_data.get('template11', None)

    @property
    def template12(self):
        return self._raw_data.get('template12', None)

    @property
    def template13(self):
        return self._raw_data.get('template13', None)

    @property
    def template14(self):
        return self._raw_data.get('template14', None)

    @property
    def template15(self):
        return self._raw_data.get('template15', None)

    @property
    def template16(self):
        return self._raw_data.get('template16', None)



class point_tesla(BaseEntityPoint):
    icon_sprite = "editor/ficool2/point_tesla"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def m_bon(self):
        return self._raw_data.get('m_bon', "0")

    @property
    def m_sourceentityname(self):
        return self._raw_data.get('m_sourceentityname', None)

    @property
    def m_soundname(self):
        return self._raw_data.get('m_soundname', "DoSpark")

    @property
    def texture(self):
        return self._raw_data.get('texture', "sprites/physbeam.vmt")

    @property
    def m_color(self):
        return parse_int_vector(self._raw_data.get('m_color', "255 255 255"))

    @property
    def m_flradius(self):
        return parse_source_value(self._raw_data.get('m_flradius', 200))

    @property
    def beamcount_min(self):
        return parse_source_value(self._raw_data.get('beamcount_min', 6))

    @property
    def beamcount_max(self):
        return parse_source_value(self._raw_data.get('beamcount_max', 8))

    @property
    def thick_min(self):
        return parse_source_value(self._raw_data.get('thick_min', 4))

    @property
    def thick_max(self):
        return parse_source_value(self._raw_data.get('thick_max', 5))

    @property
    def lifetime_min(self):
        return parse_source_value(self._raw_data.get('lifetime_min', 0.3))

    @property
    def lifetime_max(self):
        return parse_source_value(self._raw_data.get('lifetime_max', 0.3))

    @property
    def interval_min(self):
        return parse_source_value(self._raw_data.get('interval_min', 0.5))

    @property
    def interval_max(self):
        return parse_source_value(self._raw_data.get('interval_max', 2))



class point_velocitysensor(BaseEntityPoint):
    icon_sprite = "editor/ficool2/point_velocitysensor"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def axis(self):
        return self._raw_data.get('axis', None)

    @property
    def enabled(self):
        return self._raw_data.get('enabled', "1")



class point_viewcontrol(BaseEntityPoint):
    viewport_model = "models/editor/camera.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 0))

    @property
    def acceleration(self):
        return parse_source_value(self._raw_data.get('acceleration', 500))

    @property
    def deceleration(self):
        return parse_source_value(self._raw_data.get('deceleration', 500))

    @property
    def trackspeed(self):
        return parse_source_value(self._raw_data.get('trackspeed', 40))

    @property
    def fov(self):
        return parse_source_value(self._raw_data.get('fov', 90))

    @property
    def fov_rate(self):
        return parse_source_value(self._raw_data.get('fov_rate', 1.0))

    @property
    def dontsetplayerview(self):
        return self._raw_data.get('dontsetplayerview', "0")

    @property
    def _frustum_far(self):
        return parse_source_value(self._raw_data.get('_frustum_far', 1024))



class point_viewcontrol_multiplayer(BaseEntityPoint):
    viewport_model = "models/editor/camera.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def fov(self):
        return parse_source_value(self._raw_data.get('fov', 90))

    @property
    def fov_rate(self):
        return parse_source_value(self._raw_data.get('fov_rate', 1))

    @property
    def target_entity(self):
        return self._raw_data.get('target_entity', None)

    @property
    def interp_time(self):
        return parse_source_value(self._raw_data.get('interp_time', 1))

    @property
    def target_team(self):
        return self._raw_data.get('target_team', "-1")

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def _frustum_far(self):
        return parse_source_value(self._raw_data.get('_frustum_far', 1024))



class point_viewproxy(BaseEntityPoint):
    viewport_model = "models/editor/camera.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

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



class point_worldtext(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def message(self):
        return self._raw_data.get('message', "New Message")

    @property
    def textsize(self):
        return parse_source_value(self._raw_data.get('textsize', 10))

    @property
    def color(self):
        return parse_int_vector(self._raw_data.get('color', "255 255 255"))

    @property
    def font(self):
        return self._raw_data.get('font', "editor/worldtext")



class portalmp_gamerules(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class postprocess_controller(BaseEntityPoint):
    icon_sprite = "editor/postprocess_controller.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class projected_wall_entity(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class prop_glass_futbol_socket(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/props/futbol_socket.mdl")



class prop_glass_futbol_spawner(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def startwithfutbol(self):
        return self._raw_data.get('startwithfutbol', "1")

    @property
    def model(self):
        return self._raw_data.get('model', "models/props/futbol_dispenser.mdl")



class prop_indicator_panel(BaseEntityPoint):
    viewport_model = "models/editor/prop_indicator_panel.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def timerduration(self):
        return parse_source_value(self._raw_data.get('timerduration', 0))

    @property
    def enabled(self):
        return self._raw_data.get('enabled', "1")

    @property
    def istimer(self):
        return self._raw_data.get('istimer', "0")

    @property
    def ischecked(self):
        return self._raw_data.get('ischecked', "0")

    @property
    def indicatorlights(self):
        return self._raw_data.get('indicatorlights', None)

    @property
    def skin(self):
        return self._raw_data.get('skin', "0")



class prop_portal(PortalBase, BaseEntityPoint):
    viewport_model = "models/editor/prop_portal.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def linkagegroupid(self):
        return parse_source_value(self._raw_data.get('linkagegroupid', 0))

    @property
    def skin(self):
        return self._raw_data.get('skin', "1")



class prop_testchamber_sign(BaseEntityPoint):
    viewport_model = "models/editor/prop_testchamber_sign.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def enabled(self):
        return self._raw_data.get('enabled', "1")

    @property
    def use_mod_config(self):
        return self._raw_data.get('use_mod_config', "0")

    @property
    def mod_config_sign_id(self):
        return parse_source_value(self._raw_data.get('mod_config_sign_id', 0))

    @property
    def separator(self):
        return self._raw_data.get('separator', "0")

    @property
    def sign_levelname(self):
        return self._raw_data.get('sign_levelname', None)

    @property
    def sign_startup_sequence(self):
        return self._raw_data.get('sign_startup_sequence', "Normal Flicker")

    @property
    def sign_current(self):
        return parse_source_value(self._raw_data.get('sign_current', None))

    @property
    def sign_total(self):
        return parse_source_value(self._raw_data.get('sign_total', None))

    @property
    def sign_dirt(self):
        return self._raw_data.get('sign_dirt', "-1")

    @property
    def sign_icon0name(self):
        return self._raw_data.get('sign_icon0name', None)

    @property
    def sign_icon1name(self):
        return self._raw_data.get('sign_icon1name', None)

    @property
    def sign_icon2name(self):
        return self._raw_data.get('sign_icon2name', None)

    @property
    def sign_icon3name(self):
        return self._raw_data.get('sign_icon3name', None)

    @property
    def sign_icon4name(self):
        return self._raw_data.get('sign_icon4name', None)

    @property
    def sign_icon5name(self):
        return self._raw_data.get('sign_icon5name', None)

    @property
    def sign_icon6name(self):
        return self._raw_data.get('sign_icon6name', None)

    @property
    def sign_icon7name(self):
        return self._raw_data.get('sign_icon7name', None)

    @property
    def sign_icon8name(self):
        return self._raw_data.get('sign_icon8name', None)

    @property
    def sign_icon9name(self):
        return self._raw_data.get('sign_icon9name', None)



class prop_tic_tac_toe_panel(BaseEntityPoint):
    viewport_model = "models/editor/prop_indicator_panel.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class rocket_turret_projectile(ResponseContext, ToggleDraw, BaseEntityPoint):
    viewport_model = "models/props_bts/rocket.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class script_intro(BaseEntityPoint):
    icon_sprite = "editor/ts2do/script_intro.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def alternatefovchange(self):
        return self._raw_data.get('alternatefovchange', "0")



class scripted_sentence(BaseEntityPoint):
    icon_sprite = "editor/scripted_sentence.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def sentence(self):
        return self._raw_data.get('sentence', None)

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



class scripted_sequence(SystemLevelChoice, BaseEntityPoint):
    model = "models/editor/scriptedsequence.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def m_iszentity(self):
        return self._raw_data.get('m_iszentity', None)

    @property
    def m_iszidle(self):
        return self._raw_data.get('m_iszidle', None)

    @property
    def m_iszentry(self):
        return self._raw_data.get('m_iszentry', None)

    @property
    def m_iszplay(self):
        return self._raw_data.get('m_iszplay', None)

    @property
    def m_iszpostidle(self):
        return self._raw_data.get('m_iszpostidle', None)

    @property
    def m_iszcustommove(self):
        return self._raw_data.get('m_iszcustommove', None)

    @property
    def m_bloopactionsequence(self):
        return self._raw_data.get('m_bloopactionsequence', "0")

    @property
    def m_bsynchpostidles(self):
        return self._raw_data.get('m_bsynchpostidles', "0")

    @property
    def m_flradius(self):
        return parse_source_value(self._raw_data.get('m_flradius', 0))

    @property
    def m_flrepeat(self):
        return parse_source_value(self._raw_data.get('m_flrepeat', 0))

    @property
    def m_fmoveto(self):
        return self._raw_data.get('m_fmoveto', "1")

    @property
    def onplayerdeath(self):
        return self._raw_data.get('onplayerdeath', "0")

    @property
    def m_isznextscript(self):
        return self._raw_data.get('m_isznextscript', None)

    @property
    def m_bignoregravity(self):
        return self._raw_data.get('m_bignoregravity', "0")

    @property
    def m_bdisablenpccollisions(self):
        return self._raw_data.get('m_bdisablenpccollisions', "0")

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class scripted_target(BaseEntityPoint):
    icon_sprite = "editor/ficool2/scripted_target.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def startdisabled(self):
        return self._raw_data.get('startdisabled', "1")

    @property
    def m_iszentity(self):
        return self._raw_data.get('m_iszentity', None)

    @property
    def m_flradius(self):
        return parse_source_value(self._raw_data.get('m_flradius', 0))

    @property
    def movespeed(self):
        return parse_source_value(self._raw_data.get('movespeed', 5))

    @property
    def pauseduration(self):
        return parse_source_value(self._raw_data.get('pauseduration', 0))

    @property
    def effectduration(self):
        return parse_source_value(self._raw_data.get('effectduration', 2))

    @property
    def target(self):
        return self._raw_data.get('target', None)



class shadow_control(BaseEntityPoint):
    icon_sprite = "editor/shadow_control.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def color(self):
        return parse_int_vector(self._raw_data.get('color', "128 128 128"))

    @property
    def distance(self):
        return parse_source_value(self._raw_data.get('distance', 75))

    @property
    def disableallshadows(self):
        return self._raw_data.get('disableallshadows', "0")

    @property
    def enableshadowsfromlocallights(self):
        return self._raw_data.get('enableshadowsfromlocallights', "0")



class sky_camera(BaseEntityPoint):
    viewport_model = "models/editor/sky_camera.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def skycolor(self):
        return parse_int_vector(self._raw_data.get('skycolor', "255 255 255 0"))

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
    def use_angles_for_sky(self):
        return self._raw_data.get('use_angles_for_sky', "0")

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
        return parse_source_value(self._raw_data.get('fogstart', 500))

    @property
    def fogend(self):
        return parse_source_value(self._raw_data.get('fogend', 2000))

    @property
    def fogmaxdensity(self):
        return parse_source_value(self._raw_data.get('fogmaxdensity', 1))

    @property
    def hdrcolorscale(self):
        return parse_source_value(self._raw_data.get('hdrcolorscale', 1))



class skybox_swapper(BaseEntityPoint):
    icon_sprite = "editor/skybox_swapper"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def skyboxname(self):
        return self._raw_data.get('skyboxname', "sky_dust")



class spark_shower(Angles, BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class sunlight_shadow_control(EnableDisable, BaseEntityPoint):
    icon_sprite = "editor/shadow_control.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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



class tanktrain_ai(BaseEntityPoint):
    icon_sprite = "editor/tanktrain_ai.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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



class tanktrain_aitarget(BaseEntityPoint):
    icon_sprite = "editor/tanktrain_aitarget.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def newtarget(self):
        return self._raw_data.get('newtarget', None)



class target_changegravity(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def gravity(self):
        return parse_source_value(self._raw_data.get('gravity', 1))



class test_sidelist(BaseEntityPoint):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def sides(self):
        return self._raw_data.get('sides', None)



class test_traceline(BaseEntityPoint):
    icon_sprite = "editor/ficool2/test_traceline"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class trigger_brush(EnableDisable, BaseEntityBrush):

    @property
    def inputfilter(self):
        return parse_source_value(self._raw_data.get('inputfilter', 0))

    @property
    def dontmessageparent(self):
        return self._raw_data.get('dontmessageparent', "0")

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class vgui_level_placard_display(BaseEntityPoint):
    viewport_model = "models/editor/vgui_arrows.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class vgui_movie_display(BaseEntityPoint):
    viewport_model = "models/editor/vgui_arrows.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def startenabled(self):
        return self._raw_data.get('startenabled', "0")

    @property
    def startpaused(self):
        return self._raw_data.get('startpaused', "0")

    @property
    def displaytext(self):
        return self._raw_data.get('displaytext', None)

    @property
    def moviefilename(self):
        return self._raw_data.get('moviefilename', "media/aperture_logo.webm")

    @property
    def groupname(self):
        return self._raw_data.get('groupname', None)

    @property
    def looping(self):
        return self._raw_data.get('looping', "0")

    @property
    def width(self):
        return parse_source_value(self._raw_data.get('width', 256))

    @property
    def height(self):
        return parse_source_value(self._raw_data.get('height', 128))

    @property
    def stretch(self):
        return self._raw_data.get('stretch', "0")

    @property
    def forcedslave(self):
        return self._raw_data.get('forcedslave', "0")

    @property
    def forceprecache(self):
        return self._raw_data.get('forceprecache', "0")

    @property
    def noscanline(self):
        return self._raw_data.get('noscanline', "0")

    @property
    def custom_uv(self):
        return self._raw_data.get('custom_uv', "0")

    @property
    def u_min(self):
        return parse_source_value(self._raw_data.get('u_min', 0))

    @property
    def u_max(self):
        return parse_source_value(self._raw_data.get('u_max', 0))

    @property
    def v_min(self):
        return parse_source_value(self._raw_data.get('v_min', 1))

    @property
    def v_max(self):
        return parse_source_value(self._raw_data.get('v_max', 1))

    @property
    def volume(self):
        return parse_source_value(self._raw_data.get('volume', 0))



class vgui_mp_lobby_display(BaseEntityPoint):
    viewport_model = "models/editor/vgui_arrows.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class vgui_neurotoxin_countdown(BaseEntityPoint):
    viewport_model = "models/editor/vgui_arrows.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def width(self):
        return parse_source_value(self._raw_data.get('width', 256))

    @property
    def height(self):
        return parse_source_value(self._raw_data.get('height', 128))

    @property
    def countdown(self):
        return parse_source_value(self._raw_data.get('countdown', 60))



class vgui_screen(BaseEntityPoint):
    model = "models/editor/angle_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def panelname(self):
        return self._raw_data.get('panelname', None)

    @property
    def overlaymaterial(self):
        return self._raw_data.get('overlaymaterial', None)

    @property
    def width(self):
        return parse_source_value(self._raw_data.get('width', 256))

    @property
    def height(self):
        return parse_source_value(self._raw_data.get('height', 128))

    @property
    def istransparent(self):
        return self._raw_data.get('istransparent', "0")



class vgui_slideshow_display(BaseEntityPoint):
    viewport_model = "models/editor/vgui_arrows.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def displaytext(self):
        return self._raw_data.get('displaytext', None)

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



class vgui_world_text_panel(BaseEntityPoint):
    viewport_model = "models/editor/vgui_arrows.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def enabled(self):
        return self._raw_data.get('enabled', "1")

    @property
    def displaytext(self):
        return self._raw_data.get('displaytext', None)

    @property
    def displaytextoption(self):
        return self._raw_data.get('displaytextoption', None)

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



class water_lod_control(BaseEntityPoint):
    icon_sprite = "editor/waterlodcontrol.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def cheapwaterstartdistance(self):
        return parse_source_value(self._raw_data.get('cheapwaterstartdistance', 1000))

    @property
    def cheapwaterenddistance(self):
        return parse_source_value(self._raw_data.get('cheapwaterenddistance', 2000))



class worldspawn(BaseEntity, ResponseContext):

    @property
    def message(self):
        return self._raw_data.get('message', None)

    @property
    def skyname(self):
        return self._raw_data.get('skyname', "sky_black_nofog")

    @property
    def chaptertitle(self):
        return self._raw_data.get('chaptertitle', None)

    @property
    def startdark(self):
        return self._raw_data.get('startdark', "0")

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
    def paintinmap(self):
        return self._raw_data.get('paintinmap', "0")

    @property
    def maxblobcount(self):
        return parse_source_value(self._raw_data.get('maxblobcount', 250))

    @property
    def maxprojectedtextures(self):
        return parse_source_value(self._raw_data.get('maxprojectedtextures', 8))



class BaseLogicalNPC(BaseEntityAnimating):

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', None))

    @property
    def max_health(self):
        return parse_source_value(self._raw_data.get('max_health', None))

    @property
    def squadname(self):
        return self._raw_data.get('squadname', None)

    @property
    def hintgroup(self):
        return self._raw_data.get('hintgroup', None)

    @property
    def hintlimiting(self):
        return self._raw_data.get('hintlimiting', "0")

    @property
    def additionalequipment(self):
        return self._raw_data.get('additionalequipment', "0")

    @property
    def relationship(self):
        return self._raw_data.get('relationship', None)

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

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
        return self._raw_data.get('enemyfilter', None)

    @property
    def ignoreunseenenemies(self):
        return self._raw_data.get('ignoreunseenenemies', "0")

    @property
    def physdamagescale(self):
        return parse_source_value(self._raw_data.get('physdamagescale', 1.0))

    @property
    def velocity(self):
        return parse_float_vector(self._raw_data.get('velocity', None))

    @property
    def basevelocity(self):
        return parse_float_vector(self._raw_data.get('basevelocity', None))

    @property
    def avelocity(self):
        return parse_float_vector(self._raw_data.get('avelocity', None))

    @property
    def waterlevel(self):
        return self._raw_data.get('waterlevel', "0")



class BasePedButton(SRCIndicator, BaseEntityAnimating):

    @property
    def delay(self):
        return parse_source_value(self._raw_data.get('delay', 1))

    @property
    def istimer(self):
        return self._raw_data.get('istimer', "0")

    @property
    def preventfastreset(self):
        return self._raw_data.get('preventfastreset', "0")



class BasePortButton(SRCIndicator, BaseEntityAnimating):

    @property
    def suppressanimsounds(self):
        return self._raw_data.get('suppressanimsounds', "0")

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)



class BaseProjector(BaseEntityAnimating):

    @property
    def startenabled(self):
        return self._raw_data.get('startenabled', "1")

    @property
    def disablehelper(self):
        return self._raw_data.get('disablehelper', "0")



class BasePropPhysics(SystemLevelChoice, BaseFadeProp, BreakableProp, BaseEntityPhysics):

    @property
    def damagetype(self):
        return self._raw_data.get('damagetype', "0")

    @property
    def nodamageforces(self):
        return self._raw_data.get('nodamageforces', "0")

    @property
    def inertiascale(self):
        return parse_source_value(self._raw_data.get('inertiascale', 1.0))

    @property
    def massscale(self):
        return parse_source_value(self._raw_data.get('massscale', 0))

    @property
    def overridescript(self):
        return self._raw_data.get('overridescript', None)

    @property
    def damagetoenablemotion(self):
        return parse_source_value(self._raw_data.get('damagetoenablemotion', 0))

    @property
    def forcetoenablemotion(self):
        return parse_source_value(self._raw_data.get('forcetoenablemotion', 0))

    @property
    def puntsound(self):
        return self._raw_data.get('puntsound', None)



class BaseTrain(BaseEntityVisBrush):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def target(self):
        return self._raw_data.get('target', None)

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
    def line_basetrain(self):
        return self._raw_data.get('line_basetrain', None)

    @property
    def movesound(self):
        return self._raw_data.get('movesound', None)

    @property
    def movepingsound(self):
        return self._raw_data.get('movepingsound', None)

    @property
    def startsound(self):
        return self._raw_data.get('startsound', None)

    @property
    def stopsound(self):
        return self._raw_data.get('stopsound', None)

    @property
    def volume(self):
        return parse_source_value(self._raw_data.get('volume', 10))

    @property
    def movesoundminpitch(self):
        return parse_source_value(self._raw_data.get('movesoundminpitch', 60))

    @property
    def movesoundmaxpitch(self):
        return parse_source_value(self._raw_data.get('movesoundmaxpitch', 200))

    @property
    def movesoundmintime(self):
        return parse_source_value(self._raw_data.get('movesoundmintime', 0))

    @property
    def movesoundmaxtime(self):
        return parse_source_value(self._raw_data.get('movesoundmaxtime', 0))



class BreakableBrush(BaseEntityVisBrush, DamageFilter, _Breakable):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def propdata(self):
        return self._raw_data.get('propdata', "0")

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
    def spawnobject(self):
        return self._raw_data.get('spawnobject', "0")



class Door(BaseEntityVisBrush, MasterEnt):

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 100))

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
    def loopmovesound(self):
        return self._raw_data.get('loopmovesound', "0")

    @property
    def wait(self):
        return parse_source_value(self._raw_data.get('wait', -1))

    @property
    def lip(self):
        return parse_source_value(self._raw_data.get('lip', 0))

    @property
    def dmg(self):
        return parse_source_value(self._raw_data.get('dmg', 0))

    @property
    def chainstodoor(self):
        return self._raw_data.get('chainstodoor', None)

    @property
    def forceclosed(self):
        return self._raw_data.get('forceclosed', "0")

    @property
    def ignoredebris(self):
        return self._raw_data.get('ignoredebris', "0")

    @property
    def solidbsp(self):
        return self._raw_data.get('solidbsp', "1")

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', None))

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
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def locked_sentence(self):
        return self._raw_data.get('locked_sentence', "0")

    @property
    def unlocked_sentence(self):
        return self._raw_data.get('unlocked_sentence', "0")



class Item(Toggle, FadeDistance, BaseEntityPhysics, TeamNum, EnableDisable):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class Trigger(TriggerOnce):
    pass


class ai_goal_actbusy(BaseActBusy):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def seeentity(self):
        return self._raw_data.get('seeentity', None)

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
        return self._raw_data.get('safezone', None)

    @property
    def allowteleport(self):
        return self._raw_data.get('allowteleport', "0")



class ai_goal_actbusy_queue(BaseActBusy):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def node_exit(self):
        return self._raw_data.get('node_exit', None)

    @property
    def node01(self):
        return self._raw_data.get('node01', None)

    @property
    def node02(self):
        return self._raw_data.get('node02', None)

    @property
    def node03(self):
        return self._raw_data.get('node03', None)

    @property
    def node04(self):
        return self._raw_data.get('node04', None)

    @property
    def node05(self):
        return self._raw_data.get('node05', None)

    @property
    def node06(self):
        return self._raw_data.get('node06', None)

    @property
    def node07(self):
        return self._raw_data.get('node07', None)

    @property
    def node08(self):
        return self._raw_data.get('node08', None)

    @property
    def node09(self):
        return self._raw_data.get('node09', None)

    @property
    def node10(self):
        return self._raw_data.get('node10', None)

    @property
    def node11(self):
        return self._raw_data.get('node11', None)

    @property
    def node12(self):
        return self._raw_data.get('node12', None)

    @property
    def node13(self):
        return self._raw_data.get('node13', None)

    @property
    def node14(self):
        return self._raw_data.get('node14', None)

    @property
    def node15(self):
        return self._raw_data.get('node15', None)

    @property
    def node16(self):
        return self._raw_data.get('node16', None)

    @property
    def node17(self):
        return self._raw_data.get('node17', None)

    @property
    def node18(self):
        return self._raw_data.get('node18', None)

    @property
    def node19(self):
        return self._raw_data.get('node19', None)

    @property
    def node20(self):
        return self._raw_data.get('node20', None)

    @property
    def mustreachfront(self):
        return self._raw_data.get('mustreachfront', "0")



class ai_goal_follow(FollowGoal):
    icon_sprite = "editor/ai_goal_follow.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class ai_goal_injured_follow(FollowGoal):
    icon_sprite = "editor/ficool2/ai_goal_injured_follow.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class ai_goal_lead(LeadGoalBase):
    icon_sprite = "editor/ai_goal_lead.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def searchtype(self):
        return self._raw_data.get('searchtype', "0")



class ai_goal_lead_weapon(LeadGoalBase):
    icon_sprite = "editor/ficool2/ai_goal_lead_weapon.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def weaponname(self):
        return self._raw_data.get('weaponname', "weapon_bugbait")

    @property
    def missingweaponconceptmodifier(self):
        return self._raw_data.get('missingweaponconceptmodifier', None)

    @property
    def searchtype(self):
        return self._raw_data.get('searchtype', "0")



class bounce_bomb(combine_mine):
    model = "models/props_combine/combine_mine01.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class combine_bouncemine(combine_mine):
    model = "models/props_combine/combine_mine01.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class comp_numeric_transition(logic_relay):
    icon_sprite = "editor/comp_numeric_transition"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def opt_name(self):
        return self._raw_data.get('opt_name', "SetSpeed")

    @property
    def io_type(self):
        return self._raw_data.get('io_type', "auto")

    @property
    def transform(self):
        return self._raw_data.get('transform', "speed")

    @property
    def line_trans2(self):
        return self._raw_data.get('line_trans2', None)

    @property
    def beat_interval(self):
        return parse_source_value(self._raw_data.get('beat_interval', 0.1))

    @property
    def delay(self):
        return parse_source_value(self._raw_data.get('delay', 0.0))

    @property
    def duration(self):
        return parse_source_value(self._raw_data.get('duration', 5))

    @property
    def startval(self):
        return parse_source_value(self._raw_data.get('startval', 0))

    @property
    def endval(self):
        return parse_source_value(self._raw_data.get('endval', 100))

    @property
    def line_trans3(self):
        return self._raw_data.get('line_trans3', None)

    @property
    def easing_start(self):
        return self._raw_data.get('easing_start', "linear")

    @property
    def easing_end(self):
        return self._raw_data.get('easing_end', "linear")



class comp_prop_cable_dynamic(BreakableProp, BaseEntityAnimating):
    model = "models/editor/comp_prop_cable_dynamic.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def group(self):
        return self._raw_data.get('group', None)

    @property
    def skin1(self):
        return self._raw_data.get('skin1', None)

    @property
    def skin2(self):
        return self._raw_data.get('skin2', None)

    @property
    def skin3(self):
        return self._raw_data.get('skin3', None)

    @property
    def skin4(self):
        return self._raw_data.get('skin4', None)

    @property
    def skin5(self):
        return self._raw_data.get('skin5', None)



class comp_prop_rope_dynamic(BreakableProp, BaseEntityAnimating):
    model = "models/editor/comp_prop_rope_dynamic.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def group(self):
        return self._raw_data.get('group', None)



class comp_sequential_call(logic_relay):
    icon_sprite = "editor/comp_sequential_call"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def order_mode(self):
        return self._raw_data.get('order_mode', "dist")

    @property
    def uniquify(self):
        return self._raw_data.get('uniquify', "0")

    @property
    def time_mode(self):
        return self._raw_data.get('time_mode', "total")

    @property
    def time_val(self):
        return parse_source_value(self._raw_data.get('time_val', 5.0))

    @property
    def time_variance(self):
        return parse_source_value(self._raw_data.get('time_variance', 0.0))



class cycler(SetModel, BaseEntityAnimating):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def sequence(self):
        return parse_source_value(self._raw_data.get('sequence', 0))



class ent_hover_turret_tether(BaseEntityAnimating):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class env_beam(BaseBeam):
    icon_sprite = "editor/env_beam.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 16))

    @property
    def life(self):
        return parse_source_value(self._raw_data.get('life', 0))

    @property
    def boltwidth(self):
        return parse_source_value(self._raw_data.get('boltwidth', 2))

    @property
    def striketime(self):
        return parse_source_value(self._raw_data.get('striketime', 1))

    @property
    def lightningstart(self):
        return self._raw_data.get('lightningstart', None)

    @property
    def lightningend(self):
        return self._raw_data.get('lightningend', None)

    @property
    def decalname(self):
        return self._raw_data.get('decalname', "Bigshot")

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def touchtype(self):
        return self._raw_data.get('touchtype', "0")

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)

    @property
    def targetpoint(self):
        return self._raw_data.get('targetpoint', "0 0 0")

    @property
    def clipstyle(self):
        return self._raw_data.get('clipstyle', "0")



class env_effectscript(Angles, BaseEntityAnimating):
    icon_sprite = "editor/ficool2/env_effectscript"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', None)

    @property
    def scriptfile(self):
        return self._raw_data.get('scriptfile', "scripts/effects/testeffect.txt")



class env_glow(env_sprite):
    model = "models/editor/axis_helper_white.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class env_headcrabcanister(BaseEntityAnimating):
    model = "models/props_combine/headcrabcannister01b.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def headcrabtype(self):
        return self._raw_data.get('headcrabtype', "0")

    @property
    def headcrabcount(self):
        return parse_source_value(self._raw_data.get('headcrabcount', 6))

    @property
    def flightspeed(self):
        return parse_source_value(self._raw_data.get('flightspeed', 3000))

    @property
    def flighttime(self):
        return parse_source_value(self._raw_data.get('flighttime', 5))

    @property
    def startingheight(self):
        return parse_source_value(self._raw_data.get('startingheight', 0))

    @property
    def minskyboxrefiretime(self):
        return parse_source_value(self._raw_data.get('minskyboxrefiretime', 0))

    @property
    def maxskyboxrefiretime(self):
        return parse_source_value(self._raw_data.get('maxskyboxrefiretime', 0))

    @property
    def skyboxcannistercount(self):
        return parse_source_value(self._raw_data.get('skyboxcannistercount', 1))

    @property
    def damage(self):
        return parse_source_value(self._raw_data.get('damage', 150))

    @property
    def damageradius(self):
        return parse_source_value(self._raw_data.get('damageradius', 750))

    @property
    def smokelifetime(self):
        return parse_source_value(self._raw_data.get('smokelifetime', 30))

    @property
    def launchpositionname(self):
        return self._raw_data.get('launchpositionname', None)



class env_laser(BaseBeam):
    icon_sprite = "editor/ficool2/env_laser.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def lasertarget(self):
        return self._raw_data.get('lasertarget', None)

    @property
    def width(self):
        return parse_source_value(self._raw_data.get('width', 2))

    @property
    def endsprite(self):
        return self._raw_data.get('endsprite', None)

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class env_portal_laser(BaseEntityAnimating):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def noplacementhelper(self):
        return self._raw_data.get('noplacementhelper', "0")

    @property
    def model(self):
        return self._raw_data.get('model', "models/props/laser_emitter_center.mdl")

    @property
    def startstate(self):
        return self._raw_data.get('startstate', "0")

    @property
    def lethaldamage(self):
        return self._raw_data.get('lethaldamage', "0")

    @property
    def autoaimenabled(self):
        return self._raw_data.get('autoaimenabled', "1")

    @property
    def skin(self):
        return self._raw_data.get('skin', "0")

    @property
    def beamcolor(self):
        return self._raw_data.get('beamcolor', "255 0 0 255")

    @property
    def disableplayercollision(self):
        return self._raw_data.get('disableplayercollision', "Default Behavior")



class env_portal_path_track(path_track):
    model = "models/editor/angle_helper.mdl"
    icon_sprite = "editor/ficool2/path_track"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def track_beam_scale(self):
        return parse_source_value(self._raw_data.get('track_beam_scale', 0))

    @property
    def end_point_scale(self):
        return parse_source_value(self._raw_data.get('end_point_scale', 0))

    @property
    def end_point_fadeout(self):
        return parse_source_value(self._raw_data.get('end_point_fadeout', 0))

    @property
    def end_point_fadein(self):
        return parse_source_value(self._raw_data.get('end_point_fadein', 0))



class env_rotorshooter(gibshooterbase):
    icon_sprite = "editor/ficool2/env_rotorshooter.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def shootmodel(self):
        return self._raw_data.get('shootmodel', None)

    @property
    def shootsounds(self):
        return self._raw_data.get('shootsounds', "-1")

    @property
    def skin(self):
        return parse_source_value(self._raw_data.get('skin', 0))

    @property
    def rotortime(self):
        return parse_source_value(self._raw_data.get('rotortime', 1))

    @property
    def rotortimevariance(self):
        return parse_source_value(self._raw_data.get('rotortimevariance', 0.3))



class env_shooter(gibshooterbase):
    icon_sprite = "editor/env_shooter.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def shootmodel(self):
        return self._raw_data.get('shootmodel', None)

    @property
    def shootsounds(self):
        return self._raw_data.get('shootsounds', "-1")

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



class env_soundscape_triggerable(env_soundscape):
    icon_sprite = "editor/env_soundscape_triggerable.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class env_sprite_clientside(env_sprite):
    model = "models/editor/axis_helper_white.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class env_sprite_oriented(env_sprite):
    model = "models/editor/axis_helper_white.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class filter_activator_class(filter_base):
    icon_sprite = "editor/filter_class.vmt"

    @property
    def filterclass(self):
        return self._raw_data.get('filterclass', None)



class filter_activator_context(filter_base):
    icon_sprite = "editor/filter_context.vmt"

    @property
    def responsecontext(self):
        return self._raw_data.get('responsecontext', None)

    @property
    def any(self):
        return self._raw_data.get('any', "0")



class filter_activator_involume(filter_base):
    icon_sprite = "editor/filter_involume.vmt"

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def tester(self):
        return self._raw_data.get('tester', "!activator")



class filter_activator_keyfield(filter_base):
    icon_sprite = "editor/filter_keyfield.vmt"

    @property
    def keyname(self):
        return self._raw_data.get('keyname', None)

    @property
    def value(self):
        return self._raw_data.get('value', None)



class filter_activator_mass_greater(filter_base):
    icon_sprite = "editor/filter_mass.vmt"

    @property
    def filtermass(self):
        return parse_source_value(self._raw_data.get('filtermass', None))



class filter_activator_model(filter_base):
    icon_sprite = "editor/filter_model.vmt"

    @property
    def model(self):
        return self._raw_data.get('model', None)

    @property
    def skin(self):
        return self._raw_data.get('skin', "-1")



class filter_activator_name(filter_base):
    icon_sprite = "editor/filter_name.vmt"

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)



class filter_activator_surfacedata(filter_base):
    icon_sprite = "editor/l2/filter_activator_surfacedata.vmt"

    @property
    def filtersurfaceprop(self):
        return self._raw_data.get('filtersurfaceprop', None)

    @property
    def surfacetype(self):
        return self._raw_data.get('surfacetype', "1")



class filter_activator_team(filter_base):
    icon_sprite = "editor/filter_team.vmt"

    @property
    def filterteam(self):
        return self._raw_data.get('filterteam', "0")



class filter_combineball_type(filter_base):
    icon_sprite = "editor/filter_pellet.vmt"

    @property
    def balltype(self):
        return self._raw_data.get('balltype', "1")



class filter_damage_type(filter_base):
    icon_sprite = "editor/filter_damage_type.vmt"

    @property
    def damagetype(self):
        return self._raw_data.get('damagetype', "64")



class filter_enemy(filter_base):
    icon_sprite = "editor/ficool2/filter_enemy.vmt"

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

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class filter_multi(filter_base):
    icon_sprite = "editor/filter_multiple.vmt"

    @property
    def filtertype(self):
        return self._raw_data.get('filtertype', "0")

    @property
    def filter01(self):
        return self._raw_data.get('filter01', None)

    @property
    def filter02(self):
        return self._raw_data.get('filter02', None)

    @property
    def filter03(self):
        return self._raw_data.get('filter03', None)

    @property
    def filter04(self):
        return self._raw_data.get('filter04', None)

    @property
    def filter05(self):
        return self._raw_data.get('filter05', None)

    @property
    def filter06(self):
        return self._raw_data.get('filter06', None)

    @property
    def filter07(self):
        return self._raw_data.get('filter07', None)

    @property
    def filter08(self):
        return self._raw_data.get('filter08', None)

    @property
    def filter09(self):
        return self._raw_data.get('filter09', None)

    @property
    def filter10(self):
        return self._raw_data.get('filter10', None)



class filter_paint_power(filter_base):
    icon_sprite = "editor/filter_paint_power.vmt"

    @property
    def paint_type(self):
        return self._raw_data.get('paint_type', "0")

    @property
    def paint_mode(self):
        return self._raw_data.get('paint_mode', "0")



class filter_player_held(filter_base):
    icon_sprite = "editor/filter_held.vmt"
    pass


class filter_velocity(filter_base):
    icon_sprite = "editor/ficool2/filter_base.vmt"

    @property
    def speedthreshold(self):
        return parse_source_value(self._raw_data.get('speedthreshold', 500))

    @property
    def dimension(self):
        return self._raw_data.get('dimension', "0")

    @property
    def matchmode(self):
        return self._raw_data.get('matchmode', "0")

    @property
    def yawangle(self):
        return parse_source_value(self._raw_data.get('yawangle', 0))



class func_brush(BaseEntityVisBrush, EnableDisable, Toggle):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def solidity(self):
        return self._raw_data.get('solidity', "0")

    @property
    def excludednpc(self):
        return self._raw_data.get('excludednpc', None)

    @property
    def invert_exclusion(self):
        return self._raw_data.get('invert_exclusion', "0")

    @property
    def solidbsp(self):
        return self._raw_data.get('solidbsp', "1")



class func_button(BaseEntityVisBrush, Button):

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
    def wait(self):
        return parse_source_value(self._raw_data.get('wait', 3))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def customsound(self):
        return self._raw_data.get('customsound', None)

    @property
    def min_use_angle(self):
        return parse_source_value(self._raw_data.get('min_use_angle', 0.8))



class func_combine_ball_spawner(CombineBallSpawners):
    pass


class func_conveyor(BaseEntityVisBrush):

    @property
    def movedir(self):
        return parse_float_vector(self._raw_data.get('movedir', "0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 100))



class func_guntarget(BaseEntityVisBrush):

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 100))

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 0))



class func_healthcharger(BaseEntityVisBrush, EnableDisable):
    pass


class func_illusionary(BaseEntityVisBrush):
    pass


class func_lod(BaseEntityVisBrush):

    @property
    def disappearmindist(self):
        return parse_source_value(self._raw_data.get('disappearmindist', 2000))

    @property
    def disappearmaxdist(self):
        return parse_source_value(self._raw_data.get('disappearmaxdist', 2200))



class func_movelinear(BaseEntityVisBrush, Angles):

    @property
    def movedir(self):
        return parse_float_vector(self._raw_data.get('movedir', "0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

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

    @property
    def solidbsp(self):
        return self._raw_data.get('solidbsp', "1")



class func_nav_avoid(NavCost):
    pass


class func_nav_prefer(NavCost):
    pass


class func_platrot(BaseEntityVisBrush):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

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



class func_portalled(func_portal_detector):

    @property
    def fireondeparture(self):
        return self._raw_data.get('fireondeparture', "1")

    @property
    def fireonarrival(self):
        return self._raw_data.get('fireonarrival', "1")

    @property
    def fireonplayer(self):
        return self._raw_data.get('fireonplayer', "1")



class func_recharge(BaseEntityVisBrush):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class func_rot_button(BaseEntityVisBrush, MasterEnt, Button):

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 50))

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 0))

    @property
    def wait(self):
        return parse_source_value(self._raw_data.get('wait', 3))

    @property
    def distance(self):
        return parse_source_value(self._raw_data.get('distance', 90))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class func_rotating(BaseEntityVisBrush, Origin, Angles):

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
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def dmg(self):
        return parse_source_value(self._raw_data.get('dmg', 0))

    @property
    def solidbsp(self):
        return self._raw_data.get('solidbsp', "1")



class func_tank(BaseTank):

    @property
    def ammotype(self):
        return self._raw_data.get('ammotype', None)



class func_tank_combine_cannon(BaseTank):

    @property
    def ammotype(self):
        return self._raw_data.get('ammotype', None)



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



class func_tanklaser(BaseTank):

    @property
    def laserentity(self):
        return self._raw_data.get('laserentity', None)



class func_tankmortar(BaseTank):

    @property
    def imagnitude(self):
        return parse_source_value(self._raw_data.get('imagnitude', 100))

    @property
    def firedelay(self):
        return self._raw_data.get('firedelay', "2")

    @property
    def firestartsound(self):
        return self._raw_data.get('firestartsound', None)

    @property
    def fireendsound(self):
        return self._raw_data.get('fireendsound', None)

    @property
    def incomingsound(self):
        return self._raw_data.get('incomingsound', None)

    @property
    def warningtime(self):
        return parse_source_value(self._raw_data.get('warningtime', 1))

    @property
    def firevariance(self):
        return parse_source_value(self._raw_data.get('firevariance', 0))



class func_tankphyscannister(BaseTank):

    @property
    def barrel_volume(self):
        return self._raw_data.get('barrel_volume', None)



class func_tankpulselaser(BaseTank):

    @property
    def pulsespeed(self):
        return parse_source_value(self._raw_data.get('pulsespeed', 1000))

    @property
    def pulsecolor(self):
        return parse_int_vector(self._raw_data.get('pulsecolor', "255 0 0"))

    @property
    def pulsewidth(self):
        return parse_source_value(self._raw_data.get('pulsewidth', 20))

    @property
    def pulselife(self):
        return parse_source_value(self._raw_data.get('pulselife', 2))

    @property
    def pulselag(self):
        return parse_source_value(self._raw_data.get('pulselag', 0.05))

    @property
    def pulsefiresound(self):
        return self._raw_data.get('pulsefiresound', None)



class func_tankrocket(BaseTank):

    @property
    def rocketspeed(self):
        return parse_source_value(self._raw_data.get('rocketspeed', 800))



class func_trackchange(BaseEntityVisBrush):

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

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class func_train(BaseEntityVisBrush):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 50))

    @property
    def noise1(self):
        return self._raw_data.get('noise1', "Default.Null")

    @property
    def noise2(self):
        return self._raw_data.get('noise2', "Default.Null")

    @property
    def volume(self):
        return parse_source_value(self._raw_data.get('volume', 10))

    @property
    def dmg(self):
        return parse_source_value(self._raw_data.get('dmg', 0))



class func_traincontrols(BaseEntityVisBrush):

    @property
    def target(self):
        return self._raw_data.get('target', None)



class func_wall_toggle(func_wall):

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class func_weight_button(BaseEntityVisBrush):

    @property
    def weighttoactivate(self):
        return parse_source_value(self._raw_data.get('weighttoactivate', None))



class gibshooter(gibshooterbase):
    icon_sprite = "editor/gibshooter.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class grenade_helicopter(BaseEntityPhysics):
    model = "models/combine_helicopter/helicopter_bomb01.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class hot_potato_catcher(prop_glass_futbol_socket):
    model = "models/editor/axis_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class hot_potato_socket(prop_glass_futbol_socket):
    model = "models/editor/axis_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class hot_potato_spawner(prop_glass_futbol_spawner):
    viewport_model = "models/props/futbol_dispenser.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_ammo_crate(BaseEntityAnimating):
    viewport_model = "models/items/ammocrate_pistol.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def ammotype(self):
        return self._raw_data.get('ammotype', "0")



class item_healthcharger(BaseEntityAnimating):
    model = "models/props_combine/health_charger001.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_nugget(BaseEntityAnimating):
    viewport_model = "models/effects/cappoint_hologram.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def groupname(self):
        return self._raw_data.get('groupname', None)

    @property
    def respawntime(self):
        return parse_source_value(self._raw_data.get('respawntime', 30))

    @property
    def pointvalue(self):
        return self._raw_data.get('pointvalue', "1")



class item_paint_power_pickup(BaseEntityAnimating):
    viewport_model = "models/props/water_bottle/water_bottle.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def painttype(self):
        return self._raw_data.get('painttype', "0")



class item_suitcharger(BaseEntityAnimating):
    model = "models/editor/item_suit_charger_hl2.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class light(BasePointLight, BaseLightFalloff):
    icon_sprite = "editor/light.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class light_rt(BasePointLight, BaseLightFalloff, BaseClusteredDynLight):
    icon_sprite = "editor/light.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class light_rt_spot(BaseSpotLight, BaseLightFalloff, BaseClusteredDynLight):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class light_spot(BaseSpotLight, BaseLightFalloff):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class logic_measure_direction(logic_measure_movement):
    icon_sprite = "editor/logic_measure_movement.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def measuretarget(self):
        return self._raw_data.get('measuretarget', None)

    @property
    def measurereference(self):
        return self._raw_data.get('measurereference', None)

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def targetreference(self):
        return self._raw_data.get('targetreference', None)

    @property
    def targetscale(self):
        return parse_source_value(self._raw_data.get('targetscale', 1))

    @property
    def measuretype(self):
        return self._raw_data.get('measuretype', "0")

    @property
    def tracedistance(self):
        return parse_source_value(self._raw_data.get('tracedistance', 8192))

    @property
    def mask(self):
        return self._raw_data.get('mask', "1174421507")

    @property
    def collisiongroup(self):
        return self._raw_data.get('collisiongroup', "0")

    @property
    def damagefilter(self):
        return self._raw_data.get('damagefilter', None)

    @property
    def hitifpassed(self):
        return self._raw_data.get('hitifpassed', "0")

    @property
    def tracetargetreference(self):
        return self._raw_data.get('tracetargetreference', "0")



class math_counter_advanced(math_counter):
    icon_sprite = "editor/l2/math_counter_advanced.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def preservevalue(self):
        return self._raw_data.get('preservevalue', "0")

    @property
    def alwaysoutputasint(self):
        return self._raw_data.get('alwaysoutputasint', "0")

    @property
    def setlerppercent(self):
        return parse_source_value(self._raw_data.get('setlerppercent', 0.5))



class momentary_rot_button(BaseEntityVisBrush, MasterEnt, Angles):

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 50))

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
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def startposition(self):
        return parse_source_value(self._raw_data.get('startposition', 0))

    @property
    def startdirection(self):
        return self._raw_data.get('startdirection', "-1")

    @property
    def solidbsp(self):
        return self._raw_data.get('solidbsp', "1")



class npc_antlion_grub(BaseEntityPhysics):
    model = "models/antlion_grub.mdl"

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)



class npc_antlion_template_maker(Angles, BaseNPCMaker):
    icon_sprite = "editor/npc_maker.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def templatename(self):
        return self._raw_data.get('templatename', None)

    @property
    def spawngroup(self):
        return self._raw_data.get('spawngroup', None)

    @property
    def spawnradius(self):
        return parse_source_value(self._raw_data.get('spawnradius', 512))

    @property
    def spawntarget(self):
        return self._raw_data.get('spawntarget', None)

    @property
    def fighttarget(self):
        return self._raw_data.get('fighttarget', None)

    @property
    def followtarget(self):
        return self._raw_data.get('followtarget', None)

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



class npc_maker(Angles, BaseNPCMaker):
    icon_sprite = "editor/npc_maker.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def npctype(self):
        return self._raw_data.get('npctype', None)

    @property
    def npctargetname(self):
        return self._raw_data.get('npctargetname', None)

    @property
    def npcsquadname(self):
        return self._raw_data.get('npcsquadname', None)

    @property
    def npchintgroup(self):
        return self._raw_data.get('npchintgroup', None)

    @property
    def additionalequipment(self):
        return self._raw_data.get('additionalequipment', "0")



class npc_template_maker(BaseNPCMaker):
    icon_sprite = "editor/ficool2/npc_template_maker.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def templatename(self):
        return self._raw_data.get('templatename', None)

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 256))

    @property
    def destinationgroup(self):
        return self._raw_data.get('destinationgroup', None)

    @property
    def criterionvisibility(self):
        return self._raw_data.get('criterionvisibility', "2")

    @property
    def criteriondistance(self):
        return self._raw_data.get('criteriondistance', "2")

    @property
    def minspawndistance(self):
        return parse_source_value(self._raw_data.get('minspawndistance', 0))



class npc_tripmine(Angles, BaseEntityAnimating):
    model = "models/weapons/w_slam.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class npc_turret_ceiling(BaseEntityAnimating):
    model = "models/combine_turrets/ceiling_turret.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def minhealthdmg(self):
        return parse_source_value(self._raw_data.get('minhealthdmg', 0))



class phys_ballsocket(TwoObjectPhysics):
    icon_sprite = "editor/phys_ballsocket.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class phys_constraint(TwoObjectPhysics):
    icon_sprite = "editor/phys_constraint"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class phys_hinge(TwoObjectPhysics):
    icon_sprite = "editor/ficool2/phys_hinge"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def hingefriction(self):
        return parse_source_value(self._raw_data.get('hingefriction', 0))

    @property
    def hingeaxis(self):
        return self._raw_data.get('hingeaxis', None)

    @property
    def systemloadscale(self):
        return parse_source_value(self._raw_data.get('systemloadscale', 1))

    @property
    def minsoundthreshold(self):
        return parse_source_value(self._raw_data.get('minsoundthreshold', 6))

    @property
    def maxsoundthreshold(self):
        return parse_source_value(self._raw_data.get('maxsoundthreshold', 80))

    @property
    def slidesoundfwd(self):
        return self._raw_data.get('slidesoundfwd', None)

    @property
    def slidesoundback(self):
        return self._raw_data.get('slidesoundback', None)

    @property
    def reversalsoundthresholdsmall(self):
        return parse_source_value(self._raw_data.get('reversalsoundthresholdsmall', 0))

    @property
    def reversalsoundthresholdmedium(self):
        return parse_source_value(self._raw_data.get('reversalsoundthresholdmedium', 0))

    @property
    def reversalsoundthresholdlarge(self):
        return parse_source_value(self._raw_data.get('reversalsoundthresholdlarge', 0))

    @property
    def reversalsoundsmall(self):
        return self._raw_data.get('reversalsoundsmall', None)

    @property
    def reversalsoundmedium(self):
        return self._raw_data.get('reversalsoundmedium', None)

    @property
    def reversalsoundlarge(self):
        return self._raw_data.get('reversalsoundlarge', None)



class phys_lengthconstraint(TwoObjectPhysics):
    model = "models/editor/axis_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def addlength(self):
        return parse_source_value(self._raw_data.get('addlength', 0))

    @property
    def minlength(self):
        return parse_source_value(self._raw_data.get('minlength', 0))

    @property
    def attachpoint(self):
        return self._raw_data.get('attachpoint', None)  # Set to none due to bug in BlackMesa base.fgd file



class phys_magnet(SetModel, BaseEntityAnimating):
    icon_sprite = "editor/phys_magnet"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def forcelimit(self):
        return parse_source_value(self._raw_data.get('forcelimit', 0))

    @property
    def torquelimit(self):
        return parse_source_value(self._raw_data.get('torquelimit', 0))

    @property
    def massscale(self):
        return parse_source_value(self._raw_data.get('massscale', 0))

    @property
    def overridescript(self):
        return self._raw_data.get('overridescript', None)

    @property
    def maxobjects(self):
        return parse_source_value(self._raw_data.get('maxobjects', 0))



class phys_pulleyconstraint(TwoObjectPhysics):
    model = "models/editor/axis_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def addlength(self):
        return parse_source_value(self._raw_data.get('addlength', 0))

    @property
    def gearratio(self):
        return parse_source_value(self._raw_data.get('gearratio', 1))

    @property
    def position2(self):
        return self._raw_data.get('position2', None)



class phys_ragdollconstraint(TwoObjectPhysics):
    model = "models/editor/axis_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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



class phys_slideconstraint(TwoObjectPhysics):
    icon_sprite = "editor/phys_slideconstraint"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def slideaxis(self):
        return self._raw_data.get('slideaxis', None)

    @property
    def slidefriction(self):
        return parse_source_value(self._raw_data.get('slidefriction', 0))

    @property
    def systemloadscale(self):
        return parse_source_value(self._raw_data.get('systemloadscale', 1))

    @property
    def minsoundthreshold(self):
        return parse_source_value(self._raw_data.get('minsoundthreshold', 6))

    @property
    def maxsoundthreshold(self):
        return parse_source_value(self._raw_data.get('maxsoundthreshold', 80))

    @property
    def slidesoundfwd(self):
        return self._raw_data.get('slidesoundfwd', None)

    @property
    def slidesoundback(self):
        return self._raw_data.get('slidesoundback', None)

    @property
    def reversalsoundthresholdsmall(self):
        return parse_source_value(self._raw_data.get('reversalsoundthresholdsmall', 0))

    @property
    def reversalsoundthresholdmedium(self):
        return parse_source_value(self._raw_data.get('reversalsoundthresholdmedium', 0))

    @property
    def reversalsoundthresholdlarge(self):
        return parse_source_value(self._raw_data.get('reversalsoundthresholdlarge', 0))

    @property
    def reversalsoundsmall(self):
        return self._raw_data.get('reversalsoundsmall', None)

    @property
    def reversalsoundmedium(self):
        return self._raw_data.get('reversalsoundmedium', None)

    @property
    def reversalsoundlarge(self):
        return self._raw_data.get('reversalsoundlarge', None)



class phys_thruster(Angles, ForceController):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def force(self):
        return self._raw_data.get('force', "0")



class phys_torque(ForceController):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def force(self):
        return self._raw_data.get('force', "0")

    @property
    def axis(self):
        return self._raw_data.get('axis', None)



class physics_cannister(BaseEntityPhysics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', None)

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def expdamage(self):
        return parse_source_value(self._raw_data.get('expdamage', 200.0))

    @property
    def expradius(self):
        return parse_source_value(self._raw_data.get('expradius', 250.0))

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 25))

    @property
    def thrust(self):
        return parse_source_value(self._raw_data.get('thrust', 3000.0))

    @property
    def fuel(self):
        return parse_source_value(self._raw_data.get('fuel', 12.0))

    @property
    def gassound(self):
        return self._raw_data.get('gassound', "ambient/objects/cannister_loop.wav")



class point_combine_ball_launcher(CombineBallSpawners):
    icon_sprite = "editor/energy_ball.vmt"
    model = "models/editor/cone_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def launchconenoise(self):
        return parse_source_value(self._raw_data.get('launchconenoise', 0.0))

    @property
    def bullseyename(self):
        return self._raw_data.get('bullseyename', None)

    @property
    def maxballbounces(self):
        return parse_source_value(self._raw_data.get('maxballbounces', 8))



class point_commentary_node(EnableDisable, BaseEntityAnimating):
    model = "models/extras/info_speech.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def commentaryfile(self):
        return self._raw_data.get('commentaryfile', None)

    @property
    def commentaryfile_nohdr(self):
        return self._raw_data.get('commentaryfile_nohdr', None)

    @property
    def speakers(self):
        return self._raw_data.get('speakers', None)

    @property
    def prevent_movement(self):
        return self._raw_data.get('prevent_movement', "0")

    @property
    def precommands(self):
        return self._raw_data.get('precommands', None)

    @property
    def postcommands(self):
        return self._raw_data.get('postcommands', None)

    @property
    def viewposition(self):
        return self._raw_data.get('viewposition', None)

    @property
    def viewtarget(self):
        return self._raw_data.get('viewtarget', None)



class point_energy_ball_launcher(CombineBallSpawners):
    icon_sprite = "editor/energy_ball.vmt"
    model = "models/editor/cone_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def maxballbounces(self):
        return parse_source_value(self._raw_data.get('maxballbounces', -1))

    @property
    def balllifetime(self):
        return parse_source_value(self._raw_data.get('balllifetime', 12))

    @property
    def balldamage(self):
        return parse_source_value(self._raw_data.get('balldamage', 1500))

    @property
    def ballexplodeplayer(self):
        return self._raw_data.get('ballexplodeplayer', "1")

    @property
    def ballmaterial1(self):
        return self._raw_data.get('ballmaterial1', "effects/eball_infinite_life")

    @property
    def ballmaterial2(self):
        return self._raw_data.get('ballmaterial2', "effects/combinemuzzle1")

    @property
    def ballknockback(self):
        return parse_source_value(self._raw_data.get('ballknockback', 0))

    @property
    def minlifeafterportal(self):
        return parse_source_value(self._raw_data.get('minlifeafterportal', 6))



class portal_race_checkpoint(BaseEntityAnimating):
    viewport_model = "models/effects/cappoint_hologram.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def resettime(self):
        return parse_source_value(self._raw_data.get('resettime', 5.0))



class prop_coreball(BaseEntityAnimating):
    model = "models/props_combine/coreball.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class prop_door_rotating(SetModel, BaseEntityAnimating):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

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



class prop_dropship_container(BaseEntityAnimating):
    model = "models/combine_dropship_container.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class prop_dynamic_base(SetModel, BreakableProp, BaseEntityAnimating):

    @property
    def defaultanim(self):
        return self._raw_data.get('defaultanim', None)

    @property
    def randomanimation(self):
        return self._raw_data.get('randomanimation', "0")

    @property
    def minanimtime(self):
        return parse_source_value(self._raw_data.get('minanimtime', 5))

    @property
    def maxanimtime(self):
        return parse_source_value(self._raw_data.get('maxanimtime', 10))

    @property
    def disablebonefollowers(self):
        return self._raw_data.get('disablebonefollowers', "0")

    @property
    def holdanimation(self):
        return self._raw_data.get('holdanimation', "0")

    @property
    def suppressanimsounds(self):
        return self._raw_data.get('suppressanimsounds', "0")

    @property
    def animateeveryframe(self):
        return self._raw_data.get('animateeveryframe', "1")



class prop_exploding_futbol(SRCModel, BaseEntityPhysics):
    viewport_model = "models/npcs/personality_sphere_angry.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def explodeontouch(self):
        return self._raw_data.get('explodeontouch', "1")

    @property
    def model(self):
        return self._raw_data.get('model', "models/npcs/personality_sphere_angry.mdl")



class prop_hallucination(SetModel, BaseEntityAnimating):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def enabledchance(self):
        return parse_source_value(self._raw_data.get('enabledchance', 100.0))

    @property
    def visibletime(self):
        return parse_source_value(self._raw_data.get('visibletime', 0.215))

    @property
    def rechargetime(self):
        return parse_source_value(self._raw_data.get('rechargetime', 0.0))



class prop_laser_catcher(SRCIndicator, BaseEntityAnimating):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def skintype(self):
        return self._raw_data.get('skintype', "0")

    @property
    def skin(self):
        return self._raw_data.get('skin', "0")

    @property
    def model(self):
        return self._raw_data.get('model', "models/props/laser_catcher_center.mdl")

    @property
    def uselaserfilter(self):
        return self._raw_data.get('uselaserfilter', "0")

    @property
    def filtercolor(self):
        return self._raw_data.get('filtercolor', "255 255 255 255")

    @property
    def src_fix_skins(self):
        return self._raw_data.get('src_fix_skins', "1")



class prop_laser_relay(SetSkin, SRCIndicator, BaseEntityAnimating):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/props/laser_receptacle.mdl")

    @property
    def uselaserfilter(self):
        return self._raw_data.get('uselaserfilter', "0")

    @property
    def filtercolor(self):
        return self._raw_data.get('filtercolor', "255 255 255 255")



class prop_linked_portal_door(LinkedPortalDoor, BaseEntityAnimating):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def partnername(self):
        return self._raw_data.get('partnername', None)

    @property
    def lightingorigin(self):
        return self._raw_data.get('lightingorigin', None)

    @property
    def model(self):
        return self._raw_data.get('model', "models/props/portal_door.mdl")



class prop_mirror(SetModel, BaseEntityAnimating):
    model = "models/editor/angle_helper.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def width(self):
        return parse_source_value(self._raw_data.get('width', 128.0))

    @property
    def height(self):
        return parse_source_value(self._raw_data.get('height', 216.0))

    @property
    def physicsenabled(self):
        return self._raw_data.get('physicsenabled', "0")



class prop_monster_box(PaintableProp, BaseEntityPhysics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def startasbox(self):
        return self._raw_data.get('startasbox', "0")

    @property
    def boxswitchspeed(self):
        return parse_source_value(self._raw_data.get('boxswitchspeed', 400))

    @property
    def model(self):
        return self._raw_data.get('model', "models/npcs/monsters/monster_a.mdl")



class prop_paint_bomb(BasePaintType, BaseEntityPhysics):
    model = "models/editor/prop_paint_bomb.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def bombtype(self):
        return self._raw_data.get('bombtype', "0")

    @property
    def allowfunnel(self):
        return self._raw_data.get('allowfunnel', "1")

    @property
    def playspawnsound(self):
        return self._raw_data.get('playspawnsound', "1")

    @property
    def model(self):
        return self._raw_data.get('model', "models/error.mdl")



class prop_portal_stats_display(BaseEntityAnimating):
    viewport_model = "models/props/Round_elevator_body.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class prop_ragdoll(EnableDisable, SetModel, SystemLevelChoice, BaseEntityPhysics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def angleoverride(self):
        return self._raw_data.get('angleoverride', None)



class prop_rocket_tripwire(BaseEntityAnimating):
    viewport_model = "models/props/tripwire_turret.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def rocketspeed(self):
        return parse_source_value(self._raw_data.get('rocketspeed', 450))

    @property
    def rocketlifetime(self):
        return parse_source_value(self._raw_data.get('rocketlifetime', 20))

    @property
    def startdisabled(self):
        return self._raw_data.get('startdisabled', "0")



class prop_scalable(SetModel, BaseEntityAnimating):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class prop_telescopic_arm(BaseEntityAnimating):
    viewport_model = "models/props/telescopic_arm.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class prop_testchamber_door(BaseEntityAnimating):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def lightingorigin(self):
        return self._raw_data.get('lightingorigin', None)

    @property
    def areaportalwindow(self):
        return self._raw_data.get('areaportalwindow', None)

    @property
    def useareaportalfade(self):
        return self._raw_data.get('useareaportalfade', "0")

    @property
    def areaportalfadestart(self):
        return parse_source_value(self._raw_data.get('areaportalfadestart', 0))

    @property
    def areaportalfadeend(self):
        return parse_source_value(self._raw_data.get('areaportalfadeend', 0))

    @property
    def open(self):
        return self._raw_data.get('open', "0")

    @property
    def locked(self):
        return self._raw_data.get('locked', "0")

    @property
    def skin(self):
        return parse_source_value(self._raw_data.get('skin', 0))

    @property
    def model(self):
        return self._raw_data.get('model', "models/props/portal_door_combined.mdl")

    @property
    def defaultanim(self):
        return self._raw_data.get('defaultanim', "idleclose")



class prop_thumper(BaseEntityAnimating):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/props_combine/CombineThumper002.mdl")

    @property
    def dustscale(self):
        return self._raw_data.get('dustscale', "128")

    @property
    def effectradius(self):
        return parse_source_value(self._raw_data.get('effectradius', 1000))



class scripted_scene(logic_choreographed_scene):
    icon_sprite = "editor/choreo_scene.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class simple_physics_brush(BaseEntityVisBrush):
    pass


class trigger_autosave(TriggerOnce):

    @property
    def newlevelunit(self):
        return self._raw_data.get('newlevelunit', "0")

    @property
    def dangeroustimer(self):
        return parse_source_value(self._raw_data.get('dangeroustimer', 0))

    @property
    def minimumhitpoints(self):
        return parse_source_value(self._raw_data.get('minimumhitpoints', 0))



class trigger_once(TriggerOnce):
    pass


class weapon_357(Weapon):
    viewport_model = "models/weapons/w_357.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class weapon_alyxgun(Weapon):
    viewport_model = "models/weapons/W_Alyx_Gun.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class weapon_annabelle(Weapon):
    viewport_model = "models/weapons/W_annabelle.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class weapon_ar2(Weapon):
    viewport_model = "models/weapons/w_irifle.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class weapon_bugbait(Weapon):
    viewport_model = "models/weapons/w_bugbait.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class weapon_citizenpackage(Weapon):
    viewport_model = "models/weapons/w_package.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class weapon_citizensuitcase(Weapon):
    viewport_model = "models/weapons/w_suitcase_passenger.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class weapon_crossbow(Weapon):
    viewport_model = "models/weapons/w_crossbow.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class weapon_crowbar(Weapon):
    viewport_model = "models/weapons/w_crowbar.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class weapon_cubemap(Weapon):
    viewport_model = "models/shadertest/envballs.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class weapon_frag(Weapon):
    viewport_model = "models/weapons/w_grenade.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class weapon_paintgun(Weapon):
    viewport_model = "models/weapons/w_portalgun.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class weapon_physcannon(Weapon):
    viewport_model = "models/weapons/w_physics.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class weapon_physgun(Weapon):
    viewport_model = "models/weapons/w_physics.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class weapon_pistol(Weapon):
    viewport_model = "models/weapons/w_pistol.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class weapon_portalgun(Weapon):
    viewport_model = "models/weapons/w_portalgun.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def canfireportal1(self):
        return self._raw_data.get('canfireportal1', "1")

    @property
    def canfireportal2(self):
        return self._raw_data.get('canfireportal2', "1")

    @property
    def showingpotatos(self):
        return self._raw_data.get('showingpotatos', "0")

    @property
    def startingteamnum(self):
        return self._raw_data.get('startingteamnum', "0")

    @property
    def skin(self):
        return self._raw_data.get('skin', "0")

    @property
    def model(self):
        return self._raw_data.get('model', "models/weapons/w_portalgun.mdl")

    @property
    def viewmodel(self):
        return self._raw_data.get('viewmodel', "models/weapons/v_portalgun.mdl")

    @property
    def customportalcolors(self):
        return self._raw_data.get('customportalcolors', "0")

    @property
    def portal1color(self):
        return parse_int_vector(self._raw_data.get('portal1color', "64 160 255"))

    @property
    def portal2color(self):
        return parse_int_vector(self._raw_data.get('portal2color', "255 160 32"))



class weapon_rpg(Weapon):
    viewport_model = "models/weapons/w_rocket_launcher.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class weapon_shotgun(Weapon):
    viewport_model = "models/weapons/w_shotgun.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class weapon_smg1(Weapon):
    viewport_model = "models/weapons/w_smg1.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class weapon_stunstick(Weapon):
    viewport_model = "models/weapons/w_stunbaton.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class BaseNPC(BaseLogicalNPC):

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def expressionoverride(self):
        return self._raw_data.get('expressionoverride', None)

    @property
    def dontusespeechsemaphore(self):
        return self._raw_data.get('dontusespeechsemaphore', "0")

    @property
    def linedivider_npc(self):
        return self._raw_data.get('linedivider_npc', None)



class BaseVehicle(BasePropPhysics):

    @property
    def vehiclescript(self):
        return self._raw_data.get('vehiclescript', "scripts/vehicles/jeep_test.txt")

    @property
    def actionscale(self):
        return parse_source_value(self._raw_data.get('actionscale', 1))



class comp_trigger_p2_goo(Trigger):

    @property
    def enablefade(self):
        return self._raw_data.get('enablefade', "0")

    @property
    def fadepreset(self):
        return self._raw_data.get('fadepreset', "0 0 0")

    @property
    def fadecolor(self):
        return parse_int_vector(self._raw_data.get('fadecolor', "0 0 0"))

    @property
    def fadetime(self):
        return parse_source_value(self._raw_data.get('fadetime', 0.5))

    @property
    def damagetype(self):
        return self._raw_data.get('damagetype', "1327104")

    @property
    def phys_offset(self):
        return parse_source_value(self._raw_data.get('phys_offset', 70.0))

    @property
    def failsafe_delay(self):
        return parse_source_value(self._raw_data.get('failsafe_delay', 1.0))

    @property
    def dissolve_filter(self):
        return self._raw_data.get('dissolve_filter', None)



class func_breakable(BreakableBrush):
    pass


class func_breakable_surf(BreakableBrush):

    @property
    def fragility(self):
        return parse_source_value(self._raw_data.get('fragility', 100))

    @property
    def surfacetype(self):
        return self._raw_data.get('surfacetype', "0")

    @property
    def lowerleft(self):
        return self._raw_data.get('lowerleft', None)

    @property
    def lowerright(self):
        return self._raw_data.get('lowerright', None)

    @property
    def upperleft(self):
        return self._raw_data.get('upperleft', None)

    @property
    def upperright(self):
        return self._raw_data.get('upperright', None)

    @property
    def error(self):
        return self._raw_data.get('error', "0")



class func_bulletshield(func_brush):
    pass


class func_door(Origin, Angles, Door):

    @property
    def movedir(self):
        return parse_float_vector(self._raw_data.get('movedir', "0 0 0"))



class func_door_rotating(Origin, Angles, Door):

    @property
    def distance(self):
        return parse_source_value(self._raw_data.get('distance', 90))



class func_lookdoor(func_movelinear):

    @property
    def proximitydistance(self):
        return self._raw_data.get('proximitydistance', "0.0")

    @property
    def proximityoffset(self):
        return self._raw_data.get('proximityoffset', "0.0")

    @property
    def fieldofview(self):
        return self._raw_data.get('fieldofview', "0.0")



class func_monitor(func_brush):

    @property
    def target(self):
        return self._raw_data.get('target', None)



class func_physbox(BreakableBrush):

    @property
    def damagetype(self):
        return self._raw_data.get('damagetype', "0")

    @property
    def massscale(self):
        return parse_source_value(self._raw_data.get('massscale', 0))

    @property
    def overridescript(self):
        return self._raw_data.get('overridescript', None)

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
    def exploitablebyplayer(self):
        return self._raw_data.get('exploitablebyplayer', "0")

    @property
    def ha_override_mass(self):
        return parse_source_value(self._raw_data.get('ha_override_mass', None))



class func_placement_clip(Trigger):
    pass


class func_reflective_glass(func_brush):
    pass


class func_tanktrain(BaseTrain):

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 100))



class func_trackautochange(func_trackchange):
    pass


class func_tracktrain(BaseTrain):

    @property
    def manualspeedchanges(self):
        return self._raw_data.get('manualspeedchanges', "0")

    @property
    def manualaccelspeed(self):
        return parse_source_value(self._raw_data.get('manualaccelspeed', 0))

    @property
    def manualdecelspeed(self):
        return parse_source_value(self._raw_data.get('manualdecelspeed', 0))



class func_water_analog(func_movelinear):

    @property
    def waveheight(self):
        return self._raw_data.get('waveheight', "3.0")



class hunter_flechette(BasePropPhysics):
    viewport_model = "models/weapons/hunter_flechette.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_ammo_357(Item):
    model = "models/items/357ammo.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_ammo_357_large(Item):
    model = "models/items/357ammobox.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_ammo_ar2(Item):
    model = "models/items/combine_rifle_cartridge01.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_ammo_ar2_altfire(Item):
    model = "models/items/combine_rifle_ammo01.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_ammo_ar2_large(Item):
    model = "models/items/combine_rifle_cartridge01.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_ammo_crossbow(Item):
    model = "models/items/crossbowrounds.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_ammo_pistol(Item):
    model = "models/items/boxsrounds.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_ammo_pistol_large(Item):
    model = "models/items/boxsrounds.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_ammo_smg1(Item):
    model = "models/items/BoxMRounds.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_ammo_smg1_grenade(Item):
    model = "models/items/AR2_Grenade.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_ammo_smg1_large(Item):
    model = "models/items/boxmrounds.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_ar2_grenade(Item):
    model = "models/items/AR2_Grenade.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_battery(Item):
    model = "models/items/battery.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_boots(Item):
    model = "models/items/item_boots.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_box_buckshot(Item):
    model = "models/items/BoxBuckshot.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_dynamic_resupply(Item):
    icon_sprite = "editor/item_dynamic_resupply.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def desiredhealth(self):
        return parse_source_value(self._raw_data.get('desiredhealth', 1))

    @property
    def desiredarmor(self):
        return parse_source_value(self._raw_data.get('desiredarmor', 0.3))

    @property
    def desiredammopistol(self):
        return parse_source_value(self._raw_data.get('desiredammopistol', 0.5))

    @property
    def desiredammosmg1(self):
        return parse_source_value(self._raw_data.get('desiredammosmg1', 0.5))

    @property
    def desiredammosmg1_grenade(self):
        return parse_source_value(self._raw_data.get('desiredammosmg1_grenade', 0.1))

    @property
    def desiredammoar2(self):
        return parse_source_value(self._raw_data.get('desiredammoar2', 0.4))

    @property
    def desiredammobuckshot(self):
        return parse_source_value(self._raw_data.get('desiredammobuckshot', 0.5))

    @property
    def desiredammorpg_round(self):
        return parse_source_value(self._raw_data.get('desiredammorpg_round', 0))

    @property
    def desiredammogrenade(self):
        return parse_source_value(self._raw_data.get('desiredammogrenade', 0.1))

    @property
    def desiredammo357(self):
        return parse_source_value(self._raw_data.get('desiredammo357', 0))

    @property
    def desiredammocrossbow(self):
        return parse_source_value(self._raw_data.get('desiredammocrossbow', 0))

    @property
    def desiredammoar2_altfire(self):
        return parse_source_value(self._raw_data.get('desiredammoar2_altfire', 0))



class item_grubnugget(Item):
    model = "models/grub_nugget_small.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_healthkit(Item):
    model = "models/items/healthkit.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_healthvial(Item):
    model = "models/healthvial.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_item_crate(Angles, BasePropPhysics):
    viewport_model = "models/items/item_item_crate.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def cratetype(self):
        return self._raw_data.get('cratetype', "0")

    @property
    def crateappearance(self):
        return self._raw_data.get('crateappearance', "0")

    @property
    def itemclass(self):
        return self._raw_data.get('itemclass', "item_dynamic_resupply")

    @property
    def itemcount(self):
        return parse_source_value(self._raw_data.get('itemcount', 1))

    @property
    def specificresupply(self):
        return self._raw_data.get('specificresupply', None)



class item_rpg_round(Item):
    model = "models/weapons/w_missile_closed.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_suit(Item):
    model = "models/items/hevsuit.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class npc_bullseye(BaseLogicalNPC):
    icon_sprite = "editor/bullseye.vmt"

    @property
    def linedivider_npc(self):
        return self._raw_data.get('linedivider_npc', None)

    @property
    def minangle(self):
        return parse_source_value(self._raw_data.get('minangle', 360))

    @property
    def mindist(self):
        return parse_source_value(self._raw_data.get('mindist', 0))

    @property
    def autoaimradius(self):
        return parse_source_value(self._raw_data.get('autoaimradius', 0))

    @property
    def alwaystransmit(self):
        return self._raw_data.get('alwaystransmit', "0")



class npc_enemyfinder(BaseLogicalNPC):
    icon_sprite = "editor/npc_enemyfinder.vmt"

    @property
    def fieldofview(self):
        return self._raw_data.get('fieldofview', "0.2")

    @property
    def minsearchdist(self):
        return parse_source_value(self._raw_data.get('minsearchdist', 0))

    @property
    def maxsearchdist(self):
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
    def starton(self):
        return self._raw_data.get('starton', "1")



class npc_hunter_maker(npc_template_maker):
    icon_sprite = "editor/npc_maker.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class npc_launcher(BaseLogicalNPC):
    model = "models/weapons/w_rocket_launcher.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def starton(self):
        return self._raw_data.get('starton', "0")

    @property
    def missilemodel(self):
        return self._raw_data.get('missilemodel', "models/weapons/w_missile.mdl")

    @property
    def launchsound(self):
        return self._raw_data.get('launchsound', "npc/waste_scanner/grenade_fire.wav")

    @property
    def flysound(self):
        return self._raw_data.get('flysound', "Missile.Accelerate")

    @property
    def smoketrail(self):
        return self._raw_data.get('smoketrail', "1")

    @property
    def launchsmoke(self):
        return self._raw_data.get('launchsmoke', "1")

    @property
    def launchdelay(self):
        return parse_source_value(self._raw_data.get('launchdelay', 8))

    @property
    def launchspeed(self):
        return parse_source_value(self._raw_data.get('launchspeed', 200))

    @property
    def pathcornername(self):
        return self._raw_data.get('pathcornername', None)

    @property
    def homingspeed(self):
        return parse_source_value(self._raw_data.get('homingspeed', 0))

    @property
    def homingstrength(self):
        return parse_source_value(self._raw_data.get('homingstrength', 10))

    @property
    def homingdelay(self):
        return parse_source_value(self._raw_data.get('homingdelay', 0))

    @property
    def homingrampup(self):
        return parse_source_value(self._raw_data.get('homingrampup', 0.5))

    @property
    def homingduration(self):
        return parse_source_value(self._raw_data.get('homingduration', 5))

    @property
    def homingrampdown(self):
        return parse_source_value(self._raw_data.get('homingrampdown', 1.0))

    @property
    def gravity(self):
        return parse_source_value(self._raw_data.get('gravity', 1.0))

    @property
    def minrange(self):
        return parse_source_value(self._raw_data.get('minrange', 100))

    @property
    def maxrange(self):
        return parse_source_value(self._raw_data.get('maxrange', 2048))

    @property
    def spinmagnitude(self):
        return self._raw_data.get('spinmagnitude', "0")

    @property
    def spinspeed(self):
        return parse_source_value(self._raw_data.get('spinspeed', 0))

    @property
    def damage(self):
        return parse_source_value(self._raw_data.get('damage', 50))

    @property
    def damageradius(self):
        return parse_source_value(self._raw_data.get('damageradius', 200))



class prop_button(BasePedButton, SRCModel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnflags(self):
        return self._raw_data.get('spawnflags', None)

    @property
    def skin(self):
        return self._raw_data.get('skin', "0")

    @property
    def model(self):
        return self._raw_data.get('model', "models/props/switch001.mdl")



class prop_dynamic(prop_dynamic_base, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class prop_dynamic_ornament(prop_dynamic_base):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def initialowner(self):
        return self._raw_data.get('initialowner', None)



class prop_dynamic_override(prop_dynamic_base, EnableDisable):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class prop_floor_ball_button(BasePortButton, SRCModel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/props/ball_button.mdl")



class prop_floor_button(BasePortButton):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/props/portal_button.mdl")



class prop_floor_cube_button(BasePortButton, SRCModel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/props/box_socket.mdl")

    @property
    def acceptsball(self):
        return self._raw_data.get('acceptsball', "0")



class prop_glados_core(BasePropPhysics, SRCModel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def coretype(self):
        return self._raw_data.get('coretype', "0")

    @property
    def delaybetweenlines(self):
        return parse_source_value(self._raw_data.get('delaybetweenlines', 0.4))

    @property
    def model(self):
        return self._raw_data.get('model', "models/npcs/personality_sphere/personality_sphere.mdl")



class prop_glass_futbol(SRCModel, BasePropPhysics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def spawnername(self):
        return self._raw_data.get('spawnername', None)

    @property
    def model(self):
        return self._raw_data.get('model', "models/props/futbol.mdl")



class prop_physics(SetModel, BasePropPhysics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def allowfunnel(self):
        return self._raw_data.get('allowfunnel', "1")

    @property
    def exploitablebyplayer(self):
        return self._raw_data.get('exploitablebyplayer', "0")



class prop_physics_ragdoll(prop_ragdoll):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class prop_sphere(SetModel, BasePropPhysics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 12))



class prop_tractor_beam(BaseProjector):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def linearforce(self):
        return self._raw_data.get('linearforce', "250")

    @property
    def noemitterparticles(self):
        return self._raw_data.get('noemitterparticles', "0")

    @property
    def use128model(self):
        return self._raw_data.get('use128model', "0")

    @property
    def model(self):
        return self._raw_data.get('model', "models/props/tractor_beam_emitter.mdl")

    @property
    def skin(self):
        return parse_source_value(self._raw_data.get('skin', 0))

    @property
    def primarycolor(self):
        return parse_int_vector(self._raw_data.get('primarycolor', "10 160 255 255"))

    @property
    def secondarycolor(self):
        return parse_int_vector(self._raw_data.get('secondarycolor', "255 160 32 255"))



class prop_under_button(BasePedButton, SRCModel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/props_underground/underground_testchamber_button.mdl")



class prop_under_floor_button(BasePortButton, SRCModel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/props_underground/underground_floor_button.mdl")



class prop_wall_projector(BaseProjector, SRCModel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def skin(self):
        return self._raw_data.get('skin', "0")

    @property
    def model(self):
        return self._raw_data.get('model', "models/props/wall_emitter.mdl")



class prop_weighted_cube(PaintableProp, BasePropPhysics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def line_cube(self):
        return self._raw_data.get('line_cube', None)

    @property
    def cubebehavior(self):
        return self._raw_data.get('cubebehavior', "0")

    @property
    def cubeshape(self):
        return self._raw_data.get('cubeshape', "0")

    @property
    def model(self):
        return self._raw_data.get('model', "models/props/cubes/standard_cube.mdl")

    @property
    def paintpower(self):
        return self._raw_data.get('paintpower', "4")

    @property
    def allowfunnel(self):
        return self._raw_data.get('allowfunnel', "1")

    @property
    def uselasermodifier(self):
        return self._raw_data.get('uselasermodifier', "0")

    @property
    def reflectmodifycolor(self):
        return self._raw_data.get('reflectmodifycolor', "255 0 0 255")

    @property
    def uselaserfilter(self):
        return self._raw_data.get('uselaserfilter', "0")

    @property
    def reflectfiltercolor(self):
        return self._raw_data.get('reflectfiltercolor', "255 0 0 255")

    @property
    def line_cube_oldskins(self):
        return self._raw_data.get('line_cube_oldskins', None)

    @property
    def newskins(self):
        return self._raw_data.get('newskins', "2")

    @property
    def cubetype(self):
        return self._raw_data.get('cubetype', "0")

    @property
    def skintype(self):
        return self._raw_data.get('skintype', "0")

    @property
    def skin(self):
        return parse_source_value(self._raw_data.get('skin', 0))

    @property
    def comp_custom_model_type(self):
        return self._raw_data.get('comp_custom_model_type', "0")



class simple_physics_prop(BasePropPhysics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def allowfunnel(self):
        return self._raw_data.get('allowfunnel', "1")



class trigger_catapult(Trigger):

    @property
    def playerspeed(self):
        return parse_source_value(self._raw_data.get('playerspeed', 450))

    @property
    def physicsspeed(self):
        return parse_source_value(self._raw_data.get('physicsspeed', 450))

    @property
    def launchdirection(self):
        return parse_float_vector(self._raw_data.get('launchdirection', "0 0 0"))

    @property
    def launchtarget(self):
        return self._raw_data.get('launchtarget', None)

    @property
    def useexactvelocity(self):
        return self._raw_data.get('useexactvelocity', "0")

    @property
    def exactvelocitychoicetype(self):
        return self._raw_data.get('exactvelocitychoicetype', "0")

    @property
    def applyangularimpulse(self):
        return self._raw_data.get('applyangularimpulse', "1")

    @property
    def airctrlsupressiontime(self):
        return parse_source_value(self._raw_data.get('airctrlsupressiontime', -1.0))

    @property
    def directionsuppressaircontrol(self):
        return self._raw_data.get('directionsuppressaircontrol', "0")

    @property
    def usethresholdcheck(self):
        return self._raw_data.get('usethresholdcheck', "0")

    @property
    def onlyvelocitycheck(self):
        return self._raw_data.get('onlyvelocitycheck', "0")

    @property
    def absolutevelocitycheck(self):
        return self._raw_data.get('absolutevelocitycheck', "0")

    @property
    def lowerthreshold(self):
        return parse_source_value(self._raw_data.get('lowerthreshold', 0.15))

    @property
    def upperthreshold(self):
        return parse_source_value(self._raw_data.get('upperthreshold', 0.30))

    @property
    def entryangletolerance(self):
        return parse_source_value(self._raw_data.get('entryangletolerance', 0.0))

    @property
    def launchsound(self):
        return self._raw_data.get('launchsound', None)



class trigger_changelevel(Trigger):

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

    @property
    def persist(self):
        return self._raw_data.get('persist', "0")

    @property
    def gravityvector(self):
        return parse_float_vector(self._raw_data.get('gravityvector', "90 0 0"))



class trigger_hurt(MasterEnt, Trigger, DamageType):

    @property
    def damage(self):
        return parse_source_value(self._raw_data.get('damage', 100000))

    @property
    def damagecap(self):
        return parse_source_value(self._raw_data.get('damagecap', 20))

    @property
    def damagemodel(self):
        return self._raw_data.get('damagemodel', "0")

    @property
    def nodmgforce(self):
        return self._raw_data.get('nodmgforce', "0")



class trigger_impact(Angles, Origin, Trigger):

    @property
    def magnitude(self):
        return parse_source_value(self._raw_data.get('magnitude', 200))

    @property
    def noise(self):
        return parse_source_value(self._raw_data.get('noise', 0.1))

    @property
    def viewkick(self):
        return parse_source_value(self._raw_data.get('viewkick', 0.05))



class trigger_jumppad(Trigger):

    @property
    def launchtarget(self):
        return self._raw_data.get('launchtarget', None)

    @property
    def launchsound(self):
        return self._raw_data.get('launchsound', None)

    @property
    def keephorizontalspeed(self):
        return parse_source_value(self._raw_data.get('keephorizontalspeed', 0))

    @property
    def keepverticalspeed(self):
        return parse_source_value(self._raw_data.get('keepverticalspeed', 0))



class trigger_look(Trigger):

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def looktime(self):
        return self._raw_data.get('looktime', "0.5")

    @property
    def fieldofview(self):
        return self._raw_data.get('fieldofview', "0.9")

    @property
    def timeout(self):
        return parse_source_value(self._raw_data.get('timeout', 0))



class trigger_multiple(Trigger):

    @property
    def wait(self):
        return parse_source_value(self._raw_data.get('wait', 1))



class trigger_paint_cleanser(Trigger):
    pass


class trigger_physics_trap(Angles, Trigger):

    @property
    def dissolvetype(self):
        return self._raw_data.get('dissolvetype', "0")



class trigger_ping_detector(Trigger):
    pass


class trigger_playermovement(Trigger):
    pass


class trigger_playerteam(Trigger):

    @property
    def target_team(self):
        return self._raw_data.get('target_team', "0")

    @property
    def trigger_once(self):
        return self._raw_data.get('trigger_once', "0")



class trigger_portal_cleanser(Reflection, RenderFields, Trigger):

    @property
    def visible(self):
        return self._raw_data.get('visible', "1")

    @property
    def usescanline(self):
        return self._raw_data.get('usescanline', "1")



class trigger_proximity(Trigger):

    @property
    def measuretarget(self):
        return self._raw_data.get('measuretarget', None)

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 256))



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



class trigger_remove(Trigger):
    pass


class trigger_rpgfire(Trigger):
    pass


class trigger_serverragdoll(Trigger):
    pass


class trigger_setspeed(Trigger):

    @property
    def horizontalspeedmode(self):
        return self._raw_data.get('horizontalspeedmode', "0")

    @property
    def horizontalspeed(self):
        return parse_source_value(self._raw_data.get('horizontalspeed', 500))

    @property
    def horizontalspeedangle(self):
        return parse_float_vector(self._raw_data.get('horizontalspeedangle', "0 0 0"))

    @property
    def verticalspeedmode(self):
        return self._raw_data.get('verticalspeedmode', "4")

    @property
    def verticalspeed(self):
        return parse_source_value(self._raw_data.get('verticalspeed', 0))

    @property
    def strictmode(self):
        return self._raw_data.get('strictmode', "0")



class trigger_softbarrier(Trigger):

    @property
    def pushdir(self):
        return parse_float_vector(self._raw_data.get('pushdir', "0 0 0"))



class trigger_soundoperator(Trigger):

    @property
    def sosvar(self):
        return self._raw_data.get('sosvar', "0")



class trigger_soundscape(Trigger):

    @property
    def soundscape(self):
        return self._raw_data.get('soundscape', None)



class trigger_teleport(Trigger):

    @property
    def target(self):
        return self._raw_data.get('target', None)

    @property
    def landmark(self):
        return self._raw_data.get('landmark', None)

    @property
    def uselandmarkangles(self):
        return self._raw_data.get('uselandmarkangles', "1")

    @property
    def checkdestifclearforplayer(self):
        return self._raw_data.get('checkdestifclearforplayer', "0")

    @property
    def teleportonendtouch(self):
        return self._raw_data.get('teleportonendtouch', "0")

    @property
    def velocitymode(self):
        return self._raw_data.get('velocitymode', "0")

    @property
    def setspeed(self):
        return parse_source_value(self._raw_data.get('setspeed', 400))



class trigger_teleport_relative(Trigger):

    @property
    def teleportoffset(self):
        return parse_float_vector(self._raw_data.get('teleportoffset', "0 0 0"))



class trigger_togglesave(EnableDisable, Trigger):
    pass


class trigger_tonemap(Trigger):

    @property
    def tonemapname(self):
        return self._raw_data.get('tonemapname', None)



class trigger_transition(Origin, Trigger):
    pass


class trigger_userinput(Trigger):

    @property
    def lookedkey(self):
        return self._raw_data.get('lookedkey', "0")

    @property
    def lookedkey2(self):
        return self._raw_data.get('lookedkey2', "-1")

    @property
    def lookedkey3(self):
        return self._raw_data.get('lookedkey3', "-1")



class trigger_vphysics_motion(Trigger):

    @property
    def setgravityscale(self):
        return parse_source_value(self._raw_data.get('setgravityscale', 1.0))

    @property
    def setgravitydirection(self):
        return parse_float_vector(self._raw_data.get('setgravitydirection', "0 0 0"))

    @property
    def disableaircontrol(self):
        return self._raw_data.get('disableaircontrol', "1")

    @property
    def setadditionalairdensity(self):
        return parse_source_value(self._raw_data.get('setadditionalairdensity', 0))

    @property
    def setvelocitylimit(self):
        return parse_source_value(self._raw_data.get('setvelocitylimit', 0.0))

    @property
    def setvelocitylimitdelta(self):
        return parse_source_value(self._raw_data.get('setvelocitylimitdelta', 0.0))

    @property
    def setvelocityscale(self):
        return parse_source_value(self._raw_data.get('setvelocityscale', 1.0))

    @property
    def setangvelocitylimit(self):
        return parse_source_value(self._raw_data.get('setangvelocitylimit', 0.0))

    @property
    def setangvelocityscale(self):
        return parse_source_value(self._raw_data.get('setangvelocityscale', 1.0))

    @property
    def setlinearforce(self):
        return parse_source_value(self._raw_data.get('setlinearforce', 0.0))

    @property
    def setlinearforceangles(self):
        return parse_float_vector(self._raw_data.get('setlinearforceangles', "0 0 0"))

    @property
    def particletrailmaterial(self):
        return self._raw_data.get('particletrailmaterial', None)

    @property
    def particletraillifetime(self):
        return parse_source_value(self._raw_data.get('particletraillifetime', 4))

    @property
    def particletrailstartsize(self):
        return parse_source_value(self._raw_data.get('particletrailstartsize', 2))

    @property
    def particletrailendsize(self):
        return parse_source_value(self._raw_data.get('particletrailendsize', 3))



class trigger_waterydeath(Trigger):
    pass


class trigger_weapon_dissolve(Trigger):

    @property
    def emittername(self):
        return self._raw_data.get('emittername', None)



class trigger_weapon_strip(Trigger):

    @property
    def killweapons(self):
        return self._raw_data.get('killweapons', "0")



class trigger_wind(Angles, Trigger):

    @property
    def speed(self):
        return parse_source_value(self._raw_data.get('speed', 200))

    @property
    def speednoise(self):
        return parse_source_value(self._raw_data.get('speednoise', 0))

    @property
    def directionnoise(self):
        return parse_source_value(self._raw_data.get('directionnoise', 10))

    @property
    def holdtime(self):
        return parse_source_value(self._raw_data.get('holdtime', 0))

    @property
    def holdnoise(self):
        return parse_source_value(self._raw_data.get('holdnoise', 0))



class weapon_striderbuster(BasePropPhysics):
    viewport_model = "models/magnusson_device.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def dud(self):
        return self._raw_data.get('dud', "0")



class BaseDriveableVehicle(BaseVehicle):

    @property
    def vehiclelocked(self):
        return self._raw_data.get('vehiclelocked', "0")



class BaseHeadcrab(BaseNPC):

    @property
    def startburrowed(self):
        return self._raw_data.get('startburrowed', "0")



class BaseHelicopter(BaseNPC):

    @property
    def initialspeed(self):
        return parse_source_value(self._raw_data.get('initialspeed', 0))



class PlayerCompanion(BaseNPC):

    @property
    def alwaystransition(self):
        return self._raw_data.get('alwaystransition', "0")

    @property
    def dontpickupweapons(self):
        return self._raw_data.get('dontpickupweapons', "0")

    @property
    def gameendally(self):
        return self._raw_data.get('gameendally', "0")



class RappelNPC(BaseNPC):

    @property
    def waitingtorappel(self):
        return self._raw_data.get('waitingtorappel', "0")



class TalkNPC(BaseNPC):

    @property
    def usesentence(self):
        return self._raw_data.get('usesentence', None)

    @property
    def unusesentence(self):
        return self._raw_data.get('unusesentence', None)

    @property
    def dontusespeechsemaphore(self):
        return self._raw_data.get('dontusespeechsemaphore', "0")



class VehicleDriverNPC(BaseNPC):

    @property
    def vehicle(self):
        return self._raw_data.get('vehicle', None)



class comp_trigger_coop(trigger_playerteam):
    pass


class cycler_actor(SetModel, BaseNPC):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def sentence(self):
        return self._raw_data.get('sentence', None)



class func_physbox_multiplayer(func_physbox):
    pass


class func_pushable(func_breakable):

    @property
    def size(self):
        return self._raw_data.get('size', "0")

    @property
    def friction(self):
        return parse_source_value(self._raw_data.get('friction', 50))



class func_touch(BaseEntityVisBrush, trigger_multiple):
    pass


class func_water(func_door):
    pass


class generic_actor(SetModel, BaseNPC):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def hull_name(self):
        return self._raw_data.get('hull_name', "HUMAN_HULL")

    @property
    def defaultanim(self):
        return self._raw_data.get('defaultanim', None)

    @property
    def randomanimation(self):
        return self._raw_data.get('randomanimation', "0")



class hot_potato(prop_exploding_futbol, prop_glass_futbol):
    viewport_model = "models/props/futbol.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_box_lrounds(item_ammo_ar2):
    model = "models/items/combine_rifle_cartridge01.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_box_mrounds(item_ammo_smg1):
    model = "models/items/BoxMRounds.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_box_srounds(item_ammo_pistol):
    model = "models/items/boxsrounds.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_large_box_lrounds(item_ammo_ar2_large):
    model = "models/items/combine_rifle_cartridge01.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_large_box_mrounds(item_ammo_smg1_large):
    model = "models/items/boxmrounds.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_large_box_srounds(item_ammo_pistol_large):
    model = "models/items/boxsrounds.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class monster_generic(BaseNPC):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', None)



class npc_advisor(BaseNPC):

    @property
    def model(self):
        return self._raw_data.get('model', "models/advisor.mdl")

    @property
    def levitationarea(self):
        return self._raw_data.get('levitationarea', None)

    @property
    def levitategoal_bottom(self):
        return self._raw_data.get('levitategoal_bottom', None)

    @property
    def levitategoal_top(self):
        return self._raw_data.get('levitategoal_top', None)

    @property
    def staging_ent_names(self):
        return self._raw_data.get('staging_ent_names', None)

    @property
    def priority_grab_name(self):
        return self._raw_data.get('priority_grab_name', None)



class npc_antlion(BaseNPC):
    model = "models/antlion.mdl"

    @property
    def radius(self):
        return parse_source_value(self._raw_data.get('radius', 256))

    @property
    def eludedist(self):
        return parse_source_value(self._raw_data.get('eludedist', 1024))

    @property
    def ignorebugbait(self):
        return self._raw_data.get('ignorebugbait', "0")

    @property
    def unburroweffects(self):
        return self._raw_data.get('unburroweffects', "0")

    @property
    def startburrowed(self):
        return self._raw_data.get('startburrowed', "0")



class npc_antlionguard(BaseNPC):
    model = "models/antlion_guard.mdl"

    @property
    def startburrowed(self):
        return self._raw_data.get('startburrowed', "0")

    @property
    def allowbark(self):
        return self._raw_data.get('allowbark', "0")

    @property
    def cavernbreed(self):
        return self._raw_data.get('cavernbreed', "0")

    @property
    def incavern(self):
        return self._raw_data.get('incavern', "0")

    @property
    def shovetargets(self):
        return self._raw_data.get('shovetargets', None)



class npc_barnacle(BaseFadeProp, BaseNPC):
    model = "models/Barnacle.mdl"

    @property
    def restdist(self):
        return parse_source_value(self._raw_data.get('restdist', 16))



class npc_clawscanner(CombineScanner, BaseNPC):
    viewport_model = "models/shield_scanner.mdl"
    pass


class npc_combine_camera(BaseNPC):
    model = "models/combine_camera/combine_camera.mdl"

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
        return self._raw_data.get('defaulttarget', None)



class npc_combine_cannon(BaseNPC):
    model = "models/combine_soldier.mdl"

    @property
    def sightdist(self):
        return parse_source_value(self._raw_data.get('sightdist', 1024))



class npc_crow(BaseNPC):
    model = "models/crow.mdl"

    @property
    def deaf(self):
        return self._raw_data.get('deaf', "0")



class npc_cscanner(CombineScanner, BaseNPC):

    @property
    def model(self):
        return self._raw_data.get('model', "models/combine_scanner.mdl")



class npc_dog(BaseNPC):
    model = "models/dog.mdl"
    pass


class npc_enemyfinder_combinecannon(npc_enemyfinder):
    icon_sprite = "editor/npc_enemyfinder.vmt"

    @property
    def snaptoent(self):
        return self._raw_data.get('snaptoent', None)



class npc_fastzombie(BaseNPC):
    model = "models/Zombie/fast.mdl"
    pass


class npc_fastzombie_torso(BaseNPC):
    model = "models/Zombie/Fast_torso.mdl"
    pass


class npc_fisherman(BaseNPC):
    model = "models/lostcoast/fisherman/fisherman.mdl"
    pass


class npc_furniture(SetModel, BaseNPC):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class npc_grenade_frag(BaseNPC):
    model = "models/Weapons/w_grenade.mdl"
    pass


class npc_heardanger(SetModel, BaseNPC):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class npc_hunter(BaseNPC):
    model = "models/hunter.mdl"

    @property
    def followtarget(self):
        return self._raw_data.get('followtarget', None)



class npc_ichthyosaur(BaseNPC):
    model = "models/ichthyosaur.mdl"
    pass


class npc_manhack(AlyxInteractable, BaseNPC):
    model = "models/manhack.mdl"

    @property
    def ignoreclipbrushes(self):
        return self._raw_data.get('ignoreclipbrushes', "0")



class npc_missiledefense(SetModel, BaseNPC):
    model = "models/missile_defense.mdl"
    pass


class npc_pigeon(BaseNPC):
    model = "models/pigeon.mdl"

    @property
    def deaf(self):
        return self._raw_data.get('deaf', "0")



class npc_poisonzombie(BaseNPC):
    model = "models/Zombie/Poison.mdl"

    @property
    def crabcount(self):
        return self._raw_data.get('crabcount', "3")



class npc_portal_turret_floor(PaintableProp, BaseNPC):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def damageforce(self):
        return self._raw_data.get('damageforce', "0")

    @property
    def modelindex(self):
        return self._raw_data.get('modelindex', "0")

    @property
    def skinnumber(self):
        return self._raw_data.get('skinnumber', "0")

    @property
    def gagged(self):
        return self._raw_data.get('gagged', "0")

    @property
    def usedasactor(self):
        return self._raw_data.get('usedasactor', "0")

    @property
    def pickupenabled(self):
        return self._raw_data.get('pickupenabled', "1")

    @property
    def disablemotion(self):
        return self._raw_data.get('disablemotion', "0")

    @property
    def allowshootthroughportals(self):
        return self._raw_data.get('allowshootthroughportals', "0")

    @property
    def turretrange(self):
        return parse_source_value(self._raw_data.get('turretrange', 1024))

    @property
    def loadalternativemodels(self):
        return self._raw_data.get('loadalternativemodels', "0")

    @property
    def usesuperdamagescale(self):
        return self._raw_data.get('usesuperdamagescale', "0")

    @property
    def collisiontype(self):
        return self._raw_data.get('collisiontype', "0")

    @property
    def model(self):
        return self._raw_data.get('model', "models/npcs/turret/turret.mdl")

    @property
    def skin(self):
        return parse_source_value(self._raw_data.get('skin', 0))

    @property
    def _frustum_color(self):
        return self._raw_data.get('_frustum_color', "255 0 0")



class npc_puppet(SetModel, BaseNPC):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def animationtarget(self):
        return self._raw_data.get('animationtarget', None)

    @property
    def attachmentname(self):
        return self._raw_data.get('attachmentname', None)



class npc_rocket_turret(ResponseContext, BaseNPC, SRCModel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def rocketspeed(self):
        return parse_source_value(self._raw_data.get('rocketspeed', 450))

    @property
    def rocketlifetime(self):
        return parse_source_value(self._raw_data.get('rocketlifetime', 20))

    @property
    def firecooldown(self):
        return parse_source_value(self._raw_data.get('firecooldown', 4))

    @property
    def lockontime(self):
        return parse_source_value(self._raw_data.get('lockontime', 1.5))

    @property
    def tripwiremode(self):
        return self._raw_data.get('tripwiremode', "0")

    @property
    def tripwireaimtarget(self):
        return self._raw_data.get('tripwireaimtarget', None)

    @property
    def model(self):
        return self._raw_data.get('model', "models/props_bts/rocket_sentry.mdl")

    @property
    def turretrange(self):
        return parse_source_value(self._raw_data.get('turretrange', 8192))



class npc_rollermine(AlyxInteractable, BaseNPC):
    model = "models/roller.mdl"

    @property
    def startburied(self):
        return self._raw_data.get('startburied', "0")

    @property
    def uniformsightdist(self):
        return self._raw_data.get('uniformsightdist', "0")



class npc_seagull(BaseNPC):
    model = "models/seagull.mdl"

    @property
    def deaf(self):
        return self._raw_data.get('deaf', "0")



class npc_security_camera(SetSkin, BaseNPC):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def lookatplayerpings(self):
        return self._raw_data.get('lookatplayerpings', "0")

    @property
    def teamtolookat(self):
        return self._raw_data.get('teamtolookat', "1")

    @property
    def teamplayertolookat(self):
        return self._raw_data.get('teamplayertolookat', "0")

    @property
    def model(self):
        return self._raw_data.get('model', "models/props/security_camera.mdl")



class npc_sniper(BaseNPC):
    model = "models/combine_soldier.mdl"

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
    def shootzombiesinchest(self):
        return self._raw_data.get('shootzombiesinchest', "0")

    @property
    def paintinterval(self):
        return parse_source_value(self._raw_data.get('paintinterval', 1))

    @property
    def paintintervalvariance(self):
        return parse_source_value(self._raw_data.get('paintintervalvariance', 0.75))



class npc_spotlight(BaseNPC):

    @property
    def yawrange(self):
        return parse_source_value(self._raw_data.get('yawrange', 90))

    @property
    def pitchmin(self):
        return parse_source_value(self._raw_data.get('pitchmin', 35))

    @property
    def pitchmax(self):
        return parse_source_value(self._raw_data.get('pitchmax', 50))

    @property
    def idlespeed(self):
        return parse_source_value(self._raw_data.get('idlespeed', 2))

    @property
    def alertspeed(self):
        return parse_source_value(self._raw_data.get('alertspeed', 5))

    @property
    def spotlightlength(self):
        return parse_source_value(self._raw_data.get('spotlightlength', 500))

    @property
    def spotlightwidth(self):
        return parse_source_value(self._raw_data.get('spotlightwidth', 50))



class npc_stalker(BaseNPC):
    model = "models/Stalker.mdl"

    @property
    def beampower(self):
        return self._raw_data.get('beampower', "0")



class npc_strider(BaseNPC):

    @property
    def model(self):
        return self._raw_data.get('model', "models/combine_strider.mdl")

    @property
    def disablephysics(self):
        return self._raw_data.get('disablephysics', "0")



class npc_turret_floor(BaseNPC):
    viewport_model = "models/combine_turrets/floor_turret.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def skinnumber(self):
        return parse_source_value(self._raw_data.get('skinnumber', 0))

    @property
    def _frustum_far(self):
        return parse_source_value(self._raw_data.get('_frustum_far', 1200))

    @property
    def _frustum_color(self):
        return self._raw_data.get('_frustum_color', "255 0 0")



class npc_turret_ground(AlyxInteractable, BaseNPC):
    model = "models/combine_turrets/ground_turret.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class npc_wheatley_boss(SetSkin, BaseNPC):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/npcs/glados/glados_wheatley_boss.mdl")



class npc_zombie(BaseNPC):
    model = "models/Zombie/Classic.mdl"
    pass


class npc_zombie_torso(BaseNPC):
    model = "models/Zombie/Classic_torso.mdl"
    pass


class npc_zombine(BaseNPC):
    model = "models/Zombie/zombie_soldier.mdl"
    pass


class prop_dynamic_glow(prop_dynamic):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

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



class prop_physics_multiplayer(prop_physics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def physicsmode(self):
        return self._raw_data.get('physicsmode', "0")



class prop_physics_override(prop_physics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class prop_physics_paintable(prop_physics, PaintableProp):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def paintpower(self):
        return self._raw_data.get('paintpower', "4")

    @property
    def cleanskin(self):
        return parse_source_value(self._raw_data.get('cleanskin', -1))

    @property
    def paintskin(self):
        return parse_source_value(self._raw_data.get('paintskin', -1))



class prop_physics_respawnable(prop_physics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def respawntime(self):
        return parse_source_value(self._raw_data.get('respawntime', 60))



class prop_stickybomb(weapon_striderbuster):
    viewport_model = "models/magnusson_device.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class prop_vehicle(BaseVehicle, SetModel):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class simple_bot(EnableDisable, BaseNPC):
    model = "models/humans/group01/female_01.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class trigger_hierarchy(trigger_multiple):

    @property
    def childfiltername(self):
        return self._raw_data.get('childfiltername', None)



class npc_alyx(PlayerCompanion, TalkNPC):

    @property
    def model(self):
        return self._raw_data.get('model', "models/alyx.mdl")

    @property
    def shouldhaveemp(self):
        return self._raw_data.get('shouldhaveemp', "1")



class npc_apcdriver(VehicleDriverNPC):
    model = "models/roller.mdl"

    @property
    def drivermaxspeed(self):
        return parse_source_value(self._raw_data.get('drivermaxspeed', 1))

    @property
    def driverminspeed(self):
        return parse_source_value(self._raw_data.get('driverminspeed', 0))



class npc_barney(PlayerCompanion, TalkNPC):
    model = "models/Barney.mdl"
    pass


class npc_breen(TalkNPC):

    @property
    def model(self):
        return self._raw_data.get('model', "models/breen.mdl")



class npc_citizen(PlayerCompanion, TalkNPC):

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
    def notifynavfailblocked(self):
        return self._raw_data.get('notifynavfailblocked', "0")

    @property
    def neverleaveplayersquad(self):
        return self._raw_data.get('neverleaveplayersquad', "0")

    @property
    def denycommandconcept(self):
        return self._raw_data.get('denycommandconcept', None)

    @property
    def alternateaiming(self):
        return self._raw_data.get('alternateaiming', "0")



class npc_combine_s(RappelNPC, GrenadeUser):

    @property
    def model(self):
        return self._raw_data.get('model', "models/combine_soldier.mdl")

    @property
    def tacticalvariant(self):
        return self._raw_data.get('tacticalvariant', "0")

    @property
    def usemarch(self):
        return self._raw_data.get('usemarch', "0")



class npc_combinedropship(BaseHelicopter):
    model = "models/combine_dropship.mdl"

    @property
    def landtarget(self):
        return self._raw_data.get('landtarget', None)

    @property
    def gunrange(self):
        return parse_source_value(self._raw_data.get('gunrange', 2048))

    @property
    def rollerminetemplate(self):
        return self._raw_data.get('rollerminetemplate', None)

    @property
    def npctemplate(self):
        return self._raw_data.get('npctemplate', None)

    @property
    def npctemplate2(self):
        return self._raw_data.get('npctemplate2', None)

    @property
    def npctemplate3(self):
        return self._raw_data.get('npctemplate3', None)

    @property
    def npctemplate4(self):
        return self._raw_data.get('npctemplate4', None)

    @property
    def npctemplate5(self):
        return self._raw_data.get('npctemplate5', None)

    @property
    def npctemplate6(self):
        return self._raw_data.get('npctemplate6', None)

    @property
    def dustoff1(self):
        return self._raw_data.get('dustoff1', None)

    @property
    def dustoff2(self):
        return self._raw_data.get('dustoff2', None)

    @property
    def dustoff3(self):
        return self._raw_data.get('dustoff3', None)

    @property
    def dustoff4(self):
        return self._raw_data.get('dustoff4', None)

    @property
    def dustoff5(self):
        return self._raw_data.get('dustoff5', None)

    @property
    def dustoff6(self):
        return self._raw_data.get('dustoff6', None)

    @property
    def apcvehiclename(self):
        return self._raw_data.get('apcvehiclename', None)

    @property
    def invulnerable(self):
        return self._raw_data.get('invulnerable', "0")

    @property
    def cratetype(self):
        return self._raw_data.get('cratetype', "2")



class npc_combinegunship(BaseHelicopter):
    model = "models/gunship.mdl"

    @property
    def maxangaccel(self):
        return parse_source_value(self._raw_data.get('maxangaccel', 1000))

    @property
    def maxangvelocity(self):
        return parse_float_vector(self._raw_data.get('maxangvelocity', "300 120 300"))



class npc_cranedriver(VehicleDriverNPC):
    model = "models/roller.mdl"

    @property
    def releasepause(self):
        return parse_source_value(self._raw_data.get('releasepause', 0))



class npc_eli(TalkNPC):

    @property
    def model(self):
        return self._raw_data.get('model', "models/eli.mdl")



class npc_gman(TalkNPC):
    model = "models/gman.mdl"
    pass


class npc_headcrab(BaseHeadcrab):
    model = "models/Headcrabclassic.mdl"
    pass


class npc_headcrab_fast(BaseHeadcrab):
    model = "models/Headcrab.mdl"
    pass


class npc_headcrab_poison(BaseHeadcrab):
    model = "models/Headcrabblack.mdl"
    pass


class npc_helicopter(BaseHelicopter):
    model = "models/combine_helicopter.mdl"

    @property
    def graceperiod(self):
        return parse_source_value(self._raw_data.get('graceperiod', 2.0))

    @property
    def patrolspeed(self):
        return parse_source_value(self._raw_data.get('patrolspeed', 0))

    @property
    def noncombat(self):
        return self._raw_data.get('noncombat', "0")



class npc_kleiner(TalkNPC):

    @property
    def model(self):
        return self._raw_data.get('model', "models/kleiner.mdl")



class npc_magnusson(TalkNPC):

    @property
    def model(self):
        return self._raw_data.get('model', "models/magnusson.mdl")



class npc_metropolice(RappelNPC):
    model = "models/Police.mdl"

    @property
    def manhacks(self):
        return self._raw_data.get('manhacks', "0")

    @property
    def weapondrawn(self):
        return self._raw_data.get('weapondrawn', "0")



class npc_monk(TalkNPC):
    model = "models/Monk.mdl"

    @property
    def hasgun(self):
        return self._raw_data.get('hasgun', "1")



class npc_mossman(TalkNPC):
    model = "models/mossman.mdl"
    pass


class npc_personality_core(SRCModel, TalkNPC):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def modelskin(self):
        return self._raw_data.get('modelskin', "0")

    @property
    def altmodel(self):
        return self._raw_data.get('altmodel', "0")

    @property
    def model(self):
        return self._raw_data.get('model', "models/npcs/personality_sphere/personality_sphere.mdl")

    @property
    def skin(self):
        return self._raw_data.get('skin', "0")

    @property
    def flashlightcolor(self):
        return parse_int_vector(self._raw_data.get('flashlightcolor', "255 255 255 500"))

    @property
    def flashlightfov(self):
        return parse_source_value(self._raw_data.get('flashlightfov', 85.0))

    @property
    def flashlightfarz(self):
        return parse_source_value(self._raw_data.get('flashlightfarz', 750.0))



class npc_vehicledriver(VehicleDriverNPC):
    model = "models/roller.mdl"

    @property
    def drivermaxspeed(self):
        return parse_source_value(self._raw_data.get('drivermaxspeed', 1))

    @property
    def driverminspeed(self):
        return parse_source_value(self._raw_data.get('driverminspeed', 0))



class npc_vortigaunt(PlayerCompanion, TalkNPC):

    @property
    def model(self):
        return self._raw_data.get('model', "models/vortigaunt.mdl")

    @property
    def armorrechargeenabled(self):
        return self._raw_data.get('armorrechargeenabled', "1")

    @property
    def healthregenerateenabled(self):
        return self._raw_data.get('healthregenerateenabled', "0")



class prop_vehicle_airboat(BaseDriveableVehicle):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/airboat.mdl")

    @property
    def enablegun(self):
        return self._raw_data.get('enablegun', "0")



class prop_vehicle_apc(BaseDriveableVehicle):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/combine_apc.mdl")

    @property
    def missilehint(self):
        return self._raw_data.get('missilehint', None)



class prop_vehicle_cannon(SetModel, BaseDriveableVehicle):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class prop_vehicle_choreo_generic(SetModel, BaseDriveableVehicle):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def ignoremoveparent(self):
        return self._raw_data.get('ignoremoveparent', "0")

    @property
    def ignoreplayer(self):
        return self._raw_data.get('ignoreplayer', "0")

    @property
    def playercanshoot(self):
        return self._raw_data.get('playercanshoot', "0")

    @property
    def useattachmenteyes(self):
        return self._raw_data.get('useattachmenteyes', "0")



class prop_vehicle_crane(BaseDriveableVehicle):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/cranes/crane_docks.mdl")

    @property
    def magnetname(self):
        return self._raw_data.get('magnetname', None)



class prop_vehicle_driveable(SetModel, BaseDriveableVehicle):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class prop_vehicle_jeep(BaseDriveableVehicle):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/buggy.mdl")

    @property
    def cargovisible(self):
        return self._raw_data.get('cargovisible', "0")

    @property
    def nohazardlights(self):
        return self._raw_data.get('nohazardlights', "0")

    @property
    def enablegun(self):
        return self._raw_data.get('enablegun', "0")



class prop_vehicle_prisoner_pod(SetSkin, BaseDriveableVehicle):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def model(self):
        return self._raw_data.get('model', "models/vehicles/prisoner_pod.mdl")



class vehicle_viewcontroller(BaseDriveableVehicle):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class npc_headcrab_black(npc_headcrab_poison):
    model = "models/Headcrabblack.mdl"
    pass



entity_class_handle = {
    'AlyxInteractable': AlyxInteractable,
    'Angles': Angles,
    'BaseClusteredLight': BaseClusteredLight,
    'BaseEffectBrush': BaseEffectBrush,
    'BaseEntityInputs': BaseEntityInputs,
    'BaseEntityOutputs': BaseEntityOutputs,
    'BaseFadeProp': BaseFadeProp,
    'BaseLight': BaseLight,
    'BaseLightFalloff': BaseLightFalloff,
    'BasePaintType': BasePaintType,
    'CombineScanner': CombineScanner,
    'ControlEnables': ControlEnables,
    'DamageFilter': DamageFilter,
    'DamageType': DamageType,
    'DetailPropBase': DetailPropBase,
    'EnableDisable': EnableDisable,
    'FadeDistance': FadeDistance,
    'GrenadeUser': GrenadeUser,
    'KeyFrame': KeyFrame,
    'LinkedPortalDoor': LinkedPortalDoor,
    'MasterEnt': MasterEnt,
    'Mover': Mover,
    'Node': Node,
    'Origin': Origin,
    'PaintableProp': PaintableProp,
    'PortalBase': PortalBase,
    'Reflection': Reflection,
    'RenderFields': RenderFields,
    'ResponseContext': ResponseContext,
    'SRCIndicator': SRCIndicator,
    'SRCModel': SRCModel,
    'SetSkin': SetSkin,
    'StaticTargetName': StaticTargetName,
    'SystemLevelChoice': SystemLevelChoice,
    'TeamNum': TeamNum,
    'Toggle': Toggle,
    'ToggleDraw': ToggleDraw,
    '_Breakable': _Breakable,
    'comp_entity_finder': comp_entity_finder,
    'comp_entity_mover': comp_entity_mover,
    'comp_player_input_helper': comp_player_input_helper,
    'comp_propcombine_volume': comp_propcombine_volume,
    'comp_vactube_object': comp_vactube_object,
    'env_cubemap': env_cubemap,
    'func_detail': func_detail,
    'func_detail_blocker': func_detail_blocker,
    'func_fish_pool': func_fish_pool,
    'func_instance_io_proxy': func_instance_io_proxy,
    'func_instance_origin': func_instance_origin,
    'func_instance_parms': func_instance_parms,
    'func_ladder': func_ladder,
    'func_viscluster': func_viscluster,
    'hammer_notes': hammer_notes,
    'info_intermission': info_intermission,
    'info_mass_center': info_mass_center,
    'info_no_dynamic_shadow': info_no_dynamic_shadow,
    'info_overlay_transition': info_overlay_transition,
    'parallax_obb': parallax_obb,
    'BaseClusteredDynLight': BaseClusteredDynLight,
    'BaseDustParticleSpawner': BaseDustParticleSpawner,
    'BaseEntityIO': BaseEntityIO,
    'BreakableProp': BreakableProp,
    'Button': Button,
    'HintNode': HintNode,
    'RopeKeyFrame': RopeKeyFrame,
    'SetModel': SetModel,
    'comp_adv_output': comp_adv_output,
    'comp_case': comp_case,
    'comp_kv_setter': comp_kv_setter,
    'comp_pack': comp_pack,
    'comp_pack_rename': comp_pack_rename,
    'comp_pack_replace_soundscript': comp_pack_replace_soundscript,
    'comp_precache_model': comp_precache_model,
    'comp_precache_sound': comp_precache_sound,
    'comp_prop_cable': comp_prop_cable,
    'comp_prop_rope': comp_prop_rope,
    'comp_prop_rope_bunting': comp_prop_rope_bunting,
    'comp_propcombine_set': comp_propcombine_set,
    'comp_relay': comp_relay,
    'comp_scriptvar_setter': comp_scriptvar_setter,
    'comp_vactube_junction': comp_vactube_junction,
    'comp_vactube_sensor': comp_vactube_sensor,
    'comp_vactube_spline': comp_vactube_spline,
    'env_bubbles': env_bubbles,
    'env_embers': env_embers,
    'func_instance': func_instance,
    'func_precipitation': func_precipitation,
    'func_smokevolume': func_smokevolume,
    'info_lighting': info_lighting,
    'info_node': info_node,
    'info_node_air': info_node_air,
    'prop_detail': prop_detail,
    'prop_detail_sprite': prop_detail_sprite,
    'prop_static': prop_static,
    'BaseEntity': BaseEntity,
    'BaseEntityBrush': BaseEntityBrush,
    'BaseEntityPoint': BaseEntityPoint,
    'func_dustcloud': func_dustcloud,
    'func_dustmotes': func_dustmotes,
    'BaseActBusy': BaseActBusy,
    'BaseBeam': BaseBeam,
    'BaseEntityAnimating': BaseEntityAnimating,
    'BaseEntityPhysics': BaseEntityPhysics,
    'BaseEntityVisBrush': BaseEntityVisBrush,
    'BaseNPCMaker': BaseNPCMaker,
    'BasePointLight': BasePointLight,
    'BaseSpotLight': BaseSpotLight,
    'BaseTank': BaseTank,
    'CombineBallSpawners': CombineBallSpawners,
    'FollowGoal': FollowGoal,
    'ForceController': ForceController,
    'LeadGoalBase': LeadGoalBase,
    'NavCost': NavCost,
    'TriggerOnce': TriggerOnce,
    'TwoObjectPhysics': TwoObjectPhysics,
    'Weapon': Weapon,
    'ai_ally_manager': ai_ally_manager,
    'ai_battle_line': ai_battle_line,
    'ai_changehintgroup': ai_changehintgroup,
    'ai_changetarget': ai_changetarget,
    'ai_citizen_response_system': ai_citizen_response_system,
    'ai_goal_assault': ai_goal_assault,
    'ai_goal_fightfromcover': ai_goal_fightfromcover,
    'ai_goal_operator': ai_goal_operator,
    'ai_goal_police': ai_goal_police,
    'ai_goal_standoff': ai_goal_standoff,
    'ai_npc_eventresponsesystem': ai_npc_eventresponsesystem,
    'ai_relationship': ai_relationship,
    'ai_script_conditions': ai_script_conditions,
    'ai_sound': ai_sound,
    'ai_speechfilter': ai_speechfilter,
    'aiscripted_schedule': aiscripted_schedule,
    'ambient_generic': ambient_generic,
    'assault_assaultpoint': assault_assaultpoint,
    'assault_rallypoint': assault_rallypoint,
    'beam_spotlight': beam_spotlight,
    'color_correction': color_correction,
    'color_correction_volume': color_correction_volume,
    'combine_mine': combine_mine,
    'commentary_auto': commentary_auto,
    'comp_choreo_sceneset': comp_choreo_sceneset,
    'comp_flicker': comp_flicker,
    'comp_vactube_end': comp_vactube_end,
    'comp_vactube_start': comp_vactube_start,
    'env_alyxemp': env_alyxemp,
    'env_ambient_light': env_ambient_light,
    'env_ar2explosion': env_ar2explosion,
    'env_blood': env_blood,
    'env_cascade_light': env_cascade_light,
    'env_citadel_energy_core': env_citadel_energy_core,
    'env_credits': env_credits,
    'env_detail_controller': env_detail_controller,
    'env_dof_controller': env_dof_controller,
    'env_dustpuff': env_dustpuff,
    'env_entity_dissolver': env_entity_dissolver,
    'env_entity_igniter': env_entity_igniter,
    'env_entity_maker': env_entity_maker,
    'env_explosion': env_explosion,
    'env_fade': env_fade,
    'env_fire': env_fire,
    'env_firesensor': env_firesensor,
    'env_firesource': env_firesource,
    'env_flare': env_flare,
    'env_fog_controller': env_fog_controller,
    'env_funnel': env_funnel,
    'env_global': env_global,
    'env_gunfire': env_gunfire,
    'env_hudhint': env_hudhint,
    'env_instructor_hint': env_instructor_hint,
    'env_lightglow': env_lightglow,
    'env_lightrail_endpoint': env_lightrail_endpoint,
    'env_message': env_message,
    'env_microphone': env_microphone,
    'env_movieexplosion': env_movieexplosion,
    'env_muzzleflash': env_muzzleflash,
    'env_particle_performance_monitor': env_particle_performance_monitor,
    'env_particlelight': env_particlelight,
    'env_particlescript': env_particlescript,
    'env_physexplosion': env_physexplosion,
    'env_physimpact': env_physimpact,
    'env_player_surface_trigger': env_player_surface_trigger,
    'env_player_viewfinder': env_player_viewfinder,
    'env_portal_credits': env_portal_credits,
    'env_projectedtexture': env_projectedtexture,
    'env_rockettrail': env_rockettrail,
    'env_rotorwash_emitter': env_rotorwash_emitter,
    'env_screeneffect': env_screeneffect,
    'env_screenoverlay': env_screenoverlay,
    'env_shake': env_shake,
    'env_smokestack': env_smokestack,
    'env_smoketrail': env_smoketrail,
    'env_soundscape': env_soundscape,
    'env_soundscape_proxy': env_soundscape_proxy,
    'env_spark': env_spark,
    'env_speaker': env_speaker,
    'env_splash': env_splash,
    'env_sporeexplosion': env_sporeexplosion,
    'env_sprite': env_sprite,
    'env_spritetrail': env_spritetrail,
    'env_starfield': env_starfield,
    'env_steam': env_steam,
    'env_sun': env_sun,
    'env_texturetoggle': env_texturetoggle,
    'env_tilt': env_tilt,
    'env_tonemap_controller': env_tonemap_controller,
    'env_viewpunch': env_viewpunch,
    'env_wind': env_wind,
    'env_zoom': env_zoom,
    'filter_base': filter_base,
    'fog_volume': fog_volume,
    'func_areaportal': func_areaportal,
    'func_areaportalwindow': func_areaportalwindow,
    'func_clip_vphysics': func_clip_vphysics,
    'func_ladderendpoint': func_ladderendpoint,
    'func_nav_avoidance_obstacle': func_nav_avoidance_obstacle,
    'func_nav_blocker': func_nav_blocker,
    'func_noportal_volume': func_noportal_volume,
    'func_occluder': func_occluder,
    'func_portal_bumper': func_portal_bumper,
    'func_portal_detector': func_portal_detector,
    'func_portal_orientation': func_portal_orientation,
    'func_precipitation_blocker': func_precipitation_blocker,
    'func_proprrespawnzone': func_proprrespawnzone,
    'func_useableladder': func_useableladder,
    'func_vehicleclip': func_vehicleclip,
    'func_wall': func_wall,
    'game_end': game_end,
    'game_gib_manager': game_gib_manager,
    'game_globalvars': game_globalvars,
    'game_player_equip': game_player_equip,
    'game_player_team': game_player_team,
    'game_ragdoll_manager': game_ragdoll_manager,
    'game_score': game_score,
    'game_text': game_text,
    'game_ui': game_ui,
    'game_weapon_manager': game_weapon_manager,
    'game_zone_player': game_zone_player,
    'gibshooterbase': gibshooterbase,
    'hammer_updateignorelist': hammer_updateignorelist,
    'info_apc_missile_hint': info_apc_missile_hint,
    'info_camera_link': info_camera_link,
    'info_constraint_anchor': info_constraint_anchor,
    'info_coop_spawn': info_coop_spawn,
    'info_darknessmode_lightsource': info_darknessmode_lightsource,
    'info_game_event_proxy': info_game_event_proxy,
    'info_hint': info_hint,
    'info_ladder_dismount': info_ladder_dismount,
    'info_landmark': info_landmark,
    'info_landmark_entry': info_landmark_entry,
    'info_landmark_exit': info_landmark_exit,
    'info_lighting_relative': info_lighting_relative,
    'info_node_air_hint': info_node_air_hint,
    'info_node_climb': info_node_climb,
    'info_node_hint': info_node_hint,
    'info_node_link': info_node_link,
    'info_node_link_controller': info_node_link_controller,
    'info_npc_spawn_destination': info_npc_spawn_destination,
    'info_null': info_null,
    'info_overlay': info_overlay,
    'info_paint_sprayer': info_paint_sprayer,
    'info_particle_system': info_particle_system,
    'info_placement_helper': info_placement_helper,
    'info_player_deathmatch': info_player_deathmatch,
    'info_player_ping_detector': info_player_ping_detector,
    'info_player_start': info_player_start,
    'info_playtest_manager': info_playtest_manager,
    'info_portal_gamerules': info_portal_gamerules,
    'info_projecteddecal': info_projecteddecal,
    'info_radar_target': info_radar_target,
    'info_radial_link_controller': info_radial_link_controller,
    'info_snipertarget': info_snipertarget,
    'info_target': info_target,
    'info_target_gunshipcrash': info_target_gunshipcrash,
    'info_target_helicopter_crash': info_target_helicopter_crash,
    'info_target_instructor_hint': info_target_instructor_hint,
    'info_target_personality_sphere': info_target_personality_sphere,
    'info_target_vehicle_transition': info_target_vehicle_transition,
    'info_teleport_destination': info_teleport_destination,
    'info_teleporter_countdown': info_teleporter_countdown,
    'infodecal': infodecal,
    'keyframe_rope': keyframe_rope,
    'keyframe_track': keyframe_track,
    'light_directional': light_directional,
    'light_dynamic': light_dynamic,
    'light_environment': light_environment,
    'linked_portal_door': linked_portal_door,
    'logic_achievement': logic_achievement,
    'logic_active_autosave': logic_active_autosave,
    'logic_auto': logic_auto,
    'logic_autosave': logic_autosave,
    'logic_branch': logic_branch,
    'logic_branch_listener': logic_branch_listener,
    'logic_case': logic_case,
    'logic_choreographed_scene': logic_choreographed_scene,
    'logic_collision_pair': logic_collision_pair,
    'logic_compare': logic_compare,
    'logic_console': logic_console,
    'logic_context_accessor': logic_context_accessor,
    'logic_convar': logic_convar,
    'logic_coop_manager': logic_coop_manager,
    'logic_datadesc_accessor': logic_datadesc_accessor,
    'logic_entity_position': logic_entity_position,
    'logic_eventlistener': logic_eventlistener,
    'logic_eventlistener_itemequip': logic_eventlistener_itemequip,
    'logic_format': logic_format,
    'logic_gate': logic_gate,
    'logic_keyfield': logic_keyfield,
    'logic_lineto': logic_lineto,
    'logic_measure_movement': logic_measure_movement,
    'logic_modelinfo': logic_modelinfo,
    'logic_multicompare': logic_multicompare,
    'logic_navigation': logic_navigation,
    'logic_player_slowtime': logic_player_slowtime,
    'logic_playerproxy': logic_playerproxy,
    'logic_playmovie': logic_playmovie,
    'logic_random_outputs': logic_random_outputs,
    'logic_register_activator': logic_register_activator,
    'logic_relay': logic_relay,
    'logic_relay_queue': logic_relay_queue,
    'logic_scene_list_manager': logic_scene_list_manager,
    'logic_script': logic_script,
    'logic_sequence': logic_sequence,
    'logic_timer': logic_timer,
    'logic_timescale': logic_timescale,
    'material_modify_control': material_modify_control,
    'math_bits': math_bits,
    'math_clamp': math_clamp,
    'math_colorblend': math_colorblend,
    'math_counter': math_counter,
    'math_generate': math_generate,
    'math_lightpattern': math_lightpattern,
    'math_mod': math_mod,
    'math_remap': math_remap,
    'math_vector': math_vector,
    'move_keyframed': move_keyframed,
    'move_rope': move_rope,
    'move_track': move_track,
    'npc_heli_avoidbox': npc_heli_avoidbox,
    'npc_heli_avoidsphere': npc_heli_avoidsphere,
    'npc_heli_nobomb': npc_heli_nobomb,
    'obb_volumefog': obb_volumefog,
    'paint_sphere': paint_sphere,
    'panorama_screen': panorama_screen,
    'path_corner': path_corner,
    'path_corner_crash': path_corner_crash,
    'path_track': path_track,
    'path_vphysics': path_vphysics,
    'phys_constraintsystem': phys_constraintsystem,
    'phys_convert': phys_convert,
    'phys_keepupright': phys_keepupright,
    'phys_motor': phys_motor,
    'phys_ragdollmagnet': phys_ragdollmagnet,
    'phys_spring': phys_spring,
    'player_loadsaved': player_loadsaved,
    'player_speedmod': player_speedmod,
    'player_weaponstrip': player_weaponstrip,
    'point_anglesensor': point_anglesensor,
    'point_angularvelocitysensor': point_angularvelocitysensor,
    'point_antlion_repellant': point_antlion_repellant,
    'point_apc_controller': point_apc_controller,
    'point_bonusmaps_accessor': point_bonusmaps_accessor,
    'point_broadcastclientcommand': point_broadcastclientcommand,
    'point_bugbait': point_bugbait,
    'point_camera': point_camera,
    'point_changelevel': point_changelevel,
    'point_clientcommand': point_clientcommand,
    'point_devshot_camera': point_devshot_camera,
    'point_enable_motion_fixup': point_enable_motion_fixup,
    'point_entity_finder': point_entity_finder,
    'point_flesh_effect_target': point_flesh_effect_target,
    'point_futbol_shooter': point_futbol_shooter,
    'point_gamestats_counter': point_gamestats_counter,
    'point_hiding_spot': point_hiding_spot,
    'point_hurt': point_hurt,
    'point_laser_target': point_laser_target,
    'point_message': point_message,
    'point_paint_sensor': point_paint_sensor,
    'point_playermoveconstraint': point_playermoveconstraint,
    'point_posecontroller': point_posecontroller,
    'point_proximity_sensor': point_proximity_sensor,
    'point_push': point_push,
    'point_servercommand': point_servercommand,
    'point_spotlight': point_spotlight,
    'point_survey': point_survey,
    'point_teleport': point_teleport,
    'point_template': point_template,
    'point_tesla': point_tesla,
    'point_velocitysensor': point_velocitysensor,
    'point_viewcontrol': point_viewcontrol,
    'point_viewcontrol_multiplayer': point_viewcontrol_multiplayer,
    'point_viewproxy': point_viewproxy,
    'point_worldtext': point_worldtext,
    'portalmp_gamerules': portalmp_gamerules,
    'postprocess_controller': postprocess_controller,
    'projected_wall_entity': projected_wall_entity,
    'prop_glass_futbol_socket': prop_glass_futbol_socket,
    'prop_glass_futbol_spawner': prop_glass_futbol_spawner,
    'prop_indicator_panel': prop_indicator_panel,
    'prop_portal': prop_portal,
    'prop_testchamber_sign': prop_testchamber_sign,
    'prop_tic_tac_toe_panel': prop_tic_tac_toe_panel,
    'rocket_turret_projectile': rocket_turret_projectile,
    'script_intro': script_intro,
    'scripted_sentence': scripted_sentence,
    'scripted_sequence': scripted_sequence,
    'scripted_target': scripted_target,
    'shadow_control': shadow_control,
    'sky_camera': sky_camera,
    'skybox_swapper': skybox_swapper,
    'spark_shower': spark_shower,
    'sunlight_shadow_control': sunlight_shadow_control,
    'tanktrain_ai': tanktrain_ai,
    'tanktrain_aitarget': tanktrain_aitarget,
    'target_changegravity': target_changegravity,
    'test_sidelist': test_sidelist,
    'test_traceline': test_traceline,
    'trigger_brush': trigger_brush,
    'vgui_level_placard_display': vgui_level_placard_display,
    'vgui_movie_display': vgui_movie_display,
    'vgui_mp_lobby_display': vgui_mp_lobby_display,
    'vgui_neurotoxin_countdown': vgui_neurotoxin_countdown,
    'vgui_screen': vgui_screen,
    'vgui_slideshow_display': vgui_slideshow_display,
    'vgui_world_text_panel': vgui_world_text_panel,
    'water_lod_control': water_lod_control,
    'worldspawn': worldspawn,
    'BaseLogicalNPC': BaseLogicalNPC,
    'BasePedButton': BasePedButton,
    'BasePortButton': BasePortButton,
    'BaseProjector': BaseProjector,
    'BasePropPhysics': BasePropPhysics,
    'BaseTrain': BaseTrain,
    'BreakableBrush': BreakableBrush,
    'Door': Door,
    'Item': Item,
    'Trigger': Trigger,
    'ai_goal_actbusy': ai_goal_actbusy,
    'ai_goal_actbusy_queue': ai_goal_actbusy_queue,
    'ai_goal_follow': ai_goal_follow,
    'ai_goal_injured_follow': ai_goal_injured_follow,
    'ai_goal_lead': ai_goal_lead,
    'ai_goal_lead_weapon': ai_goal_lead_weapon,
    'bounce_bomb': bounce_bomb,
    'combine_bouncemine': combine_bouncemine,
    'comp_numeric_transition': comp_numeric_transition,
    'comp_prop_cable_dynamic': comp_prop_cable_dynamic,
    'comp_prop_rope_dynamic': comp_prop_rope_dynamic,
    'comp_sequential_call': comp_sequential_call,
    'cycler': cycler,
    'ent_hover_turret_tether': ent_hover_turret_tether,
    'env_beam': env_beam,
    'env_effectscript': env_effectscript,
    'env_glow': env_glow,
    'env_headcrabcanister': env_headcrabcanister,
    'env_laser': env_laser,
    'env_portal_laser': env_portal_laser,
    'env_portal_path_track': env_portal_path_track,
    'env_rotorshooter': env_rotorshooter,
    'env_shooter': env_shooter,
    'env_soundscape_triggerable': env_soundscape_triggerable,
    'env_sprite_clientside': env_sprite_clientside,
    'env_sprite_oriented': env_sprite_oriented,
    'filter_activator_class': filter_activator_class,
    'filter_activator_context': filter_activator_context,
    'filter_activator_involume': filter_activator_involume,
    'filter_activator_keyfield': filter_activator_keyfield,
    'filter_activator_mass_greater': filter_activator_mass_greater,
    'filter_activator_model': filter_activator_model,
    'filter_activator_name': filter_activator_name,
    'filter_activator_surfacedata': filter_activator_surfacedata,
    'filter_activator_team': filter_activator_team,
    'filter_combineball_type': filter_combineball_type,
    'filter_damage_type': filter_damage_type,
    'filter_enemy': filter_enemy,
    'filter_multi': filter_multi,
    'filter_paint_power': filter_paint_power,
    'filter_player_held': filter_player_held,
    'filter_velocity': filter_velocity,
    'func_brush': func_brush,
    'func_button': func_button,
    'func_combine_ball_spawner': func_combine_ball_spawner,
    'func_conveyor': func_conveyor,
    'func_guntarget': func_guntarget,
    'func_healthcharger': func_healthcharger,
    'func_illusionary': func_illusionary,
    'func_lod': func_lod,
    'func_movelinear': func_movelinear,
    'func_nav_avoid': func_nav_avoid,
    'func_nav_prefer': func_nav_prefer,
    'func_platrot': func_platrot,
    'func_portalled': func_portalled,
    'func_recharge': func_recharge,
    'func_rot_button': func_rot_button,
    'func_rotating': func_rotating,
    'func_tank': func_tank,
    'func_tank_combine_cannon': func_tank_combine_cannon,
    'func_tankairboatgun': func_tankairboatgun,
    'func_tankapcrocket': func_tankapcrocket,
    'func_tanklaser': func_tanklaser,
    'func_tankmortar': func_tankmortar,
    'func_tankphyscannister': func_tankphyscannister,
    'func_tankpulselaser': func_tankpulselaser,
    'func_tankrocket': func_tankrocket,
    'func_trackchange': func_trackchange,
    'func_train': func_train,
    'func_traincontrols': func_traincontrols,
    'func_wall_toggle': func_wall_toggle,
    'func_weight_button': func_weight_button,
    'gibshooter': gibshooter,
    'grenade_helicopter': grenade_helicopter,
    'hot_potato_catcher': hot_potato_catcher,
    'hot_potato_socket': hot_potato_socket,
    'hot_potato_spawner': hot_potato_spawner,
    'item_ammo_crate': item_ammo_crate,
    'item_healthcharger': item_healthcharger,
    'item_nugget': item_nugget,
    'item_paint_power_pickup': item_paint_power_pickup,
    'item_suitcharger': item_suitcharger,
    'light': light,
    'light_rt': light_rt,
    'light_rt_spot': light_rt_spot,
    'light_spot': light_spot,
    'logic_measure_direction': logic_measure_direction,
    'math_counter_advanced': math_counter_advanced,
    'momentary_rot_button': momentary_rot_button,
    'npc_antlion_grub': npc_antlion_grub,
    'npc_antlion_template_maker': npc_antlion_template_maker,
    'npc_maker': npc_maker,
    'npc_template_maker': npc_template_maker,
    'npc_tripmine': npc_tripmine,
    'npc_turret_ceiling': npc_turret_ceiling,
    'phys_ballsocket': phys_ballsocket,
    'phys_constraint': phys_constraint,
    'phys_hinge': phys_hinge,
    'phys_lengthconstraint': phys_lengthconstraint,
    'phys_magnet': phys_magnet,
    'phys_pulleyconstraint': phys_pulleyconstraint,
    'phys_ragdollconstraint': phys_ragdollconstraint,
    'phys_slideconstraint': phys_slideconstraint,
    'phys_thruster': phys_thruster,
    'phys_torque': phys_torque,
    'physics_cannister': physics_cannister,
    'point_combine_ball_launcher': point_combine_ball_launcher,
    'point_commentary_node': point_commentary_node,
    'point_energy_ball_launcher': point_energy_ball_launcher,
    'portal_race_checkpoint': portal_race_checkpoint,
    'prop_coreball': prop_coreball,
    'prop_door_rotating': prop_door_rotating,
    'prop_dropship_container': prop_dropship_container,
    'prop_dynamic_base': prop_dynamic_base,
    'prop_exploding_futbol': prop_exploding_futbol,
    'prop_hallucination': prop_hallucination,
    'prop_laser_catcher': prop_laser_catcher,
    'prop_laser_relay': prop_laser_relay,
    'prop_linked_portal_door': prop_linked_portal_door,
    'prop_mirror': prop_mirror,
    'prop_monster_box': prop_monster_box,
    'prop_paint_bomb': prop_paint_bomb,
    'prop_portal_stats_display': prop_portal_stats_display,
    'prop_ragdoll': prop_ragdoll,
    'prop_rocket_tripwire': prop_rocket_tripwire,
    'prop_scalable': prop_scalable,
    'prop_telescopic_arm': prop_telescopic_arm,
    'prop_testchamber_door': prop_testchamber_door,
    'prop_thumper': prop_thumper,
    'scripted_scene': scripted_scene,
    'simple_physics_brush': simple_physics_brush,
    'trigger_autosave': trigger_autosave,
    'trigger_once': trigger_once,
    'weapon_357': weapon_357,
    'weapon_alyxgun': weapon_alyxgun,
    'weapon_annabelle': weapon_annabelle,
    'weapon_ar2': weapon_ar2,
    'weapon_bugbait': weapon_bugbait,
    'weapon_citizenpackage': weapon_citizenpackage,
    'weapon_citizensuitcase': weapon_citizensuitcase,
    'weapon_crossbow': weapon_crossbow,
    'weapon_crowbar': weapon_crowbar,
    'weapon_cubemap': weapon_cubemap,
    'weapon_frag': weapon_frag,
    'weapon_paintgun': weapon_paintgun,
    'weapon_physcannon': weapon_physcannon,
    'weapon_physgun': weapon_physgun,
    'weapon_pistol': weapon_pistol,
    'weapon_portalgun': weapon_portalgun,
    'weapon_rpg': weapon_rpg,
    'weapon_shotgun': weapon_shotgun,
    'weapon_smg1': weapon_smg1,
    'weapon_stunstick': weapon_stunstick,
    'BaseNPC': BaseNPC,
    'BaseVehicle': BaseVehicle,
    'comp_trigger_p2_goo': comp_trigger_p2_goo,
    'func_breakable': func_breakable,
    'func_breakable_surf': func_breakable_surf,
    'func_bulletshield': func_bulletshield,
    'func_door': func_door,
    'func_door_rotating': func_door_rotating,
    'func_lookdoor': func_lookdoor,
    'func_monitor': func_monitor,
    'func_physbox': func_physbox,
    'func_placement_clip': func_placement_clip,
    'func_reflective_glass': func_reflective_glass,
    'func_tanktrain': func_tanktrain,
    'func_trackautochange': func_trackautochange,
    'func_tracktrain': func_tracktrain,
    'func_water_analog': func_water_analog,
    'hunter_flechette': hunter_flechette,
    'item_ammo_357': item_ammo_357,
    'item_ammo_357_large': item_ammo_357_large,
    'item_ammo_ar2': item_ammo_ar2,
    'item_ammo_ar2_altfire': item_ammo_ar2_altfire,
    'item_ammo_ar2_large': item_ammo_ar2_large,
    'item_ammo_crossbow': item_ammo_crossbow,
    'item_ammo_pistol': item_ammo_pistol,
    'item_ammo_pistol_large': item_ammo_pistol_large,
    'item_ammo_smg1': item_ammo_smg1,
    'item_ammo_smg1_grenade': item_ammo_smg1_grenade,
    'item_ammo_smg1_large': item_ammo_smg1_large,
    'item_ar2_grenade': item_ar2_grenade,
    'item_battery': item_battery,
    'item_boots': item_boots,
    'item_box_buckshot': item_box_buckshot,
    'item_dynamic_resupply': item_dynamic_resupply,
    'item_grubnugget': item_grubnugget,
    'item_healthkit': item_healthkit,
    'item_healthvial': item_healthvial,
    'item_item_crate': item_item_crate,
    'item_rpg_round': item_rpg_round,
    'item_suit': item_suit,
    'npc_bullseye': npc_bullseye,
    'npc_enemyfinder': npc_enemyfinder,
    'npc_hunter_maker': npc_hunter_maker,
    'npc_launcher': npc_launcher,
    'prop_button': prop_button,
    'prop_dynamic': prop_dynamic,
    'prop_dynamic_ornament': prop_dynamic_ornament,
    'prop_dynamic_override': prop_dynamic_override,
    'prop_floor_ball_button': prop_floor_ball_button,
    'prop_floor_button': prop_floor_button,
    'prop_floor_cube_button': prop_floor_cube_button,
    'prop_glados_core': prop_glados_core,
    'prop_glass_futbol': prop_glass_futbol,
    'prop_physics': prop_physics,
    'prop_physics_ragdoll': prop_physics_ragdoll,
    'prop_sphere': prop_sphere,
    'prop_tractor_beam': prop_tractor_beam,
    'prop_under_button': prop_under_button,
    'prop_under_floor_button': prop_under_floor_button,
    'prop_wall_projector': prop_wall_projector,
    'prop_weighted_cube': prop_weighted_cube,
    'simple_physics_prop': simple_physics_prop,
    'trigger_catapult': trigger_catapult,
    'trigger_changelevel': trigger_changelevel,
    'trigger_gravity': trigger_gravity,
    'trigger_hurt': trigger_hurt,
    'trigger_impact': trigger_impact,
    'trigger_jumppad': trigger_jumppad,
    'trigger_look': trigger_look,
    'trigger_multiple': trigger_multiple,
    'trigger_paint_cleanser': trigger_paint_cleanser,
    'trigger_physics_trap': trigger_physics_trap,
    'trigger_ping_detector': trigger_ping_detector,
    'trigger_playermovement': trigger_playermovement,
    'trigger_playerteam': trigger_playerteam,
    'trigger_portal_cleanser': trigger_portal_cleanser,
    'trigger_proximity': trigger_proximity,
    'trigger_push': trigger_push,
    'trigger_remove': trigger_remove,
    'trigger_rpgfire': trigger_rpgfire,
    'trigger_serverragdoll': trigger_serverragdoll,
    'trigger_setspeed': trigger_setspeed,
    'trigger_softbarrier': trigger_softbarrier,
    'trigger_soundoperator': trigger_soundoperator,
    'trigger_soundscape': trigger_soundscape,
    'trigger_teleport': trigger_teleport,
    'trigger_teleport_relative': trigger_teleport_relative,
    'trigger_togglesave': trigger_togglesave,
    'trigger_tonemap': trigger_tonemap,
    'trigger_transition': trigger_transition,
    'trigger_userinput': trigger_userinput,
    'trigger_vphysics_motion': trigger_vphysics_motion,
    'trigger_waterydeath': trigger_waterydeath,
    'trigger_weapon_dissolve': trigger_weapon_dissolve,
    'trigger_weapon_strip': trigger_weapon_strip,
    'trigger_wind': trigger_wind,
    'weapon_striderbuster': weapon_striderbuster,
    'BaseDriveableVehicle': BaseDriveableVehicle,
    'BaseHeadcrab': BaseHeadcrab,
    'BaseHelicopter': BaseHelicopter,
    'PlayerCompanion': PlayerCompanion,
    'RappelNPC': RappelNPC,
    'TalkNPC': TalkNPC,
    'VehicleDriverNPC': VehicleDriverNPC,
    'comp_trigger_coop': comp_trigger_coop,
    'cycler_actor': cycler_actor,
    'func_physbox_multiplayer': func_physbox_multiplayer,
    'func_pushable': func_pushable,
    'func_touch': func_touch,
    'func_water': func_water,
    'generic_actor': generic_actor,
    'hot_potato': hot_potato,
    'item_box_lrounds': item_box_lrounds,
    'item_box_mrounds': item_box_mrounds,
    'item_box_srounds': item_box_srounds,
    'item_large_box_lrounds': item_large_box_lrounds,
    'item_large_box_mrounds': item_large_box_mrounds,
    'item_large_box_srounds': item_large_box_srounds,
    'monster_generic': monster_generic,
    'npc_advisor': npc_advisor,
    'npc_antlion': npc_antlion,
    'npc_antlionguard': npc_antlionguard,
    'npc_barnacle': npc_barnacle,
    'npc_clawscanner': npc_clawscanner,
    'npc_combine_camera': npc_combine_camera,
    'npc_combine_cannon': npc_combine_cannon,
    'npc_crow': npc_crow,
    'npc_cscanner': npc_cscanner,
    'npc_dog': npc_dog,
    'npc_enemyfinder_combinecannon': npc_enemyfinder_combinecannon,
    'npc_fastzombie': npc_fastzombie,
    'npc_fastzombie_torso': npc_fastzombie_torso,
    'npc_fisherman': npc_fisherman,
    'npc_furniture': npc_furniture,
    'npc_grenade_frag': npc_grenade_frag,
    'npc_heardanger': npc_heardanger,
    'npc_hunter': npc_hunter,
    'npc_ichthyosaur': npc_ichthyosaur,
    'npc_manhack': npc_manhack,
    'npc_missiledefense': npc_missiledefense,
    'npc_pigeon': npc_pigeon,
    'npc_poisonzombie': npc_poisonzombie,
    'npc_portal_turret_floor': npc_portal_turret_floor,
    'npc_puppet': npc_puppet,
    'npc_rocket_turret': npc_rocket_turret,
    'npc_rollermine': npc_rollermine,
    'npc_seagull': npc_seagull,
    'npc_security_camera': npc_security_camera,
    'npc_sniper': npc_sniper,
    'npc_spotlight': npc_spotlight,
    'npc_stalker': npc_stalker,
    'npc_strider': npc_strider,
    'npc_turret_floor': npc_turret_floor,
    'npc_turret_ground': npc_turret_ground,
    'npc_wheatley_boss': npc_wheatley_boss,
    'npc_zombie': npc_zombie,
    'npc_zombie_torso': npc_zombie_torso,
    'npc_zombine': npc_zombine,
    'prop_dynamic_glow': prop_dynamic_glow,
    'prop_physics_multiplayer': prop_physics_multiplayer,
    'prop_physics_override': prop_physics_override,
    'prop_physics_paintable': prop_physics_paintable,
    'prop_physics_respawnable': prop_physics_respawnable,
    'prop_stickybomb': prop_stickybomb,
    'prop_vehicle': prop_vehicle,
    'simple_bot': simple_bot,
    'trigger_hierarchy': trigger_hierarchy,
    'npc_alyx': npc_alyx,
    'npc_apcdriver': npc_apcdriver,
    'npc_barney': npc_barney,
    'npc_breen': npc_breen,
    'npc_citizen': npc_citizen,
    'npc_combine_s': npc_combine_s,
    'npc_combinedropship': npc_combinedropship,
    'npc_combinegunship': npc_combinegunship,
    'npc_cranedriver': npc_cranedriver,
    'npc_eli': npc_eli,
    'npc_gman': npc_gman,
    'npc_headcrab': npc_headcrab,
    'npc_headcrab_fast': npc_headcrab_fast,
    'npc_headcrab_poison': npc_headcrab_poison,
    'npc_helicopter': npc_helicopter,
    'npc_kleiner': npc_kleiner,
    'npc_magnusson': npc_magnusson,
    'npc_metropolice': npc_metropolice,
    'npc_monk': npc_monk,
    'npc_mossman': npc_mossman,
    'npc_personality_core': npc_personality_core,
    'npc_vehicledriver': npc_vehicledriver,
    'npc_vortigaunt': npc_vortigaunt,
    'prop_vehicle_airboat': prop_vehicle_airboat,
    'prop_vehicle_apc': prop_vehicle_apc,
    'prop_vehicle_cannon': prop_vehicle_cannon,
    'prop_vehicle_choreo_generic': prop_vehicle_choreo_generic,
    'prop_vehicle_crane': prop_vehicle_crane,
    'prop_vehicle_driveable': prop_vehicle_driveable,
    'prop_vehicle_jeep': prop_vehicle_jeep,
    'prop_vehicle_prisoner_pod': prop_vehicle_prisoner_pod,
    'vehicle_viewcontroller': vehicle_viewcontroller,
    'npc_headcrab_black': npc_headcrab_black,
}