from .base_entity_classes import (EnableDisable, Global, Light, Parentname,
                                  keyframe_rope, move_rope, prop_dynamic)


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
        return parse_float_vector(self._raw_data.get('origin', "None"))


class Targetname(Base):

    @property
    def targetname(self):
        return self._raw_data.get('targetname', "None")


class ResponseContext(Base):

    @property
    def ResponseContext(self):
        return self._raw_data.get('responsecontext', "")


class worldbase(Base):

    @property
    def message(self):
        return self._raw_data.get('message', "None")

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


class worldspawn(worldbase, ResponseContext, Targetname):
    pass


class func_window_hint(Origin):
    @property
    def model(self):
        return self._raw_data.get('model', None)

    @property
    def halfheight(self):
        return self._raw_data.get('halfheight', None)

    @property
    def halfwidth(self):
        return self._raw_data.get('halfwidth', None)

    @property
    def right(self):
        return parse_float_vector(self._raw_data.get('right', '0 0 0'))


class TriggerOnce(Parentname, Origin, Global, EnableDisable, Targetname):
    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)


class Trigger(TriggerOnce):
    pass


class trigger_indoor_area(Trigger):
    pass


class trigger_capture_point(Trigger):
    pass


class trigger_out_of_bounds(Trigger):
    pass


class trigger_soundscape(Trigger):
    pass


class light(Targetname, Light):
    icon_sprite = "editor/light.vmt"
    @property
    def origin(self):
        return parse_float_vector(self._raw_data.get('origin',"0 0 0"))

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
        return parse_float_vector(self._raw_data.get('origin',"0 0 0"))

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


class light_spot(Targetname, Light, Angles):
    @property
    def origin(self):
        return parse_float_vector(self._raw_data.get('origin',"0 0 0"))

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
        return parse_source_value(self._raw_data.get('origin', "[0 0 0]"))

    @property
    def target(self):
        return self._raw_data.get('target', "None")

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


entity_class_handle = {

    'worldbase': worldbase,
    'worldspawn': worldspawn,
    'func_window_hint': func_window_hint,
    'trigger_indoor_area': trigger_indoor_area,
    'trigger_capture_point': trigger_capture_point,
    'trigger_out_of_bounds': trigger_out_of_bounds,
    'trigger_soundscape': trigger_soundscape,
    'info_particle_system': Base,  # TODO
    'info_node': Base,  # TODO
    'info_target': Base,  # TODO
    'info_hint': Base,  # TODO
    'info_target_clientside': Base,  # TODO
    'info_node_safe_hint': Base,  # TODO
    'info_node_cover_stand': Base,  # TODO
    'info_spawnpoint_dropship_start': Base,  # TODO
    'info_spawnpoint_titan_start': Base,  # TODO
    'info_spawnpoint_droppod_start': Base,  # TODO
    'info_spawnpoint_droppod': Base,  # TODO
    'info_spawnpoint_human_start': Base,  # TODO
    'info_spawnpoint_human': Base,  # TODO
    'info_spawnpoint_titan': Base,  # TODO
    'info_frontline': Base,  # TODO
    'info_hardpoint': Base,  # TODO
    'ambient_generic': Base,  # TODO
    'traverse': Base,  # TODO
    'assault_assaultpoint': Base,  # TODO
    'trigger_hurt': trigger_hurt,
    'light': light,
    'light_environment': light_environment,
    'light_spot': light_spot,
    'light_dynamic': light_dynamic,
    'keyframe_rope': keyframe_rope,
    'move_rope': move_rope,
    'prop_dynamic': prop_dynamic,
}
