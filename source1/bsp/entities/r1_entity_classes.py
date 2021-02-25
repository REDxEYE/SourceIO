from .base_entity_classes import Parentname, Global, EnableDisable, Light, keyframe_rope, \
    move_rope, prop_dynamic


def parse_int_vector(string):
    return [int(val) for val in string.split(' ')]


def parse_float_vector(string):
    return [float(val) for val in string.split(' ')]


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


class Targetname(Base):
    def __init__(self):
        super().__init__()
        self.targetname = None  # Type: target_source

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.targetname = entity_data.get('targetname', None)  # Type: target_source


class ResponseContext(Base):
    def __init__(self):
        super().__init__()
        self.ResponseContext = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.ResponseContext = entity_data.get('responsecontext', None)  # Type: string


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


class worldspawn(Targetname, ResponseContext, worldbase):
    def __init__(self):
        super(Targetname).__init__()
        super(ResponseContext).__init__()
        super(worldbase).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        ResponseContext.from_dict(instance, entity_data)
        worldbase.from_dict(instance, entity_data)


class func_window_hint(Origin):
    def __init__(self):
        super(Origin).__init__()
        self.model = ''
        self.halfheight = 0
        self.halfwidth = 0
        self.right = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Origin.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', None)
        instance.halfheight = int(entity_data.get('halfheight', None))
        instance.halfwidth = int(entity_data.get('halfwidth', None))
        instance.right = parse_float_vector(entity_data.get('right', '0 0 0'))


class TriggerOnce(Parentname, Origin, Global, EnableDisable, Targetname):
    def __init__(self):
        super(Parentname).__init__()
        super(Origin).__init__()
        super(Global).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()
        self.filtername = None  # Type: filterclass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Global.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.filtername = entity_data.get('filtername', None)  # Type: filterclass


class Trigger(TriggerOnce):
    def __init__(self):
        super(TriggerOnce).__init__()
        super(Parentname).__init__()
        super(Origin).__init__()
        super(EnableDisable).__init__()
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TriggerOnce.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)


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
        instance._distance = int(entity_data.get('_distance', 0))  # Type: integer


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
        instance.pitch = float(entity_data.get('pitch', 0))  # Type: float
        instance._light = parse_int_vector(entity_data.get('_light', "255 255 255 200"))  # Type: color255
        instance._ambient = parse_int_vector(entity_data.get('_ambient', "255 255 255 20"))  # Type: color255
        instance._lightHDR = parse_int_vector(entity_data.get('_lighthdr', "-1 -1 -1 1"))  # Type: color255
        instance._lightscaleHDR = float(entity_data.get('_lightscalehdr', 1))  # Type: float
        instance._ambientHDR = parse_int_vector(entity_data.get('_ambienthdr', "-1 -1 -1 1"))  # Type: color255
        instance._AmbientScaleHDR = float(entity_data.get('_ambientscalehdr', 1))  # Type: float
        instance.SunSpreadAngle = float(entity_data.get('sunspreadangle', 0))  # Type: float


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
        instance.damage = int(entity_data.get('damage', 10))  # Type: integer
        instance.damagecap = int(entity_data.get('damagecap', 20))  # Type: integer
        instance.damagetype = entity_data.get('damagetype', None)  # Type: choices
        instance.damagemodel = entity_data.get('damagemodel', None)  # Type: choices
        instance.nodmgforce = entity_data.get('nodmgforce', None)  # Type: choices


class light_spot(Targetname, Light, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(Light).__init__()
        super(Angles).__init__()
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
        Light.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance._inner_cone = int(entity_data.get('_inner_cone', 30))  # Type: integer
        instance._cone = int(entity_data.get('_cone', 45))  # Type: integer
        instance._exponent = int(entity_data.get('_exponent', 1))  # Type: integer
        instance._distance = int(entity_data.get('_distance', 0))  # Type: integer
        instance.pitch = float(entity_data.get('pitch', -90))  # Type: angle_negative_pitch


class light_dynamic(Targetname, Parentname, Angles):
    icon_sprite = "editor/light.vmt"

    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
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
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance._light = parse_int_vector(entity_data.get('_light', "255 255 255 200"))  # Type: color255
        instance.brightness = int(entity_data.get('brightness', 0))  # Type: integer
        instance._inner_cone = int(entity_data.get('_inner_cone', 30))  # Type: integer
        instance._cone = int(entity_data.get('_cone', 45))  # Type: integer
        instance.pitch = int(entity_data.get('pitch', -90))  # Type: integer
        instance.distance = float(entity_data.get('distance', 120))  # Type: float
        instance.spotlight_radius = float(entity_data.get('spotlight_radius', 80))  # Type: float
        instance.style = entity_data.get('style', None)  # Type: choices


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
