
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


class TriggerOnce(Targetname, Origin):
    def __init__(self):
        super(Targetname).__init__()
        super(Origin).__init__()
        self.filtername = None  # Type: filterclass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.filtername = entity_data.get('filtername', None)  # Type: filterclass


class worldbase(Base):
    def __init__(self):
        super().__init__()
        self.message = None  # Type: string
        self.skyname = "blacksky"  # Type: string
        self.chaptertitle = None  # Type: string
        self.startdark = None  # Type: boolean
        self.gametitle = None  # Type: boolean
        self.newunit = None  # Type: choices
        self.maxoccludeearea = 0  # Type: float
        self.minoccluderarea = 0  # Type: float
        self.maxoccludeearea_x360 = 0  # Type: float
        self.minoccluderarea_x360 = 0  # Type: float
        self.maxpropscreenwidth = -1  # Type: float
        self.minpropscreenwidth = None  # Type: float
        self.detailvbsp = "detail.vbsp"  # Type: string
        self.detailmaterial = "detail/detailsprites"  # Type: string
        self.coldworld = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.message = entity_data.get('message', None)  # Type: string
        instance.skyname = entity_data.get('skyname', "blacksky")  # Type: string
        instance.chaptertitle = entity_data.get('chaptertitle', None)  # Type: string
        instance.startdark = entity_data.get('startdark', None)  # Type: boolean
        instance.gametitle = entity_data.get('gametitle', None)  # Type: boolean
        instance.newunit = entity_data.get('newunit', None)  # Type: choices
        instance.maxoccludeearea = float(entity_data.get('maxoccludeearea', 0))  # Type: float
        instance.minoccluderarea = float(entity_data.get('minoccluderarea', 0))  # Type: float
        instance.maxoccludeearea_x360 = float(entity_data.get('maxoccludeearea_x360', 0))  # Type: float
        instance.minoccluderarea_x360 = float(entity_data.get('minoccluderarea_x360', 0))  # Type: float
        instance.maxpropscreenwidth = float(entity_data.get('maxpropscreenwidth', -1))  # Type: float
        instance.minpropscreenwidth = float(entity_data.get('minpropscreenwidth', 0))  # Type: float
        instance.detailvbsp = entity_data.get('detailvbsp', "detail.vbsp")  # Type: string
        instance.detailmaterial = entity_data.get('detailmaterial', "detail/detailsprites")  # Type: string
        instance.coldworld = entity_data.get('coldworld', None)  # Type: boolean


class env_fire(Targetname):
    icon_sprite = "editor/env_fire"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.health = 30  # Type: integer
        self.firesize = 64  # Type: integer
        self.fireattack = 4  # Type: integer
        self.firetype = None  # Type: choices
        self.ignitionpoint = 32  # Type: float
        self.damagescale = 1.0  # Type: float
        self.LightRadiusScale = 1.0  # Type: float
        self.LightBrightness = 1  # Type: integer
        self.LightColor = [255, 255, 255]  # Type: color255
        self.LoopSound = "d1_town.LargeFireLoop"  # Type: sound
        self.IgniteSound = "ASW_Flare.IgniteFlare"  # Type: sound

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.health = parse_source_value(entity_data.get('health', 30))  # Type: integer
        instance.firesize = parse_source_value(entity_data.get('firesize', 64))  # Type: integer
        instance.fireattack = parse_source_value(entity_data.get('fireattack', 4))  # Type: integer
        instance.firetype = entity_data.get('firetype', None)  # Type: choices
        instance.ignitionpoint = float(entity_data.get('ignitionpoint', 32))  # Type: float
        instance.damagescale = float(entity_data.get('damagescale', 1.0))  # Type: float
        instance.LightRadiusScale = float(entity_data.get('lightradiusscale', 1.0))  # Type: float
        instance.LightBrightness = parse_source_value(entity_data.get('lightbrightness', 1))  # Type: integer
        instance.LightColor = parse_int_vector(entity_data.get('lightcolor', "255 255 255"))  # Type: color255
        instance.LoopSound = entity_data.get('loopsound', "d1_town.LargeFireLoop")  # Type: sound
        instance.IgniteSound = entity_data.get('ignitesound', "ASW_Flare.IgniteFlare")  # Type: sound


class sky_camera(Angles):
    def __init__(self):
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.scale = 16  # Type: integer
        self.fogenable = None  # Type: boolean
        self.fogblend = None  # Type: boolean
        self.use_angles = None  # Type: boolean
        self.fogcolor = [255, 255, 255]  # Type: color255
        self.fogcolor2 = [255, 255, 255]  # Type: color255
        self.fogdir = "1 0 0"  # Type: string
        self.fogstart = "500.0"  # Type: string
        self.fogend = "2000.0"  # Type: string
        self.targetname = None  # Type: target_source

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.scale = parse_source_value(entity_data.get('scale', 16))  # Type: integer
        instance.fogenable = entity_data.get('fogenable', None)  # Type: boolean
        instance.fogblend = entity_data.get('fogblend', None)  # Type: boolean
        instance.use_angles = entity_data.get('use_angles', None)  # Type: boolean
        instance.fogcolor = parse_int_vector(entity_data.get('fogcolor', "255 255 255"))  # Type: color255
        instance.fogcolor2 = parse_int_vector(entity_data.get('fogcolor2', "255 255 255"))  # Type: color255
        instance.fogdir = entity_data.get('fogdir', "1 0 0")  # Type: string
        instance.fogstart = entity_data.get('fogstart', "500.0")  # Type: string
        instance.fogend = entity_data.get('fogend', "2000.0")  # Type: string
        instance.targetname = entity_data.get('targetname', None)  # Type: target_source


class light(Targetname):
    icon_sprite = "editor/light.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target = None  # Type: target_destination
        self._distance = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance._distance = parse_source_value(entity_data.get('_distance', 0))  # Type: integer


class BasePropPhysics(Targetname, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.minhealthdmg = None  # Type: integer
        self.shadowcastdist = None  # Type: integer
        self.physdamagescale = 0.1  # Type: float
        self.Damagetype = None  # Type: choices
        self.nodamageforces = None  # Type: boolean
        self.inertiaScale = 1.0  # Type: float
        self.massScale = 0  # Type: float
        self.overridescript = None  # Type: string
        self.damagetoenablemotion = None  # Type: integer
        self.forcetoenablemotion = None  # Type: float
        self.puntsound = None  # Type: sound
        self.addon = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.minhealthdmg = parse_source_value(entity_data.get('minhealthdmg', 0))  # Type: integer
        instance.shadowcastdist = parse_source_value(entity_data.get('shadowcastdist', 0))  # Type: integer
        instance.physdamagescale = float(entity_data.get('physdamagescale', 0.1))  # Type: float
        instance.Damagetype = entity_data.get('damagetype', None)  # Type: choices
        instance.nodamageforces = entity_data.get('nodamageforces', None)  # Type: boolean
        instance.inertiaScale = float(entity_data.get('inertiascale', 1.0))  # Type: float
        instance.massScale = float(entity_data.get('massscale', 0))  # Type: float
        instance.overridescript = entity_data.get('overridescript', None)  # Type: string
        instance.damagetoenablemotion = parse_source_value(entity_data.get('damagetoenablemotion', 0))  # Type: integer
        instance.forcetoenablemotion = float(entity_data.get('forcetoenablemotion', 0))  # Type: float
        instance.puntsound = entity_data.get('puntsound', None)  # Type: sound
        instance.addon = entity_data.get('addon', None)  # Type: string


class prop_physics(BasePropPhysics):
    def __init__(self):
        super(BasePropPhysics).__init__()
        self.origin = [0, 0, 0]
        self.BulletForceImmune = None  # Type: boolean

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePropPhysics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.BulletForceImmune = entity_data.get('bulletforceimmune', None)  # Type: boolean


class info_node(Base):
    model = "models/editor/ground_node.mdl"
    def __init__(self):
        super().__init__()
        self.origin = [0, 0, 0]
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
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
        self.InvertAllow = None  # Type: boolean
        self.preciseMovement = None  # Type: boolean
        self.priority = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.StartNode = parse_source_value(entity_data.get('startnode', 0))  # Type: node_dest
        instance.EndNode = parse_source_value(entity_data.get('endnode', 0))  # Type: node_dest
        instance.initialstate = entity_data.get('initialstate', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.linktype = entity_data.get('linktype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.AllowUse = entity_data.get('allowuse', None)  # Type: string
        instance.InvertAllow = entity_data.get('invertallow', None)  # Type: boolean
        instance.preciseMovement = entity_data.get('precisemovement', None)  # Type: boolean
        instance.priority = entity_data.get('priority', None)  # Type: choices


class TalkNPC(Base):
    def __init__(self):
        super().__init__()
        self.UseSentence = None  # Type: string
        self.UnUseSentence = None  # Type: string
        self.DontUseSpeechSemaphore = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.UseSentence = entity_data.get('usesentence', None)  # Type: string
        instance.UnUseSentence = entity_data.get('unusesentence', None)  # Type: string
        instance.DontUseSpeechSemaphore = entity_data.get('dontusespeechsemaphore', None)  # Type: choices


class PlayerCompanion(Base):
    def __init__(self):
        super().__init__()
        self.AlwaysTransition = "CHOICES NOT SUPPORTED"  # Type: choices
        self.DontPickupWeapons = "CHOICES NOT SUPPORTED"  # Type: choices
        self.GameEndAlly = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.AlwaysTransition = entity_data.get('alwaystransition', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.DontPickupWeapons = entity_data.get('dontpickupweapons', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.GameEndAlly = entity_data.get('gameendally', "CHOICES NOT SUPPORTED")  # Type: choices


class RappelNPC(Base):
    def __init__(self):
        super().__init__()
        self.waitingtorappel = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.waitingtorappel = entity_data.get('waitingtorappel', "CHOICES NOT SUPPORTED")  # Type: choices


class AlyxInteractable(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class CombineBallSpawners(Targetname, Angles, Origin):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Origin).__init__()
        self.ballcount = 3  # Type: integer
        self.minspeed = 300.0  # Type: float
        self.maxspeed = 600.0  # Type: float
        self.ballradius = 20.0  # Type: float
        self.balltype = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ballrespawntime = 4.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.ballcount = parse_source_value(entity_data.get('ballcount', 3))  # Type: integer
        instance.minspeed = float(entity_data.get('minspeed', 300.0))  # Type: float
        instance.maxspeed = float(entity_data.get('maxspeed', 600.0))  # Type: float
        instance.ballradius = float(entity_data.get('ballradius', 20.0))  # Type: float
        instance.balltype = entity_data.get('balltype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ballrespawntime = float(entity_data.get('ballrespawntime', 4.0))  # Type: float


class trigger_asw_use_area(Base):
    def __init__(self):
        super().__init__()
        self.usetargetname = None  # Type: target_destination
        self.playersrequired = 1  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.usetargetname = entity_data.get('usetargetname', None)  # Type: target_destination
        instance.playersrequired = parse_source_value(entity_data.get('playersrequired', 1))  # Type: integer


class prop_combine_ball(BasePropPhysics):
    def __init__(self):
        super(BasePropPhysics).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BasePropPhysics.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class trigger_physics_trap(Angles):
    def __init__(self):
        super(Angles).__init__()
        self.dissolvetype = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        instance.dissolvetype = entity_data.get('dissolvetype', "CHOICES NOT SUPPORTED")  # Type: choices


class trigger_weapon_dissolve(Base):
    def __init__(self):
        super().__init__()
        self.emittername = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.emittername = entity_data.get('emittername', None)  # Type: target_destination


class trigger_weapon_strip(Base):
    def __init__(self):
        super().__init__()
        self.KillWeapons = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
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


class npc_grenade_frag(Base):
    model = "models/Weapons/w_grenade.mdl"
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class npc_combine_cannon(Base):
    model = "models/combine_soldier.mdl"
    def __init__(self):
        super().__init__()
        self.sightdist = 1024  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.sightdist = float(entity_data.get('sightdist', 1024))  # Type: float


class npc_combine_camera(Base):
    model = "models/combine_camera/combine_camera.mdl"
    def __init__(self):
        super().__init__()
        self.innerradius = 300  # Type: integer
        self.outerradius = 450  # Type: integer
        self.minhealthdmg = None  # Type: integer
        self.defaulttarget = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.innerradius = parse_source_value(entity_data.get('innerradius', 300))  # Type: integer
        instance.outerradius = parse_source_value(entity_data.get('outerradius', 450))  # Type: integer
        instance.minhealthdmg = parse_source_value(entity_data.get('minhealthdmg', 0))  # Type: integer
        instance.defaulttarget = entity_data.get('defaulttarget', None)  # Type: target_destination


class npc_turret_ground(AlyxInteractable):
    model = "models/combine_turrets/ground_turret.mdl"
    def __init__(self):
        super(AlyxInteractable).__init__()
        self.origin = [0, 0, 0]
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        AlyxInteractable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class npc_turret_ceiling(Targetname, Angles):
    model = "models/combine_turrets/ceiling_turret.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.minhealthdmg = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.minhealthdmg = parse_source_value(entity_data.get('minhealthdmg', 0))  # Type: integer


class npc_turret_floor(Targetname, Angles):
    model = "models/combine_turrets/floor_turret.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.SkinNumber = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.SkinNumber = parse_source_value(entity_data.get('skinnumber', 0))  # Type: integer


class VehicleDriverNPC(Base):
    def __init__(self):
        super().__init__()
        self.vehicle = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
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


class npc_rollermine(AlyxInteractable):
    model = "models/roller.mdl"
    def __init__(self):
        super(AlyxInteractable).__init__()
        self.startburied = "CHOICES NOT SUPPORTED"  # Type: choices
        self.uniformsightdist = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        AlyxInteractable.from_dict(instance, entity_data)
        instance.startburied = entity_data.get('startburied', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.uniformsightdist = entity_data.get('uniformsightdist', None)  # Type: choices


class npc_missiledefense(Base):
    model = "models/missile_defense.mdl"
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class npc_sniper(Base):
    model = "models/combine_soldier.mdl"
    def __init__(self):
        super().__init__()
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
        Base.from_dict(instance, entity_data)
        instance.radius = parse_source_value(entity_data.get('radius', 0))  # Type: integer
        instance.misses = parse_source_value(entity_data.get('misses', 0))  # Type: integer
        instance.beambrightness = parse_source_value(entity_data.get('beambrightness', 100))  # Type: integer
        instance.shootZombiesInChest = entity_data.get('shootzombiesinchest', None)  # Type: choices
        instance.shielddistance = float(entity_data.get('shielddistance', 64))  # Type: float
        instance.shieldradius = float(entity_data.get('shieldradius', 48))  # Type: float
        instance.PaintInterval = float(entity_data.get('paintinterval', 1))  # Type: float
        instance.PaintIntervalVariance = float(entity_data.get('paintintervalvariance', 0.75))  # Type: float


class info_radar_target(Targetname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.radius = 6000  # Type: float
        self.type = None  # Type: choices
        self.mode = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.radius = float(entity_data.get('radius', 6000))  # Type: float
        instance.type = entity_data.get('type', None)  # Type: choices
        instance.mode = entity_data.get('mode', None)  # Type: choices


class info_target_vehicle_transition(Targetname, Angles):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_snipertarget(Targetname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.speed = 2  # Type: integer
        self.groupname = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.speed = parse_source_value(entity_data.get('speed', 2))  # Type: integer
        instance.groupname = entity_data.get('groupname', None)  # Type: string


class prop_thumper(Targetname, Angles):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.model = "models/props_combine/CombineThumper002.mdl"  # Type: studio
        self.dustscale = "CHOICES NOT SUPPORTED"  # Type: choices
        self.EffectRadius = 1000  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.model = entity_data.get('model', "models/props_combine/CombineThumper002.mdl")  # Type: studio
        instance.dustscale = entity_data.get('dustscale', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.EffectRadius = parse_source_value(entity_data.get('effectradius', 1000))  # Type: integer


class npc_antlion(Base):
    model = "models/antlion.mdl"
    def __init__(self):
        super().__init__()
        self.startburrowed = "CHOICES NOT SUPPORTED"  # Type: choices
        self.radius = 256  # Type: integer
        self.eludedist = 1024  # Type: integer
        self.ignorebugbait = "CHOICES NOT SUPPORTED"  # Type: choices
        self.unburroweffects = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.startburrowed = entity_data.get('startburrowed', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.radius = parse_source_value(entity_data.get('radius', 256))  # Type: integer
        instance.eludedist = parse_source_value(entity_data.get('eludedist', 1024))  # Type: integer
        instance.ignorebugbait = entity_data.get('ignorebugbait', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.unburroweffects = entity_data.get('unburroweffects', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_antlionguard(Base):
    model = "models/antlion_guard.mdl"
    def __init__(self):
        super().__init__()
        self.startburrowed = "CHOICES NOT SUPPORTED"  # Type: choices
        self.allowbark = "CHOICES NOT SUPPORTED"  # Type: choices
        self.cavernbreed = "CHOICES NOT SUPPORTED"  # Type: choices
        self.incavern = "CHOICES NOT SUPPORTED"  # Type: choices
        self.shovetargets = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.startburrowed = entity_data.get('startburrowed', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.allowbark = entity_data.get('allowbark', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.cavernbreed = entity_data.get('cavernbreed', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.incavern = entity_data.get('incavern', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.shovetargets = entity_data.get('shovetargets', None)  # Type: string


class npc_crow(Base):
    model = "models/crow.mdl"
    def __init__(self):
        super().__init__()
        self.deaf = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.deaf = entity_data.get('deaf', None)  # Type: choices


class npc_seagull(Base):
    model = "models/seagull.mdl"
    def __init__(self):
        super().__init__()
        self.deaf = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.deaf = entity_data.get('deaf', None)  # Type: choices


class npc_pigeon(Base):
    model = "models/pigeon.mdl"
    def __init__(self):
        super().__init__()
        self.deaf = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.deaf = entity_data.get('deaf', None)  # Type: choices


class npc_ichthyosaur(Base):
    model = "models/ichthyosaur.mdl"
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class BaseHeadcrab(Base):
    def __init__(self):
        super().__init__()
        self.startburrowed = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.startburrowed = entity_data.get('startburrowed', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_headcrab(BaseHeadcrab):
    model = "models/Headcrabclassic.mdl"
    def __init__(self):
        super(BaseHeadcrab).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseHeadcrab.from_dict(instance, entity_data)


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


class npc_stalker(Base):
    model = "models/Stalker.mdl"
    def __init__(self):
        super().__init__()
        self.BeamPower = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.BeamPower = entity_data.get('beampower', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_bullseye(Base):
    icon_sprite = "editor/bullseye.vmt"
    def __init__(self):
        super().__init__()
        self.health = 35  # Type: integer
        self.minangle = "360"  # Type: string
        self.mindist = "0"  # Type: string
        self.autoaimradius = 0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.health = parse_source_value(entity_data.get('health', 35))  # Type: integer
        instance.minangle = entity_data.get('minangle', "360")  # Type: string
        instance.mindist = entity_data.get('mindist', "0")  # Type: string
        instance.autoaimradius = float(entity_data.get('autoaimradius', 0))  # Type: float


class npc_enemyfinder(Base):
    def __init__(self):
        super().__init__()
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
        Base.from_dict(instance, entity_data)
        instance.FieldOfView = entity_data.get('fieldofview', "0.2")  # Type: string
        instance.MinSearchDist = parse_source_value(entity_data.get('minsearchdist', 0))  # Type: integer
        instance.MaxSearchDist = parse_source_value(entity_data.get('maxsearchdist', 2048))  # Type: integer
        instance.freepass_timetotrigger = float(entity_data.get('freepass_timetotrigger', 0))  # Type: float
        instance.freepass_duration = float(entity_data.get('freepass_duration', 0))  # Type: float
        instance.freepass_movetolerance = float(entity_data.get('freepass_movetolerance', 120))  # Type: float
        instance.freepass_refillrate = float(entity_data.get('freepass_refillrate', 0.5))  # Type: float
        instance.freepass_peektime = float(entity_data.get('freepass_peektime', 0))  # Type: float
        instance.StartOn = entity_data.get('starton', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_enemyfinder_combinecannon(Base):
    def __init__(self):
        super().__init__()
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
        Base.from_dict(instance, entity_data)
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


class npc_citizen(PlayerCompanion, TalkNPC):
    def __init__(self):
        super(PlayerCompanion).__init__()
        super(TalkNPC).__init__()
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
        PlayerCompanion.from_dict(instance, entity_data)
        TalkNPC.from_dict(instance, entity_data)
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


class npc_fisherman(Base):
    model = "models/Barney.mdl"
    def __init__(self):
        super().__init__()
        self.ExpressionOverride = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.ExpressionOverride = entity_data.get('expressionoverride', None)  # Type: string


class npc_barney(PlayerCompanion, TalkNPC):
    model = "models/Barney.mdl"
    def __init__(self):
        super(PlayerCompanion).__init__()
        super(TalkNPC).__init__()
        self.additionalequipment = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ExpressionOverride = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        PlayerCompanion.from_dict(instance, entity_data)
        TalkNPC.from_dict(instance, entity_data)
        instance.additionalequipment = entity_data.get('additionalequipment', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ExpressionOverride = entity_data.get('expressionoverride', None)  # Type: string


class BaseCombine(RappelNPC):
    def __init__(self):
        super(RappelNPC).__init__()
        self.additionalequipment = "CHOICES NOT SUPPORTED"  # Type: choices
        self.NumGrenades = "CHOICES NOT SUPPORTED"  # Type: choices
        self.TeleportGrenades = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        RappelNPC.from_dict(instance, entity_data)
        instance.additionalequipment = entity_data.get('additionalequipment', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.NumGrenades = entity_data.get('numgrenades', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.TeleportGrenades = entity_data.get('teleportgrenades', "CHOICES NOT SUPPORTED")  # Type: choices


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


class npc_launcher(Base):
    model = "models/junk/w_traffcone.mdl"
    def __init__(self):
        super().__init__()
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
        Base.from_dict(instance, entity_data)
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


class npc_hunter(Base):
    model = "models/hunter.mdl"
    def __init__(self):
        super().__init__()
        self.FollowTarget = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.FollowTarget = entity_data.get('followtarget', None)  # Type: target_destination


class npc_hunter_maker(Base):
    icon_sprite = "editor/npc_maker.vmt"
    def __init__(self):
        super().__init__()
        self.origin = [0, 0, 0]
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class npc_advisor(Base):
    def __init__(self):
        super().__init__()
        self.model = "models/advisor.mdl"  # Type: studio
        self.levitationarea = None  # Type: string
        self.levitategoal_bottom = None  # Type: target_destination
        self.levitategoal_top = None  # Type: target_destination
        self.staging_ent_names = None  # Type: string
        self.priority_grab_name = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', "models/advisor.mdl")  # Type: studio
        instance.levitationarea = entity_data.get('levitationarea', None)  # Type: string
        instance.levitategoal_bottom = entity_data.get('levitategoal_bottom', None)  # Type: target_destination
        instance.levitategoal_top = entity_data.get('levitategoal_top', None)  # Type: target_destination
        instance.staging_ent_names = entity_data.get('staging_ent_names', None)  # Type: string
        instance.priority_grab_name = entity_data.get('priority_grab_name', None)  # Type: string


class env_sporeexplosion(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.spawnrate = 25  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.spawnrate = float(entity_data.get('spawnrate', 25))  # Type: float


class env_gunfire(Targetname):
    def __init__(self):
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


class env_headcrabcanister(Targetname, Angles):
    model = "models/props_combine/headcrabcannister01b.mdl"
    def __init__(self):
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


class npc_vortigaunt(PlayerCompanion, TalkNPC):
    def __init__(self):
        super(PlayerCompanion).__init__()
        super(TalkNPC).__init__()
        self.model = "models/vortigaunt.mdl"  # Type: studio
        self.ArmorRechargeEnabled = "CHOICES NOT SUPPORTED"  # Type: choices
        self.HealthRegenerateEnabled = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        PlayerCompanion.from_dict(instance, entity_data)
        TalkNPC.from_dict(instance, entity_data)
        instance.model = entity_data.get('model', "models/vortigaunt.mdl")  # Type: studio
        instance.ArmorRechargeEnabled = entity_data.get('armorrechargeenabled', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.HealthRegenerateEnabled = entity_data.get('healthregenerateenabled', None)  # Type: choices


class npc_spotlight(Base):
    def __init__(self):
        super().__init__()
        self.health = 100  # Type: integer
        self.YawRange = 90  # Type: integer
        self.PitchMin = 35  # Type: integer
        self.PitchMax = 50  # Type: integer
        self.IdleSpeed = 2  # Type: integer
        self.AlertSpeed = 5  # Type: integer
        self.spotlightlength = 500  # Type: integer
        self.spotlightwidth = 50  # Type: integer
        self.rendercolor = [255, 255, 255]  # Type: color255

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.health = parse_source_value(entity_data.get('health', 100))  # Type: integer
        instance.YawRange = parse_source_value(entity_data.get('yawrange', 90))  # Type: integer
        instance.PitchMin = parse_source_value(entity_data.get('pitchmin', 35))  # Type: integer
        instance.PitchMax = parse_source_value(entity_data.get('pitchmax', 50))  # Type: integer
        instance.IdleSpeed = parse_source_value(entity_data.get('idlespeed', 2))  # Type: integer
        instance.AlertSpeed = parse_source_value(entity_data.get('alertspeed', 5))  # Type: integer
        instance.spotlightlength = parse_source_value(entity_data.get('spotlightlength', 500))  # Type: integer
        instance.spotlightwidth = parse_source_value(entity_data.get('spotlightwidth', 50))  # Type: integer
        instance.rendercolor = parse_int_vector(entity_data.get('rendercolor', "255 255 255"))  # Type: color255



entity_class_handle = {
    'Angles': Angles,
    'Origin': Origin,
    'Targetname': Targetname,
    'TriggerOnce': TriggerOnce,
    'worldbase': worldbase,
    'env_fire': env_fire,
    'sky_camera': sky_camera,
    'light': light,
    'BasePropPhysics': BasePropPhysics,
    'prop_physics': prop_physics,
    'info_node': info_node,
    'info_node_link': info_node_link,
    'TalkNPC': TalkNPC,
    'PlayerCompanion': PlayerCompanion,
    'RappelNPC': RappelNPC,
    'AlyxInteractable': AlyxInteractable,
    'CombineBallSpawners': CombineBallSpawners,
    'trigger_asw_use_area': trigger_asw_use_area,
    'prop_combine_ball': prop_combine_ball,
    'trigger_physics_trap': trigger_physics_trap,
    'trigger_weapon_dissolve': trigger_weapon_dissolve,
    'trigger_weapon_strip': trigger_weapon_strip,
    'func_combine_ball_spawner': func_combine_ball_spawner,
    'point_combine_ball_launcher': point_combine_ball_launcher,
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
}