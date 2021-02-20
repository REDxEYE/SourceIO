from .base_entity_classes import *

def parse_int_vector(string):
    return [int(val) for val in string.split(' ')]


def parse_float_vector(string):
    return [float(val) for val in string.split(' ')]


class Base:
    def __init__(self):
        self.hammer_id = 0
        self.class_name = 'ANY'

    @staticmethod
    def from_dict(instance, entity_data: dict):
        instance.hammer_id = int(entity_data.get('hammerid'))
        instance.class_name = entity_data.get('classname')


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


class CombineBallSpawners(Targetname, Angles, Global, Origin):
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Global).__init__()
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
        Global.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.ballcount = int(entity_data.get('ballcount', 3))  # Type: integer
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
        instance.maxballbounces = int(entity_data.get('maxballbounces', 8))  # Type: integer


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
        instance.innerradius = int(entity_data.get('innerradius', 300))  # Type: integer
        instance.outerradius = int(entity_data.get('outerradius', 450))  # Type: integer
        instance.minhealthdmg = int(entity_data.get('minhealthdmg', 0))  # Type: integer
        instance.defaulttarget = entity_data.get('defaulttarget', None)  # Type: target_destination


class npc_turret_ground(BaseNPC, Parentname, AlyxInteractable):
    model = "models/combine_turrets/ground_turret.mdl"
    def __init__(self):
        super(BaseNPC).__init__()
        super(Parentname).__init__()
        super(AlyxInteractable).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseNPC.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        AlyxInteractable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class npc_turret_ceiling(Targetname, Angles, Studiomodel):
    model = "models/combine_turrets/ceiling_turret.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(Studiomodel).__init__()
        self.origin = [0, 0, 0]
        self.minhealthdmg = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Studiomodel.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.minhealthdmg = int(entity_data.get('minhealthdmg', 0))  # Type: integer


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
        instance.SkinNumber = int(entity_data.get('skinnumber', 0))  # Type: integer


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
        instance.radius = int(entity_data.get('radius', 0))  # Type: integer
        instance.misses = int(entity_data.get('misses', 0))  # Type: integer
        instance.beambrightness = int(entity_data.get('beambrightness', 100))  # Type: integer
        instance.shootZombiesInChest = entity_data.get('shootzombiesinchest', None)  # Type: choices
        instance.shielddistance = float(entity_data.get('shielddistance', 64))  # Type: float
        instance.shieldradius = float(entity_data.get('shieldradius', 48))  # Type: float
        instance.PaintInterval = float(entity_data.get('paintinterval', 1))  # Type: float
        instance.PaintIntervalVariance = float(entity_data.get('paintintervalvariance', 0.75))  # Type: float


class info_radar_target(Targetname, EnableDisable, Parentname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.radius = 6000  # Type: float
        self.type = None  # Type: choices
        self.mode = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.radius = float(entity_data.get('radius', 6000))  # Type: float
        instance.type = entity_data.get('type', None)  # Type: choices
        instance.mode = entity_data.get('mode', None)  # Type: choices


class info_target_vehicle_transition(Targetname, Angles, EnableDisable):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(Angles).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_snipertarget(Targetname, Parentname):
    icon_sprite = "editor/info_target.vmt"
    def __init__(self):
        super(Targetname).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.speed = 2  # Type: integer
        self.groupname = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.speed = int(entity_data.get('speed', 2))  # Type: integer
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
        instance.EffectRadius = int(entity_data.get('effectradius', 1000))  # Type: integer


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
        instance.radius = int(entity_data.get('radius', 256))  # Type: integer
        instance.eludedist = int(entity_data.get('eludedist', 1024))  # Type: integer
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


class npc_bullseye(BaseNPC, Parentname):
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
        BaseNPC.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.health = int(entity_data.get('health', 35))  # Type: integer
        instance.minangle = entity_data.get('minangle', "360")  # Type: string
        instance.mindist = entity_data.get('mindist', "0")  # Type: string
        instance.autoaimradius = float(entity_data.get('autoaimradius', 0))  # Type: float


class npc_enemyfinder(BaseNPC, Parentname):
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
        BaseNPC.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.FieldOfView = entity_data.get('fieldofview', "0.2")  # Type: string
        instance.MinSearchDist = int(entity_data.get('minsearchdist', 0))  # Type: integer
        instance.MaxSearchDist = int(entity_data.get('maxsearchdist', 2048))  # Type: integer
        instance.freepass_timetotrigger = float(entity_data.get('freepass_timetotrigger', 0))  # Type: float
        instance.freepass_duration = float(entity_data.get('freepass_duration', 0))  # Type: float
        instance.freepass_movetolerance = float(entity_data.get('freepass_movetolerance', 120))  # Type: float
        instance.freepass_refillrate = float(entity_data.get('freepass_refillrate', 0.5))  # Type: float
        instance.freepass_peektime = float(entity_data.get('freepass_peektime', 0))  # Type: float
        instance.StartOn = entity_data.get('starton', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_enemyfinder_combinecannon(BaseNPC, Parentname):
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
        BaseNPC.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.FieldOfView = entity_data.get('fieldofview', "0.2")  # Type: string
        instance.MinSearchDist = int(entity_data.get('minsearchdist', 0))  # Type: integer
        instance.MaxSearchDist = int(entity_data.get('maxsearchdist', 2048))  # Type: integer
        instance.SnapToEnt = entity_data.get('snaptoent', None)  # Type: target_destination
        instance.freepass_timetotrigger = float(entity_data.get('freepass_timetotrigger', 0))  # Type: float
        instance.freepass_duration = float(entity_data.get('freepass_duration', 0))  # Type: float
        instance.freepass_movetolerance = float(entity_data.get('freepass_movetolerance', 120))  # Type: float
        instance.freepass_refillrate = float(entity_data.get('freepass_refillrate', 0.5))  # Type: float
        instance.freepass_peektime = float(entity_data.get('freepass_peektime', 0))  # Type: float
        instance.StartOn = entity_data.get('starton', "CHOICES NOT SUPPORTED")  # Type: choices


class npc_citizen(TalkNPC, Parentname, PlayerCompanion):
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
        BaseNPC.from_dict(instance, entity_data)
        TalkNPC.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        PlayerCompanion.from_dict(instance, entity_data)
        instance.additionalequipment = entity_data.get('additionalequipment', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ammosupply = entity_data.get('ammosupply', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ammoamount = int(entity_data.get('ammoamount', 1))  # Type: integer
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
        BaseNPC.from_dict(instance, entity_data)
        RappelNPC.from_dict(instance, entity_data)
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


class npc_launcher(BaseNPC, Parentname):
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
        BaseNPC.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.StartOn = entity_data.get('starton', None)  # Type: choices
        instance.MissileModel = entity_data.get('missilemodel', "models/Weapons/wscanner_grenade.mdl")  # Type: studio
        instance.LaunchSound = entity_data.get('launchsound', "npc/waste_scanner/grenade_fire.wav")  # Type: sound
        instance.FlySound = entity_data.get('flysound', "ambient/objects/machine2.wav")  # Type: sound
        instance.SmokeTrail = entity_data.get('smoketrail', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.LaunchSmoke = entity_data.get('launchsmoke', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.LaunchDelay = int(entity_data.get('launchdelay', 8))  # Type: integer
        instance.LaunchSpeed = entity_data.get('launchspeed', "200")  # Type: string
        instance.PathCornerName = entity_data.get('pathcornername', None)  # Type: target_destination
        instance.HomingSpeed = entity_data.get('homingspeed', None)  # Type: string
        instance.HomingStrength = int(entity_data.get('homingstrength', 10))  # Type: integer
        instance.HomingDelay = entity_data.get('homingdelay', None)  # Type: string
        instance.HomingRampUp = entity_data.get('homingrampup', "0.5")  # Type: string
        instance.HomingDuration = entity_data.get('homingduration', "5")  # Type: string
        instance.HomingRampDown = entity_data.get('homingrampdown', "1.0")  # Type: string
        instance.Gravity = entity_data.get('gravity', "1.0")  # Type: string
        instance.MinRange = int(entity_data.get('minrange', 100))  # Type: integer
        instance.MaxRange = int(entity_data.get('maxrange', 2048))  # Type: integer
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


class env_sporeexplosion(Targetname, EnableDisable, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.spawnrate = 25  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.spawnrate = float(entity_data.get('spawnrate', 25))  # Type: float


class env_gunfire(Targetname, EnableDisable, Parentname):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(Parentname).__init__()
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
        EnableDisable.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.minburstsize = int(entity_data.get('minburstsize', 2))  # Type: integer
        instance.maxburstsize = int(entity_data.get('maxburstsize', 7))  # Type: integer
        instance.minburstdelay = float(entity_data.get('minburstdelay', 2))  # Type: float
        instance.maxburstdelay = float(entity_data.get('maxburstdelay', 5))  # Type: float
        instance.rateoffire = float(entity_data.get('rateoffire', 10))  # Type: float
        instance.spread = entity_data.get('spread', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.bias = entity_data.get('bias', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.collisions = entity_data.get('collisions', None)  # Type: choices
        instance.shootsound = entity_data.get('shootsound', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.tracertype = entity_data.get('tracertype', "CHOICES NOT SUPPORTED")  # Type: choices


class env_headcrabcanister(Angles, Targetname, Parentname):
    model = "models/props_combine/headcrabcannister01b.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
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
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.HeadcrabType = entity_data.get('headcrabtype', None)  # Type: choices
        instance.HeadcrabCount = int(entity_data.get('headcrabcount', 6))  # Type: integer
        instance.FlightSpeed = float(entity_data.get('flightspeed', 3000))  # Type: float
        instance.FlightTime = float(entity_data.get('flighttime', 5))  # Type: float
        instance.StartingHeight = float(entity_data.get('startingheight', 0))  # Type: float
        instance.MinSkyboxRefireTime = float(entity_data.get('minskyboxrefiretime', 0))  # Type: float
        instance.MaxSkyboxRefireTime = float(entity_data.get('maxskyboxrefiretime', 0))  # Type: float
        instance.SkyboxCannisterCount = int(entity_data.get('skyboxcannistercount', 1))  # Type: integer
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
        BaseNPC.from_dict(instance, entity_data)
        TalkNPC.from_dict(instance, entity_data)
        PlayerCompanion.from_dict(instance, entity_data)
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
        instance.health = int(entity_data.get('health', 100))  # Type: integer
        instance.YawRange = int(entity_data.get('yawrange', 90))  # Type: integer
        instance.PitchMin = int(entity_data.get('pitchmin', 35))  # Type: integer
        instance.PitchMax = int(entity_data.get('pitchmax', 50))  # Type: integer
        instance.IdleSpeed = int(entity_data.get('idlespeed', 2))  # Type: integer
        instance.AlertSpeed = int(entity_data.get('alertspeed', 5))  # Type: integer
        instance.spotlightlength = int(entity_data.get('spotlightlength', 500))  # Type: integer
        instance.spotlightwidth = int(entity_data.get('spotlightwidth', 50))  # Type: integer



entity_class_handle = {
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
}