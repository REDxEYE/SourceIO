
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



class TriggerOnce(Origin, Targetname):

    @property
    def filtername(self):
        return self._raw_data.get('filtername', None)



class worldbase(Base):

    @property
    def message(self):
        return self._raw_data.get('message', None)

    @property
    def skyname(self):
        return self._raw_data.get('skyname', "blacksky")

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



class env_fire(Targetname):
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

    @property
    def LightRadiusScale(self):
        return parse_source_value(self._raw_data.get('lightradiusscale', 1.0))

    @property
    def LightBrightness(self):
        return parse_source_value(self._raw_data.get('lightbrightness', 1))

    @property
    def LightColor(self):
        return parse_int_vector(self._raw_data.get('lightcolor', "255 255 255"))

    @property
    def LoopSound(self):
        return self._raw_data.get('loopsound', "d1_town.LargeFireLoop")

    @property
    def IgniteSound(self):
        return self._raw_data.get('ignitesound', "ASW_Flare.IgniteFlare")



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
    def targetname(self):
        return self._raw_data.get('targetname', None)



class light(Targetname):
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



class BasePropPhysics(Targetname, Angles):

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



class prop_physics(BasePropPhysics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def BulletForceImmune(self):
        return self._raw_data.get('bulletforceimmune', "0")



class info_node(Base):
    model_ = "models/editor/ground_node.mdl"
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



class TalkNPC(Base):

    @property
    def UseSentence(self):
        return self._raw_data.get('usesentence', None)

    @property
    def UnUseSentence(self):
        return self._raw_data.get('unusesentence', None)

    @property
    def DontUseSpeechSemaphore(self):
        return self._raw_data.get('dontusespeechsemaphore', "0")



class PlayerCompanion(Base):

    @property
    def AlwaysTransition(self):
        return self._raw_data.get('alwaystransition', "No")

    @property
    def DontPickupWeapons(self):
        return self._raw_data.get('dontpickupweapons', "No")

    @property
    def GameEndAlly(self):
        return self._raw_data.get('gameendally', "No")



class RappelNPC(Base):

    @property
    def waitingtorappel(self):
        return self._raw_data.get('waitingtorappel', "No")



class AlyxInteractable(Base):
    pass


class CombineBallSpawners(Origin, Targetname, Angles):

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



class trigger_asw_use_area(Base):

    @property
    def usetargetname(self):
        return self._raw_data.get('usetargetname', None)

    @property
    def playersrequired(self):
        return parse_source_value(self._raw_data.get('playersrequired', 1))



class prop_combine_ball(BasePropPhysics):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class trigger_physics_trap(Angles):

    @property
    def dissolvetype(self):
        return self._raw_data.get('dissolvetype', "Energy")



class trigger_weapon_dissolve(Base):

    @property
    def emittername(self):
        return self._raw_data.get('emittername', "")



class trigger_weapon_strip(Base):

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



class npc_grenade_frag(Base):
    model_ = "models/Weapons/w_grenade.mdl"
    pass


class npc_combine_cannon(Base):
    model_ = "models/combine_soldier.mdl"

    @property
    def sightdist(self):
        return parse_source_value(self._raw_data.get('sightdist', 1024))



class npc_combine_camera(Base):
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



class npc_turret_ground(AlyxInteractable):
    model_ = "models/combine_turrets/ground_turret.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class npc_turret_ceiling(Targetname, Angles):
    model_ = "models/combine_turrets/ceiling_turret.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def minhealthdmg(self):
        return parse_source_value(self._raw_data.get('minhealthdmg', 0))



class npc_turret_floor(Targetname, Angles):
    model_ = "models/combine_turrets/floor_turret.mdl"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def SkinNumber(self):
        return parse_source_value(self._raw_data.get('skinnumber', 0))



class VehicleDriverNPC(Base):

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



class npc_rollermine(AlyxInteractable):
    model_ = "models/roller.mdl"

    @property
    def startburied(self):
        return self._raw_data.get('startburied', "No")

    @property
    def uniformsightdist(self):
        return self._raw_data.get('uniformsightdist', "0")



class npc_missiledefense(Base):
    model_ = "models/missile_defense.mdl"
    pass


class npc_sniper(Base):
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



class info_radar_target(Targetname):
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



class info_target_vehicle_transition(Targetname, Angles):
    icon_sprite = "editor/info_target.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class info_snipertarget(Targetname):
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



class prop_thumper(Targetname, Angles):
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



class npc_antlion(Base):
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



class npc_antlionguard(Base):
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



class npc_crow(Base):
    model_ = "models/crow.mdl"

    @property
    def deaf(self):
        return self._raw_data.get('deaf', "0")



class npc_seagull(Base):
    model_ = "models/seagull.mdl"

    @property
    def deaf(self):
        return self._raw_data.get('deaf', "0")



class npc_pigeon(Base):
    model_ = "models/pigeon.mdl"

    @property
    def deaf(self):
        return self._raw_data.get('deaf', "0")



class npc_ichthyosaur(Base):
    model_ = "models/ichthyosaur.mdl"
    pass


class BaseHeadcrab(Base):

    @property
    def startburrowed(self):
        return self._raw_data.get('startburrowed', "No")



class npc_headcrab(BaseHeadcrab):
    model_ = "models/Headcrabclassic.mdl"
    pass


class npc_headcrab_fast(BaseHeadcrab):
    model_ = "models/Headcrab.mdl"
    pass


class npc_headcrab_black(BaseHeadcrab):
    model_ = "models/Headcrabblack.mdl"
    pass


class npc_stalker(Base):
    model_ = "models/Stalker.mdl"

    @property
    def BeamPower(self):
        return self._raw_data.get('beampower', "Low")



class npc_bullseye(Base):
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



class npc_enemyfinder(Base):

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



class npc_enemyfinder_combinecannon(Base):

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



class npc_citizen(PlayerCompanion, TalkNPC):

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



class npc_fisherman(Base):
    model_ = "models/Barney.mdl"

    @property
    def ExpressionOverride(self):
        return self._raw_data.get('expressionoverride', None)



class npc_barney(PlayerCompanion, TalkNPC):
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

    @property
    def TeleportGrenades(self):
        return self._raw_data.get('teleportgrenades', "0")



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



class npc_launcher(Base):
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



class npc_hunter(Base):
    model_ = "models/hunter.mdl"

    @property
    def FollowTarget(self):
        return self._raw_data.get('followtarget', "")



class npc_hunter_maker(Base):
    icon_sprite = "editor/npc_maker.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))
    pass


class npc_advisor(Base):

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



class env_sporeexplosion(Targetname):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin', "0 0 0"))

    @property
    def spawnrate(self):
        return parse_source_value(self._raw_data.get('spawnrate', 25))



class env_gunfire(Targetname):
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



class env_headcrabcanister(Angles, Targetname):
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



class npc_vortigaunt(PlayerCompanion, TalkNPC):

    @property
    def model(self):
        return self._raw_data.get('model', "models/vortigaunt.mdl")

    @property
    def ArmorRechargeEnabled(self):
        return self._raw_data.get('armorrechargeenabled', "1")

    @property
    def HealthRegenerateEnabled(self):
        return self._raw_data.get('healthregenerateenabled', "0")



class npc_spotlight(Base):

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

    @property
    def rendercolor(self):
        return parse_int_vector(self._raw_data.get('rendercolor', "255 255 255"))




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