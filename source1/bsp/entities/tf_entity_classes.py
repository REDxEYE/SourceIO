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


class TeamNum(Base):
    def __init__(self):
        super().__init__()
        self.TeamNum = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.TeamNum = entity_data.get('teamnum', None)  # Type: choices


class MatchSummary(Base):
    def __init__(self):
        super().__init__()
        self.MatchSummary = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.MatchSummary = entity_data.get('matchsummary', None)  # Type: choices


class FadeDistance(Base):
    def __init__(self):
        super().__init__()
        self.fademindist = -1  # Type: float
        self.fademaxdist = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.fademindist = float(entity_data.get('fademindist', -1))  # Type: float
        instance.fademaxdist = float(entity_data.get('fademaxdist', 0))  # Type: float


class GameType(Base):
    def __init__(self):
        super().__init__()
        self.GameType = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.GameType = entity_data.get('gametype', None)  # Type: choices


class Condition(Base):
    def __init__(self):
        super().__init__()
        self.condition = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.condition = entity_data.get('condition', "CHOICES NOT SUPPORTED")  # Type: choices


class PlayerTouch(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class Toggle(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class tf_gamerules(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.hud_type = None  # Type: choices
        self.ctf_overtime = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.hud_type = entity_data.get('hud_type', None)  # Type: choices
        instance.ctf_overtime = entity_data.get('ctf_overtime', "CHOICES NOT SUPPORTED")  # Type: choices


class info_player_teamspawn(Targetname, MatchSummary, EnableDisable, TeamNum, Angles):
    model = "models/editor/playerstart.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(MatchSummary).__init__()
        super(EnableDisable).__init__()
        super(TeamNum).__init__()
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.controlpoint = None  # Type: target_destination
        self.SpawnMode = None  # Type: choices
        self.round_bluespawn = None  # Type: target_destination
        self.round_redspawn = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        MatchSummary.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        TeamNum.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.controlpoint = entity_data.get('controlpoint', None)  # Type: target_destination
        instance.SpawnMode = entity_data.get('spawnmode', None)  # Type: choices
        instance.round_bluespawn = entity_data.get('round_bluespawn', None)  # Type: target_destination
        instance.round_redspawn = entity_data.get('round_redspawn', None)  # Type: target_destination


class game_forcerespawn(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_teamflag(Targetname, EnableDisable, Parentname, TeamNum, Angles, GameType):
    model = "models/flag/briefcase.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(Parentname).__init__()
        super(TeamNum).__init__()
        super(Angles).__init__()
        super(GameType).__init__()
        self.origin = [0, 0, 0]
        self.ReturnTime = 60  # Type: integer
        self.NeutralType = "CHOICES NOT SUPPORTED"  # Type: choices
        self.ScoringType = None  # Type: choices
        self.flag_model = "models/flag/briefcase.mdl"  # Type: string
        self.flag_icon = "../hud/objectives_flagpanel_carried"  # Type: string
        self.flag_paper = "player_intel_papertrail"  # Type: string
        self.flag_trail = "flagtrail"  # Type: string
        self.trail_effect = "CHOICES NOT SUPPORTED"  # Type: choices
        self.VisibleWhenDisabled = None  # Type: choices
        self.ShotClockMode = None  # Type: choices
        self.PointValue = None  # Type: integer
        self.ReturnBetweenWaves = "CHOICES NOT SUPPORTED"  # Type: choices
        self.tags = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        TeamNum.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        GameType.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.ReturnTime = int(entity_data.get('returntime', 60))  # Type: integer
        instance.NeutralType = entity_data.get('neutraltype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.ScoringType = entity_data.get('scoringtype', None)  # Type: choices
        instance.flag_model = entity_data.get('flag_model', "models/flag/briefcase.mdl")  # Type: string
        instance.flag_icon = entity_data.get('flag_icon', "../hud/objectives_flagpanel_carried")  # Type: string
        instance.flag_paper = entity_data.get('flag_paper', "player_intel_papertrail")  # Type: string
        instance.flag_trail = entity_data.get('flag_trail', "flagtrail")  # Type: string
        instance.trail_effect = entity_data.get('trail_effect', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.VisibleWhenDisabled = entity_data.get('visiblewhendisabled', None)  # Type: choices
        instance.ShotClockMode = entity_data.get('shotclockmode', None)  # Type: choices
        instance.PointValue = int(entity_data.get('pointvalue', 0))  # Type: integer
        instance.ReturnBetweenWaves = entity_data.get('returnbetweenwaves', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.tags = entity_data.get('tags', None)  # Type: string


class point_intermission(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class info_observer_point(Targetname, EnableDisable, TeamNum, Angles, Parentname):
    viewport_model = "models/editor/camera.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(TeamNum).__init__()
        super(Angles).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.associated_team_entity = None  # Type: target_destination
        self.defaultwelcome = None  # Type: choices
        self.fov = None  # Type: float
        self.match_summary = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        TeamNum.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.associated_team_entity = entity_data.get('associated_team_entity', None)  # Type: target_destination
        instance.defaultwelcome = entity_data.get('defaultwelcome', None)  # Type: choices
        instance.fov = float(entity_data.get('fov', 0))  # Type: float
        instance.match_summary = entity_data.get('match_summary', None)  # Type: choices


class game_round_win(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.TeamNum = None  # Type: choices
        self.force_map_reset = "CHOICES NOT SUPPORTED"  # Type: choices
        self.switch_teams = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.TeamNum = entity_data.get('teamnum', None)  # Type: choices
        instance.force_map_reset = entity_data.get('force_map_reset', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.switch_teams = entity_data.get('switch_teams', None)  # Type: choices


class team_round_timer(Targetname, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.timer_length = 600  # Type: integer
        self.max_length = None  # Type: integer
        self.start_paused = "CHOICES NOT SUPPORTED"  # Type: choices
        self.show_time_remaining = "CHOICES NOT SUPPORTED"  # Type: choices
        self.setup_length = None  # Type: integer
        self.reset_time = None  # Type: choices
        self.auto_countdown = "CHOICES NOT SUPPORTED"  # Type: choices
        self.show_in_hud = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.timer_length = int(entity_data.get('timer_length', 600))  # Type: integer
        instance.max_length = int(entity_data.get('max_length', 0))  # Type: integer
        instance.start_paused = entity_data.get('start_paused', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.show_time_remaining = entity_data.get('show_time_remaining', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.setup_length = int(entity_data.get('setup_length', 0))  # Type: integer
        instance.reset_time = entity_data.get('reset_time', None)  # Type: choices
        instance.auto_countdown = entity_data.get('auto_countdown', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.show_in_hud = entity_data.get('show_in_hud', "CHOICES NOT SUPPORTED")  # Type: choices


class Item(Targetname, Toggle, EnableDisable, PlayerTouch, TeamNum, Angles, FadeDistance):
    def __init__(self):
        super(Targetname).__init__()
        super(Toggle).__init__()
        super(EnableDisable).__init__()
        super(PlayerTouch).__init__()
        super(TeamNum).__init__()
        super(Angles).__init__()
        super(FadeDistance).__init__()
        self.powerup_model = None  # Type: string
        self.AutoMaterialize = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Toggle.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        PlayerTouch.from_dict(instance, entity_data)
        TeamNum.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        FadeDistance.from_dict(instance, entity_data)
        instance.powerup_model = entity_data.get('powerup_model', None)  # Type: string
        instance.AutoMaterialize = entity_data.get('automaterialize', "CHOICES NOT SUPPORTED")  # Type: choices


class item_healthkit_full(Item):
    model = "models/items/medkit_large.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_healthkit_small(Item):
    model = "models/items/medkit_small.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_healthkit_medium(Item):
    model = "models/items/medkit_medium.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammopack_full(Item):
    model = "models/items/ammopack_large.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammopack_small(Item):
    model = "models/items/ammopack_small.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_ammopack_medium(Item):
    model = "models/items/ammopack_medium.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class tf_spell_pickup(Item):
    model = "models/props_halloween/hwn_spellbook_flying.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]
        self.tier = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.tier = entity_data.get('tier', None)  # Type: choices


class item_bonuspack(Targetname, Toggle, EnableDisable, PlayerTouch, TeamNum, Angles, FadeDistance):
    model = "models/crafting/moustachium.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(Toggle).__init__()
        super(EnableDisable).__init__()
        super(PlayerTouch).__init__()
        super(TeamNum).__init__()
        super(Angles).__init__()
        super(FadeDistance).__init__()
        self.origin = [0, 0, 0]
        self.powerup_model = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Toggle.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        PlayerTouch.from_dict(instance, entity_data)
        TeamNum.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        FadeDistance.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.powerup_model = entity_data.get('powerup_model', None)  # Type: string


class tf_halloween_pickup(Item, Parentname):
    model = "models/items/target_duck.mdl"
    def __init__(self):
        super(Item).__init__()
        super(Parentname).__init__()
        self.origin = [0, 0, 0]
        self.pickup_sound = None  # Type: string
        self.pickup_particle = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.pickup_sound = entity_data.get('pickup_sound', None)  # Type: string
        instance.pickup_particle = entity_data.get('pickup_particle', None)  # Type: string


class info_powerup_spawn(Targetname, EnableDisable):
    model = "models/pickups/pickup_powerup_regen.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.team = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.team = entity_data.get('team', "CHOICES NOT SUPPORTED")  # Type: choices


class item_powerup_crit(Targetname, EnableDisable):
    model = "models/pickups/pickup_powerup_crit.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class item_powerup_uber(Targetname, EnableDisable):
    model = "models/pickups/pickup_powerup_uber.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class filter_activator_tfteam(TeamNum, BaseFilter):
    icon_sprite = "editor/filter_team.vmt"
    def __init__(self):
        super(BaseFilter).__init__()
        super(TeamNum).__init__()
        self.controlpoint = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TeamNum.from_dict(instance, entity_data)
        BaseFilter.from_dict(instance, entity_data)
        instance.controlpoint = entity_data.get('controlpoint', None)  # Type: target_destination


class filter_tf_player_can_cap(TeamNum, BaseFilter):
    icon_sprite = "editor/filter_team.vmt"
    def __init__(self):
        super(BaseFilter).__init__()
        super(TeamNum).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TeamNum.from_dict(instance, entity_data)
        BaseFilter.from_dict(instance, entity_data)


class filter_tf_damaged_by_weapon_in_slot(BaseFilter):
    def __init__(self):
        super(BaseFilter).__init__()
        self.weaponSlot = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.weaponSlot = entity_data.get('weaponslot', None)  # Type: choices


class filter_tf_condition(BaseFilter, Condition):
    def __init__(self):
        super(BaseFilter).__init__()
        super(Condition).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        Condition.from_dict(instance, entity_data)


class filter_tf_class(BaseFilter):
    icon_sprite = "editor/filter_class.vmt"
    def __init__(self):
        super(BaseFilter).__init__()
        self.tfclass = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.tfclass = entity_data.get('tfclass', None)  # Type: choices


class func_capturezone(TeamNum, Targetname, EnableDisable):
    def __init__(self):
        super(TeamNum).__init__()
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.capturepoint = 1  # Type: integer
        self.capture_delay = 1.1  # Type: float
        self.capture_delay_offset = 0.025  # Type: float
        self.shouldBlock = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TeamNum.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.capturepoint = int(entity_data.get('capturepoint', 1))  # Type: integer
        instance.capture_delay = float(entity_data.get('capture_delay', 1.1))  # Type: float
        instance.capture_delay_offset = float(entity_data.get('capture_delay_offset', 0.025))  # Type: float
        instance.shouldBlock = entity_data.get('shouldblock', "CHOICES NOT SUPPORTED")  # Type: choices


class func_flagdetectionzone(Parentname, Targetname, TeamNum, EnableDisable):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(TeamNum).__init__()
        super(EnableDisable).__init__()
        self.alarm = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        TeamNum.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.alarm = entity_data.get('alarm', None)  # Type: choices


class func_nogrenades(TeamNum, Targetname, Toggle, EnableDisable):
    def __init__(self):
        super(TeamNum).__init__()
        super(Targetname).__init__()
        super(Toggle).__init__()
        super(EnableDisable).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TeamNum.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Toggle.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)


class func_achievement(TeamNum, Targetname, Toggle, EnableDisable):
    def __init__(self):
        super(TeamNum).__init__()
        super(Targetname).__init__()
        super(Toggle).__init__()
        super(EnableDisable).__init__()
        self.zone_id = None  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TeamNum.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Toggle.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.zone_id = int(entity_data.get('zone_id', 0))  # Type: integer


class func_nobuild(TeamNum, Targetname, Toggle, EnableDisable):
    def __init__(self):
        super(TeamNum).__init__()
        super(Targetname).__init__()
        super(Toggle).__init__()
        super(EnableDisable).__init__()
        self.AllowSentry = None  # Type: choices
        self.AllowDispenser = None  # Type: choices
        self.AllowTeleporters = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TeamNum.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Toggle.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.AllowSentry = entity_data.get('allowsentry', None)  # Type: choices
        instance.AllowDispenser = entity_data.get('allowdispenser', None)  # Type: choices
        instance.AllowTeleporters = entity_data.get('allowteleporters', None)  # Type: choices


class func_croc(TeamNum, Targetname, Toggle, EnableDisable):
    def __init__(self):
        super(TeamNum).__init__()
        super(Targetname).__init__()
        super(Toggle).__init__()
        super(EnableDisable).__init__()
        self.filtername = None  # Type: filterclass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TeamNum.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Toggle.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.filtername = entity_data.get('filtername', None)  # Type: filterclass


class func_suggested_build(Targetname, Toggle, EnableDisable, TeamNum, Origin):
    def __init__(self):
        super(Targetname).__init__()
        super(Toggle).__init__()
        super(EnableDisable).__init__()
        super(TeamNum).__init__()
        super(Origin).__init__()
        self.object_type = None  # Type: choices
        self.face_entity = None  # Type: target_destination
        self.face_entity_fov = 90  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Toggle.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        TeamNum.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.object_type = entity_data.get('object_type', None)  # Type: choices
        instance.face_entity = entity_data.get('face_entity', None)  # Type: target_destination
        instance.face_entity_fov = float(entity_data.get('face_entity_fov', 90))  # Type: float


class func_regenerate(TeamNum, Targetname, Toggle, EnableDisable):
    def __init__(self):
        super(TeamNum).__init__()
        super(Targetname).__init__()
        super(Toggle).__init__()
        super(EnableDisable).__init__()
        self.associatedmodel = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TeamNum.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Toggle.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.associatedmodel = entity_data.get('associatedmodel', None)  # Type: target_destination


class func_powerupvolume(Toggle, Trigger, TeamNum):
    def __init__(self):
        super(Trigger).__init__()
        super(Toggle).__init__()
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(TeamNum).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Toggle.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Trigger.from_dict(instance, entity_data)
        TeamNum.from_dict(instance, entity_data)


class func_respawnflag(Targetname, Toggle, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(Toggle).__init__()
        super(EnableDisable).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        Toggle.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)


class func_respawnroom(TeamNum, Targetname, Toggle, EnableDisable):
    def __init__(self):
        super(TeamNum).__init__()
        super(Targetname).__init__()
        super(Toggle).__init__()
        super(EnableDisable).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TeamNum.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Toggle.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)


class func_flag_alert(TeamNum, Targetname, Toggle, EnableDisable):
    def __init__(self):
        super(TeamNum).__init__()
        super(Targetname).__init__()
        super(Toggle).__init__()
        super(EnableDisable).__init__()
        self.playsound = "CHOICES NOT SUPPORTED"  # Type: choices
        self.alert_delay = 10  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TeamNum.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Toggle.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.playsound = entity_data.get('playsound', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.alert_delay = int(entity_data.get('alert_delay', 10))  # Type: integer


class func_respawnroomvisualizer(Global, Shadow, Targetname, EnableDisable, Parentname, Inputfilter, RenderFields, Origin):
    def __init__(self):
        super(RenderFields).__init__()
        super(Global).__init__()
        super(Shadow).__init__()
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(Parentname).__init__()
        super(Inputfilter).__init__()
        super(Origin).__init__()
        self.respawnroomname = None  # Type: target_destination
        self.Solidity = "CHOICES NOT SUPPORTED"  # Type: choices
        self.vrad_brush_cast_shadows = None  # Type: choices
        self.solid_to_enemies = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Global.from_dict(instance, entity_data)
        Shadow.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Inputfilter.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        instance.respawnroomname = entity_data.get('respawnroomname', None)  # Type: target_destination
        instance.Solidity = entity_data.get('solidity', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.vrad_brush_cast_shadows = entity_data.get('vrad_brush_cast_shadows', None)  # Type: choices
        instance.solid_to_enemies = entity_data.get('solid_to_enemies', "CHOICES NOT SUPPORTED")  # Type: choices


class func_forcefield(Targetname, EnableDisable, TeamNum, Parentname, RenderFields, Origin):
    def __init__(self):
        super(RenderFields).__init__()
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(TeamNum).__init__()
        super(Parentname).__init__()
        super(Origin).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        TeamNum.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        RenderFields.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)


class func_changeclass(TeamNum, Targetname, Toggle, EnableDisable):
    def __init__(self):
        super(TeamNum).__init__()
        super(Targetname).__init__()
        super(Toggle).__init__()
        super(EnableDisable).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TeamNum.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Toggle.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)


class game_intro_viewpoint(Angles):
    viewport_model = "models/editor/camera.mdl"
    def __init__(self):
        super(Angles).__init__()
        self.origin = [0, 0, 0]
        self.TeamNum = None  # Type: choices
        self.step_number = 1  # Type: integer
        self.time_delay = 12  # Type: float
        self.hint_message = None  # Type: string
        self.event_to_fire = None  # Type: string
        self.event_delay = 3  # Type: float
        self.event_data_int = None  # Type: integer
        self.fov = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.TeamNum = entity_data.get('teamnum', None)  # Type: choices
        instance.step_number = int(entity_data.get('step_number', 1))  # Type: integer
        instance.time_delay = float(entity_data.get('time_delay', 12))  # Type: float
        instance.hint_message = entity_data.get('hint_message', None)  # Type: string
        instance.event_to_fire = entity_data.get('event_to_fire', None)  # Type: string
        instance.event_delay = float(entity_data.get('event_delay', 3))  # Type: float
        instance.event_data_int = int(entity_data.get('event_data_int', 0))  # Type: integer
        instance.fov = float(entity_data.get('fov', 0))  # Type: float


class func_proprrespawnzone(Targetname):
    def __init__(self):
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)


class team_control_point(Angles, Targetname, Parentname, EnableDisable):
    model = "models/effects/cappoint_hologram.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.point_start_locked = None  # Type: choices
        self.point_printname = "TODO: Set Name"  # Type: string
        self.point_group = None  # Type: integer
        self.point_default_owner = None  # Type: choices
        self.point_index = None  # Type: integer
        self.point_warn_on_cap = None  # Type: choices
        self.point_warn_sound = "ControlPoint.CaptureWarn"  # Type: string
        self.random_owner_on_restart = None  # Type: choices
        self.team_timedpoints_2 = None  # Type: integer
        self.team_timedpoints_3 = None  # Type: integer
        self.team_capsound_0 = None  # Type: sound
        self.team_capsound_2 = None  # Type: sound
        self.team_capsound_3 = None  # Type: sound
        self.team_model_0 = "models/effects/cappoint_hologram.mdl"  # Type: studio
        self.team_model_2 = "models/effects/cappoint_hologram.mdl"  # Type: studio
        self.team_model_3 = "models/effects/cappoint_hologram.mdl"  # Type: studio
        self.team_bodygroup_0 = 3  # Type: integer
        self.team_bodygroup_2 = 1  # Type: integer
        self.team_bodygroup_3 = 1  # Type: integer
        self.team_icon_0 = "sprites/obj_icons/icon_obj_neutral"  # Type: material
        self.team_icon_2 = "sprites/obj_icons/icon_obj_red"  # Type: material
        self.team_icon_3 = "sprites/obj_icons/icon_obj_blu"  # Type: material
        self.team_overlay_0 = None  # Type: material
        self.team_overlay_2 = None  # Type: material
        self.team_overlay_3 = None  # Type: material
        self.team_previouspoint_2_0 = None  # Type: target_source
        self.team_previouspoint_2_1 = None  # Type: target_source
        self.team_previouspoint_2_2 = None  # Type: target_source
        self.team_previouspoint_3_0 = None  # Type: target_source
        self.team_previouspoint_3_1 = None  # Type: target_source
        self.team_previouspoint_3_2 = None  # Type: target_source

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.point_start_locked = entity_data.get('point_start_locked', None)  # Type: choices
        instance.point_printname = entity_data.get('point_printname', "TODO: Set Name")  # Type: string
        instance.point_group = int(entity_data.get('point_group', 0))  # Type: integer
        instance.point_default_owner = entity_data.get('point_default_owner', None)  # Type: choices
        instance.point_index = int(entity_data.get('point_index', 0))  # Type: integer
        instance.point_warn_on_cap = entity_data.get('point_warn_on_cap', None)  # Type: choices
        instance.point_warn_sound = entity_data.get('point_warn_sound', "ControlPoint.CaptureWarn")  # Type: string
        instance.random_owner_on_restart = entity_data.get('random_owner_on_restart', None)  # Type: choices
        instance.team_timedpoints_2 = int(entity_data.get('team_timedpoints_2', 0))  # Type: integer
        instance.team_timedpoints_3 = int(entity_data.get('team_timedpoints_3', 0))  # Type: integer
        instance.team_capsound_0 = entity_data.get('team_capsound_0', None)  # Type: sound
        instance.team_capsound_2 = entity_data.get('team_capsound_2', None)  # Type: sound
        instance.team_capsound_3 = entity_data.get('team_capsound_3', None)  # Type: sound
        instance.team_model_0 = entity_data.get('team_model_0', "models/effects/cappoint_hologram.mdl")  # Type: studio
        instance.team_model_2 = entity_data.get('team_model_2', "models/effects/cappoint_hologram.mdl")  # Type: studio
        instance.team_model_3 = entity_data.get('team_model_3', "models/effects/cappoint_hologram.mdl")  # Type: studio
        instance.team_bodygroup_0 = int(entity_data.get('team_bodygroup_0', 3))  # Type: integer
        instance.team_bodygroup_2 = int(entity_data.get('team_bodygroup_2', 1))  # Type: integer
        instance.team_bodygroup_3 = int(entity_data.get('team_bodygroup_3', 1))  # Type: integer
        instance.team_icon_0 = entity_data.get('team_icon_0', "sprites/obj_icons/icon_obj_neutral")  # Type: material
        instance.team_icon_2 = entity_data.get('team_icon_2', "sprites/obj_icons/icon_obj_red")  # Type: material
        instance.team_icon_3 = entity_data.get('team_icon_3', "sprites/obj_icons/icon_obj_blu")  # Type: material
        instance.team_overlay_0 = entity_data.get('team_overlay_0', None)  # Type: material
        instance.team_overlay_2 = entity_data.get('team_overlay_2', None)  # Type: material
        instance.team_overlay_3 = entity_data.get('team_overlay_3', None)  # Type: material
        instance.team_previouspoint_2_0 = entity_data.get('team_previouspoint_2_0', None)  # Type: target_source
        instance.team_previouspoint_2_1 = entity_data.get('team_previouspoint_2_1', None)  # Type: target_source
        instance.team_previouspoint_2_2 = entity_data.get('team_previouspoint_2_2', None)  # Type: target_source
        instance.team_previouspoint_3_0 = entity_data.get('team_previouspoint_3_0', None)  # Type: target_source
        instance.team_previouspoint_3_1 = entity_data.get('team_previouspoint_3_1', None)  # Type: target_source
        instance.team_previouspoint_3_2 = entity_data.get('team_previouspoint_3_2', None)  # Type: target_source


class team_control_point_round(Targetname, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.cpr_printname = None  # Type: string
        self.cpr_priority = None  # Type: integer
        self.cpr_cp_names = None  # Type: string
        self.cpr_restrict_team_cap_win = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.cpr_printname = entity_data.get('cpr_printname', None)  # Type: string
        instance.cpr_priority = int(entity_data.get('cpr_priority', 0))  # Type: integer
        instance.cpr_cp_names = entity_data.get('cpr_cp_names', None)  # Type: string
        instance.cpr_restrict_team_cap_win = entity_data.get('cpr_restrict_team_cap_win', None)  # Type: choices


class team_control_point_master(Targetname, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.team_base_icon_2 = "sprites/obj_icons/icon_base_red"  # Type: material
        self.team_base_icon_3 = "sprites/obj_icons/icon_base_blu"  # Type: material
        self.caplayout = None  # Type: string
        self.custom_position_x = -1  # Type: float
        self.custom_position_y = -1  # Type: float
        self.cpm_restrict_team_cap_win = None  # Type: choices
        self.switch_teams = None  # Type: choices
        self.score_style = None  # Type: choices
        self.play_all_rounds = None  # Type: choices
        self.partial_cap_points_rate = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.team_base_icon_2 = entity_data.get('team_base_icon_2', "sprites/obj_icons/icon_base_red")  # Type: material
        instance.team_base_icon_3 = entity_data.get('team_base_icon_3', "sprites/obj_icons/icon_base_blu")  # Type: material
        instance.caplayout = entity_data.get('caplayout', None)  # Type: string
        instance.custom_position_x = float(entity_data.get('custom_position_x', -1))  # Type: float
        instance.custom_position_y = float(entity_data.get('custom_position_y', -1))  # Type: float
        instance.cpm_restrict_team_cap_win = entity_data.get('cpm_restrict_team_cap_win', None)  # Type: choices
        instance.switch_teams = entity_data.get('switch_teams', None)  # Type: choices
        instance.score_style = entity_data.get('score_style', None)  # Type: choices
        instance.play_all_rounds = entity_data.get('play_all_rounds', None)  # Type: choices
        instance.partial_cap_points_rate = float(entity_data.get('partial_cap_points_rate', 0))  # Type: float


class trigger_capture_area(Parentname, Targetname, EnableDisable):
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.area_cap_point = None  # Type: target_source
        self.team_cancap_2 = "CHOICES NOT SUPPORTED"  # Type: choices
        self.team_cancap_3 = "CHOICES NOT SUPPORTED"  # Type: choices
        self.team_startcap_2 = 1  # Type: integer
        self.team_startcap_3 = 1  # Type: integer
        self.team_numcap_2 = 1  # Type: integer
        self.team_numcap_3 = 1  # Type: integer
        self.team_spawn_2 = None  # Type: integer
        self.team_spawn_3 = None  # Type: integer
        self.area_time_to_cap = 5  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.area_cap_point = entity_data.get('area_cap_point', None)  # Type: target_source
        instance.team_cancap_2 = entity_data.get('team_cancap_2', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.team_cancap_3 = entity_data.get('team_cancap_3', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.team_startcap_2 = int(entity_data.get('team_startcap_2', 1))  # Type: integer
        instance.team_startcap_3 = int(entity_data.get('team_startcap_3', 1))  # Type: integer
        instance.team_numcap_2 = int(entity_data.get('team_numcap_2', 1))  # Type: integer
        instance.team_numcap_3 = int(entity_data.get('team_numcap_3', 1))  # Type: integer
        instance.team_spawn_2 = int(entity_data.get('team_spawn_2', 0))  # Type: integer
        instance.team_spawn_3 = int(entity_data.get('team_spawn_3', 0))  # Type: integer
        instance.area_time_to_cap = int(entity_data.get('area_time_to_cap', 5))  # Type: integer


class team_train_watcher(TeamNum, Targetname, EnableDisable):
    def __init__(self):
        super(TeamNum).__init__()
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.train_can_recede = "CHOICES NOT SUPPORTED"  # Type: choices
        self.train_recede_time = None  # Type: integer
        self.handle_train_movement = None  # Type: choices
        self.speed_forward_modifier = 1  # Type: float
        self.env_spark_name = None  # Type: string
        self.train = None  # Type: target_destination
        self.start_node = None  # Type: target_destination
        self.goal_node = None  # Type: target_destination
        self.linked_pathtrack_1 = None  # Type: target_destination
        self.linked_cp_1 = None  # Type: target_destination
        self.linked_pathtrack_2 = None  # Type: target_destination
        self.linked_cp_2 = None  # Type: target_destination
        self.linked_pathtrack_3 = None  # Type: target_destination
        self.linked_cp_3 = None  # Type: target_destination
        self.linked_pathtrack_4 = None  # Type: target_destination
        self.linked_cp_4 = None  # Type: target_destination
        self.linked_pathtrack_5 = None  # Type: target_destination
        self.linked_cp_5 = None  # Type: target_destination
        self.linked_pathtrack_6 = None  # Type: target_destination
        self.linked_cp_6 = None  # Type: target_destination
        self.linked_pathtrack_7 = None  # Type: target_destination
        self.linked_cp_7 = None  # Type: target_destination
        self.linked_pathtrack_8 = None  # Type: target_destination
        self.linked_cp_8 = None  # Type: target_destination
        self.hud_min_speed_level_1 = 30  # Type: float
        self.hud_min_speed_level_2 = 60  # Type: float
        self.hud_min_speed_level_3 = 90  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TeamNum.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.train_can_recede = entity_data.get('train_can_recede', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.train_recede_time = int(entity_data.get('train_recede_time', 0))  # Type: integer
        instance.handle_train_movement = entity_data.get('handle_train_movement', None)  # Type: choices
        instance.speed_forward_modifier = float(entity_data.get('speed_forward_modifier', 1))  # Type: float
        instance.env_spark_name = entity_data.get('env_spark_name', None)  # Type: string
        instance.train = entity_data.get('train', None)  # Type: target_destination
        instance.start_node = entity_data.get('start_node', None)  # Type: target_destination
        instance.goal_node = entity_data.get('goal_node', None)  # Type: target_destination
        instance.linked_pathtrack_1 = entity_data.get('linked_pathtrack_1', None)  # Type: target_destination
        instance.linked_cp_1 = entity_data.get('linked_cp_1', None)  # Type: target_destination
        instance.linked_pathtrack_2 = entity_data.get('linked_pathtrack_2', None)  # Type: target_destination
        instance.linked_cp_2 = entity_data.get('linked_cp_2', None)  # Type: target_destination
        instance.linked_pathtrack_3 = entity_data.get('linked_pathtrack_3', None)  # Type: target_destination
        instance.linked_cp_3 = entity_data.get('linked_cp_3', None)  # Type: target_destination
        instance.linked_pathtrack_4 = entity_data.get('linked_pathtrack_4', None)  # Type: target_destination
        instance.linked_cp_4 = entity_data.get('linked_cp_4', None)  # Type: target_destination
        instance.linked_pathtrack_5 = entity_data.get('linked_pathtrack_5', None)  # Type: target_destination
        instance.linked_cp_5 = entity_data.get('linked_cp_5', None)  # Type: target_destination
        instance.linked_pathtrack_6 = entity_data.get('linked_pathtrack_6', None)  # Type: target_destination
        instance.linked_cp_6 = entity_data.get('linked_cp_6', None)  # Type: target_destination
        instance.linked_pathtrack_7 = entity_data.get('linked_pathtrack_7', None)  # Type: target_destination
        instance.linked_cp_7 = entity_data.get('linked_cp_7', None)  # Type: target_destination
        instance.linked_pathtrack_8 = entity_data.get('linked_pathtrack_8', None)  # Type: target_destination
        instance.linked_cp_8 = entity_data.get('linked_cp_8', None)  # Type: target_destination
        instance.hud_min_speed_level_1 = float(entity_data.get('hud_min_speed_level_1', 30))  # Type: float
        instance.hud_min_speed_level_2 = float(entity_data.get('hud_min_speed_level_2', 60))  # Type: float
        instance.hud_min_speed_level_3 = float(entity_data.get('hud_min_speed_level_3', 90))  # Type: float


class game_text_tf(Targetname):
    icon_sprite = "editor/game_text.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.message = None  # Type: string
        self.icon = None  # Type: string
        self.display_to_team = None  # Type: choices
        self.background = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.message = entity_data.get('message', None)  # Type: string
        instance.icon = entity_data.get('icon', None)  # Type: string
        instance.display_to_team = entity_data.get('display_to_team', None)  # Type: choices
        instance.background = entity_data.get('background', None)  # Type: choices


class BaseObject(Base):
    def __init__(self):
        super().__init__()
        self.TeamNum = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)
        instance.TeamNum = entity_data.get('teamnum', "CHOICES NOT SUPPORTED")  # Type: choices


class obj_dispenser(Parentname, BaseObject, Angles, Targetname):
    model = "models/buildables/dispenser_light.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(BaseObject).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.defaultupgrade = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        BaseObject.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.defaultupgrade = entity_data.get('defaultupgrade', None)  # Type: choices


class obj_sentrygun(Parentname, BaseObject, Angles, Targetname):
    model = "models/buildables/sentry3.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(BaseObject).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.defaultupgrade = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        BaseObject.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.defaultupgrade = entity_data.get('defaultupgrade', None)  # Type: choices


class obj_teleporter(Parentname, BaseObject, Angles, Targetname):
    model = "models/buildables/teleporter_light.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(BaseObject).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.defaultupgrade = None  # Type: choices
        self.teleporterType = "CHOICES NOT SUPPORTED"  # Type: choices
        self.matchingTeleporter = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        BaseObject.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.defaultupgrade = entity_data.get('defaultupgrade', None)  # Type: choices
        instance.teleporterType = entity_data.get('teleportertype', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.matchingTeleporter = entity_data.get('matchingteleporter', None)  # Type: target_destination


class bot_hint_sentrygun(Targetname, EnableDisable, Parentname, Angles, BaseObject):
    model = "models/buildables/sentry3.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(BaseObject).__init__()
        self.origin = [0, 0, 0]
        self.sequence = 5  # Type: integer
        self.sticky = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        BaseObject.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.sequence = int(entity_data.get('sequence', 5))  # Type: integer
        instance.sticky = entity_data.get('sticky', None)  # Type: choices


class bot_hint_teleporter_exit(Targetname, EnableDisable, Parentname, Angles, BaseObject):
    model = "models/buildables/teleporter_blueprint_exit.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(BaseObject).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        BaseObject.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class bot_hint_engineer_nest(Targetname, EnableDisable, Parentname, Angles, BaseObject):
    model = "models/bots/engineer/bot_engineer.mdl"
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        super(Parentname).__init__()
        super(Angles).__init__()
        super(BaseObject).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        BaseObject.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class bot_hint_sniper_spot(Parentname, BaseObject, Angles, Targetname):
    model = "models/player/sniper.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(BaseObject).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.radius = 100  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        BaseObject.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.radius = float(entity_data.get('radius', 100))  # Type: float


class mapobj_cart_dispenser(Parentname, BaseObject, Targetname):
    icon_sprite = "editor/bullseye.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(BaseObject).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.touch_trigger = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        BaseObject.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.touch_trigger = entity_data.get('touch_trigger', None)  # Type: target_destination


class dispenser_touch_trigger(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        super(Parentname).__init__()
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Trigger.from_dict(instance, entity_data)


class tf_logic_arena(Targetname, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.CapEnableDelay = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.CapEnableDelay = float(entity_data.get('capenabledelay', 0))  # Type: float


class tf_logic_competitive(Targetname, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class tf_logic_mannpower(Targetname, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class bot_controller(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.TeamNum = "CHOICES NOT SUPPORTED"  # Type: choices
        self.bot_class = None  # Type: choices
        self.bot_name = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.TeamNum = entity_data.get('teamnum', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.bot_class = entity_data.get('bot_class', None)  # Type: choices
        instance.bot_name = entity_data.get('bot_name', None)  # Type: string


class tf_logic_training_mode(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.nextMap = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.nextMap = entity_data.get('nextmap', None)  # Type: choices


class boss_alpha(Angles, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class base_boss(Parentname, Angles, Targetname, TeamNum):
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        super(TeamNum).__init__()
        self.origin = [0, 0, 0]
        self.health = 1000  # Type: integer
        self.model = "models/bots/boss_bot/boss_tank.mdl"  # Type: string
        self.speed = 75  # Type: float
        self.start_disabled = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        TeamNum.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.health = int(entity_data.get('health', 1000))  # Type: integer
        instance.model = entity_data.get('model', "models/bots/boss_bot/boss_tank.mdl")  # Type: string
        instance.speed = float(entity_data.get('speed', 75))  # Type: float
        instance.start_disabled = entity_data.get('start_disabled', None)  # Type: choices


class tank_boss(base_boss):
    def __init__(self):
        super(base_boss).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        base_boss.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class tf_logic_multiple_escort(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class tf_logic_koth(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.timer_length = 180  # Type: integer
        self.unlock_point = 30  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.timer_length = int(entity_data.get('timer_length', 180))  # Type: integer
        instance.unlock_point = int(entity_data.get('unlock_point', 30))  # Type: integer


class tf_robot_destruction_robot_spawn(Parentname, Angles, Targetname):
    model = "models/bots/bot_worker/bot_worker_a.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.health = 500  # Type: integer
        self.gibs = None  # Type: integer
        self.type = None  # Type: choices
        self.spawngroup = None  # Type: target_source
        self.startpath = None  # Type: target_source

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.health = int(entity_data.get('health', 500))  # Type: integer
        instance.gibs = int(entity_data.get('gibs', 0))  # Type: integer
        instance.type = entity_data.get('type', None)  # Type: choices
        instance.spawngroup = entity_data.get('spawngroup', None)  # Type: target_source
        instance.startpath = entity_data.get('startpath', None)  # Type: target_source


class tf_robot_destruction_spawn_group(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.respawn_time = None  # Type: float
        self.group_number = None  # Type: integer
        self.team_number = "CHOICES NOT SUPPORTED"  # Type: choices
        self.hud_icon = "../HUD/hud_bot_worker_outline_blue"  # Type: string
        self.respawn_reduction_scale = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.respawn_time = float(entity_data.get('respawn_time', 0))  # Type: float
        instance.group_number = int(entity_data.get('group_number', 0))  # Type: integer
        instance.team_number = entity_data.get('team_number', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.hud_icon = entity_data.get('hud_icon', "../HUD/hud_bot_worker_outline_blue")  # Type: string
        instance.respawn_reduction_scale = float(entity_data.get('respawn_reduction_scale', 0))  # Type: float


class RobotDestruction(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class tf_logic_robot_destruction(RobotDestruction, Targetname):
    def __init__(self):
        super(RobotDestruction).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.score_interval = 1  # Type: float
        self.loser_respawn_bonus_per_bot = None  # Type: float
        self.red_respawn_time = 10  # Type: float
        self.blue_respawn_time = 10  # Type: float
        self.max_points = 200  # Type: integer
        self.finale_length = 30  # Type: float
        self.res_file = "resource/UI/HudObjectiveRobotDestruction.res"  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        RobotDestruction.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.score_interval = float(entity_data.get('score_interval', 1))  # Type: float
        instance.loser_respawn_bonus_per_bot = float(entity_data.get('loser_respawn_bonus_per_bot', 0))  # Type: float
        instance.red_respawn_time = float(entity_data.get('red_respawn_time', 10))  # Type: float
        instance.blue_respawn_time = float(entity_data.get('blue_respawn_time', 10))  # Type: float
        instance.max_points = int(entity_data.get('max_points', 200))  # Type: integer
        instance.finale_length = float(entity_data.get('finale_length', 30))  # Type: float
        instance.res_file = entity_data.get('res_file', "resource/UI/HudObjectiveRobotDestruction.res")  # Type: string


class tf_logic_player_destruction(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.prop_model_name = "models/flag/flag.mdl"  # Type: studio
        self.prop_drop_sound = None  # Type: sound
        self.prop_pickup_sound = None  # Type: sound
        self.red_respawn_time = 10  # Type: float
        self.blue_respawn_time = 10  # Type: float
        self.min_points = 10  # Type: integer
        self.points_per_player = 5  # Type: integer
        self.finale_length = 30  # Type: float
        self.res_file = "resource/UI/HudObjectivePlayerDestruction.res"  # Type: string
        self.flag_reset_delay = 60  # Type: integer
        self.heal_distance = 450  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.prop_model_name = entity_data.get('prop_model_name', "models/flag/flag.mdl")  # Type: studio
        instance.prop_drop_sound = entity_data.get('prop_drop_sound', None)  # Type: sound
        instance.prop_pickup_sound = entity_data.get('prop_pickup_sound', None)  # Type: sound
        instance.red_respawn_time = float(entity_data.get('red_respawn_time', 10))  # Type: float
        instance.blue_respawn_time = float(entity_data.get('blue_respawn_time', 10))  # Type: float
        instance.min_points = int(entity_data.get('min_points', 10))  # Type: integer
        instance.points_per_player = int(entity_data.get('points_per_player', 5))  # Type: integer
        instance.finale_length = float(entity_data.get('finale_length', 30))  # Type: float
        instance.res_file = entity_data.get('res_file', "resource/UI/HudObjectivePlayerDestruction.res")  # Type: string
        instance.flag_reset_delay = int(entity_data.get('flag_reset_delay', 60))  # Type: integer
        instance.heal_distance = int(entity_data.get('heal_distance', 450))  # Type: integer


class trigger_rd_vault_trigger(Trigger, TeamNum):
    def __init__(self):
        super(Trigger).__init__()
        super(Targetname).__init__()
        super(TeamNum).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        TeamNum.from_dict(instance, entity_data)


class tf_logic_medieval(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class tf_logic_cp_timer(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.controlpoint = None  # Type: target_destination
        self.timer_length = 60  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.controlpoint = entity_data.get('controlpoint', None)  # Type: target_destination
        instance.timer_length = int(entity_data.get('timer_length', 60))  # Type: integer


class tf_logic_hybrid_ctf_cp(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class tf_logic_raid(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class tf_logic_boss_battle(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class tf_logic_mann_vs_machine(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class tf_logic_holiday(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.holiday_type = "CHOICES NOT SUPPORTED"  # Type: choices
        self.tauntInHell = None  # Type: choices
        self.allowHaunting = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.holiday_type = entity_data.get('holiday_type', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.tauntInHell = entity_data.get('tauntinhell', None)  # Type: choices
        instance.allowHaunting = entity_data.get('allowhaunting', None)  # Type: choices


class func_upgradestation(Targetname, EnableDisable):
    def __init__(self):
        super(Targetname).__init__()
        super(EnableDisable).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)


class bot_generator(Angles, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.team = "CHOICES NOT SUPPORTED"  # Type: choices
        self.class_ = "CHOICES NOT SUPPORTED"  # Type: choices
        self.count = 1  # Type: integer
        self.maxActive = 1  # Type: integer
        self.interval = None  # Type: float
        self.action_point = None  # Type: target_destination
        self.initial_command = None  # Type: choices
        self.suppressFire = None  # Type: choices
        self.disableDodge = None  # Type: choices
        self.actionOnDeath = "CHOICES NOT SUPPORTED"  # Type: choices
        self.spectateOnDeath = None  # Type: choices
        self.useTeamSpawnPoint = None  # Type: choices
        self.retainBuildings = None  # Type: choices
        self.difficulty = None  # Type: choices
        self.spawnOnlyWhenTriggered = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.team = entity_data.get('team', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.class_ = entity_data.get('class', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.count = int(entity_data.get('count', 1))  # Type: integer
        instance.maxActive = int(entity_data.get('maxactive', 1))  # Type: integer
        instance.interval = float(entity_data.get('interval', 0))  # Type: float
        instance.action_point = entity_data.get('action_point', None)  # Type: target_destination
        instance.initial_command = entity_data.get('initial_command', None)  # Type: choices
        instance.suppressFire = entity_data.get('suppressfire', None)  # Type: choices
        instance.disableDodge = entity_data.get('disabledodge', None)  # Type: choices
        instance.actionOnDeath = entity_data.get('actionondeath', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.spectateOnDeath = entity_data.get('spectateondeath', None)  # Type: choices
        instance.useTeamSpawnPoint = entity_data.get('useteamspawnpoint', None)  # Type: choices
        instance.retainBuildings = entity_data.get('retainbuildings', None)  # Type: choices
        instance.difficulty = entity_data.get('difficulty', None)  # Type: choices
        instance.spawnOnlyWhenTriggered = entity_data.get('spawnonlywhentriggered', None)  # Type: choices


class bot_roster(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.team = "CHOICES NOT SUPPORTED"  # Type: choices
        self.allowClassChanges = "CHOICES NOT SUPPORTED"  # Type: choices
        self.allowScout = "CHOICES NOT SUPPORTED"  # Type: choices
        self.allowSniper = "CHOICES NOT SUPPORTED"  # Type: choices
        self.allowSoldier = "CHOICES NOT SUPPORTED"  # Type: choices
        self.allowDemoman = "CHOICES NOT SUPPORTED"  # Type: choices
        self.allowMedic = "CHOICES NOT SUPPORTED"  # Type: choices
        self.allowHeavy = "CHOICES NOT SUPPORTED"  # Type: choices
        self.allowPyro = "CHOICES NOT SUPPORTED"  # Type: choices
        self.allowSpy = "CHOICES NOT SUPPORTED"  # Type: choices
        self.allowEngineer = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.team = entity_data.get('team', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.allowClassChanges = entity_data.get('allowclasschanges', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.allowScout = entity_data.get('allowscout', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.allowSniper = entity_data.get('allowsniper', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.allowSoldier = entity_data.get('allowsoldier', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.allowDemoman = entity_data.get('allowdemoman', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.allowMedic = entity_data.get('allowmedic', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.allowHeavy = entity_data.get('allowheavy', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.allowPyro = entity_data.get('allowpyro', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.allowSpy = entity_data.get('allowspy', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.allowEngineer = entity_data.get('allowengineer', "CHOICES NOT SUPPORTED")  # Type: choices


class bot_action_point(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.next_action_point = None  # Type: target_destination
        self.desired_distance = 5  # Type: float
        self.stay_time = None  # Type: float
        self.command = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.next_action_point = entity_data.get('next_action_point', None)  # Type: target_destination
        instance.desired_distance = float(entity_data.get('desired_distance', 5))  # Type: float
        instance.stay_time = float(entity_data.get('stay_time', 0))  # Type: float
        instance.command = entity_data.get('command', None)  # Type: choices


class bot_proxy(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.bot_name = "TFBot"  # Type: string
        self.team = "CHOICES NOT SUPPORTED"  # Type: choices
        self.class_ = "CHOICES NOT SUPPORTED"  # Type: choices
        self.spawn_on_start = None  # Type: choices
        self.respawn_interval = None  # Type: float
        self.action_point = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.bot_name = entity_data.get('bot_name', "TFBot")  # Type: string
        instance.team = entity_data.get('team', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.class_ = entity_data.get('class', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.spawn_on_start = entity_data.get('spawn_on_start', None)  # Type: choices
        instance.respawn_interval = float(entity_data.get('respawn_interval', 0))  # Type: float
        instance.action_point = entity_data.get('action_point', None)  # Type: target_destination


class tf_spawner(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.count = 1  # Type: integer
        self.maxActive = 1  # Type: integer
        self.interval = None  # Type: float
        self.template = None  # Type: target_destination

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.count = int(entity_data.get('count', 1))  # Type: integer
        instance.maxActive = int(entity_data.get('maxactive', 1))  # Type: integer
        instance.interval = float(entity_data.get('interval', 0))  # Type: float
        instance.template = entity_data.get('template', None)  # Type: target_destination


class tf_template_stun_drone(Angles, Targetname, EnableDisable):
    def __init__(self):
        super(Angles).__init__()
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class func_nav_blocker(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.teamToBlock = "CHOICES NOT SUPPORTED"  # Type: choices
        self.affectsFlow = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.teamToBlock = entity_data.get('teamtoblock', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.affectsFlow = entity_data.get('affectsflow', None)  # Type: choices


class func_nav_avoid(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.tags = None  # Type: string
        self.team = "CHOICES NOT SUPPORTED"  # Type: choices
        self.start_disabled = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.tags = entity_data.get('tags', None)  # Type: string
        instance.team = entity_data.get('team', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.start_disabled = entity_data.get('start_disabled', None)  # Type: choices


class func_nav_prefer(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.tags = None  # Type: string
        self.team = "CHOICES NOT SUPPORTED"  # Type: choices
        self.start_disabled = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.tags = entity_data.get('tags', None)  # Type: string
        instance.team = entity_data.get('team', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.start_disabled = entity_data.get('start_disabled', None)  # Type: choices


class func_nav_prerequisite(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        super(Targetname).__init__()
        self.Task = "CHOICES NOT SUPPORTED"  # Type: choices
        self.Entity = None  # Type: target_destination
        self.Value = None  # Type: float
        self.start_disabled = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.Task = entity_data.get('task', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.Entity = entity_data.get('entity', None)  # Type: target_destination
        instance.Value = float(entity_data.get('value', 0))  # Type: float
        instance.start_disabled = entity_data.get('start_disabled', None)  # Type: choices


class func_tfbot_hint(Origin, Targetname, EnableDisable):
    def __init__(self):
        super(Origin).__init__()
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.team = "CHOICES NOT SUPPORTED"  # Type: choices
        self.hint = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.team = entity_data.get('team', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.hint = entity_data.get('hint', None)  # Type: choices


class trigger_stun(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        super(Targetname).__init__()
        self.trigger_delay = None  # Type: float
        self.stun_duration = None  # Type: float
        self.move_speed_reduction = None  # Type: float
        self.stun_type = None  # Type: choices
        self.stun_effects = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.trigger_delay = float(entity_data.get('trigger_delay', 0))  # Type: float
        instance.stun_duration = float(entity_data.get('stun_duration', 0))  # Type: float
        instance.move_speed_reduction = float(entity_data.get('move_speed_reduction', 0))  # Type: float
        instance.stun_type = entity_data.get('stun_type', None)  # Type: choices
        instance.stun_effects = entity_data.get('stun_effects', None)  # Type: choices


class entity_spawn_point(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.spawn_manager_name = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.spawn_manager_name = entity_data.get('spawn_manager_name', None)  # Type: string


class entity_spawn_manager(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.entity_name = None  # Type: string
        self.entity_count = None  # Type: integer
        self.respawn_time = None  # Type: integer
        self.drop_to_ground = None  # Type: choices
        self.random_rotation = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.entity_name = entity_data.get('entity_name', None)  # Type: string
        instance.entity_count = int(entity_data.get('entity_count', 0))  # Type: integer
        instance.respawn_time = int(entity_data.get('respawn_time', 0))  # Type: integer
        instance.drop_to_ground = entity_data.get('drop_to_ground', None)  # Type: choices
        instance.random_rotation = entity_data.get('random_rotation', None)  # Type: choices


class tf_pumpkin_bomb(Item):
    model = "models/props_halloween/pumpkin_explode.mdl"
    def __init__(self):
        super(Item).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Item.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class tf_generic_bomb(Studiomodel, Origin, Angles, Targetname):
    def __init__(self):
        super(Studiomodel).__init__()
        super(Origin).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.damage = 50  # Type: float
        self.radius = 100  # Type: float
        self.health = 1  # Type: integer
        self.explode_particle = None  # Type: string
        self.sound = None  # Type: sound
        self.friendlyfire = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Studiomodel.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.damage = float(entity_data.get('damage', 50))  # Type: float
        instance.radius = float(entity_data.get('radius', 100))  # Type: float
        instance.health = int(entity_data.get('health', 1))  # Type: integer
        instance.explode_particle = entity_data.get('explode_particle', None)  # Type: string
        instance.sound = entity_data.get('sound', None)  # Type: sound
        instance.friendlyfire = entity_data.get('friendlyfire', None)  # Type: choices


class training_annotation(Targetname):
    model = "models/extras/info_speech.mdl"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.display_text = "<Add Text Here>"  # Type: string
        self.lifetime = None  # Type: float
        self.offset = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.display_text = entity_data.get('display_text', "<Add Text Here>")  # Type: string
        instance.lifetime = float(entity_data.get('lifetime', 0))  # Type: float
        instance.offset = float(entity_data.get('offset', 0))  # Type: float


class training_prop_dynamic(prop_dynamic):
    def __init__(self):
        super(prop_dynamic).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        prop_dynamic.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class trigger_ignite_arrows(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        super(Parentname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Parentname.from_dict(instance, entity_data)


class wheel_of_doom(Parentname, Angles, Targetname):
    model = "models/editor/cone_helper.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.effect_duration = 30  # Type: float
        self.has_spiral = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.effect_duration = float(entity_data.get('effect_duration', 30))  # Type: float
        instance.has_spiral = entity_data.get('has_spiral', None)  # Type: choices


class point_populator_interface(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class tf_point_weapon_mimic(Parentname, Angles, Targetname):
    model = "models/editor/cone_helper.mdl"
    def __init__(self):
        super(Parentname).__init__()
        super(Angles).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.WeaponType = None  # Type: choices
        self.FireSound = None  # Type: string
        self.ParticleEffect = None  # Type: string
        self.ModelOverride = None  # Type: string
        self.ModelScale = 1  # Type: float
        self.SpeedMin = 1000  # Type: float
        self.SpeedMax = 1000  # Type: float
        self.Damage = 75  # Type: float
        self.SplashRadius = 50  # Type: float
        self.SpreadAngle = 0  # Type: float
        self.Crits = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Angles.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.WeaponType = entity_data.get('weapontype', None)  # Type: choices
        instance.FireSound = entity_data.get('firesound', None)  # Type: string
        instance.ParticleEffect = entity_data.get('particleeffect', None)  # Type: string
        instance.ModelOverride = entity_data.get('modeloverride', None)  # Type: string
        instance.ModelScale = float(entity_data.get('modelscale', 1))  # Type: float
        instance.SpeedMin = float(entity_data.get('speedmin', 1000))  # Type: float
        instance.SpeedMax = float(entity_data.get('speedmax', 1000))  # Type: float
        instance.Damage = float(entity_data.get('damage', 75))  # Type: float
        instance.SplashRadius = float(entity_data.get('splashradius', 50))  # Type: float
        instance.SpreadAngle = float(entity_data.get('spreadangle', 0))  # Type: float
        instance.Crits = entity_data.get('crits', None)  # Type: choices


class tf_point_nav_interface(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class trigger_timer_door(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        self.door_name = None  # Type: string
        self.area_cap_point = None  # Type: target_source
        self.team_cancap_2 = "CHOICES NOT SUPPORTED"  # Type: choices
        self.team_cancap_3 = "CHOICES NOT SUPPORTED"  # Type: choices
        self.team_startcap_2 = 1  # Type: integer
        self.team_startcap_3 = 1  # Type: integer
        self.team_numcap_2 = 1  # Type: integer
        self.team_numcap_3 = 1  # Type: integer
        self.team_spawn_2 = None  # Type: integer
        self.team_spawn_3 = None  # Type: integer
        self.area_time_to_cap = 5  # Type: integer

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.door_name = entity_data.get('door_name', None)  # Type: string
        instance.area_cap_point = entity_data.get('area_cap_point', None)  # Type: target_source
        instance.team_cancap_2 = entity_data.get('team_cancap_2', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.team_cancap_3 = entity_data.get('team_cancap_3', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.team_startcap_2 = int(entity_data.get('team_startcap_2', 1))  # Type: integer
        instance.team_startcap_3 = int(entity_data.get('team_startcap_3', 1))  # Type: integer
        instance.team_numcap_2 = int(entity_data.get('team_numcap_2', 1))  # Type: integer
        instance.team_numcap_3 = int(entity_data.get('team_numcap_3', 1))  # Type: integer
        instance.team_spawn_2 = int(entity_data.get('team_spawn_2', 0))  # Type: integer
        instance.team_spawn_3 = int(entity_data.get('team_spawn_3', 0))  # Type: integer
        instance.area_time_to_cap = int(entity_data.get('area_time_to_cap', 5))  # Type: integer


class trigger_bot_tag(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        self.tags = None  # Type: string
        self.add = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.tags = entity_data.get('tags', None)  # Type: string
        instance.add = entity_data.get('add', "CHOICES NOT SUPPORTED")  # Type: choices


class filter_tf_bot_has_tag(BaseFilter):
    def __init__(self):
        super(BaseFilter).__init__()
        self.tags = None  # Type: string
        self.require_all_tags = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        BaseFilter.from_dict(instance, entity_data)
        instance.tags = entity_data.get('tags', None)  # Type: string
        instance.require_all_tags = entity_data.get('require_all_tags', "CHOICES NOT SUPPORTED")  # Type: choices


class trigger_add_tf_player_condition(Trigger, Condition):
    def __init__(self):
        super(Trigger).__init__()
        super(Condition).__init__()
        self.duration = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        Condition.from_dict(instance, entity_data)
        instance.duration = float(entity_data.get('duration', 0))  # Type: float


class trigger_remove_tf_player_condition(Trigger, Condition):
    def __init__(self):
        super(Trigger).__init__()
        super(Condition).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        Condition.from_dict(instance, entity_data)


class hightower_teleport_vortex(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.target_base_name = None  # Type: string
        self.lifetime = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.target_base_name = entity_data.get('target_base_name', None)  # Type: string
        instance.lifetime = float(entity_data.get('lifetime', 0))  # Type: float


class tf_zombie_spawner(Targetname):
    model = "models/bots/skeleton_sniper/skeleton_sniper.mdl"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.zombie_lifetime = None  # Type: float
        self.max_zombies = 1  # Type: integer
        self.infinite_zombies = None  # Type: choices
        self.zombie_type = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.zombie_lifetime = float(entity_data.get('zombie_lifetime', 0))  # Type: float
        instance.max_zombies = int(entity_data.get('max_zombies', 1))  # Type: integer
        instance.infinite_zombies = entity_data.get('infinite_zombies', None)  # Type: choices
        instance.zombie_type = entity_data.get('zombie_type', None)  # Type: choices


class halloween_zapper(Parentname, Targetname):
    icon_sprite = "editor/bullseye.vmt"
    def __init__(self):
        super(Parentname).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.touch_trigger = None  # Type: target_destination
        self.ParticleEffect = None  # Type: string
        self.ZapperType = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Parentname.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.touch_trigger = entity_data.get('touch_trigger', None)  # Type: target_destination
        instance.ParticleEffect = entity_data.get('particleeffect', None)  # Type: string
        instance.ZapperType = entity_data.get('zappertype', None)  # Type: choices


class trigger_player_respawn_override(Trigger, Toggle):
    def __init__(self):
        super(Trigger).__init__()
        super(Targetname).__init__()
        super(Toggle).__init__()
        super(EnableDisable).__init__()
        self.RespawnTime = -1  # Type: float
        self.RespawnName = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        Toggle.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.RespawnTime = float(entity_data.get('respawntime', -1))  # Type: float
        instance.RespawnName = entity_data.get('respawnname', None)  # Type: string


class prop_soccer_ball(Studiomodel, Targetname):
    def __init__(self):
        super(Studiomodel).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.trigger_name = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Studiomodel.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.trigger_name = entity_data.get('trigger_name', None)  # Type: string


class MiniGame(Base):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Base.from_dict(instance, entity_data)


class tf_logic_minigames(Targetname, MiniGame):
    def __init__(self):
        super(Targetname).__init__()
        super(MiniGame).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        MiniGame.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class tf_base_minigame(Targetname):
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.RedSpawn = None  # Type: target_source
        self.BlueSpawn = None  # Type: target_source
        self.InRandomPool = "CHOICES NOT SUPPORTED"  # Type: choices
        self.MaxScore = 5  # Type: integer
        self.hud_res_file = None  # Type: string
        self.your_team_score_sound = None  # Type: string
        self.enemy_team_score_sound = None  # Type: string
        self.ScoreType = None  # Type: choices
        self.SuddenDeathTime = -1  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.RedSpawn = entity_data.get('redspawn', None)  # Type: target_source
        instance.BlueSpawn = entity_data.get('bluespawn', None)  # Type: target_source
        instance.InRandomPool = entity_data.get('inrandompool', "CHOICES NOT SUPPORTED")  # Type: choices
        instance.MaxScore = int(entity_data.get('maxscore', 5))  # Type: integer
        instance.hud_res_file = entity_data.get('hud_res_file', None)  # Type: string
        instance.your_team_score_sound = entity_data.get('your_team_score_sound', None)  # Type: string
        instance.enemy_team_score_sound = entity_data.get('enemy_team_score_sound', None)  # Type: string
        instance.ScoreType = entity_data.get('scoretype', None)  # Type: choices
        instance.SuddenDeathTime = float(entity_data.get('suddendeathtime', -1))  # Type: float


class tf_halloween_minigame(tf_base_minigame):
    def __init__(self):
        super(tf_base_minigame).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.MinigameType = "CHOICES NOT SUPPORTED"  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        tf_base_minigame.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.MinigameType = entity_data.get('minigametype', "CHOICES NOT SUPPORTED")  # Type: choices


class tf_halloween_minigame_falling_platforms(tf_halloween_minigame):
    def __init__(self):
        super(tf_halloween_minigame).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        tf_halloween_minigame.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class halloween_fortune_teller(Angles, Origin, Targetname):
    model = "models/bots/merasmus/merasmas_misfortune_teller.mdl"
    def __init__(self):
        super(Angles).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.red_teleport = None  # Type: string
        self.blue_teleport = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.red_teleport = entity_data.get('red_teleport', None)  # Type: string
        instance.blue_teleport = entity_data.get('blue_teleport', None)  # Type: string


class tf_teleport_location(Angles, Origin, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class func_passtime_goal(TeamNum, Origin, Targetname, EnableDisable):
    def __init__(self):
        super(TeamNum).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.points = 1  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TeamNum.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.points = float(entity_data.get('points', 1))  # Type: float


class info_passtime_ball_spawn(TeamNum, Targetname, EnableDisable):
    icon_sprite = "editor/passtime_ball_spawner.vmt"
    def __init__(self):
        super(TeamNum).__init__()
        super(Targetname).__init__()
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TeamNum.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class passtime_logic(Targetname):
    icon_sprite = "editor/passtime_master.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]
        self.num_sections = None  # Type: integer
        self.ball_spawn_countdown = 15  # Type: integer
        self.max_pass_range = None  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.num_sections = int(entity_data.get('num_sections', 0))  # Type: integer
        instance.ball_spawn_countdown = int(entity_data.get('ball_spawn_countdown', 15))  # Type: integer
        instance.max_pass_range = float(entity_data.get('max_pass_range', 0))  # Type: float


class func_passtime_goalie_zone(TeamNum, Targetname, EnableDisable):
    def __init__(self):
        super(TeamNum).__init__()
        super(Targetname).__init__()
        super(EnableDisable).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        TeamNum.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        EnableDisable.from_dict(instance, entity_data)


class func_passtime_no_ball_zone(Targetname):
    def __init__(self):
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)


class trigger_passtime_ball(Targetname):
    def __init__(self):
        super(Targetname).__init__()

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)


class trigger_catapult(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        self.playerSpeed = 450  # Type: float
        self.physicsSpeed = 450  # Type: float
        self.useThresholdCheck = None  # Type: integer
        self.entryAngleTolerance = 0.0  # Type: float
        self.useExactVelocity = None  # Type: integer
        self.exactVelocityChoiceType = None  # Type: choices
        self.lowerThreshold = 0.15  # Type: float
        self.upperThreshold = 0.30  # Type: float
        self.launchDirection = [0.0, 0.0, 0.0]  # Type: angle
        self.launchTarget = None  # Type: target_destination
        self.onlyVelocityCheck = None  # Type: integer
        self.applyAngularImpulse = 1  # Type: integer
        self.AirCtrlSupressionTime = -1.0  # Type: float

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.playerSpeed = float(entity_data.get('playerspeed', 450))  # Type: float
        instance.physicsSpeed = float(entity_data.get('physicsspeed', 450))  # Type: float
        instance.useThresholdCheck = int(entity_data.get('usethresholdcheck', 0))  # Type: integer
        instance.entryAngleTolerance = float(entity_data.get('entryangletolerance', 0.0))  # Type: float
        instance.useExactVelocity = int(entity_data.get('useexactvelocity', 0))  # Type: integer
        instance.exactVelocityChoiceType = entity_data.get('exactvelocitychoicetype', None)  # Type: choices
        instance.lowerThreshold = float(entity_data.get('lowerthreshold', 0.15))  # Type: float
        instance.upperThreshold = float(entity_data.get('upperthreshold', 0.30))  # Type: float
        instance.launchDirection = parse_float_vector(entity_data.get('launchdirection', "0 0 0"))  # Type: angle
        instance.launchTarget = entity_data.get('launchtarget', None)  # Type: target_destination
        instance.onlyVelocityCheck = int(entity_data.get('onlyvelocitycheck', 0))  # Type: integer
        instance.applyAngularImpulse = int(entity_data.get('applyangularimpulse', 1))  # Type: integer
        instance.AirCtrlSupressionTime = float(entity_data.get('airctrlsupressiontime', -1.0))  # Type: float


class trigger_ignite(Trigger):
    def __init__(self):
        super(Trigger).__init__()
        self.burn_duration = 5  # Type: float
        self.damage_percent_per_second = 10  # Type: float
        self.ignite_particle_name = None  # Type: string
        self.ignite_sound_name = None  # Type: string

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Trigger.from_dict(instance, entity_data)
        instance.burn_duration = float(entity_data.get('burn_duration', 5))  # Type: float
        instance.damage_percent_per_second = float(entity_data.get('damage_percent_per_second', 10))  # Type: float
        instance.ignite_particle_name = entity_data.get('ignite_particle_name', None)  # Type: string
        instance.ignite_sound_name = entity_data.get('ignite_sound_name', None)  # Type: string


class tf_halloween_gift_spawn_location(Angles, Origin, Targetname):
    def __init__(self):
        super(Angles).__init__()
        super(Origin).__init__()
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Angles.from_dict(instance, entity_data)
        Origin.from_dict(instance, entity_data)
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class tf_logic_on_holiday(Targetname):
    icon_sprite = "editor/logic_auto.vmt"
    def __init__(self):
        super(Targetname).__init__()
        self.origin = [0, 0, 0]

    @staticmethod
    def from_dict(instance, entity_data: dict):
        Targetname.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))


class tf_glow(EnableDisable):
    def __init__(self):
        super(EnableDisable).__init__()
        self.origin = [0, 0, 0]
        self.targetname = None  # Type: target_source
        self.target = None  # Type: target_destination
        self.GlowColor = [255, 0, 0, 255]  # Type: color255
        self.Mode = None  # Type: choices

    @staticmethod
    def from_dict(instance, entity_data: dict):
        EnableDisable.from_dict(instance, entity_data)
        instance.origin = parse_float_vector(entity_data.get('origin', "0 0 0"))
        instance.targetname = entity_data.get('targetname', None)  # Type: target_source
        instance.target = entity_data.get('target', None)  # Type: target_destination
        instance.GlowColor = parse_int_vector(entity_data.get('glowcolor', "255 0 0 255"))  # Type: color255
        instance.Mode = entity_data.get('mode', None)  # Type: choices



entity_class_handle = {
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