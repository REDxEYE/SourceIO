from dataclasses import dataclass
from enum import auto, IntEnum
from functools import partial

from SourceIO.blender_bindings.source1.bsp.entities.base_entity_classes import Base
from SourceIO.blender_bindings.source1.bsp.entities.quake3.sof_entity_handler import RavenQ3EntityHandler
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.source1.bsp.bsp_file import RavenBSPFile
from SourceIO.library.utils import SOURCE1_HAMMER_UNIT_TO_METERS


class ItemType(IntEnum):
    BAD = auto()
    WEAPON = auto()  # EFX: rotate + upscale + minlight
    AMMO = auto()  # EFX: rotate
    ARMOR = auto()  # EFX: rotate + minlight
    HEALTH = auto()  # EFX: static external sphere + rotating internal
    POWERUP = auto()  # instant on, timer based. EFX: rotate + external ring that rotates
    HOLDABLE = auto()  # single use, holdable item. EFX: rotate + bob
    PERSISTANT_POWERUP = auto()
    TEAM = auto()


class AmmoType(IntEnum):
    NONE = auto()
    FORCE = auto()
    BLASTER = auto()
    POWERCELL = auto()
    METAL_BOLTS = auto()
    ROCKETS = auto()
    EMPLACED = auto()
    THERMAL = auto()
    TRIPMINE = auto()
    DETPACK = auto()


class WeaponType(IntEnum):
    NONE = auto()
    GAUNTLET = auto()
    MACHINEGUN = auto()
    SHOTGUN = auto()
    GRENADE_LAUNCHER = auto()
    ROCKET_LAUNCHER = auto()
    LIGHTNING = auto()
    RAILGUN = auto()
    PLASMAGUN = auto()
    BFG = auto()
    GRAPPLING_HOOK = auto()
    NAILGUN = auto()
    PROX_LAUNCHER = auto()
    CHAINGUN = auto()

    STUN_BATON = auto()
    MELEE = auto()
    SABER = auto()
    BRYAR_PISTOL = auto()
    BLASTER = auto()
    DISRUPTOR = auto()
    BOWCASTER = auto()
    REPEATER = auto()
    DEMP2 = auto()
    FLECHETTE = auto()
    THERMAL = auto()
    TRIP_MINE = auto()
    DET_PACK = auto()
    CONCUSSION = auto()
    BRYAR_OLD = auto()
    EMPLACED_GUN = auto()
    TURRET = auto()


class PowerUpType(IntEnum):
    NONE = auto()
    QUAD = auto()
    BATTLESUIT = auto()
    HASTE = auto()
    INVIS = auto()
    REGEN = auto()
    FLIGHT = auto()
    REDFLAG = auto()
    BLUEFLAG = auto()
    NEUTRALFLAG = auto()
    SCOUT = auto()
    GUARD = auto()
    DOUBLER = auto()
    AMMOREGEN = auto()
    INVULNERABILITY = auto()

    SPEEDBURST = auto()
    DISINT_4 = auto()
    SPEED = auto()
    CLOAKED = auto()
    FORCE_ENLIGHTENED_LIGHT = auto()
    FORCE_ENLIGHTENED_DARK = auto()
    FORCE_BOON = auto()
    YSALAMIRI = auto()


class HoldableType(IntEnum):
    NONE = auto()
    TELEPORTER = auto()
    MEDKIT = auto()
    KAMIKAZE = auto()
    PORTAL = auto()
    INVULNERABILITY = auto()

    SEEKER = auto()
    SHIELD = auto()
    MEDPAC = auto()
    MEDPAC_BIG = auto()
    BINOCULARS = auto()
    SENTRY_GUN = auto()
    JETPACK = auto()
    HEALTHDISP = auto()
    AMMODISP = auto()
    EWEB = auto()
    CLOAK = auto()


@dataclass
class Item:
    classname: str  # entity classname
    pickup_sound: str | None  # sound to play when item is picked up
    models: tuple[str | None, ...]  # (world model, icon model, view model, pickup model)
    icon: str | None  # icon to use in status bar
    pickup_name: str  # name to display in pickup message
    quantity: int  # for ammo how much, or duration of powerup
    item_type: ItemType  # ItemType enum
    tag: WeaponType | HoldableType | PowerUpType | int  # for weapons, WeaponType enum; for powerups, PowerUpType enum
    precaches: str  # string of all the media this item will use
    sounds: str  # string of all the sounds this item will use
    description: str  # string to display in the help menu


class StarWarsJediKnights2(RavenQ3EntityHandler):
    entity_lookup_table = RavenQ3EntityHandler.entity_lookup_table.copy()
    entity_lookup_table.update({

    })

    def __init__(self, bsp_file: RavenBSPFile, content_manager: ContentManager, parent_collection,
                 world_scale: float = SOURCE1_HAMMER_UNIT_TO_METERS, light_scale: float = 1.0):
        super().__init__(bsp_file, content_manager, parent_collection, world_scale, light_scale)

        point_entities = [
            ("target_secret", "target"),
            ("target_scriptrunner", "target"),
            ("target_counter", "target"),
            ("target_deactivate", "target"),
            ("target_activate", "target"),
            ("target_level_change", "target"),
            ("point_combat", "point"),
            ("fx_runner", "fx"),
            ("ref_tag", "ref"),
            ("waypoint", "waypoints"),
            ("waypoint_navgoal", "waypoints"),
            ("waypoint_navgoal_4", "waypoints"),
            ("waypoint_navgoal_8", "waypoints"),
            ("misc_camera", "misc"),
        ]

        for classname, category in point_entities:
            self.entity_lookup_table[classname] = Base
            setattr(self, "handle_" + classname, partial(self._handle_point_entity, classname, category))

        brush_entities = [
            ("func_breakable", "func"),
            ("func_usable", "func"),
            ("func_glass", "func"),
        ]

        for classname, category in brush_entities:
            self.entity_lookup_table[classname] = Base
            setattr(self, "handle_" + classname, partial(self._handle_brush_entity, classname, category))

        simple_model_entities = [
            ("misc_model_breakable", "misc"),
            ("misc_model_ammo_rack", "misc"),
            ("misc_model_gun_rack", "misc"),
        ]

        for classname, category in simple_model_entities:
            self.entity_lookup_table[classname] = Base
            setattr(self, "handle_" + classname, partial(self._handle_model_entity, classname, category, None))

        npc_entities = [
            ("NPC_Stormtrooper", "npc", r"models\players\stormtrooper\model.glm"),
            ("NPC_Imperial", "npc", r"models\players\imperial\model.glm"),
            ("NPC_Droid_Gonk", "npc", r"models\players\gonk\model.glm"),
            ("NPC_Prisoner", "npc", r"models\players\prisoner\model.glm"),
            ("NPC_MineMonster", "npc", r"models\players\minemonster\model.glm"),
            ("NPC_Droid_R5D2", "npc", r"models\players\r5d2\model.glm"),
            ("NPC_Kyle", "npc", r"models\players\kyle\model.glm"),
            ("NPC_Droid_Sentry", "npc", r"models\players\sentry\model.glm"),
        ]

        for classname, category, model in npc_entities:
            self.entity_lookup_table[classname] = Base
            setattr(self, "handle_" + classname, partial(self._handle_model_entity, classname, category, model))

        items = [Item(classname='item_shield_sm_instant',
                      pickup_sound='sound/player/pickupshield.wav',
                      models=('models/map_objects/mp/psd_sm.md3', None, None, None),
                      icon=None,
                      pickup_name='gfx/mp/small_shield',
                      quantity=25,
                      item_type=ItemType.ARMOR,
                      tag=1,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='item_shield_lrg_instant',
                      pickup_sound='sound/player/pickupshield.wav',
                      models=('models/map_objects/mp/psd.md3', None, None, None),
                      icon=None,
                      pickup_name='gfx/mp/large_shield',
                      quantity=100,
                      item_type=ItemType.ARMOR,
                      tag=2,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='item_medpak_instant',
                      pickup_sound='sound/player/pickuphealth.wav',
                      models=('models/map_objects/mp/medpac.md3', None, None, None),
                      icon=None,
                      pickup_name='gfx/hud/i_icon_medkit',
                      quantity=25,
                      item_type=ItemType.HEALTH,
                      tag=0,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='item_seeker',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/items/remote.md3', None, None, None),
                      icon=None,
                      pickup_name='gfx/hud/i_icon_seeker',
                      quantity=120,
                      item_type=ItemType.HOLDABLE,
                      tag=HoldableType.SEEKER,
                      precaches='',
                      sounds='',
                      description='@MENUS_AN_ATTACK_DRONE_SIMILAR'),
                 Item(classname='item_shield',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/map_objects/mp/shield.md3', None, None, None),
                      icon=None,
                      pickup_name='gfx/hud/i_icon_shieldwall',
                      quantity=120,
                      item_type=ItemType.HOLDABLE,
                      tag=HoldableType.SHIELD,
                      precaches='',
                      sounds='sound/weapons/detpack/stick.wav '
                             'sound/movers/doors/forcefield_on.wav '
                             'sound/movers/doors/forcefield_off.wav '
                             'sound/movers/doors/forcefield_lp.wav sound/effects/bumpfield.wav',
                      description='@MENUS_THIS_STATIONARY_ENERGY'),
                 Item(classname='item_medpac',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/map_objects/mp/bacta.md3', None, None, None),
                      icon=None,
                      pickup_name='gfx/hud/i_icon_bacta',
                      quantity=25,
                      item_type=ItemType.HOLDABLE,
                      tag=HoldableType.MEDPAC,
                      precaches='',
                      sounds='',
                      description='@SP_INGAME_BACTA_DESC'),
                 Item(classname='item_medpac_big',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/items/big_bacta.md3', None, None, None),
                      icon=None,
                      pickup_name='gfx/hud/i_icon_big_bacta',
                      quantity=25,
                      item_type=ItemType.HOLDABLE,
                      tag=HoldableType.MEDPAC_BIG,
                      precaches='',
                      sounds='',
                      description='@SP_INGAME_BACTA_DESC'),
                 Item(classname='item_binoculars',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/items/binoculars.md3', None, None, None),
                      icon=None,
                      pickup_name='gfx/hud/i_icon_zoom',
                      quantity=60,
                      item_type=ItemType.HOLDABLE,
                      tag=HoldableType.BINOCULARS,
                      precaches='',
                      sounds='',
                      description='@SP_INGAME_LA_GOGGLES_DESC'),
                 Item(classname='item_sentry_gun',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/items/psgun.glm', None, None, None),
                      icon=None,
                      pickup_name='gfx/hud/i_icon_sentrygun',
                      quantity=120,
                      item_type=ItemType.HOLDABLE,
                      tag=HoldableType.SENTRY_GUN,
                      precaches='',
                      sounds='',
                      description='@MENUS_THIS_DEADLY_WEAPON_IS'),
                 Item(classname='item_jetpack',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/items/psgun.glm', None, None, None),
                      icon=None,
                      pickup_name='gfx/hud/i_icon_jetpack',
                      quantity=120,
                      item_type=ItemType.HOLDABLE,
                      tag=HoldableType.JETPACK,
                      precaches='effects/boba/jet.efx',
                      sounds='sound/chars/boba/JETON.wav sound/chars/boba/JETHOVER.wav '
                             'sound/effects/fire_lp.wav',
                      description='@MENUS_JETPACK_DESC'),
                 Item(classname='item_healthdisp',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/map_objects/mp/bacta.md3', None, None, None),
                      icon=None,
                      pickup_name='gfx/hud/i_icon_healthdisp',
                      quantity=120,
                      item_type=ItemType.HOLDABLE,
                      tag=HoldableType.HEALTHDISP,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='item_ammodisp',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/map_objects/mp/bacta.md3', None, None, None),
                      icon=None,
                      pickup_name='gfx/hud/i_icon_ammodisp',
                      quantity=120,
                      item_type=ItemType.HOLDABLE,
                      tag=HoldableType.AMMODISP,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='item_eweb_holdable',
                      pickup_sound='sound/interface/shieldcon_empty',
                      models=('models/map_objects/hoth/eweb_model.glm', None, None, None),
                      icon=None,
                      pickup_name='gfx/hud/i_icon_eweb',
                      quantity=120,
                      item_type=ItemType.HOLDABLE,
                      tag=HoldableType.EWEB,
                      precaches='',
                      sounds='',
                      description='@MENUS_EWEB_DESC'),
                 Item(classname='item_cloak',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/items/psgun.glm', None, None, None),
                      icon=None,
                      pickup_name='gfx/hud/i_icon_cloak',
                      quantity=120,
                      item_type=ItemType.HOLDABLE,
                      tag=HoldableType.CLOAK,
                      precaches='',
                      sounds='',
                      description='@MENUS_CLOAK_DESC'),
                 Item(classname='item_force_enlighten_light',
                      pickup_sound='sound/player/enlightenment.wav',
                      models=('models/map_objects/mp/jedi_enlightenment.md3', None, None, None),
                      icon=None,
                      pickup_name='gfx/hud/mpi_jlight',
                      quantity=25,
                      item_type=ItemType.POWERUP,
                      tag=PowerUpType.FORCE_ENLIGHTENED_LIGHT,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='item_force_enlighten_dark',
                      pickup_sound='sound/player/enlightenment.wav',
                      models=('models/map_objects/mp/dk_enlightenment.md3', None, None, None),
                      icon=None,
                      pickup_name='gfx/hud/mpi_dklight',
                      quantity=25,
                      item_type=ItemType.POWERUP,
                      tag=PowerUpType.FORCE_ENLIGHTENED_DARK,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='item_force_boon',
                      pickup_sound='sound/player/boon.wav',
                      models=('models/map_objects/mp/force_boon.md3', None, None, None),
                      icon=None,
                      pickup_name='gfx/hud/mpi_fboon',
                      quantity=25,
                      item_type=ItemType.POWERUP,
                      tag=PowerUpType.FORCE_BOON,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='item_ysalimari',
                      pickup_sound='sound/player/ysalimari.wav',
                      models=('models/map_objects/mp/ysalimari.md3', None, None, None),
                      icon=None,
                      pickup_name='gfx/hud/mpi_ysamari',
                      quantity=25,
                      item_type=ItemType.POWERUP,
                      tag=PowerUpType.YSALAMIRI,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='weapon_stun_baton',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/stun_baton/baton_w.glm', None, None, None),
                      icon='models/weapons2/stun_baton/baton.md3',
                      pickup_name='gfx/hud/w_icon_stunbaton',
                      quantity=100,
                      item_type=ItemType.WEAPON,
                      tag=WeaponType.STUN_BATON,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='weapon_melee',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/stun_baton/baton_w.glm', None, None, None),
                      icon='models/weapons2/stun_baton/baton.md3',
                      pickup_name='gfx/hud/w_icon_melee',
                      quantity=100,
                      item_type=ItemType.WEAPON,
                      tag=WeaponType.MELEE,
                      precaches='',
                      sounds='',
                      description='@MENUS_MELEE_DESC'),
                 Item(classname='weapon_saber',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/saber/saber_w.glm', None, None, None),
                      icon='models/weapons2/saber/saber_w.md3',
                      pickup_name='gfx/hud/w_icon_lightsaber',
                      quantity=100,
                      item_type=ItemType.WEAPON,
                      tag=WeaponType.SABER,
                      precaches='',
                      sounds='',
                      description='@MENUS_AN_ELEGANT_WEAPON_FOR'),
                 Item(classname='weapon_blaster_pistol',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/blaster_pistol/blaster_pistol_w.glm',
                              None,
                              None,
                              None),
                      icon='models/weapons2/blaster_pistol/blaster_pistol.md3',
                      pickup_name='gfx/hud/w_icon_blaster_pistol',
                      quantity=100,
                      item_type=ItemType.WEAPON,
                      tag=WeaponType.BRYAR_PISTOL,
                      precaches='',
                      sounds='',
                      description='@MENUS_BLASTER_PISTOL_DESC'),
                 Item(classname='weapon_concussion_rifle',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/concussion/c_rifle_w.glm', None, None, None),
                      icon='models/weapons2/concussion/c_rifle.md3',
                      pickup_name='gfx/hud/w_icon_c_rifle',
                      quantity=50,
                      item_type=ItemType.WEAPON,
                      tag=WeaponType.CONCUSSION,
                      precaches='',
                      sounds='',
                      description='@MENUS_CONC_RIFLE_DESC'),
                 Item(classname='weapon_bryar_pistol',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/briar_pistol/briar_pistol_w.glm',
                              None,
                              None,
                              None),
                      icon='models/weapons2/briar_pistol/briar_pistol.md3',
                      pickup_name='gfx/hud/w_icon_briar',
                      quantity=100,
                      item_type=ItemType.WEAPON,
                      tag=WeaponType.BRYAR_OLD,
                      precaches='',
                      sounds='',
                      description='@SP_INGAME_BLASTER_PISTOL'),
                 Item(classname='weapon_blaster',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/blaster_r/blaster_w.glm', None, None, None),
                      icon='models/weapons2/blaster_r/blaster.md3',
                      pickup_name='gfx/hud/w_icon_blaster',
                      quantity=100,
                      item_type=ItemType.WEAPON,
                      tag=WeaponType.BLASTER,
                      precaches='',
                      sounds='',
                      description='@MENUS_THE_PRIMARY_WEAPON_OF'),
                 Item(classname='weapon_disruptor',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/disruptor/disruptor_w.glm', None, None, None),
                      icon='models/weapons2/disruptor/disruptor.md3',
                      pickup_name='gfx/hud/w_icon_disruptor',
                      quantity=100,
                      item_type=ItemType.WEAPON,
                      tag=WeaponType.DISRUPTOR,
                      precaches='',
                      sounds='',
                      description='@MENUS_THIS_NEFARIOUS_WEAPON'),
                 Item(classname='weapon_bowcaster',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/bowcaster/bowcaster_w.glm', None, None, None),
                      icon='models/weapons2/bowcaster/bowcaster.md3',
                      pickup_name='gfx/hud/w_icon_bowcaster',
                      quantity=100,
                      item_type=ItemType.WEAPON,
                      tag=WeaponType.BOWCASTER,
                      precaches='',
                      sounds='',
                      description='@MENUS_THIS_ARCHAIC_LOOKING'),
                 Item(classname='weapon_repeater',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/heavy_repeater/heavy_repeater_w.glm',
                              None,
                              None,
                              None),
                      icon='models/weapons2/heavy_repeater/heavy_repeater.md3',
                      pickup_name='gfx/hud/w_icon_repeater',
                      quantity=100,
                      item_type=ItemType.WEAPON,
                      tag=WeaponType.REPEATER,
                      precaches='',
                      sounds='',
                      description='@MENUS_THIS_DESTRUCTIVE_PROJECTILE'),
                 Item(classname='weapon_demp2',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/demp2/demp2_w.glm', None, None, None),
                      icon='models/weapons2/demp2/demp2.md3',
                      pickup_name='gfx/hud/w_icon_demp2',
                      quantity=100,
                      item_type=ItemType.WEAPON,
                      tag=WeaponType.DEMP2,
                      precaches='',
                      sounds='',
                      description='@MENUS_COMMONLY_REFERRED_TO'),
                 Item(classname='weapon_flechette',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/golan_arms/golan_arms_w.glm', None, None, None),
                      icon='models/weapons2/golan_arms/golan_arms.md3',
                      pickup_name='gfx/hud/w_icon_flechette',
                      quantity=100,
                      item_type=ItemType.WEAPON,
                      tag=WeaponType.FLECHETTE,
                      precaches='',
                      sounds='',
                      description='@MENUS_WIDELY_USED_BY_THE_CORPORATE'),
                 Item(classname='weapon_rocket_launcher',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/merr_sonn/merr_sonn_w.glm', None, None, None),
                      icon='models/weapons2/merr_sonn/merr_sonn.md3',
                      pickup_name='gfx/hud/w_icon_merrsonn',
                      quantity=3,
                      item_type=ItemType.WEAPON,
                      tag=WeaponType.ROCKET_LAUNCHER,
                      precaches='',
                      sounds='',
                      description='@MENUS_THE_PLX_2M_IS_AN_EXTREMELY'),
                 Item(classname='ammo_thermal',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/thermal/thermal_pu.md3',
                              'models/weapons2/thermal/thermal_w.glm',
                              None,
                              None),
                      icon='models/weapons2/thermal/thermal.md3',
                      pickup_name='gfx/hud/w_icon_thermal',
                      quantity=4,
                      item_type=ItemType.AMMO,
                      tag=AmmoType.THERMAL,
                      precaches='',
                      sounds='',
                      description='@MENUS_THE_THERMAL_DETONATOR'),
                 Item(classname='ammo_tripmine',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/laser_trap/laser_trap_pu.md3',
                              'models/weapons2/laser_trap/laser_trap_w.glm',
                              None,
                              None),
                      icon='models/weapons2/laser_trap/laser_trap.md3',
                      pickup_name='gfx/hud/w_icon_tripmine',
                      quantity=3,
                      item_type=ItemType.AMMO,
                      tag=AmmoType.TRIPMINE,
                      precaches='',
                      sounds='',
                      description='@MENUS_TRIP_MINES_CONSIST_OF'),
                 Item(classname='ammo_detpack',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/detpack/det_pack_pu.md3',
                              'models/weapons2/detpack/det_pack_proj.glm',
                              'models/weapons2/detpack/det_pack_w.glm',
                              None),
                      icon='models/weapons2/detpack/det_pack.md3',
                      pickup_name='gfx/hud/w_icon_detpack',
                      quantity=3,
                      item_type=ItemType.AMMO,
                      tag=AmmoType.DETPACK,
                      precaches='',
                      sounds='',
                      description='@MENUS_A_DETONATION_PACK_IS'),
                 Item(classname='weapon_thermal',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/thermal/thermal_w.glm',
                              'models/weapons2/thermal/thermal_pu.md3',
                              None,
                              None),
                      icon='models/weapons2/thermal/thermal.md3',
                      pickup_name='gfx/hud/w_icon_thermal',
                      quantity=4,
                      item_type=ItemType.WEAPON,
                      tag=WeaponType.THERMAL,
                      precaches='',
                      sounds='',
                      description='@MENUS_THE_THERMAL_DETONATOR'),
                 Item(classname='weapon_trip_mine',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/laser_trap/laser_trap_w.glm',
                              'models/weapons2/laser_trap/laser_trap_pu.md3',
                              None,
                              None),
                      icon='models/weapons2/laser_trap/laser_trap.md3',
                      pickup_name='gfx/hud/w_icon_tripmine',
                      quantity=3,
                      item_type=ItemType.WEAPON,
                      tag=WeaponType.TRIP_MINE,
                      precaches='',
                      sounds='',
                      description='@MENUS_TRIP_MINES_CONSIST_OF'),
                 Item(classname='weapon_det_pack',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/detpack/det_pack_proj.glm',
                              'models/weapons2/detpack/det_pack_pu.md3',
                              'models/weapons2/detpack/det_pack_w.glm',
                              None),
                      icon='models/weapons2/detpack/det_pack.md3',
                      pickup_name='gfx/hud/w_icon_detpack',
                      quantity=3,
                      item_type=ItemType.WEAPON,
                      tag=WeaponType.DET_PACK,
                      precaches='',
                      sounds='',
                      description='@MENUS_A_DETONATION_PACK_IS'),
                 Item(classname='weapon_emplaced',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/blaster_r/blaster_w.glm', None, None, None),
                      icon='models/weapons2/blaster_r/blaster.md3',
                      pickup_name='gfx/hud/w_icon_blaster',
                      quantity=50,
                      item_type=ItemType.WEAPON,
                      tag=WeaponType.EMPLACED_GUN,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='weapon_turretwp',
                      pickup_sound='sound/weapons/w_pkup.wav',
                      models=('models/weapons2/blaster_r/blaster_w.glm', None, None, None),
                      icon='models/weapons2/blaster_r/blaster.md3',
                      pickup_name='gfx/hud/w_icon_blaster',
                      quantity=50,
                      item_type=ItemType.WEAPON,
                      tag=WeaponType.TURRET,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='ammo_force',
                      pickup_sound='sound/player/pickupenergy.wav',
                      models=('models/items/energy_cell.md3', None, None, None),
                      icon=None,
                      pickup_name='gfx/hud/w_icon_blaster',
                      quantity=100,
                      item_type=ItemType.AMMO,
                      tag=AmmoType.FORCE,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='ammo_blaster',
                      pickup_sound='sound/player/pickupenergy.wav',
                      models=('models/items/energy_cell.md3', None, None, None),
                      icon=None,
                      pickup_name='gfx/hud/i_icon_battery',
                      quantity=100,
                      item_type=ItemType.AMMO,
                      tag=AmmoType.BLASTER,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='ammo_powercell',
                      pickup_sound='sound/player/pickupenergy.wav',
                      models=('models/items/power_cell.md3', None, None, None),
                      icon=None,
                      pickup_name='gfx/mp/ammo_power_cell',
                      quantity=100,
                      item_type=ItemType.AMMO,
                      tag=AmmoType.POWERCELL,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='ammo_metallic_bolts',
                      pickup_sound='sound/player/pickupenergy.wav',
                      models=('models/items/metallic_bolts.md3', None, None, None),
                      icon=None,
                      pickup_name='gfx/mp/ammo_metallic_bolts',
                      quantity=100,
                      item_type=ItemType.AMMO,
                      tag=AmmoType.METAL_BOLTS,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='ammo_rockets',
                      pickup_sound='sound/player/pickupenergy.wav',
                      models=('models/items/rockets.md3', None, None, None),
                      icon=None,
                      pickup_name='gfx/mp/ammo_rockets',
                      quantity=3,
                      item_type=ItemType.AMMO,
                      tag=AmmoType.ROCKETS,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='ammo_all',
                      pickup_sound='sound/player/pickupenergy.wav',
                      models=('models/items/battery.md3', None, None, None),
                      icon=None,
                      pickup_name='gfx/mp/ammo_rockets',
                      quantity=0,
                      item_type=ItemType.AMMO,
                      tag=-1,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='team_CTF_redflag',
                      pickup_sound=None,
                      models=('models/flags/r_flag.md3',
                              'models/flags/r_flag_ysal.md3',
                              None,
                              None),
                      icon=None,
                      pickup_name='gfx/hud/mpi_rflag',
                      quantity=0,
                      item_type=ItemType.TEAM,
                      tag=PowerUpType.REDFLAG,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='team_CTF_blueflag',
                      pickup_sound=None,
                      models=('models/flags/b_flag.md3',
                              'models/flags/b_flag_ysal.md3',
                              None,
                              None),
                      icon=None,
                      pickup_name='gfx/hud/mpi_bflag',
                      quantity=0,
                      item_type=ItemType.TEAM,
                      tag=PowerUpType.BLUEFLAG,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='team_CTF_neutralflag',
                      pickup_sound=None,
                      models=('models/flags/n_flag.md3', None, None, None),
                      icon=None,
                      pickup_name='icons/iconf_neutral1',
                      quantity=0,
                      item_type=ItemType.TEAM,
                      tag=PowerUpType.NEUTRALFLAG,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='item_redcube',
                      pickup_sound='sound/player/pickupenergy.wav',
                      models=('models/powerups/orb/r_orb.md3', None, None, None),
                      icon=None,
                      pickup_name='icons/iconh_rorb',
                      quantity=0,
                      item_type=ItemType.TEAM,
                      tag=0,
                      precaches='',
                      sounds='',
                      description=''),
                 Item(classname='item_bluecube',
                      pickup_sound='sound/player/pickupenergy.wav',
                      models=('models/powerups/orb/b_orb.md3', None, None, None),
                      icon=None,
                      pickup_name='icons/iconh_borb',
                      quantity=0,
                      item_type=ItemType.TEAM,
                      tag=0,
                      precaches='',
                      sounds='',
                      description='')]

        for item in items:
            self.entity_lookup_table[item.classname] = Base
            setattr(self, "handle_" + item.classname, partial(self._handle_model_entity, item.classname, "item", item.models[0]))