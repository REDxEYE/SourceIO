import math
from functools import partial
from pathlib import Path

import bpy

from ....library.shared.content_providers.content_manager import ContentManager
from ....library.utils.math_utilities import parse_hammer_vector
from ...utils.utils import get_new_unique_collection

content_manager = ContentManager()


def handle_generic_model_prop(entity_data, scale, parent_collection, fix_rotation=True):
    model_name = Path(entity_data['model'])
    return handle_model_prop(model_name, entity_data, scale, parent_collection, fix_rotation=fix_rotation)


def handle_model_prop(model_name, entity_data, scale, parent_collection, fix_rotation=True):
    from .. import import_model
    origin = parse_hammer_vector(entity_data.get('origin', '0 0 0')) * scale
    angles = [math.radians(a) for a in parse_hammer_vector(entity_data.get('angles', '0 0 0'))]

    if fix_rotation:
        x, y, z = angles
        y += math.pi / 2
        angles = [x, z, y]
    if not angles:
        angles = 0.0, math.radians(entity_data.get('angle', '0')), 0.0
    target_name = entity_data.get('targetname', entity_data['classname'])
    model_path = content_manager.find_file(str(model_name))
    if model_path:
        master_collection = get_new_unique_collection(target_name, parent_collection)
        model_texture_path = content_manager.find_file(str(model_name.with_name(model_name.stem + 't.mdl')))
        model_container = import_model(model_path,
                                       model_texture_path if model_texture_path is not None else None, scale,
                                       master_collection, disable_collection_sort=True, re_use_meshes=True)
        if model_container.armature:
            model_container.armature.location = origin
            model_container.armature.rotation_euler = angles
        else:
            for o in model_container.objects:
                o.location = origin
                o.rotation_euler = angles
        entity_data_holder = bpy.data.objects.new(target_name, None)
        entity_data_holder.location = origin
        entity_data_holder.rotation_euler = angles
        entity_data_holder.scale *= scale
        entity_data_holder['entity_data'] = {'entity': entity_data}
        master_collection.objects.link(entity_data_holder)
    pass


entity_handlers = {
    'monster_scientist': partial(handle_model_prop, Path('models/scientist.mdl')),
    'monster_sitting_scientist': partial(handle_model_prop, Path('models/scientist.mdl')),
    'monster_barney': partial(handle_model_prop, Path('models/barney.mdl')),
    'monster_cine_barney': partial(handle_model_prop, Path('models/cine-barney.mdl')),
    'monster_cine_panther': partial(handle_model_prop, Path('models/cine-panther.mdl')),
    'monster_cine_scientist': partial(handle_model_prop, Path('models/cine-scientist.mdl')),
    'monster_gman': partial(handle_model_prop, Path('models/gman.mdl')),
    'monster_faceless': partial(handle_model_prop, Path('models/Faceless.mdl')),
    'monster_polyrobo': partial(handle_model_prop, Path('models/polyrobo.mdl')),
    'monster_boid': partial(handle_model_prop, Path('models/boid.mdl')),
    'monster_boid_flock': partial(handle_model_prop, Path('models/boid.mdl')),
    'item_battery': partial(handle_model_prop, Path('models/w_battery.mdl')),
    'item_healthkit': partial(handle_model_prop, Path('models/w_medkit.mdl')),
    'weapon_crossbow': partial(handle_model_prop, Path('models/w_crossbow.mdl')),
    'ammo_crossbow': partial(handle_model_prop, Path('models/w_crossbow_clip.mdl')),
    'ammo_buckshot': partial(handle_model_prop, Path('models/w_shotbox.mdl')),
    'ammo_gaussclip': partial(handle_model_prop, Path('models/w_gaussammo.mdl')),
    'ammo_rpgclip': partial(handle_model_prop, Path('models/w_rpgammo.mdl')),
    'weapon_rpg': partial(handle_model_prop, Path('models/w_rpg.mdl')),
    'weapon_9mmAR': partial(handle_model_prop, Path('models/w_9mmAR.mdl')),
    'weapon_snark': partial(handle_model_prop, Path('models/w_sqknest.mdl')),
    'weapon_gauss': partial(handle_model_prop, Path('models/w_gauss.mdl')),
    'ammo_9mmAR': partial(handle_model_prop, Path('models/w_9mmARclip.mdl')),
    'item_longjump': partial(handle_model_prop, Path('models/w_longjump.mdl')),
    'weapon_handgrenade': partial(handle_model_prop, Path('models/w_grenade.mdl')),
    'ammo_ARgrenades': partial(handle_model_prop, Path('models/w_ARgrenade.mdl')),
    'weapon_egon': partial(handle_model_prop, Path('models/w_egon.mdl')),
    'weapon_hornetgun': partial(handle_model_prop, Path('models/w_hgun.mdl')),
    'weapon_357': partial(handle_model_prop, Path('models/w_357.mdl')),
    'ammo_357': partial(handle_model_prop, Path('models/w_357ammobox.mdl')),
    'weapon_satchel': partial(handle_model_prop, Path('models/w_satchel.mdl')),
    'weapon_shotgun': partial(handle_model_prop, Path('models/w_shotgun.mdl')),

    'ammo_9mmbox': partial(handle_model_prop, Path('models/w_chainammo.mdl')),
    'ammo_9mmclip': partial(handle_model_prop, Path('models/w_9mmclip.mdl')),
    'ammo_egonclip': partial(handle_model_prop, Path('models/w_chainammo.mdl')),
    'ammo_glockclip': partial(handle_model_prop, Path('models/w_9mmclip.mdl')),
    'ammo_mp5clip': partial(handle_model_prop, Path('models/w_9mmARclip.mdl')),
    'ammo_mp5grenades': partial(handle_model_prop, Path('models/w_ARgrenade.mdl')),
    'crossbow_bolt': partial(handle_model_prop, Path('models/crossbow_bolt.mdl')),
    'cycler_prdroid': partial(handle_model_prop, Path('models/prdroid.mdl')),
    'grenade': partial(handle_model_prop, Path('models/grenade.mdl')),
    'hornet': partial(handle_model_prop, Path('models/hornet.mdl')),
    'hvr_rocket': partial(handle_model_prop, Path('models/HVR.mdl')),
    'item_airtank': partial(handle_model_prop, Path('models/w_oxygen.mdl')),
    'item_antidote': partial(handle_model_prop, Path('models/w_antidote.mdl')),
    'item_security': partial(handle_model_prop, Path('models/w_security.mdl')),
    'item_sodacan': partial(handle_model_prop, Path('models/can.mdl')),
    'item_suit': partial(handle_model_prop, Path('models/w_suit.mdl')),
    'monster_alien_grunt': partial(handle_model_prop, Path('models/agrunt.mdl')),
    'monster_alien_slave': partial(handle_model_prop, Path('models/islave.mdl')),
    'monster_apache': partial(handle_model_prop, Path('models/apache.mdl')),
    'monster_babycrab': partial(handle_model_prop, Path('models/baby_headcrab.mdl')),
    'monster_barnacle': partial(handle_model_prop, Path('models/barnacle.mdl')),
    'monster_barney_dead': partial(handle_model_prop, Path('models/barney.mdl')),
    'monster_bigmomma': partial(handle_model_prop, Path('models/big_mom.mdl')),
    'monster_bloater': partial(handle_model_prop, Path('models/floater.mdl')),
    'monster_bullchicken': partial(handle_model_prop, Path('models/bullsquid.mdl')),
    'monster_cine2_hvyweapons': partial(handle_model_prop, Path('models/cine2_hvyweapons.mdl')),
    'monster_cine2_scientist': partial(handle_model_prop, Path('models/cine2-scientist.mdl')),
    'monster_cine2_slave': partial(handle_model_prop, Path('models/cine2_slave.mdl')),
    'monster_cine3_barney': partial(handle_model_prop, Path('models/cine3-barney.mdl')),
    'monster_cine3_scientist': partial(handle_model_prop, Path('models/cine3-scientist.mdl')),
    'monster_cockroach': partial(handle_model_prop, Path('models/roach.mdl')),
    'monster_flyer': partial(handle_model_prop, Path('models/boid.mdl')),
    'monster_gargantua': partial(handle_model_prop, Path('models/garg.mdl')),
    'monster_headcrab': partial(handle_model_prop, Path('models/headcrab.mdl')),
    'monster_hevsuit_dead': partial(handle_model_prop, Path('models/player.mdl')),
    'monster_hgrunt_dead': partial(handle_model_prop, Path('models/hgrunt.mdl')),
    'monster_houndeye': partial(handle_model_prop, Path('models/houndeye.mdl')),
    'monster_human_assassin': partial(handle_model_prop, Path('models/hassassin.mdl')),
    'monster_human_grunt': partial(handle_model_prop, Path('models/hgrunt.mdl')),
    'monster_ichthyosaur': partial(handle_model_prop, Path('models/icky.mdl')),
    'monster_leech': partial(handle_model_prop, Path('models/leech.mdl')),
    'monster_miniturret': partial(handle_model_prop, Path('models/miniturret.mdl')),
    'monster_nihilanth': partial(handle_model_prop, Path('models/nihilanth.mdl')),
    'monster_osprey': partial(handle_model_prop, Path('models/osprey.mdl')),
    'monster_rat': partial(handle_model_prop, Path('models/bigrat.mdl')),
    'monster_satchel': partial(handle_model_prop, Path('models/w_satchel.mdl')),
    'monster_scientist_dead': partial(handle_model_prop, Path('models/scientist.mdl')),
    'monster_sentry': partial(handle_model_prop, Path('models/sentry.mdl')),
    'monster_snark': partial(handle_model_prop, Path('models/w_squeak.mdl')),
    'monster_tentacle': partial(handle_model_prop, Path('models/tentacle2.mdl')),
    'monster_tentaclemaw': partial(handle_model_prop, Path('models/maw.mdl')),
    'monster_tripmine': partial(handle_model_prop, Path('models/v_tripmine.mdl')),
    'monster_vortigaunt': partial(handle_model_prop, Path('models/islave.mdl')),
    'monster_zombie': partial(handle_model_prop, Path('models/zombie.mdl')),
    'player': partial(handle_model_prop, Path('models/player.mdl')),
    'rpg_rocket': partial(handle_model_prop, Path('models/rpgrocket.mdl')),
    'spark_shower': partial(handle_model_prop, Path('models/grenade.mdl')),
    'weapon_9mmhandgun': partial(handle_model_prop, Path('models/w_9mmhandgun.mdl')),
    'weapon_crowbar': partial(handle_model_prop, Path('models/w_crowbar.mdl')),
    'weapon_glock': partial(handle_model_prop, Path('models/w_9mmhandgun.mdl')),
    'weapon_mp5': partial(handle_model_prop, Path('models/w_9mmAR.mdl')),
    'weapon_python': partial(handle_model_prop, Path('models/w_357.mdl')),
    'weapon_tripmine': partial(handle_model_prop, Path('models/v_tripmine.mdl')),
    'weaponbox': partial(handle_model_prop, Path('models/w_weaponbox.mdl')),
    'xen_hair': partial(handle_model_prop, Path('models/hair.mdl')),
    'xen_plantlight': partial(handle_model_prop, Path('models/light.mdl')),
    'xen_tree': partial(handle_model_prop, Path('models/tree.mdl')),

    'monster_generic': handle_generic_model_prop,
    'cycler': handle_generic_model_prop,
    'cycler_sprite': handle_generic_model_prop,
    'env_model': handle_generic_model_prop,
}
