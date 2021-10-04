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
    'monster_generic': handle_generic_model_prop,
    'cycler': handle_generic_model_prop,
    'cycler_sprite': handle_generic_model_prop,
    'env_model': handle_generic_model_prop,
}
