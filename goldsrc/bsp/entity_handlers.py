from functools import partial
from pathlib import Path

import bpy

from .mgr import GoldSrcContentManager
from ..mdl.import_mdl import import_model
from ...bpy_utilities.utils import get_new_unique_collection
from ...utilities.math_utilities import parse_hammer_vector

content_manager = GoldSrcContentManager()


def handle_generic_model_prop(entity_data, scale, parent_collection):
    model_name = Path(entity_data['model'])
    return handle_model_prop(model_name, entity_data, scale, parent_collection)


def handle_model_prop(model_name, entity_data, scale, parent_collection):
    origin = parse_hammer_vector(entity_data.get('origin', '0 0 0')) * scale
    angles = parse_hammer_vector(entity_data.get('angles', '0 0 0'))
    target_name = entity_data.get('targetname', entity_data['classname'])
    model_path = content_manager.get_game_resource(str(model_name))
    if model_path:
        master_collection = get_new_unique_collection(target_name, parent_collection)
        model_texture_path = content_manager.get_game_resource(str(model_name.with_name(model_name.stem + 't.mdl')))
        model_container = import_model(model_path.open('rb'),
                                       model_texture_path.open('rb') if model_texture_path is not None else None, scale,
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
    'monster_gman': partial(handle_model_prop, Path('models/gman.mdl')),
    'monster_generic': handle_generic_model_prop,
    'cycler': handle_generic_model_prop,
}
