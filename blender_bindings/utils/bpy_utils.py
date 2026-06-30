import contextlib
import random

import bpy

from SourceIO.library.utils.perf_sampler import timed
from SourceIO.library.utils.tiny_path import TinyPath


@contextlib.contextmanager
def pause_view_layer_update():
    from bpy.ops import _BPyOpsSubModOp
    view_layer_update = _BPyOpsSubModOp._view_layer_update

    def dummy_view_layer_update(context):
        pass

    _BPyOpsSubModOp._view_layer_update = dummy_view_layer_update
    try:
        yield
    finally:
        _BPyOpsSubModOp._view_layer_update = view_layer_update


def is_blender_4():
    return bpy.app.version >= (4, 0, 0)


def is_blender_4_1():
    return bpy.app.version >= (4, 1, 0)


def is_blender_4_2():
    return bpy.app.version >= (4, 2, 0)


def is_blender_4_3():
    return bpy.app.version >= (4, 3, 0)


def is_blender_5():
    return bpy.app.version >= (5, 0, 0)


class ActionCurveFactory:
    """Compatibility wrapper for creating FCurves across Blender 4.x and 5.x.

    In Blender 5.0+ action.fcurves and action.groups were removed in favor of
    the slotted channelbag API.
    """

    def __init__(self, action: bpy.types.Action, armature_obj: bpy.types.Object):
        self.action = action
        self._armature = armature_obj
        self._use_channelbag = is_blender_5()
        adt = armature_obj.animation_data
        if adt is None:
            adt = armature_obj.animation_data_create()
        if self._use_channelbag:
            slot = action.slots.new(id_type='OBJECT', name=armature_obj.name)
            adt.action = action
            adt.action_slot = slot
            layer = action.layers.new(name="Layer")
            strip = layer.strips.new(type='KEYFRAME')
            self._channelbag = strip.channelbags.new(slot=slot)
        else:
            adt.action = action

    def new_group(self, name: str):
        if self._use_channelbag:
            return self._channelbag.groups.new(name=name)
        return self.action.groups.new(name=name)

    def new_fcurve(self, data_path: str, index: int = 0, group=None):
        if self._use_channelbag:
            curve = self._channelbag.fcurves.new(data_path=data_path, index=index)
        else:
            curve = self.action.fcurves.new(data_path=data_path, index=index)
        if group is not None:
            curve.group = group
        return curve


def find_layer_collection(layer_collection, name):
    if layer_collection.name == name:
        return layer_collection
    for layer in layer_collection.children:
        found = find_layer_collection(layer, name)
        if found:
            return found



def add_material(material, model_ob):
    md = model_ob.data
    for i, ob_material in enumerate(md.materials):
        if (ob_material.name == material.name and
            ob_material.get("full_path", "Not match") == material.get("full_path", "Not match too")
        ):
            return i
    else:
        md.materials.append(material)
        return len(md.materials) - 1



def get_or_create_material(name: str, full_path: str):
    full_path = full_path.lstrip('/').casefold()
    for mat in bpy.data.materials:
        #if (fp := mat.get('full_path', None)) is None:
        #    continue
        if TinyPath(mat.get('full_path', '').casefold()) == TinyPath(full_path):
            return mat
    mat = bpy.data.materials.new(name)
    mat["full_path"] = full_path
    mat.diffuse_color = [random.uniform(.4, 1) for _ in range(3)] + [1.0]
    return mat


KNOWN_COLLECTIONS_CACHE = {}


def get_or_create_collection(name, parent: bpy.types.Collection) -> bpy.types.Collection:
    if (key := KNOWN_COLLECTIONS_CACHE.get(name, None)) is not None:
        if (collection := bpy.data.collections.get(key, None)) is not None:
            return collection
        KNOWN_COLLECTIONS_CACHE.clear()

    new_collection = (bpy.data.collections.get(name, None) or bpy.data.collections.new(name))
    if new_collection.name not in parent.children:
        parent.children.link(new_collection)
    KNOWN_COLLECTIONS_CACHE[name] = new_collection.name
    return new_collection


# def get_new_unique_collection(model_name, parent_collection):
#     copy_count = len([collection for collection in bpy.data.collections if model_name in collection.name])
#
#     master_collection = get_or_create_collection(model_name + (f'_{copy_count}' if copy_count > 0 else ''),
#                                                  parent_collection)
#     return master_collection

_name_next_idx = {}

def get_new_unique_collection(model_name, parent_collection):
    """Faster for repeated calls: caches the next free suffix per base name and verifies only once per new base."""
    from bpy import data as _d

    if model_name not in _name_next_idx:
        if _d.collections.get(model_name) is None:
            _name_next_idx[model_name] = 1
            return get_or_create_collection(model_name, parent_collection)
        i = 1
        while _d.collections.get(f"{model_name}_{i}") is not None:
            i += 1
        _name_next_idx[model_name] = i + 1
        return get_or_create_collection(f"{model_name}_{i}", parent_collection)

    i = _name_next_idx[model_name]
    name = f"{model_name}_{i}"
    _name_next_idx[model_name] = i + 1
    return get_or_create_collection(name, parent_collection)


def append_blend(filepath, type_name, link=False):
    with bpy.data.libraries.load(filepath, link=link) as (data_from, data_to):
        setattr(data_to, type_name, [asset for asset in getattr(data_from, type_name)])
    for o in getattr(data_to, type_name):
        o.use_fake_user = True


def new_collection(name: str, parent: bpy.types.Collection):
    collection = bpy.data.collections.new(name)
    if collection.name not in parent.children:
        parent.children.link(collection)
    collection.name = name
    return collection
