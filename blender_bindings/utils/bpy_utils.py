import contextlib
import random

import bpy

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
    for mat in bpy.data.materials:
        if (fp := mat.get('full_path', None)) is None:
            continue
        if TinyPath(fp.lower()) == TinyPath(full_path.lower()):
            return mat
    mat = bpy.data.materials.new(name)
    mat["full_path"] = full_path
    mat.diffuse_color = [random.uniform(.4, 1) for _ in range(3)] + [1.0]
    return mat


def get_or_create_collection(name, parent: bpy.types.Collection) -> bpy.types.Collection:
    new_collection = (bpy.data.collections.get(name, None) or
                      bpy.data.collections.new(name))
    if new_collection.name not in parent.children:
        parent.children.link(new_collection)
    new_collection.name = name
    return new_collection


def get_new_unique_collection(model_name, parent_collection):
    copy_count = len([collection for collection in bpy.data.collections if model_name in collection.name])

    master_collection = get_or_create_collection(model_name + (f'_{copy_count}' if copy_count > 0 else ''),
                                                 parent_collection)
    return master_collection


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
