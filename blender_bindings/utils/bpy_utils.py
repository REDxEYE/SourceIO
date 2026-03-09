import contextlib
import random
from typing import Optional

import bpy
import numpy as np

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

def get_vertex_indices(mesh_data: bpy.types.Mesh) -> np.ndarray:
    vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
    mesh_data.loops.foreach_get('vertex_index', vertex_indices)
    return vertex_indices

def add_uv_layer(name: str, uv_data: np.ndarray, mesh_data: bpy.types.Mesh,
                 vertex_indices: Optional[np.ndarray] = None,
                 flip_uv: bool = True):
    uv_layer = mesh_data.uv_layers.new(name=name)
    uv_data = uv_data.copy()
    if flip_uv:
        uv_data[:, 1] = 1 - uv_data[:, 1]
    if vertex_indices is None:
        vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
        mesh_data.loops.foreach_get('vertex_index', vertex_indices)

    uv_layer.data.foreach_set('uv', uv_data[vertex_indices].ravel())


def add_vertex_color_layer(name: str, v_color_data: np.ndarray, mesh_data: bpy.types.Mesh,
                           vertex_indices: Optional[np.ndarray] = None):
    v_color_data = v_color_data.copy()
    if vertex_indices is None:
        vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
        mesh_data.loops.foreach_get('vertex_index', vertex_indices)

    vertex_colors = mesh_data.vertex_colors.get(name, False) or mesh_data.vertex_colors.new(name=name)
    vertex_colors.data.foreach_set('color', v_color_data[vertex_indices].flatten())


def add_custom_normals(normals: np.ndarray, mesh_data: bpy.types.Mesh):
    mesh_data.polygons.foreach_set("use_smooth", np.ones(len(mesh_data.polygons), np.uint32))
    if not is_blender_4_1():
        mesh_data.use_auto_smooth = True
    mesh_data.normals_split_custom_set_from_vertices(normals)

def add_custom_normals_from_faces(normals: np.ndarray, mesh_data: bpy.types.Mesh):
    mesh_data.polygons.foreach_set("use_smooth", np.ones(len(mesh_data.polygons), np.uint32))
    if not is_blender_4_1():
        mesh_data.use_auto_smooth = True
    mesh_data.normals_split_custom_set(normals)


def add_weights(bone_indices: np.ndarray, bone_weights: np.ndarray, bone_names: list[str], mesh_obj: bpy.types.Object):
    weight_groups = {name: mesh_obj.vertex_groups.new(name=name) for name in bone_names}
    for n, (index_group, weight_group), in enumerate(zip(bone_indices, bone_weights)):
        for index, weight in zip(index_group,weight_group):
            if weight > 0:
                weight_groups[bone_names[index]].add([n], weight, 'REPLACE')
