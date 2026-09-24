"""Skin selection shared by model tools and map prop importers."""
import bpy

from SourceIO.blender_bindings.utils.bpy_utils import get_or_create_material
from SourceIO.library.utils.tiny_path import TinyPath


def prop_skin(obj, default):
    data = obj.get('entity_data', {})
    skin = data.get('skin', data.get('entity', {}).get('skin', obj.get('skin', default)))
    return str(skin) if skin not in (None, '') else default


def _material(value, source2=False):
    if isinstance(value, bpy.types.Material):
        return value
    if not value:
        return None
    if source2:
        return get_or_create_material(TinyPath(value).stem, TinyPath(value).as_posix())
    return bpy.data.materials.get(value) or bpy.data.materials.new(value)


def set_skin(obj, skin):
    """Apply a skin to one mesh, preserving slot identity across repeated swaps.

    Object-linked materials keep linked mesh duplicates independent. The stored
    slot-to-skin-table mapping also permits switching back from skins which use
    the same material for more than one slot.
    """
    groups = obj.get('skin_groups')
    if obj.type != 'MESH' or not groups or not hasattr(groups, 'keys'):
        return False
    names = list(groups.keys())
    skin = str(skin)
    if skin not in groups:
        # Source 2 entities can store a material group by its numeric index.
        if obj.get('model_type', '').lower() == 's2' and skin.isdecimal() and int(skin) < len(names):
            skin = names[int(skin)]
        else:
            return False
    current = obj.get('active_skin', names[0])
    if current not in groups:
        current = names[0]
    source2 = isinstance(groups[skin], str)
    old = [groups[current]] if source2 else list(groups[current])
    new = [groups[skin]] if source2 else list(groups[skin])
    old_materials = [_material(value, source2) for value in old]
    slots = obj.get('skin_slots')
    if slots is None or len(slots) != len(obj.material_slots):
        slots = [old_materials.index(slot.material) if slot.material in old_materials else -1
                 for slot in obj.material_slots]
        obj['skin_slots'] = slots
    for slot, index in zip(obj.material_slots, slots):
        if 0 <= index < len(new):
            material = _material(new[index], source2)
            if material is not None:
                slot.link = 'OBJECT'
                slot.material = material
    obj['active_skin'] = skin
    return True


def set_model_skin(container, skin):
    for obj in container.objects:
        set_skin(obj, skin)
