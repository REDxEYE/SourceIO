"""Incremental export: fingerprints of export inputs stored next to the QC.

Each exported file is recorded with the fingerprint of everything that
produced it plus the size and modification time it had when written. An item
is re-exported when its fingerprint changes, or when the output file is
missing or was modified outside the exporter. Fingerprints cover:

* exporter code (the model_export and export_nodes sources), so add-on
  updates invalidate everything;
* textures: the VTF Output node and every node upstream of it (types,
  properties, socket defaults and links) plus cheap source markers:
  path/size/mtime for files, the packed bytes for packed images and the
  pixels for generated or modified (dirty) images;
* meshes: the Blender data the capture reads (mesh attributes and topology,
  shape keys, vertex groups, modifiers, transforms, parenting, material
  slots, rig rest and pose) plus every export setting.

Not covered: data a modifier pulls from other objects (geometry nodes inputs,
shrinkwrap targets and the like) and vertex-animation clips, which are always
exported. Disable "Use export cache" on the model output to force a full
export.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import bpy
import numpy as np

CACHE_VERSION = 1
CACHE_SUFFIX = '.sourceio-cache.json'

# Node RNA properties that do not affect the exported result.
_SKIPPED_PROPERTIES = (frozenset(bpy.types.Node.bl_rna.properties.keys())
                       | {'rna_type'})


def cache_path(qc_path: Path) -> Path:
    return qc_path.with_name(qc_path.stem + CACHE_SUFFIX)


class _Hasher:
    def __init__(self):
        self._hash = hashlib.blake2b(digest_size=20)

    def feed(self, *values):
        for value in values:
            if isinstance(value, np.ndarray):
                self._hash.update(str((value.dtype.str, value.shape)).encode())
                self._hash.update(np.ascontiguousarray(value).tobytes())
            elif isinstance(value, (bytes, bytearray)):
                self._hash.update(bytes(value))
            else:
                self._hash.update(repr(value).encode())
            self._hash.update(b'\x00')
        return self

    def hexdigest(self) -> str:
        return self._hash.hexdigest()


@lru_cache(maxsize=1)
def code_fingerprint() -> str:
    """Hash of the exporter sources; computed once per Blender session."""
    here = Path(__file__).resolve().parent
    roots = (here, here.parent.parent / 'source1' / 'model_export')
    hasher = _Hasher().feed(CACHE_VERSION)
    for root in roots:
        for path in sorted(root.rglob('*.py')):
            hasher.feed(path.relative_to(root.parent).as_posix(),
                        path.read_bytes())
    vtf_root = here.parents[2] / 'library' / 'utils' / 'pylib'
    for path in sorted(vtf_root.glob('**/*vtf*')):
        if path.is_file():
            stat = path.stat()
            hasher.feed(path.name, stat.st_size, stat.st_mtime_ns)
    return hasher.hexdigest()


def _file_marker(path_value: str):
    path = Path(bpy.path.abspath(path_value)) if path_value else None
    if path is None:
        return ('no-file',)
    try:
        stat = path.stat()
    except OSError:
        return ('missing', str(path))
    return (str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _image_marker(image: bpy.types.Image):
    settings = (image.name_full, image.source, image.colorspace_settings.name,
                image.alpha_mode, tuple(image.size))
    if image.packed_file is not None:
        digest = hashlib.blake2b(image.packed_file.data, digest_size=20)
        return settings + ('packed', digest.hexdigest())
    if image.source == 'FILE' and not image.is_dirty:
        return settings + ('file',) + _file_marker(image.filepath)
    # Generated or edited in Blender: the pixels are the only reliable marker.
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    digest = hashlib.blake2b(pixels.tobytes(), digest_size=20)
    return settings + ('pixels', digest.hexdigest())


def _value(value):
    if isinstance(value, bpy.types.Image):
        return _image_marker(value)
    if isinstance(value, bpy.types.Text):
        return ('text', value.name_full, value.as_string())
    if isinstance(value, bpy.types.ID):
        return ('id', type(value).__name__, value.name_full)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return tuple(_value(item) for item in value)
    except TypeError:
        return repr(value)


def _rna_values(struct, skipped=frozenset()):
    result = []
    for prop in struct.bl_rna.properties:
        name = prop.identifier
        if name in skipped or name == 'rna_type' or prop.type == 'COLLECTION':
            continue
        result.append((name, _value(getattr(struct, name, None))))
    return tuple(result)


def node_fingerprint(node) -> str:
    """Fingerprint of ``node`` and everything upstream of it."""
    hasher = _Hasher().feed(code_fingerprint())
    seen = set()
    pending = [node]
    while pending:
        current = pending.pop()
        if current.name in seen:
            continue
        seen.add(current.name)
        hasher.feed(current.name, current.bl_idname, current.mute,
                    _rna_values(current, _SKIPPED_PROPERTIES))
        if current.bl_idname == 'SourceIOExternalTextureNode':
            hasher.feed(_file_marker(current.source_path))
        for socket in current.inputs:
            if socket.is_linked:
                for link in socket.links:
                    hasher.feed(socket.identifier, link.from_node.name,
                                link.from_socket.identifier)
                    pending.append(link.from_node)
            elif hasattr(socket, 'default_value'):
                hasher.feed(socket.identifier, _value(socket.default_value))
    return hasher.hexdigest()


_ATTRIBUTE_LAYOUT = {
    'FLOAT': ('value', 1, np.float32),
    'INT': ('value', 1, np.int32),
    'BOOLEAN': ('value', 1, bool),
    'INT8': ('value', 1, np.int32),
    'FLOAT_VECTOR': ('vector', 3, np.float32),
    'FLOAT2': ('vector', 2, np.float32),
    'FLOAT_COLOR': ('color', 4, np.float32),
    'BYTE_COLOR': ('color', 4, np.float32),
    'INT32_2D': ('value', 2, np.int32),
    'QUATERNION': ('value', 4, np.float32),
    'FLOAT4X4': ('value', 16, np.float32),
}


def _array(collection, prop, components, dtype):
    values = np.empty(len(collection) * components, dtype=dtype)
    collection.foreach_get(prop, values)
    return values


def _mesh_data(hasher, mesh):
    hasher.feed(_array(mesh.vertices, 'co', 3, np.float32),
                _array(mesh.edges, 'vertices', 2, np.int32),
                _array(mesh.loops, 'vertex_index', 1, np.int32),
                _array(mesh.polygons, 'loop_start', 1, np.int32))
    for attribute in sorted(mesh.attributes, key=lambda item: item.name):
        layout = _ATTRIBUTE_LAYOUT.get(attribute.data_type)
        hasher.feed(attribute.name, attribute.domain, attribute.data_type)
        if layout is not None:
            prop, components, dtype = layout
            hasher.feed(_array(attribute.data, prop, components, dtype))
    hasher.feed(mesh.uv_layers.active.name if mesh.uv_layers.active else None)
    keys = mesh.shape_keys
    if keys is not None:
        hasher.feed(keys.use_relative, keys.eval_time, keys.reference_key.name)
        for block in keys.key_blocks:
            hasher.feed(block.name, block.mute, block.slider_min,
                        block.slider_max, block.relative_key.name,
                        block.vertex_group, block.interpolation,
                        _array(block.data, 'co', 3, np.float32))


def geometry_fingerprint(geometry, rig, settings) -> str | None:
    """Fingerprint of everything a geometry export reads.

    None when it cannot be cached.
    """
    if geometry.vertex_animations:
        return None
    hasher = _Hasher().feed(
        code_fingerprint(), settings, geometry.key, geometry.auto_merge,
        tuple(sorted(dict(geometry.stereo_balance_groups).items())))
    source = geometry.source
    objects = (sorted(source.all_objects, key=lambda item: item.name)
               if isinstance(source, bpy.types.Collection) else [source])
    hasher.feed(source.name, [obj.name for obj in objects])
    for obj in objects:
        if obj.type not in {'MESH', 'EMPTY', 'ARMATURE', 'LIGHT', 'CAMERA'}:
            # Curves and other evaluated types are always exported.
            return None
        hasher.feed(obj.name, obj.type, _value(obj.matrix_world),
                    obj.parent.name if obj.parent else None,
                    obj.parent_type, obj.parent_bone, obj.hide_viewport,
                    obj.hide_get())
        if obj.type != 'MESH':
            continue
        hasher.feed(obj.data.name)
        _mesh_data(hasher, obj.data)
        hasher.feed([(slot.link, slot.material.name if slot.material else None)
                     for slot in obj.material_slots])
        for modifier in obj.modifiers:
            hasher.feed(modifier.type, _rna_values(modifier))
        groups = {group.index: group.name for group in obj.vertex_groups}
        hasher.feed(sorted(groups.items()))
        weights = [(vertex.index, group.group, group.weight)
                   for vertex in obj.data.vertices for group in vertex.groups]
        hasher.feed(np.asarray(weights, dtype=np.float64))
    if rig is not None:
        hasher.feed(rig.name, _value(rig.matrix_world))
        for bone in rig.data.bones:
            hasher.feed(bone.name, bone.parent.name if bone.parent else None,
                        _value(bone.matrix_local))
        if rig.pose is not None:
            for bone in rig.pose.bones:
                hasher.feed(bone.name, _value(bone.matrix))
    return hasher.hexdigest()


class ExportCache:
    """Recorded fingerprints of exported files for one model output.

    Stored as ``<qc stem>.sourceio-cache.json`` next to the QC.
    """

    def __init__(self, path: Path):
        self.path = path
        self.entries = {}
        self.written = 0
        self.skipped = 0
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if data.get('version') == CACHE_VERSION:
                self.entries = dict(data.get('files', {}))
        except (OSError, ValueError, AttributeError):
            pass

    def fresh(self, output: Path, key: str | None) -> bool:
        """True when ``output`` was written from ``key`` and is unchanged.

        Unchanged means the size and modification time on disk still match.
        """
        if key is None:
            return False
        entry = self.entries.get(str(output))
        if not entry or entry.get('key') != key:
            return False
        try:
            stat = output.stat()
        except OSError:
            return False
        if (entry.get('size') != stat.st_size
                or entry.get('mtime_ns') != stat.st_mtime_ns):
            return False
        self.skipped += 1
        return True

    def record(self, output: Path, key: str | None, written: bool = True):
        if written:
            self.written += 1
        else:
            self.skipped += 1
        if key is None:
            self.entries.pop(str(output), None)
            return
        stat = output.stat()
        self.entries[str(output)] = {
            'key': key, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + '.tmp')
        temporary.write_text(
            json.dumps({'version': CACHE_VERSION, 'files': self.entries},
                       indent=1, sort_keys=True),
            encoding='utf-8')
        os.replace(temporary, self.path)
