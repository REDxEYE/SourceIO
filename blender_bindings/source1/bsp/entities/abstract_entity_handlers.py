from array import array
import math
import re
from pprint import pformat

import bpy
import numpy as np
from mathutils import Euler

from SourceIO.blender_bindings.source1.bsp.entities.base_entity_classes import *
from SourceIO.blender_bindings.source1.vtf import import_texture
from SourceIO.blender_bindings.operators.import_settings_base import Source1BSPSettings
from SourceIO.blender_bindings.utils.bpy_utils import add_material, get_or_create_collection, get_or_create_material
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.source1.bsp.datatypes.face import Face
from SourceIO.library.source1.bsp.datatypes.model import Model
from SourceIO.library.source1.bsp.datatypes.texture_data import TextureData
from SourceIO.library.source1.bsp.datatypes.texture_info import TextureInfo
from SourceIO.library.source1.vmt import VMT
from SourceIO.library.utils.math_utilities import SOURCE1_HAMMER_UNIT_TO_METERS
from SourceIO.library.utils.path_utilities import path_stem
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.logger import SourceLogMan

strip_patch_coordinates = re.compile(r"_-?\d+_-?\d+_-?\d+.*$")
log_manager = SourceLogMan()


def gather_vertex_ids(model: Model, faces: list[Face], surf_edges: np.ndarray, edges: np.ndarray):
    vertex_offset = 0
    material_ids = []
    vertex_count = 0
    for map_face in faces[model.first_face:model.first_face + model.face_count]:
        vertex_count += map_face.edge_count
    vertex_ids = np.zeros(vertex_count, dtype=np.uint32)
    for map_face in faces[model.first_face:model.first_face + model.face_count]:
        if map_face.disp_info_id != -1:
            continue
        first_edge = map_face.first_edge
        edge_count = map_face.edge_count
        material_ids.append(map_face.tex_info_id)

        used_surf_edges = surf_edges[first_edge:first_edge + edge_count]
        reverse = np.subtract(1, (used_surf_edges > 0).astype(np.uint8))
        used_edges = edges[np.abs(used_surf_edges)]
        tmp = np.arange(len(used_edges))
        face_vertex_ids = used_edges[tmp, reverse]
        vertex_ids[vertex_offset:vertex_offset + edge_count] = face_vertex_ids
        vertex_offset += edge_count

    return vertex_ids, material_ids


def _set_uv(mesh_data, uv_data, uvs_per_face):
    loop_count = len(mesh_data.loops)
    flat_uvs = np.empty((loop_count, 2), dtype=np.float32)

    for poly in mesh_data.polygons:
        face_uvs = np.asarray(uvs_per_face[poly.index], dtype=np.float32)

        if len(face_uvs) != poly.loop_total:
            raise ValueError(f"UV size mismatch for polygon {poly.index}: {len(face_uvs)} UVs, {poly.loop_total} loops")

        start = poly.loop_start
        end = start + poly.loop_total
        flat_uvs[start:end] = face_uvs

    uv_data.foreach_set("uv", flat_uvs.ravel())


def corner_hash(vertex_id, uv, luv, ndigits=6):
    return hash((
        int(vertex_id),
        tuple(round(x, ndigits) for x in uv),
        tuple(round(x, ndigits) for x in luv),
    ))


def remove_dupe_face_vertices(face_vertex_ids, face_uvs, face_luvs):
    cleaned_ids = array("I")
    cleaned_uvs = array("f")
    cleaned_luvs = array("f")
    seen_hashes = set()

    for vertex_id, uv, luv in zip(face_vertex_ids, face_uvs, face_luvs):
        key = corner_hash(vertex_id, uv, luv)

        if key in seen_hashes:
            continue

        seen_hashes.add(key)

        cleaned_ids.append(int(vertex_id))

        cleaned_uvs.extend((float(uv[0]), float(uv[1])))
        cleaned_luvs.extend((float(luv[0]), float(luv[1])))

    ids = np.frombuffer(cleaned_ids, dtype=np.uint32).copy()
    uvs = np.frombuffer(cleaned_uvs, dtype=np.float32).reshape((-1, 2)).copy()
    luvs = np.frombuffer(cleaned_luvs, dtype=np.float32).reshape((-1, 2)).copy()

    return ids, uvs, luvs


def register_entity_handlers(handler_class):
    """Generate ``handle_<class>`` methods from a handler's declarative tables.

    ``handle_entity`` dispatches on ``handle_<classname>`` existing, so every
    supported entity needs a method -- but the overwhelming majority of them are
    one of four fixed shapes (brush model, studio model, point empty, or an
    intentional no-op). Declaring those in
    :attr:`~AbstractEntityHandler.BRUSH_ENTITIES` and friends keeps the
    hand-written methods for entities that genuinely need custom work, instead of
    hundreds of identical five-line copies.

    Never overwrites an existing method, so a table entry can be promoted to a
    real implementation just by writing one.
    """
    _ensure_lookup_entries(handler_class)
    for class_name, group in handler_class.BRUSH_ENTITIES.items():
        _add_generated_handler(handler_class, class_name,
                               lambda self, e, raw, n=class_name, g=group:
                               self._handle_brush_entity(n, g, e, raw))
    for class_name, group in handler_class.MODEL_ENTITIES.items():
        _add_generated_handler(handler_class, class_name,
                               lambda self, e, raw, n=class_name, g=group:
                               self._handle_model_entity(n, g, e, raw))
    for class_name, group in handler_class.POINT_ENTITIES.items():
        _add_generated_handler(handler_class, class_name,
                               lambda self, e, raw, n=class_name, g=group:
                               self._handle_point_entity(n, g, e, raw))
    for class_name in handler_class.NOOP_ENTITIES:
        # Claimed so `load_entities` stops logging them as unhandled; these carry
        # no importable world presence.
        _add_generated_handler(handler_class, class_name, lambda self, e, raw: None)
    return handler_class


def _ensure_lookup_entries(handler_class):
    """Give every declared entity a lookup-table entry.

    ``handle_entity`` requires one in addition to the method, and the tables are
    generated from what real maps contain -- which includes classes absent from the
    FGDs the ``*_entity_classes`` modules were generated from (e.g. HL2's
    ``func_train`` and ``item_box_*``). Fall back to ``Base``, which parses the
    shared keyvalues (``origin``, ``angles``, ``targetname``) that the generated
    handlers actually read.
    """
    declared = set(handler_class.BRUSH_ENTITIES) | set(handler_class.MODEL_ENTITIES) | \
               set(handler_class.POINT_ENTITIES) | set(handler_class.NOOP_ENTITIES)
    missing = declared - set(handler_class.entity_lookup_table)
    if not missing:
        return
    # Copy first: the table is often shared with the parent class.
    handler_class.entity_lookup_table = dict(handler_class.entity_lookup_table)
    for class_name in missing:
        handler_class.entity_lookup_table[class_name] = Base


def _add_generated_handler(handler_class, class_name: str, function):
    method_name = f'handle_{class_name}'
    if method_name in vars(handler_class):
        return  # hand-written implementation wins
    function.__name__ = method_name
    function.__qualname__ = f'{handler_class.__name__}.{method_name}'
    setattr(handler_class, method_name, function)


class AbstractEntityHandler:
    entity_lookup_table = {}

    #: ``classname -> collection group`` for entities whose ``model`` is a brush
    #: model (``*N``) stored in the BSP.
    BRUSH_ENTITIES: dict[str, str] = {}
    #: ``classname -> collection group`` for entities that reference a ``.mdl``.
    MODEL_ENTITIES: dict[str, str] = {}
    #: ``classname -> collection group`` for entities that are only a point in
    #: space; imported as an empty so their placement survives the round trip.
    POINT_ENTITIES: dict[str, str] = {}
    #: Entities deliberately not imported. Listed so they are not reported as
    #: unhandled -- they have no world presence to represent.
    NOOP_ENTITIES: frozenset[str] = frozenset()

    def __init__(self, bsp_file: BSPFile, content_manager: ContentManager, parent_collection,
                 world_scale: float = SOURCE1_HAMMER_UNIT_TO_METERS, light_scale: float = 1.0):
        self.logger = log_manager.get_logger(self.__class__.__name__)
        self._bsp: BSPFile = bsp_file
        self.content_manager = content_manager
        self.scale = world_scale
        self.light_scale = light_scale
        self.parent_collection = parent_collection

        self._entites = self._bsp.get_lump('LUMP_ENTITIES').entities
        self._handled_paths = set()
        self._entity_by_name_cache = {}
        self._world_geometry_name = ""
        self.settings: Source1BSPSettings | None = None

    def load_entities(self, settings: Source1BSPSettings):
        self.settings = settings
        entity_lump = self._bsp.get_lump('LUMP_ENTITIES')
        for entity_data in entity_lump.entities:
            entity_class: str = entity_data['classname']
            if entity_class.startswith("info_") and not settings.load_info:
                continue
            elif "decal" in entity_class and not settings.load_decals:
                continue
            elif "light" in entity_class and not settings.load_lights:
                continue
            elif entity_class.startswith("trigger_") and not settings.load_triggers:
                continue
            elif entity_class.startswith("prop_") and not settings.load_props:
                continue
            elif entity_class.startswith("logic_") and not settings.load_logic:
                continue
            elif entity_class.endswith("rope") and not settings.load_ropes:
                continue
            if not self.handle_entity(entity_data):
                self.logger.warn(pformat(entity_data))
        bpy.context.view_layer.update()
        # for entity_data in entity_lump.entities:
        #     self.resolve_parents(entity_data)
        pass

    def handle_entity(self, entity_data: dict):
        entity_class = entity_data['classname']
        if hasattr(self, f'handle_{entity_class}') and entity_class in self.entity_lookup_table:
            entity_class_obj = self._get_class(entity_class)
            entity_object = entity_class_obj(entity_data)
            handler_function = getattr(self, f'handle_{entity_class}')
            try:
                handler_function(entity_object, entity_data)
            except ValueError as e:
                import traceback
                self.logger.error(f'Exception during handling {entity_class} entity: {e.__class__.__name__}("{e}")')
                self.logger.error(traceback.format_exc())
                return False
            return True
        return False

    def _get_entity_by_name(self, name):
        if not self._entity_by_name_cache:
            self._entity_by_name_cache = {e['targetname']: e for e in self._entites if 'targetname' in e}
        entity = self._entity_by_name_cache.get(name, None)
        if entity is None:
            return None, None
        entity_class = self._get_class(entity['classname'])
        entity_obj = entity_class(entity)
        return entity_obj, entity

    def _load_brush_model(self, model_id, model_name):
        def _get_string(string_id: int) -> str:
            strings: list[str] = self._bsp.get_lump('LUMP_TEXDATA_STRING_TABLE').strings
            return strings[string_id] or "NO_NAME"

        model = self._bsp.get_lump("LUMP_MODELS").models[model_id]
        mesh_data = bpy.data.meshes.new(f"{model_name}_MESH")
        mesh_obj = bpy.data.objects.new(model_name, mesh_data)

        bsp_surf_edges: np.ndarray = self._bsp.get_lump('LUMP_SURFEDGES').surf_edges
        bsp_vertices: np.ndarray = self._bsp.get_lump('LUMP_VERTICES').vertices
        bsp_edges: np.ndarray = self._bsp.get_lump('LUMP_EDGES').edges
        bsp_faces: list[Face] = self._bsp.get_lump('LUMP_FACES').faces
        bsp_textures_info: list[TextureInfo] = self._bsp.get_lump('LUMP_TEXINFO').texture_info
        bsp_textures_data: list[TextureData] = self._bsp.get_lump('LUMP_TEXDATA').texture_data

        vertex_ids, material_ids = gather_vertex_ids(model, bsp_faces, bsp_surf_edges, bsp_edges)
        unique_vertex_ids = np.unique(vertex_ids)

        tmp2 = np.searchsorted(unique_vertex_ids, vertex_ids)
        remapped = dict(zip(vertex_ids, tmp2))

        material_lookup_table = {}
        skippable_materials = set()
        for texture_info_id in sorted(set(material_ids)):
            texture_info = bsp_textures_info[texture_info_id]
            texture_data = bsp_textures_data[texture_info.texture_data_id]
            material_name = _get_string(texture_data.name_id)
            material_name = material_name.rstrip("/\\").lstrip("/\\")
            if self.settings and self.settings.import_textures:
                material_file = self.content_manager.find_file(TinyPath("materials") / (material_name + ".vmt"))
                if material_file:
                    vmt = VMT(material_file, material_name, self.content_manager)
                    material_name = strip_patch_coordinates.sub("", material_name)
                    if vmt.get_int("$abovewater", 1) == 0:
                        skippable_materials.add(texture_info_id)
                else:
                    material_name = strip_patch_coordinates.sub("", material_name)
                    material_name = material_name.rstrip("/\\").lstrip("/\\")
                    material_file = self.content_manager.find_file(TinyPath("materials") / (material_name + ".vmt"))
                    if material_file:
                        vmt = VMT(material_file, material_name, self.content_manager)
                        if vmt.get_int("$abovewater", 1) == 0:
                            skippable_materials.add(texture_info_id)
            material = get_or_create_material(path_stem(material_name), material_name)
            material_lookup_table[texture_data.name_id] = add_material(material, mesh_obj)

        faces = []
        uvs_per_face = []
        luvs_per_face = []
        material_indices = []

        for map_face in bsp_faces[model.first_face:model.first_face + model.face_count]:
            if map_face.disp_info_id != -1:
                continue

            if map_face.tex_info_id in skippable_materials:
                continue

            used_surf_edges = bsp_surf_edges[map_face.first_edge:map_face.first_edge + map_face.edge_count]

            used_edges = bsp_edges[np.abs(used_surf_edges)]
            reverse = (used_surf_edges < 0).astype(np.uint8)

            face_vertex_ids = used_edges[np.arange(len(used_edges)), reverse]

            if len(face_vertex_ids) < 3:
                continue

            uv_vertices = bsp_vertices[face_vertex_ids]

            texture_info = bsp_textures_info[map_face.tex_info_id]
            texture_data = bsp_textures_data[texture_info.texture_data_id]

            tv1, tv2 = texture_info.texture_vectors
            lv1, lv2 = texture_info.lightmap_vectors

            tex_w = texture_data.width or 512
            tex_h = texture_data.height or 512

            u = (np.dot(uv_vertices, tv1[:3]) + tv1[3]) / tex_w
            v = 1.0 - ((np.dot(uv_vertices, tv2[:3]) + tv2[3]) / tex_h)

            lu = (np.dot(uv_vertices, lv1[:3]) + lv1[3]) / tex_w
            lv = 1.0 - ((np.dot(uv_vertices, lv2[:3]) + lv2[3]) / tex_h)

            face_uvs = np.stack([u, v], axis=1)
            face_luvs = np.stack([lu, lv], axis=1)

            face_vertex_ids, face_uvs, face_luvs = remove_dupe_face_vertices(
                face_vertex_ids,
                face_uvs,
                face_luvs,
            )

            if len(face_vertex_ids) < 3:
                continue

            face = []
            remapped_face_uvs = []
            remapped_face_luvs = []

            for vertex_id, uv, luv in zip(face_vertex_ids, face_uvs, face_luvs):
                new_vertex_id = remapped[int(vertex_id)]

                face.append(new_vertex_id)
                remapped_face_uvs.append(uv)
                remapped_face_luvs.append(luv)

            face = face[::-1]
            remapped_face_uvs = remapped_face_uvs[::-1]
            remapped_face_luvs = remapped_face_luvs[::-1]

            if len(face) < 3:
                print("Got invalid face len < 3")
                continue

            material_index = material_lookup_table[texture_data.name_id]

            faces.append(face)
            uvs_per_face.append(remapped_face_uvs)
            luvs_per_face.append(remapped_face_luvs)
            material_indices.append(material_index)

        mesh_data.from_pydata(bsp_vertices[unique_vertex_ids] * self.scale, [], faces)
        mesh_data.update()
        mesh_data.polygons.foreach_set('material_index', material_indices)

        main_uv = mesh_data.uv_layers.new()
        uv_data = main_uv.data
        _set_uv(mesh_data, uv_data, uvs_per_face)

        lightmap_uv = mesh_data.uv_layers.new(name='lightmap')
        uv_data = lightmap_uv.data
        _set_uv(mesh_data, uv_data, luvs_per_face)
        if mesh_data.validate(verbose=True):
            self.logger.warn(f"Mesh(*{model_id}) had some invalid geometry")
        return mesh_obj

    def _handle_brush_model(self, class_name, group, entity, entity_raw):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        mesh_object = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_location_and_scale(mesh_object, parse_float_vector(entity_raw.get('origin', '0 0 0')))
        self._set_rotation(mesh_object, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection(class_name, mesh_object, group)

    def _handle_brush_entity(self, class_name: str, group: str, entity, entity_raw: dict):
        """Import a brush-model entity, matching the hand-written handlers.

        Those use ``entity.origin`` and ``_set_location`` rather than
        ``_handle_brush_model``'s ``_set_location_and_scale``: brush vertices are
        already scaled by :meth:`_load_brush_model`, so scaling the object too
        would apply it twice.
        """
        model = entity_raw.get('model', '')
        if not model.startswith('*'):
            # Brush entities can also be pointed at a studio model (e.g. a
            # func_breakable with a gib model); fall back rather than crash on
            # int('') below.
            if model:
                self._handle_model_entity(class_name, group, entity, entity_raw)
            return
        mesh_object = self._load_brush_model(int(model[1:]), self._get_entity_name(entity))
        self._set_location(mesh_object, parse_float_vector(entity_raw.get('origin', '0 0 0')))
        self._set_rotation(mesh_object, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_entity_data(mesh_object, {'entity': entity_raw})
        self._put_into_collection(class_name, mesh_object, group)

    def _handle_model_entity(self, class_name: str, group: str, entity, entity_raw: dict):
        """Import a studio-model entity as a placeholder for the model loader."""
        model = entity_raw.get('model', '')
        if model.endswith('.vmt') or model.endswith('.spr'):
            # Sprite entities (e.g. env_sprite_clientside) put a material in
            # `model`, not a studio model. Handing that to the model loader would
            # fail, so keep the placement as an empty instead.
            self._handle_point_entity(class_name, group, entity, entity_raw)
            return
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._post_process_entity(obj, entity, entity_raw)
        self._put_into_collection(class_name, obj, group)

    def _post_process_entity(self, obj, entity, entity_raw: dict):
        """Hook for work that has to happen after the object exists.

        Applies the keyvalues that are common enough to be worth doing for every
        generated entity. A subclass needing more can either override this or write
        a full ``handle_<class>`` method -- a hand-written method always takes
        precedence over the generated one.
        """
        skin = entity_raw.get('skin')
        if skin not in (None, ''):
            obj['skin'] = parse_source_value(skin)
        # `$scale` on sprites and prop_scalable; `modelscale` is already applied by
        # `_handle_entity_with_model`.
        if 'scale' in entity_raw and entity_raw['scale'] not in (None, ''):
            try:
                scale = float(entity_raw['scale'])
            except (TypeError, ValueError):
                scale = 0.0
            if scale > 0.0:
                obj.scale *= scale

    #: Directory holding Hammer's editor-only helper models (axis/cone/camera
    #: gizmos). Entities default to these so they are visible while editing; they are
    #: not part of the map and must not be imported as geometry.
    EDITOR_MODEL_PREFIX = 'models/editor/'

    def _entity_default_model(self, entity) -> str | None:
        """A game model the entity class supplies rather than the map.

        Some entities never write a ``model`` keyvalue because the game hardcodes it
        -- Portal 2's ``prop_button`` is always ``props/switch001.mdl``, its turrets
        always ``props/turret_01.mdl``. The FGD-generated classes record these as
        ``model_``/``viewport_model``, so a class default means the entity has real
        geometry even though the map is silent about it.
        """
        for attribute in ('model_', 'viewport_model'):
            model = getattr(entity, attribute, None)
            if not isinstance(model, str) or not model.endswith('.mdl'):
                continue
            if model.lower().startswith(self.EDITOR_MODEL_PREFIX):
                continue  # Hammer gizmo, not map geometry
            return model
        return None

    def _handle_point_entity(self, class_name: str, group: str, entity, entity_raw: dict):
        """Import a point entity as an empty, preserving placement and keyvalues.

        Uses ``_set_location_and_scale``: ``_create_empty`` sizes the empty in Hammer
        units, so without the world scale applied the empties dwarf the map.
        """
        if 'model' not in entity_raw and self._entity_default_model(entity):
            # The class knows a model even though the map does not; import it as one.
            self._handle_model_entity(class_name, group, entity, entity_raw)
            return
        obj = self._create_empty(self._get_entity_name(entity))
        self._set_location_and_scale(obj, parse_float_vector(entity_raw.get('origin', '0 0 0')))
        self._set_rotation(obj, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        self._set_icon_if_present(obj, entity)
        self._set_entity_data(obj, {'entity': entity_raw})
        self._post_process_entity(obj, entity, entity_raw)
        self._put_into_collection(class_name, obj, group)

    def _set_entity_data(self, obj, entity_raw: dict):
        obj['entity_data'] = entity_raw

    @staticmethod
    def _get_entity_name(entity: Base):
        if hasattr(entity, 'targetname') and entity.targetname:
            return str(entity.targetname)
        else:
            return f'{entity.class_name}_{entity.hammer_id}'

    def _put_into_collection(self, name, obj, grouping_collection_name=None):
        if grouping_collection_name is not None:
            parent_collection = get_or_create_collection(grouping_collection_name, self.parent_collection)
            parent_collection = get_or_create_collection(name, parent_collection)
        else:
            parent_collection = get_or_create_collection(name, self.parent_collection)
        parent_collection.objects.link(obj)

    @staticmethod
    def _apply_light_rotation(obj, entity):
        obj.rotation_euler = Euler((0, math.radians(-90), 0))
        if len(entity.angles) == 1:
            obj.rotation_euler.rotate(Euler((
                math.radians(0),
                math.radians(-entity.pitch),
                math.radians(0)
            )))
        elif len(entity.angles) == 2:
            obj.rotation_euler.rotate(Euler((
                math.radians(0),
                math.radians(-entity.pitch),
                math.radians(entity.angles[1])
            )))
        else:
            obj.rotation_euler.rotate(Euler((
                math.radians(entity.angles[2]),
                math.radians(-entity.pitch),
                math.radians(entity.angles[1])
            )))

    def _set_location_and_scale(self, obj, location, additional_scale=1.0):
        scale = self.scale * additional_scale
        obj.location = location
        obj.location *= scale
        obj.scale *= scale

    def _set_location(self, obj, location):
        obj.location = location
        obj.location *= self.scale

    @staticmethod
    def _set_rotation(obj, angles):
        if len(angles) < 3:
            return
        obj.rotation_euler.rotate(Euler((math.radians(angles[2]),
                                         math.radians(angles[0]),
                                         math.radians(angles[1]))))

    @staticmethod
    def _set_single_angle(obj, angle: float):
        obj.rotation_euler.rotate(Euler((0, 0, math.radians(angle))))

    @staticmethod
    def _set_parent_if_exist(obj, parent_name):
        if parent_name is None:
            return
        if parent_name in bpy.data.objects:
            pass
            before = obj.matrix_world.copy()
            obj.parent = bpy.data.objects[parent_name]
            obj.matrix_world = before

    def _set_icon_if_present(self, obj, entity):
        icon_path = getattr(entity, 'icon_sprite', None)

        if icon_path is not None:
            icon_path = TinyPath(icon_path)
            icon = bpy.data.images.get(icon_path.stem, None)
            if icon is None:
                icon_material_file = self.content_manager.find_file(
                    TinyPath("materials") / icon_path.with_suffix(".vmt"))
                if not icon_material_file:
                    return
                vmt = VMT(icon_material_file, icon_path, self.content_manager)
                base_texture = vmt.get_string('$basetexture', None)
                if not base_texture:
                    return
                texture = self.content_manager.find_file(TinyPath("materials") / (base_texture + ".vtf"))
                if not texture:
                    return
                icon = import_texture(TinyPath(icon_path.stem), texture)

            obj.empty_display_type = 'IMAGE'
            obj.empty_display_size = (1 / self.scale)
            obj.data = icon

    @staticmethod
    def _create_lines(name, points, closed=False):
        line_data = bpy.data.curves.new(name=f'{name}_data', type='CURVE')
        line_data.dimensions = '3D'
        line_data.fill_mode = 'FULL'
        line_data.bevel_depth = 0

        polyline = line_data.splines.new('POLY')
        polyline.use_cyclic_u = closed
        polyline.points.add(len(points) - 1)
        for idx in range(len(points)):
            polyline.points[idx].co = tuple(points[idx]) + (1.0,)

        line = bpy.data.objects.new(f'{name}', line_data)
        line.location = [0, 0, 0]
        return line

    def _get_class(self, class_name) -> type(Base):
        if class_name in self.entity_lookup_table:
            entity_object = self.entity_lookup_table[class_name]
            return entity_object
        else:
            return Base

    def resolve_parents(self, entity_raw: dict):
        entity = self._get_class(entity_raw['classname'])
        entity.from_dict(entity, entity_raw)
        if hasattr(entity, 'targetname') and hasattr(entity, 'parentname'):
            if entity.targetname and str(entity.targetname) in bpy.data.objects:
                obj = bpy.data.objects[entity.targetname]
                self._set_parent_if_exist(obj, entity.parentname)

    @staticmethod
    def _create_empty(name):
        empty = bpy.data.objects.new(name, None)
        empty.empty_display_size = 16
        return empty

    def _handle_entity_with_model(self, entity, entity_raw: dict):
        if hasattr(entity, 'model') and entity.model:
            model_path = entity.model
        elif hasattr(entity, 'model_') and entity.model_:
            model_path = entity.model_
        elif hasattr(entity, 'viewport_model') and entity.viewport_model:
            model_path = entity.viewport_model
        elif "model" in entity_raw:
            model_path = entity_raw["model"]
        elif "viewport_model" in entity_raw:
            model_path = entity_raw["viewport_model"]
        else:
            model_path = 'error.mdl'
        obj = self._create_empty(self._get_entity_name(entity))
        properties = {'prop_path': model_path,
                      'type': entity.class_name,
                      'scale': self.scale,
                      'entity': entity_raw}

        self._set_location_and_scale(obj, parse_float_vector(entity_raw.get('origin', '0 0 0')))
        self._set_rotation(obj, parse_float_vector(entity_raw.get('angles', '0 0 0')))
        obj.scale *= parse_source_value(entity_raw.get("modelscale", 1))
        self._set_entity_data(obj, properties)

        return obj
