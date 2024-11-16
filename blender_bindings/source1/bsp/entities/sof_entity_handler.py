import re
from typing import Optional

import bmesh
import bpy
import numpy as np
from mathutils import Vector, geometry

from .abstract_entity_handlers import _srgb2lin
from .base_entity_classes import *
from .base_entity_handler import BaseEntityHandler
from SourceIO.blender_bindings.utils.bpy_utils import add_material, get_or_create_material
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.source1.bsp.datatypes.plane import RavenPlane
from SourceIO.library.source1.bsp.lumps import RavenModelLump, RavenFaceLump, VertexLump, ShadersLump, RavenBrushLump, \
    RavenBrushSidesLump
from SourceIO.library.source1.bsp.lumps.plane_lump import RavenPlaneLump
from SourceIO.library.source1.bsp.lumps.surf_edge_lump import RavenIndicesLump
from SourceIO.library.utils.math_utilities import ensure_length, SOURCE1_HAMMER_UNIT_TO_METERS
from SourceIO.library.utils.path_utilities import path_stem
from SourceIO.logger import SourceLogMan

strip_patch_coordinates = re.compile(r"_-?\d+_-?\d+_-?\d+.*$")
log_manager = SourceLogMan()


def srgb_to_linear(srgb: tuple[float]) -> tuple[list[float], float]:
    final_color = []
    if len(srgb) == 4:
        scale = srgb[3] / 255
    else:
        scale = 1
    for component in srgb[:3]:
        component = _srgb2lin(component / 255)
        final_color.append(component)
    if len(final_color) == 1:
        return ensure_length(final_color, 3, final_color[0]), 1
    return final_color, scale


def plane_to_point_normal(plane: RavenPlane):
    """Convert plane normal and dist to a point on the plane and its normal vector."""
    return Vector(plane.normal).normalized() * plane.dist, Vector(plane.normal)


def intersect_line_plane(line_point, line_dir, plane_point, plane_normal):
    """Find intersection point of a line and a plane."""
    line_dir.normalize()
    denom = plane_normal.dot(line_dir)
    if abs(denom) > 1e-6:
        t = (plane_point - line_point).dot(plane_normal) / denom
        return line_point + line_dir * t
    else:
        return None


def compute_brush_geometry(planes):
    """Compute vertices and a simple set of faces for a brush defined by planes."""
    vertices = []
    lines = []
    for i, plane_a in enumerate(planes):
        for plane_b in planes[i + 1:]:
            line = geometry.intersect_plane_plane(*plane_to_point_normal(plane_a), *plane_to_point_normal(plane_b))
            if all(line):
                lines.append(line)

    for line_point, line_dir in lines:
        for plane in planes:
            vertex = intersect_line_plane(line_point, line_dir, *plane_to_point_normal(plane))
            if vertex and all(plane_normal.dot(vertex - plane_point) <= 0 for plane_point, plane_normal in
                              [plane_to_point_normal(p) for p in planes]):
                vertices.append(vertex)

    # Deduplicate vertices
    unique_vertices = list({v.to_tuple(): v for v in vertices}.values())
    return unique_vertices


class SOFEntityHandler(BaseEntityHandler):
    entity_lookup_table = {
        "worldspawn": Base,
        "pickup_ammo_556": Base,
        "pickup_ammo_9mm": Base,
        "pickup_ammo_45": Base,
        "pickup_ammo_762": Base,
        "pickup_ammo_12gauge": Base,
        "pickup_weapon_M590": Base,
        "pickup_armor_medium": Base,
        "pickup_armor_big": Base,
        "pickup_health_small": Base,
        "pickup_health_big": Base,
        "pickup_weapon_SIG551": Base,
        "pickup_weapon_US_SOCOM": Base,
        "pickup_weapon_SMOHG92": Base,
        "pickup_weapon_silvertalon": Base,
        "pickup_weapon_AN_M14": Base,
        "pickup_weapon_M84": Base,
        "pickup_weapon_M3A1": Base,
        "pickup_weapon_microuzi": Base,
        "pickup_weapon_M60": Base,
        "pickup_weapon_AK_74": Base,
        "pickup_weapon_USAS_12": Base,
        "pickup_weapon_MSG90A1": Base,
        "pickup_weapon_M4": Base,
        "pickup_backpack": Base,
        "func_static": Base,
        "func_wall": Base,
        "trigger_ladder": Base,
        "gametype_trigger": Base,
        "trigger_hurt": Base,
        "light": Base,
        "misc_model": Base,
    }
    light_power_multiplier = 100

    def __init__(self, bsp_file: BSPFile, parent_collection, world_scale: float = SOURCE1_HAMMER_UNIT_TO_METERS,
                 light_scale: float = 1.0):
        super().__init__(bsp_file, parent_collection, world_scale, light_scale)
        for name in ["pickup_ammo_556",
                     "pickup_ammo_9mm",
                     "pickup_ammo_45",
                     "pickup_ammo_762",
                     "pickup_ammo_12gauge",
                     "pickup_weapon_M590",
                     "pickup_armor_medium",
                     "pickup_armor_big",
                     "pickup_health_small",
                     "pickup_health_big",
                     "pickup_weapon_USAS_12",
                     "pickup_weapon_M4",
                     "pickup_weapon_AK_74",
                     "pickup_weapon_SIG551",
                     "pickup_weapon_US_SOCOM",
                     "pickup_weapon_SMOHG92",
                     "pickup_weapon_MSG90A1",
                     "pickup_weapon_silvertalon",
                     "pickup_weapon_AN_M14",
                     "pickup_weapon_M84",
                     "pickup_weapon_M3A1",
                     "pickup_weapon_microuzi",
                     "pickup_weapon_M60",
                     "pickup_backpack"]:
            setattr(self, 'handle_' + name, self._handle_generic_pickup)

    @classmethod
    def _get_entity_name(cls, entity: Base):
        if entity._raw_data.get('targetname', None):
            return str(entity._raw_data['targetname'])
        else:
            return f'{entity.class_name}_{entity.new_hammer_id()}'

    def _handle_generic_pickup(self, entity: Base, entity_raw: dict):
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location_and_scale(obj, parse_float_vector(entity_raw.get("origin", "0 0 0")))
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection(entity.class_name, obj, 'pickup')

    def handle_worldspawn(self, entity: Base, entity_raw: dict):
        world = self._load_brush_model(0, 'world_geometry')
        if not world:
            return
        self._world_geometry_name = world.name
        self._set_entity_data(world, {'entity': entity_raw})
        self.parent_collection.objects.link(world)

    def handle_func_static(self, entity: Base, entity_raw: dict):
        obj = self._load_brush_model(int(entity_raw["model"][1:]), self._get_entity_name(entity))
        if not obj:
            return
        self._set_location(obj, parse_float_vector(entity_raw.get("origin", "0 0 0")))
        self._set_rotation(obj, parse_float_vector(entity_raw.get("angles", "0 0 0")))
        self._put_into_collection(entity.class_name, obj, 'brushes')

    def handle_func_wall(self, entity: Base, entity_raw: dict):
        obj = self._load_brush_model(int(entity_raw["model"][1:]), self._get_entity_name(entity))
        if not obj:
            return
        self._set_location(obj, parse_float_vector(entity_raw.get("origin", "0 0 0")))
        self._set_rotation(obj, parse_float_vector(entity_raw.get("angles", "0 0 0")))
        self._put_into_collection(entity.class_name, obj, 'brushes')

    def handle_trigger_ladder(self, entity: Base, entity_raw: dict):
        obj = self._load_brush_model(int(entity_raw["model"][1:]), self._get_entity_name(entity))
        if not obj:
            return
        self._set_location(obj, parse_float_vector(entity_raw.get("origin", "0 0 0")))
        self._set_rotation(obj, parse_float_vector(entity_raw.get("angles", "0 0 0")))
        self._put_into_collection(entity.class_name, obj, 'triggers')

    def handle_gametype_trigger(self, entity: Base, entity_raw: dict):
        obj = self._load_brush_model(int(entity_raw["model"][1:]), self._get_entity_name(entity))
        if not obj:
            return
        self._set_location(obj, parse_float_vector(entity_raw.get("origin", "0 0 0")))
        self._set_rotation(obj, parse_float_vector(entity_raw.get("angles", "0 0 0")))
        self._put_into_collection(entity.class_name, obj, 'triggers')

    def handle_trigger_hurt(self, entity: Base, entity_raw: dict):
        obj = self._load_brush_model(int(entity_raw["model"][1:]), self._get_entity_name(entity))
        if not obj:
            return
        self._set_location(obj, parse_float_vector(entity_raw.get("origin", "0 0 0")))
        self._set_rotation(obj, parse_float_vector(entity_raw.get("angles", "0 0 0")))
        self._put_into_collection(entity.class_name, obj, 'triggers')

    def handle_light(self, entity: light, entity_raw: dict):
        brightness = float(entity_raw.get("light", 1.0))
        scale = float(entity_raw.get("scale", 1.0))
        light: bpy.types.PointLight = bpy.data.lights.new(self._get_entity_name(entity), 'POINT')
        light.cycles.use_multiple_importance_sampling = True
        light.color = parse_float_vector(entity_raw.get("_color", "1 1 1"))
        light.energy = brightness * scale * self.light_power_multiplier * self.scale * self.light_scale
        # TODO: possible to convert constant-linear-quadratic attenuation into blender?
        obj: bpy.types.Object = bpy.data.objects.new(self._get_entity_name(entity), object_data=light)
        self._set_location(obj, parse_float_vector(entity_raw.get("origin", "0 0 0")))
        self._set_rotation(obj, parse_float_vector(entity_raw.get("angles", "0 0 0")))
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('light', obj, 'lights')

    def handle_misc_model(self, entity: Base, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._set_location(obj, parse_float_vector(entity_raw.get("origin", "0 0 0")))
        if "angle" in entity_raw:
            self._set_rotation(obj, (0, float(entity_raw["angle"]), 0))
        if "angles" in entity_raw:
            self._set_rotation(obj, parse_float_vector(entity_raw.get("angles", "0 0 0")))
        self._put_into_collection('misc_model', obj, 'props')

    def _load_brush_model(self, model_id, model_name):
        bsp = self._bsp
        models_lump: Optional[RavenModelLump] = bsp.get_lump("LUMP_MODELS")
        faces_lump: Optional[RavenFaceLump] = bsp.get_lump("LUMP_FACES")
        vertices_lump: Optional[VertexLump] = bsp.get_lump("LUMP_DRAWVERTS")
        indices_lump: Optional[RavenIndicesLump] = bsp.get_lump("LUMP_DRAWINDEXES")
        brushes_lump: Optional[RavenBrushLump] = bsp.get_lump("LUMP_BRUSHES")
        brush_sides_lump: Optional[RavenBrushSidesLump] = bsp.get_lump("LUMP_BRUSHSIDES")
        planes_lump: Optional[RavenPlaneLump] = bsp.get_lump("LUMP_PLANES")
        shaders_lump: Optional[ShadersLump] = bsp.get_lump("LUMP_SHADERS")
        if not all((faces_lump, vertices_lump, indices_lump, indices_lump,
                    brushes_lump, brush_sides_lump, planes_lump)):
            self.logger.warn("No LUMP_MODELS or LUMP_FACES found!")
            return None
        model = models_lump.models[model_id]
        unique_materials = []
        material_ids = []
        vertices = []
        indices = []
        if model.face_count:
            for face in faces_lump.faces[model.face_offset:model.face_offset + model.face_count]:
                if face.indices_count == 0:
                    continue
                if face.shader_id not in unique_materials:
                    unique_materials.append(face.shader_id)
                face_indices = indices_lump.indices[face.index_offset:face.index_offset + face.indices_count]
                indices.extend(face_indices + len(vertices))
                face_vertices = vertices_lump.vertices[face.vertex_offset:face.vertex_offset + face.vertex_count]
                vertices.extend(face_vertices)
                material_ids.extend([unique_materials.index(face.shader_id)] * (face.indices_count // 3))

            indices = np.asarray(indices).reshape((-1, 3))
            vertices = np.asarray(vertices)
            if len(indices) == 0:
                return None
            mesh_obj = bpy.data.objects.new(model_name, bpy.data.meshes.new(f"{model_name}_MESH"))
            mesh_data = mesh_obj.data

            mesh_data.from_pydata(vertices["pos"] * self.scale, [], indices)

            for mat_id in unique_materials:
                shader = shaders_lump.shaders[mat_id]
                material = get_or_create_material(path_stem(shader.name), shader.name)
                add_material(material, mesh_obj)
            mesh_data.polygons.foreach_set('material_index', material_ids)

            mesh_data.polygons.foreach_set("use_smooth", np.ones(len(mesh_data.polygons), np.uint32))
            mesh_data.normals_split_custom_set_from_vertices(vertices['normal'])

            vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
            mesh_data.loops.foreach_get('vertex_index', vertex_indices)

            for i in range(4):
                vc = mesh_data.vertex_colors.new()
                colors = vertices["color"][vertex_indices, i].astype(np.float32) / 255
                vc.data.foreach_set('color', colors.flatten())

            uv_data = mesh_data.uv_layers.new()
            uvs = vertices['st']
            uvs[:, 1] = 1 - uvs[:, 1]
            uv_data.data.foreach_set('uv', uvs[vertex_indices].flatten())
            mesh_data.validate()
        elif model.brush_count:
            def join_bmesh(target_bm, source_bm):
                source_bm.verts.layers.int.new('index')
                idx_layer = source_bm.verts.layers.int['index']

                for face in source_bm.faces:
                    new_verts = []
                    for old_vert in face.verts:
                        if not old_vert.tag:
                            new_vert = target_bm.verts.new(old_vert.co)
                            target_bm.verts.index_update()
                            old_vert[idx_layer] = new_vert.index
                            old_vert.tag = True

                        target_bm.verts.ensure_lookup_table()
                        idx = old_vert[idx_layer]
                        new_verts.append(target_bm.verts[idx])

                    target_bm.faces.new(new_verts)
                return target_bm

            mesh = bpy.data.meshes.new(model_name)
            mesh_obj = bpy.data.objects.new(model_name, mesh)
            bmeshes = []
            for brush in brushes_lump.brushes[model.brush_offset:model.brush_offset + model.brush_count]:
                planes = []
                for side in brush_sides_lump.brush_sides[brush.side_offset:brush.side_offset + brush.side_count]:
                    planes.append(planes_lump.planes[side.plane_id])

                vertices = compute_brush_geometry(planes)
                bm = bmesh.new()
                for v in vertices:
                    bm.verts.new(v * self.scale)
                bm.verts.ensure_lookup_table()
                bmesh.ops.convex_hull(bm, input=bm.verts)
                bmeshes.append(bm)
            bm = bmesh.new()
            for t in bmeshes:
                join_bmesh(bm, t)
            bm.to_mesh(mesh)
            bm.free()
        else:
            return None
        return mesh_obj
