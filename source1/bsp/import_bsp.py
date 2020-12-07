import random
import re
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from .bsp_file import BSPFile, Model, Face
from .lump import LumpTypes, ModelLump, VertexLump, EdgeLump, FaceLump, SurfEdgeLump, StringsLump, StringOffsetLump, \
    TextureDataLump, TextureInfoLump, WorldLightLump
from .datatypes.world_light import EmitType, Color32

import bpy

from ..vtf.blender_material import BlenderMaterial
from ..vtf.import_vtf import import_texture
from ..vtf.vmt import VMT
from ...utilities import valve_utils
from ...utilities.gameinfo import Gameinfo
from ...utilities.path_utilities import NonSourceInstall
from ...utilities.valve_utils import fix_workshop_not_having_gameinfo_file

material_name_fix = re.compile(r"_-?[\d]+_-?[\d]+_-?[\d]+")


def fix_material_name(material_name):
    if material_name:
        if material_name_fix.search(material_name):
            material_name = material_name_fix.sub("", material_name)
            material_name = str(Path(material_name).with_suffix(''))
    return material_name


def get_material(mat_name, model_ob):
    if mat_name:
        mat_name = fix_material_name(mat_name)
    else:
        mat_name = "Material"
    mat_ind = 0
    md = model_ob.data
    mat = None
    for candidate in bpy.data.materials:  # Do we have this material already?
        if candidate.name == mat_name:
            mat = candidate
    if mat:
        if md.materials.get(mat.name):  # Look for it on this mesh_data
            for i in range(len(md.materials)):
                if md.materials[i].name == mat.name:
                    mat_ind = i
                    break
        else:  # material exists, but not on this mesh_data
            md.materials.append(mat)
            mat_ind = len(md.materials) - 1
    else:  # material does not exist
        mat = bpy.data.materials.new(mat_name)
        md.materials.append(mat)
        # Give it a random colour
        rand_col = []
        for i in range(3):
            rand_col.append(random.uniform(.4, 1))
        rand_col.append(1.0)
        mat.diffuse_color = rand_col

        mat_ind = len(md.materials) - 1

    return mat_ind


class BSP:
    scale = 0.01

    def __init__(self, map_path):
        self.filepath = Path(map_path)
        print('Loading map from', self.filepath)
        self.map_file = BSPFile(self.filepath)
        self.map_file.parse()
        self.main_collection = bpy.data.collections.new(self.filepath.name)
        bpy.context.scene.collection.children.link(self.main_collection)

    @staticmethod
    def gather_vertex_ids(model: Model,
                          faces: List[Face],
                          surf_edges: List[Tuple[int, int]],
                          edges: List[Tuple[int, int]]):
        vertex_ids = []
        for map_face in faces[model.first_face:model.first_face + model.face_count]:
            first_edge = map_face.first_edge
            edge_count = map_face.edge_count
            for surf_edge in surf_edges[first_edge:first_edge + edge_count]:
                reverse = surf_edge >= 0
                edge = edges[abs(surf_edge)]
                vertex_id = edge[0] if reverse else edge[1]
                vertex_ids.append(vertex_id)
        return len(vertex_ids)

    def get_string(self, string_id):
        strings_lump: Optional[StringsLump] = self.map_file.lumps.get(LumpTypes.LUMP_TEXDATA_STRING_TABLE, None)
        return strings_lump.strings[string_id] or "NO_NAME"

    def load_map_mesh(self):

        model_lump: Optional[ModelLump] = self.map_file.lumps.get(LumpTypes.LUMP_MODELS, None)
        vertex_lump: Optional[VertexLump] = self.map_file.lumps.get(LumpTypes.LUMP_VERTICES, None)
        edge_lump: Optional[EdgeLump] = self.map_file.lumps.get(LumpTypes.LUMP_EDGES, None)
        surf_edge_lump: Optional[SurfEdgeLump] = self.map_file.lumps.get(LumpTypes.LUMP_SURFEDGES, None)
        face_lump: Optional[FaceLump] = self.map_file.lumps.get(LumpTypes.LUMP_FACES, None)
        texture_info_lump: Optional[TextureInfoLump] = self.map_file.lumps.get(LumpTypes.LUMP_TEXINFO, None)
        texture_data_lump: Optional[TextureDataLump] = self.map_file.lumps.get(LumpTypes.LUMP_TEXDATA, None)

        if vertex_lump and face_lump and model_lump:
            vertex_lump.vertices = np.multiply(vertex_lump.vertices, self.scale)
            for n, model in enumerate(model_lump.models):
                print(f"Loading model {n}/{len(model_lump.models)}")
                mesh_obj = bpy.data.objects.new(f"{self.filepath.stem}_{n}",
                                                bpy.data.meshes.new(f"{self.filepath.stem}_{n}_MESH"))
                mesh_data = mesh_obj.data
                self.main_collection.objects.link(mesh_obj)
                mesh_obj.location = model.origin

                material_lookup_table = {}
                for texture_info in texture_info_lump.texture_info:
                    texture_data = texture_data_lump.texture_data[texture_info.texture_data_id]
                    material_name = self.get_string(texture_data.name_id)
                    material_lookup_table[texture_data.name_id] = get_material(material_name, mesh_obj)

                faces = []
                material_indices = []
                vertices = []
                vertex_count = self.gather_vertex_ids(model,
                                                      face_lump.faces,
                                                      surf_edge_lump.surf_edges,
                                                      edge_lump.edges)

                uvs_per_face = []
                for map_face in face_lump.faces[model.first_face:model.first_face + model.face_count]:
                    uvs = np.zeros((vertex_count, 2), dtype=np.float)
                    face = []
                    first_edge = map_face.first_edge
                    edge_count = map_face.edge_count

                    texture_info = texture_info_lump.texture_info[map_face.tex_info_id]
                    texture_data = texture_data_lump.texture_data[texture_info.texture_data_id]
                    for surf_edge in surf_edge_lump.surf_edges[first_edge:first_edge + edge_count]:
                        reverse = surf_edge >= 0
                        edge = edge_lump.edges[abs(surf_edge)]
                        vertex_id = edge[0] if reverse else edge[1]

                        vert = tuple(vertex_lump.vertices[vertex_id])
                        if vert in vertices:
                            face.append(vertices.index(vert))
                        else:
                            face.append(len(vertices))
                            vertices.append(vert)

                        tv1, tv2 = texture_info.texture_vectors
                        uco = np.array(tv1[:3])
                        vco = np.array(tv2[:3])
                        u = np.dot(np.array(np.multiply(vert, 1.0 / self.scale)), uco) + tv1[3]
                        v = np.dot(np.array(np.multiply(vert, 1.0 / self.scale)), vco) + tv2[3]
                        uvs[vertices.index(vert)] = [u / texture_data.width, 1 - (v / texture_data.height)]

                    material_indices.append(material_lookup_table[texture_data.name_id])
                    uvs_per_face.append(uvs)
                    faces.append(face)

                mesh_data.from_pydata(vertices, [], faces)
                mesh_data.polygons.foreach_set('material_index', material_indices)

                mesh_data.uv_layers.new()
                uv_data = mesh_data.uv_layers[0].data
                for poly in mesh_data.polygons:
                    for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                        uv_data[loop_index].uv = uvs_per_face[poly.index][mesh_data.loops[loop_index].vertex_index]

    def load_lights(self):
        lights_lump: Optional[WorldLightLump] = self.map_file.lumps.get(LumpTypes.LUMP_WORLDLIGHTS, None)
        if lights_lump:
            for light in lights_lump.lights:
                loc = np.multiply(light.origin, self.scale)
                if light.type == EmitType.emit_point:
                    bpy.ops.object.light_add(type='POINT', align='WORLD', location=loc)
                elif light.type == EmitType.emit_spotlight:
                    bpy.ops.object.light_add(type='SPOT', align='WORLD', location=loc)
                elif light.type == EmitType.emit_skylight:
                    bpy.ops.object.light_add(type='SUN', align='WORLD', location=loc)
                elif light.type == EmitType.emit_skyambient:
                    bpy.ops.object.light_add(type='AREA', radius=1, align='WORLD', location=loc)
                else:
                    print("unsupported light type", light.type.name)
                    continue
                lamp = bpy.context.object
                lamp_data = lamp.data
                lamp_data.energy = light.intensity.magnitude() * self.scale * 10
                lamp_data.color = light.intensity.normalized().rgb
                lamp.rotation_euler = (light.normal[0], light.normal[1], light.normal[2])

    def load_materials(self):
        mod_path = valve_utils.get_mod_path(self.filepath)
        rel_model_path = self.filepath.relative_to(mod_path)
        print('Mod path', mod_path)
        print('Relative map path', rel_model_path)
        mod_path = fix_workshop_not_having_gameinfo_file(mod_path)
        gi_path = mod_path / 'gameinfo.txt'
        if gi_path.exists():
            path_resolver = Gameinfo(gi_path)
        else:
            path_resolver = NonSourceInstall(rel_model_path)

        texture_data_lump: Optional[TextureDataLump] = self.map_file.lumps.get(LumpTypes.LUMP_TEXDATA, None)
        for texture_data in texture_data_lump.texture_data:
            material_name = fix_material_name(self.get_string(texture_data.name_id))
            print(f"Loading {material_name} material")
            try:
                material_path = path_resolver.find_material(material_name, True)

                if material_path and material_path.exists():
                    try:
                        vmt = VMT(material_path)
                        vmt.parse()
                        for name, tex in vmt.textures.items():
                            import_texture(tex)
                        mat = BlenderMaterial(vmt)
                        mat.load_textures(True)
                        mat.create_material(material_name, True)
                    except Exception as m_ex:
                        print(f'Failed to import material "{material_name}", caused by {m_ex}')
                        import traceback
                        traceback.print_exc()
                else:
                    print(f'Failed to find {material_name} material')
            except Exception as t_ex:
                print(f'Failed to import materials, caused by {t_ex}')
                import traceback
                traceback.print_exc()
