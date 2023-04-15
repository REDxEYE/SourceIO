from collections import defaultdict
from itertools import chain
from pprint import pformat

import bpy

from .....library.source1.bsp.datatypes.model import RespawnModel
from .....library.source1.bsp.datatypes.texture_data import TextureData
from .....library.source1.bsp.lumps.face_indices_lump import *
from .....library.source1.bsp.lumps.lightmap_header_lump import *
from .....library.source1.bsp.lumps.lightmap_lump import *
from .....library.source1.bsp.lumps.material_sort_lump import *
from .....library.source1.bsp.lumps.mesh_lump import *
from .....library.source1.bsp.lumps.vertex_lump import *
from .....library.source1.bsp.lumps.vertex_normal_lump import *
from ....utils.utils import add_material
from ..entities.base_entity_handler import BaseEntityHandler
from ..entities.r1_entity_classes import (Base, entity_class_handle,
                                          func_window_hint,
                                          trigger_capture_point,
                                          trigger_indoor_area,
                                          trigger_out_of_bounds,
                                          trigger_soundscape, worldspawn)


class TitanfallEntityHandler(BaseEntityHandler):
    entity_lookup_table = entity_class_handle

    def _load_brush_model(self, model_id, model_name):
        objs = []
        model: RespawnModel = self._bsp.get_lump("LUMP_MODELS").models[model_id]
        tex_data: List[TextureData] = self._bsp.get_lump("LUMP_TEXDATA").texture_data
        indices: np.ndarray = self._bsp.get_lump("LUMP_INDICES").indices
        bsp_vertices: np.ndarray = self._bsp.get_lump('LUMP_VERTICES').vertices

        grouped_by_lightmap = defaultdict(list)

        for mesh_id in range(model.first_mesh, model.first_mesh + model.mesh_count):
            mesh: Mesh = self._bsp.get_lump("LUMP_MESHES").meshes[mesh_id]
            material_sort: MaterialSort = self._bsp.get_lump('LUMP_MATERIALSORT').materials[mesh.material_sort]
            grouped_by_lightmap[material_sort.lightmap_header_index].append(mesh_id)

        for lightmap_id, meshes in grouped_by_lightmap.items():
            merged_vertex_ids = np.array([], np.uint32)
            merged_uv_data = np.array([], np.float32)
            merged_lightmap_uv_data = np.array([], np.float32)
            merged_materials_ids = np.array([], np.uint32)
            material_indices = []

            l_headers = self._bsp.get_lump('LUMP_LIGHTMAP_HEADERS').lightmap_headers
            l_data = self._bsp.get_lump('LUMP_LIGHTMAP_DATA_SKY').lightmap_data
            offset = 0
            for n, header in enumerate(l_headers):
                for c in range(header.count + 1):
                    pixel_count = header.width * header.height
                    if n == lightmap_id:
                        name = f'lightmap_{n}_{c}'
                        if name in bpy.data.images:
                            continue
                        pixel_data: np.ndarray = l_data[offset:offset + pixel_count]
                        pixel_data = pixel_data.astype(np.float32) / 255
                        image = bpy.data.images.get(name, None) or bpy.data.images.new(
                            name,
                            width=header.width,
                            height=header.height,
                            alpha=True,
                        )
                        image.filepath = name + '.png'
                        image.alpha_mode = 'CHANNEL_PACKED'
                        image.file_format = 'PNG'

                        image.pixels.foreach_set(pixel_data.ravel())
                        image.pack()
                        del pixel_data

                    offset += pixel_count

            for mesh_id in meshes:
                self.logger.info(f'Loading Mesh {mesh_id - model.first_mesh}/{model.mesh_count} from {model_name}')
                mesh: Mesh = self._bsp.get_lump("LUMP_MESHES").meshes[mesh_id]
                material_sort: MaterialSort = self._bsp.get_lump('LUMP_MATERIALSORT').materials[mesh.material_sort]
                material_data = tex_data[material_sort.texdata_index]
                if material_data.name in material_indices:
                    mat_id = material_indices.index(material_data.name)
                else:
                    mat_id = len(material_indices)
                    material_indices.append(material_data.name)
                vertex_type = mesh.flags & 0x600
                if vertex_type == 0x000:
                    vertex_info_lump = self._bsp.get_lump("LUMP_LITVERTEXFLAT").vertex_info
                elif vertex_type == 0x200:
                    vertex_info_lump = self._bsp.get_lump("LUMP_BUMPLITVERTEX").vertex_info
                elif vertex_type == 0x400:
                    vertex_info_lump = self._bsp.get_lump("LUMP_UNLITVERTEX").vertex_info
                elif vertex_type == 0x600:
                    vertex_info_lump = self._bsp.get_lump("LUMP_UNLITTSVERTEX").vertex_info
                else:
                    raise NotImplementedError(f'Unknown mesh format {mesh.flags:016b}')
                mesh_indices = indices[
                               mesh.triangle_start:mesh.triangle_start + mesh.triangle_count * 3].astype(
                    np.uint32) + material_sort.vertex_offset
                used_vertices_info = vertex_info_lump[mesh_indices]

                vertex_indices = used_vertices_info['vpi'].flatten()
                uvs = used_vertices_info['uv']

                uvs[:, 1] = 1 - uvs[:, 1]

                merged_vertex_ids = np.append(merged_vertex_ids, vertex_indices, axis=0)
                if merged_uv_data.shape[0] == 0:
                    merged_uv_data = uvs
                else:
                    merged_uv_data = np.append(merged_uv_data, uvs, axis=0)

                if mesh.flags & 0x200 > 0:
                    uvs_lm = used_vertices_info['uv_lm']
                    uvs_lm[:, 1] = 1 - uvs_lm[:, 1]
                    if merged_lightmap_uv_data.shape[0] == 0:
                        merged_lightmap_uv_data = uvs_lm
                    else:
                        merged_lightmap_uv_data = np.append(merged_lightmap_uv_data, uvs_lm, axis=0)
                else:
                    if merged_lightmap_uv_data.shape[0] == 0:
                        merged_lightmap_uv_data = np.zeros_like(uvs)
                    else:
                        merged_lightmap_uv_data = np.append(merged_lightmap_uv_data, np.zeros_like(uvs), axis=0)

                if merged_materials_ids.shape[0] == 0:
                    merged_materials_ids = np.full((vertex_indices.shape[0] // 3,), mat_id)
                else:
                    merged_materials_ids = np.append(merged_materials_ids,
                                                     np.full((vertex_indices.shape[0] // 3,), mat_id), axis=0)
            mesh_obj = bpy.data.objects.new(f'{model_name}_{lightmap_id}',
                                            bpy.data.meshes.new(f"{model_name}_{lightmap_id}_MESH"))
            objs.append(mesh_obj)
            mesh_data = mesh_obj.data
            for mat in material_indices:
                add_material(mat, mesh_obj)

            unique_vertex_ids = np.unique(merged_vertex_ids)

            tmp2 = np.searchsorted(unique_vertex_ids, merged_vertex_ids)
            remapped = dict(zip(merged_vertex_ids, tmp2))

            faces = []
            uvs_per_face = []
            for triplet, tri_uv, tri_uv_lm in zip(merged_vertex_ids.reshape((-1, 3)),
                                                  merged_uv_data.reshape((-1, 3, 2)),
                                                  merged_lightmap_uv_data.reshape((-1, 3, 2))):
                faces.append((remapped[triplet[0]], remapped[triplet[1]], remapped[triplet[2]]))
                uvs_per_face.append(
                    {remapped[triplet[0]]: (tri_uv[0], tri_uv_lm[0]),
                     remapped[triplet[1]]: (tri_uv[1], tri_uv_lm[1]),
                     remapped[triplet[2]]: (tri_uv[2], tri_uv_lm[2])})

            mesh_data.from_pydata(bsp_vertices[unique_vertex_ids] * self.scale, [], faces)
            mesh_data.polygons.foreach_set('material_index', merged_materials_ids)

            mesh_data.uv_layers.new()
            uv_data = mesh_data.uv_layers[0].data
            uv_lm_data = mesh_data.uv_layers.new(name='LIGHTMAP').data

            for poly in mesh_data.polygons:
                for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                    uv_data[loop_index].uv = uvs_per_face[poly.index][mesh_data.loops[loop_index].vertex_index][0]
                    uv_lm_data[loop_index].uv = uvs_per_face[poly.index][mesh_data.loops[loop_index].vertex_index][1]

        return objs

    def _set_location_and_scale(self, obj, location, additional_scale=1.0):
        if isinstance(obj, list):
            for o in obj:
                super()._set_location_and_scale(o, location, additional_scale)
        else:
            super()._set_location_and_scale(obj, location, additional_scale)

    def _set_location(self, obj, location):
        if isinstance(obj, list):
            for o in obj:
                super()._set_location(o, location)
        else:
            super()._set_location(obj, location)

    @staticmethod
    def _set_parent_if_exist(obj, parent_name):
        if isinstance(obj, list):
            for o in obj:
                BaseEntityHandler._set_parent_if_exist(o, parent_name)
        else:
            BaseEntityHandler._set_parent_if_exist(obj, parent_name)

    def _put_into_collection(self, name, obj, grouping_collection_name=None):
        if isinstance(obj, list):
            for o in obj:
                super()._put_into_collection(name, o, grouping_collection_name)
        else:
            super()._put_into_collection(name, obj, grouping_collection_name)

    def _set_entity_data(self, obj, entity_raw: dict):
        if isinstance(obj, list):
            for o in obj:
                super()._set_entity_data(o, entity_raw)
        else:
            super()._set_entity_data(obj, entity_raw)

    def load_entities(self):
        entity_lump = self._bsp.get_lump('LUMP_ENTITIES')
        additional_entity_lump = self._bsp.get_lump('LUMP_ENTITYPARTITIONS')
        for entity_data in chain(entity_lump.entities, additional_entity_lump.entities):
            if not self.handle_entity(entity_data):
                self.logger.warn(pformat(entity_data))
        # bpy.context.view_layer.update()
        # for entity_data in entity_lump.entities:
        #     self.resolve_parents(entity_data)
        pass

    def handle_worldspawn(self, entity: worldspawn, entity_raw: dict):
        world = self._load_brush_model(0, 'world_geometry')
        for obj in world:
            self._set_entity_data(obj, {'entity': entity_raw})
            self.parent_collection.objects.link(obj)

    def handle_func_window_hint(self, entity: func_window_hint, entity_raw: dict):
        obj = self._create_empty(f'func_window_hint_{entity.hammer_id}')
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('func_window_hint', obj, 'brushes')
        pass

    def handle_trigger_indoor_area(self, entity: trigger_indoor_area, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        world = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_entity_data(world, {'entity': entity_raw})
        self._set_location(world, entity.origin)
        self._put_into_collection('trigger_indoor_area', world, 'triggers')

    def handle_trigger_capture_point(self, entity: trigger_capture_point, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        obj = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_entity_data(obj, {'entity': entity_raw})
        self._set_location(obj, entity.origin)
        self._put_into_collection('trigger_capture_point', obj, 'triggers')

    def handle_trigger_out_of_bounds(self, entity: trigger_out_of_bounds, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        obj = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_entity_data(obj, {'entity': entity_raw})
        self._set_location(obj, entity.origin)
        self._put_into_collection('trigger_out_of_bounds', obj, 'triggers')

    def handle_trigger_soundscape(self, entity: trigger_soundscape, entity_raw: dict):
        if 'model' not in entity_raw:
            return
        model_id = int(entity_raw.get('model')[1:])
        obj = self._load_brush_model(model_id, self._get_entity_name(entity))
        self._set_entity_data(obj, {'entity': entity_raw})
        self._set_location(obj, entity.origin)
        self._put_into_collection('trigger_soundscape', obj, 'triggers')

    def handle_info_particle_system(self, entity: Base, entity_raw: dict):
        pass

    def handle_info_node(self, entity: Base, entity_raw: dict):
        pass

    def handle_info_node_cover_stand(self, entity: Base, entity_raw: dict):
        pass

    def handle_info_spawnpoint_dropship_start(self, entity: Base, entity_raw: dict):
        pass

    def handle_info_spawnpoint_titan(self, entity: Base, entity_raw: dict):
        pass

    def handle_info_spawnpoint_titan_start(self, entity: Base, entity_raw: dict):
        pass

    def handle_info_spawnpoint_droppod(self, entity: Base, entity_raw: dict):
        pass

    def handle_info_spawnpoint_droppod_start(self, entity: Base, entity_raw: dict):
        pass

    def handle_info_spawnpoint_human(self, entity: Base, entity_raw: dict):
        pass

    def handle_info_spawnpoint_human_start(self, entity: Base, entity_raw: dict):
        pass

    def handle_info_frontline(self, entity: Base, entity_raw: dict):
        pass

    def handle_ambient_generic(self, entity: Base, entity_raw: dict):
        pass

    def handle_traverse(self, entity: Base, entity_raw: dict):
        pass

    def handle_info_hardpoint(self, entity: Base, entity_raw: dict):
        pass

    def handle_assault_assaultpoint(self, entity: Base, entity_raw: dict):
        pass

    def handle_info_target(self, entity: Base, entity_raw: dict):
        pass

    def handle_info_hint(self, entity: Base, entity_raw: dict):
        pass

    def handle_info_target_clientside(self, entity: Base, entity_raw: dict):
        pass

    def handle_info_node_safe_hint(self, entity: Base, entity_raw: dict):
        pass
