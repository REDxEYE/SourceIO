from pathlib import Path

# noinspection PyUnresolvedReferences
import bpy
# noinspection PyUnresolvedReferences
from mathutils import Vector, Matrix, Quaternion, Euler

import math

from ..utils.decode_animations import parse_anim_data
from ..source2 import ValveCompiledFile
import numpy as np

from ...bpy_utilities.utils import get_material, get_or_create_collection, get_new_unique_collection
from ...source_shared.content_manager import ContentManager
from ...utilities.math_utilities import HAMMER_UNIT_TO_METERS


class ValveCompiledPhysics(ValveCompiledFile):
    def __init__(self, path_or_file, scale=HAMMER_UNIT_TO_METERS):
        super().__init__(path_or_file)
        self.scale = scale
        self.data_block = self.get_data_block(block_name='DATA')[0]
        self.spheres = []
        self.capsules = []
        self.hulls = []
        self.meshes = []

    @staticmethod
    def gather_edges(start_edge_index: int, edges: dict):
        vertex_ids = [edges[start_edge_index]['m_nOrigin']]
        next_edge_index = edges[start_edge_index]['m_nNext']
        while True:
            if next_edge_index == start_edge_index:
                break
            vertex_ids.append(edges[next_edge_index]['m_nOrigin'])
            next_edge_index = edges[next_edge_index]['m_nNext']
        return vertex_ids

    def parse_meshes(self):
        data = self.data_block.data
        for part in data['m_parts']:
            shapes = part['m_rnShape']
            for sphere in shapes['m_spheres']:
                sphere_data = sphere['m_Sphere']
                self.spheres.append((sphere_data['m_vCenter'], sphere_data['m_flRadius']))
            for capsule in shapes['m_capsules']:
                capsule_data = capsule['m_Capsule']
                self.capsules.append((capsule_data['m_vCenter'], capsule_data['m_flRadius']))
            for n, hull in enumerate(shapes['m_hulls']):
                hull_data = hull['m_Hull']
                hull_name = hull['m_UserFriendlyName'] or f'hull_{n}'
                vertices = np.array(hull_data['m_Vertices'], np.float32)
                polygons = []
                for face in hull_data['m_Faces']:
                    edges = self.gather_edges(face['m_nEdge'], hull_data['m_Edges'])
                    polygons.append(edges)
                self.hulls.append((hull_name, polygons, vertices))
            for mesh in shapes['m_meshes']:
                mesh_data = mesh['m_Mesh']

    def build_mesh(self):
        meshes = []
        for sphere in self.spheres:
            print(sphere)
        for capsule in self.capsules:
            print(capsule)
        for mesh in self.meshes:
            print(mesh)
        for name, polygons, vertices in self.hulls:
            mesh_data = bpy.data.meshes.new(name=f'{name}_mesh')
            mesh_obj = bpy.data.objects.new(name=name, object_data=mesh_data)

            mesh_data.from_pydata(vertices * self.scale, [], polygons)
            mesh_data.update()
            meshes.append(mesh_obj)
        return meshes
