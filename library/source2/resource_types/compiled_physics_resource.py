from typing import Collection, List

import numpy as np

from ...shared.intermidiate_data.bone import Bone
from ..exceptions import MissingBlock
from .resource import CompiledResource


class CompiledPhysicsResource(CompiledResource):
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
        data, = self.get_data_block(block_name='DATA')
        if data is None:
            raise MissingBlock('Required block "DATA" is missing')
        spheres = []
        capsules = []
        hulls = []
        for part in data['m_parts']:
            shapes = part['m_rnShape']
            for sphere in shapes['m_spheres']:
                sphere_data = sphere['m_Sphere']
                spheres.append((sphere_data['m_vCenter'], sphere_data['m_flRadius']))
            for capsule in shapes['m_capsules']:
                capsule_data = capsule['m_Capsule']
                capsules.append((capsule_data['m_vCenter'], capsule_data['m_flRadius']))
            for n, hull in enumerate(shapes['m_hulls']):
                hull_data = hull['m_Hull']
                hull_name = hull.get('m_UserFriendlyName', None) or f'hull_{n}'
                vertices = np.array(hull_data['m_Vertices'], np.float32)
                polygons = []
                for face in hull_data['m_Faces']:
                    edges = self.gather_edges(face['m_nEdge'], hull_data['m_Edges'])
                    polygons.append(edges)
                hulls.append((hull_name, polygons, vertices))
            for mesh in shapes['m_meshes']:
                mesh_data = mesh['m_Mesh']
        return spheres, capsules, hulls,
