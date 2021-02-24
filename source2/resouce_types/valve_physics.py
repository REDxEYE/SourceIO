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


class ValveCompiledPhysics(ValveCompiledFile):
    def __init__(self, path_or_file):
        super().__init__(path_or_file)
        self.data_block = self.get_data_block(block_name='DATA')[0]
        self.spheres = []
        self.capsules = []
        self.hulls = []
        self.meshes = []

    def parse_meshes(self):
        for part in self.data_block['m_parts']:
            shapes = part['m_rnShape']
            for sphere in shapes['m_spheres']:
                sphere_data = sphere['m_Sphere']
                self.spheres.append((sphere_data['m_vCenter'], sphere_data['m_flRadius']))
            for capsule in shapes['m_capsules']:
                capsule_data = capsule['m_Capsule']
                self.capsules.append((capsule_data['m_vCenter'], capsule_data['m_flRadius']))
            for hull in shapes['m_hulls']:
                hull_data = hull['m_Hull']
                vertices = np.array(hull_data['m_Vertices'], np.float32)
                faces = np.array(hull_data['m_Faces'].values(), np.int32)

            for mesh in shapes['m_meshes']:
                mesh_data = mesh['m_Mesh']
