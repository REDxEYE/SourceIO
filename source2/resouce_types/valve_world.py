import os.path
import random
from pathlib import Path
from typing import List

import bpy
import math
from mathutils import Vector, Matrix, Quaternion, Euler

from ...utilities.path_utilities import backwalk_file_resolver
from ..common import SourceVector, SourceVertex, SourceVector4D
from ..source2 import ValveFile
from .valve_model import ValveModel


class ValveWorld:
    def __init__(self, vmdl_path):
        self.valve_file = ValveFile(vmdl_path)
        self.valve_file.read_block_info()
        self.valve_file.check_external_resources()

    def load(self, invert_uv=False):
        data_block = self.valve_file.get_data_block(block_name='DATA')[0]
        if data_block:
            for world_node_t in data_block.data['m_worldNodes']:
                node_path = world_node_t['m_worldNodePrefix'] + '.vwnod_c'
                full_node_path = backwalk_file_resolver(self.valve_file.filepath.parent, node_path)
                world_node_file = ValveFile(full_node_path)
                world_node_file.read_block_info()
                world_node_file.check_external_resources()
                world_data = world_node_file.get_data_block(block_name="DATA")[0]
                for n, static_object in enumerate(world_data.data['m_sceneObjects']):
                    model_file = world_node_file.get_child_resource(static_object['m_renderableModel'])
                    if model_file is not None:
                        print(f"Loading ({n}/{len(world_data.data['m_sceneObjects'])}){model_file.filepath.stem} mesh")
                        model = ValveModel("", model_file)
                        to_remove = Path(node_path)
                        model.load_mesh(invert_uv, True, to_remove.stem+"_")
                        mat_rows = static_object['m_vTransform']  # type:List[SourceVector4D]
                        transform_mat = Matrix([mat_rows[0].as_list, mat_rows[1].as_list, mat_rows[2].as_list]).to_4x4()
                        for obj in model.objects:
                            obj.matrix_world = transform_mat
