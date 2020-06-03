import math
from typing import List, Union

import numpy as np


def convert_rotation_matrix_to_degrees(m0, m1, m2, m3, m4, m5, m8):
    angle_y = -math.asin(round(m2, 6))
    c = math.cos(angle_y)
    if abs(c) > 0.005:
        translate_x = m8 / c
        translate_y = -m5 / c
        angle_x = (math.atan2(translate_y, translate_x))
        translate_x = m0 / c
        translate_y = -m1 / c
    else:
        angle_x = 0
        translate_x = m4
        translate_y = m3
    angle_z = (math.atan2(translate_y, translate_x))
    return angle_x, angle_y, angle_z


def convert_rotation_source2_to_blender(source2_rotation: Union[List[float], np.ndarray]) -> List[float]:
    # XYZ -> ZXY
    return [math.radians(source2_rotation[2]), math.radians(source2_rotation[0]), math.radians(source2_rotation[1])]


def parse_source2_hammer_vector(string: str) -> np.ndarray:
    return np.array([float(x) for x in string.split(" ")])


# def vector_i_transform(input: 'SourceVector',
#                        matrix_c0: 'SourceVector',
#                        matrix_c1: 'SourceVector',
#                        matrix_c2: 'SourceVector',
#                        matrix_c3: 'SourceVector'):
#     from SourceIO.source1.data_structures.source_shared import SourceVector
#     temp = SourceVector()
#     output = SourceVector()
#
#     temp.x = input.x - matrix_c3.x
#     temp.y = input.y - matrix_c3.y
#     temp.z = input.z - matrix_c3.z
#
#     output.x = temp.x * matrix_c0.x + temp.y * matrix_c0.y + temp.z * matrix_c0.z
#     output.y = temp.x * matrix_c1.x + temp.y * matrix_c1.y + temp.z * matrix_c1.z
#     output.z = temp.x * matrix_c2.x + temp.y * matrix_c2.y + temp.z * matrix_c2.z
#
#     return output
