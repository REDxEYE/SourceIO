import math
from typing import List, Union, Tuple

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
    return [math.radians(source2_rotation[2]), math.radians(source2_rotation[0]),
            math.radians(source2_rotation[1])]


def convert_rotation_source1_to_blender(source2_rotation: Union[List[float], np.ndarray]) -> List[float]:
    # XYZ -> ZXY
    return [math.radians(source2_rotation[2]), math.radians(source2_rotation[0]),
            math.radians(source2_rotation[1])]


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


def lumen_to_candela_by_apex_angle(flux: float, angle: float):
    """
    Compute the luminous intensity from the luminous flux,
    assuming that the flux of <flux> is distributed equally around
    a cone with apex angle <angle>.
    Keyword parameters
    ------------------
    flux : value, engineer string or NumPy array
        The luminous flux in Lux.
    angle : value, engineer string or NumPy array
        The apex angle of the emission cone, in degrees
        For many LEDs, this is
    >>> lumen_to_candela_by_apex_angle(25., 120.)
    7.957747154594769
    """
    solid_angle = 2 * math.pi * (1. - math.cos((angle * math.pi / 180.) / 2.0))
    return flux / solid_angle


MAX_LIGHT_EFFICIENCY_EFFICACY = 683


def srgb_to_luminance(color: Union[List, Tuple]):
    return 0.2126729 * color[0] + 0.7151522 * color[1] + 0.072175 * color[2]


def watt_power_point(lumen, color):
    return lumen * ((1 / MAX_LIGHT_EFFICIENCY_EFFICACY) / srgb_to_luminance(color))


def watt_power_spot(lumen, color, cone):
    return lumen * (1 / (MAX_LIGHT_EFFICIENCY_EFFICACY * 2 * math.pi * (
            1 - math.cos(math.radians(cone) / 2))) * 4 * math.pi) / srgb_to_luminance(color)
