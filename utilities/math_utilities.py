import math
from typing import List, Union, Tuple

import numpy as np

# According to the Valve documentation,
# one hammer unit is 1/16 of feet, and one feet is 30.48 cm
HAMMER_UNIT_TO_METERS = ((1 / 16) * 30.48) / 100


def clamp_value(value, min_value=0.0, max_value=1.0):
    return min(max_value, max(value, min_value))


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


def convert_to_radians(vector: Union[List[float], np.ndarray]):
    return np.deg2rad(vector)


def parse_hammer_vector(string: str) -> np.ndarray:
    return np.array([float(x) for x in string.split(" ") if x], np.float32)


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


def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1


def lerp_vec(v0, v1, t):
    return list(map(lambda x: lerp(x[0], x[1], t), zip(v0, v1)))
