import math
from typing import List, Union, Tuple

import numpy as np

# According to the Valve documentation,
# one hammer unit is 1/16 of feet, and one feet is 30.48 cm
SOURCE1_HAMMER_UNIT_TO_METERS = ((1 / 16) * 30.48) / 100
# one hammer unit is 1/12 of feet, and one feet is xx.xx cm
SOURCE2_HAMMER_UNIT_TO_METERS = ((1 / 12) * 30.48) / 100

EPSILON = 0.000001


def clamp_value(value, min_value=0.0, max_value=1.0):
    return min(max_value, max(value, min_value))


def quaternion_to_euler_angle(w, x, y, z):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    x = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    z = math.degrees(math.atan2(t3, t4))

    return x, y, z


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


# extracted from scipy Rotation class
def matrix_to_quat(matrix):
    is_single = False
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim not in [2, 3] or matrix.shape[-2:] != (3, 3):
        raise ValueError("Expected `matrix` to have shape (3, 3) or "
                         "(N, 3, 3), got {}".format(matrix.shape))

    # If a single matrix is given, convert it to 3D 1 x 3 x 3 matrix but
    # set self._single to True so that we can return appropriate objects in
    # the `to_...` methods
    if matrix.shape == (3, 3):
        matrix = matrix.reshape((1, 3, 3))
        is_single = True

    num_rotations = matrix.shape[0]

    decision_matrix = np.empty((num_rotations, 4))
    decision_matrix[:, :3] = matrix.diagonal(axis1=1, axis2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
    choices = decision_matrix.argmax(axis=1)

    quat = np.empty((num_rotations, 4))

    ind = np.nonzero(choices != 3)[0]
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * matrix[ind, i, i]
    quat[ind, j] = matrix[ind, j, i] + matrix[ind, i, j]
    quat[ind, k] = matrix[ind, k, i] + matrix[ind, i, k]
    quat[ind, 3] = matrix[ind, k, j] - matrix[ind, j, k]

    ind = np.nonzero(choices == 3)[0]
    quat[ind, 0] = matrix[ind, 2, 1] - matrix[ind, 1, 2]
    quat[ind, 1] = matrix[ind, 0, 2] - matrix[ind, 2, 0]
    quat[ind, 2] = matrix[ind, 1, 0] - matrix[ind, 0, 1]
    quat[ind, 3] = 1 + decision_matrix[ind, -1]

    quat /= np.linalg.norm(quat, axis=1)[:, None]

    if is_single:
        return quat[0]
    else:
        return quat


def quat_to_matrix(quat):
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    matrix = np.empty((3, 3))

    matrix[0, 0] = x2 - y2 - z2 + w2
    matrix[1, 0] = 2 * (xy + zw)
    matrix[2, 0] = 2 * (xz - yw)

    matrix[0, 1] = 2 * (xy - zw)
    matrix[1, 1] = - x2 + y2 - z2 + w2
    matrix[2, 1] = 2 * (yz + xw)

    matrix[0, 2] = 2 * (xz + yw)
    matrix[1, 2] = 2 * (yz - xw)
    matrix[2, 2] = - x2 - y2 + z2 + w2

    return matrix


def euler_to_matrix(theta):
    r_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    r_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    r_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    return np.dot(r_z, np.dot(r_y, r_x))


def euler_to_quat(euler: np.ndarray):
    euler *= 0.5
    roll, pitch, yaw = euler
    sin_r = math.sin(roll)
    cos_r = math.cos(roll)
    sin_p = math.sin(pitch)
    cos_p = math.cos(pitch)
    sin_y = math.sin(yaw)
    cos_y = math.cos(yaw)
    quat = np.zeros((4,), np.float32)
    quat[0] = sin_r * cos_p * cos_y - cos_r * sin_p * sin_y
    quat[1] = cos_r * sin_p * cos_y + sin_r * cos_p * sin_y
    quat[2] = cos_r * cos_p * sin_y - sin_r * sin_p * cos_y
    quat[3] = cos_r * cos_p * cos_y + sin_r * sin_p * sin_y
    return quat


def quat_slerp(a, b, factor):
    ax = a[0]
    ay = a[1]
    az = a[2]
    aw = a[3]
    bx = b[0]
    by = b[1]
    bz = b[2]
    bw = b[3]

    omega, cosom, sinom, scale0, scale1 = 0, 0, 0, 0, 0

    cosom = ax * bx + ay * by + az * bz + aw * bw
    if cosom < 0.0:
        cosom = -cosom
        bx = -bx
        by = -by
        bz = -bz
        bw = -bw

    if 1.0 - cosom > EPSILON:
        omega = math.acos(cosom)
        sinom = math.sin(omega)
        scale0 = math.sin((1.0 - factor) * omega) / sinom
        scale1 = math.sin(factor * omega) / sinom
    else:
        scale0 = 1.0 - factor
        scale1 = factor
    out = np.zeros((4,), np.float32)
    out[0] = scale0 * ax + scale1 * bx
    out[1] = scale0 * ay + scale1 * by
    out[2] = scale0 * az + scale1 * bz
    out[3] = scale0 * aw + scale1 * bw

    return out


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


def sizeof_fmt(num):
    unit_list = list(zip(['bytes', 'kB', 'MB', 'GB', 'TB', 'PB'], [0, 0, 1, 2, 2, 2]))
    """Human friendly file size"""
    if num > 1:
        exponent = min(int(math.log(num, 1024)), len(unit_list) - 1)
        quotient = float(num) / 1024 ** exponent
        unit, num_decimals = unit_list[exponent]
        format_string = '{:.%sf} {}' % (num_decimals)
        return format_string.format(quotient, unit)
    if num == 0:
        return '0 bytes'
    if num == 1:
        return '1 byte'


def vector_normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


class UnitPlane:
    def __init__(self):
        self.normal = np.zeros((3,))
        self.d = 0.0

    def normal_from_tri(self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray):
        scratch = np.zeros((2, 3))
        scratch[0] = p1 - p0
        scratch[1] = p2 - p0
        normal = np.cross(scratch[0], scratch[1])
        self.normal[:] = vector_normalize(normal)
        self.d = -np.dot(self.normal, p0)

    def distance_vec3(self, p):
        return np.dot(p, self.normal) + self.d
