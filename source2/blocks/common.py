import importlib
import math
from enum import IntEnum

from ...byte_io_mdl import ByteIO
from .dummy import DataBlock


class KeyValueDataType(IntEnum):
    STRUCT = 1
    ENUM = 2
    POINTER = 3
    STRING = 4
    UBYTE = 10
    BYTE = 11
    SHORT = 12
    USHORT = 13
    INTEGER = 14
    UINTEGER = 15
    INT64 = 16
    UINT64 = 17
    FLOAT = 18
    VECTOR2 = 21
    VECTOR3 = 22
    VECTOR4 = 23
    QUATERNION = 25
    Fltx4 = 27
    COLOR = 28  # Standard RGBA, 1 byte per channel
    BOOLEAN = 30
    NAME = 31  # Also used for notes as well? idk... seems to be some kind of special string
    Matrix3x4 = 33
    Matrix3x4a = 36
    CTransform = 40
    Vector4D_44 = 44

    UNKNOWN = -1
    # def _missing_(cls, *args):
    #     return KeyValueDataType.UNKNOWN


def short_to_float(value):
    value = int(value)
    s = (value >> 14) & 2  # sign*2
    e = (value >> 10) & 31  # exponent
    m = (value & 1023)  # mantissa
    if e == 0:
        # either zero or a subnormal number
        if m != 0:
            return (1 - s) * pow(2, -14) * (m / 1024)
        elif s == 0:
            return -0
        else:
            return 0
    elif e != 31:
        # normal number
        return (1 - s) * pow(2, e - 15) * (1 + m / 1024)
    elif (value & 1023) != 0:
        return -float('Inf')
    elif value < 0:
        return float('Inf')
    else:
        return float('Inf')


def lerp(a, b, f):
    return (a * (1.0 - f)) + (b * f)


class SourceVector:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __len__(self):
        return 3

    def __iter__(self):
        return iter(self.as_list)

    def read(self, reader: ByteIO):
        self.x, self.y, self.z = reader.read_fmt('fff')

        return self

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __add__(self, other):
        return SourceVector(*[self.x + other.x, self.y + other.y, self.z + other.z])

    def __sub__(self, other):
        return SourceVector(*[self.x - other.x, self.y - other.y, self.z - other.z])

    def to_degrees(self):
        return SourceVector(*[math.degrees(self.x), math.degrees(self.y), math.degrees(self.z)])

    @property
    def as_list(self):
        return [self.x, self.y, self.z]

    @property
    def as_string_smd(self):
        return "{:.6f} {:.6f} {:.6f}".format(self.x, self.y, self.z)

    def as_rounded(self, n):
        return "{} {} {}".format(round(self.x, n), round(self.y, n), round(self.z, n))

    @property
    def as_string(self):
        return " X:{} Y:{} Z:{}".format(self.x, self.y, self.z)

    @staticmethod
    def convert(x, y):
        # print('before',self)
        out_normal = SourceVector()
        z_sing = -math.floor((x - 128) / 128)
        t_sing = -math.floor((y - 128) / 128)
        x_abs = abs(x - 128) - z_sing
        y_abs = abs(y - 128) - t_sing
        x_sing = -math.floor((x_abs - 64) / 64)
        y_sing = -math.floor((y_abs - 64) / 64)
        out_normal.x = (abs(x_abs - 64) - x_sing) / 64
        out_normal.y = (abs(y_abs - 64) - y_sing) / 64
        out_normal.z = 1 - out_normal.x - out_normal.y
        out_normal.normalize()
        out_normal.x *= lerp(1, -1, x_sing)
        out_normal.y *= lerp(1, -1, y_sing)
        out_normal.z *= lerp(1, -1, z_sing)
        return out_normal

    def magnitude(self):
        magn = math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        return magn

    def normalize(self):
        magn = self.magnitude()
        if magn == 0:
            return self
        self.x = self.x / magn
        self.y = self.y / magn
        self.z = self.z / magn
        return self

    @property
    def as_normalized(self):
        magn = self.magnitude()
        if magn == 0:
            return self
        x = self.x / magn
        y = self.y / magn
        z = self.z / magn
        return SourceVector(*[x, y, z])

    def __repr__(self):
        return "<Vector 3D X:{:3} Y:{:3} Z:{:3} W:{:3}>".format(self.x, self.y, self.z, self.w)


class SourceVector2D:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def read(self, reader: ByteIO):
        self.x = reader.read_float()
        self.y = reader.read_float()
        return self

    def __repr__(self):
        return "<Vector 3D X:{} Y:{}>".format(self.x, self.y)


class SourceVector4D:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    @property
    def to_floats(self):
        return SourceVector4D(self.x / 255, self.y / 255, self.z / 255, self.w / 255)

    def read(self, reader: ByteIO):
        self.x = reader.read_float()
        self.y = reader.read_float()
        self.z = reader.read_float()
        self.w = reader.read_float()
        return self

    @property
    def as_list(self):
        return [self.x, self.y, self.z, self.w]

    def __repr__(self):
        return "<Vector 4D X:{:2f} Y:{:2f} Z:{:2f} W:{:2f}>".format(self.x, self.y, self.z, self.w)


class SourceBoneWeight:
    def __init__(self):
        self.weight = []
        self.bone = []
        self.boneCount = b"\x00"

    def __repr__(self):
        return '<SourceBoneWeight bones:{} weights:{}>'.format(self.bone, self.weight)


class SourceVertex:
    def __init__(self):
        self.position = []
        self.normal = []
        self.uvs = {}
        self.bone_weight = SourceBoneWeight()
        self.tangent = []
        self.lightmap = []
        self.color = []

    def __str__(self):
        return "<Vertex pos:{}>".format(self.position)

    def __repr__(self):
        return self.__str__()


class Matrix(DataBlock):
    mat_rows = 0
    mat_cols = 0

    def __init__(self, cols, rows):
        self.mat_rows = rows
        self.mat_cols = cols
        self.mat = []
        col = []
        for _ in range(rows):
            for _ in range(cols):
                col.append(None)
            self.mat.append(col)

    def read(self, reader: ByteIO):
        for x in range(self.mat_cols):
            for y in range(self.mat_rows):
                self.mat[x][y] = reader.read_float()

    def __repr__(self):
        return '<Matrix{}x{}>'.format(self.mat_cols, self.mat_rows)


class CTransform(DataBlock):

    def __init__(self):
        self.quat = SourceVector4D()
        self.pos = SourceVector()

    def read(self, reader: ByteIO):
        self.quat.read(reader)
        self.pos.read(reader)

    def __repr__(self):
        return '<CTransform pos:{} quat:{}>'.format(self.pos.as_rounded(3), self.quat)


kv_type_to_c_type = {
    1: 'STRUCT',
    2: 'int',
    3: 'int',
    4: 'char*',
    10: 'unsigned byte',
    11: 'byte',
    12: 'short',
    13: 'unsigned short',
    14: 'int',
    15: 'unsigned int',
    16: 'long long',
    17: 'unsigned long long',
    18: 'float',
    21: 'vector2',
    22: 'vector3',
    23: 'vector4',
    25: 'quaternion',
    28: 'RGB',
    30: 'byte',
    31: 'char*'
}
