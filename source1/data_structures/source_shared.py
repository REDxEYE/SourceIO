import math
import random
import struct

from ...byte_io_mdl import ByteIO


class SourceVector:
    def __init__(self, init_vec=None):
        if init_vec:
            self.x, self.y, self.z = init_vec
        else:
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    def read(self, reader: ByteIO):
        self.x, self.y, self.z = reader.read_fmt('fff')

        return self

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __add__(self, other):
        return SourceVector(
            [self.x + other.x, self.y + other.y, self.z + other.z])

    def __sub__(self, other):
        return SourceVector(
            [self.x - other.x, self.y - other.y, self.z - other.z])

    def to_degrees(self):
        return SourceVector(init_vec=[math.degrees(
            self.x), math.degrees(self.y), math.degrees(self.z)])

    @property
    def as_list(self):
        return [self.x, self.y, self.z]

    @property
    def as_string_smd(self):
        return "{:.6f} {:.6f} {:.6f}".format(self.x, self.y, self.z)

    def as_rounded(self, n):
        return "{} {} {}".format(
            round(self.x, n), round(self.y, n), round(self.z, n))

    @property
    def as_string(self):
        return " X:{} Y:{} Z:{}".format(self.x, self.y, self.z)

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
        return SourceVector(init_vec=[x, y, z])

    def __str__(self):
        return "<Vector 3D X:{} Y:{} Z:{}".format(self.x, self.y, self.z)

    def __repr__(self):
        return "<Vector 3D X:{} Y:{} Z:{}".format(self.x, self.y, self.z)


class SourceQuaternion:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 0.0

    def read(self, reader: ByteIO):
        self.x, self.y, self.z, self.w = reader.read_fmt('ffff')
        return self

    def __str__(self):
        return "<Quaternion X:{} Y:{} Z:{} W:{}".format(
            self.x, self.y, self.z, self.w)

    def __repr__(self):
        return "<Quaternion X:{} Y:{} Z:{} W:{}".format(
            self.x, self.y, self.z, self.w)


class SourceFloat16bits:
    float32bias = 127
    float16bias = 15
    max_float_16bits = 65504.0
    half_denorm = (1.0 / 16384.0)

    def __init__(self):
        self.value_16bit = 0

    def read(self, reader: ByteIO):
        self.value_16bit = reader.read_uint16()
        return self

    @property
    def float_value(self):
        sign = self.get_sign(self.value_16bit)
        if sign == 1:
            float_sign = -1
        else:
            float_sign = 0
        if self.is_infinity(self.value_16bit):
            return self.max_float_16bits * float_sign
        if self.is_nan(self.value_16bit):
            return 0
        mantissa = self.get_mantissa(self.value_16bit)
        biased_exponent = self.get_biased_exponent(self.value_16bit)
        if biased_exponent == 0 and mantissa != 0:
            float_mantissa = mantissa / 1024.0
            result = float_sign * float_mantissa * self.half_denorm
        else:
            result = self.get_single()
        return result

    @staticmethod
    def get_mantissa(value):
        return value & 0x3FF

    @staticmethod
    def get_biased_exponent(value):
        return (value & 0x7C00) >> 10

    @staticmethod
    def get_sign(value):
        return (value & 0x8000) >> 15

    def get_single(self):
        bits_result = IntegerAndSingleUnion()
        bits_result.i = 0
        mantissa = self.get_mantissa(self.value_16bit)
        biased_exponent = self.get_biased_exponent(self.value_16bit)
        sign = self.get_sign(self.value_16bit)
        result_mantissa = mantissa << (23 - 10)
        if biased_exponent == 0:
            result_biased_exponent = 0
        else:
            result_biased_exponent = (
                biased_exponent -
                self.float16bias +
                self.float32bias) << 23
        result_sign = sign << 31

        bits_result.i = result_sign | result_biased_exponent | result_mantissa

        return bits_result.s

    def is_infinity(self, value):
        mantissa = self.get_mantissa(value)
        biased_exponent = self.get_biased_exponent(value)
        return (biased_exponent == 31) and (mantissa == 0)

    def is_nan(self, value):
        mantissa = self.get_mantissa(value)
        biased_exponent = self.get_biased_exponent(value)
        return (biased_exponent == 31) and (mantissa != 0)

    def __str__(self):
        return self.float_value

    def __repr__(self):
        return self.float_value


class IntegerAndSingleUnion:
    def __init__(self):
        self.i = 0

    @property
    def s(self):
        a = struct.pack('!I', self.i)
        return struct.unpack('!f', a)[0]


class SourceVertex:
    def __init__(self):
        self.boneWeight = SourceBoneWeight()
        self.position = SourceVector()

        self.normal = SourceVector()
        self.texCoordX = 0
        self.texCoordY = 0

    def read(self, reader: ByteIO):
        self.boneWeight.read(reader)
        self.position.read(reader)
        self.normal.read(reader)
        self.texCoordX = reader.read_float()
        self.texCoordY = reader.read_float()
        return self

    def __str__(self):
        return "<Vertex pos:{}>".format(self.position.as_string)

    def __repr__(self):
        return self.__str__()


class SourceMdlTexture:
    def __init__(self):
        self.nameOffset = 0
        self.flags = 0
        self.used = 0
        self.unused1 = 0
        self.materialP = 0
        self.clientMaterialP = 0
        self.unused = []  # len 10
        self.thePathFileName = 'texture' + str(random.randint(0, 256))

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.nameOffset = reader.read_uint32()
        self.thePathFileName = reader.read_from_offset(
            entry + self.nameOffset, reader.read_ascii_string)
        self.flags = reader.read_uint32()
        self.used = reader.read_uint32()
        self.unused1 = reader.read_uint32()
        self.materialP = reader.read_uint32()
        self.clientMaterialP = reader.read_uint32()
        self.unused = [reader.read_uint32() for _ in range(5)]

    def __repr__(self):
        return "<SourceModel texture name:{}>".format(self.thePathFileName)


class SourceBoneWeight:
    def __init__(self):
        self.weight = []
        self.bone = []
        self.boneCount = 0

    def read(self, reader: ByteIO):
        self.weight = reader.read_fmt('fff')
        self.bone = reader.read_fmt('BBB')
        self.boneCount = reader.read_uint8()
        return self

    def __str__(self):
        return '<BoneWeight Weight: {} Bone: {} BoneCount: {}>'.format(
            self.weight, self.bone, self.boneCount)

    def __repr__(self):
        return self.__str__()
