import math

from ....utilities.byte_io_mdl import ByteIO


class Quat:
    @staticmethod
    def read(reader: ByteIO):
        raise NotImplementedError('Override me')


class Quat64(Quat):
    @staticmethod
    def read(reader: ByteIO):
        raw_value = reader.read_uint64()

        x = ((raw_value & 0x1FFFFF) - 1048576) / 1048576.5
        y = ((raw_value >> 21 & 0x1FFFFF) - 1048576) / 1048576.5
        z = ((raw_value >> 42 & 0x1FFFFF) - 1048576) / 1048576.5
        w = math.sqrt(1 - x * x - y * y - z * z)
        w_neg = raw_value >> 61 & 0x1
        if w_neg:
            w *= -1
        return x, y, z, w


class Quat48(Quat):
    @staticmethod
    def read(reader: ByteIO):
        x = (reader.read_uint16() - 32768) / 32768
        y = (reader.read_uint16() - 32768) / 32768
        raw_z = reader.read_uint16()
        z = ((raw_z & 0x7FFF) - 16384) / 16384
        w_neg = (raw_z >> 15) & 0x1
        w = math.sqrt(1 - x * x - y * y - z * z)
        if w_neg:
            w *= -1
        return x, y, z, w
