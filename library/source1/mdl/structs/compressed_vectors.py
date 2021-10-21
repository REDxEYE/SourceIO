import math

from . import ByteIO


class Quat:
    @staticmethod
    def read(reader: ByteIO):
        raise NotImplementedError('Override me')


class Quat64(Quat):
    @staticmethod
    def read(reader: ByteIO):
        b0 = reader.read_uint32()
        b1 = reader.read_uint32()
        xs = b0 & 0x1FFFFF
        ys = (((b1 & 0x03FF) << 11) | (b0 >> 21)) >> 0
        zs = (b1 >> 10) & 0x1FFFFF
        wn = -1 if (b1 & 0x80000000) else 1

        x = (xs - 0x100000) / 0x100000
        y = (ys - 0x100000) / 0x100000
        z = (zs - 0x100000) / 0x100000
        w = wn * math.sqrt(1.0 - x * x - y * y - z * z)
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
