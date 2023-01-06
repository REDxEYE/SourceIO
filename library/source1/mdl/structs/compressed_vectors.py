import math

from ....utils import Buffer


class Quat:
    @staticmethod
    def read(buffer: Buffer):
        raise NotImplementedError('Override me')


class Quat64(Quat):
    @staticmethod
    def read(buffer: Buffer):
        b0 = buffer.read_uint32()
        b1 = buffer.read_uint32()
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
    def read(buffer: Buffer):
        x = (buffer.read_uint16() - 32768) / 32768
        y = (buffer.read_uint16() - 32768) / 32768
        raw_z = buffer.read_uint16()
        z = ((raw_z & 0x7FFF) - 16384) / 16384
        w_neg = (raw_z >> 15) & 0x1
        w = math.sqrt(1 - x * x - y * y - z * z)
        if w_neg:
            w *= -1
        return x, y, z, w


class Quat48S(Quat):
    SCALE48S = 23168.0
    SHIFT48S = 16384

    @staticmethod
    def read(buffer: Buffer):
        data = buffer.read_fmt('3H')
        ia = (data[1] & 1) + (data[0] & 1) * 2
        ib = (ia + 1) % 4
        ic = (ia + 2) % 4
        id = (ia + 3) % 4
        quat = [0, 0, 0, 0]
        quat[ia] = ((data[0] >> 1) - Quat48S.SCALE48S) * (1 / Quat48S.SCALE48S)
        quat[ib] = ((data[1] >> 1) - Quat48S.SCALE48S) * (1 / Quat48S.SCALE48S)
        quat[ic] = ((data[2] >> 1) - Quat48S.SCALE48S) * (1 / Quat48S.SCALE48S)
        quat[id] = math.sqrt(1.0 - quat[ia] * quat[ia] - quat[ib] * quat[ib] - quat[ic] * quat[ic])
        if data[2] & 1:
            quat[id] = -quat[id]
        return quat
