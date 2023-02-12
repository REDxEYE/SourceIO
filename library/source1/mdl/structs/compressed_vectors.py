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

        x = (xs - 1048576) * (1 / 1048576.5)
        y = (ys - 1048576) * (1 / 1048576.5)
        z = (zs - 1048576) * (1 / 1048576.5)
        w = wn * math.sqrt(1.0 - x * x - y * y - z * z)
        return x, y, z, w


class Quat48(Quat):
    @staticmethod
    def read(buffer: Buffer):
        raw_x = buffer.read_uint16()
        raw_y = buffer.read_uint16()
        raw_z = buffer.read_uint16()
        x = (raw_x - 32768) * (1 / 32768)
        y = (raw_y - 32768) * (1 / 32768)
        z = ((raw_z & 0x7FFF) - 16384) * (1 / 16384)
        w_neg = (raw_z >> 15) & 0x1
        w = math.sqrt(1 - x * x - y * y - z * z)
        if w_neg:
            w = -w
        return x, y, z, w


class Quat48S(Quat):
    SCALE48S = 23168.0
    SHIFT48S = 16384

    @staticmethod
    def read(buffer: Buffer):
        quat = [0, 0, 0, 0]
        data = buffer.read_fmt('3H')
        a = data[0] & 0x7FFF
        offset_h = data[0] >> 15
        b = data[1] & 0x7FFF
        offset_l = data[1] >> 15
        c = data[2] & 0x7FFF
        d_neg = data[2] >> 15

        ia = offset_l + offset_h * 2
        ib = (ia + 1) % 4
        ic = (ia + 2) % 4
        id = (ia + 3) % 4

        quat[ia] = (a - Quat48S.SHIFT48S) * (1 / Quat48S.SCALE48S)
        quat[ib] = (b - Quat48S.SHIFT48S) * (1 / Quat48S.SCALE48S)
        quat[ic] = (c - Quat48S.SHIFT48S) * (1 / Quat48S.SCALE48S)
        quat[id] = math.sqrt(1.0 - quat[ia] * quat[ia] - quat[ib] * quat[ib] - quat[ic] * quat[ic])
        if d_neg:
            quat[id] = -quat[id]
        return quat
