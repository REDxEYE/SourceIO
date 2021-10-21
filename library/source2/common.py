import numpy as np

from ..utils.byte_io_mdl import ByteIO


def short_to_float(value):
    value = int(value)
    s = (value >> 14) & 2  # sign*2
    e = (value >> 10) & 31  # exponent
    m = (value & 1023)  # mantissa
    if e == 0:
        # either zero or a subnormal number
        if m != 0:
            return (1 - s) * pow(2, -14) * (m / 1024)
        else:
            return 0
    elif e != 31:
        # normal number
        return (1 - s) * pow(2, e - 15) * (1 + m / 1024)
    elif value & 1023 != 0:
        return -float('Inf')
    else:
        return float('Inf')


def lerp(a, b, f):
    return (a * (1.0 - f)) + (b * f)


def convert_normals(inpurt_array: np.ndarray):
    # X Y Z
    inpurt_array = inpurt_array.astype(np.int32)
    output = np.zeros((len(inpurt_array), 3), dtype=np.float32)
    xs = inpurt_array[:, 0]
    ys = inpurt_array[:, 1]

    z_signs = -np.floor((xs - 128) / 128)
    t_signs = -np.floor((ys - 128) / 128)

    x_abss = np.abs(xs - 128) - z_signs
    y_abss = np.abs(ys - 128) - t_signs

    x_sings = -np.floor((x_abss - 64) / 64)
    y_sings = -np.floor((y_abss - 64) / 64)

    output[:, 0] = (np.abs(x_abss - 64) - x_sings) / 64
    output[:, 1] = (np.abs(y_abss - 64) - y_sings) / 64
    output[:, 2] = (1 - output[:, 0]) - output[:, 1]

    sq = np.sqrt(np.sum(output ** 2, 1))
    output[:, 0] /= sq
    output[:, 1] /= sq
    output[:, 2] /= sq

    output[:, 0] *= lerp(1, -1, np.abs(x_sings))
    output[:, 1] *= lerp(1, -1, np.abs(y_sings))
    output[:, 2] *= lerp(1, -1, np.abs(z_signs))
    return output


def magnitude(array: np.ndarray):
    array = np.sum(array ** 2)
    return np.sqrt(array)


def normalize(array: np.ndarray):
    magn = magnitude(array)
    if magn == 0:
        return array
    return array / magn


class Matrix:
    def __init__(self, cols, rows):
        self.n_rows = rows
        self.n_cols = cols
        self.mat: np.ndarray = np.zeros((rows, cols))

    def read(self, reader: ByteIO):
        self.mat = np.frombuffer(reader.read(self.n_cols * self.n_cols * 4), dtype=np.float32)
        self.mat = self.mat.reshape((self.n_cols, self.n_cols))

    def __repr__(self):
        return '<Matrix{}x{}>'.format(self.n_cols, self.n_rows)


class CTransform:

    def __init__(self):
        self.quat = []
        self.pos = []

    def read(self, reader: ByteIO):
        self.quat = reader.read_fmt('4f')
        self.pos = reader.read_fmt('3f')

    def __repr__(self):
        return f'<CTransform pos:{self.pos} quat:{self.quat}>'


