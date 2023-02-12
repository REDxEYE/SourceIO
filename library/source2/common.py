import numpy as np


def lerp(a, b, f):
    return (a * (1.0 - f)) + (b * f)


def convert_normals(input_array: np.ndarray):
    # X Y Z
    input_array = input_array.astype(np.int32)
    output = np.zeros((len(input_array), 3), dtype=np.float32)
    xs = input_array[:, 0]
    ys = input_array[:, 1]

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
