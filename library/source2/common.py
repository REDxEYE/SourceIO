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


def convert_normals_2(input_array: np.ndarray):
    # X Y Z
    input_array = input_array.ravel()
    output = np.zeros((len(input_array), 3), dtype=np.float32)
    # sign_bit = input_array & 1
    # tbits = (input_array >> 1) & 0x7ff
    xbits = (input_array >> 12) & 0x3ff
    ybits = (input_array >> 22) & 0x3ff

    n_packed_frame_x = (xbits / 1023.0) * 2.0 - 1.0
    n_packed_frame_y = (ybits / 1023.0) * 2.0 - 1.0
    derived_normal_z = 1.0 - np.abs(n_packed_frame_x) - np.abs(n_packed_frame_y)
    negative_z_compensation = np.clip(-derived_normal_z, 0.0, 1.0)
    unpacked_normal_x_positive = np.zeros(len(input_array), float)
    unpacked_normal_x_positive[(n_packed_frame_x >= 0.0)] = 1.0
    unpacked_normal_y_positive = np.zeros(len(input_array), float)
    unpacked_normal_y_positive[(n_packed_frame_y >= 0.0)] = 1.0
    n_packed_frame_x += negative_z_compensation * (
            1 - unpacked_normal_x_positive) + -negative_z_compensation * unpacked_normal_x_positive
    n_packed_frame_y += negative_z_compensation * (
        1 - unpacked_normal_y_positive) + -negative_z_compensation * unpacked_normal_y_positive

    output[:, 0] = n_packed_frame_x
    output[:, 1] = n_packed_frame_y
    output[:, 2] = derived_normal_z

    sq = np.sqrt(np.sum(output ** 2, 1))
    output[:, 0] /= sq
    output[:, 1] /= sq
    output[:, 2] /= sq
    return output
