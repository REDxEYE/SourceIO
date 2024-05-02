#!/usr/bin/env python3

import numpy as np


def linear_interp(v0, v1, d):
    """Basic Linear Interpolation"""
    return v0 * (1 - d) + v1 * d


def interp2d(
        Q: list[np.ndarray],
        dy: np.ndarray,
        dx: np.ndarray,
        mode: str = "bilinear",
) -> np.ndarray:
    """Naive Interpolation
    (y,x): target pixel
    mode: interpolation mode
    """
    q00, q10, q01, q11 = Q
    if mode == "bilinear":
        f0 = linear_interp(q00, q01, dx)
        f1 = linear_interp(q10, q11, dx)
        return linear_interp(f0, f1, dy)
    else:
        raise NotImplementedError


def grid_sample(
        img: np.ndarray,
        grid: np.ndarray,
        mode: str = "bilinear",
) -> np.ndarray:
    """Optimized Numpy Grid Sample using advanced indexing and avoiding unnecessary type conversions."""
    channels, h_in, w_in = img.shape
    _, h_out, w_out = grid.shape

    if img.dtype not in (np.uint8, np.float32):
        raise ValueError(f"{img.dtype} is not supported")

    # Initialize output image
    if mode == "bilinear":
        y_min = np.floor(grid[0]).astype(int) % h_in
        x_min = np.floor(grid[1]).astype(int) % w_in
        y_max = (y_min + 1) % h_in
        x_max = (x_min + 1) % w_in
        y_d = grid[0] - np.floor(grid[0])
        x_d = grid[1] - np.floor(grid[1])

        Q00 = img[:, y_min, x_min]
        Q10 = img[:, y_max, x_min]
        Q01 = img[:, y_min, x_max]
        Q11 = img[:, y_max, x_max]

        f0 = (1 - y_d) * Q00 + y_d * Q10
        f1 = (1 - y_d) * Q01 + y_d * Q11
        out = (1 - x_d) * f0 + x_d * f1

    elif mode == "nearest":
        y_nearest = np.rint(grid[0]).astype(int) % h_in
        x_nearest = np.rint(grid[1]).astype(int) % w_in
        out = img[:, y_nearest, x_nearest]

    else:
        raise ValueError("{} is not available".format(mode))

    # out = np.where(out >= _max, _max, out)
    # out = np.where(out < _min, _min, out)
    out = out.reshape(channels, h_out, w_out)
    return out.astype(img.dtype)
