#!/usr/bin/env python3

from typing import Dict, List, Tuple, Union

import numpy as np

from . import numpy_grid_sample as numpy_func


def cube_list2h(cube_list: List[np.ndarray]) -> np.ndarray:
    assert len(cube_list) == 6
    assert sum(face.shape == cube_list[0].shape for face in cube_list) == 6
    return np.concatenate(cube_list, axis=-1)


def cube_dict2h(
        cube_dict: Dict[str, np.ndarray],
        face_k: Union[Tuple[str], List[str]] = ("F", "R", "B", "L", "U", "D"),
) -> np.ndarray:
    assert len(face_k) == 6
    return cube_list2h([cube_dict[k] for k in face_k])


def cube_dice2h(cube_dice: np.ndarray) -> np.ndarray:
    """dice to horizion

    params:
    - cube_dice: (C, H, W)
    """
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    w = cube_dice.shape[-2] // 3
    assert cube_dice.shape[-2] == w * 3 and cube_dice.shape[-1] == w * 4
    cube_h = np.zeros((cube_dice.shape[-3], w, w * 6), dtype=cube_dice.dtype)
    for i, (sx, sy) in enumerate(sxy):
        face = cube_dice[:, sy * w: (sy + 1) * w, sx * w: (sx + 1) * w]
        cube_h[:, :, i * w: (i + 1) * w] = face
    return cube_h


def _to_horizon(cubemap: np.ndarray, cube_format: str) -> np.ndarray:
    if cube_format == "horizon":
        pass
    elif cube_format == "list":
        cubemap = cube_list2h(cubemap)
    elif cube_format == "dict":
        cubemap = cube_dict2h(cubemap)
    elif cube_format == "dice":
        cubemap = cube_dice2h(cubemap)
    else:
        raise NotImplementedError("unknown cube_format")

    assert len(cubemap.shape) == 3
    assert cubemap.shape[-2] * 6 == cubemap.shape[-1]

    return cubemap


def _equirect_facetype(h: int, w: int) -> np.ndarray:
    """0F 1R 2B 3L 4U 5D"""

    tp = np.roll(
        np.arange(4).repeat(w // 4)[None, :].repeat(h, 0), 3 * w // 8, 1
    )

    # Prepare ceil mask
    mask = np.zeros((h, w // 4), bool)
    idx = np.linspace(-np.pi, np.pi, w // 4, dtype=np.float32) / 4
    idx = h // 2 - np.around(np.arctan(np.cos(idx)) * h / np.pi).astype(int)
    for i, j in enumerate(idx):
        mask[:j, i] = 1
    mask = np.roll(np.concatenate([mask] * 4, 1), 3 * w // 8, 1)

    tp[mask] = 4
    tp[np.flip(mask, 0)] = 5

    return tp.astype(np.int32)


def create_equi_grid(h_out: int, w_out: int) -> Tuple[np.ndarray, np.ndarray]:
    _dtype = np.float32
    theta = np.linspace(-np.pi, np.pi, num=w_out, dtype=_dtype)
    phi = np.linspace(np.pi, -np.pi, num=h_out, dtype=_dtype) / 2
    theta, phi = np.meshgrid(theta, phi)
    return theta, phi


def _run_single(
        cubemap: np.ndarray,  # `horizon`
        w_out: int,
        h_out: int,
        sampling_method: str,
        mode: str,
) -> np.ndarray:
    """Run a single batch of transformation"""
    w_face = cubemap.shape[-2]

    theta, phi = create_equi_grid(h_out, w_out)

    # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
    tp = _equirect_facetype(h_out, w_out)

    # xy coordinate map
    coor_x = np.zeros((h_out, w_out), dtype=np.float32)
    coor_y = np.zeros((h_out, w_out), dtype=np.float32)

    for i in range(6):
        mask = tp == i

        if i < 4:
            coor_x[mask] = 0.5 * np.tan(theta[mask] - np.pi * i / 2)
            coor_y[mask] = (
                    -0.5 * np.tan(phi[mask]) / np.cos(theta[mask] - np.pi * i / 2)
            )
        elif i == 4:
            c = 0.5 * np.tan(np.pi / 2 - phi[mask])
            coor_x[mask] = c * np.sin(theta[mask])
            coor_y[mask] = c * np.cos(theta[mask])
        elif i == 5:
            c = 0.5 * np.tan(np.pi / 2 - np.abs(phi[mask]))
            coor_x[mask] = c * np.sin(theta[mask])
            coor_y[mask] = -c * np.cos(theta[mask])

    # Final renormalize
    coor_x = np.clip(np.clip(coor_x + 0.5, 0, 1) * w_face, 0, w_face - 1)
    coor_y = np.clip(np.clip(coor_y + 0.5, 0, 1) * w_face, 0, w_face - 1)

    # change x axis of the x coordinate map
    for i in range(6):
        mask = tp == i
        coor_x[mask] = coor_x[mask] + w_face * i

    grid = np.stack((coor_y, coor_x), axis=0)
    grid_sample = getattr(numpy_func, sampling_method)
    equi = grid_sample(cubemap, grid, mode=mode)

    return equi


def run(
        cubemap: Union[np.ndarray, Dict[str, np.ndarray], List[np.ndarray]],
        cube_format: str,
        w_out: int,
        h_out: int,
        sampling_method: str,
        mode: str,
) -> np.ndarray:
    """Run cube to equirectangular image transformation

    params:
    - cubemap: np.ndarray
    - cube_format: ('dice', 'horizon', 'list', 'dict')
    - sampling_method: str
    - mode: str
    """

    # Convert all cubemap format to `horizon` and batched
    is_single = False
    if cube_format in ["dice", "horizon"]:
        if isinstance(cubemap, np.ndarray):
            # could be single or batched
            assert (
                    len(cubemap.shape) >= 3
            ), "input shape {} is not valid".format(cubemap.shape)
            if len(cubemap.shape) == 4:
                # batch
                cubemap = [_to_horizon(c, cube_format) for c in cubemap]
            else:
                # single
                is_single = True
                cubemap = [_to_horizon(cubemap, cube_format)]
        elif isinstance(cubemap, list):
            # could only be batched
            cubemap = [_to_horizon(c, cube_format) for c in cubemap]
        else:
            raise ValueError
    elif cube_format == "dict":
        if isinstance(cubemap, dict):
            # single
            is_single = True
            cubemap = [_to_horizon(cubemap, cube_format)]
        elif isinstance(cubemap, list):
            # batch
            cubemap = [_to_horizon(c, cube_format) for c in cubemap]
        else:
            raise ValueError
    elif cube_format == "list":
        if isinstance(cubemap[0], np.ndarray):
            # single
            is_single = True
            cubemap = [_to_horizon(cubemap, cube_format)]
        elif isinstance(cubemap[0], list):
            # batch
            cubemap = [_to_horizon(c, cube_format) for c in cubemap]
        else:
            raise ValueError
    else:
        raise ValueError

    equis = []
    for cm in cubemap:
        equi = _run_single(
            cm,
            w_out=w_out,
            h_out=h_out,
            sampling_method=sampling_method,
            mode=mode,
        )
        equis.append(equi)

    equis = np.stack(equis, axis=0)

    if is_single:
        equis = equis.squeeze(0)

    return equis
