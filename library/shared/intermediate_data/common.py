from dataclasses import dataclass
from typing import TypeVar, Generic, Iterator

from SourceIO.library.utils.file_utils import Buffer

import numpy as np

T = TypeVar("T", int, float, contravariant=True)


class _TupleLike(Generic[T]):
    __slots__ = ()

    def __iter__(self) -> Iterator[T]:
        for f in self.__dataclass_fields__.values():  # type: ignore[attr-defined]
            yield getattr(self, f.name)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)  # type: ignore[attr-defined]

    def __getitem__(self, i: int | slice) -> T | tuple[T, ...]:
        t = tuple(self)
        return t[i]


@dataclass(frozen=True, slots=True)
class Vector2(_TupleLike[T]):
    x: T
    y: T

    def __len__(self) -> int:
        return 2

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'Vector2[float]':
        return Vector2(*buffer.read_fmt('2f'))


@dataclass(frozen=True, slots=True)
class Vector3(_TupleLike[T]):
    x: T
    y: T
    z: T

    def __len__(self) -> int:
        return 3

    @classmethod
    def one(cls):
        return cls(1, 1, 1)

    @classmethod
    def zero(cls):
        return cls(0, 0, 0)

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'Vector3[float]':
        return cls(*buffer.read_fmt('3f'))


@dataclass(frozen=True, slots=True)
class Vector4(_TupleLike[T]):
    x: T
    y: T
    z: T
    w: T

    def __len__(self) -> int:
        return 4

    @classmethod
    def one(cls):
        return cls(1, 1, 1, 1)

    @classmethod
    def zero(cls):
        return cls(0, 0, 0, 0)

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'Vector4[float]':
        return cls(*buffer.read_fmt('4f'))


class Quaternion(Vector4[float]):

    @classmethod
    def zero(cls):
        return cls(0.0, 0.0, 0.0, 1.0)


class Matrix4x4:

    def __init__(self, matrix: np.ndarray = None):
        if matrix is None:
            self._matrix = np.eye(4)
        else:
            if matrix.shape != (4, 4):
                raise ValueError("Matrix must be 4x4")
            self._matrix = matrix

    def __array__(self, dtype=None):
        return np.asarray(self._matrix, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        arrays = [x._matrix if isinstance(x, Matrix4x4) else x for x in inputs]
        result = getattr(ufunc, method)(*arrays, **kwargs)
        if isinstance(result, np.ndarray) and result.shape == (4, 4):
            return Matrix4x4(result)
        return result

    def __array_function__(self, func, types, args, kwargs):
        arrays = tuple(x._matrix if isinstance(x, Matrix4x4) else x for x in args)
        result = func(*arrays, **kwargs)
        if isinstance(result, np.ndarray) and result.shape == (4, 4):
            return Matrix4x4(result)
        return result

    def __getitem__(self, key: tuple[int] | tuple[int, int]) -> float | np.ndarray:
        return self._matrix[key]

    def __iter__(self) -> Iterator[T]:
        return self._matrix.__iter__()

    def __len__(self) -> int:
        return 16


    @classmethod
    def from_3x4(cls, matrix: np.ndarray) -> 'Matrix4x4':
        if matrix.shape != (3, 4):
            raise ValueError("Input matrix must be 4x3")
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :4] = matrix
        return cls(mat)

    @classmethod
    def from_translation(cls, translation: Vector3) -> "Matrix4x4":
        mat = np.eye(4, dtype=np.float32)
        mat[0, 3] = translation.x
        mat[1, 3] = translation.y
        mat[2, 3] = translation.z
        return cls(mat)

    @classmethod
    def from_euler_angles(cls, pitch: float, yaw: float, roll: float) -> "Matrix4x4":
        cx, sx = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cz, sz = np.cos(roll), np.sin(roll)

        # X rotation
        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, cx, -sx],
                       [0.0, sx, cx]], dtype=np.float32)

        # Y rotation
        Ry = np.array([[cy, 0.0, sy],
                       [0.0, 1.0, 0.0],
                       [-sy, 0.0, cy]], dtype=np.float32)

        # Z rotation
        Rz = np.array([[cz, -sz, 0.0],
                       [sz, cz, 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float32)

        # Blender 'XYZ': apply X then Y then Z  →  R = Rz * Ry * Rx
        R = Rz @ (Ry @ Rx)

        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = R
        return cls(mat)

    @classmethod
    def from_quaternion(cls, quat: Quaternion) -> "Matrix4x4":
        x, y, z, w = quat
        mat = np.eye(4, dtype=np.float32)
        mat[0, 0] = 1 - 2 * (y * y + z * z)
        mat[0, 1] = 2 * (x * y - z * w)
        mat[0, 2] = 2 * (x * z + y * w)
        mat[1, 0] = 2 * (x * y + z * w)
        mat[1, 1] = 1 - 2 * (x * x + z * z)
        mat[1, 2] = 2 * (y * z - x * w)
        mat[2, 0] = 2 * (x * z - y * w)
        mat[2, 1] = 2 * (y * z + x * w)
        mat[2, 2] = 1 - 2 * (x * x + y * y)
        return cls(mat)

    @classmethod
    def from_scale(cls, scale: Vector3) -> "Matrix4x4":
        mat = np.eye(4, dtype=np.float32)
        mat[0, 0] = scale.x
        mat[1, 1] = scale.y
        mat[2, 2] = scale.z
        return cls(mat)

    @classmethod
    def from_trs(cls, translation: Vector3, rotation: Quaternion | Vector3, scale: Vector3) -> "Matrix4x4":
        t_mat = cls.from_translation(translation)
        if isinstance(rotation, Quaternion):
            r_mat = cls.from_quaternion(rotation)
        else:
            r_mat = cls.from_euler_angles(*rotation)
        s_mat = cls.from_scale(scale)
        return t_mat @ r_mat @ s_mat


    def to_translation(self) -> Vector3:
        return Vector3(self[0, 3], self[1, 3], self[2, 3])

    def to_scale(self) -> Vector3:
        return Vector3(self[0, 0], self[1, 1], self[2, 2])

    def to_quaternion(self) -> Quaternion:
        m = self._matrix
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        return Quaternion(x, y, z, w)

    def to_euler(self) -> Vector3[float]:
        # Assuming the order is pitch (X), yaw (Y), roll (Z)
        m = self._matrix
        sy = np.clip(m[0, 2], -1.0, 1.0)
        pitch = np.arcsin(sy)
        if abs(sy) < 0.999999:
            yaw = np.arctan2(-m[1, 2], m[2, 2])
            roll = np.arctan2(-m[0, 1], m[0, 0])
        else:
            yaw = np.arctan2(m[2, 1], m[1, 1])
            roll = 0.0
        return Vector3(pitch, yaw, roll)

    def to_numpy(self) -> np.ndarray:
        return self._matrix.copy()

    def __mul__(self, other: "Matrix4x4") -> "Matrix4x4":
        return Matrix4x4(np.dot(self._matrix, other._matrix))

    def __matmul__(self, other: "Matrix4x4") -> "Matrix4x4":
        return Matrix4x4(np.matmul(self._matrix, other._matrix))

