import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from SourceIO.library.shared.types import Vector3
from SourceIO.library.utils import Buffer


def euler_to_quat(euler: tuple[float, float, float]):
    rx, ry, rz = euler

    sx, cx = math.sin(rx * 0.5), math.cos(rx * 0.5)
    sy, cy = math.sin(ry * 0.5), math.cos(ry * 0.5)
    sz, cz = math.sin(rz * 0.5), math.cos(rz * 0.5)

    return (
        cx * cy * cz + sx * sy * sz,  # w
        sx * cy * cz - cx * sy * sz,  # x
        cx * sy * cz + sx * cy * sz,  # y
        cx * cy * sz - sx * sy * cz,  # z
    )


@dataclass(slots=True)
class SequenceFrame:
    global_frame_id: float
    unk: tuple[int, ...]
    root_motion: Vector3[float]
    animation_per_bone_rot: np.ndarray

    @classmethod
    def from_buffer(cls, reader: Buffer, bone_count: int):
        global_frame_id = reader.read_float()
        unk = reader.read_fmt('11I')
        root_motion = reader.read_fmt('3f')
        animation_per_bone_rot = np.frombuffer(reader.read(6 * bone_count), dtype=np.uint16).astype(np.float32)
        animation_per_bone_rot *= 0.0001745329354889691
        animation_per_bone_rot = animation_per_bone_rot.reshape((-1, 3))
        return cls(global_frame_id, unk, root_motion, animation_per_bone_rot)


@dataclass(slots=True)
class StudioSequence:
    name: str
    frame_count: int
    unk: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(buffer.read_ascii_string(32), buffer.read_int32(), buffer.read_int32())

    def read_anim_values(self, buffer: Buffer, bone_count) -> list[tuple[Vector3[float], npt.NDArray]]:
        frames = []
        for _ in range(self.frame_count):
            frame = SequenceFrame.from_buffer(buffer, bone_count)
            frames.append((frame.root_motion, frame.animation_per_bone_rot))
        return frames
