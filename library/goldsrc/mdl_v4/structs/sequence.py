import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from ....shared.types import Vector3
from ....utils import Buffer


def euler_to_quat(euler):
    eulerd = euler[2] * 0.5
    v8 = math.sin(eulerd)
    v9 = math.cos(eulerd)
    eulerd = euler[1] * 0.5
    v12 = math.sin(eulerd)
    v10 = math.cos(eulerd)
    eulerd = euler[0] * 0.5
    v11 = math.sin(eulerd)
    eulerd = math.cos(eulerd)
    v4 = v11 * v10
    v5 = eulerd * v12
    x = v9 * v4 - v8 * v5
    y = v4 * v8 + v5 * v9
    v6 = v10 * eulerd
    v7 = v11 * v12
    z = v8 * v6 - v9 * v7
    w = v7 * v8 + v9 * v6
    quat = w, x, y, z
    return quat


@dataclass(slots=True)
class SequenceFrame:
    global_frame_id: float
    unk: Tuple[int, ...]
    root_motion: Vector3[float]
    animation_per_bone_rot: npt.NDArray[np.float32]

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

    def read_anim_values(self, buffer: Buffer, bone_count) -> List[Tuple[Vector3[float], npt.NDArray]]:
        frames = []
        for _ in range(self.frame_count):
            frame = SequenceFrame.from_buffer(buffer, bone_count)
            frames.append((frame.root_motion, frame.animation_per_bone_rot))
        return frames
