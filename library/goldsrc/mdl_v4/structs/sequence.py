import math
from typing import List

import numpy as np

from ....utils.byte_io_mdl import ByteIO


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


class SequenceFrame:
    def __init__(self):
        self.sequence_id = 0.0
        self.unk = []
        self.unk_vec = []
        self.animation_per_bone_rot = np.array([])

    def read(self, reader: ByteIO, bone_count):
        self.sequence_id = reader.read_float()
        self.unk = reader.read_fmt('11I')
        self.unk_vec = reader.read_fmt('3f')
        self.animation_per_bone_rot = np.frombuffer(reader.read(6 * bone_count), dtype=np.int16).astype(np.float32)
        self.animation_per_bone_rot *= 0.0001745329354889691
        self.animation_per_bone_rot = self.animation_per_bone_rot.reshape((-1, 3))


class StudioSequence:
    def __init__(self):
        self.name = ''
        self.frame_count = 0
        self.unk = 0
        self.frame_helpers: List[SequenceFrame] = []
        self.frames = []

    def read(self, reader: ByteIO):
        self.name = reader.read_ascii_string(32)
        self.frame_count = reader.read_int32()
        self.unk = reader.read_int32()

    def read_anim_values(self, reader, bone_count):
        for _ in range(self.frame_count):
            frame = SequenceFrame()
            frame.read(reader, bone_count)
            self.frame_helpers.append(frame)
            self.frames.append(frame.animation_per_bone_rot)
