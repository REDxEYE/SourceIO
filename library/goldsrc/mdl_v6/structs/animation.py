import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ....utils import Buffer

raw_pos_dtype = np.dtype([
    ('frame_id', np.int32, (1,)),
    ('value', np.float32, (3,)),
])
raw_rot_dtype = np.dtype([
    ('frame_id', np.int16, (1,)),
    ('value', np.int16, (3,)),
])


@dataclass(slots=True)
class StudioAnimation:
    frames: npt.NDArray

    @classmethod
    def from_buffer(cls, buffer: Buffer, frame_count: int):
        pos_count, pos_offset, rot_count, rot_offset = buffer.read_fmt('4I')
        buffer.seek(pos_offset)
        raw_pos_data = np.frombuffer(buffer.read(pos_count * raw_pos_dtype.itemsize), raw_pos_dtype)

        buffer.seek(rot_offset)
        raw_rot_data = np.frombuffer(buffer.read(rot_count * raw_rot_dtype.itemsize), raw_rot_dtype)

        frames = np.zeros((frame_count, 2, 3,), np.float32)

        total_count = frames.shape[0]
        start_frame_id = 0
        for index, bone_pos in enumerate(raw_pos_data):
            if index == pos_count - 1:
                stop_frame_id = total_count
            else:
                stop_frame_id = raw_pos_data[index + 1]['frame_id'][0]

            for frame_id in range(start_frame_id, stop_frame_id):
                frames[frame_id][0] = bone_pos['value']
            start_frame_id = stop_frame_id

        start_frame_id = 0
        for index, bone_rot in enumerate(raw_rot_data):
            if index == rot_count - 1:
                stop_frame_id = total_count
            else:
                stop_frame_id = raw_rot_data[index + 1]['frame_id'][0]

            for frame_id in range(start_frame_id, stop_frame_id):
                frames[frame_id][1] = bone_rot['value'] * (math.pi / 18000)
            start_frame_id = stop_frame_id
        return cls(frames)
