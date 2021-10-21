import math

import numpy as np

from ....utils.byte_io_mdl import ByteIO

raw_pos_dtype = np.dtype([
    ('frame_id', np.int32, (1,)),
    ('value', np.float32, (3,)),
])
raw_rot_dtype = np.dtype([
    ('frame_id', np.int16, (1,)),
    ('value', np.int16, (3,)),
])


class StudioAnimation:
    def __init__(self):
        self.pos_count = 0
        self.pos_offset = 0
        self.rot_count = 0
        self.rot_offset = 0

        self.start_frame_id = 0
        self.stop_frame_id = 0

        self.frames = np.array([])

    def read(self, reader: ByteIO):
        self.pos_count, self.pos_offset, self.rot_count, self.rot_offset = reader.read_fmt('4I')

    def read_anim_values(self, reader: ByteIO):
        reader.seek(self.pos_offset)
        raw_pos_data = np.frombuffer(reader.read(self.pos_count * raw_pos_dtype.itemsize), raw_pos_dtype)

        reader.seek(self.rot_offset)
        raw_rot_data = np.frombuffer(reader.read(self.rot_count * raw_rot_dtype.itemsize), raw_rot_dtype)

        total_count = self.frames.shape[0]
        self.start_frame_id = 0
        for index, bone_pos in enumerate(raw_pos_data):
            if index == self.pos_count - 1:
                self.stop_frame_id = total_count
            else:
                self.stop_frame_id = raw_pos_data[index + 1]['frame_id'][0]

            for frame_id in range(self.start_frame_id, self.stop_frame_id):
                self.frames[frame_id][0] = bone_pos['value']
            self.start_frame_id = self.stop_frame_id

        self.start_frame_id = 0
        for index, bone_rot in enumerate(raw_rot_data):
            if index == self.rot_count - 1:
                self.stop_frame_id = total_count
            else:
                self.stop_frame_id = raw_rot_data[index + 1]['frame_id'][0]

            for frame_id in range(self.start_frame_id, self.stop_frame_id):
                self.frames[frame_id][1] = bone_rot['value'] * (math.pi / 18000)
            self.start_frame_id = self.stop_frame_id
