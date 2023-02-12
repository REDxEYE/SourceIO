import math
from typing import List

import numpy as np

from ...utils.byte_io_mdl import ByteIO


class _Decoder:
    def __init__(self, name, n_type, version):
        self.name = name
        self.n_type = n_type
        self.version = version
        self.size = 0
        self._type = 'x'
        self.calc_size()

    def calc_size(self):
        if self.name in ["CCompressedStaticFullVector3",
                         "CCompressedFullVector3", ]:
            self.size = 4 * 3
            self._type = '3f'
        elif self.name in ["CCompressedAnimVector3",
                           "CCompressedDeltaVector3",
                           "CCompressedStaticVector3"]:
            self.size = 2 * 3
            self._type = '3Y'

        elif self.name in ["CCompressedAnimQuaternion",
                           "CCompressedFullQuaternion",
                           "CCompressedStaticQuaternion"]:
            self.size = 6
            self._type = 'O'

        elif self.name in ["CCompressedStaticFloat", "CCompressedFullFloat"]:
            self.size = 4
            self._type = 'f'

        else:
            raise NotImplementedError(f"Unknown decoder type {self.name}")

    def read_element(self, reader: ByteIO):
        if 'O' in self._type:
            return self._read_quat(reader.read(6))
        elif 'Y' in self._type:
            count = int(self._type[:-1])
            return [reader.read_float16() for _ in range(count)]
        else:
            return reader.read_fmt(self._type)

    @staticmethod
    def _read_quat(data):

        i1 = data[0] + ((data[1] & 63) << 8)
        i2 = data[2] + ((data[3] & 63) << 8)
        i3 = data[4] + ((data[5] & 63) << 8)

        # Signs
        s1 = data[1] & 128
        s2 = data[3] & 128
        s3 = data[5] & 128

        c = math.sin(math.pi / 4.0) / 16384
        t1 = math.sin(math.pi / 4.0)
        x = c * (i1 - 16384) if (data[1] & 64) == 0 else c * i1
        y = c * (i2 - 16384) if (data[3] & 64) == 0 else c * i2
        z = c * (i3 - 16384) if (data[5] & 64) == 0 else c * i3

        w = math.sqrt(1 - (x * x) - (y * y) - (z * z))

        # Apply sign 3:
        if s3 == 128:
            w *= -1

        # Apply sign 1 and 2
        if s1 == 128:
            return (y, z, w, x) if s2 == 128 else (z, w, x, y)

        return (w, x, y, z) if s2 == 128 else (x, y, z, w)


def parse_anim_data(anim_block: dict, agroup_block: dict):
    print("Parsing animation data")
    anim_array = anim_block['m_animArray']
    animations: List[Animation] = []
    if len(anim_array) == 0:
        return []

    decoder_array = anim_block['m_decoderArray']
    segment_array = anim_block['m_segmentArray']
    decode_key = agroup_block['m_decodeKey']
    for anim in anim_array:
        print(f"Parsing {anim['m_name']}")
        animations.append(parse_anim(anim, decode_key, decoder_array, segment_array))
    return animations


def parse_anim(anim_desc, decode_key, decoder_array, segment_array):
    p_data = anim_desc['m_pData']
    frame_block_array = p_data['m_frameblockArray']
    frame_count = p_data['m_nFrames']
    animation = Animation(anim_desc['m_name'], anim_desc['fps'])
    for frame_id in range(frame_count):
        frame = Frame()
        for frame_block in frame_block_array:
            start = frame_block['m_nStartFrame']
            end = frame_block['m_nEndFrame']
            if start <= frame_id <= end:
                for segment_index in frame_block['m_segmentIndexArray']:
                    parse_segment(min(max(frame_id - start, 0), frame_count),
                                  frame,
                                  segment_array[segment_index],
                                  decode_key,
                                  decoder_array)
        animation.add_frame(frame)
    return animation


def parse_segment(frame_id, frame: 'Frame', segment, decode_key, decoder_array):
    local_channel = segment['m_nLocalChannel']
    data_channel = decode_key['m_dataChannelArray'][local_channel]
    container = ByteIO(segment['m_container'])

    element_index_array = data_channel['m_nElementIndexArray']
    element_bones = np.zeros(decode_key['m_nChannelElements'], dtype=np.uint32)

    for i, index in enumerate(element_index_array):
        element_bones[index] = i

    if container.size():
        d = decoder_array[container.read_int16()]
        decoder = _Decoder(d['m_szName'], d['m_nType'], d['m_nVersion'])
        # print(f"Reading {d}")
        cardinality = container.read_int16()
        bone_count = container.read_int16()
        total_size = container.read_int16()

        elements = np.frombuffer(container.read(2 * bone_count), dtype=np.uint16)

        if container.tell() + (decoder.size * frame_id * bone_count) < container.size():
            container.skip(decoder.size * frame_id * bone_count)

        bone_names = data_channel['m_szElementNameArray']
        channel_name = data_channel['m_szChannelClass']
        channel_attr_name = data_channel['m_szVariableName']

        for bone_id in range(bone_count):
            bone = element_bones[elements[bone_id]]
            value = decoder.read_element(container)
            frame.set_attribute(bone_names[bone], channel_name, channel_attr_name,
                                (decoder.name,value ))


class Frame:
    def __init__(self):
        self.morph_data = {}
        self.bone_data = {}

    def set_attribute(self, bone_name, channel_type, attr_name, value):
        if channel_type == 'BoneChannel':
            if bone_name not in self.bone_data:
                self.bone_data[bone_name] = {}
            self.bone_data[bone_name][attr_name] = value
        elif channel_type == 'MorphChannel':
            if bone_name not in self.morph_data:
                self.morph_data[bone_name] = {}
            self.morph_data[bone_name][attr_name] = value


class Animation:
    def __init__(self, name, fps):
        self.name = name
        self.fps = fps
        self.frames = []  # type:List[Frame]

    def add_frame(self, frame: Frame):
        self.frames.append(frame)

    def __repr__(self):
        return f'Animation "{self.name}" (fps:{self.fps})'
