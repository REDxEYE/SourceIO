from enum import IntEnum
from typing import List, Dict
import numpy as np

try:
    from ..utils.PySourceIOUtils import (decode_vertex_buffer as decode_vertex_buffer_, \
                                         decode_index_buffer as decode_index_buffer_)


    def decode_vertex_buffer(data, size, count):
        return decode_vertex_buffer_(data, len(data), size, count)


    def decode_index_buffer(data, size, count):
        return decode_index_buffer_(data, len(data), size, count)

except ImportError:
    print("Failed to import native binary!\nUsing python version")
    from ..utils.compressed_buffers import decode_vertex_buffer, decode_index_buffer, slice

from ..utils.compressed_buffers import slice
from ...byte_io_mdl import ByteIO

from ..common import SourceVertex, short_to_float
from .dummy import DataBlock


class DxgiFormat(IntEnum):
    UNKNOWN = 0,
    R32G32B32A32_TYPELESS = 1,
    R32G32B32A32_FLOAT = 2,
    R32G32B32A32_UINT = 3,
    R32G32B32A32_SINT = 4,
    R32G32B32_TYPELESS = 5,
    R32G32B32_FLOAT = 6,
    R32G32B32_UINT = 7,
    R32G32B32_SINT = 8,
    R16G16B16A16_TYPELESS = 9,
    R16G16B16A16_FLOAT = 10,
    R16G16B16A16_UNORM = 11,
    R16G16B16A16_UINT = 12,
    R16G16B16A16_SNORM = 13,
    R16G16B16A16_SINT = 14,
    R32G32_TYPELESS = 15,
    R32G32_FLOAT = 16,
    R32G32_UINT = 17,
    R32G32_SINT = 18,
    R32G8X24_TYPELESS = 19,
    D32_FLOAT_S8X24_UINT = 20,
    R32_FLOAT_X8X24_TYPELESS = 21,
    X32_TYPELESS_G8X24_UINT = 22,
    R10G10B10A2_TYPELESS = 23,
    R10G10B10A2_UNORM = 24,
    R10G10B10A2_UINT = 25,
    R11G11B10_FLOAT = 26,
    R8G8B8A8_TYPELESS = 27,
    R8G8B8A8_UNORM = 28,
    R8G8B8A8_UNORM_SRGB = 29,
    R8G8B8A8_UINT = 30,
    R8G8B8A8_SNORM = 31,
    R8G8B8A8_SINT = 32,
    R16G16_TYPELESS = 33,
    R16G16_FLOAT = 34,
    R16G16_UNORM = 35,
    R16G16_UINT = 36,
    R16G16_SNORM = 37,
    R16G16_SINT = 38,
    R32_TYPELESS = 39,
    D32_FLOAT = 40,
    R32_FLOAT = 41,
    R32_UINT = 42,
    R32_SINT = 43,
    R24G8_TYPELESS = 44,
    D24_UNORM_S8_UINT = 45,
    R24_UNORM_X8_TYPELESS = 46,
    X24_TYPELESS_G8_UINT = 47,
    R8G8_TYPELESS = 48,
    R8G8_UNORM = 49,
    R8G8_UINT = 50,
    R8G8_SNORM = 51,
    R8G8_SINT = 52,
    R16_TYPELESS = 53,
    R16_FLOAT = 54,
    D16_UNORM = 55,
    R16_UNORM = 56,
    R16_UINT = 57,
    R16_SNORM = 58,
    R16_SINT = 59,
    R8_TYPELESS = 60,
    R8_UNORM = 61,
    R8_UINT = 62,
    R8_SNORM = 63,
    R8_SINT = 64,
    A8_UNORM = 65,
    R1_UNORM = 66,
    R9G9B9E5_SHAREDEXP = 67,
    R8G8_B8G8_UNORM = 68,
    G8R8_G8B8_UNORM = 69,
    BC1_TYPELESS = 70,
    BC1_UNORM = 71,
    BC1_UNORM_SRGB = 72,
    BC2_TYPELESS = 73,
    BC2_UNORM = 74,
    BC2_UNORM_SRGB = 75,
    BC3_TYPELESS = 76,
    BC3_UNORM = 77,
    BC3_UNORM_SRGB = 78,
    BC4_TYPELESS = 79,
    BC4_UNORM = 80,
    BC4_SNORM = 81,
    BC5_TYPELESS = 82,
    BC5_UNORM = 83,
    BC5_SNORM = 84,
    B5G6R5_UNORM = 85,
    B5G5R5A1_UNORM = 86,
    B8G8R8A8_UNORM = 87,
    B8G8R8X8_UNORM = 88,
    R10G10B10_XR_BIAS_A2_UNORM = 89,
    B8G8R8A8_TYPELESS = 90,
    B8G8R8A8_UNORM_SRGB = 91,
    B8G8R8X8_TYPELESS = 92,
    B8G8R8X8_UNORM_SRGB = 93,
    BC6H_TYPELESS = 94,
    BC6H_UF16 = 95,
    BC6H_SF16 = 96,
    BC7_TYPELESS = 97,
    BC7_UNORM = 98,
    BC7_UNORM_SRGB = 99,
    AYUV = 100,
    Y410 = 101,
    Y416 = 102,
    NV12 = 103,
    P010 = 104,
    P016 = 105,
    _OPAQUE = 106,
    YUY2 = 107,
    Y210 = 108,
    Y216 = 109,
    NV11 = 110,
    AI44 = 111,
    IA44 = 112,
    P8 = 113,
    A8P8 = 114,
    B4G4R4A4_UNORM = 115,
    P208 = 130,
    V208 = 131,
    V408 = 132,


class VertexBuffer:
    def __init__(self):
        super().__init__()
        self.vertex_count = 0
        self.vertex_size = 0
        self.offset = 0
        self.total_size = 0
        self.attributes = []  # type:List[VertexAttribute]
        self.buffer = ByteIO()  # type: ByteIO
        # self.vertexes = []  # type: List[SourceVertex]
        self.vertexes = {}  # type: Dict[str,List]
        self.vertex_struct = ''

    def __repr__(self):
        buff = ''
        for attrib in self.attributes:
            buff += attrib.name + ' ' + attrib.format.name + '; '
        return '<VertexBuffer vertexes:{} ' \
               'attributes:{} vertex size:{} ' \
               'vertex attributes: {} >'.format(self.vertex_count, len(self.attributes), self.vertex_size, buff, )

    def read(self, reader: ByteIO):
        self.vertex_count = reader.read_uint32()
        self.vertex_size = reader.read_uint32()
        entry = reader.tell()
        attributes_offset = reader.read_uint32()
        attributes_count = reader.read_uint32()
        with reader.save_current_pos():
            reader.seek(entry + attributes_offset)
            used_names = []
            for _ in range(attributes_count):
                v_attrib = VertexAttribute()
                v_attrib.read(reader)
                if v_attrib.name in used_names:
                    tmp = v_attrib.name
                    v_attrib.name += f"_{used_names.count(v_attrib.name)}"
                    used_names.append(tmp)
                else:
                    used_names.append(v_attrib.name)
                self.vertex_struct += v_attrib.get_struct()
                self.attributes.append(v_attrib)
        entry = reader.tell()
        self.offset = reader.read_uint32()
        self.total_size = reader.read_uint32()
        with reader.save_current_pos():
            reader.seek(entry + self.offset)
            if self.total_size != self.vertex_size * self.vertex_count:
                data = reader.read_bytes(self.total_size)
                self.buffer.write_bytes(decode_vertex_buffer(data, self.vertex_size, self.vertex_count))
            else:
                self.buffer.write_bytes(reader.read_bytes(self.vertex_count * self.vertex_size))
            self.buffer.seek(0)
        self.read_buffer()

    def read_buffer(self):
        for attrib in self.attributes:
            self.vertexes[attrib.name] = []
        for _ in range(self.vertex_count):
            vertex_data = list(self.buffer.read_fmt(self.vertex_struct))
            offset = 0
            for attrib in self.attributes:
                attrib_len = attrib.element_count
                if attrib.name == 'POSITION':
                    self.vertexes[attrib.name].append(slice(vertex_data, offset, attrib_len))
                elif 'TEXCOORD' in attrib.name:
                    self.vertexes[attrib.name].append(slice(vertex_data, offset, attrib_len))
                elif attrib.name == 'COLOR':
                    self.vertexes[attrib.name].append(slice(vertex_data, offset, attrib_len))
                elif attrib.name == 'NORMAL':
                    self.vertexes[attrib.name].append(slice(vertex_data, offset, attrib_len))
                elif attrib.name == 'TANGENT':
                    self.vertexes[attrib.name].append(slice(vertex_data, offset, attrib_len))
                elif 'texcoord' in attrib.name:
                    self.vertexes[attrib.name].append(slice(vertex_data, offset, attrib_len))
                elif attrib.name == "BLENDINDICES":
                    self.vertexes[attrib.name].append(slice(vertex_data, offset, attrib_len))
                elif attrib.name == "BLENDWEIGHT":
                    self.vertexes[attrib.name].append(slice(vertex_data, offset, attrib_len))
                else:
                    print(f"UNKNOWN ATTRIBUTE {attrib.name}!!!!")
                offset += attrib_len
        for attrib in self.attributes:
            attrib_array = self.vertexes[attrib.name]
            # if "UNORM" in attrib.format.name:
            #     attrib_array = list(np.array(attrib_array) / 255.0)
            # elif "SNORM" in attrib.format.name:
            #     attrib_array = list(
            #         np.array(attrib_array) / 128.0)
            if attrib.name == "BLENDWEIGHT":
                attrib_array = list(np.array(attrib_array) / 255.0)
            self.vertexes[attrib.name] = attrib_array


class VertexAttribute:
    def __init__(self):
        self.name = ''
        self.format = DxgiFormat(0)  # type:DxgiFormat
        self.offset = 0
        self.abs_offset = 0
        self.element_count = 0

    def __repr__(self):
        return '<VertexAttribute "{}" format:{} offset:{}>'.format(self.name, self.format.name, self.offset)

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.name = reader.read_ascii_string()
        reader.seek(entry + 36)
        self.format = DxgiFormat(reader.read_int32())
        self.offset = reader.read_uint32()
        self.element_count = self.get_element_count()
        reader.skip(12)

    def get_struct(self):
        if self.format == DxgiFormat.R32G32B32_FLOAT:
            return '3f'
        elif self.format == DxgiFormat.R32G32_FLOAT:
            return '2f'
        elif self.format == DxgiFormat.R32_FLOAT:
            return '1f'
        elif self.format == DxgiFormat.R32G32B32_UINT:
            return '3I'
        elif self.format == DxgiFormat.R32G32B32_SINT:
            return '3i'
        elif self.format == DxgiFormat.R32G32B32A32_FLOAT:
            return '4f'
        elif self.format == DxgiFormat.R32G32B32A32_UINT:
            return '4I'
        elif self.format == DxgiFormat.R32G32B32A32_SINT:
            return '4i'
        elif self.format == DxgiFormat.R16G16_FLOAT:
            return '2e'
        elif self.format == DxgiFormat.R16G16_SINT:
            return '2h'
        elif self.format == DxgiFormat.R16G16_UINT:
            return '2H'
        elif self.format == DxgiFormat.R16G16B16A16_SINT:
            return '4h'
        elif self.format == DxgiFormat.R8G8B8A8_SNORM:
            return '4b'
        elif self.format == DxgiFormat.R8G8B8A8_UNORM:
            return '4B'
        elif self.format == DxgiFormat.R8G8B8A8_UINT:
            return '4B'
        elif self.format == DxgiFormat.R16G16_UNORM:
            return '2H'
        else:
            raise NotImplementedError(f"UNSUPPORTED DXGI format {self.format.name}")

    def get_element_count(self):
        if self.format == DxgiFormat.R32G32B32_FLOAT:
            return 3
        elif self.format == DxgiFormat.R32G32_FLOAT:
            return 2
        elif self.format == DxgiFormat.R32_FLOAT:
            return 1
        elif self.format == DxgiFormat.R32G32B32_UINT:
            return 3
        elif self.format == DxgiFormat.R32G32B32_SINT:
            return 3
        elif self.format == DxgiFormat.R32G32B32A32_FLOAT:
            return 4
        elif self.format == DxgiFormat.R32G32B32A32_UINT:
            return 4
        elif self.format == DxgiFormat.R32G32B32A32_SINT:
            return 4
        elif self.format == DxgiFormat.R16G16_FLOAT:
            return 2
        elif self.format == DxgiFormat.R16G16_SINT:
            return 2
        elif self.format == DxgiFormat.R16G16_UINT:
            return 2
        elif self.format == DxgiFormat.R16G16B16A16_SINT:
            return 4
        elif self.format == DxgiFormat.R8G8B8A8_SNORM:
            return 4
        elif self.format == DxgiFormat.R8G8B8A8_UNORM:
            return 4
        elif self.format == DxgiFormat.R8G8B8A8_UINT:
            return 4
        elif self.format == DxgiFormat.R16G16_UNORM:
            return 4
        elif self.format == DxgiFormat.R16G16_UNORM:
            return 2
        else:
            raise NotImplementedError(f"UNSUPPORTED DXGI format {self.format.name}")

    def read_from_buffer(self, reader: ByteIO):
        if self.format == DxgiFormat.R32G32B32_FLOAT:
            return reader.read_fmt(f'3f')
        elif self.format == DxgiFormat.R32G32_FLOAT:
            return reader.read_fmt(f'2f')
        elif self.format == DxgiFormat.R32_FLOAT:
            return reader.read_fmt(f'f')
        elif self.format == DxgiFormat.R32G32B32_UINT:
            return reader.read_fmt(f'3I')
        elif self.format == DxgiFormat.R32G32B32_SINT:
            return reader.read_fmt(f'3i')
        elif self.format == DxgiFormat.R32G32B32A32_FLOAT:
            return reader.read_fmt(f'4f')
        elif self.format == DxgiFormat.R32G32B32A32_UINT:
            return reader.read_fmt(f'4I')
        elif self.format == DxgiFormat.R32G32B32A32_SINT:
            return reader.read_fmt(f'4i')
        elif self.format == DxgiFormat.R16G16_FLOAT:
            return [short_to_float(reader.read_int16()) for _ in range(self.format.name.count('16'))]
        elif self.format == DxgiFormat.R16G16_SINT:
            return reader.read_fmt(f'2h')
        elif self.format == DxgiFormat.R16G16_UINT:
            return reader.read_fmt(f'2H')
        elif self.format == DxgiFormat.R16G16B16A16_SINT:
            return reader.read_fmt(f'4h')
        elif self.format == DxgiFormat.R8G8B8A8_SNORM:
            return reader.read_fmt(f'4b')
        elif self.format == DxgiFormat.R8G8B8A8_UNORM:
            return reader.read_fmt(f'4B')
        elif self.format == DxgiFormat.R8G8B8A8_UINT:
            return reader.read_fmt(f'4B')
        elif self.format == DxgiFormat.R16G16_UNORM:
            return reader.read_fmt(f'2H')
        else:
            raise NotImplementedError(f"UNSUPPORTED DXGI format {self.format.name}")


class IndexBuffer:
    def __init__(self):
        self.index_count = 0
        self.index_size = 0
        self.offset = 0
        self.unk1 = 0
        self.unk2 = 0
        self.total_size = 0
        self.buffer = ByteIO()  # type:ByteIO
        self.indexes = []

    def __repr__(self):
        return '<IndexBuffer indexes:{} size:{}>'.format(self.index_count, self.index_size)

    def read(self, reader: ByteIO):
        self.index_count = reader.read_uint32()
        self.index_size = reader.read_uint32()
        self.unk1 = reader.read_uint32()
        self.unk2 = reader.read_uint32()
        entry = reader.tell()
        self.offset = reader.read_uint32()
        self.total_size = reader.read_uint32()
        with reader.save_current_pos():
            reader.seek(entry + self.offset)
            if self.total_size != self.index_size * self.index_count:
                data = reader.read_bytes(self.total_size)
                self.buffer.write_bytes(decode_index_buffer(data, self.index_size, self.index_count))
            else:
                self.buffer.write_bytes(reader.read_bytes(self.index_count * self.index_size))
            self.buffer.seek(0)
        # with open("test.bin", 'wb') as f:
        #     f.write(self.buffer.read_bytes(-1))
        self.buffer.seek(0)

        self.read_buffer()

    def read_buffer(self):
        reader = self.buffer.read_uint32 if self.index_size == 4 else self.buffer.read_uint16
        for n in range(0, self.index_count, 3):
            polygon = [reader(), reader(), reader()]
            self.indexes.append(polygon)


class VBIB(DataBlock):

    def __init__(self, valve_file, info_block):
        super().__init__(valve_file, info_block)
        self.vertex_offset = 0
        self.vertex_count = 0
        self.index_offset = 0
        self.index_count = 0
        self.vertex_buffer: List[VertexBuffer] = []
        self.index_buffer: List[IndexBuffer] = []

    def __repr__(self):
        return '<VBIB vertex buffers:{} index buffers:{}>'.format(self.vertex_count, self.index_count)

    def read(self):
        reader = self.reader
        self.vertex_offset = reader.read_uint32()
        self.vertex_count = reader.read_uint32()
        entry = reader.tell()
        self.index_offset = reader.read_uint32()
        self.index_count = reader.read_uint32()
        with reader.save_current_pos():
            reader.seek(self.vertex_offset)
            for _ in range(self.vertex_count):
                v_buffer = VertexBuffer()
                v_buffer.read(reader)
                self.vertex_buffer.append(v_buffer)

        with reader.save_current_pos():
            reader.seek(entry + self.index_offset)
            for _ in range(self.index_count):
                i_buffer = IndexBuffer()
                i_buffer.read(reader)
                self.index_buffer.append(i_buffer)
