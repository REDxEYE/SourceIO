import struct
from copy import copy
from enum import IntEnum
from typing import List
import numpy as np

from ..source2 import ValveFile
from ...byte_io_mdl import ByteIO

from .common import SourceVertex, SourceVector, short_to_float, SourceVector4D, SourceVector2D
from .header_block import InfoBlock
from .dummy import DataBlock


def unzigzag8(v):
    return (-(v & 1) ^ (v >> 1)) & 0xFF


def slice(data: np.ndarray, start, len=None):
    if len is None:
        len = data.size - start
    return data[start:start + len]


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


index_header = 0xe0
vertex_header = 0xa0
vertex_block_size_bytes = 8192
vertex_block_max_size = 256
byte_group_size = 16
tail_max_size = 32


class VertexBuffer:
    def __init__(self):
        super().__init__()
        self.vertex_count = 0
        self.vertex_size = 0
        self.attributes_offset = 0
        self.attributes_count = 0
        self.offset = 0
        self.total_size = 0
        self.attributes = []  # type:List[VertexAttribute]
        self.buffer = ByteIO()  # type: ByteIO
        self.vertexes = []  # type: List[SourceVertex]
        self.vertex_struct = ''

    def __repr__(self):
        buff = ''
        for attrib in self.attributes:
            buff += attrib.name + ' ' + attrib.format.name + '; '
        return '<VertexBuffer vertexes:{} ' \
               'attributes:{} vertex size:{} ' \
               'vertex attributes: {} >'.format(self.vertex_count, self.attributes_count, self.vertex_size, buff, )

    def read(self, reader: ByteIO):
        self.vertex_count = reader.read_uint32()
        self.vertex_size = reader.read_uint32()
        entry = reader.tell()
        self.attributes_offset = reader.read_uint32()
        self.attributes_count = reader.read_uint32()
        with reader.save_current_pos():
            reader.seek(entry + self.attributes_offset)
            used_names = []
            for _ in range(self.attributes_count):
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
                self.buffer.write_bytes(self.decode_vertex_buffer(reader.read_bytes(self.total_size)))
            else:
                self.buffer.write_bytes(reader.read_bytes(self.vertex_count * self.vertex_size))
            self.buffer.seek(0)
        self.read_buffer()

    def read_buffer(self):
        for n in range(self.vertex_count):
            vertex = SourceVertex()
            vertex_data = self.buffer.read_fmt(self.vertex_struct)
            offset = 0
            for attrib in self.attributes:
                attrib_len = attrib.element_count
                if attrib.name == 'POSITION':
                    vertex.position = slice(vertex_data, offset, attrib_len)
                elif 'TEXCOORD' in attrib.name:
                    if "_" in attrib.name:
                        t_id = int(attrib.name.split("_")[-1])
                    else:
                        t_id = 0
                    if vertex.uvs.get(t_id, None) is None:
                        vertex.uvs[t_id] = []

                    vertex.uvs[t_id] = slice(vertex_data, offset, attrib_len)
                elif attrib.name == 'COLOR':
                    vertex.color = slice(vertex_data, offset, attrib_len)
                elif attrib.name == 'NORMAL':
                    vertex.normal = slice(vertex_data, offset, attrib_len)
                elif attrib.name == 'TANGENT':
                    vertex.tangent = slice(vertex_data, offset, attrib_len)
                elif attrib.name == 'texcoord':
                    vertex.lightmap = slice(vertex_data, offset, attrib_len)
                elif attrib.name == "BLENDINDICES":
                    vertex.bone_weight.bone = slice(vertex_data, offset, attrib_len)
                    vertex.bone_weight.boneCount = len(vertex.bone_weight.bone)
                elif attrib.name == "BLENDWEIGHT":
                    vertex.bone_weight.weight = np.divide(slice(vertex_data, offset, attrib_len), 255)
                else:
                    print(f"UNKNOWN ATTRIBUTE {attrib.name}!!!!")
                offset += attrib_len
            self.vertexes.append(vertex)

    @staticmethod
    def decode_bytes_group(data, destination, bitslog2):
        data_offset = 0
        data_var = 0
        b = 0

        def next(bits, encv):
            enc = b >> (8 - bits)
            is_same = enc == (1 << bits) - 1
            return (b << bits) & 0xFF, data_var + is_same, encv if is_same else enc & 0xFF

        if bitslog2 == 0:
            for k in range(byte_group_size):
                destination[k] = 0
            return data
        elif bitslog2 == 1:
            data_var = 4
            b = data[data_offset]
            data_offset += 1
            b, data_var, destination[0] = next(2, data[data_var])
            b, data_var, destination[1] = next(2, data[data_var])
            b, data_var, destination[2] = next(2, data[data_var])
            b, data_var, destination[3] = next(2, data[data_var])
            b = data[data_offset]
            data_offset += 1
            b, data_var, destination[4] = next(2, data[data_var])
            b, data_var, destination[5] = next(2, data[data_var])
            b, data_var, destination[6] = next(2, data[data_var])
            b, data_var, destination[7] = next(2, data[data_var])
            b = data[data_offset]
            data_offset += 1
            b, data_var, destination[8] = next(2, data[data_var])
            b, data_var, destination[9] = next(2, data[data_var])
            b, data_var, destination[10] = next(2, data[data_var])
            b, data_var, destination[11] = next(2, data[data_var])
            b = data[data_offset]
            data_offset += 1
            b, data_var, destination[12] = next(2, data[data_var])
            b, data_var, destination[13] = next(2, data[data_var])
            b, data_var, destination[14] = next(2, data[data_var])
            b, data_var, destination[15] = next(2, data[data_var])

            return slice(data, data_var)
        elif bitslog2 == 2:
            data_var = 8

            b = data[data_offset]
            data_offset += 1
            b, data_var, destination[0] = next(4, data[data_var])
            b, data_var, destination[1] = next(4, data[data_var])
            b = data[data_offset]
            data_offset += 1
            b, data_var, destination[2] = next(4, data[data_var])
            b, data_var, destination[3] = next(4, data[data_var])
            b = data[data_offset]
            data_offset += 1
            b, data_var, destination[4] = next(4, data[data_var])
            b, data_var, destination[5] = next(4, data[data_var])
            b = data[data_offset]
            data_offset += 1
            b, data_var, destination[6] = next(4, data[data_var])
            b, data_var, destination[7] = next(4, data[data_var])
            b = data[data_offset]
            data_offset += 1
            b, data_var, destination[8] = next(4, data[data_var])
            b, data_var, destination[9] = next(4, data[data_var])
            b = data[data_offset]
            data_offset += 1
            b, data_var, destination[10] = next(4, data[data_var])
            b, data_var, destination[11] = next(4, data[data_var])
            b = data[data_offset]
            data_offset += 1
            b, data_var, destination[12] = next(4, data[data_var])
            b, data_var, destination[13] = next(4, data[data_var])
            b = data[data_offset]
            data_offset += 1
            b, data_var, destination[14] = next(4, data[data_var])
            b, data_var, destination[15] = next(4, data[data_var])
            return slice(data, data_var)

        elif bitslog2 == 3:
            destination[:byte_group_size] = data[0:byte_group_size]
            return slice(data, byte_group_size)
        else:
            raise Exception("Unexpected bit length")

    def decode_bytes(self, data: np.ndarray, destination: np.ndarray):
        assert destination.size % byte_group_size == 0, "Expected data length to be a multiple of ByteGroupSize."
        header_size = ((destination.size // byte_group_size) + 3) // 4
        header = slice(data, 0)
        data: np.ndarray = slice(data, header_size)
        for i in range(0, destination.size, byte_group_size):
            assert data.size >= tail_max_size, "Cannot decode"
            header_offset = i // byte_group_size
            bitslog2 = (header[header_offset // 4] >> ((header_offset % 4) * 2)) & 3
            data = self.decode_bytes_group(data, slice(destination, i), bitslog2)
        return data

    def get_vertex_block_size(self):
        result = vertex_block_size_bytes // self.vertex_size
        result &= ~(byte_group_size - 1)
        return result if result < vertex_block_max_size \
            else vertex_block_max_size

    def decode_vertex_block(self, data: np.ndarray, vertex_data: np.ndarray, vertex_count, last_vertex: np.ndarray):
        assert 0 < vertex_count <= vertex_block_max_size, \
            f"Expected vertexCount({vertex_count}) to be between 0 and VertexMaxBlockSize"
        buffer = np.zeros((vertex_block_max_size,), dtype=np.uint8)
        transposed = np.zeros((vertex_block_size_bytes,), dtype=np.uint8)
        vertex_count_aligned = (vertex_count + byte_group_size - 1) & ~(
                byte_group_size - 1)
        for k in range(self.vertex_size):
            data = self.decode_bytes(data, slice(buffer, 0, vertex_count_aligned))
            vertex_offset = k
            p = last_vertex[k]
            for i in range(vertex_count):
                a = buffer[i]
                v = (((-(a & 1) ^ (a >> 1)) & 0xFF) + p) & 0xFF
                transposed[vertex_offset] = v
                p = v
                vertex_offset += self.vertex_size
        vertex_data[:vertex_count * self.vertex_size] = slice(transposed, 0, vertex_count * self.vertex_size)
        last_vertex[:self.vertex_size] = slice(transposed, self.vertex_size * (vertex_count - 1), self.vertex_size)
        return data

    def decode_vertex_buffer(self, buffer: bytes):
        buffer: np.ndarray = np.array(list(buffer), dtype=np.uint8)
        assert 0 < self.vertex_size < 256, f"Vertex size is expected to be between 1 and 256 = {self.vertex_size}"
        assert self.vertex_size % 4 == 0, "Vertex size is expected to be a multiple of 4."
        assert len(buffer) > 1 + self.vertex_size, "Vertex buffer is too short."
        vertex_span = buffer.copy()
        header = vertex_span[0]
        assert header == vertex_header, \
            f"Invalid vertex buffer header, expected {vertex_header} but got {header}."
        vertex_span: np.ndarray = slice(vertex_span, 1)
        last_vertex: np.ndarray = slice(vertex_span, buffer.size - 1 - self.vertex_size, self.vertex_size)
        vertex_block_size = self.get_vertex_block_size()
        vertex_offset = 0
        result = np.zeros((self.vertex_count * self.vertex_size,), dtype=np.uint8)

        while vertex_offset < self.vertex_count:
            print(f"BLOCK {vertex_offset}/{self.vertex_count}")
            block_size = vertex_block_size if vertex_offset + vertex_block_size < self.vertex_count else \
                self.vertex_count - vertex_offset
            vertex_span = self.decode_vertex_block(vertex_span, slice(result, vertex_offset * self.vertex_size),
                                                   block_size,
                                                   last_vertex)
            vertex_offset += block_size
        return bytes(result)


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
                self.buffer.write_bytes(self.decode_index_buffer(reader.read_bytes(self.total_size)))
            else:
                self.buffer.write_bytes(reader.read_bytes(self.index_count * self.index_size))
            self.buffer.seek(0)
        with open("test.bin", 'wb') as f:
            f.write(self.buffer.read_bytes(-1))
        self.buffer.seek(0)

        self.read_buffer()

    def decode_index_buffer(self, buffer: bytes):
        buffer: np.ndarray = np.array(list(buffer), dtype=np.uint8)
        assert self.index_count % 3 == 0, "Expected indexCount to be a multiple of 3."
        assert self.index_size in [2, 4], "Expected indexSize to be either 2 or 4"
        data_offset = 1 + (self.index_count // 3)
        assert buffer.size >= data_offset + 16, "Index buffer is too short."
        assert buffer[0] == index_header, "Incorrect index buffer header."
        vertex_fifo = np.zeros((16,), dtype=np.uint32)
        edge_fifo = np.zeros((16, 2), dtype=np.uint32)
        edge_fifo_offset = 0
        vertex_fifo_offset = 0

        next = 0
        last = 0

        buffer_index = 1
        data = slice(buffer, data_offset, buffer.size - 16 - data_offset)
        codeaux_table = slice(buffer, buffer.size - 16)
        destination = np.zeros((self.index_count * self.index_size),
                               dtype=np.uint8)
        ds = ByteIO(byte_object=bytes(data))
        for i in range(0, self.index_count, 3):
            code_tri = buffer[buffer_index]
            buffer_index += 1

            if code_tri < 0xF0:
                fe = code_tri >> 4
                a, b = edge_fifo[((edge_fifo_offset - 1 - fe) & 15)]
                fec = code_tri & 15
                if fec != 15:
                    c = next if fec == 0 else vertex_fifo[(vertex_fifo_offset - 1 - fec) & 15]
                    fec0 = fec == 0
                    next += fec0
                    self.write_triangle(destination, i, a, b, c)
                    vertex_fifo_offset = self.push_vertex_fifo(vertex_fifo, vertex_fifo_offset, c, fec0)
                    edge_fifo_offset = self.push_edge_fifo(edge_fifo, edge_fifo_offset, c, b)
                    edge_fifo_offset = self.push_edge_fifo(edge_fifo, edge_fifo_offset, a, c)
                else:
                    c = last = self.decode_index(ds, next, last)
                    self.write_triangle(destination, i, a, b, c)

                    vertex_fifo_offset = self.push_vertex_fifo(vertex_fifo, vertex_fifo_offset, c)

                    edge_fifo_offset = self.push_edge_fifo(edge_fifo, edge_fifo_offset, c, b)
                    edge_fifo_offset = self.push_edge_fifo(edge_fifo, edge_fifo_offset, a, c)
            else:
                if code_tri < 0xfe:
                    codeaux = codeaux_table[code_tri & 15]
                    feb = codeaux >> 4
                    fec = codeaux & 15

                    a = next
                    next += 1

                    b = next if feb == 0 else vertex_fifo[(vertex_fifo_offset - feb) & 15]
                    feb0 = feb == 0
                    next += feb0

                    c = next if fec == 0 else vertex_fifo[(vertex_fifo_offset - fec) & 15]
                    fec0 = 1 if not fec else 0
                    next += fec0

                    self.write_triangle(destination, i, a, b, c)

                    vertex_fifo_offset = self.push_vertex_fifo(vertex_fifo, vertex_fifo_offset, a)
                    vertex_fifo_offset = self.push_vertex_fifo(vertex_fifo, vertex_fifo_offset, b, feb0 == 1)
                    vertex_fifo_offset = self.push_vertex_fifo(vertex_fifo, vertex_fifo_offset, c, fec0 == 1)

                    edge_fifo_offset = self.push_edge_fifo(edge_fifo, edge_fifo_offset, b, a)
                    edge_fifo_offset = self.push_edge_fifo(edge_fifo, edge_fifo_offset, c, b)
                    edge_fifo_offset = self.push_edge_fifo(edge_fifo, edge_fifo_offset, a, c)
                else:
                    codeaux = ds.read_uint8()
                    fea = 0 if code_tri == 0xfe else 15
                    feb = codeaux >> 4
                    fec = codeaux & 15

                    if fea == 0:
                        a = next
                        next += 1
                    else:
                        a = 0

                    if feb == 0:
                        b = next
                        next += 1
                    else:
                        b = vertex_fifo[(vertex_fifo_offset - feb) & 15]

                    if fec == 0:
                        c = next
                        next += 1
                    else:
                        c = vertex_fifo[(vertex_fifo_offset - fec) & 15]

                    if fea == 15:
                        last = a = self.decode_index(ds, next, last)
                    if feb == 15:
                        last = b = self.decode_index(ds, next, last)
                    if fec == 15:
                        last = c = self.decode_index(ds, next, last)

                    self.write_triangle(destination, i, a, b, c)

                    vertex_fifo_offset = self.push_vertex_fifo(vertex_fifo, vertex_fifo_offset, a)
                    vertex_fifo_offset = self.push_vertex_fifo(vertex_fifo, vertex_fifo_offset, b,
                                                               (feb == 0) or (feb == 15))
                    vertex_fifo_offset = self.push_vertex_fifo(vertex_fifo, vertex_fifo_offset, c,
                                                               (fec == 0) or (fec == 15))
                    edge_fifo_offset = self.push_edge_fifo(edge_fifo, edge_fifo_offset, b, a)
                    edge_fifo_offset = self.push_edge_fifo(edge_fifo, edge_fifo_offset, c, b)
                    edge_fifo_offset = self.push_edge_fifo(edge_fifo, edge_fifo_offset, a, c)
        assert ds.size() == ds.tell(), "we didn't read all data bytes and stopped before the boundary between data and codeaux table"
        return bytes(destination)

    def write_triangle(self, destination: np.ndarray, offset, a, b, c):
        offset *= self.index_size
        if self.index_size == 2:
            ad = struct.pack('H', a)
            bd = struct.pack('H', b)
            cd = struct.pack('H', c)
            destination[offset:offset + 6] = np.array(list(ad) + list(bd) + list(cd), dtype=np.uint8)
        else:
            ad = struct.pack('I', a)
            bd = struct.pack('I', b)
            cd = struct.pack('I', c)
            destination[offset:offset + 12] = np.array(list(ad) + list(bd) + list(cd), dtype=np.uint8)

    def push_vertex_fifo(self, fifo: np.ndarray, offset, v, cond=True):
        fifo[offset] = v
        return (offset + cond) & 15

    def push_edge_fifo(self, fifo: np.ndarray, offset, a, b):
        fifo[offset, :] = [a, b]
        return (offset + 1) & 15

    def decode_index(self, data: ByteIO, next, last):
        v = self.decode_vbyte(data)
        mm = 0xFF_FF_FF_FF if self.index_size == 4 else 0xFF_FF
        d = ((v >> 1) ^ -(v & 1)) & mm
        return (last + d) & mm

    def decode_vbyte(self, data: ByteIO):
        lead = data.read_uint8()
        if lead < 128:
            return lead
        result = lead & 127
        shift = 7
        for i in range(4):
            group = data.read_uint8()
            result |= (group & 127) << shift
            shift += 7
            if group < 128:
                break
        return result

    def read_buffer(self):
        reader = self.buffer.read_uint32 if self.index_size == 4 else self.buffer.read_uint16
        for n in range(0, self.index_count, 3):
            polygon = [reader(), reader(), reader()]
            self.indexes.append(polygon)


class VBIB(DataBlock):

    def __init__(self, valve_file: ValveFile, info_block):
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
