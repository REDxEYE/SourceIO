import struct

import numpy as np

from ...byte_io_mdl import ByteIO

index_header = 0xe0
vertex_header = 0xa0
vertex_block_size_bytes = 8192
vertex_block_max_size = 256
byte_group_size = 16
tail_max_size = 32


def unzigzag8(v):
    return (-(v & 1) ^ (v >> 1)) & 0xFF


def slice(data: np.ndarray, start, len=None):
    if len is None:
        len = data.size - start
    return data[start:start + len]


def decode_vertex_buffer(data, size, count):
    buffer = CompressedVertexBuffer(size, count)
    return buffer.decode_vertex_buffer(data)


def decode_index_buffer(data, size, count):
    buffer = CompressedIndexBuffer(size, count)
    return buffer.decode_index_buffer(data)


class CompressedVertexBuffer:

    def __init__(self, vertex_size, vertex_count):
        self.vertex_size = vertex_size
        self.vertex_count = vertex_count

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
            print(f"Decoding vertex block {vertex_offset}/{self.vertex_count}", end='\r')
            block_size = vertex_block_size if vertex_offset + vertex_block_size < self.vertex_count else \
                self.vertex_count - vertex_offset
            vertex_span = self.decode_vertex_block(vertex_span, slice(result, vertex_offset * self.vertex_size),
                                                   block_size,
                                                   last_vertex)
            vertex_offset += block_size
        return bytes(result)


class CompressedIndexBuffer:
    def __init__(self, size, count):
        self.index_size = size
        self.index_count = count
        pass

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

        next_id = 0
        last_id = 0

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
                    c = next_id if fec == 0 else vertex_fifo[(vertex_fifo_offset - 1 - fec) & 15]
                    fec0 = fec == 0
                    next_id += fec0
                    self.write_triangle(destination, i, a, b, c)
                    vertex_fifo_offset = self.push_vertex_fifo(vertex_fifo, vertex_fifo_offset, c, fec0)
                    edge_fifo_offset = self.push_edge_fifo(edge_fifo, edge_fifo_offset, c, b)
                    edge_fifo_offset = self.push_edge_fifo(edge_fifo, edge_fifo_offset, a, c)
                else:
                    c = last_id = self.decode_index(ds, last_id)
                    self.write_triangle(destination, i, a, b, c)

                    vertex_fifo_offset = self.push_vertex_fifo(vertex_fifo, vertex_fifo_offset, c)

                    edge_fifo_offset = self.push_edge_fifo(edge_fifo, edge_fifo_offset, c, b)
                    edge_fifo_offset = self.push_edge_fifo(edge_fifo, edge_fifo_offset, a, c)
            else:
                if code_tri < 0xfe:
                    codeaux = codeaux_table[code_tri & 15]
                    feb = codeaux >> 4
                    fec = codeaux & 15

                    a = next_id
                    next_id += 1

                    b = next_id if feb == 0 else vertex_fifo[(vertex_fifo_offset - feb) & 15]
                    feb0 = feb == 0
                    next_id += feb0

                    c = next_id if fec == 0 else vertex_fifo[(vertex_fifo_offset - fec) & 15]
                    fec0 = 1 if not fec else 0
                    next_id += fec0

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
                        a = next_id
                        next_id += 1
                    else:
                        a = 0

                    if feb == 0:
                        b = next_id
                        next_id += 1
                    else:
                        b = vertex_fifo[(vertex_fifo_offset - feb) & 15]

                    if fec == 0:
                        c = next_id
                        next_id += 1
                    else:
                        c = vertex_fifo[(vertex_fifo_offset - fec) & 15]

                    if fea == 15:
                        last_id = a = self.decode_index(ds, last_id)
                    if feb == 15:
                        last_id = b = self.decode_index(ds, last_id)
                    if fec == 15:
                        last_id = c = self.decode_index(ds, last_id)

                    self.write_triangle(destination, i, a, b, c)

                    vertex_fifo_offset = self.push_vertex_fifo(vertex_fifo, vertex_fifo_offset, a)
                    vertex_fifo_offset = self.push_vertex_fifo(vertex_fifo, vertex_fifo_offset, b,
                                                               (feb == 0) or (feb == 15))
                    vertex_fifo_offset = self.push_vertex_fifo(vertex_fifo, vertex_fifo_offset, c,
                                                               (fec == 0) or (fec == 15))
                    edge_fifo_offset = self.push_edge_fifo(edge_fifo, edge_fifo_offset, b, a)
                    edge_fifo_offset = self.push_edge_fifo(edge_fifo, edge_fifo_offset, c, b)
                    edge_fifo_offset = self.push_edge_fifo(edge_fifo, edge_fifo_offset, a, c)
        assert ds.size() == ds.tell(), "we didn't read all data bytes and " \
                                       "stopped before the boundary between data and codeaux table"
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

    @staticmethod
    def push_vertex_fifo(fifo: np.ndarray, offset, v, cond=True):
        fifo[offset] = v
        return (offset + cond) & 15

    @staticmethod
    def push_edge_fifo(fifo: np.ndarray, offset, a, b):
        fifo[offset, :] = [a, b]
        return (offset + 1) & 15

    def decode_index(self, data: ByteIO, last):
        v = self.decode_vbyte(data)
        mm = 0xFF_FF_FF_FF if self.index_size == 4 else 0xFF_FF
        d = ((v >> 1) ^ -(v & 1)) & mm
        return (last + d) & mm

    @staticmethod
    def decode_vbyte(data: ByteIO):
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
