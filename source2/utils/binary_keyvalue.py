from enum import IntEnum, IntFlag

import numpy as np

from ...utilities.byte_io_mdl import ByteIO
from ..blocks.compiled_file_header import InfoBlock
from ...utilities.lz4_wrapper import LZ4ChainDecoder, LZ4Wrapper


def uncompress(compressed_data, _b, decompressed_size):
    decoder = LZ4Wrapper()
    return decoder.decompress_safe(compressed_data, decompressed_size)


class KVFlag(IntFlag):
    Nothing = 0
    Resource = 1
    DeferredResource = 2


class KVType(IntEnum):
    STRING_MULTI = 0  # STRING_MULTI doesn't have an ID
    NULL = 1
    BOOLEAN = 2
    INT64 = 3
    UINT64 = 4
    DOUBLE = 5
    STRING = 6
    BINARY_BLOB = 7
    ARRAY = 8
    OBJECT = 9
    ARRAY_TYPED = 10
    INT32 = 11
    UINT32 = 12
    BOOLEAN_TRUE = 13
    BOOLEAN_FALSE = 14
    INT64_ZERO = 15
    INT64_ONE = 16
    DOUBLE_ZERO = 17
    DOUBLE_ONE = 18
    UNK = 21


class BinaryKeyValue:
    KV3_ENCODING_BINARY_BLOCK_COMPRESSED = (
        0x46, 0x1A, 0x79, 0x95, 0xBC, 0x95, 0x6C, 0x4F, 0xA7, 0x0B, 0x05, 0xBC, 0xA1, 0xB7, 0xDF, 0xD2)
    KV3_ENCODING_BINARY_UNCOMPRESSED = (
        0x00, 0x05, 0x86, 0x1B, 0xD8, 0xF7, 0xC1, 0x40, 0xAD, 0x82, 0x75, 0xA4, 0x82, 0x67, 0xE7, 0x14)
    KV3_ENCODING_BINARY_BLOCK_LZ4 = (
        0x8A, 0x34, 0x47, 0x68, 0xA1, 0x63, 0x5C, 0x4F, 0xA1, 0x97, 0x53, 0x80, 0x6F, 0xD9, 0xB1, 0x19)
    KV3_FORMAT_GENERIC = (
        0x7C, 0x16, 0x12, 0x74, 0xE9, 0x06, 0x98, 0x46, 0xAF, 0xF2, 0xE6, 0x3E, 0xB5, 0x90, 0x37, 0xE7)
    KV3_SIG = (0x56, 0x4B, 0x56, 0x03)
    VKV3_SIG = (0x01, 0x33, 0x56, 0x4B)
    VKV3_v2_SIG = (0x02, 0x33, 0x56, 0x4B)

    KNOWN_SIGNATURES = [KV3_SIG, VKV3_SIG, VKV3_v2_SIG]

    indent = 0

    def __init__(self, block_info: InfoBlock = None):
        super().__init__()
        self.block_info = block_info
        self.mode = 0
        self.strings = []
        self.types = np.array([])
        self.current_type = 0
        self.kv = []
        self.flags = 0
        self.buffer = ByteIO()  # type: ByteIO

        self.bin_blob_count = 0
        self.bin_blob_offset = -1
        self.int_count = 0
        self.int_offset = -1
        self.double_count = 0
        self.double_offset = -1

        self.byte_buffer = ByteIO()
        self.int_buffer = ByteIO()
        self.double_buffer = ByteIO()

        self.block_data = bytearray()
        self.block_reader = ByteIO()
        self.block_sizes = []
        self.next_block_id = 0

    def read(self, reader: ByteIO):
        fourcc = reader.read(4)
        assert tuple(fourcc) in self.KNOWN_SIGNATURES, 'Invalid KV Signature'
        if tuple(fourcc) == self.VKV3_SIG:
            self.read_v1(reader)
        if tuple(fourcc) == self.VKV3_v2_SIG:
            self.read_v2(reader)
        elif tuple(fourcc) == self.KV3_SIG:
            self.read_v3(reader)

    def block_decompress(self, reader):
        self.flags = reader.read(4)
        if self.flags[3] & 0x80:
            self.buffer.write_bytes(reader.read(-1))
        working = True
        while reader.tell() != reader.size() and working:
            block_mask = reader.read_uint16()
            for i in range(16):
                if block_mask & (1 << i) > 0:
                    offset_and_size = reader.read_uint16()
                    offset = ((offset_and_size & 0xFFF0) >> 4) + 1
                    size = (offset_and_size & 0x000F) + 3
                    lookup_size = offset if offset < size else size

                    entry = self.buffer.tell()
                    self.buffer.seek(entry - offset)
                    data = self.buffer.read(lookup_size)
                    self.buffer.seek(entry)
                    while size > 0:
                        self.buffer.write_bytes(data[:lookup_size if lookup_size < size else size])
                        size -= lookup_size
                else:
                    data = reader.read_int8()
                    self.buffer.write_int8(data)
                if self.buffer.size() == (self.flags[2] << 16) + (self.flags[1] << 8) + self.flags[0]:
                    working = False
                    break
        self.buffer.seek(0)

    def decompress_lz4(self, reader):
        decompressed_size = reader.read_uint32()
        compressed_size = reader.size() - reader.tell()
        data = reader.read(-1)
        data = uncompress(data, compressed_size, decompressed_size)
        self.buffer.write_bytes(data)
        self.buffer.seek(0)

    def read_v3(self, reader):
        encoding = reader.read(16)
        assert tuple(encoding) in [
            self.KV3_ENCODING_BINARY_BLOCK_COMPRESSED,
            self.KV3_ENCODING_BINARY_BLOCK_LZ4,
            self.KV3_ENCODING_BINARY_UNCOMPRESSED,
        ], 'Unrecognized KV3 Encoding'

        fmt = reader.read(16)

        assert tuple(fmt) == self.KV3_FORMAT_GENERIC, 'Unrecognised KV3 Format'
        if tuple(encoding) == self.KV3_ENCODING_BINARY_BLOCK_COMPRESSED:
            self.block_decompress(reader)
        elif tuple(encoding) == self.KV3_ENCODING_BINARY_BLOCK_LZ4:
            self.decompress_lz4(reader)
        elif tuple(encoding) == self.KV3_ENCODING_BINARY_UNCOMPRESSED:
            self.buffer.write_bytes(reader.read(-1))
            self.buffer.seek(0)
        string_count = self.buffer.read_uint32()
        for _ in range(string_count):
            self.strings.append(self.buffer.read_ascii_string())
        self.int_buffer = self.buffer
        self.double_buffer = self.buffer
        self.byte_buffer = self.buffer
        self.parse(self.buffer, self.kv, True)
        assert len(self.kv) == 1, "Never yet seen that state of vkv3 v1"
        self.kv = self.kv[0]
        self.buffer.close()
        del self.buffer

    def read_v1(self, reader: ByteIO):
        fmt = reader.read(16)
        assert tuple(fmt) == self.KV3_FORMAT_GENERIC, 'Unrecognised KV3 Format'

        compression_method = reader.read_uint32()
        self.bin_blob_count = reader.read_uint32()
        self.int_count = reader.read_uint32()
        self.double_count = reader.read_uint32()
        if compression_method == 0:
            length = reader.read_uint32()
            self.buffer.write_bytes(reader.read(length))
        elif compression_method == 1:
            uncompressed_size = reader.read_uint32()
            compressed_size = self.block_info.block_size - reader.tell()
            data = reader.read(compressed_size)
            u_data = uncompress(data, compressed_size, uncompressed_size)
            assert len(u_data) == uncompressed_size, "Decompressed data size does not match expected size"
            self.buffer.write_bytes(u_data)
        else:
            raise NotImplementedError("Unknown KV3 compression method")

        self.buffer.seek(0)

        self.byte_buffer.write_bytes(self.buffer.read(self.bin_blob_count))
        self.byte_buffer.seek(0)

        if self.buffer.tell() % 4 != 0:
            self.buffer.seek(self.buffer.tell() + (4 - (self.buffer.tell() % 4)))

        self.int_buffer.write_bytes(self.buffer.read(self.int_count * 4))
        self.int_buffer.seek(0)

        if self.buffer.tell() % 8 != 0:
            self.buffer.seek(self.buffer.tell() + (8 - (self.buffer.tell() % 8)))

        self.double_buffer.write_bytes(self.buffer.read(self.double_count * 8))
        self.double_buffer.seek(0)

        for _ in range(self.int_buffer.read_uint32()):
            self.strings.append(self.buffer.read_ascii_string())

        types_len = self.buffer.size() - self.buffer.tell() - 4

        self.types = np.frombuffer(self.buffer.read(types_len), np.uint8)

        self.parse(self.buffer, self.kv, True)
        self.kv = self.kv[0]

        self.buffer.close()
        del self.buffer
        self.byte_buffer.close()
        del self.byte_buffer
        self.int_buffer.close()
        del self.int_buffer
        self.double_buffer.close()
        del self.double_buffer

    def read_v2(self, reader: ByteIO):
        fmt = reader.read(16)
        assert tuple(fmt) == self.KV3_FORMAT_GENERIC, 'Unrecognised KV3 Format'

        compression_method = reader.read_uint32()
        compression_dict_id = reader.read_uint16()
        compression_frame_size = reader.read_uint16()

        self.bin_blob_count = reader.read_uint32()
        self.int_count = reader.read_uint32()
        self.double_count = reader.read_uint32()

        string_and_types_buffer_size, b, c = reader.read_fmt('I2H')

        uncompressed_size = reader.read_uint32()
        compressed_size = reader.read_uint32()
        block_count = reader.read_uint32()
        block_total_size = reader.read_uint32()

        if compression_method == 0:
            if compression_dict_id != 0:
                raise NotImplementedError('Unknown compression method in KV3 v2 block')
            if compression_frame_size != 0:
                raise NotImplementedError('Unknown compression method in KV3 v2 block')
            self.buffer.write_bytes(reader.read(compressed_size))
        elif compression_method == 1:

            if compression_dict_id != 0:
                raise NotImplementedError('Unknown compression method in KV3 v2 block')

            if compression_frame_size != 16384:
                raise NotImplementedError('Unknown compression method in KV3 v2 block')

            data = reader.read(compressed_size)
            u_data = uncompress(data, compressed_size, uncompressed_size)
            assert len(u_data) == uncompressed_size, "Decompressed data size does not match expected size"
            self.buffer.write_bytes(u_data)
        else:
            raise NotImplementedError("Unknown KV3 compression method")

        self.buffer.seek(0)

        self.byte_buffer.write_bytes(self.buffer.read(self.bin_blob_count))
        self.byte_buffer.seek(0)

        if self.buffer.tell() % 4 != 0:
            self.buffer.seek(self.buffer.tell() + (4 - (self.buffer.tell() % 4)))

        self.int_buffer.write_bytes(self.buffer.read(self.int_count * 4))
        self.int_buffer.seek(0)

        if self.buffer.tell() % 8 != 0:
            self.buffer.seek(self.buffer.tell() + (8 - (self.buffer.tell() % 8)))

        self.double_buffer.write_bytes(self.buffer.read(self.double_count * 8))
        self.double_buffer.seek(0)

        string_start = self.buffer.tell()

        for _ in range(self.int_buffer.read_uint32()):
            self.strings.append(self.buffer.read_ascii_string())

        types_len = string_and_types_buffer_size - (self.buffer.tell() - string_start)
        self.types = np.frombuffer(self.buffer.read(types_len),np.uint8)
        if block_count == 0:
            assert self.buffer.read_uint32() == 0xFFEEDD00, 'Invalid terminator'
            self.parse(self.buffer, self.kv, True)
            self.kv = self.kv[0]
        else:
            self.block_sizes = [self.buffer.read_uint32() for _ in range(block_count)]
            assert self.buffer.read_uint32() == 0xFFEEDD00, 'Invalid terminator'
            cd = LZ4ChainDecoder(block_total_size, 0)
            for uncompressed_block_size in self.block_sizes:
                if compression_method == 0:
                    self.block_data += reader.read(uncompressed_block_size)
                elif compression_method == 1:
                    compressed_block_size = self.buffer.read_uint16()
                    data = reader.read(compressed_block_size)
                    data = cd.decompress(data, uncompressed_block_size)
                    self.block_data += data
                else:
                    raise NotImplementedError("Unknown KV3 compression method")

            self.block_reader.write_bytes(self.block_data)
            self.block_reader.seek(0)
            self.parse(self.buffer, self.kv, True)
            self.kv = self.kv[0]

        self.buffer.close()
        del self.buffer
        self.byte_buffer.close()
        del self.byte_buffer
        self.int_buffer.close()
        del self.int_buffer
        self.double_buffer.close()
        del self.double_buffer

    def read_type(self, reader: ByteIO):
        if self.types.shape[0] > 0:
            data_type = self.types[self.current_type]
            self.current_type += 1
        else:
            data_type = reader.read_int8()

        flag_info = KVFlag.Nothing
        if data_type & 0x80:
            data_type &= 0x7F
            if self.types.shape[0]>0:
                flag_info = KVFlag(self.types[self.current_type])
                self.current_type += 1
            else:
                flag_info = KVFlag(reader.read_int8())
        return KVType(data_type), flag_info

    def parse(self, reader: ByteIO, parent=None, in_array=False):
        name = None
        if not in_array:
            str_id = self.int_buffer.read_uint32()
            name = self.strings[str_id] if str_id != -1 else ""
        data_type, flag_info = self.read_type(reader)
        self.read_value(name, reader, data_type, parent, in_array)

    def read_value(self, name, reader: ByteIO, data_type: KVType, parent, is_array=False):
        def add(v):
            if not is_array:
                parent.update({name: v})
            else:
                parent.append(v)

        if data_type == KVType.NULL:
            add(None)
            return
        elif data_type == KVType.DOUBLE:
            add(self.double_buffer.read_double())
            return
        elif data_type == KVType.BOOLEAN:
            add(self.byte_buffer.read_int8() == 1)
            return
        elif data_type == KVType.BOOLEAN_TRUE:
            add(True)
            return
        elif data_type == KVType.BOOLEAN_FALSE:
            add(False)
            return
        elif data_type == KVType.INT64:
            add(self.double_buffer.read_int64())
            return
        elif data_type == KVType.UINT64:
            add(self.double_buffer.read_uint64())
            return

        elif data_type == KVType.DOUBLE_ZERO:
            add(0.0)
            return
        elif data_type == KVType.INT64_ZERO:
            add(0)
            return
        elif data_type == KVType.INT64_ONE:
            add(1)
            return
        elif data_type == KVType.DOUBLE_ONE:
            add(1.0)
            return
        elif data_type == KVType.INT32:
            add(self.int_buffer.read_int32())
            return
        elif data_type == KVType.STRING:
            string_id = self.int_buffer.read_int32()
            if string_id == -1:
                add(None)
                return
            add(self.strings[string_id])
            return
        elif data_type == KVType.ARRAY:
            size = self.int_buffer.read_uint32()
            arr = []
            for _ in range(size):
                self.parse(reader, arr, True)
            add(arr)
            return
        elif data_type == KVType.OBJECT:
            size = self.int_buffer.read_uint32()
            tmp = {}
            for _ in range(size):
                self.parse(reader, tmp, False)
            add(tmp)
            if not parent:
                parent = tmp
        elif data_type == KVType.ARRAY_TYPED:
            t_array_size = self.int_buffer.read_uint32()
            sub_type, sub_flag = self.read_type(reader)
            tmp = []
            for _ in range(t_array_size):
                self.read_value(name, reader, sub_type, tmp, True)

            if sub_type in (KVType.DOUBLE, KVType.DOUBLE_ONE, KVType.DOUBLE_ZERO):
                tmp = np.array(tmp, dtype=np.float64)
            add(tmp)
        elif data_type == KVType.BINARY_BLOB:
            if self.block_reader.size() != 0:
                data = self.block_reader.read(self.block_sizes[self.next_block_id])
                self.next_block_id += 1
                add(data)
            else:
                size = self.int_buffer.read_uint32()
                add(self.byte_buffer.read(size))
            return
        else:
            raise NotImplementedError("Unknown KVType.{}".format(data_type.name))

        return parent
