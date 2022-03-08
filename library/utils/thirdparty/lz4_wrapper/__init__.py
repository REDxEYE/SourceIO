import ctypes
import platform
from ctypes import *

from pathlib import Path
from typing import Tuple


class Mem:
    K1 = 1024
    K2 = 2 * K1
    K4 = 4 * K1
    K8 = 8 * K1
    K16 = 16 * K1
    K32 = 32 * K1
    K64 = 64 * K1
    K128 = 128 * K1
    K256 = 256 * K1
    K512 = 512 * K1
    M1 = 1024 * K1
    M4 = 4 * M1

    @staticmethod
    def round_up(value: int, step: int):
        return (value + step - 1) // step * step


def _offset_pointer(pointer, pointer_class, offset):
    return pointer_class(ctypes.addressof(pointer) + offset)


def load_library(path: Path):
    return cdll.LoadLibrary(str(path))


platform_name = platform.system()

if platform_name == "Windows":
    lz4_libname = "msys-lz4-1.dll"

elif platform_name == "Linux":
    lz4_libname = "liblz4.so.1"
else:
    raise NotImplementedError(f"{platform_name}:{platform.architecture()} is not supported")


class LZ4Wrapper:
    lib_cdll: ctypes.CDLL = load_library(Path(__file__).parent / lz4_libname)

    @classmethod
    def reload_library(cls, path: Path):
        cls.lib_cdll = cdll.LoadLibrary(str(path))

    # LZ4LIB_API int LZ4_compressBound(int inputSize);
    _lz4_compress_bound = lib_cdll.LZ4_compressBound
    _lz4_compress_bound.argtypes = [c_int32]
    _lz4_compress_bound.restype = c_int32

    def compress_bound(self, size):
        return self._lz4_compress_bound(size)

    # LZ4LIB_API int LZ4_decompress_safe (const char* src, char* dst, int compressedSize, int dstCapacity);
    _lz4_decompress_safe = lib_cdll.LZ4_decompress_safe
    _lz4_decompress_safe.argtypes = [c_char_p, c_char_p, c_int32, c_int32]
    _lz4_decompress_safe.restype = c_int32

    def decompress_safe(self, compressed_data: bytes, decompressed_size: int):
        compressed_buffer = create_string_buffer(compressed_data)
        decompressed_buffer = create_string_buffer(decompressed_size)
        rv = self._lz4_decompress_safe(compressed_buffer, decompressed_buffer, len(compressed_data), decompressed_size)
        assert rv > 1, f'Received error code from LZ4:{rv}'
        assert rv == decompressed_size
        decompressed_data = bytes(decompressed_buffer.raw[:rv])
        del compressed_buffer
        del decompressed_buffer
        return decompressed_data

    # LZ4LIB_API int LZ4_compress_fast (const char* src, char* dst, int srcSize, int dstCapacity, int acceleration);
    _lz4_compress_fast = lib_cdll.LZ4_compress_fast
    _lz4_compress_fast.argtypes = [c_char_p, c_char_p, c_int32, c_int32, c_int32]
    _lz4_compress_fast.restype = c_int32

    def compress_fast(self, data: bytes, acceleration=1):
        assert acceleration < 65537, f'{acceleration} is higher than LZ4_ACCELERATION_MAX(65537)'
        data_buffer = create_string_buffer(data)
        minimum_buffer_size = self.compress_bound(len(data))
        compressed_buffer = create_string_buffer(minimum_buffer_size)
        rv = self._lz4_compress_fast(data_buffer, compressed_buffer, len(data), minimum_buffer_size, acceleration)
        assert rv > 1, f'Received error code from LZ4:{rv}'
        compressed_data = bytes(compressed_buffer.raw[:rv])
        del data_buffer
        del compressed_buffer
        return compressed_data


class LZ4StreamWrapper(LZ4Wrapper):
    class LZ4_streamDecode_t(ctypes.Structure):
        pass

    LZ4_streamDecode_t._fields_ = [
        ('externalDict', c_char_p),
        ('extDictSize', c_uint32),
        ('prefixEnd', c_char_p),
        ('prefixSize', c_uint32),
    ]

    lib_cdll = LZ4Wrapper.lib_cdll

    def __init__(self):
        self._stream_state = self._lz4_create_steam_decode()
        pass

    def __del__(self):
        rv = self._lz4_free_steam_decode(self._stream_state)
        assert rv == 0, f'Received error code from LZ4:{rv}'

    # LZ4LIB_API LZ4_streamDecode_t* LZ4_createStreamDecode(void);
    _lz4_create_steam_decode = lib_cdll.LZ4_createStreamDecode
    _lz4_create_steam_decode.argtypes = []
    _lz4_create_steam_decode.restype = POINTER(LZ4_streamDecode_t)

    # LZ4LIB_API int                 LZ4_freeStreamDecode (LZ4_streamDecode_t* LZ4_stream);
    _lz4_free_steam_decode = lib_cdll.LZ4_freeStreamDecode
    _lz4_free_steam_decode.argtypes = [POINTER(LZ4_streamDecode_t)]
    _lz4_free_steam_decode.restype = c_int32


class LZ4ChainDecoder(LZ4StreamWrapper):
    lib_cdll = LZ4Wrapper.lib_cdll

    def __init__(self, block_size, extra_blocks=0):
        super().__init__()
        self._block_size = Mem.round_up(max(block_size, Mem.K1), Mem.K1)
        self._extra_blocks = max(extra_blocks, 0)
        self._output_length = Mem.K64 + (1 + extra_blocks) * self._block_size + 32
        self._output_index = 0
        self._output_buffer = create_string_buffer(self._output_length + 8)

    def _decode(self, source: c_char_p, source_size, block_size: int) -> int:
        if block_size <= 0:
            block_size = self._block_size
        self._prepare(block_size)
        tmp = _offset_pointer(self._output_buffer, c_char_p, self._output_index)
        decoded_size = self._decode_block(source, source_size, tmp, block_size)
        assert decoded_size > 0, f'Received error code from LZ4:{decoded_size}'
        self._output_index += decoded_size
        return decoded_size

    def _prepare(self, block_size: int) -> None:
        if self._output_index + block_size <= self._output_length:
            return
        self._output_index = self._copy_dict(self._output_index)

    def _copy_dict(self, index):
        dict_start = max(index - Mem.K64, 0)
        dict_size = index - dict_start
        self._output_buffer.raw[:] = self._output_buffer[dict_size:]
        self._stream_state.prefixSize = dict_size
        self._stream_state.prefixEnd = _offset_pointer(self._output_buffer, c_char_p, dict_size)
        self._stream_state.externalDict = 0
        self._stream_state.extDictSize = 0
        return dict_size

    # LZ4LIB_API int LZ4_decompress_safe_continue (LZ4_streamDecode_t* LZ4_streamDecode, const char* src, char* dst, int srcSize, int dstCapacity);
    _lz4_decompress_safe_continue = lib_cdll.LZ4_decompress_safe_continue
    _lz4_decompress_safe_continue.argtypes = [POINTER(LZ4StreamWrapper.LZ4_streamDecode_t), c_char_p, c_char_p, c_int32,
                                              c_int32]
    _lz4_decompress_safe_continue.restype = c_int32

    def _decode_block(self, data, data_size: int, target, target_size):
        rv = self._lz4_decompress_safe_continue(self._stream_state, data, target, data_size, target_size)
        del data
        return rv

    def _drain(self, target: c_char_p, offset: int, size: int) -> None:
        offset = self._output_index + offset
        if offset < 0 or size < 0 or offset + size > self._output_index:
            raise AssertionError('Invalid operation')

        memmove(target, _offset_pointer(self._output_buffer, c_char_p, offset), size)

    def _decode_and_drain(self, source, source_size, target, target_size) -> Tuple[bool, int]:
        decoded = 0
        if source_size <= 0:
            return False, decoded
        decoded = self._decode(source, source_size, target_size)
        if decoded <= 0 or target_size < decoded:
            return False, decoded
        self._drain(target, -decoded, decoded)
        return True, decoded

    def decompress(self, compressed_data, decompressed_size):
        data = create_string_buffer(compressed_data)
        output = create_string_buffer(decompressed_size)
        self._decode_and_drain(data, len(compressed_data), output, decompressed_size)
        return bytes(output.raw)


if __name__ == '__main__':
    a = LZ4ChainDecoder(1)
    d = b'b\x01\x00\x00\x00\x13\x00\x01\x00\xe00H\x0ef\x06\x00\x00\x00\x01\xfd=\xdc\x0b\x1e\x11\x00p\x00\xc6V\xa5#\x1a\x00\x14\x00\x02#\x00fHw\x17\x12\x1a\x00\x01\x00B\x12\xda\xd3v*\x00B\x80i;\x1c\n\x00SkXW\xe5\x01 \x00B\xb7\xb0h=\x0c\x00b\x80\xbf\x19\xbf[5"\x00@\xf0i\xb6\xfc\n\x00r1\x00\xb2\x01\x0cx\t+\x00`\xff#\x99\x00\xcd\x06\n\x00@b\x1c\x1b\xc6\x1f\x00\xf1\x00worldspawn\x00\x16\x02 \xe4<\x00!.0\x01\x00\x1d \t\x00_\x00\xcf\xda\x98\xba#\x00\x0cA,\xe4\xc1\x19x\x00\x04\x1a\x00\r\t\x00R\x008\xa0c\xa9\x9b\x00@\x86\x81\x8b\x9c\n\x00\xe0construct\x00\xb3b\xf2\xd3\x12\x00\xd0sky_day01_01\x00'
    b = create_string_buffer(321)
    cd = a._decode_block(d, len(d), b, 321)
    print(cd)
