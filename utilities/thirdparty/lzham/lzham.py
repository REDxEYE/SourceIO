import ctypes
import os
from pathlib import Path
import platform

from ctypes import CDLL, c_uint32, c_char_p, POINTER, Structure, create_string_buffer, pointer, cast
from enum import IntEnum, auto

platform_name = platform.system()


def pointer_to_array(poiter, size, type=ctypes.c_ubyte):
    return cast(poiter, POINTER(type * size))


class DecompressStatus(IntEnum):
    NotFinished = 0
    HasMoreOutput = 1
    NeedsMoreInput = 2
    Success = 3
    Failure = 4
    DestinationBufferTooSmall = 5
    ExpectedMoreRawBytes = 6
    BadCode = 7
    Adler32 = 8
    BadRawBlock = 9
    BadCompBlockSyncCheck = auto()
    BadZlibHeader = auto()
    NeedSeedBytes = auto()
    BadSeedBytes = auto()
    BadSyncBlock = auto()
    InvalidParameter = auto()


class DecompressionParameters(Structure):
    _fields_ = [
        ('m_struct_size', c_uint32),
        ('m_dict_size_log2', c_uint32),
        ('m_decompress_flags', c_uint32),
        ('m_num_seed_bytes', c_uint32),
        ('m_pSeed_bytes', c_char_p),
    ]

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.m_struct_size = ctypes.sizeof(self)


class LZHAM:
    try:
        lib = CDLL(
            str(Path(__file__).absolute().parent / ('lzham_x64' + ('.dll' if platform_name == 'Windows' else '.so'))))

        _get_version = lib.lzham_get_version
        _get_version.argtypes = []
        _get_version.restype = c_uint32

        _compress_init = lib.lzham_compress_init
        _compress_init.argtypes = [c_char_p]
        _compress_init.restype = POINTER(c_uint32)

        _compress_reinit = lib.lzham_compress_reinit
        _compress_reinit.argtypes = [POINTER(c_uint32)]
        _compress_reinit.restype = POINTER(c_uint32)

        _compress = lib.lzham_compress
        _compress.argtypes = [POINTER(c_uint32), c_char_p, POINTER(c_uint32), c_char_p, POINTER(c_uint32), c_uint32]
        _compress.restype = c_uint32

        _compress2 = lib.lzham_compress2
        _compress2.argtypes = [POINTER(c_uint32), c_char_p, POINTER(c_uint32), c_char_p, POINTER(c_uint32), c_uint32]
        _compress2.restype = c_uint32

        _compress_memory = lib.lzham_compress_memory
        _compress_memory.argtypes = [c_char_p, c_char_p, POINTER(c_uint32), c_char_p, c_uint32, POINTER(c_uint32)]
        _compress_memory.restype = c_uint32

        _compress_deinit = lib.lzham_compress_deinit
        _compress_deinit.argtypes = [POINTER(c_uint32)]
        _compress_deinit.restype = c_uint32

        _decompress_init = lib.lzham_decompress_init
        _decompress_init.argtypes = [POINTER(DecompressionParameters)]
        _decompress_init.restype = POINTER(c_uint32)

        _decompress_reinit = lib.lzham_decompress_reinit
        _decompress_reinit.argtypes = [POINTER(c_uint32), POINTER(c_uint32)]
        _decompress_reinit.restype = POINTER(c_uint32)

        _decompress = lib.lzham_decompress
        _decompress.argtypes = [POINTER(c_uint32), c_char_p, POINTER(c_uint32), c_char_p, POINTER(c_uint32), c_uint32]
        _decompress.restype = POINTER(c_uint32)

        _decompress_memory = lib.lzham_decompress_memory
        _decompress_memory.argtypes = [POINTER(DecompressionParameters), c_char_p, POINTER(c_uint32), c_char_p,
                                       POINTER(c_uint32),
                                       POINTER(c_uint32)]
        _decompress_memory.restype = DecompressStatus
    except:
        lib = None

    def __init__(self):
        self.decompress_handle = POINTER(c_uint32)()
        pass

    def version(self):
        return self._get_version()

    def init_decompress(self, dict_size=15):
        params = DecompressionParameters()
        params.m_dict_size_log2 = dict_size
        self.decompress_handle = self._decompress_init(params)
        assert self.decompress_handle.contents.value != 0

    def reinit_decompress(self, dict_size=15):
        params = DecompressionParameters()
        params.m_dict_size_log2 = dict_size
        self.decompress_handle = self._compress_reinit(self.decompress_handle, params)

    @classmethod
    def decompress_memory(cls, compressed_data, decompressed_size, dict_size=15, flags=0):
        compressed_size = len(compressed_data)
        compressed_ptr = create_string_buffer(compressed_data)
        decompressed_ptr = create_string_buffer(decompressed_size)
        compressed_size_ptr = pointer(c_uint32(compressed_size))
        decompressed_size_ptr = pointer(c_uint32(decompressed_size))
        decompressed_params = DecompressionParameters()
        decompressed_params.m_dict_size_log2 = dict_size
        decompressed_params.m_decompress_flags = flags
        adler_prt = pointer(c_uint32(0))
        result = cls._decompress_memory(decompressed_params,
                                        decompressed_ptr, decompressed_size_ptr,
                                        compressed_ptr, compressed_size_ptr,
                                        adler_prt)
        if result == DecompressStatus.Success:
            return pointer_to_array(decompressed_ptr, decompressed_size_ptr.contents.value).contents
        else:
            raise Exception(f'LZHAM decompression error: {result.name}')
