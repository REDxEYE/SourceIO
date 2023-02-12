from pathlib import Path

import numpy as np

from ...utils import MemoryBuffer
from ...utils.singleton import SingletonMeta
from . import tables

ROUND_ITER = 16
BLOCK_ITER = 4
STATIC_KEY = "MBHEROES!@0u9"
KEY_ITER = 0x3F
KEY_SIZE = 128
HASH_SIZE = 16

try:
    import ctypes

    rel_path = Path(__file__).absolute().parent


    class Serpent(metaclass=SingletonMeta):
        def __init__(self):
            self.serpent_dll = ctypes.cdll.LoadLibrary(str(rel_path / 'Serpent.dll'))
            self._set_key = self.serpent_dll.set_key
            self._set_key.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),
                                      ctypes.c_uint32]
            self._get_key = self.serpent_dll.get_key
            self._get_key.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),
                                      ctypes.c_uint32]
            self._get_sw = self.serpent_dll.get_sw
            self._get_sw.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='C_CONTIGUOUS'),
                                     ctypes.c_uint32]
            self._get_ks = self.serpent_dll.get_ks
            self._get_ks.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='C_CONTIGUOUS'),
                                     ctypes.c_uint32]
            self._update_stream_keys = self.serpent_dll.update_stream_keys
            self._decrypt = self.serpent_dll.decrypt
            self._decrypt.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),
                                      ctypes.c_uint32]

        def set_key(self, key):
            assert len(key) == 128
            self._set_key(key, 128)
            assert (key == self.get_key()).all()

        def get_key(self):
            key = np.zeros(128, np.uint8)
            self._get_key(key, 128)
            return key

        def get_sw(self):
            key = np.zeros(16, np.uint32)
            self._get_sw(key, 64)
            return key

        def get_ks(self):
            key = np.zeros(16, np.uint32)
            self._get_ks(key, 64)
            return key

        @property
        def key(self):
            return self.get_key()

        @property
        def sw(self):
            return self.get_sw()

        @property
        def ks(self):
            return self.get_ks()

        def update_stream_keys(self):
            self._update_stream_keys()

        def decrypt(self, buffer: np.ndarray):
            self._decrypt(buffer, buffer.shape[0])
            return buffer

        def decrypt_to_reader(self, data: bytes):
            buffer = np.frombuffer(data, np.uint8).copy()
            return MemoryBuffer(self.decrypt(buffer).tobytes())


except:

    class Serpent(metaclass=SingletonMeta):

        def __init__(self):
            self.key = np.zeros(128, np.uint8)
            self.sw = np.zeros(16, np.uint32)
            self.ks = np.zeros(16, np.uint32)
            self.r1 = 0
            self.r2 = 0
            self.key_index = 0
            self.round = 0

        def set_key(self, key):
            self.key[0:128] = key
            sw = self.sw

            sw[15] = get_int_at_big(self.key, 0)
            sw[14] = get_int_at_big(self.key, 4)
            sw[13] = get_int_at_big(self.key, 8)
            sw[12] = get_int_at_big(self.key, 12)
            sw[11] = ~sw[15]
            sw[10] = ~sw[14]
            sw[9] = ~sw[13]
            sw[8] = ~sw[12]

            sw[7] = sw[15]
            sw[6] = sw[14]
            sw[5] = sw[13]
            sw[4] = sw[12]

            sw[3] = ~sw[15]
            sw[2] = ~sw[14]
            sw[1] = ~sw[13]
            sw[0] = ~sw[12]

            self.r1 = 0
            self.r2 = 0

            for i in range(2):
                for j in range(16):
                    w1 = self.r1 + sw[(j + 15) & 15] ^ self.r2
                    sw[j] = tables.mul(sw[j]) ^ sw[(j + 2) & 15] ^ tables.div(sw[(j + 11) & 15]) ^ w1
                    w2 = (self.r2 + sw[(j + 5) & 15]) & 0xFFFFFFFF
                    self.r2 = tables.S1_T0[self.r1 & 0xff] ^ \
                              tables.S1_T1[(self.r1 >> 8) & 0xff] ^ \
                              tables.S1_T2[(self.r1 >> 16) & 0xff] ^ \
                              tables.S1_T3[(self.r1 >> 24) & 0xff]
                    self.r1 = w2
            self.key_index = 0
            self.round = 16

        def update_stream_keys(self):
            # Avoid object lookup
            sw = self.sw
            ks = self.ks
            mul = tables.mul
            div = tables.div
            s1_t0 = tables.S1_T0
            s1_t1 = tables.S1_T1
            s1_t2 = tables.S1_T2
            s1_t3 = tables.S1_T3
            r1 = self.r1
            r2 = self.r2

            for j in range(16):
                sw[j] = mul(sw[j]) ^ sw[(j + 2) & 15] ^ div(sw[(j + 11) & 15])
                w2 = (r2 + sw[(j + 5) & 15])
                r2 = s1_t0[r1 & 0xff] ^ s1_t1[(r1 >> 8) & 0xff] ^ s1_t2[(r1 >> 16) & 0xff] ^ s1_t3[(r1 >> 24) & 0xff]
                r1 = w2
                ks[j] = (r1 + sw[j]) ^ r2 ^ sw[(j + 1) & 15]
            self.r1 = r1
            self.r2 = r2

        def decrypt(self, buffer: np.ndarray):
            int_buffer = buffer.view(np.uint32)
            for i in range(int_buffer.shape[0]):
                if self.round >= 16:
                    self.update_stream_keys()
                    self.round = 0
                int_buffer[i] -= self.ks[self.key_index & 15]
                self.round += 1
                self.key_index += 1
            return buffer

        def decrypt_to_reader(self, data: bytes):
            buffer = np.frombuffer(data, np.uint8).copy()
            return MemoryBuffer(self.decrypt(buffer).tobytes())


def get_int_at_big(data: np.ndarray, index):
    data = data.astype(np.int8)
    return ((data[index] << 24) | (data[index + 1] << 16) | (data[index + 2] << 8) | data[index + 3]) & 0xFFFFFFFF


def generate_key(key: str):
    key_buffer = key.lower() + STATIC_KEY
    key_blob = np.zeros(KEY_SIZE, np.uint8)
    for i in range(KEY_SIZE):
        key_blob[i] = ((ord(key_buffer[i % KEY_ITER]) & 0xFF) + i) & 0xFF
    return key_blob


def generate_encoding_key(key):
    key_buffer = key.lower() + STATIC_KEY
    key_blob = np.zeros(KEY_SIZE, np.uint8)
    for i in range(KEY_SIZE):
        key_blob[i] = (i + (i % 3 + 2) * ord(key_buffer[-(i % KEY_ITER + 1)])) & 0xFF
    return key_blob


def generate_hashed_key(key: str, hash_table: bytes):
    key_blob = np.zeros(KEY_SIZE, np.uint8)
    for i in range(KEY_SIZE):
        key_blob[i] = ((hash_table[i % HASH_SIZE] + 2 + i % 5) * ord(key[i % len(key)]) + i)
    return key_blob


__all__ = ["Serpent", "generate_key", "generate_encoding_key", "generate_hashed_key"]
