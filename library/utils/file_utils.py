import abc
import binascii
import contextlib
import io
import os
import struct
from pathlib import Path

from struct import unpack, calcsize, pack
from typing import Protocol, Optional


class IBuffer(abc.ABC, io.RawIOBase):
    def __init__(self):
        io.RawIOBase.__init__(self)
        self._endian = '<'

    @contextlib.contextmanager
    def save_current_offset(self):
        entry = self.tell()
        yield
        self.seek(entry)

    @contextlib.contextmanager
    def read_from_offset(self, offset: int):
        entry = self.tell()
        self.seek(offset)
        yield
        self.seek(entry)

    @property
    @abc.abstractmethod
    def data(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def size(self):
        raise NotImplementedError()

    def remaining(self):
        return self.size() - self.tell()

    @property
    def preview(self):
        with self.save_current_offset():
            return binascii.hexlify(self.read(64), sep=' ', bytes_per_sep=4).decode('ascii').upper()

    def align(self, align_to):
        value = self.tell()
        padding = (align_to - value % align_to) % align_to
        self.seek(padding, io.SEEK_CUR)

    def skip(self, size):
        self.seek(size, io.SEEK_CUR)

    def read_fmt(self, fmt):
        return unpack(self._endian + fmt, self.read(calcsize(fmt)))

    def _read(self, fmt):
        return unpack(self._endian + fmt, self.read(calcsize(fmt)))[0]

    def read_relative_offset32(self):
        return self.tell() + self.read_uint32()

    def read_uint64(self):
        return self._read('Q')

    def read_int64(self):
        return self._read('q')

    def read_uint32(self):
        return self._read('I')

    def read_int32(self):
        return self._read('i')

    def read_uint16(self):
        return self._read('H')

    def read_int16(self):
        return self._read('h')

    def read_uint8(self):
        return self._read('B')

    def read_int8(self):
        return self._read('b')

    def read_float(self):
        return self._read('f')

    def read_double(self):
        return self._read('d')

    def read_ascii_string(self, length=None):
        if length is not None:
            buffer = self.read(length).strip(b'\x00').rstrip(b'\x00')
            return buffer.decode('latin', errors='replace')

        buffer = bytearray()

        while True:
            chunk = self.read(32)
            if chunk:
                chunk_end = chunk.find(b'\x00')
            else:
                chunk_end = 0
            if chunk_end >= 0:
                buffer += chunk[:chunk_end]
            else:
                buffer += chunk
            if chunk_end >= 0:
                self.seek(-(len(chunk) - chunk_end - 1), io.SEEK_CUR)
                return buffer.decode('latin', errors='replace')

    def read_fourcc(self):
        return self.read_ascii_string(4)

    def write_fmt(self, fmt: str, *values):
        self.write(pack(self._endian + fmt, *values))

    def write_uint64(self, value):
        self.write_fmt('Q', value)

    def write_int64(self, value):
        self.write_fmt('q', value)

    def write_uint32(self, value):
        self.write_fmt('I', value)

    def write_int32(self, value):
        self.write_fmt('i', value)

    def write_uint16(self, value):
        self.write_fmt('H', value)

    def write_int16(self, value):
        self.write_fmt('h', value)

    def write_uint8(self, value):
        self.write_fmt('B', value)

    def write_int8(self, value):
        self.write_fmt('b', value)

    def write_float(self, value):
        self.write_fmt('f', value)

    def write_double(self, value):
        self.write_fmt('d', value)

    def write_ascii_string(self, string, zero_terminated=False, length=-1):
        pos = self.tell()
        for c in string:
            self.write(c.encode('ascii'))
        if zero_terminated:
            self.write(b'\x00')
        elif length != -1:
            to_fill = length - (self.tell() - pos)
            if to_fill > 0:
                for _ in range(to_fill):
                    self.write_uint8(0)

    def write_fourcc(self, fourcc):
        self.write_ascii_string(fourcc)

    def peek_uint32(self):
        with self.save_current_offset():
            return self.read_uint32()

    def set_big_endian(self):
        self._endian = '>'

    def set_little_endian(self):
        self._endian = '<'

    def __bool__(self):
        return self.tell() < self.size()

    def slice(self, offset: Optional[int] = None, size: int = -1) -> 'IBuffer':
        raise NotImplementedError


class MemoryBuffer(IBuffer):

    def __init__(self, buffer: bytes | bytearray | memoryview):
        super().__init__()
        self._buffer = memoryview(buffer)
        self._offset = 0

    @property
    def data(self) -> memoryview:
        return self._buffer

    def size(self):
        return len(self._buffer)

    def _read(self, fmt: str):
        data = struct.unpack_from(self._endian + fmt, self._buffer, self._offset)
        self._offset += struct.calcsize(self._endian + fmt)
        return data[0]

    def read_fmt(self, fmt):
        data = struct.unpack_from(self._endian + fmt, self._buffer, self._offset)
        self._offset += struct.calcsize(self._endian + fmt)
        return data

    def write(self, __b: bytes | bytearray) -> int | None:
        raise NotImplementedError()

    def read(self, __size: int = -1) -> bytes | None:
        if __size == -1:
            data = self._buffer[self._offset:]
            self._offset += len(data)
            return data.tobytes()
        data = self._buffer[self._offset:self._offset + __size]
        self._offset += __size
        return data.tobytes()

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._offset = offset
        elif whence == io.SEEK_CUR:
            self._offset += offset
        elif whence == io.SEEK_END:
            self._offset = self.size() - offset
        else:
            raise ValueError("Invalid whence argument")

        if self._offset > self.size():
            raise BufferError('Offset is out of bounds')

        return self._offset

    def __str__(self) -> str:
        return f'<MemoryBuffer {self.tell()}/{self.size()}>'

    def tell(self) -> int:
        return self._offset

    @property
    def closed(self) -> bool:
        return self._buffer is not None

    def close(self) -> None:
        self._buffer = None

    def slice(self, offset: Optional[int] = None, size: int = -1) -> 'IBuffer':
        if offset is None:
            offset = self._offset

        if size == -1:
            return MemoryBuffer(self._buffer[offset:])
        return MemoryBuffer(self._buffer[offset:offset + size])


class FileBuffer(io.FileIO, IBuffer):

    def __init__(self, file: str | Path | int, mode: str = 'r', closefd: bool = True, opener=None) -> None:
        io.FileIO.__init__(self, file, mode, closefd, opener)
        IBuffer.__init__(self)

    def size(self):
        return os.fstat(self.fileno()).st_size

    @property
    def data(self):
        offset = self.tell()
        self.seek(0)
        _data = self.read()
        self.seek(offset)
        return _data

    def __str__(self) -> str:
        return f'<FileBuffer: {self.name!r} {self.tell()}/{self.size()}>'

    def slice(self, offset: Optional[int] = None, size: int = -1) -> 'IBuffer':
        with self.save_current_offset():
            if offset is not None:
                self.seek(offset)

            if size == -1:
                return MemoryBuffer(self.read())
            return MemoryBuffer(self.read(size))


class Readable(Protocol):
    @classmethod
    def from_buffer(cls, buffer: IBuffer):
        ...


__all__ = ['IBuffer', 'MemoryBuffer', 'FileBuffer', 'Readable']
