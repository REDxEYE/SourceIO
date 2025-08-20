import abc
import binascii
import contextlib
import io
import os
import struct
import sys
import array
from pathlib import Path
from struct import calcsize, pack, unpack
from typing import Optional, Protocol, Union, TypeVar, Type

try:
    from SourceIO.library.utils.tiny_path import TinyPath
except ImportError:
    TinyPath = Path

NATIVE_LITTLE = "<" if (sys.byteorder == "little") else ">"


class Buffer(abc.ABC, io.RawIOBase):
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

    def read_source1_string(self, entry):
        offset = self.read_int32()
        if offset:
            with self.read_from_offset(entry + offset):
                return self.read_nt_string()
        else:
            return ""

    def read_source2_string(self):
        with self.read_from_offset(self.tell() + self.read_int32()):
            return self.read_nt_string()

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
        if padding + self.tell() > self.size():
            return
        self.seek(padding, io.SEEK_CUR)

    def skip(self, size):
        self.seek(size, io.SEEK_CUR)

    def read_fmt(self, fmt):
        return unpack(self._endian + fmt, self.read(calcsize(self._endian + fmt)))

    def _read(self, fmt):
        return unpack(self._endian + fmt, self.read(calcsize(self._endian + fmt)))[0]

    def read_relative_offset32(self):
        return self.tell() + self.read_uint32()

    def read_uint64(self):
        return self._read

    def read_int64(self):
        return self._read("q")

    def read_uint32(self):
        return self._read("I")

    def read_int32(self):
        return self._read("i")

    def read_uint16(self):
        return self._read("H")

    def read_int16(self):
        return self._read("h")

    def read_uint8(self):
        return self._read("B")

    def read_int8(self):
        return self._read("b")

    def read_float(self):
        return self._read("f")

    def read_double(self):
        return self._read("d")

    def read_half(self):
        return self._read("h")

    @abc.abstractmethod
    def read_array(self, fmt: str, count: int):
        pass

    def read_nt_string(self):
        buffer = bytearray()

        while True:
            chunk = self.read(min(32, self.remaining()))
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

    def read_ascii_string(self, length: Optional[int] = None):
        if length is not None:
            buffer = self.read(length).strip(b'\x00').rstrip(b'\x00')
            if b'\x00' in buffer:
                buffer = buffer[:buffer.index(b'\x00')]
            return buffer.decode('latin', errors='replace')

        return self.read_nt_string()

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

    def peek_fmt(self, fmt):
        with self.save_current_offset():
            return self.read_fmt(fmt)

    def peek(self, size: int):
        with self.save_current_offset():
            return self.read(size)

    def set_big_endian(self):
        self._endian = '>'

    def set_little_endian(self):
        self._endian = '<'

    def __bool__(self):
        return self.tell() < self.size()

    def slice(self, offset: Optional[int] = None, size: int = -1) -> 'Buffer':
        raise NotImplementedError

    def read_structure_array(self, offset, count, data_class: Type['Readable']):
        if count == 0:
            return []
        self.seek(offset)
        object_list = []
        for _ in range(count):
            obj = data_class.from_buffer(self)
            object_list.append(obj)
        return object_list


class MemoryBuffer(Buffer):

    def __init__(self, buffer: bytes | bytearray | memoryview):
        super().__init__()
        self._buffer = memoryview(buffer)
        self._offset = 0
        self._struct_cache_le: dict[str, struct.Struct] = {}
        self._struct_cache_be: dict[str, struct.Struct] = {}
        self._struct_cache = self._struct_cache_le

    @property
    def data(self) -> memoryview:
        return self._buffer

    def size(self):
        return len(self._buffer)

    def _get_struct(self, fmt: str) -> struct.Struct:
        s = self._struct_cache.get(fmt)
        if s is None:
            s = struct.Struct(self._endian + fmt)
            self._struct_cache[fmt] = s
        return s

    def _read_struct(self, s: struct.Struct):
        val = s.unpack_from(self._buffer, self._offset)
        self._offset += s.size
        return val

    _uint64_le = struct.Struct('<Q')
    _uint64_be = struct.Struct('>Q')
    _int64_le = struct.Struct('<q')
    _int64_be = struct.Struct('>q')
    _uint32_le = struct.Struct('<I')
    _uint32_be = struct.Struct('>I')
    _int32_le = struct.Struct('<i')
    _int32_be = struct.Struct('>i')
    _uint16_le = struct.Struct('<H')
    _uint16_be = struct.Struct('>H')
    _int16_le = struct.Struct('<h')
    _int16_be = struct.Struct('>h')
    _float32_le = struct.Struct('<f')
    _float32_be = struct.Struct('>f')
    _double_le = struct.Struct('<d')
    _double_be = struct.Struct('>d')
    _half_le = struct.Struct('<e')
    _half_be = struct.Struct('>e')

    def read_uint64(self):
        return (self._read_struct(self._uint64_le) if self._endian == "<" else self._read_struct(self._uint64_be))[0]

    def read_int64(self):
        return (self._read_struct(self._int64_le) if self._endian == "<" else self._read_struct(self._int64_be))[0]

    def read_uint32(self):
        return (self._read_struct(self._uint32_le) if self._endian == "<" else self._read_struct(self._uint32_be))[0]

    def read_int32(self):
        return (self._read_struct(self._int32_le) if self._endian == "<" else self._read_struct(self._int32_be))[0]

    def read_uint16(self):
        return (self._read_struct(self._uint16_le) if self._endian == "<" else self._read_struct(self._uint16_be))[0]

    def read_int16(self):
        return (self._read_struct(self._int16_le) if self._endian == "<" else self._read_struct(self._int16_be))[0]

    def read_uint8(self):
        return self._read("B")

    def read_int8(self):
        return self._read("b")

    def read_float(self):
        return (self._read_struct(self._float32_le) if self._endian == "<" else self._read_struct(self._float32_be))[0]

    def read_double(self):
        return (self._read_struct(self._double_le) if self._endian == "<" else self._read_struct(self._double_be))[0]

    def read_half(self):
        return (self._read_struct(self._half_le) if self._endian == "<" else self._read_struct(self._half_be))[0]

    def _read(self, fmt: str):
        return self._read_struct(self._get_struct(fmt))[0]

    def read_fmt(self, fmt: str):
        return self._read_struct(self._get_struct(fmt))

    def read_array(self, fmt: str, count: int):
        itemsize = struct.calcsize(fmt)
        nbytes = count * itemsize
        mv = self.read(nbytes)
        a = array.array(fmt)
        a.frombytes(mv)  # accepts memoryview; no temporary bytes object created
        if self._endian != NATIVE_LITTLE and fmt not in ("b", "B"):
            a.byteswap()
        return a.tolist()

    def write(self, _b: Union[bytes, bytearray]) -> Optional[int]:
        if self._offset + len(_b) > self.size():
            raise BufferError(f"Not enough space left({self.remaining()}) in buffer to write {len(_b)} bytes")
        self._buffer[self._offset:self._offset + len(_b)] = _b
        self._offset += len(_b)
        return len(_b)

    def read(self, _size: int = -1) -> Optional[bytes]:
        if _size == -1:
            data = self._buffer[self._offset:]
            self._offset += len(data)
            return data.tobytes()
        data = self._buffer[self._offset:self._offset + _size]
        self._offset += _size
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
        return self._buffer is None

    def close(self) -> None:
        self._buffer = None

    def read_nt_string(self):
        obj = self._buffer.obj
        try:
            end = obj.find(b"\x00", self._offset)
        except AttributeError:
            end = self._buffer.tobytes().find(b"\x00", self._offset)  # rare fallback
        buffer_size = len(self._buffer)
        if end == -1:
            end = buffer_size
        s = bytes(self._buffer[self._offset:end]).decode("utf8", "replace")
        self._offset = end + (1 if end < buffer_size else 0)
        return s

    def slice(self, offset: Optional[int] = None, size: int = -1) -> 'MemorySlice':
        if offset is None:
            offset = self._offset
        slice_offset = self.tell()
        if size == -1:
            return MemorySlice(self._buffer[offset:], slice_offset)
        return MemorySlice(self._buffer[offset:offset + size], slice_offset)

    def set_big_endian(self):
        """Switch active cache without rebuilding keys."""
        self._endian = '>'
        self._struct_cache = self._struct_cache_be

    def set_little_endian(self):
        """Switch active cache without rebuilding keys."""
        self._endian = '<'
        self._struct_cache = self._struct_cache_le


class WritableMemoryBuffer(io.BytesIO, Buffer):
    def __init__(self, initial_bytes=None):
        io.BytesIO.__init__(self, initial_bytes)
        Buffer.__init__(self)

    @property
    def data(self):
        return self.getbuffer()

    def size(self):
        return len(self.getbuffer())

    def slice(self, offset: Optional[int] = None, size: int = -1) -> 'MemorySlice':
        if offset is None:
            offset = self.tell()

        if size == -1:
            return MemoryBuffer(self.data[offset:])
        return MemoryBuffer(self.data[offset:offset + size])

    def read_array(self, fmt: str, count: int):
        itemsize = struct.calcsize(fmt)
        nbytes = count * itemsize
        buffer = self.getbuffer()
        offset = self.tell()
        mv = buffer[offset:offset + nbytes]
        offset += nbytes
        self.seek(offset)
        a = array.array(fmt)
        a.frombytes(mv)  # accepts memoryview; no temporary bytes object created
        mv.release()  # release memoryview to avoid keeping the whole buffer in memory
        if self._endian != NATIVE_LITTLE and fmt not in ("b", "B"):
            a.byteswap()
        return a


class FileBuffer(io.FileIO, Buffer):

    def __init__(self, file: Union[str, TinyPath, Path, int], mode: str = 'r', closefd: bool = True,
                 opener=None) -> None:
        io.FileIO.__init__(self, file, mode, closefd, opener)
        Buffer.__init__(self)
        self._cached_size = None
        self._is_read_only = mode == "r" or mode == "rb"

    def size(self):
        if self._is_read_only:
            if self._cached_size is None:
                self._cached_size = os.fstat(self.fileno()).st_size
            return self._cached_size
        with self.save_current_offset():
            self.seek(0, io.SEEK_END)
            size = self.tell()
            return size

    def remaining(self):
        return self.size() - self.tell()

    @property
    def data(self):
        offset = self.tell()
        self.seek(0)
        _data = self.read()
        self.seek(offset)
        return _data

    def read_array(self, fmt: str, count: int):
        """Bulk read from file directly into array.array using fromfile; copies once."""
        a = array.array(fmt)
        a.fromfile(self, count)  # fast C-level read; advances file offset
        if self._endian != NATIVE_LITTLE and fmt not in ("b", "B"):
            a.byteswap()
        return a

    def __str__(self) -> str:
        return f'<FileBuffer: {self.name!r} {self.tell()}/{self.size()}>'

    def slice(self, offset: Optional[int] = None, size: int = -1) -> 'MemorySlice':
        with self.save_current_offset():
            if offset is not None:
                self.seek(offset)
            slice_offset = self.tell()
            if size == -1:
                return MemorySlice(self.read(), slice_offset)
            return MemorySlice(self.read(size), slice_offset)


class MMapBuffer(MemoryBuffer):
    def __init__(self, path: str | Path):
        import mmap, os
        fd = os.open(os.fspath(path), os.O_RDONLY)
        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        os.close(fd)
        super().__init__(mm)
        self._mm = mm

    def close(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = None



class MemorySlice(MemoryBuffer):
    def __init__(self, buffer: Union[bytes, bytearray, memoryview], offset: int):
        super().__init__(buffer)
        self._slice_offset = offset

    def abs_tell(self):
        return self.tell() + self._slice_offset


T = TypeVar("T")


class Readable(Protocol):
    @classmethod
    def from_buffer(cls: Type[T], buffer: Buffer) -> T:
        ...


__all__ = ['Buffer', 'MemoryBuffer', 'WritableMemoryBuffer', 'FileBuffer', 'MMapBuffer', 'MemorySlice', 'Readable']
