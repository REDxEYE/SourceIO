from __future__ import annotations
import abc
import binascii
import contextlib
import io
import os
import struct
import sys
import array
import typing
from dataclasses import dataclass, field
from pathlib import Path
from struct import calcsize, pack, unpack
from typing import Optional, Protocol, Union, TypeVar, Type, Callable, Any

try:
    from SourceIO.library.utils.tiny_path import TinyPath
except ImportError:
    TinyPath = Path

NATIVE_LITTLE = "<" if (sys.byteorder == "little") else ">"


@dataclass(slots=True)
class Label:
    name: str
    buffer: Buffer
    offset: int
    size: int
    callback: Callable[[Buffer, Label], ...]

    extra_data: dict[str, Any] = field(default_factory=dict)

    def write(self, fmt: str, *value: Any):
        if self.size < calcsize(fmt):
            raise BufferError(f"Not enough space left in label '{self.name}' ({self.size} bytes left) to write {fmt}")
        with self.buffer.read_from_offset(self.offset):
            size = self.buffer.write_fmt(fmt, *value)
            self.size -= size
            self.offset += size

    def __setitem__(self, key: str, value: Any):
        self.extra_data[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.extra_data.get(key, None)

    def get(self, key: str, default: Any = None) -> Any:
        return self.extra_data.get(key, default)


class Buffer(abc.ABC, io.RawIOBase):
    def __init__(self):
        io.RawIOBase.__init__(self)
        self._endian = '<'
        self._labels: list[Label] = []

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

    def align_pad_to(self, size: int):
        """Align the current position to the next multiple of `size`."""
        if not self.writable():
            raise BufferError("Buffer is not writable.")
        if size <= 0:
            raise ValueError("Alignment size must be greater than zero.")
        current_position = self.tell()
        padding = (size - (current_position % size)) % size
        self.seek(padding, io.SEEK_CUR)

    def skip(self, size):
        self.seek(size, io.SEEK_CUR)

    @abc.abstractmethod
    def ro_view(self, offset: int = -1, size: int = -1) -> memoryview:
        """Create a read-only memoryview of the current buffer's data."""
        pass

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

    def read_nt_string(self, encoding="latin1"):
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
                return buffer.decode(encoding, errors='replace')

    def read_ascii_string(self, length: Optional[int] = None, encoding="latin1"):
        if length is not None:
            buffer = self.read(length).strip(b'\x00').rstrip(b'\x00')
            if b'\x00' in buffer:
                buffer = buffer[:buffer.index(b'\x00')]
            return buffer.decode(encoding, errors='replace')

        return self.read_nt_string()

    def read_fourcc(self):
        return self.read_ascii_string(4)

    def write_fmt(self, fmt: str, *values):
        return self.write(pack(self._endian + fmt, *values))

    def write_uint64(self, value):
        return self.write_fmt('Q', value)

    def write_int64(self, value):
        return self.write_fmt('q', value)

    def write_uint32(self, value):
        return self.write_fmt('I', value)

    def write_int32(self, value):
        return self.write_fmt('i', value)

    def write_uint16(self, value):
        return self.write_fmt('H', value)

    def write_int16(self, value):
        return self.write_fmt('h', value)

    def write_uint8(self, value):
        return self.write_fmt('B', value)

    def write_int8(self, value):
        return self.write_fmt('b', value)

    def write_float(self, value):
        return self.write_fmt('f', value)

    def write_double(self, value):
        return self.write_fmt('d', value)

    def write_ascii_string(self, string, zero_terminated=False, length=-1, encoding="latin1"):
        pos = self.tell()
        self.write(string.encode(encoding))
        if zero_terminated:
            self.write(b'\x00')
            return len(string) + 1
        elif length != -1:
            to_fill = length - (self.tell() - pos)
            if to_fill > 0:
                for _ in range(to_fill):
                    self.write_uint8(0)
            return length
        return len(string)

    def write_fourcc(self, fourcc):
        return self.write_ascii_string(fourcc)

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

    def new_label(self, name: str, size: int, callback: Callable[[Buffer, Label], ...] | None = None) -> Label:
        label = Label(name, self, self.tell(), size, callback)
        self._labels.append(label)
        self.write(b'\x00' * size)  # Preallocate space for the label
        return label


class MemoryBuffer(Buffer):

    def __init__(self, buffer: bytes | bytearray | memoryview):
        super().__init__()
        self._buffer = memoryview(buffer)
        self._offset = 0
        self._struct_cache_le: dict[str, struct.Struct] = {}
        self._struct_cache_be: dict[str, struct.Struct] = {}
        self._struct_cache = self._struct_cache_le

    def new_label(self, name: str, size: int, callback: Callable[[Buffer, Label], ...] | None = None) -> Label:
        raise NotImplementedError("Labels are not supported in MemoryBuffer")

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

    def ro_view(self, offset: int = -1, size: int = -1) -> memoryview:
        if offset == -1:
            offset = self._offset
        if size == -1:
            size = self.size() - offset
        if offset < 0 or offset + size > self.size():
            raise ValueError("Offset and size must be within the bounds of the buffer")
        return self._buffer[offset:offset + size]

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
        if self._offset + _size > self.size():
            raise BufferError(
                f'Read exceeds buffer size: buffer has {self.remaining()} bytes left, tried to read {_size}')

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

    def read_nt_string(self, encoding="latin1"):
        obj = self._buffer.obj
        try:
            end = obj.find(b"\x00", self._offset)
        except AttributeError:
            end = self._buffer.tobytes().find(b"\x00", self._offset)  # rare fallback
        buffer_size = len(self._buffer)
        if end == -1:
            end = buffer_size
        s = bytes(self._buffer[self._offset:end]).decode(encoding, "replace")
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

    def ro_view(self, offset: int = -1, size: int = -1) -> memoryview:
        if offset == -1:
            offset = self.tell()
        if size == -1:
            size = self.size() - offset
        if offset < 0 or offset + size > self.size():
            raise ValueError("Offset and size must be within the bounds of the buffer")
        return self.getbuffer()[offset:offset + size]

    def __del__(self):
        for label in self._labels:
            if label.callback is not None:
                label.callback(self, label)
        self._labels.clear()
        self.close()


class FileBuffer(io.FileIO, Buffer):

    def __init__(self, file: Union[str, TinyPath, Path, int], mode: str = 'r', closefd: bool = True,
                 opener=None) -> None:
        io.FileIO.__init__(self, file, mode, closefd, opener)
        Buffer.__init__(self)
        self._cached_size = None
        self._is_read_only = mode == "r" or mode == "rb"

    def close(self):
        for label in self._labels:
            if label.callback is not None:
                label.callback(self, label)
        self._labels.clear()
        super().close()

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

    def ro_view(self, offset: int = -1, size: int = -1) -> memoryview:
        """Not very efficient; reads data into bytes first, but there is nothing else you can do with simple file objects."""
        if offset == -1:
            offset = self.tell()
        if size == -1:
            size = self.size() - offset
        if offset < 0 or offset + size > self.size():
            raise ValueError("Offset and size must be within the bounds of the buffer")
        with self.save_current_offset():
            self.seek(offset)
            return memoryview(self.read(size))

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
        super().__init__(typing.cast(memoryview,mm))
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


__all__ = ['Buffer', 'MemoryBuffer', 'WritableMemoryBuffer', 'FileBuffer', 'MMapBuffer', 'MemorySlice', 'Readable',
           'Label']
