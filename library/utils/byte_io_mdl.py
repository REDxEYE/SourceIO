import binascii
import contextlib
import io
import struct
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, List, Union

from . import Buffer


class OffsetOutOfBounds(Exception):
    pass


def split(array, n=3):
    return [array[i:i + n] for i in range(0, len(array), n)]


@dataclass
class Region:
    name: str
    start: int
    end: int


class ByteIO:
    @contextlib.contextmanager
    def save_current_pos(self):
        entry = self.tell()
        yield
        self.seek(entry)

    def __init__(self, path_or_file_or_data: Union[str, Path, BinaryIO, bytes, bytearray, Buffer] = None,
                 open_to_read=True):
        if path_or_file_or_data is not None:
            self.assert_file_exists(path_or_file_or_data)
        if hasattr(path_or_file_or_data, 'mode'):
            file = path_or_file_or_data
            self.file = file
        elif type(path_or_file_or_data) is str or isinstance(path_or_file_or_data, Path):
            mode = 'rb' if open_to_read else 'wb'
            self.file = open(path_or_file_or_data, mode)
        elif type(path_or_file_or_data) in [bytes, bytearray]:
            self.file = io.BytesIO(path_or_file_or_data)
        elif issubclass(type(path_or_file_or_data), io.IOBase):
            self.file = path_or_file_or_data
        elif isinstance(path_or_file_or_data, ByteIO):
            self.file = path_or_file_or_data.file
        elif isinstance(path_or_file_or_data, Buffer):
            self.file = path_or_file_or_data
        else:
            self.file = BytesIO()

        self.regions: List[Region] = []

        self.used_regions = []

    @staticmethod
    def assert_file_exists(input_data: Union[str, Path, BinaryIO, bytes, bytearray]):
        if isinstance(input_data, str):
            res = Path(input_data).exists()
        elif isinstance(input_data, Path):
            res = input_data.exists()
        elif isinstance(input_data, (bytes, bytearray)):
            res = bool(input_data)
        elif isinstance(input_data, ByteIO):
            res = bool(input_data.file)
        elif isinstance(input_data, Buffer):
            res = not input_data.closed
        elif isinstance(input_data, (BinaryIO, io.BufferedReader, BytesIO)):
            res = not input_data.closed
        else:
            raise Exception(f'Unknown input data: {input_data}:{type(input_data)}')
        assert res, f'Failed to open file: {input_data}'

    @property
    def sorted_regions(self):
        return sorted(self.regions, key=lambda region: region.start)

    def begin_region(self, name):
        self.regions.append(Region(name, self.tell(), -1))

    def end_region(self):
        self.regions[-1].end = self.tell()

    def __del__(self):
        if isinstance(self.file, BytesIO):
            return
        self.close()

    @property
    def preview(self):
        with self.save_current_pos():
            return self.read(64)

    @property
    def preview_f(self):
        with self.save_current_pos():
            block = self.read(64)
            hex_values = split(split(binascii.hexlify(block).decode().upper(), 2), 4)
            return [' '.join(b) for b in hex_values]

    def __repr__(self):
        return "<ByteIO {}/{}>".format(self.tell(), self.size())

    def close(self):
        if hasattr(self.file, 'close'):
            self.file.close()

    def rewind(self, amount):
        self.file.seek(-amount, io.SEEK_CUR)

    def skip(self, amount):
        self.file.seek(amount, io.SEEK_CUR)

    def seek(self, off, pos: int = io.SEEK_SET):
        self.file.seek(off, pos)

    def tell(self):
        return self.file.tell()

    def remaining(self):
        return self.size() - self.tell()

    def size(self):
        curr_offset = self.tell()
        self.seek(0, io.SEEK_END)
        ret = self.tell()
        self.seek(curr_offset, io.SEEK_SET)
        return ret

    def fill(self, amount):
        for _ in range(amount):
            self._write(b'\x00')

    def insert_begin(self, to_insert):
        self.seek(0)
        buffer = self.read(-1)

        del self.file
        self.file = BytesIO()
        self.file.write(to_insert)
        self.file.write(buffer)
        self.file.seek(0)

    # ------------ PEEK SECTION ------------ #

    def peek(self, size=1):
        with self.save_current_pos():
            return self.read(size)

    def peek_single(self, t):
        size = struct.calcsize(t)
        return struct.unpack(t, self.peek(size))[0]

    def peek_fmt(self, fmt):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.peek(size))

    def peek_uint64(self):
        return self.peek_single('Q')

    def peek_int64(self):
        return self.peek_single('q')

    def peek_uint32(self):
        return self.peek_single('I')

    def peek_int32(self):
        return self.peek_single('i')

    def peek_uint16(self):
        return self.peek_single('H')

    def peek_int16(self):
        return self.peek_single('h')

    def peek_uint8(self):
        return self.peek_single('B')

    def peek_int8(self):
        return self.peek_single('b')

    def peek_float(self):
        return self.peek_single('f')

    def peek_double(self):
        return self.peek_single('d')

    def peek_fourcc(self):
        with self.save_current_pos():
            return self.read_ascii_string(4)

    # ------------ READ SECTION ------------ #

    def read(self, size=-1) -> bytes:
        self.used_regions.append((self.tell(), self.tell() + size))
        return self.file.read(size)

    def _read(self, t):
        return struct.unpack(t, self.file.read(struct.calcsize(t)))[0]

    def read_fmt(self, fmt):
        return struct.unpack(fmt, self.file.read(struct.calcsize(fmt)))

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
            buffer = self.file.read(length).strip(b'\x00')
            if b'\x00' in buffer:
                buffer = buffer[:buffer.index(b'\x00')]
            return buffer.decode('latin', errors='replace').strip()

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

    def read_from_offset(self, offset, reader, **reader_args):
        if offset > self.size():
            raise OffsetOutOfBounds()
        with self.save_current_pos():
            self.seek(offset, io.SEEK_SET)
            ret = reader(**reader_args)
        return ret

    def read_source1_string(self, entry):
        offset = self.read_int32()
        if offset:
            with self.save_current_pos():
                self.seek(entry + offset)
                return self.read_ascii_string()
        else:
            return ""

    def read_source2_string(self):
        entry = self.tell()
        offset = self.read_int32()
        with self.save_current_pos():
            self.seek(entry + offset)
            return self.read_ascii_string()

    # ------------ WRITE SECTION ------------ #

    def _write(self, data):
        self.file.write(data)

    def write(self, t, value):
        self._write(struct.pack(t, value))

    def write_fmt(self, fmt: str, *values):
        self._write(struct.pack(fmt, *values))

    def write_uint64(self, value):
        self.write('Q', value)

    def write_int64(self, value):
        self.write('q', value)

    def write_uint32(self, value):
        self.write('I', value)

    def write_int32(self, value):
        self.write('i', value)

    def write_uint16(self, value):
        self.write('H', value)

    def write_int16(self, value):
        self.write('h', value)

    def write_uint8(self, value):
        self.write('B', value)

    def write_int8(self, value):
        self.write('b', value)

    def write_float(self, value):
        self.write('f', value)

    def write_double(self, value):
        self.write('d', value)

    def write_ascii_string(self, string, zero_terminated=False, length=-1):
        pos = self.tell()
        for c in string:
            self._write(c.encode('ascii'))
        if zero_terminated:
            self._write(b'\x00')
        elif length != -1:
            to_fill = length - (self.tell() - pos)
            if to_fill > 0:
                for _ in range(to_fill):
                    self.write_uint8(0)

    def write_fourcc(self, fourcc):
        self.write_ascii_string(fourcc)

    def write_to_offset(self, offset, writer, value, fill_to_target=False):
        if offset > self.size() and not fill_to_target:
            raise OffsetOutOfBounds()
        curr_offset = self.tell()
        self.seek(offset, io.SEEK_SET)
        ret = writer(value)
        self.seek(curr_offset, io.SEEK_SET)
        return ret

    def read_float16(self):
        return self._read('e')

    def write_bytes(self, data):
        self._write(data)

    def __bool__(self):
        return self.tell() < self.size()

    # EXTENSION

    def read_rle_shorts(self, count):
        values = []
        total_count = 0
        for i in range(count):
            raw_count, encoded_count = self.read_fmt('2B')
            vals = self.read_fmt(f'{raw_count}h')
            for j in range(encoded_count):
                if total_count == count:
                    break
                total_count += 1
                idx = min(raw_count - 1, j)
                values.append(vals[idx])
            if total_count == count:
                break
        return values

    def read_structure_array(self, offset, count, data_class, *args, **kwargs):
        if count == 0:
            return []
        self.seek(offset)
        object_list = []
        for _ in range(count):
            obj = data_class()
            obj.read(self, *args, **kwargs)
            object_list.append(obj)
        return object_list

    def align(self, align_to):
        value = self.tell()
        padding = (align_to - value % align_to) % align_to
        self.seek(padding, io.SEEK_CUR)


if __name__ == '__main__':
    a = ByteIO(r'./test.bin')
    a.write_fourcc("IDST")
    # a.write_int8(108)
    # a.write_uint32(104)
    # a.write_to_offset(1024,a.write_uint32,84,True)
    # a.write_double(15.58)
    # a.write_float(18.58)
    # a.write_uint64(18564846516)
    # a.write_ascii_string('Test123')
    a.close()
    a = ByteIO(open(r'./test.bin', mode='rb'))
    print(a.peek_uint32())
    # print(a.read_from_offset(1024,a.read_uint32))
    # print(a.read_uint32())
    # print(a.read_double())
    # print(a.read_float())
    # print(a.read_uint64())
    # print(a.read_ascii_string())
