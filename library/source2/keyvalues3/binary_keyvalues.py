import struct
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from SourceIO.library.utils import Buffer, MemoryBuffer, WritableMemoryBuffer
from SourceIO.library.utils.pylib.compression import LZ4ChainDecoder, lz4_decompress, zstd_decompress_stream, \
    zstd_decompress, zstd_compress_stream, lz4_compress, zstd_compress, LZ4ChainEncoder
from .enums import *
from .types import *


class KV3UnsupportedVersion(Exception):
    pass


class KV3RecursiveReferenceError(Exception):
    pass


def _legacy_block_decompress(in_buffer: Buffer) -> Buffer:
    out_buffer = WritableMemoryBuffer()
    flags = in_buffer.read(4)
    if flags[3] & 0x80:
        out_buffer.write(in_buffer.read(-1))
    working = True
    while in_buffer.tell() != in_buffer.size() and working:
        block_mask = in_buffer.read_uint16()
        for i in range(16):
            if block_mask & (1 << i) > 0:
                offset_and_size = in_buffer.read_uint16()
                offset = ((offset_and_size & 0xFFF0) >> 4) + 1
                size = (offset_and_size & 0x000F) + 3
                lookup_size = offset if offset < size else size

                entry = out_buffer.tell()
                out_buffer.seek(entry - offset)
                data = out_buffer.read(lookup_size)
                out_buffer.seek(entry)
                while size > 0:
                    out_buffer.write(data[:lookup_size if lookup_size < size else size])
                    size -= lookup_size
            else:
                data = in_buffer.read_int8()
                out_buffer.write_int8(data)
            if out_buffer.size() == (flags[2] << 16) + (flags[1] << 8) + flags[0]:
                working = False
                break
    out_buffer.seek(0)
    return out_buffer


def _lz4_chain_blocks_compress(block_raw, block_sizes, frame_size=16384):
    """LZ4-chain compress blocks in fixed-size frames; return (u16 sizes table, concatenated frames)."""
    encoder = LZ4ChainEncoder(frame_size, 0)
    chunk_sizes = WritableMemoryBuffer()
    frames = []
    off = 0
    for size in block_sizes:
        end = off + size
        pos = off
        while pos < end:
            take = min(frame_size, end - pos)
            comp = encoder.compress(block_raw[pos:pos + take])
            chunk_sizes.write_uint16(len(comp))
            frames.append(comp)
            pos += take
        off = end
    return chunk_sizes.data, b"".join(frames)


def read_valve_keyvalue3(buffer: Buffer) -> AnyKVType:
    sig = buffer.read(4)
    if not KV3Signature.is_valid(sig):
        raise BufferError("Not a KV3 buffer")
    sig = KV3Signature(sig)
    if sig == KV3Signature.VKV_LEGACY:
        return read_legacy(buffer)
    elif sig == KV3Signature.KV3_V1:
        return read_v1(buffer)
    elif sig == KV3Signature.KV3_V2:
        return read_v2(buffer)
    elif sig == KV3Signature.KV3_V3:
        return read_v3(buffer)
    elif sig == KV3Signature.KV3_V4:
        return read_v4(buffer)
    elif sig == KV3Signature.KV3_V5:
        return read_v5(buffer)
    raise KV3UnsupportedVersion(f"Unsupported KV3 version: {sig!r}")


def write_valve_keyvalue3(buffer: Buffer, data: AnyKVType, fmt: KV3Format, sig: KV3Signature,
                          compression: KV3CompressionMethod):
    buffer.write_fmt("4s", sig)
    if sig == KV3Signature.VKV_LEGACY:
        write_legacy(buffer, fmt, data, compression)
        return
    buffer.write_fmt("16s", fmt)
    if sig == KV3Signature.KV3_V1:
        write_v1(buffer, data, compression)
    elif sig == KV3Signature.KV3_V2:
        write_v2(buffer, data, compression)
    elif sig == KV3Signature.KV3_V3:
        write_v3(buffer, data, compression)
    elif sig == KV3Signature.KV3_V4:
        write_v4(buffer, data, compression)
    elif sig == KV3Signature.KV3_V5:
        write_v5(buffer, data, compression)


@dataclass
class KV3Buffers:
    byte_buffer: Buffer
    short_buffer: Buffer | None
    int_buffer: Buffer
    double_buffer: Buffer


@dataclass
class KV3Context:
    strings: list[str]
    buffer0: KV3Buffers
    buffer1: KV3Buffers

    types_buffer: Buffer | list[tuple[KV3Type, Specifier]]
    object_member_count_buffer: Buffer
    binary_blob_sizes: list[int] | None
    binary_blob_buffer: Buffer | None

    read_type: Callable[['KV3Context'], tuple[KV3Type, Specifier]]
    read_value: Callable[['KV3Context'], AnyKVType]
    active_buffer: KV3Buffers | None = None


@dataclass
class KV3WriteContext:
    strings: list[str]
    buffer0: KV3Buffers
    buffer1: KV3Buffers

    types_buffer: Buffer
    object_member_count_buffer: Buffer
    binary_blobs: list[bytes] | None

    write_type: Callable[['KV3WriteContext', KV3Type, Specifier], None]
    write_value: Callable[['KV3WriteContext', AnyKVType], None]
    active_buffer: KV3Buffers | None = None

    object_count: int = 0
    array_count: int = 0

    use_extended_types: bool = False
    already_written_objects: set[int] = field(default_factory=set)


def _read_boolean(context: KV3Context, specifier: Specifier):
    value = Bool(context.active_buffer.byte_buffer.read_uint8() == 1)
    value.specifier = specifier
    return value


def _write_boolean(context: KV3WriteContext, value: Bool, specifier: Specifier):
    context.active_buffer.byte_buffer.write_uint8(value)


def _read_int64(context: KV3Context, specifier: Specifier):
    value = Int64(context.active_buffer.double_buffer.read_int64())
    value.specifier = specifier
    return value


def _write_int64(context: KV3WriteContext, value: Int64, specifier: Specifier):
    context.active_buffer.double_buffer.write_int64(value)


def _read_uint64(context: KV3Context, specifier: Specifier):
    value = UInt64(context.active_buffer.double_buffer.read_uint64())
    value.specifier = specifier
    return value


def _write_uint64(context: KV3WriteContext, value: UInt64, specifier: Specifier):
    context.active_buffer.double_buffer.write_uint64(value)


def _read_double(context: KV3Context, specifier: Specifier):
    value = Double(context.active_buffer.double_buffer.read_double())
    value.specifier = specifier
    return value


def _write_double(context: KV3WriteContext, value: Double, specifier: Specifier):
    context.active_buffer.double_buffer.write_double(value)


def _read_string(context: KV3Context, specifier: Specifier):
    str_id = context.active_buffer.int_buffer.read_int32()
    if str_id == -1:
        value = String('')
    else:
        value = String(context.strings[str_id])
    value.specifier = specifier
    return value


def _write_string(context: KV3WriteContext, value: String | str, specifier: Specifier):
    if value == "":
        context.active_buffer.int_buffer.write_int32(-1)
        return
    context.active_buffer.int_buffer.write_int32(_get_string_id(context, value))


def _read_blob(context: KV3Context, specifier: Specifier):
    if context.binary_blob_sizes is not None:
        expected_size = context.binary_blob_sizes.pop(0)
        if expected_size == 0:
            value = BinaryBlob(b"")
        else:
            data = context.binary_blob_buffer.read(expected_size)
            assert len(data) == expected_size, "Binary blob is smaller than expected"
            value = BinaryBlob(data)
    else:
        value = BinaryBlob(context.active_buffer.byte_buffer.read(context.active_buffer.int_buffer.read_int32()))
    value.specifier = specifier
    return value


def _write_blob(context: KV3WriteContext, value: BinaryBlob, specifier: Specifier):
    if context.binary_blobs is not None:
        context.binary_blobs.append(value)
    else:
        context.active_buffer.int_buffer.write_int32(len(value))
        context.active_buffer.byte_buffer.write(value)


def _read_array(context: KV3Context, specifier: Specifier):
    count = context.active_buffer.int_buffer.read_int32()
    array = Array([None] * count)
    for i in range(count):
        array[i] = context.read_value(context)
    return array


def _write_array(context: KV3WriteContext, value: Array, specifier: Specifier):
    if id(value) in context.already_written_objects:  # Empty are all equal to each other, so they are ignored
        raise KV3RecursiveReferenceError("Recursive reference detected while writing array")
    context.already_written_objects.add(id(value))
    context.array_count += 1
    context.active_buffer.int_buffer.write_int32(len(value))
    for item in value:
        context.write_value(context, item)


def _read_object(context: KV3Context, specifier: Specifier):
    member_count = context.object_member_count_buffer.read_uint32()
    obj = Object()
    names = context.strings
    read_name_id = context.active_buffer.int_buffer.read_int32
    read_value = context.read_value
    for i in range(member_count):
        name_id = read_name_id()
        name = names[name_id] if name_id != -1 else str(i)
        obj[name] = read_value(context)
    obj.specifier = specifier
    return obj


def _get_string_id(context: KV3WriteContext, value: str) -> int:
    if value in context.strings:
        return context.strings.index(value)
    else:
        context.strings.append(value)
        return len(context.strings) - 1


def _write_object(context: KV3WriteContext, value: Object, specifier: Specifier):
    if id(value) in context.already_written_objects:
        raise KV3RecursiveReferenceError("Recursive reference detected while writing object")
    context.already_written_objects.add(id(value))
    context.object_member_count_buffer.write_uint32(len(value))
    context.object_count += 1
    for k, v in value.items():
        context.active_buffer.int_buffer.write_int32(_get_string_id(context, k))
        context.write_value(context, v)


def _read_array_typed_helper(context: KV3Context, count, specifier: Specifier):
    buffers = context.active_buffer
    data_type, data_specifier = context.read_type(context)
    if data_type == KV3Type.DOUBLE_ZERO:
        return np.zeros(count, np.float64)
    elif data_type == KV3Type.DOUBLE_ONE:
        return np.ones(count, np.float64)
    elif data_type == KV3Type.INT64_ZERO:
        return np.zeros(count, np.int64)
    elif data_type == KV3Type.INT64_ONE:
        return np.ones(count, np.int64)
    elif data_type == KV3Type.DOUBLE:
        return np.frombuffer(buffers.double_buffer.read(8 * count), np.float64)
    elif data_type == KV3Type.INT64:
        return np.frombuffer(buffers.double_buffer.read(8 * count), np.int64)
    elif data_type == KV3Type.UINT64:
        return np.frombuffer(buffers.double_buffer.read(8 * count), np.uint64)
    elif data_type == KV3Type.INT32:
        return np.frombuffer(buffers.int_buffer.read(4 * count), np.int32)
    elif data_type == KV3Type.UINT32:
        return np.frombuffer(buffers.int_buffer.read(4 * count), np.uint32)
    else:
        reader = _kv3_readers[data_type]
        array = TypedArray(data_type, data_specifier, [reader(context, data_specifier) for _ in range(count)])
        array.specifier = specifier
        return array


def _write_array_typed_helper(context: KV3WriteContext, value: TypedArray, specifier: Specifier):
    if id(value) in context.already_written_objects:
        raise KV3RecursiveReferenceError("Recursive reference detected while writing array")
    context.already_written_objects.add(id(value))
    buffers = context.active_buffer
    if isinstance(value, np.ndarray):
        if value.dtype == np.float64:
            if np.all(value == 0.0):
                context.write_type(context, KV3Type.DOUBLE_ZERO, specifier)
                return
            elif np.all(value == 1.0):
                context.write_type(context, KV3Type.DOUBLE_ONE, specifier)
                return
            context.write_type(context, KV3Type.DOUBLE, specifier)
            buffers.double_buffer.write(value.tobytes())
        elif value.dtype == np.int64:
            if np.all(value == 0):
                context.write_type(context, KV3Type.INT64_ZERO, specifier)
                return
            elif np.all(value == 1):
                context.write_type(context, KV3Type.INT64_ONE, specifier)
                return
            context.write_type(context, KV3Type.INT64, specifier)
            buffers.double_buffer.write(value.tobytes())
        elif value.dtype == np.uint64:
            context.write_type(context, KV3Type.UINT64, specifier)
            buffers.double_buffer.write(value.tobytes())
        elif value.dtype == np.int32:
            context.write_type(context, KV3Type.INT32, specifier)
            buffers.int_buffer.write(value.tobytes())
        elif value.dtype == np.uint32:
            context.write_type(context, KV3Type.UINT32, specifier)
            buffers.int_buffer.write(value.tobytes())
        elif value.dtype == np.int8:
            if context.use_extended_types:
                context.write_type(context, KV3Type.INT8, specifier)
                buffers.byte_buffer.write(value.tobytes())
            else:
                context.write_type(context, KV3Type.INT32, specifier)
                buffers.int_buffer.write(value.astype(np.int32).tobytes())
        elif value.dtype == np.uint8:
            if context.use_extended_types:
                context.write_type(context, KV3Type.UINT8, specifier)
                buffers.byte_buffer.write(value.tobytes())
            else:
                context.write_type(context, KV3Type.UINT32, specifier)
                buffers.int_buffer.write(value.astype(np.uint32).tobytes())
        elif value.dtype == np.int16:
            if buffers.short_buffer:
                context.write_type(context, KV3Type.INT16, specifier)
                buffers.short_buffer.write(value.tobytes())
            else:
                context.write_type(context, KV3Type.INT32, specifier)
                buffers.int_buffer.write(value.astype(np.int32).tobytes())
        elif value.dtype == np.uint16:
            if buffers.short_buffer:
                context.write_type(context, KV3Type.UINT16, specifier)
                buffers.short_buffer.write(value.tobytes())
            else:
                context.write_type(context, KV3Type.UINT32, specifier)
                buffers.int_buffer.write(value.astype(np.uint32).tobytes())
        else:
            raise NotImplementedError(f"Writer for numpy dtype {value.dtype!r} not implemented")
    else:
        if len(value) == 0:
            raise TypeError("TypedArrays of size 0 are not supported")

        first = value[0]
        elem_type = guess_kv_type_legacy(first)
        if elem_type in [KV3Type.BOOLEAN_TRUE, KV3Type.BOOLEAN_FALSE,
                         KV3Type.INT64_ZERO, KV3Type.INT64_ONE,
                         KV3Type.DOUBLE_ZERO, KV3Type.DOUBLE_ONE]:
            all_same = all(guess_kv_type_legacy(v) == elem_type for v in value)
            if not all_same:
                if elem_type in [KV3Type.BOOLEAN_TRUE, KV3Type.BOOLEAN_FALSE]:
                    elem_type = KV3Type.BOOLEAN
                elif elem_type in [KV3Type.INT64_ZERO, KV3Type.INT64_ONE]:
                    elem_type = KV3Type.INT64
                elif elem_type in [KV3Type.DOUBLE_ZERO, KV3Type.DOUBLE_ONE]:
                    elem_type = KV3Type.DOUBLE
        elem_spec = first.specifier if isinstance(first, BaseType) else Specifier.UNSPECIFIED
        context.write_type(context, elem_type, elem_spec)
        _kv3_writer = _kv3_writers[elem_type]
        if _kv3_writer is None:
            raise NotImplementedError(f"Writer for {type!r} not implemented")
        for item in value:
            _kv3_writer(context, item, specifier)


def _read_array_typed(context: KV3Context, specifier: Specifier):
    count = context.active_buffer.int_buffer.read_uint32()
    array = _read_array_typed_helper(context, count, specifier)
    if isinstance(array, BaseType):
        array.specifier = specifier
    return array


def _write_typed_array(context: KV3WriteContext, value: TypedArray, specifier: Specifier):
    context.active_buffer.int_buffer.write_uint32(len(value))
    context.array_count += 1
    _write_array_typed_helper(context, value, specifier)


def _read_array_typed_byte_size(context: KV3Context, specifier: Specifier):
    count = context.active_buffer.byte_buffer.read_uint8()
    array = _read_array_typed_helper(context, count, specifier)
    if isinstance(array, BaseType):
        array.specifier = specifier
    return array


def _write_array_typed_byte_size(context: KV3WriteContext, value: TypedArray, specifier: Specifier):
    context.active_buffer.byte_buffer.write_uint8(len(value))
    context.array_count += 1
    _write_array_typed_helper(context, value, specifier)


def _read_array_typed_byte_size2(context: KV3Context, specifier: Specifier):
    count = context.active_buffer.byte_buffer.read_uint8()
    assert specifier == Specifier.UNSPECIFIED, f"Unsupported specifier {specifier!r}"
    context.active_buffer = context.buffer0
    array = _read_array_typed_helper(context, count, specifier)
    context.active_buffer = context.buffer1
    if isinstance(array, BaseType):
        array.specifier = specifier
    return array


def _write_array_typed_byte_size2(context: KV3WriteContext, value: TypedArray, specifier: Specifier):
    assert specifier == Specifier.UNSPECIFIED, f"Unsupported specifier {specifier!r}"
    context.array_count += 1
    context.active_buffer.byte_buffer.write_uint8(len(value))
    context.active_buffer = context.buffer0
    _write_array_typed_helper(context, value, specifier)
    context.active_buffer = context.buffer1


def _read_int32(context: KV3Context, specifier: Specifier):
    value = Int32(context.active_buffer.int_buffer.read_int32())
    value.specifier = specifier
    return value


def _write_int32(context: KV3WriteContext, value: Int32, specifier: Specifier):
    context.active_buffer.int_buffer.write_int32(value)


def _read_uint32(context: KV3Context, specifier: Specifier):
    value = UInt32(context.active_buffer.int_buffer.read_uint32())
    value.specifier = specifier
    return value


def _write_uint32(context: KV3WriteContext, value: UInt32, specifier: Specifier):
    context.active_buffer.int_buffer.write_uint32(value)


def _read_float(context: KV3Context, specifier: Specifier):
    value = Float(context.active_buffer.int_buffer.read_float())
    value.specifier = specifier
    return value


def _write_float(context: KV3WriteContext, value: Float, specifier: Specifier):
    context.active_buffer.int_buffer.write_float(value)


def _read_int16(context: KV3Context, specifier: Specifier):
    value = Int32(context.active_buffer.short_buffer.read_int16())
    value.specifier = specifier
    return value


def _write_int16(context: KV3WriteContext, value: Int32, specifier: Specifier):
    context.active_buffer.short_buffer.write_int16(value)


def _read_uint16(context: KV3Context, specifier: Specifier):
    value = UInt32(context.active_buffer.short_buffer.read_uint16())
    value.specifier = specifier
    return value


def _write_uint16(context: KV3WriteContext, value: UInt32, specifier: Specifier):
    context.active_buffer.short_buffer.write_uint16(value)


def _read_int8(context: KV3Context, specifier: Specifier):
    value = Int32(context.active_buffer.byte_buffer.read_int8())
    value.specifier = specifier
    return value


def _write_int8(context: KV3WriteContext, value: Int32, specifier: Specifier):
    context.active_buffer.byte_buffer.write_int8(value)


def _read_uint8(context: KV3Context, specifier: Specifier):
    value = UInt32(context.active_buffer.byte_buffer.read_uint8())
    value.specifier = specifier
    return value


def _write_uint8(context: KV3WriteContext, value: UInt32, specifier: Specifier):
    context.active_buffer.byte_buffer.write_uint8(value)


_kv3_readers: list[Callable[['KV3Context', Specifier], Any] | None] = [
    None,
    lambda a, c: None,
    _read_boolean,
    _read_int64,
    _read_uint64,
    _read_double,
    _read_string,
    _read_blob,
    _read_array,
    _read_object,
    _read_array_typed,
    _read_int32,
    _read_uint32,
    lambda a, c: Bool(True),
    lambda a, c: Bool(False),
    lambda a, c: Int64(0),
    lambda a, c: Int64(1),
    lambda a, c: Double(0.0),
    lambda a, c: Double(1.0),
    _read_float,
    _read_int16,
    _read_uint16,
    _read_int8,
    _read_uint8,
    _read_array_typed_byte_size,
    _read_array_typed_byte_size2,
]
_kv3_writers: list[Callable[['KV3WriteContext', AnyKVType, Specifier], None] | None] = [
    None,
    lambda a, c, s: None,
    _write_boolean,
    _write_int64,
    _write_uint64,
    _write_double,
    _write_string,
    _write_blob,
    _write_array,
    _write_object,
    _write_typed_array,
    _write_int32,
    _write_uint32,
    lambda a, c, s: None,
    lambda a, c, s: None,
    lambda a, c, s: None,
    lambda a, c, s: None,
    lambda a, c, s: None,
    lambda a, c, s: None,
    _write_float,
    _write_int16,
    _write_uint16,
    _write_int8,
    _write_uint8,
    _write_array_typed_byte_size,
    _write_array_typed_byte_size2,
]


def _read_value_legacy(context: KV3Context):
    value_type, specifier = context.read_type(context)
    reader = _kv3_readers[value_type]
    if reader is None:
        raise NotImplementedError(f"Reader for {value_type!r} not implemented")

    return reader(context, specifier)


def _write_value_legacy(context: KV3WriteContext, value: AnyKVType):
    specifier = value.specifier if isinstance(value, BaseType) else Specifier.UNSPECIFIED

    value_type = guess_kv_type_legacy(value)

    if not context.use_extended_types and value_type >= KV3Type.INT16:
        if value_type is KV3Type.INT16:
            value_type = KV3Type.INT32
        elif value_type is KV3Type.UINT16:
            value_type = KV3Type.UINT32
        elif value_type is KV3Type.INT8:
            value_type = KV3Type.INT32
        elif value_type is KV3Type.UINT8:
            value_type = KV3Type.UINT32
        elif value_type is KV3Type.ARRAY_TYPED_BYTE_LENGTH:
            value_type = KV3Type.ARRAY_TYPED
        elif value_type is KV3Type.ARRAY_TYPED_BYTE_LENGTH2:
            value_type = KV3Type.ARRAY_TYPED

    if value_type is KV3Type.ARRAY_TYPED and len(value) == 0:  # Special case for empty typed arrays
        value_type = KV3Type.ARRAY

    writer = _kv3_writers[value_type]
    if writer is None:
        raise NotImplementedError(f"Writer for {value_type!r} not implemented")

    context.write_type(context, value_type, specifier)
    writer(context, value, specifier)


def _read_type_legacy(context: KV3Context):
    data_type = context.types_buffer.read_uint8()
    specifier = Specifier.UNSPECIFIED

    if data_type & 0x80:
        data_type &= 0x7F
        flag = context.types_buffer.read_uint8()
        if flag & 1:
            specifier = Specifier.RESOURCE
        elif flag & 2:
            specifier = Specifier.RESOURCE_NAME
        elif flag & 8:
            specifier = Specifier.PANORAMA
        elif flag & 16:
            specifier = Specifier.SOUNDEVENT
        elif flag & 32:
            specifier = Specifier.SUBCLASS
    return KV3Type(data_type), specifier


def _write_type_legacy(context: KV3WriteContext, data_type: KV3Type, specifier: Specifier) -> None:
    """Write legacy KV3 type+specifier to context.types_buffer, mirroring _read_type_legacy."""
    if specifier == Specifier.UNSPECIFIED:
        context.types_buffer.write_uint8(data_type)
        return
    flag_map = {
        Specifier.RESOURCE: 1,
        Specifier.RESOURCE_NAME: 2,
        Specifier.PANORAMA: 8,
        Specifier.SOUNDEVENT: 16,
        Specifier.SUBCLASS: 32,
    }
    if specifier not in flag_map:
        raise ValueError(f"Unsupported legacy specifier: {specifier!r}")
    context.types_buffer.write_uint8(data_type | 0x80)
    context.types_buffer.write_uint8(flag_map[specifier])


def _read_type_v3(context: KV3Context):
    data_type = context.types_buffer.read_uint8()
    specifier = Specifier.UNSPECIFIED

    if data_type & 0x80:
        data_type &= 0x3F
        flag = context.types_buffer.read_uint8()
        if flag & 1:
            specifier = Specifier.RESOURCE
        elif flag & 2:
            specifier = Specifier.RESOURCE_NAME
        elif flag & 8:
            specifier = Specifier.PANORAMA
        elif flag & 16:
            specifier = Specifier.SOUNDEVENT
        elif flag & 32:
            specifier = Specifier.SUBCLASS
    return KV3Type(data_type), specifier


def _write_type_v3(context: KV3WriteContext, data_type: KV3Type, specifier: Specifier) -> None:
    dt = data_type
    if specifier != Specifier.UNSPECIFIED:
        dt |= 0x80
        context.types_buffer.write_uint8(dt)
        flag_map = {
            Specifier.RESOURCE: 1,
            Specifier.RESOURCE_NAME: 2,
            Specifier.PANORAMA: 8,
            Specifier.SOUNDEVENT: 16,
            Specifier.SUBCLASS: 32,
        }
        if specifier not in flag_map:
            raise ValueError(f"Unsupported v3 specifier: {specifier!r}")
        context.types_buffer.write_uint8(flag_map[specifier])
    else:
        context.types_buffer.write_uint8(dt)


def split_buffer(data_buffer: Buffer, bytes_count: int, short_count: int, int_count: int, double_count: int,
                 force_align=False):
    bytes_buffer = MemoryBuffer(data_buffer.read(bytes_count))
    if short_count or force_align:
        data_buffer.align(2)
    shorts_buffer = MemoryBuffer(data_buffer.read(short_count * 2))
    if int_count or force_align:
        data_buffer.align(4)
    ints_buffer = MemoryBuffer(data_buffer.read(int_count * 4))
    if double_count or force_align:
        data_buffer.align(8)
    doubles_buffer = MemoryBuffer(data_buffer.read(double_count * 8))

    return KV3Buffers(bytes_buffer, shorts_buffer, ints_buffer, doubles_buffer)


def read_legacy(buffer: Buffer):
    encoding = buffer.read(16)
    if not KV3Encodings.is_valid(encoding):
        raise BufferError(f'Buffer contains unknown encoding: {encoding!r}')
    encoding = KV3Encodings(encoding)

    fmt = KV3Format(buffer.read(16))

    if encoding == KV3Encodings.binary:
        buffer = MemoryBuffer(buffer.read())
    elif encoding == KV3Encodings.binary_bc:
        buffer = _legacy_block_decompress(buffer)
    elif encoding == KV3Encodings.binary_lz4:
        decompressed_size = buffer.read_uint32()
        buffer = MemoryBuffer(lz4_decompress(buffer.read(-1), decompressed_size))
    else:
        raise ValueError("Unsupported Legacy encoding")

    strings = [buffer.read_ascii_string(encoding="utf8") for _ in range(buffer.read_uint32())]

    buffers = KV3Buffers(buffer, None, buffer, buffer)
    context = KV3Context(
        strings=strings,
        buffer0=buffers,
        buffer1=buffers,
        types_buffer=buffer,
        object_member_count_buffer=buffer,
        binary_blob_sizes=None,
        binary_blob_buffer=None,
        read_type=_read_type_legacy,
        read_value=_read_value_legacy,

        active_buffer=buffers
    )
    root = context.read_value(context)
    meta = {"compression": Int32(-1), "encoding": String(encoding.name), "format": String(fmt.name),
            "signature": String(KV3Signature.VKV_LEGACY.name)}
    root["__METADATA__"] = Object(meta)
    return root


def read_v1(buffer: Buffer):
    fmt = KV3Format(buffer.read(16))

    compression_method = buffer.read_uint32()

    bytes_count = buffer.read_uint32()
    ints_count = buffer.read_uint32()
    doubles_count = buffer.read_uint32()

    uncompressed_size = buffer.read_uint32()

    if compression_method == 0:
        buffer = MemoryBuffer(buffer.read(uncompressed_size))
    elif compression_method == 1:
        u_data = lz4_decompress(buffer.read(-1), uncompressed_size)
        assert len(u_data) == uncompressed_size, "Decompressed data size does not match expected size"
        buffer = MemoryBuffer(u_data)
        del u_data
    else:
        raise NotImplementedError(f"Unknown {compression_method} KV3 compression method")

    kv_buffer = split_buffer(buffer, bytes_count, 0, ints_count, doubles_count, force_align=True)

    strings = [buffer.read_ascii_string(encoding="utf8") for _ in range(kv_buffer.int_buffer.read_uint32())]
    types_size = buffer.remaining() - 4
    types_buffer = buffer.ro_view(size=types_size)
    buffer.skip(types_size)
    types_offset = 0
    assert buffer.read_uint32() == 0xFFEEDD00

    type_array = []
    while types_offset < types_size:
        data_type = types_buffer[types_offset]
        types_offset += 1
        specifier = Specifier.UNSPECIFIED

        if data_type & 0x80:
            data_type &= 0x7F
            flag = types_buffer[types_offset]
            types_offset += 1
            if flag & 1:
                specifier = Specifier.RESOURCE
            elif flag & 2:
                specifier = Specifier.RESOURCE_NAME
            elif flag & 8:
                specifier = Specifier.PANORAMA
            elif flag & 16:
                specifier = Specifier.SOUNDEVENT
            elif flag & 32:
                specifier = Specifier.SUBCLASS
        type_array.append((KV3Type(data_type), specifier))
    type_array.reverse()
    context = KV3Context(
        strings=strings,
        buffer0=kv_buffer,
        buffer1=kv_buffer,
        types_buffer=type_array,
        object_member_count_buffer=kv_buffer.int_buffer,
        binary_blob_sizes=None,
        binary_blob_buffer=None,
        read_type=lambda c: c.types_buffer.pop(),
        read_value=_read_value_legacy,

        active_buffer=kv_buffer,
    )
    root = context.read_value(context)
    meta = {"compression": Int32(compression_method), "format": String(fmt.name),
            "signature": String(KV3Signature.KV3_V1.name)}
    root["__METADATA__"] = Object(meta)
    return root


def read_v2(buffer: Buffer):
    fmt = KV3Format(buffer.read(16))
    compression_method = buffer.read_uint32()
    compression_dict_id = buffer.read_uint16()
    compression_frame_size = buffer.read_uint16()

    bytes_count = buffer.read_uint32()
    ints_count = buffer.read_uint32()
    doubles_count = buffer.read_uint32()

    strings_types_size, object_count, array_count = buffer.read_fmt('I2H')

    uncompressed_size = buffer.read_uint32()
    compressed_size = buffer.read_uint32()
    block_count = buffer.read_uint32()
    block_total_size = buffer.read_uint32()

    if compression_method == 0:
        if compression_dict_id != 0:
            raise NotImplementedError('Unknown compression method in KV3 v2 block')
        if compression_frame_size != 0:
            raise NotImplementedError('Unknown compression method in KV3 v2 block')
        data_buffer = MemoryBuffer(buffer.read(compressed_size))
    elif compression_method == 1:

        if compression_dict_id != 0:
            raise NotImplementedError('Unknown compression method in KV3 v2 block')

        if compression_frame_size != 16384:
            raise NotImplementedError('Unknown compression method in KV3 v2 block')

        data = buffer.read(compressed_size)
        u_data = lz4_decompress(data, uncompressed_size)
        assert len(u_data) == uncompressed_size, "Decompressed data size does not match expected size"
        data_buffer = MemoryBuffer(u_data)
        del u_data, data
    elif compression_method == 2:
        data = buffer.read(compressed_size)
        u_data = zstd_decompress_stream(data, )
        assert len(
            u_data) == uncompressed_size + block_total_size, "Decompressed data size does not match expected size"
        data_buffer = MemoryBuffer(u_data)
        del u_data, data
    else:
        raise NotImplementedError(f"Unknown {compression_method} KV3 compression method")

    bytes_buffer = MemoryBuffer(data_buffer.read(bytes_count))
    if bytes_count:
        data_buffer.align(4)
    ints_buffer = MemoryBuffer(data_buffer.read(ints_count * 4))
    if ints_count:
        data_buffer.align(8)
    doubles_buffer = MemoryBuffer(data_buffer.read(doubles_count * 8))

    strings_and_types_buffer = MemoryBuffer(data_buffer.read(strings_types_size))

    strings = [strings_and_types_buffer.read_ascii_string(encoding="utf8") for _ in range(ints_buffer.read_uint32())]

    if block_count == 0:
        block_sizes = []
        assert data_buffer.read_uint32() == 0xFFEEDD00
        block_buffer = None
    else:
        block_sizes = data_buffer.read_array("I", block_count)
        assert data_buffer.read_uint32() == 0xFFEEDD00
        block_data = b''
        if block_total_size > 0:
            if compression_method == 0:
                for uncompressed_block_size in block_sizes:
                    block_data += data_buffer.read(uncompressed_block_size)
            elif compression_method == 1:
                cd = LZ4ChainDecoder(compression_frame_size, 0)
                for block_size in block_sizes:
                    block_size_tmp = block_size
                    while data_buffer.tell() < data_buffer.size() and block_size_tmp > 0:
                        compressed_block_size = data_buffer.read_uint16()
                        decompressed = cd.decompress(buffer.read(compressed_block_size), compression_frame_size)
                        if len(decompressed) > block_size_tmp:
                            decompressed = decompressed[:block_size_tmp]
                            block_size_tmp = 0
                        elif block_size_tmp < 0:
                            raise ValueError("Failed to decompress blocks!")
                        else:
                            block_size_tmp -= len(decompressed)
                        block_data += decompressed
            elif compression_method == 2:
                block_data += data_buffer.read()
            else:
                raise NotImplementedError(f"Unknown {compression_method} KV3 compression method")
        block_buffer = MemoryBuffer(block_data)

    types_buffer = strings_and_types_buffer.ro_view()
    types_size = len(types_buffer)
    type_array = []
    types_offset = 0
    while types_offset < types_size:
        data_type = types_buffer[types_offset]
        types_offset += 1
        specifier = Specifier.UNSPECIFIED

        if data_type & 0x80:
            data_type &= 0x7F
            flag = types_buffer[types_offset]
            types_offset += 1
            if flag & 1:
                specifier = Specifier.RESOURCE
            elif flag & 2:
                specifier = Specifier.RESOURCE_NAME
            elif flag & 8:
                specifier = Specifier.PANORAMA
            elif flag & 16:
                specifier = Specifier.SOUNDEVENT
            elif flag & 32:
                specifier = Specifier.SUBCLASS
        type_array.append((KV3Type(data_type), specifier))
    type_array.reverse()
    buffers = KV3Buffers(bytes_buffer, None, ints_buffer, doubles_buffer)
    context = KV3Context(
        strings=strings,
        buffer0=buffers,
        buffer1=buffers,
        types_buffer=type_array,
        object_member_count_buffer=ints_buffer,
        binary_blob_sizes=block_sizes,
        binary_blob_buffer=block_buffer,
        read_type=lambda c: c.types_buffer.pop(),
        read_value=_read_value_legacy,
        active_buffer=buffers
    )
    root = context.read_value(context)
    meta = {"compression": Int32(compression_method), "format": String(fmt.name),
            "signature": String(KV3Signature.KV3_V2.name)}
    root["__METADATA__"] = Object(meta)
    return root


def read_v3(buffer: Buffer):
    fmt = KV3Format(buffer.read(16))
    compression_method = buffer.read_uint32()
    compression_dict_id = buffer.read_uint16()
    compression_frame_size = buffer.read_uint16()

    bytes_count = buffer.read_uint32()
    ints_count = buffer.read_uint32()
    doubles_count = buffer.read_uint32()

    strings_types_size, object_count, array_count = buffer.read_fmt('I2H')

    uncompressed_size = buffer.read_uint32()
    compressed_size = buffer.read_uint32()
    block_count = buffer.read_uint32()
    block_total_size = buffer.read_uint32()

    if compression_method == 0:
        if compression_dict_id != 0:
            raise NotImplementedError('Unknown compression method in KV3 v2 block')
        if compression_frame_size != 0:
            raise NotImplementedError('Unknown compression method in KV3 v2 block')
        data_buffer = MemoryBuffer(buffer.read(compressed_size))
    elif compression_method == 1:

        if compression_dict_id != 0:
            raise NotImplementedError('Unknown compression method in KV3 v2 block')

        if compression_frame_size != 16384:
            raise NotImplementedError('Unknown compression method in KV3 v2 block')

        data = buffer.read(compressed_size)
        u_data = lz4_decompress(data, uncompressed_size)
        assert len(u_data) == uncompressed_size, "Decompressed data size does not match expected size"
        data_buffer = MemoryBuffer(u_data)
        del u_data, data
    elif compression_method == 2:
        data = buffer.read(compressed_size)
        u_data = zstd_decompress_stream(data)
        assert len(
            u_data) == uncompressed_size + block_total_size, "Decompressed data size does not match expected size"
        data_buffer = MemoryBuffer(u_data)
        del u_data, data
    else:
        raise NotImplementedError(f"Unknown {compression_method} KV3 compression method")

    kv_buffer = split_buffer(data_buffer, bytes_count, 0, ints_count, doubles_count, True)

    strings_and_types_buffer = MemoryBuffer(data_buffer.read(strings_types_size))

    strings = [strings_and_types_buffer.read_ascii_string(encoding="utf8") for _ in range(kv_buffer.int_buffer.read_uint32())]

    if block_count == 0:
        block_sizes = []
        assert data_buffer.read_uint32() == 0xFFEEDD00
        block_buffer = None

    else:
        block_sizes = list(data_buffer.read_array("I", block_count))
        assert data_buffer.read_uint32() == 0xFFEEDD00
        block_data = b''
        if block_total_size > 0:
            if compression_method == 0:
                for uncompressed_block_size in block_sizes:
                    block_data += buffer.read(uncompressed_block_size)
            elif compression_method == 1:
                compressed_sizes = [data_buffer.read_uint16() for _ in range(data_buffer.remaining() // 2)]
                block_data = decompress_lz4_chain(buffer, block_sizes, compressed_sizes, compression_frame_size)
            elif compression_method == 2:
                block_data += data_buffer.read()
            else:
                raise NotImplementedError(f"Unknown {compression_method} KV3 compression method")
        block_buffer = MemoryBuffer(block_data)

    types_buffer = strings_and_types_buffer.ro_view()
    types_size = len(types_buffer)
    type_array = []
    types_offset = 0
    while types_offset < types_size:
        data_type = types_buffer[types_offset]
        types_offset += 1
        specifier = Specifier.UNSPECIFIED

        if data_type & 0x80:
            data_type &= 0x3F
            flag = types_buffer[types_offset]
            types_offset += 1
            if flag & 1:
                specifier = Specifier.RESOURCE
            elif flag & 2:
                specifier = Specifier.RESOURCE_NAME
            elif flag & 8:
                specifier = Specifier.PANORAMA
            elif flag & 16:
                specifier = Specifier.SOUNDEVENT
            elif flag & 32:
                specifier = Specifier.SUBCLASS
        type_array.append((KV3Type(data_type), specifier))
    type_array.reverse()
    context = KV3Context(
        strings=strings,
        buffer0=kv_buffer,
        buffer1=kv_buffer,
        types_buffer=type_array,
        object_member_count_buffer=kv_buffer.int_buffer,
        binary_blob_sizes=block_sizes,
        binary_blob_buffer=block_buffer,
        read_type=lambda c: c.types_buffer.pop(),
        read_value=_read_value_legacy,
        active_buffer=kv_buffer
    )
    root = context.read_value(context)
    meta = {"compression": Int32(compression_method), "format": String(fmt.name),
            "signature": String(KV3Signature.KV3_V3.name)}
    root["__METADATA__"] = Object(meta)
    return root


def read_v4(buffer: Buffer):
    fmt = KV3Format(buffer.read(16))
    compression_method = buffer.read_uint32()
    compression_dict_id = buffer.read_uint16()
    compression_frame_size = buffer.read_uint16()

    bytes_count = buffer.read_uint32()
    ints_count = buffer.read_uint32()
    doubles_count = buffer.read_uint32()

    strings_types_size, object_count, array_count = buffer.read_fmt('I2H')

    uncompressed_size = buffer.read_uint32()
    compressed_size = buffer.read_uint32()
    block_count = buffer.read_uint32()
    block_total_size = buffer.read_uint32()

    short_count = buffer.read_uint32()
    compressed_block_sizes = buffer.read_uint32() // 2

    if compression_method == 0:
        if compression_dict_id != 0:
            raise NotImplementedError('Unknown compression method in KV3 v2 block')
        if compression_frame_size != 0:
            raise NotImplementedError('Unknown compression method in KV3 v2 block')
        data_buffer = MemoryBuffer(buffer.read(compressed_size))
    elif compression_method == 1:

        if compression_dict_id != 0:
            raise NotImplementedError('Unknown compression method in KV3 v2 block')

        if compression_frame_size != 16384:
            raise NotImplementedError('Unknown compression method in KV3 v2 block')

        data = buffer.read(compressed_size)
        u_data = lz4_decompress(data, uncompressed_size)
        assert len(
            u_data) == uncompressed_size, f"Decompressed data size does not match expected size, got {len(u_data)} expected {uncompressed_size}"
        data_buffer = MemoryBuffer(u_data)
        del u_data, data
    elif compression_method == 2:
        data = buffer.read(compressed_size)
        u_data = zstd_decompress(data, uncompressed_size + block_total_size)
        assert len(
            u_data) == uncompressed_size + block_total_size, f"Decompressed data size does not match expected size, {len(u_data)} != {uncompressed_size + block_total_size}"
        data_buffer = MemoryBuffer(u_data)
        del u_data, data
    else:
        raise NotImplementedError(f"Unknown {compression_method} KV3 compression method")

    kv_buffer = split_buffer(data_buffer, bytes_count, short_count, ints_count, doubles_count, True)

    strings_and_types_buffer = MemoryBuffer(data_buffer.read(strings_types_size))
    strings = [strings_and_types_buffer.read_ascii_string(encoding="utf8") for _ in range(kv_buffer.int_buffer.read_uint32())]

    if block_count == 0:
        block_sizes = []
        assert data_buffer.read_uint32() == 0xFFEEDD00
        block_buffer = None

    else:
        block_sizes = list(data_buffer.read_array("I", block_count))
        assert data_buffer.read_uint32() == 0xFFEEDD00
        block_data = b''
        if block_total_size > 0:
            if compression_method == 0:
                for uncompressed_block_size in block_sizes:
                    block_data += buffer.read(uncompressed_block_size)
            elif compression_method == 1:
                compressed_sizes = [data_buffer.read_uint16() for _ in range(data_buffer.remaining() // 2)]
                block_data = decompress_lz4_chain(buffer, block_sizes, compressed_sizes, compression_frame_size)
            elif compression_method == 2:
                block_data += data_buffer.read()
            else:
                raise NotImplementedError(f"Unknown {compression_method} KV3 compression method")
        block_buffer = MemoryBuffer(block_data)

    types_buffer = strings_and_types_buffer.ro_view()
    types_size = len(types_buffer)
    type_array = []
    types_offset = 0
    while types_offset < types_size:
        data_type = types_buffer[types_offset]
        types_offset += 1
        specifier = Specifier.UNSPECIFIED

        if data_type & 0x80:
            data_type &= 0x3F
            flag = types_buffer[types_offset]
            types_offset += 1
            if flag & 1:
                specifier = Specifier.RESOURCE
            elif flag & 2:
                specifier = Specifier.RESOURCE_NAME
            elif flag & 8:
                specifier = Specifier.PANORAMA
            elif flag & 16:
                specifier = Specifier.SOUNDEVENT
            elif flag & 32:
                specifier = Specifier.SUBCLASS
        type_array.append((KV3Type(data_type), specifier))
    type_array.reverse()

    context = KV3Context(
        strings=strings,
        buffer0=kv_buffer,
        buffer1=kv_buffer,
        types_buffer=type_array,
        object_member_count_buffer=kv_buffer.int_buffer,
        binary_blob_sizes=block_sizes,
        binary_blob_buffer=block_buffer,
        read_type=lambda c: c.types_buffer.pop(),
        read_value=_read_value_legacy,
        active_buffer=kv_buffer
    )
    root = context.read_value(context)
    meta = {"compression": Int32(compression_method), "format": String(fmt.name),
            "signature": String(KV3Signature.KV3_V4.name)}
    root["__METADATA__"] = Object(meta)
    return root


def read_v5(buffer: Buffer):
    fmt = KV3Format(buffer.read(16))
    compression_method = buffer.read_uint32()
    compression_dict_id = buffer.read_uint16()
    compression_frame_size = buffer.read_uint16()

    bytes_count = buffer.read_uint32()
    int_count = buffer.read_uint32()
    double_count = buffer.read_uint32()

    types_size, object_count, array_count = buffer.read_fmt('I2H')

    uncompressed_total_size = buffer.read_uint32()
    compressed_total_size = buffer.read_uint32()
    block_count = buffer.read_uint32()
    block_total_size = buffer.read_uint32()
    short_count = buffer.read_uint32()
    compressed_block_sizes = buffer.read_uint32() // 2
    # assert unk == 0

    buffer0_decompressed_size, block0_compressed_size = buffer.read_fmt("2I")
    buffer1_decompressed_size, block1_compressed_size = buffer.read_fmt("2I")
    bytes_count2, short_count2, int_count2, double_count2 = buffer.read_fmt("4I")
    (field_54, object_count_v5, array_count_v5, field_60) = buffer.read_fmt("4I")

    if compression_method > 0:
        compressed_buffer0 = buffer.read(block0_compressed_size)
        compressed_buffer1 = buffer.read(block1_compressed_size)
    else:
        compressed_buffer0 = buffer.read(buffer0_decompressed_size)
        compressed_buffer1 = buffer.read(buffer1_decompressed_size)

    if compression_method == 0:
        if compression_dict_id != 0:
            raise NotImplementedError('Unknown compression method in KV3 v2 block')
        if compression_frame_size != 0:
            raise NotImplementedError('Unknown compression method in KV3 v2 block')
        buffer0 = MemoryBuffer(compressed_buffer0)
        buffer1 = MemoryBuffer(compressed_buffer1)
    elif compression_method == 1:

        if compression_dict_id != 0:
            raise NotImplementedError('Unknown compression method in KV3 v2 block')

        if compression_frame_size != 16384:
            raise NotImplementedError('Unknown compression method in KV3 v2 block')

        u_data = lz4_decompress_wrp(compressed_buffer0, buffer0_decompressed_size)
        assert len(u_data) == buffer0_decompressed_size, "Decompressed data size does not match expected size"
        buffer0 = MemoryBuffer(u_data)
        u_data = lz4_decompress_wrp(compressed_buffer1, buffer1_decompressed_size)
        assert len(u_data) == buffer1_decompressed_size, "Decompressed data size does not match expected size"
        buffer1 = MemoryBuffer(u_data)
    elif compression_method == 2:
        u_data = zstd_decompress(compressed_buffer0, buffer0_decompressed_size)
        assert len(u_data) == buffer0_decompressed_size, "Decompressed data size does not match expected size"
        buffer0 = MemoryBuffer(u_data)
        u_data = zstd_decompress(compressed_buffer1, buffer1_decompressed_size)
        assert len(u_data) == buffer1_decompressed_size, "Decompressed data size does not match expected size"
        buffer1 = MemoryBuffer(u_data)
    else:
        raise NotImplementedError(f"Unknown {compression_method} KV3 compression method")

    del compressed_buffer0, compressed_buffer1

    kv_buffer0 = split_buffer(buffer0, bytes_count, short_count, int_count, double_count)
    strings = [kv_buffer0.byte_buffer.read_ascii_string(encoding="utf8") for _ in range(kv_buffer0.int_buffer.read_uint32())]
    object_member_count_buffer = MemoryBuffer(buffer1.read(object_count_v5 * 4))
    kv_buffer1 = split_buffer(buffer1, bytes_count2, short_count2, int_count2, double_count2)

    types_buffer = buffer1.ro_view(size=types_size)
    buffer1.skip(types_size)

    if block_count == 0:
        block_sizes = None
        blocks_buffer = None
        assert buffer1.read_uint32() == 0xFFEEDD00
    else:
        block_sizes = list(buffer1.read_array("I", block_count))
        assert buffer1.read_uint32() == 0xFFEEDD00

        block_data = b''
        if block_total_size > 0:
            if compression_method == 0:
                for uncompressed_block_size in block_sizes:
                    block_data += buffer.read(uncompressed_block_size)
            elif compression_method == 1:
                compressed_sizes = [buffer1.read_uint16() for _ in range(compressed_block_sizes)]
                block_data = decompress_lz4_chain(buffer, block_sizes, compressed_sizes, compression_frame_size)
            elif compression_method == 2:
                zstd_compressed_data = buffer.read(
                    compressed_total_size - block0_compressed_size - block1_compressed_size)
                block_data = zstd_decompress(zstd_compressed_data, block_total_size)
            else:
                raise NotImplementedError(f"Unknown {compression_method} KV3 compression method")
            assert buffer.read_uint32() == 0xFFEEDD00
        blocks_buffer = MemoryBuffer(block_data)

    type_array = []
    types_offset = 0
    while types_offset < types_size:
        t = types_buffer[types_offset]
        types_offset += 1
        if t & 0x80:
            specific_type = Specifier(types_buffer[types_offset])
            types_offset += 1
        else:
            specific_type = Specifier.UNSPECIFIED
        if t & 0x40 != 0:
            raise NotImplementedError(f"t & 0x40 != 0: {t & 0x40}")
        type_array.append((KV3Type(t & 0x3F), specific_type))
    type_array.reverse()
    context = KV3Context(
        strings=strings,
        buffer0=kv_buffer0,
        buffer1=kv_buffer1,
        types_buffer=type_array,
        object_member_count_buffer=object_member_count_buffer,
        binary_blob_sizes=block_sizes,
        binary_blob_buffer=blocks_buffer,
        read_type=lambda c: c.types_buffer.pop(),
        read_value=_read_value_legacy,
        active_buffer=kv_buffer1
    )
    root = context.read_value(context)
    if isinstance(root, Object):
        meta = {"compression": Int32(compression_method), "format": String(fmt.name),
                "signature": String(KV3Signature.KV3_V5.name)}
        root["__METADATA__"] = Object(meta)
    return root


def write_legacy(buffer: Buffer, fmt: KV3Format, data: AnyKVType, compression: KV3CompressionMethod):
    if compression is KV3CompressionMethod.UNCOMPRESSED:
        encoding = KV3Encodings.binary
    elif compression is KV3CompressionMethod.LZ4 or compression is KV3CompressionMethod.ZSTD:
        encoding = KV3Encodings.binary_lz4
    else:
        raise NotImplementedError(f"Unsupported compression: {compression}")
    buffer.write(encoding)
    buffer.write(fmt)

    data_buffer = WritableMemoryBuffer()

    buffers = KV3Buffers(data_buffer, None, data_buffer, data_buffer)

    context = KV3WriteContext(
        strings=[],
        buffer0=buffers,
        buffer1=buffers,
        types_buffer=data_buffer,
        object_member_count_buffer=data_buffer,
        binary_blobs=None,
        write_type=_write_type_legacy,
        write_value=_write_value_legacy,
        active_buffer=buffers
    )

    context.write_value(context, data)
    tmp_buffer = WritableMemoryBuffer()
    tmp_buffer.write_uint32(len(context.strings))
    for string in context.strings:
        tmp_buffer.write_ascii_string(string, zero_terminated=True, encoding="utf8")
    tmp_buffer.write(data_buffer.data)
    tmp_buffer.write_int32(-1)
    data_buffer = tmp_buffer
    if encoding == KV3Encodings.binary:
        buffer.write(data_buffer.data)
    elif encoding == KV3Encodings.binary_bc:
        raise NotImplementedError(f"Unsupported encoding: {encoding}")
    elif encoding == KV3Encodings.binary_lz4:
        compressed_data = lz4_compress(data_buffer.data)
        buffer.write_uint32(len(data_buffer.data))
        buffer.write(compressed_data)
    else:
        raise NotImplementedError(f"Unsupported encoding: {encoding}")


def write_v1(buffer: Buffer, data: AnyKVType, compression: KV3CompressionMethod = KV3CompressionMethod.UNCOMPRESSED):
    buffers = KV3Buffers(WritableMemoryBuffer(), None, WritableMemoryBuffer(),
                         WritableMemoryBuffer())
    types_buffer = WritableMemoryBuffer()
    context = KV3WriteContext(
        strings=[],
        buffer0=buffers,
        buffer1=buffers,
        types_buffer=types_buffer,
        object_member_count_buffer=buffers.int_buffer,
        binary_blobs=None,
        write_type=_write_type_legacy,
        write_value=_write_value_legacy,
        active_buffer=buffers
    )

    context.write_value(context, data)

    tmp_buffer = WritableMemoryBuffer()
    tmp_buffer.write(buffers.byte_buffer.data)
    tmp_buffer.align_pad_to(4)
    tmp_buffer.write_uint32(len(context.strings))
    tmp_buffer.write(buffers.int_buffer.data)
    tmp_buffer.align_pad_to(8)
    tmp_buffer.write(buffers.double_buffer.data)
    for string in context.strings:
        tmp_buffer.write_ascii_string(string, zero_terminated=True, encoding="utf8")

    tmp_buffer.write(types_buffer.data)
    tmp_buffer.write_uint32(0xFFEEDD00)

    buffer.write_uint32(compression)
    buffer.write_uint32(context.buffer0.byte_buffer.size())
    buffer.write_uint32(context.buffer0.int_buffer.size() // 4 + 1)  # +1 for string count
    buffer.write_uint32(context.buffer0.double_buffer.size() // 8)
    buffer.write_uint32(tmp_buffer.size())
    if compression is KV3CompressionMethod.UNCOMPRESSED:
        buffer.write(tmp_buffer.data)
    elif compression is KV3CompressionMethod.LZ4:
        compressed_data = lz4_compress(tmp_buffer.data)
        buffer.write(compressed_data)
    else:
        raise NotImplementedError(f"Unknown compression: {compression}")


def write_v2(buffer: Buffer, data: AnyKVType, compression: KV3CompressionMethod = KV3CompressionMethod.UNCOMPRESSED):
    buffers = KV3Buffers(WritableMemoryBuffer(), None, WritableMemoryBuffer(), WritableMemoryBuffer())
    types_buffer = WritableMemoryBuffer()
    context = KV3WriteContext(
        strings=[],
        buffer0=buffers,
        buffer1=buffers,
        types_buffer=types_buffer,
        binary_blobs=[],
        object_member_count_buffer=buffers.int_buffer,
        write_type=_write_type_legacy,
        write_value=_write_value_legacy,
        active_buffer=buffers,
    )

    context.write_value(context, data)

    bytes_count = buffers.byte_buffer.size()
    ints_count = buffers.int_buffer.size() // 4 + 1
    doubles_count = buffers.double_buffer.size() // 8

    strings_types_size = types_buffer.size()
    for s in context.strings:
        strings_types_size += len(s) + 1

    block_sizes = [len(a) for a in context.binary_blobs]
    block_count = len(block_sizes)
    block_total_size = sum(block_sizes) if block_sizes else 0
    block_raw = b"".join(context.binary_blobs)

    to_be_compressed = WritableMemoryBuffer()
    to_be_compressed.write(buffers.byte_buffer.data)
    if buffers.byte_buffer.size():
        to_be_compressed.align_pad_to(4)
    to_be_compressed.write_uint32(len(context.strings))
    to_be_compressed.write(buffers.int_buffer.data)
    if buffers.int_buffer.size():
        to_be_compressed.align_pad_to(8)
    to_be_compressed.write(buffers.double_buffer.data)

    for s in context.strings:
        to_be_compressed.write_ascii_string(s, zero_terminated=True, encoding="utf8")

    to_be_compressed.write(types_buffer.data)

    if block_count == 0:
        to_be_compressed.write_uint32(0xFFEEDD00)
        compressed_blocks_bytes = b""
    else:
        to_be_compressed.write_fmt(f"{block_count}I", *block_sizes)
        to_be_compressed.write_uint32(0xFFEEDD00)
        if compression is KV3CompressionMethod.UNCOMPRESSED:
            compressed_blocks_bytes = b""
            to_be_compressed.write(block_raw)
        elif compression is KV3CompressionMethod.LZ4:
            frame_size = 16384
            chain_encoder = LZ4ChainEncoder(frame_size, 0)
            chunk_sizes_buf = WritableMemoryBuffer()
            comp_blocks_buf = WritableMemoryBuffer()
            off = 0
            for size in block_sizes:
                end = off + size
                pos = off
                while pos < end:
                    take = min(frame_size, end - pos)
                    comp = chain_encoder.compress(block_raw[pos:pos + take])
                    chunk_sizes_buf.write_uint16(len(comp))
                    comp_blocks_buf.write(comp)
                    pos += take
                off = end
            chunk_sizes_table = chunk_sizes_buf.data
            compressed_blocks_bytes = comp_blocks_buf.data
            to_be_compressed.write(chunk_sizes_table)
        elif compression == KV3CompressionMethod.ZSTD:
            compressed_blocks_bytes = b""
            pass
        else:
            raise NotImplementedError(f"Unknown compression: {compression}")

    core_main_payload = to_be_compressed.data
    uncompressed_size = to_be_compressed.size()

    buffer.write_uint32(compression)
    if compression is KV3CompressionMethod.UNCOMPRESSED:
        buffer.write_uint16(0)
        buffer.write_uint16(0)
    elif compression is KV3CompressionMethod.LZ4:
        buffer.write_uint16(0)
        buffer.write_uint16(16384)
    elif compression is KV3CompressionMethod.ZSTD:
        buffer.write_uint16(0)
        buffer.write_uint16(0)
    else:
        raise NotImplementedError(f"Unknown compression: {compression}")

    buffer.write_uint32(bytes_count)
    buffer.write_uint32(ints_count)
    buffer.write_uint32(doubles_count)

    buffer.write_uint32(strings_types_size)
    buffer.write_uint16(context.object_count)
    buffer.write_uint16(context.array_count)

    if compression is KV3CompressionMethod.UNCOMPRESSED:
        compressed_data = core_main_payload
        compressed_size = len(compressed_data)
        buffer.write_uint32(uncompressed_size)
        buffer.write_uint32(compressed_size)
        buffer.write_uint32(block_count)
        buffer.write_uint32(block_total_size)
        buffer.write(compressed_data)
    elif compression is KV3CompressionMethod.LZ4:
        compressed_data = lz4_compress(core_main_payload)
        compressed_size = len(compressed_data)
        buffer.write_uint32(uncompressed_size)
        buffer.write_uint32(compressed_size)
        buffer.write_uint32(block_count)
        buffer.write_uint32(block_total_size)
        buffer.write(compressed_data)
        if block_count > 0:
            buffer.write(compressed_blocks_bytes)
    elif compression is KV3CompressionMethod.ZSTD:
        combined = core_main_payload.tobytes() + (block_raw if block_count > 0 else b"")
        compressed_data = zstd_compress_stream(combined, 9)
        compressed_size = len(compressed_data)
        buffer.write_uint32(uncompressed_size)
        buffer.write_uint32(compressed_size)
        buffer.write_uint32(block_count)
        buffer.write_uint32(block_total_size)
        buffer.write(compressed_data)
    else:
        raise NotImplementedError(f"Unknown compression: {compression}")


def _build_main_payload(buffers, strings, types_buffer, block_sizes, compression, lz4_chunk_sizes_table):
    """Assemble the uncompressed 'main payload' that precedes block bytes."""
    out = WritableMemoryBuffer()
    out.write(buffers.byte_buffer.data)
    out.align_pad_to(4)
    out.write_uint32(len(strings))
    out.write(buffers.int_buffer.data)
    out.align_pad_to(8)
    out.write(buffers.double_buffer.data)
    for s in strings:
        out.write_ascii_string(s, zero_terminated=True, encoding="utf8")
    out.write(types_buffer.data)

    if not block_sizes:
        out.write_uint32(0xFFEEDD00)
    else:
        out.write_fmt(f"{len(block_sizes)}I", *block_sizes)
        out.write_uint32(0xFFEEDD00)
        if compression is KV3CompressionMethod.LZ4 and lz4_chunk_sizes_table:
            out.write(lz4_chunk_sizes_table)

    return out.data, out.size()


def write_v3(buffer: Buffer, data: AnyKVType, compression: KV3CompressionMethod = KV3CompressionMethod.UNCOMPRESSED):
    """Write KV3 v3 with optional compression, minimizing intermediates and clarifying flow."""
    buffers = KV3Buffers(WritableMemoryBuffer(), None, WritableMemoryBuffer(), WritableMemoryBuffer())
    types_buffer = WritableMemoryBuffer()
    context = KV3WriteContext(
        strings=[],
        buffer0=buffers,
        buffer1=buffers,
        types_buffer=types_buffer,
        binary_blobs=[],
        object_member_count_buffer=buffers.int_buffer,
        write_type=_write_type_v3,
        write_value=_write_value_legacy,
        active_buffer=buffers,
    )

    context.write_value(context, data)

    bytes_count = buffers.byte_buffer.size()
    ints_count = buffers.int_buffer.size() // 4 + 1
    doubles_count = buffers.double_buffer.size() // 8

    strings_types_size = types_buffer.size()
    for s in context.strings:
        strings_types_size += len(s) + 1

    block_sizes = [len(b) for b in context.binary_blobs]
    block_count = len(block_sizes)
    block_total_size = sum(block_sizes) if block_sizes else 0
    block_raw = b"".join(context.binary_blobs) if block_sizes else b""

    lz4_chunk_table = b""
    compressed_blocks_bytes = b""
    if compression is KV3CompressionMethod.LZ4 and block_count > 0:
        lz4_chunk_table, compressed_blocks_bytes = _lz4_chain_blocks_compress(block_raw, block_sizes, frame_size=16384)

    main_payload, uncompressed_size = _build_main_payload(
        buffers=buffers,
        strings=context.strings,
        types_buffer=types_buffer,
        block_sizes=block_sizes,
        compression=compression,
        lz4_chunk_sizes_table=lz4_chunk_table,
    )

    buffer.write_uint32(compression)
    if compression is KV3CompressionMethod.UNCOMPRESSED or compression is KV3CompressionMethod.ZSTD:
        buffer.write_uint16(0)
        buffer.write_uint16(0)
    elif compression is KV3CompressionMethod.LZ4:
        buffer.write_uint16(0)
        buffer.write_uint16(16384)
    else:
        raise NotImplementedError(f"Unknown compression: {compression}")

    buffer.write_uint32(bytes_count)
    buffer.write_uint32(ints_count)
    buffer.write_uint32(doubles_count)
    buffer.write_uint32(strings_types_size)
    buffer.write_uint16(context.object_count & 0xFFFF)
    buffer.write_uint16(context.array_count & 0xFFFF)

    if compression is KV3CompressionMethod.UNCOMPRESSED:
        buffer.write_uint32(uncompressed_size)
        buffer.write_uint32(len(main_payload))
        buffer.write_uint32(block_count)
        buffer.write_uint32(block_total_size)
        buffer.write(main_payload)
        if block_count:
            buffer.write(block_raw)
    elif compression is KV3CompressionMethod.LZ4:
        compressed_payload = lz4_compress(main_payload)
        buffer.write_uint32(uncompressed_size)
        buffer.write_uint32(len(compressed_payload))
        buffer.write_uint32(block_count)
        buffer.write_uint32(block_total_size)
        buffer.write(compressed_payload)
        if block_count:
            buffer.write(compressed_blocks_bytes)
    elif compression is KV3CompressionMethod.ZSTD:
        combined = main_payload + (block_raw if block_count else b"")
        compressed_stream = zstd_compress_stream(combined, 9)
        buffer.write_uint32(uncompressed_size)
        buffer.write_uint32(len(compressed_stream))
        buffer.write_uint32(block_count)
        buffer.write_uint32(block_total_size)
        buffer.write(compressed_stream)
    else:
        raise NotImplementedError(f"Unknown compression: {compression}")


def write_v4(buffer: Buffer, data: AnyKVType, compression: KV3CompressionMethod = KV3CompressionMethod.UNCOMPRESSED):
    buffers = KV3Buffers(WritableMemoryBuffer(), WritableMemoryBuffer(), WritableMemoryBuffer(), WritableMemoryBuffer())
    types_buffer = WritableMemoryBuffer()
    context = KV3WriteContext(
        strings=[],
        buffer0=buffers,
        buffer1=buffers,
        types_buffer=types_buffer,
        binary_blobs=[],
        object_member_count_buffer=buffers.int_buffer,
        write_type=_write_type_v3,
        write_value=_write_value_legacy,
        active_buffer=buffers,
    )
    context.write_value(context, data)

    bytes_count = buffers.byte_buffer.size()
    ints_count = buffers.int_buffer.size() // 4 + 1
    shorts_count = buffers.short_buffer.size() // 2
    doubles_count = buffers.double_buffer.size() // 8

    strings_types_size = types_buffer.size()
    for s in context.strings:
        strings_types_size += len(s) + 1

    block_sizes = [len(a) for a in context.binary_blobs]
    block_count = len(block_sizes)
    block_total_size = sum(block_sizes) if block_sizes else 0
    block_raw = b"".join(context.binary_blobs)

    to_be_compressed = WritableMemoryBuffer()
    to_be_compressed.write(buffers.byte_buffer.data)
    to_be_compressed.align_pad_to(2)
    to_be_compressed.write(buffers.short_buffer.data)
    to_be_compressed.align_pad_to(4)
    to_be_compressed.write_uint32(len(context.strings))
    to_be_compressed.write(buffers.int_buffer.data)
    to_be_compressed.align_pad_to(8)
    to_be_compressed.write(buffers.double_buffer.data)

    for s in context.strings:
        to_be_compressed.write_ascii_string(s, zero_terminated=True, encoding="utf8")

    to_be_compressed.write(types_buffer.data)

    if block_count == 0:
        compressed_blocks_stream = b""
        to_be_compressed.write_uint32(0xFFEEDD00)
    else:
        for sz in block_sizes:
            to_be_compressed.write_uint32(sz)
        to_be_compressed.write_uint32(0xFFEEDD00)
        if compression is KV3CompressionMethod.UNCOMPRESSED:
            compressed_blocks_stream = block_raw

        elif compression is KV3CompressionMethod.LZ4:
            frame = 16384
            enc = LZ4ChainEncoder(frame, 0)
            comp_blocks = WritableMemoryBuffer()
            chunk_sizes_u16 = []
            off = 0
            for sz in block_sizes:
                end = off + sz
                pos = off
                while pos < end:
                    take = min(frame, end - pos)
                    comp = enc.compress(block_raw[pos:pos + take])
                    comp_blocks.write(comp)
                    chunk_sizes_u16.append(len(comp))
                    pos += take
                off = end
            for csz in chunk_sizes_u16:
                to_be_compressed.write_uint16(csz)
            compressed_blocks_stream = comp_blocks.data
        elif compression is KV3CompressionMethod.ZSTD:
            compressed_blocks_stream = zstd_compress(block_raw, 9)
        else:
            raise NotImplementedError(f"Unknown compression: {compression}")

    buffer.write_uint32(compression)
    buffer.write_uint16(0)
    buffer.write_uint16(0)
    buffer.write_uint32(bytes_count)
    buffer.write_uint32(ints_count)
    buffer.write_uint32(doubles_count)
    buffer.write_uint32(strings_types_size)
    buffer.write_uint16(context.object_count)
    buffer.write_uint16(context.array_count)

    core_main_payload = to_be_compressed.data
    uncompressed_size = to_be_compressed.size()

    if compression is KV3CompressionMethod.UNCOMPRESSED:
        compressed_size = len(core_main_payload)
        compressed_data = core_main_payload
    elif compression is KV3CompressionMethod.LZ4:
        compressed_data = lz4_compress(core_main_payload)
        compressed_size = len(compressed_data)
    elif compression is KV3CompressionMethod.ZSTD:
        combined = core_main_payload.tobytes() + (block_raw if block_count > 0 else b"")
        compressed_data = zstd_compress_stream(combined, 9)
        compressed_size = len(compressed_data)
    else:
        raise NotImplementedError(f"Unknown compression: {compression}")

    buffer.write_uint32(uncompressed_size)
    buffer.write_uint32(compressed_size)
    buffer.write_uint32(block_count)
    buffer.write_uint32(block_total_size)
    buffer.write_uint32(shorts_count)
    buffer.write_uint32(block_count * 2)
    buffer.write(compressed_data)
    buffer.write(compressed_blocks_stream)
    buffer.write_uint32(0xFFEEDD00)


def _write_type_v5(context: KV3WriteContext, data_type: KV3Type, specifier: Specifier):
    t = data_type
    if specifier != Specifier.UNSPECIFIED:
        t |= 0x80
        context.types_buffer.write_uint8(t)
        context.types_buffer.write_uint8(specifier)
    else:
        context.types_buffer.write_uint8(t)


def write_v5(buffer: Buffer, data: AnyKVType, compression: KV3CompressionMethod = KV3CompressionMethod.UNCOMPRESSED):
    buf0 = KV3Buffers(WritableMemoryBuffer(), WritableMemoryBuffer(), WritableMemoryBuffer(), WritableMemoryBuffer())
    buf1 = KV3Buffers(WritableMemoryBuffer(), WritableMemoryBuffer(), WritableMemoryBuffer(), WritableMemoryBuffer())
    types_buffer = WritableMemoryBuffer()
    obj_counts_buf = WritableMemoryBuffer()

    ctx = KV3WriteContext(
        strings=[],
        buffer0=buf0,
        buffer1=buf1,
        types_buffer=types_buffer,
        binary_blobs=[],
        object_member_count_buffer=obj_counts_buf,
        write_type=_write_type_v5,
        write_value=_write_value_legacy,
        active_buffer=buf1,
    )

    ctx.write_value(ctx, data)

    strings_bytes = WritableMemoryBuffer()
    for s in ctx.strings:
        strings_bytes.write_ascii_string(s, zero_terminated=True, encoding="utf8")

    buf0_bytes = strings_bytes.data.tobytes() + buf0.byte_buffer.data
    buf0_shorts = buf0.short_buffer.data
    buf0_ints = struct.pack("I", len(ctx.strings)) + buf0.int_buffer.data
    buf0_doubles = buf0.double_buffer.data

    bytes_count = len(buf0_bytes)
    short_count = len(buf0_shorts) // 2
    int_count = len(buf0_ints) // 4
    double_count = len(buf0_doubles) // 8

    b0_body = WritableMemoryBuffer()
    b0_body.write(buf0_bytes)
    b0_body.align_pad_to(2)
    b0_body.write(buf0_shorts)
    b0_body.align_pad_to(4)
    b0_body.write(buf0_ints)
    b0_body.align_pad_to(8)
    b0_body.write(buf0_doubles)

    buffer0_raw = b0_body.data
    buffer0_decompressed_size = len(buffer0_raw)

    b1_bytes = buf1.byte_buffer.data
    b1_shorts = buf1.short_buffer.data
    b1_ints = buf1.int_buffer.data
    b1_doubles = buf1.double_buffer.data

    bytes_count2 = len(b1_bytes)
    short_count2 = len(b1_shorts) // 2
    int_count2 = len(b1_ints) // 4
    double_count2 = len(b1_doubles) // 8

    block_sizes = [len(b) for b in ctx.binary_blobs]
    block_count = len(block_sizes)
    block_total_size = sum(block_sizes) if block_sizes else 0
    block_raw = b"".join(ctx.binary_blobs) if block_total_size else b""

    b1_body = WritableMemoryBuffer()
    b1_body.write(obj_counts_buf.data)
    b1_body.write(b1_bytes)
    if b1_shorts:
        b1_body.align_pad_to(2)
    b1_body.write(b1_shorts)
    if b1_ints:
        b1_body.align_pad_to(4)
    b1_body.write(b1_ints)
    if b1_doubles:
        b1_body.align_pad_to(8)
    b1_body.write(b1_doubles)
    b1_body.write(types_buffer.data)

    if block_count == 0:
        compressed_blocks_stream = b""
        b1_body.write_uint32(0xFFEEDD00)
    else:
        for sz in block_sizes:
            b1_body.write_uint32(sz)
        b1_body.write_uint32(0xFFEEDD00)
        if compression is KV3CompressionMethod.UNCOMPRESSED:
            compressed_blocks_stream = b""

        elif compression is KV3CompressionMethod.LZ4:
            frame = 16384
            enc = LZ4ChainEncoder(frame, 0)
            comp_blocks = WritableMemoryBuffer()
            chunk_sizes_u16 = []
            off = 0
            for sz in block_sizes:
                end = off + sz
                pos = off
                while pos < end:
                    take = min(frame, end - pos)
                    comp = enc.compress(block_raw[pos:pos + take])
                    comp_blocks.write(comp)
                    chunk_sizes_u16.append(len(comp))
                    pos += take
                off = end
            for csz in chunk_sizes_u16:
                b1_body.write_uint16(csz)
            compressed_blocks_stream = comp_blocks.data
        elif compression is KV3CompressionMethod.ZSTD:
            compressed_blocks_stream = zstd_compress(block_raw, 9)
        else:
            raise NotImplementedError(f"Unknown compression: {compression}")

    buffer1_raw = b1_body.data
    buffer1_decompressed_size = len(buffer1_raw)

    if compression is KV3CompressionMethod.UNCOMPRESSED:
        buffer0_comp = buffer0_raw
        buffer1_comp = buffer1_raw
        compression_dict_id = 0
        compression_frame_size = 0
        compressed_total_size = len(buffer0_comp) + len(buffer1_comp)
        block0_compressed_size = len(buffer0_comp)
        block1_compressed_size = len(buffer1_comp)
    elif compression is KV3CompressionMethod.LZ4:
        buffer0_comp = lz4_compress(buffer0_raw)
        buffer1_comp = lz4_compress(buffer1_raw)
        compression_dict_id = 0
        compression_frame_size = 16384
        compressed_total_size = len(buffer0_comp) + len(buffer1_comp)
        block0_compressed_size = len(buffer0_comp)
        block1_compressed_size = len(buffer1_comp)
    elif compression is KV3CompressionMethod.ZSTD:
        buffer0_comp = zstd_compress_stream(buffer0_raw, 9)
        buffer1_comp = zstd_compress_stream(buffer1_raw, 9)
        compression_dict_id = 0
        compression_frame_size = 0
        block0_compressed_size = len(buffer0_comp)
        block1_compressed_size = len(buffer1_comp)
        compressed_total_size = block0_compressed_size + block1_compressed_size + (
            len(compressed_blocks_stream) if block_total_size else 0)
    else:
        raise NotImplementedError(f"Unknown compression: {compression}")

    compressed_block_sizes_bytes = block_count * 2
    uncompressed_total_size = buffer0_decompressed_size + buffer1_decompressed_size

    buffer.write_uint32(compression)
    buffer.write_uint16(compression_dict_id)
    buffer.write_uint16(compression_frame_size)

    buffer.write_uint32(bytes_count)
    buffer.write_uint32(int_count)
    buffer.write_uint32(double_count)

    buffer.write_uint32(types_buffer.size())
    buffer.write_uint16(ctx.object_count)
    buffer.write_uint16(ctx.array_count)

    buffer.write_uint32(uncompressed_total_size)
    buffer.write_uint32(compressed_total_size)
    buffer.write_uint32(block_count)
    buffer.write_uint32(block_total_size)
    buffer.write_uint32(short_count)
    buffer.write_uint32(compressed_block_sizes_bytes)

    buffer.write_uint32(buffer0_decompressed_size)
    buffer.write_uint32(block0_compressed_size)
    buffer.write_uint32(buffer1_decompressed_size)
    buffer.write_uint32(block1_compressed_size)

    buffer.write_uint32(bytes_count2)
    buffer.write_uint32(short_count2)
    buffer.write_uint32(int_count2)
    buffer.write_uint32(double_count2)

    buffer.write_uint32(0)
    buffer.write_uint32(ctx.object_count)
    buffer.write_uint32(ctx.array_count)
    buffer.write_uint32(0)

    buffer.write(buffer0_comp)
    buffer.write(buffer1_comp)

    if block_count == 0:
        return
    if compression is KV3CompressionMethod.UNCOMPRESSED:
        buffer.write(block_raw)
    elif compression is KV3CompressionMethod.LZ4:
        buffer.write(compressed_blocks_stream)
    elif compression is KV3CompressionMethod.ZSTD:
        buffer.write(compressed_blocks_stream)
    buffer.write_uint32(0xFFEEDD00)


def zstd_decompress_stream_wrp(data):
    return zstd_decompress_stream(data)


def lz4_decompress_wrp(data, decomp_size):
    return lz4_decompress(data, decomp_size)


def decompress_lz4_chain(buffer: Buffer, decompressed_block_sizes: list[int], compressed_block_sizes: list[int],
                         compression_frame_size: int):
    block_data = b""
    cd = LZ4ChainDecoder(compression_frame_size, 0)
    for block_size in decompressed_block_sizes:
        block_size_tmp = block_size
        while buffer.tell() < buffer.size() and block_size_tmp > 0:
            compressed_size = compressed_block_sizes.pop(0)
            block = buffer.read(compressed_size)
            decompressed = cd.decompress(block, compression_frame_size)
            actual_size = min(compression_frame_size, block_size_tmp)
            block_size_tmp -= actual_size
            block_data += decompressed[:actual_size]
    return block_data
