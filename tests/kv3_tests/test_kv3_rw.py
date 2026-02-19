import pytest
import numpy as np
from SourceIO.library.source2.keyvalues3.binary_keyvalues import read_valve_keyvalue3, write_valve_keyvalue3, \
    KV3RecursiveReferenceError
from SourceIO.library.source2.keyvalues3.enums import KV3Signature, KV3Format, KV3CompressionMethod, Specifier
from SourceIO.library.source2.keyvalues3.types import Object, Array, TypedArray, String, Bool, Int64, UInt64, Double, \
    Float, Int32, UInt32, BinaryBlob, NullObject, guess_kv_type_legacy
from SourceIO.library.utils import WritableMemoryBuffer, MemoryBuffer


# Helper: recursively compare two KV objects by structure and value, not type

def kv_struct_eq(a, b, path=None):
    """
    Recursively compare two KV objects by structure and value, not type.
    If not equal, print a detailed diff and return False.
    Special handling:
      - TypedArray(len==0) and Array(len==0) are considered equal (including numpy empty arrays)
      - numpy arrays are treated as TypedArrays
    """
    if path is None:
        path = []

    def fail(msg):
        print(f"\n[KV3 STRUCT MISMATCH] at {'.'.join(map(str, path)) or '<root>'}: {msg}")
        print(f"  a: {repr(a)}\n  b: {repr(b)}\n")
        return False

    # Handle simple types
    if isinstance(a, (Int64, UInt64, Int32, UInt32, Float, Double, Bool, String, bytes, BinaryBlob)):
        if a != b:
            return fail(f"Value mismatch: {a!r} != {b!r}")
        return True
    if isinstance(a, NullObject) and (b is None or isinstance(b, NullObject)):
        return True
    # Handle numpy arrays as TypedArrays
    if isinstance(a, np.ndarray):
        # Empty numpy array vs empty Array/TypedArray
        if (isinstance(b, (Array, TypedArray, np.ndarray)) and len(a) == 0 and len(b) == 0):
            return True
        if isinstance(b, np.ndarray):
            if not np.array_equal(a, b):
                return fail(f"Numpy array mismatch: {a} != {b}")
            return True
        # Compare to TypedArray or Array
        if isinstance(b, TypedArray) or isinstance(b, Array):
            if len(a) != len(b):
                return fail(f"Array length mismatch: {len(a)} != {len(b)}")
            for i, (x, y) in enumerate(zip(a, b)):
                if not kv_struct_eq(x, y, path + [i]):
                    return False
            return True
        return fail(f"Type mismatch: {type(a)} != {type(b)}")
    if isinstance(b, np.ndarray):
        # Empty numpy array vs empty Array/TypedArray
        if (isinstance(a, (Array, TypedArray)) and len(b) == 0 and len(a) == 0):
            return True
        # Compare to TypedArray or Array
        if isinstance(a, TypedArray) or isinstance(a, Array):
            if len(a) != len(b):
                return fail(f"Array length mismatch: {len(a)} != {len(b)}")
            for i, (x, y) in enumerate(zip(a, b)):
                if not kv_struct_eq(x, y, path + [i]):
                    return False
            return True
        return fail(f"Type mismatch: {type(a)} != {type(b)}")
    # TypedArray/Array/empty handling
    if (isinstance(a, TypedArray) and len(a) == 0 and isinstance(b, Array) and len(b) == 0) or \
            (isinstance(b, TypedArray) and len(b) == 0 and isinstance(a, Array) and len(a) == 0):
        return True
    if isinstance(a, Array) or isinstance(a, list):
        if not isinstance(b, (Array, list, TypedArray, np.ndarray)):
            return fail(f"Type mismatch: {type(a)} != {type(b)}")
        if len(a) != len(b):
            return fail(f"Array length mismatch: {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            if not kv_struct_eq(x, y, path + [i]):
                return False
        return True
    if isinstance(a, TypedArray):
        if not isinstance(b, (TypedArray, Array, np.ndarray)):
            return fail(f"Type mismatch: {type(a)} != {type(b)}")
        if len(a) == 0 and len(b) == 0:
            return True
        if len(a) != len(b):
            return fail(f"TypedArray length mismatch: {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            if not kv_struct_eq(x, y, path + [f"typed[{i}]"]):
                return False
        return True
    if isinstance(a, Object) or isinstance(a, dict):
        if "__METADATA__" in a:
            del a["__METADATA__"]

        if not isinstance(b, (Object, dict)):
            return fail(f"Type mismatch: {type(a)} != {type(b)}")

        if "__METADATA__" in b:
            del b["__METADATA__"]
        if set(a.keys()) != set(b.keys()):
            missing = set(a.keys()) - set(b.keys())
            extra = set(b.keys()) - set(a.keys())
            return fail(f"Object keys mismatch. Missing: {missing}, Extra: {extra}")
        for k in a:
            if not kv_struct_eq(a[k], b[k], path + [k]):
                return False
        return True
    if a != b:
        return fail(f"Value mismatch: {a!r} != {b!r}")
    return True


# Test data for various edge cases
def make_test_data():
    obj = Object({
        "int": Int64(42),
        "uint": UInt64(123456789012345),
        "float": Double(3.14159),
        "bool_true": Bool(True),
        "bool_false": Bool(False),
        "string": String("hello world"),
        "empty_string": String(""),
        "blob": BinaryBlob(b"\x00\x01\x02"),
        "array": Array([Int64(1), Int64(2), Int64(3)]),
        "typed_array": TypedArray(guess_kv_type_legacy(Int64(0)), 8, [Int64(10), Int64(20)]),
        "empty_array": Array([]),
        "empty_typed_array": TypedArray(guess_kv_type_legacy(Int64(0)), 8, []),
        "nested": Object({
            "a": Int64(1),
            "b": Array([String("x"), String("y")]),
            "c": Object({"d": Double(2.71)})
        }),
        "recursive": None,  # Will set below
        "np_array": np.array([1, 2, 3], dtype=np.int32),
        "np_empty": np.array([], dtype=np.float64),
        "null": NullObject(),
    })
    return obj


def roundtrip_kv3(obj, version, fmt=KV3Format.generic, compression=KV3CompressionMethod.UNCOMPRESSED):
    buf = WritableMemoryBuffer()
    write_valve_keyvalue3(buf, obj, fmt, version, compression)
    buf.seek(0)
    out = read_valve_keyvalue3(MemoryBuffer(buf.data))
    return out


@pytest.mark.parametrize("version", [
    KV3Signature.VKV_LEGACY,
    KV3Signature.KV3_V1,
    KV3Signature.KV3_V2,
    KV3Signature.KV3_V3,
    KV3Signature.KV3_V4,
    KV3Signature.KV3_V5,
])
def test_kv3_roundtrip(version):
    data = make_test_data()
    out = roundtrip_kv3(data, version)
    assert kv_struct_eq(data, out)


@pytest.mark.parametrize("version", [
    KV3Signature.VKV_LEGACY,
    KV3Signature.KV3_V1,
    KV3Signature.KV3_V2,
    KV3Signature.KV3_V3,
    KV3Signature.KV3_V4,
    KV3Signature.KV3_V5,
])
def test_kv3_recursive(version):
    data = make_test_data()
    data["recursive"] = data  # Create a recursive reference
    with pytest.raises(KV3RecursiveReferenceError):
        roundtrip_kv3(data, version)


@pytest.mark.parametrize("version", [
    KV3Signature.VKV_LEGACY,
    KV3Signature.KV3_V1,
    KV3Signature.KV3_V2,
    KV3Signature.KV3_V3,
    KV3Signature.KV3_V4,
    KV3Signature.KV3_V5,
])
def test_kv3_empty_typed_array(version):
    obj = Object({"empty_typed": TypedArray(guess_kv_type_legacy(Int64(0)), 8, [])})
    out = roundtrip_kv3(obj, version)
    assert kv_struct_eq(obj, out)


@pytest.mark.parametrize("version", [
    KV3Signature.VKV_LEGACY,
    KV3Signature.KV3_V1,
    KV3Signature.KV3_V2,
    KV3Signature.KV3_V3,
    KV3Signature.KV3_V4,
    KV3Signature.KV3_V5,
])
def test_kv3_nested_mixed(version):
    obj = Object({
        "a": Array([Int64(1), String("x"), Bool(False)]),
        "b": Object({"c": Array([Double(2.2), NullObject()])}),
        "d": TypedArray.from_python([Int64(1), Int64(2)])
    })
    out = roundtrip_kv3(obj, version)
    assert kv_struct_eq(obj, out)


@pytest.mark.parametrize("version", [
    KV3Signature.VKV_LEGACY,
    KV3Signature.KV3_V1,
    KV3Signature.KV3_V2,
    KV3Signature.KV3_V3,
    KV3Signature.KV3_V4,
    KV3Signature.KV3_V5,
])
def test_kv3_large_numbers(version):
    obj = Object({
        "bigint": Int64(2 ** 63 - 1),
        "biguint": UInt64(2 ** 64 - 1),
        "bigfloat": Double(1.79e308),
        "smallfloat": Double(5e-324),
    })
    out = roundtrip_kv3(obj, version)
    assert kv_struct_eq(obj, out)


@pytest.mark.parametrize("version", [
    KV3Signature.VKV_LEGACY,
    KV3Signature.KV3_V1,
    KV3Signature.KV3_V2,
    KV3Signature.KV3_V3,
    KV3Signature.KV3_V4,
    KV3Signature.KV3_V5,
])
def test_kv3_empty_object(version):
    obj = Object({})
    out = roundtrip_kv3(obj, version)
    assert kv_struct_eq(obj, out)


@pytest.mark.parametrize("version", [
    KV3Signature.VKV_LEGACY,
    KV3Signature.KV3_V1,
    KV3Signature.KV3_V2,
    KV3Signature.KV3_V3,
    KV3Signature.KV3_V4,
    KV3Signature.KV3_V5,
])
def test_kv3_empty_array(version):
    obj = Object({"arr": Array([])})
    out = roundtrip_kv3(obj, version)
    assert kv_struct_eq(obj, out)


@pytest.mark.parametrize("version", [
    KV3Signature.VKV_LEGACY,
    KV3Signature.KV3_V1,
    KV3Signature.KV3_V2,
    KV3Signature.KV3_V3,
    KV3Signature.KV3_V4,
    KV3Signature.KV3_V5,
])
def test_kv3_special_floats(version):
    obj = Object({"nan": Double(np.nan), "inf": Double(np.inf), "ninf": Double(-np.inf)})
    out = roundtrip_kv3(obj, version)
    # NaN != NaN, so check with np.isnan
    assert np.isnan(out["nan"]) and np.isinf(out["inf"]) and np.isinf(out["ninf"]) and out["ninf"] < 0


@pytest.mark.parametrize("version",
                         [KV3Signature.VKV_LEGACY,
                          KV3Signature.KV3_V1,
                          KV3Signature.KV3_V2,
                          KV3Signature.KV3_V3,
                          KV3Signature.KV3_V4,
                          KV3Signature.KV3_V5],
                         )
@pytest.mark.parametrize("compression",
                         [KV3CompressionMethod.UNCOMPRESSED,
                          KV3CompressionMethod.LZ4,
                          KV3CompressionMethod.ZSTD]
                         )
def test_kv3_blob(version, compression):
    obj = Object.from_python({"blob": b"\x00\x01\x02\x03\x04", "huge_blob": b"Hello World!" * 1000})
    out = roundtrip_kv3(obj, version)
    assert kv_struct_eq(obj, out)
