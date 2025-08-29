import abc
from functools import partial
from types import NoneType
from typing import Collection, Optional, TypeVar, Any

import numpy as np

from .enums import KV3Type, Specifier


class BaseType(abc.ABC):
    def __init_subclass__(cls, **kwargs):
        cls.specifier: Specifier = Specifier.UNSPECIFIED

    @abc.abstractmethod
    def to_dict(self):
        return NotImplemented

    @classmethod
    @abc.abstractmethod
    def from_python(cls, value: Any):
        pass


class NullObject(BaseType):
    def __bool__(self):
        return False

    def to_dict(self):
        return None

    @classmethod
    def from_python(cls, value: Any):
        return NullObject()


class String(BaseType, str):
    def to_dict(self):
        return str(self)

    @classmethod
    def from_python(cls, value: Any):
        return String(value)


class _BaseInt(BaseType, int):
    def to_dict(self):
        return int(self)

    @classmethod
    def from_python(cls, value: Any):
        return cls(value)


class _BaseFloat(BaseType, float):
    def to_dict(self):
        return float(self)

    @classmethod
    def from_python(cls, value: Any):
        return cls(value)


class Bool(_BaseInt):
    def to_dict(self):
        return bool(self)


class Int8(_BaseInt):
    pass


class UInt8(_BaseInt):
    pass


class Int32(_BaseInt):
    pass


class UInt32(_BaseInt):
    pass


class Int64(_BaseInt):
    pass


class UInt64(_BaseInt):
    pass


class Double(_BaseFloat):
    pass


class Float(_BaseFloat):
    pass


class BinaryBlob(BaseType, bytes):

    def to_dict(self):
        return bytes(self)

    @classmethod
    def from_python(cls, value: Any):
        return BinaryBlob(value)


T = TypeVar('T', BaseType, str, NoneType)

DEBUGGING = False


class Object(BaseType, dict[str, T]):
    if DEBUGGING:
        def __setitem__(self, key, value: T):
            if isinstance(value, np.ndarray):
                assert value.dtype in (np.float32, np.float64,
                                       np.int8, np.uint8,
                                       np.int16, np.uint16,
                                       np.int32, np.uint32,
                                       np.int64, np.uint64)
            elif not isinstance(value, (BaseType, str, NoneType)):
                raise TypeError(f'Only KV3 types are allowed, got {type(value)}')
            super(Object, self).__setitem__(key, value)
    else:
        def __setitem__(self, key, value: T):
            super(Object, self).__setitem__(key, value)

    def __contains__(self, item):
        if isinstance(item, tuple):
            for key in item:
                if dict.__contains__(self, key):
                    return True
            return False
        else:
            return dict.__contains__(self, item)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            for key in item:
                if dict.__contains__(self, key):
                    return dict.__getitem__(self, key)
            raise KeyError(item)
        else:
            return dict.__getitem__(self, item)

    def get(self, key: str | tuple[str, ...], default=None):
        if key in self:
            return self[key]
        return default

    def to_dict(self):
        if any(isinstance(i, np.ndarray) for i in self.values()):
            res = {}
            for k, v in self.items():
                if v is not None:
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    else:
                        v = v.to_dict()
                res[k] = v
            return res
        return {k: v.to_dict() for (k, v) in self.items()}

    @classmethod
    def from_python(cls, value: dict[str, Any]):
        obj = Object({})
        for k, v in value.items():
            obj[k] = _from_python_helper(v)
        return obj


class Array(BaseType, list[T]):
    def __init__(self, initial: Optional[list[T]] = None):
        super(Array, self).__init__(initial)

    def append(self, value: T):
        assert isinstance(value, BaseType)
        super(Array, self).append(value)

    def extend(self, values: Collection[T]):
        assert all(map(partial(isinstance, __class_or_tuple=BaseType), values))
        super(Array, self).extend(values)

    def to_dict(self):
        if any(isinstance(i, np.ndarray) for i in self):
            res = []
            for i in self:
                if isinstance(i, np.ndarray):
                    i = i.tolist()
                else:
                    i = i.to_dict()
                res.append(i)
            return res
        return [(i.to_dict() if i is not None else None) for i in self]

    @classmethod
    def from_python(cls, value: Collection[Any]):
        if not isinstance(value, list) and not isinstance(value, np.ndarray):
            raise TypeError(f'Expected list or numpy.ndarray, got {type(value)}')
        return Array(list(map(_from_python_helper, value)))


class TypedArray(BaseType, list[T]):
    def __init__(self, data_type: KV3Type, data_specifier: Specifier, initial: Optional[list[T]] = None):
        super(TypedArray, self).__init__(initial)
        self.data_type = data_type
        self.data_specifier = data_specifier

    def append(self, value: T):
        assert isinstance(value, BaseType)
        super(TypedArray, self).append(value)

    def extend(self, values: Collection[T]):
        assert all(map(partial(isinstance, __class_or_tuple=BaseType), values))
        super(TypedArray, self).extend(values)

    def to_dict(self):
        if any(isinstance(i, np.ndarray) for i in self):
            res = []
            for i in self:
                if isinstance(i, np.ndarray):
                    i = i.tolist()
                else:
                    i = i.to_dict()
                res.append(i)
            return res
        return [i.to_dict() for i in self]

    @classmethod
    def from_python(cls, value: Collection[T]):
        if len(value) == 0:
            raise TypeError('Cannot infer type from an empty collection')
        if not isinstance(value, np.ndarray) and len(set([type(v) for v in value])) > 1:
            raise TypeError(f'Expected numpy.ndarray, got {type(value)}')
        values = [_from_python_helper(v) for v in value]  # Ensure all values are BaseType instances

        return TypedArray(guess_kv_type_legacy(values[0]), Specifier.UNSPECIFIED, values)


AnyKVType = Object | NullObject | String | Bool | Int64 | Int32 | UInt64 | UInt32 | Double | Float | BinaryBlob | Array | TypedArray
AnyKVConvertable = dict | NoneType | str | bool | int | float | np.ndarray | list


def guess_kv_type_legacy(value: AnyKVConvertable) -> KV3Type:
    if value is None or isinstance(value, NullObject):
        return KV3Type.NULL
    elif isinstance(value, Bool):
        if value:
            return KV3Type.BOOLEAN_TRUE
        elif not value:
            return KV3Type.BOOLEAN_FALSE
        else:
            raise ValueError("Bool value must be True or False")
        return KV3Type.BOOLEAN
    elif isinstance(value, Int64):
        if value == 0:
            return KV3Type.INT64_ZERO
        elif value == 1:
            return KV3Type.INT64_ONE
        return KV3Type.INT64
    elif isinstance(value, UInt64):
        return KV3Type.UINT64
    elif isinstance(value, Double):
        if value == 0.0:
            return KV3Type.DOUBLE_ZERO
        elif value == 1.0:
            return KV3Type.DOUBLE_ONE
        return KV3Type.DOUBLE
    elif isinstance(value, Float):
        return KV3Type.FLOAT
    elif isinstance(value, String):
        return KV3Type.STRING
    elif isinstance(value, BinaryBlob):
        return KV3Type.BINARY_BLOB
    elif isinstance(value, Array):
        return KV3Type.ARRAY
    elif isinstance(value, Object):
        return KV3Type.OBJECT
    elif isinstance(value, (TypedArray, np.ndarray)):
        return KV3Type.ARRAY_TYPED
    elif isinstance(value, Int8):
        return KV3Type.INT32
    elif isinstance(value, UInt8):
        return KV3Type.UINT32
    elif isinstance(value, Int32):
        return KV3Type.INT32
    elif isinstance(value, UInt32):
        return KV3Type.UINT32
    else:
        raise TypeError(f"Unsupported value type: {type(value)}")


def _from_python_helper(value: AnyKVConvertable | AnyKVType) -> AnyKVType:
    if isinstance(value, BaseType):
        return value

    if isinstance(value, np.ndarray):
        return TypedArray.from_python(value)
    elif isinstance(value, (list, tuple)):
        if len(set([type(v) for v in value])) <= 1 and len(value) > 0:
            return TypedArray.from_python(value)
        return Array.from_python(value)
    elif isinstance(value, dict):
        return Object.from_python(value)
    elif value is None:
        return NullObject.from_python(value)
    elif isinstance(value, str):
        return String.from_python(value)
    elif isinstance(value, bool):
        return Bool.from_python(value)
    elif isinstance(value, bytes):
        return BinaryBlob.from_python(value)
    elif isinstance(value, int):
        if value >= 0:
            if value <= 0xFFFFFFFF:
                return UInt32.from_python(value)
            else:
                return UInt64.from_python(value)
        return Int64.from_python(value)
    elif isinstance(value, float):
        return Double.from_python(value)
    else:
        raise TypeError(f'Unsupported value type: {type(value)}')


__all__ = ['BaseType', 'Object', 'NullObject', 'String', 'Bool',
           'Int8', 'UInt8',
           'Int64', 'UInt32', 'UInt64', 'Int32', 'Double', 'Float',
           'BinaryBlob', 'Array', 'TypedArray', 'AnyKVType', 'guess_kv_type_legacy']
