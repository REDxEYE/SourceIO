import abc
from functools import partial
from types import NoneType
from typing import Collection, Optional, TypeVar

import numpy as np

from .enums import KV3Type, Specifier


class BaseType(abc.ABC):
    def __init_subclass__(cls, **kwargs):
        cls.specifier: Specifier = Specifier.UNSPECIFIED

    def to_dict(self):
        return NotImplemented


class NullObject(BaseType):
    def __bool__(self):
        return False

    def to_dict(self):
        return None


class String(BaseType, str):
    def to_dict(self):
        return str(self)


class _BaseInt(BaseType, int):
    def to_dict(self):
        return int(self)


class _BaseFloat(BaseType, float):
    def to_dict(self):
        return float(self)


class Bool(_BaseInt):
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


T = TypeVar('T', BaseType, str, NoneType)

DEBUGGING = False


class Object(BaseType, dict):
    def __setitem__(self, key, value: T):
        if DEBUGGING:
            if isinstance(value, np.ndarray):
                assert value.dtype in (np.float32, np.float64,
                                       np.int8, np.uint8,
                                       np.int16, np.uint16,
                                       np.int32, np.uint32,
                                       np.int64, np.uint64)
            elif not isinstance(value, (BaseType, str, NoneType)):
                raise TypeError(f'Only KV3 types are allowed, got {type(value)}')
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


AnyKVType = Object | NullObject | String | Bool | Int64 | Int32 | UInt64 | UInt32 | Double | Float | BinaryBlob | Array | TypedArray

__all__ = ['BaseType', 'Object', 'NullObject', 'String', 'Bool',
           'Int64', 'UInt32', 'UInt64', 'Int32', 'Double', 'Float',
           'BinaryBlob', 'Array', 'TypedArray', 'AnyKVType']
