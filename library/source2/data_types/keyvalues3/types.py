import abc
from functools import partial
from types import NoneType
from typing import Collection, Optional, TypeVar

import numpy as np

from .enums import KV3Type, KV3TypeFlag


class BaseType(abc.ABC):
    def __init_subclass__(cls, **kwargs):
        cls.flag = KV3TypeFlag.NONE

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


class BinaryBlob(BaseType, bytes):

    def to_dict(self):
        return bytes(self)


T = TypeVar('T', BaseType, str)


class Object(BaseType, dict):

    def __setitem__(self, key, value: T):
        if not isinstance(value, (BaseType, str, NoneType, np.ndarray)):
            raise TypeError(f'Only KV3 types are allowed, got {type(value)}')
        if isinstance(value, np.ndarray):
            assert value.dtype in (np.float32, np.float64,
                                   np.int8, np.uint8,
                                   np.int16, np.uint16,
                                   np.int32, np.uint32,
                                   np.int64, np.uint64)
        super(Object, self).__setitem__(key, value)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            for key in item:
                value = self.get(key, None)
                if value is not None:
                    return value
            raise KeyError(item)
        else:
            return dict.__getitem__(self, item)

    def to_dict(self):
        if any(isinstance(i, np.ndarray) for i in self.values()):
            res = {}
            for k, v in self.items():
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
        return [i.to_dict() for i in self]


class TypedArray(BaseType, list[T]):
    def __init__(self, data_type: KV3Type, data_flag: KV3TypeFlag, initial: Optional[list[T]] = None):
        super(TypedArray, self).__init__(initial)
        self.data_type = data_type
        self.data_flag = data_flag

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


__all__ = ['BaseType', 'Object', 'NullObject', 'String', 'Bool',
           'Int64', 'UInt32', 'UInt64', 'Int32', 'Double',
           'BinaryBlob', 'Array', 'TypedArray']
