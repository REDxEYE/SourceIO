import re
from abc import ABC
from functools import cached_property
from io import BytesIO, StringIO
from pathlib import Path
from struct import unpack, pack
from typing import Union, BinaryIO, Optional, Any, TextIO, Sequence
from uuid import UUID, uuid3, uuid4
import numpy as np
from collections import OrderedDict

header_format = "<!-- dmx encoding {:s} {:d} format {:s} {:d} -->"
header_format_regex = header_format.replace("{:d}", "([0-9]+)").replace("{:s}", r"(\S+)")

header_proto2 = "<!-- DMXVersion binary_v{:d} -->"
header_proto2_regex = header_proto2.replace("{:d}", "([0-9]+)")


class IDCollisionError(Exception):
    pass


def read_byte(file: BinaryIO):
    return unpack("B", file.read(1))[0]


def read_int(file: BinaryIO):
    return unpack("i", file.read(4))[0]


def read_quad(file: BinaryIO):
    return unpack("Q", file.read(8))[0]


def read_short(file: BinaryIO):
    return unpack("H", file.read(2))[0]


def read_float(file: BinaryIO):
    return unpack("f", file.read(4))[0]


def write_byte(file: BinaryIO, value: int):
    file.write(pack('B', value))


def write_int(file: BinaryIO, value: int):
    file.write(pack('i', value))


def write_quad(file: BinaryIO, value: int):
    file.write(pack('Q', value))


def write_short(file: BinaryIO, value: int):
    file.write(pack('H', value))


def write_float(file: BinaryIO, value: float):
    file.write(pack('f', value))


def read_string(file: BinaryIO):
    buffer = b''
    while c := file.read(1):
        if not c or c == b'\x00':
            break
        buffer += c
    return buffer.decode('utf8')


def write_string(file: BinaryIO, string: str):
    file.write(string.encode('utf8') + b'\x00')


class BaseTypeMixin:
    value: Any

    @classmethod
    def from_file(cls, file: BinaryIO) -> 'BaseTypeMixin':
        pass

    def to_file(self, file: BinaryIO):
        pass


class Binary(BaseTypeMixin, bytes):
    @classmethod
    def from_file(cls, file: BinaryIO):
        return cls(file.read(read_int(file)))

    def to_file(self, file: BinaryIO):
        write_int(file, len(self))
        file.write(self)


class Time(BaseTypeMixin, float):
    @classmethod
    def from_file(cls, file: BinaryIO):
        return cls(read_int(file) / 10_000)

    def to_file(self, file: BinaryIO):
        write_int(file, round(self * 10_000))


class AbstractArray(BaseTypeMixin, Sequence, ABC):
    data_type = None


class _NumpyArray(np.ndarray, AbstractArray, ABC):
    def __new__(cls, input_array, dtype=None):
        return np.asarray(input_array, dtype=dtype).view(cls)

    @classmethod
    def from_list(cls, initial):
        return cls(initial)

    def to_file(self, file: BinaryIO):
        write_int(file, len(self))
        file.write(self.tobytes())

    @classmethod
    def from_file(cls, file: BinaryIO):
        raise NotImplementedError()


class Vector(np.ndarray, BaseTypeMixin, Sequence, ABC):

    def __new__(cls, input_array, dtype: Optional[np.dtype] = None):
        return np.asarray(input_array, dtype=dtype).view(cls)

    def __len__(self) -> int:
        return self.shape[0]

    def __contains__(self, __x: Any) -> bool:
        return np.in1d(self, __x).any()

    @classmethod
    def from_file(cls, file: BinaryIO) -> 'Vector':
        raise NotImplementedError()

    def to_file(self, file: BinaryIO):
        file.write(self.tobytes())


class Vector2(Vector):
    def __new__(cls, input_array):
        assert len(input_array) == 2
        return super().__new__(cls, input_array, np.float32)

    @classmethod
    def from_file(cls, file: BinaryIO) -> 'Vector2':
        return np.frombuffer(file.read(8), np.float32).view(cls)


class Vector3(Vector):
    def __new__(cls, input_array):
        assert len(input_array) == 3
        return super().__new__(cls, input_array, np.float32)

    @classmethod
    def from_file(cls, file: BinaryIO) -> 'Vector3':
        return np.frombuffer(file.read(12), np.float32).view(cls)


class Vector4(Vector):
    def __new__(cls, input_array):
        assert len(input_array) == 4
        return super().__new__(cls, input_array, np.float32)

    @classmethod
    def from_file(cls, file: BinaryIO) -> 'Vector4':
        return np.frombuffer(file.read(16), np.float32).view(cls)


class Angle(Vector):
    def __new__(cls, input_array):
        assert len(input_array) == 3
        return super().__new__(cls, input_array, np.float32)

    @classmethod
    def from_file(cls, file: BinaryIO) -> 'Angle':
        return np.frombuffer(file.read(12), np.float32).view(cls)


class Quaternion(Vector):
    def __new__(cls, input_array):
        assert len(input_array) == 4
        return super().__new__(cls, input_array, np.float32)

    @classmethod
    def from_file(cls, file: BinaryIO) -> 'Quaternion':
        return np.frombuffer(file.read(16), np.float32).view(cls)


class Color(Vector):
    def __new__(cls, input_array):
        assert len(input_array) == 4
        return super().__new__(cls, input_array, np.uint8)

    @classmethod
    def from_file(cls, file: BinaryIO) -> 'Color':
        return np.frombuffer(file.read(4), np.uint8).view(cls)


class Matrix(Vector):
    def __new__(cls, input_array):
        assert len(input_array) == 16 or (isinstance(input_array, np.ndarray) and input_array.shape == (4, 4))
        return super().__new__(cls, input_array, np.float32).reshape(4, 4).view(cls)

    @classmethod
    def from_file(cls, file: BinaryIO) -> 'Matrix':
        return np.frombuffer(file.read(64), np.float32).reshape(4, 4).view(cls)


class Element(OrderedDict):
    def __init__(self, datamodel: 'DataModel', name, uid: UUID, dm_type='DmElement', placeholder=False):
        super().__init__()
        self.__dm = datamodel
        self.name = name
        self.dm_type = dm_type
        self.placeholder = placeholder
        self.id = uid

    def __eq__(self, other):
        return isinstance(other, Element) and self.id.bytes_le == other.id.bytes_le

    def __bool__(self):
        return True

    def __repr__(self):
        return "<Datamodel element \"{}\" ({})>".format(self.name, self.dm_type)

    def __hash__(self):
        return hash(self.id)

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise TypeError("Attribute name must be a string, not {}".format(type(item)))
        try:
            return super().__getitem__(item)
        except KeyError as e:
            raise AttributeError("No attribute \"{}\" on {}".format(item, self)) from e

    def add_attribute(self, name, value):
        return super().__setitem__(name, value)

    def __setitem__(self, key, item):
        key = str(key)
        if key in ["name", "id"]:
            raise KeyError("\"{}\" is a reserved name".format(key))

        if isinstance(item, np.ndarray):
            if len(item.shape) > 2:
                raise AttributeError("3+ dimmentional arrays are not supported")
            if len(item.shape) == 2 and item.shape[1] > 4:
                raise AttributeError("Only Vector2/3/4 are supported")

            return super().__setitem__(key, item)

        if isinstance(item, (Element, ElementArray)):
            visited = set()

            def import_element(elem):
                if elem.id in visited:
                    return
                visited.add(elem.id)
                if elem.id not in self.__dm.used_ids:
                    self.__dm.put_element(elem)
                for attr in elem.values():
                    if isinstance(attr, Element):
                        import_element(attr)
                    elif isinstance(attr, ElementArray):
                        for attr_elem in attr:
                            if isinstance(attr_elem, Element):
                                import_element(attr_elem)

            if isinstance(item, Element):
                import_element(item)
            else:
                for arr_elem in item:
                    if isinstance(arr_elem, Element):
                        import_element(arr_elem)
        assert type(item) in _all_known_dmx_types or isinstance(item, NoneElement), (
            f'Unsupported attribute type: {type(item)}')
        return super().__setitem__(key, item)


class NoneElement:
    pass


class IntArray(_NumpyArray):

    def __new__(cls, input_array):
        return super().__new__(cls, input_array, np.int32)

    @classmethod
    def from_file(cls, file: BinaryIO):
        data = file.read(4 * read_int(file))
        return np.frombuffer(data, np.int32).view(cls)


class UInt64Array(_NumpyArray):
    def __new__(cls, input_array):
        return super().__new__(cls, input_array, np.uint64)

    @classmethod
    def from_file(cls, file: BinaryIO):
        data = file.read(8 * read_int(file))
        return np.frombuffer(data, np.uint64).view(cls)


class UInt8Array(_NumpyArray):
    def __new__(cls, input_array):
        return super().__new__(cls, input_array, np.uint8)

    @classmethod
    def from_file(cls, file: BinaryIO):
        data = file.read(read_int(file))
        return np.frombuffer(data, np.uint8).view(cls)


class FloatArray(_NumpyArray):
    def __new__(cls, input_array):
        return super().__new__(cls, input_array, np.float32)

    @classmethod
    def from_file(cls, file: BinaryIO):
        data = file.read(4 * read_int(file))
        return np.frombuffer(data, np.float32).view(cls)


class BoolArray(_NumpyArray):
    def __new__(cls, input_array):
        return super().__new__(cls, input_array, np.bool_)

    @classmethod
    def from_file(cls, file: BinaryIO):
        data = file.read(read_int(file))
        return np.frombuffer(data, np.bool_).view(cls)


class NumpyVectorArray(_NumpyArray, ABC):
    pass


class Vector2Array(NumpyVectorArray):
    def __new__(cls, input_array):
        return super().__new__(cls, input_array, np.float32).reshape(-1, 2)

    @classmethod
    def from_file(cls, file: BinaryIO):
        data = file.read(8 * read_int(file))
        return np.frombuffer(data, "2f4").view(cls)


class ColorArray(NumpyVectorArray):
    def __new__(cls, input_array):
        return super().__new__(cls, input_array, np.uint8).reshape(-1, 4)

    @classmethod
    def from_file(cls, file: BinaryIO):
        data = file.read(4 * read_int(file))
        return np.frombuffer(data, "4u1").view(cls)


class Vector3Array(NumpyVectorArray):
    def __new__(cls, input_array):
        return super().__new__(cls, input_array, np.float32).reshape(-1, 3)

    @classmethod
    def from_file(cls, file: BinaryIO):
        data = file.read(12 * read_int(file))
        return np.frombuffer(data, "3f4").view(cls)


class Vector4Array(NumpyVectorArray):
    def __new__(cls, input_array):
        return super().__new__(cls, input_array, np.float32).reshape(-1, 4)

    @classmethod
    def from_file(cls, file: BinaryIO):
        data = file.read(16 * read_int(file))
        return np.frombuffer(data, "4f4").view(cls)


class AngleArray(Vector3Array):
    @classmethod
    def from_file(cls, file: BinaryIO):
        data = file.read(12 * read_int(file))
        return np.frombuffer(data, "3f4").view(cls)


class QuaternionArray(Vector4Array):
    @classmethod
    def from_file(cls, file: BinaryIO):
        data = file.read(16 * read_int(file))
        return np.frombuffer(data, "4f4").view(cls)


class MatrixArray(NumpyVectorArray):
    def __new__(cls, input_array):
        return super().__new__(cls, input_array, np.float32).reshape(-1, 4, 4)

    @classmethod
    def from_file(cls, file: BinaryIO):
        data = file.read(64 * read_int(file))
        return np.frombuffer(data, "16f4").reshape(-1, 4, 4).view(cls)


class ObjectArray(list, AbstractArray, ABC):
    data_type: BaseTypeMixin = None

    @classmethod
    def from_list(cls, initial):
        return cls(initial)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_file(cls, file: BinaryIO):
        self = cls()
        for _ in range(read_int(file)):
            self.append(self.data_type.from_file(file))

    def to_file(self, file: BinaryIO):
        write_int(file, len(self))
        for item in self:
            item.to_file(file)


class ElementArray(ObjectArray[Element]):
    pass


class StringArray(ObjectArray[str]):
    pass


class BinaryArray(ObjectArray[Binary]):
    pass


class TimeArray(ObjectArray[Time]):
    pass


_valid_dmx_type_to_array_type = {
    Element: ElementArray,
    int: IntArray,
    np.int32: IntArray,
    float: FloatArray,
    bool: BoolArray,
    str: StringArray,
    Binary: BinaryArray,
    Time: TimeArray,
    Color: ColorArray,
    Vector2: Vector2Array,
    Vector3: Vector3Array,
    Vector4: Vector4Array,
    Angle: AngleArray,
    Quaternion: QuaternionArray,
    Matrix: MatrixArray,
    np.uint64: UInt64Array,
    np.uint8: UInt8Array,
}

_all_known_dmx_types = list(_valid_dmx_type_to_array_type.keys()) + list(_valid_dmx_type_to_array_type.values())


class _StringDictionary(list):
    def __init__(self, encoding, encoding_ver):
        super().__init__()
        self.encoding = encoding
        self.encoding_ver = encoding_ver
        self.__string_index_writer = write_short if self.index_size == 2 else write_int

    @cached_property
    def index_size(self):
        if self.encoding == 'binary':
            if self.encoding_ver == 4:
                return 2
            elif 2 <= self.encoding_ver <= 3:
                return 2
            return 4

    @cached_property
    def length_size(self):
        if self.encoding == 'binary':
            if 2 <= self.encoding_ver <= 3:
                return 2
            return 4
        return 0

    @cached_property
    def dummy(self):
        return (self.encoding == 'binary' and self.encoding_ver == 1) or self.encoding == 'binary_proto'

    def from_file(self, file: BinaryIO):
        if self.dummy:
            return
        string_count = read_short(file) if self.length_size == 2 else read_int(file)
        for _ in range(string_count):
            self.append(read_string(file))

    def from_datamodel(self, datamodel: 'DataModel'):
        strings = set()
        for element in datamodel.elements:
            if element.name is not None:
                strings.add(element.name)
            strings.add(element.dm_type)
            for name, attr in element.items():
                strings.add(name)
                if isinstance(attr, str):
                    strings.add(attr)
        self.extend(sorted(strings))

    def read_string(self, in_file):
        if self.dummy:
            return read_string(in_file)
        else:
            string_index = read_short(in_file) if self.index_size == 2 else read_int(in_file)
            if string_index == (0xFFFF if self.index_size == 2 else -1):
                return None
            return self[string_index]

    def write_string(self, out_file, string, suppress=None):
        if suppress is None:
            suppress = self.encoding_ver < 4
        if self.dummy or suppress:
            write_string(out_file, '' if string is None else string)
        elif string is None:
            self.__string_index_writer(out_file, 0xFFFF if self.index_size == 2 else -1)
        else:
            assert (string in self)
            string_index = self.index(string)
            self.__string_index_writer(out_file, string_index)

    def write_dictionary(self, out_file):
        if not self.dummy:
            (write_short if self.length_size == 2 else write_int)(out_file, len(self))
            for string in self:
                write_string(out_file, string)


class DataModel:
    def __init__(self, encoding: str, encoding_ver: int, format_: Optional[str] = None,
                 format_ver: Optional[int] = None):
        self.encoding = encoding
        self.encoding_ver = encoding_ver
        self.format = format_
        self.format_ver = format_ver

        self.used_ids = set()
        self.elements: list[Element] = []
        self.prefix_attributes = Element(self, "", uuid4())
        self.root = None
        self.allow_random_ids = True

    def __repr__(self):
        root = f'(root == "{self.root.name}")' if self.root else ''
        return f"<Datamodel 0x{id(self)} {root}>"

    def put_element(self, element: Element):
        if element.id is None and not self.allow_random_ids:
            raise ValueError("{} does not allow random IDs.".format(self))
        if not element.placeholder and element.id in self.elements:
            raise IDCollisionError(f"{element.dm_type}({element.name!r}) invalid for {self}: "
                                   f"ID collision with {self.elements[self.elements.index(element)]}. "
                                   f"ID is {element.id}.")
        self.elements.append(element)
        self.used_ids.add(element.id)
        if len(self.elements) == 1:
            self.root = element
        return element

    def add_element(self, name, dm_element="DmElement", uuid=None, is_placeholder=False):
        if uuid is None and not self.allow_random_ids:
            raise ValueError("{} does not allow random IDs.".format(self))

        if uuid:
            if isinstance(uuid, UUID):
                uuid = uuid
            elif isinstance(uuid, str):
                uuid = uuid3(UUID('20ba94f8-59f0-4579-9e01-50aac4567d3b'), uuid)
            else:
                raise ValueError("id must be uuid.UUID or str")
        else:
            uuid = uuid4()

        if not is_placeholder and uuid in self.used_ids:
            collision = list(filter(lambda a: a.id == uuid, self.elements))
            raise IDCollisionError(
                f"({name!r}) invalid for {self}: ID collision with {collision[0]}. ID is {uuid}.")

        element = Element(self, name, uuid, dm_element, is_placeholder)
        self.used_ids.add(element.id)
        self.elements.append(element)
        if len(self.elements) == 1:
            self.root = element
        return element


class _BinaryCodec:
    """Binary DMX I/O over a DataModel; serialization state stays here."""

    def __init__(self, document: DataModel, encoding_ver: int):
        if not 1 <= encoding_ver <= 9:
            raise ValueError(f'Unsupported binary DMX version {encoding_ver}')
        self.document = document
        self.encoding = document.encoding if document.encoding == 'binary_proto' else 'binary'
        self.encoding_ver = encoding_ver
        self.format = document.format
        self.format_ver = document.format_ver
        self.elements = document.elements
        self.prefix_attributes = document.prefix_attributes
        self.__string_dict = _StringDictionary(self.encoding, encoding_ver)
        if encoding_ver <= 2:
            self.__value_reader = self._read_value_v1
        elif encoding_ver <= 5:
            self.__value_reader = self._read_value_v2
        else:
            self.__value_reader = self._read_value_v3

    def from_file(self, file: BinaryIO):
        file.seek(2, 1)  # skip header's line break and null terminator

        if self.encoding_ver >= 9:
            for prefix_elem in range(read_int(file)):
                self._read_element(file, self.prefix_attributes, use_string_dict=False)

        self.__string_dict = _StringDictionary(self.encoding, self.encoding_ver)
        self.__string_dict.from_file(file)
        element_count = read_int(file)
        for _ in range(element_count):
            dm_element = self.__string_dict.read_string(file)
            name = self.__string_dict.read_string(file) if self.encoding_ver >= 4 else read_string(file)
            uuid = UUID(bytes_le=file.read(16))
            self.document.add_element(name, dm_element, uuid)
        del dm_element, name, uuid, element_count

        for element in filter(lambda e: not e.placeholder, self.elements):
            self._read_element(file, element)
        self.document.root = self.elements[0]

    def _read_element(self, file, parent: Element, use_string_dict=True):
        value_reader = self.__value_reader
        for _ in range(read_int(file)):
            parent.add_attribute(self.__string_dict.read_string(file) if use_string_dict else read_string(file),
                                 value_reader(file, read_byte(file)))

    def _read_element_ref(self, file):
        element_index = read_int(file)
        if element_index == -1:
            return NoneElement()
        elif element_index == -2:
            self.document.add_element('Missing element', uuid=UUID(hex=read_string(file)), is_placeholder=True)
        else:
            return self.elements[element_index]

    def _read_value_v1(self, file, type_id):
        if type_id == 1:
            return self._read_element_ref(file)
        elif type_id == 2:
            return read_int(file)
        elif type_id == 3:
            return read_float(file)
        elif type_id == 4:
            return read_byte(file) != 0
        elif type_id == 5:
            return read_string(file) if self.encoding_ver < 4 else self.__string_dict.read_string(file)
        elif type_id == 6:
            return Binary.from_file(file)
        elif type_id == 7:
            raise NotImplementedError('Dunno how to read "ObjectID"')
        elif type_id == 8:
            return Color.from_file(file)
        elif type_id == 9:
            return Vector2.from_file(file)
        elif type_id == 10:
            return Vector3.from_file(file)
        elif type_id == 11:
            return Vector4.from_file(file)
        elif type_id == 12:
            return Angle.from_file(file)
        elif type_id == 13:
            return Quaternion.from_file(file)
        elif type_id == 14:
            return Matrix.from_file(file)
        elif type_id == 15:
            return ElementArray([self._read_element_ref(file) for _ in range(read_int(file))])
        elif type_id == 16:
            return IntArray.from_file(file)
        elif type_id == 17:
            return FloatArray.from_file(file)
        elif type_id == 18:
            return BoolArray.from_file(file)
        elif type_id == 19:
            return StringArray([read_string(file) for _ in range(read_int(file))])
        elif type_id == 20:
            return BinaryArray([Binary.from_file(file) for _ in range(read_int(file))])
        elif type_id == 21:
            raise NotImplementedError('Dunno how to read "ObjectID Array"')
        elif type_id == 22:
            return ColorArray.from_file(file)
        elif type_id == 23:
            return Vector2Array.from_file(file)
        elif type_id == 24:
            return Vector3Array.from_file(file)
        elif type_id == 25:
            return Vector4Array.from_file(file)
        elif type_id == 26:
            return AngleArray.from_file(file)
        elif type_id == 27:
            return QuaternionArray.from_file(file)
        elif type_id == 28:
            return MatrixArray.from_file(file)
        else:
            raise NotImplementedError(f'DMX V1 type {type_id} is not supported')

    def _read_value_v2(self, file, type_id):
        if type_id == 1:
            return self._read_element_ref(file)
        elif type_id == 2:
            return read_int(file)
        elif type_id == 3:
            return read_float(file)
        elif type_id == 4:
            return read_byte(file) != 0
        elif type_id == 5:
            return read_string(file) if self.encoding_ver < 4 else self.__string_dict.read_string(file)
        elif type_id == 6:
            return Binary.from_file(file)
        elif type_id == 7:
            return Time.from_file(file)
        elif type_id == 8:
            return Color.from_file(file)
        elif type_id == 9:
            return Vector2.from_file(file)
        elif type_id == 10:
            return Vector3.from_file(file)
        elif type_id == 11:
            return Vector4.from_file(file)
        elif type_id == 12:
            return Angle.from_file(file)
        elif type_id == 13:
            return Quaternion.from_file(file)
        elif type_id == 14:
            return Matrix.from_file(file)
        elif type_id == 15:
            item_count = read_int(file)
            return ElementArray([self._read_element_ref(file) for _ in range(item_count)])
        elif type_id == 16:
            return IntArray.from_file(file)
        elif type_id == 17:
            return FloatArray.from_file(file)
        elif type_id == 18:
            return BoolArray.from_file(file)
        elif type_id == 19:
            item_count = read_int(file)
            return StringArray([read_string(file) for _ in range(item_count)])
        elif type_id == 20:
            item_count = read_int(file)
            return BinaryArray([Binary.from_file(file) for _ in range(item_count)])
        elif type_id == 21:
            item_count = read_int(file)
            return TimeArray([Time.from_file(file) for _ in range(item_count)])
        elif type_id == 22:
            return ColorArray.from_file(file)
        elif type_id == 23:
            return Vector2Array.from_file(file)
        elif type_id == 24:
            return Vector3Array.from_file(file)
        elif type_id == 25:
            return Vector4Array.from_file(file)
        elif type_id == 26:
            return AngleArray.from_file(file)
        elif type_id == 27:
            return QuaternionArray.from_file(file)
        elif type_id == 28:
            return MatrixArray.from_file(file)
        else:
            raise NotImplementedError(f'DMX V2 type {type_id} is not supported')

    def _read_value_v3(self, file, type_id):
        if type_id == 1:
            return self._read_element_ref(file)
        elif type_id == 2:
            return read_int(file)
        elif type_id == 3:
            return read_float(file)
        elif type_id == 4:
            return read_byte(file) != 0
        elif type_id == 5:
            return self.__string_dict.read_string(file)
        elif type_id == 6:
            return Binary.from_file(file)
        elif type_id == 7:
            return Time.from_file(file)
        elif type_id == 8:
            return Color.from_file(file)
        elif type_id == 9:
            return Vector2.from_file(file)
        elif type_id == 10:
            return Vector3.from_file(file)
        elif type_id == 11:
            return Vector4.from_file(file)
        elif type_id == 12:
            return Angle.from_file(file)
        elif type_id == 13:
            return Quaternion.from_file(file)
        elif type_id == 14:
            return Matrix.from_file(file)
        elif type_id == 15:
            return np.uint64(read_quad(file))
        elif type_id == 16:
            return np.uint8(read_byte(file))
        elif type_id == 33:
            item_count = read_int(file)
            return ElementArray([self._read_element_ref(file) for _ in range(item_count)])
        elif type_id == 34:
            return IntArray.from_file(file)
        elif type_id == 35:
            return FloatArray.from_file(file)
        elif type_id == 36:
            return BoolArray.from_file(file)
        elif type_id == 37:
            item_count = read_int(file)
            return StringArray([read_string(file) for _ in range(item_count)])
        elif type_id == 38:
            item_count = read_int(file)
            return BinaryArray([Binary.from_file(file) for _ in range(item_count)])
        elif type_id == 39:
            item_count = read_int(file)
            return TimeArray([Time.from_file(file) for _ in range(item_count)])
        elif type_id == 40:
            return ColorArray.from_file(file)
        elif type_id == 41:
            return Vector2Array.from_file(file)
        elif type_id == 42:
            return Vector3Array.from_file(file)
        elif type_id == 43:
            return Vector4Array.from_file(file)
        elif type_id == 44:
            return AngleArray.from_file(file)
        elif type_id == 45:
            return QuaternionArray.from_file(file)
        elif type_id == 46:
            return MatrixArray.from_file(file)
        elif type_id == 47:
            return UInt64Array.from_file(file)
        elif type_id == 48:
            return UInt8Array.from_file(file)
        else:
            raise NotImplementedError(f'DMX V9 type {type_id} is not supported')

    def write(self, buffer: BinaryIO):
        header = header_format.format('binary', self.encoding_ver, self.format, self.format_ver)
        buffer.write(header.encode('ascii') + b'\n\x00')

        if self.encoding_ver >= 9:
            write_int(buffer, 1 if self.prefix_attributes else 0)
            if self.prefix_attributes:
                write_int(buffer, len(self.prefix_attributes))
                for name, value in self.prefix_attributes.items():
                    write_string(buffer, name)
                    self._write_value_v3(buffer, value)

        self.__string_dict = _StringDictionary('binary', self.encoding_ver)
        self.__string_dict.from_datamodel(self.document)
        self.__string_dict.write_dictionary(buffer)

        write_int(buffer, len(self.elements))
        for element in self.elements:
            self.__string_dict.write_string(buffer, element.dm_type, suppress=False)
            self.__string_dict.write_string(buffer, element.name)
            buffer.write(element.id.bytes_le)

        if self.encoding_ver <= 2:
            write_value = self._write_value_v1
        elif self.encoding_ver <= 5:
            write_value = self._write_value_v2
        else:
            write_value = self._write_value_v3

        for element in self.elements:
            write_int(buffer, len(element))
            for prop_name, prop_value in element.items():
                self.__string_dict.write_string(buffer, prop_name, suppress=False)
                write_value(buffer, prop_value)


    def _write_element_ref(self, file, element: Element):
        if isinstance(element, NoneElement):
            write_int(file, -1)
        elif element.placeholder:
            write_int(file, -2)
            write_string(file, element.id.hex)
        elif element not in self.elements:
            write_int(file, -1)
        else:
            write_int(file, self.elements.index(element))

    def _write_value_v1(self, buffer, value):
        if isinstance(value, Element):
            write_byte(buffer, 1)
            self._write_element_ref(buffer, value)
        elif isinstance(value, NoneElement):
            write_byte(buffer, 1)
            write_int(buffer, -1)
        elif isinstance(value, bool):
            write_byte(buffer, 4)
            write_byte(buffer, value)
        elif isinstance(value, int):
            write_byte(buffer, 2)
            write_int(buffer, value)
        elif isinstance(value, float):
            write_byte(buffer, 3)
            write_float(buffer, value)
        elif isinstance(value, str):
            write_byte(buffer, 5)
            str_dict_supported = self.encoding_ver < 4
            return write_string(buffer, value) if str_dict_supported else self.__string_dict.write_string(buffer, value)
        elif isinstance(value, Binary):
            write_byte(buffer, 6)
            value.to_file(buffer)
        elif isinstance(value, Color):
            write_byte(buffer, 8)
            value.to_file(buffer)
        elif isinstance(value, Vector2):
            write_byte(buffer, 9)
            buffer.write(value.tobytes())
        elif isinstance(value, Vector3):
            write_byte(buffer, 10)
            buffer.write(value.tobytes())
        elif isinstance(value, Vector4):
            write_byte(buffer, 11)
            buffer.write(value.tobytes())
        elif isinstance(value, Angle):
            write_byte(buffer, 12)
            buffer.write(value.tobytes())
        elif isinstance(value, Quaternion):
            write_byte(buffer, 13)
            buffer.write(value.tobytes())
        elif isinstance(value, Matrix):
            write_byte(buffer, 14)
            buffer.write(value.tobytes())
        elif isinstance(value, ElementArray):
            write_byte(buffer, 15)
            write_int(buffer, len(value))
            [self._write_element_ref(buffer, elem) for elem in value]
        elif isinstance(value, IntArray):
            write_byte(buffer, 16)
            value.to_file(buffer)
        elif isinstance(value, FloatArray):
            write_byte(buffer, 17)
            value.to_file(buffer)
        elif isinstance(value, BoolArray):
            write_byte(buffer, 18)
            value.to_file(buffer)
        elif isinstance(value, StringArray):
            write_byte(buffer, 19)
            write_int(buffer, len(value))
            [write_string(buffer, string) for string in value]
        elif isinstance(value, BinaryArray):
            write_byte(buffer, 20)
            write_int(buffer, len(value))
            value.to_file(buffer)
        elif isinstance(value, ColorArray):
            write_byte(buffer, 22)
            value.to_file(buffer)
        elif isinstance(value, Vector2Array):
            write_byte(buffer, 23)
            value.to_file(buffer)
        elif isinstance(value, AngleArray):
            write_byte(buffer, 26)
            value.to_file(buffer)
        elif isinstance(value, QuaternionArray):
            write_byte(buffer, 27)
            value.to_file(buffer)
        elif isinstance(value, Vector3Array):
            write_byte(buffer, 24)
            value.to_file(buffer)
        elif isinstance(value, Vector4Array):
            write_byte(buffer, 25)
            value.to_file(buffer)
        elif isinstance(value, MatrixArray):
            write_byte(buffer, 28)
            value.to_file(buffer)
        else:
            raise NotImplementedError(f'Numpy dtype {value.dtype} is not supported')

    def _write_value_v2(self, buffer, value):
        if isinstance(value, Element):
            write_byte(buffer, 1)
            self._write_element_ref(buffer, value)
        elif isinstance(value, NoneElement):
            write_byte(buffer, 1)
            write_int(buffer, -1)
        elif isinstance(value, bool):
            write_byte(buffer, 4)
            write_byte(buffer, value)
        elif isinstance(value, int):
            write_byte(buffer, 2)
            write_int(buffer, value)
        elif isinstance(value, Time):
            write_byte(buffer, 7)
            value.to_file(buffer)
        elif isinstance(value, float):
            write_byte(buffer, 3)
            write_float(buffer, value)
        elif isinstance(value, str):
            write_byte(buffer, 5)
            str_dict_supported = self.encoding_ver < 4
            return write_string(buffer, value) if str_dict_supported else self.__string_dict.write_string(buffer, value)
        elif isinstance(value, Binary):
            write_byte(buffer, 6)
            value.to_file(buffer)
        elif isinstance(value, Color):
            write_byte(buffer, 8)
            value.to_file(buffer)
        elif isinstance(value, Vector2):
            write_byte(buffer, 9)
            buffer.write(value.tobytes())
        elif isinstance(value, Vector3):
            write_byte(buffer, 10)
            buffer.write(value.tobytes())
        elif isinstance(value, Vector4):
            write_byte(buffer, 11)
            buffer.write(value.tobytes())
        elif isinstance(value, Angle):
            write_byte(buffer, 12)
            buffer.write(value.tobytes())
        elif isinstance(value, Quaternion):
            write_byte(buffer, 13)
            buffer.write(value.tobytes())
        elif isinstance(value, Matrix):
            write_byte(buffer, 14)
            buffer.write(value.tobytes())
        elif isinstance(value, ElementArray):
            write_byte(buffer, 15)
            write_int(buffer, len(value))
            [self._write_element_ref(buffer, elem) for elem in value]
        elif isinstance(value, IntArray):
            write_byte(buffer, 16)
            value.to_file(buffer)
        elif isinstance(value, FloatArray):
            write_byte(buffer, 17)
            value.to_file(buffer)
        elif isinstance(value, BoolArray):
            write_byte(buffer, 18)
            value.to_file(buffer)
        elif isinstance(value, StringArray):
            write_byte(buffer, 19)
            write_int(buffer, len(value))
            [write_string(buffer, s) for s in value]
        elif isinstance(value, BinaryArray):
            write_byte(buffer, 20)
            value.to_file(buffer)
        elif isinstance(value, TimeArray):
            write_byte(buffer, 21)
            value.to_file(buffer)
        elif isinstance(value, ColorArray):
            write_byte(buffer, 22)
            value.to_file(buffer)
        elif isinstance(value, Vector2Array):
            write_byte(buffer, 23)
            value.to_file(buffer)
        elif isinstance(value, AngleArray):
            write_byte(buffer, 26)
            value.to_file(buffer)
        elif isinstance(value, QuaternionArray):
            write_byte(buffer, 27)
            value.to_file(buffer)
        elif isinstance(value, Vector3Array):
            write_byte(buffer, 24)
            value.to_file(buffer)
        elif isinstance(value, Vector4Array):
            write_byte(buffer, 25)
            value.to_file(buffer)
        elif isinstance(value, MatrixArray):
            write_byte(buffer, 28)
            value.to_file(buffer)
        else:
            raise NotImplementedError(f'Numpy dtype {value.dtype} is not supported')

    def _write_value_v3(self, buffer, value):
        if isinstance(value, Element):
            write_byte(buffer, 1)
            self._write_element_ref(buffer, value)
        elif isinstance(value, NoneElement):
            write_byte(buffer, 1)
            write_int(buffer, -1)
        elif isinstance(value, bool):
            write_byte(buffer, 4)
            write_byte(buffer, value)
        elif isinstance(value, np.uint64):
            write_byte(buffer, 15)
            write_quad(buffer, int(value))
        elif isinstance(value, np.uint8):
            write_byte(buffer, 16)
            write_byte(buffer, int(value))
        elif isinstance(value, int):
            if value.bit_length() >= 64:
                write_byte(buffer, 15)
                write_quad(buffer, value)
            else:
                write_byte(buffer, 2)
                write_int(buffer, value)
        elif isinstance(value, Time):
            write_byte(buffer, 7)
            value.to_file(buffer)
        elif isinstance(value, float):
            write_byte(buffer, 3)
            write_float(buffer, value)
        elif isinstance(value, str):
            write_byte(buffer, 5)
            str_dict_supported = self.encoding_ver < 4
            return write_string(buffer, value) if str_dict_supported else self.__string_dict.write_string(buffer, value)
        elif isinstance(value, Binary):
            write_byte(buffer, 6)
            value.to_file(buffer)
        elif isinstance(value, Color):
            write_byte(buffer, 8)
            value.to_file(buffer)
        elif isinstance(value, Vector2):
            write_byte(buffer, 9)
            buffer.write(value.tobytes())
        elif isinstance(value, Vector3):
            write_byte(buffer, 10)
            buffer.write(value.tobytes())
        elif isinstance(value, Vector4):
            write_byte(buffer, 11)
            buffer.write(value.tobytes())
        elif isinstance(value, Angle):
            write_byte(buffer, 12)
            buffer.write(value.tobytes())
        elif isinstance(value, Quaternion):
            write_byte(buffer, 13)
            buffer.write(value.tobytes())
        elif isinstance(value, Matrix):
            write_byte(buffer, 14)
            buffer.write(value.tobytes())
        elif isinstance(value, ElementArray):
            write_byte(buffer, 33)
            write_int(buffer, len(value))
            [self._write_element_ref(buffer, elem) for elem in value]
        elif isinstance(value, IntArray):
            write_byte(buffer, 34)
            value.to_file(buffer)
        elif isinstance(value, FloatArray):
            write_byte(buffer, 35)
            value.to_file(buffer)
        elif isinstance(value, BoolArray):
            write_byte(buffer, 36)
            value.to_file(buffer)
        elif isinstance(value, StringArray):
            write_byte(buffer, 37)
            write_int(buffer, len(value))
            [write_string(buffer, s) for s in value]
        elif isinstance(value, BinaryArray):
            write_byte(buffer, 38)
            value.to_file(buffer)
        elif isinstance(value, TimeArray):
            write_byte(buffer, 39)
            value.to_file(buffer)
        elif isinstance(value, ColorArray):
            write_byte(buffer, 40)
            value.to_file(buffer)
        elif isinstance(value, Vector2Array):
            write_byte(buffer, 41)
            value.to_file(buffer)
        elif isinstance(value, AngleArray):
            write_byte(buffer, 44)
            value.to_file(buffer)
        elif isinstance(value, QuaternionArray):
            write_byte(buffer, 45)
            value.to_file(buffer)
        elif isinstance(value, Vector3Array):
            write_byte(buffer, 42)
            value.to_file(buffer)
        elif isinstance(value, Vector4Array):
            write_byte(buffer, 43)
            value.to_file(buffer)
        elif isinstance(value, MatrixArray):
            write_byte(buffer, 46)
            value.to_file(buffer)
        elif isinstance(value, UInt64Array):
            write_byte(buffer, 47)
            value.to_file(buffer)
        elif isinstance(value, UInt8Array):
            write_byte(buffer, 48)
            value.to_file(buffer)
        else:
            raise NotImplementedError(f'Numpy dtype {value.dtype} is not supported')

_KV2_SCALAR_TYPES = {
    'element': Element, 'int': int, 'float': float, 'bool': bool,
    'string': str, 'binary': Binary, 'time': Time, 'color': Color,
    'vector2': Vector2, 'vector3': Vector3, 'vector4': Vector4,
    'angle': Angle, 'quaternion': Quaternion, 'matrix': Matrix,
    'uint64': np.uint64, 'uint8': np.uint8,
}
_KV2_TYPE_NAMES = {value: name for name, value in _KV2_SCALAR_TYPES.items()}
_KV2_ARRAY_TYPES = {
    name + '_array': _valid_dmx_type_to_array_type[value]
    for name, value in _KV2_SCALAR_TYPES.items()
    if value in _valid_dmx_type_to_array_type
}
_KV2_ARRAY_NAMES = {value: name for name, value in _KV2_ARRAY_TYPES.items()}


class KeyValues2ParseError(ValueError):
    """Invalid KeyValues2 syntax, type, ID, or reference."""


class _KV2Lexer:
    def __init__(self, text: str):
        self.text = text
        self.offset = 0
        self.line = 1
        self.column = 1
        self.pending = None

    def _advance(self):
        char = self.text[self.offset]
        self.offset += 1
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char

    def _next(self):
        text = self.text
        while self.offset < len(text):
            if text[self.offset].isspace():
                self._advance()
            elif text.startswith('//', self.offset):
                while self.offset < len(text) and text[self.offset] != '\n':
                    self._advance()
            elif text.startswith('/*', self.offset):
                line, column = self.line, self.column
                self._advance()
                self._advance()
                while self.offset < len(text) and not text.startswith('*/', self.offset):
                    self._advance()
                if self.offset == len(text):
                    raise KeyValues2ParseError(f'Unterminated comment at line {line}, column {column}')
                self._advance()
                self._advance()
            else:
                break
        line, column = self.line, self.column
        if self.offset == len(text):
            return 'EOF', '', line, column
        char = self._advance()
        if char in '{}[],':
            return char, char, line, column
        if char == '"':
            value = []
            while self.offset < len(text):
                char = self._advance()
                if char == '"':
                    return 'STRING', ''.join(value), line, column
                if char == '\\':
                    if self.offset == len(text):
                        break
                    escaped = self._advance()
                    value.append({'n': '\n', 'r': '\r', 't': '\t',
                                  '"': '"', '\\': '\\'}.get(escaped, '\\' + escaped))
                else:
                    value.append(char)
            raise KeyValues2ParseError(f'Unterminated string at line {line}, column {column}')
        value = [char]
        while self.offset < len(text) and not text[self.offset].isspace() and text[self.offset] not in '{}[],':
            if text.startswith('//', self.offset) or text.startswith('/*', self.offset):
                break
            value.append(self._advance())
        return 'ATOM', ''.join(value), line, column

    def peek(self):
        if self.pending is None:
            self.pending = self._next()
        return self.pending

    def pop(self):
        token = self.peek()
        self.pending = None
        return token

    def expect(self, kind: str):
        token = self.pop()
        if token[0] != kind:
            raise KeyValues2ParseError(
                f'Expected {kind}, got {token[0]} at line {token[2]}, column {token[3]}')
        return token

    def word(self):
        token = self.pop()
        if token[0] not in ('STRING', 'ATOM'):
            raise KeyValues2ParseError(
                f'Expected a value, got {token[0]} at line {token[2]}, column {token[3]}')
        return token


class _KV2Node:
    def __init__(self, dm_type: str, line: int, column: int):
        self.dm_type = dm_type
        self.line = line
        self.column = column
        self.name = None
        self.id = None
        self.attributes = {}


class _KV2Ref:
    def __init__(self, uid: UUID, line: int, column: int):
        self.id = uid
        self.line = line
        self.column = column


class _KeyValues2Reader:
    """Parse KeyValues2 into a DataModel with per-read lexer and node state."""

    def __init__(self, document: DataModel, version: int, text: str):
        if not 1 <= version <= 4:
            raise ValueError(f'Unsupported keyvalues2 DMX version {version}')
        self.document = document
        self.lexer = _KV2Lexer(text)
        self.nodes: list[_KV2Node] = []

    @staticmethod
    def _scalar(kind: str, token):
        value, line, column = token[1:]
        try:
            if kind == 'element':
                return _KV2Ref(UUID(value), line, column) if value else NoneElement()
            if kind == 'string':
                return value
            if kind == 'bool':
                if value.lower() not in ('0', '1', 'true', 'false'):
                    raise ValueError('expected 0, 1, true, or false')
                return value.lower() in ('1', 'true')
            if kind == 'binary':
                return Binary(bytes.fromhex(value))
            if kind in ('int', 'uint8', 'uint64'):
                number = int(value, 0) if value.startswith(('0x', '0X')) else int(value)
                if kind == 'uint8' and not 0 <= number <= 255:
                    raise ValueError('uint8 out of range')
                if kind == 'uint64' and not 0 <= number < 2**64:
                    raise ValueError('uint64 out of range')
                return _KV2_SCALAR_TYPES[kind](number)
            if kind in ('float', 'time'):
                return _KV2_SCALAR_TYPES[kind](float(value))
            numbers = value.split()
            if kind == 'color':
                return Color([int(number) for number in numbers])
            return _KV2_SCALAR_TYPES[kind]([float(number) for number in numbers])
        except (ValueError, TypeError, AssertionError, OverflowError) as exc:
            raise KeyValues2ParseError(
                f'Invalid {kind} value {value!r} at line {line}, column {column}: {exc}') from exc

    def _array(self, kind: str):
        lexer = self.lexer
        lexer.expect('[')
        values = []
        while lexer.peek()[0] != ']':
            if lexer.peek()[0] == 'EOF':
                lexer.expect(']')
            if lexer.peek()[0] == ',':
                lexer.pop()
                continue
            token = lexer.word()
            if kind == 'element_array' and lexer.peek()[0] == '{':
                values.append(self._element(token))
            elif kind == 'element_array':
                if token[1] != 'element':
                    raise KeyValues2ParseError(
                        f'Expected element reference or definition at line {token[2]}, column {token[3]}')
                values.append(self._scalar('element', lexer.word()))
            else:
                values.append(self._scalar(kind[:-6], token))
            if lexer.peek()[0] == ',':
                lexer.pop()
        lexer.expect(']')
        if kind == 'element_array':
            return values
        try:
            return _KV2_ARRAY_TYPES[kind](values)
        except (ValueError, TypeError, AssertionError) as exc:
            raise KeyValues2ParseError(f'Invalid {kind}: {exc}') from exc

    def _element(self, type_token):
        lexer = self.lexer
        node = _KV2Node(type_token[1], type_token[2], type_token[3])
        self.nodes.append(node)
        lexer.expect('{')
        while lexer.peek()[0] != '}':
            if lexer.peek()[0] == 'EOF':
                lexer.expect('}')
            key_token = lexer.word()
            key = key_token[1]
            kind_token = lexer.word()
            kind = kind_token[1]
            if key == 'id':
                if kind != 'elementid' or node.id is not None:
                    raise KeyValues2ParseError(
                        f'Invalid or duplicate element ID at line {key_token[2]}, column {key_token[3]}')
                value = lexer.word()
                try:
                    node.id = UUID(value[1])
                except ValueError as exc:
                    raise KeyValues2ParseError(
                        f'Invalid element ID at line {value[2]}, column {value[3]}') from exc
            elif key == 'name':
                if kind != 'string' or node.name is not None:
                    raise KeyValues2ParseError(
                        f'Invalid or duplicate element name at line {key_token[2]}, column {key_token[3]}')
                node.name = lexer.word()[1]
            else:
                if key in node.attributes:
                    raise KeyValues2ParseError(
                        f'Duplicate attribute {key!r} at line {key_token[2]}, column {key_token[3]}')
                if kind in _KV2_ARRAY_TYPES:
                    value = self._array(kind)
                elif lexer.peek()[0] == '{':
                    value = self._element(kind_token)
                elif kind in _KV2_SCALAR_TYPES:
                    value = self._scalar(kind, lexer.word())
                else:
                    raise KeyValues2ParseError(
                        f'Unsupported DMX type {kind!r} at line {kind_token[2]}, column {kind_token[3]}')
                node.attributes[key] = value
        lexer.expect('}')
        return node

    def read(self):
        while self.lexer.peek()[0] != 'EOF':
            self._element(self.lexer.word())
        elements = {}
        for node in self.nodes:
            if node.dm_type == '$prefix_element$':
                continue
            if node.id is None:
                raise KeyValues2ParseError(
                    f'{node.dm_type} lacks an ID at line {node.line}, column {node.column}')
            if node.id in elements:
                raise KeyValues2ParseError(
                    f'Duplicate element ID {node.id} at line {node.line}, column {node.column}')
            elements[node.id] = self.document.add_element(node.name, node.dm_type, node.id)
        if not elements:
            raise KeyValues2ParseError('KeyValues2 DMX has no elements')

        def resolve(candidate):
            if isinstance(candidate, _KV2Node):
                return elements[candidate.id]
            if isinstance(candidate, _KV2Ref):
                if candidate.id not in elements:
                    raise KeyValues2ParseError(
                        f'Unresolved element {candidate.id} at line {candidate.line}, column {candidate.column}')
                return elements[candidate.id]
            if type(candidate) is list:
                return ElementArray(resolve(item) for item in candidate)
            return candidate

        for node in self.nodes:
            target = (self.document.prefix_attributes if node.dm_type == '$prefix_element$'
                      else elements[node.id])
            for key, value in node.attributes.items():
                target.add_attribute(key, resolve(value))


class _KeyValues2Writer:
    """Emit KeyValues2 from a DataModel without parser state."""

    def __init__(self, document: DataModel, version: int):
        if not 1 <= version <= 4:
            raise ValueError(f'Unsupported keyvalues2 DMX version {version}')
        self.document = document
        self.version = version

    @staticmethod
    def _quote(value: str):
        escaped = (value.replace('\\', '\\\\').replace('"', '\\"')
                   .replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t'))
        return f'"{escaped}"'

    @staticmethod
    def _scalar_text(kind: str, value, element_ids):
        if kind == 'element':
            if isinstance(value, NoneElement):
                return ''
            if value.id not in element_ids:
                raise ValueError(f'Element reference {value.id} is not in the document')
            return str(value.id)
        if kind == 'bool':
            return '1' if value else '0'
        if kind == 'binary':
            return value.hex()
        if kind in ('vector2', 'vector3', 'vector4', 'angle', 'quaternion', 'matrix'):
            return ' '.join(repr(float(number)) for number in np.asarray(value).flat)
        if kind == 'color':
            return ' '.join(str(int(number)) for number in np.asarray(value).flat)
        if kind in ('float', 'time'):
            return repr(float(value))
        return str(value)

    def _write_attribute(self, buffer: TextIO, key: str, value, element_ids):
        quoted = self._quote
        value_type = type(value)
        if value_type is ElementArray:
            buffer.write(f'\t{quoted(key)} "element_array"')
            if not value:
                buffer.write(' [ ]\n')
            else:
                buffer.write('\n\t[\n')
                for index, item in enumerate(value):
                    reference = self._scalar_text('element', item, element_ids)
                    comma = ',' if index + 1 < len(value) else ''
                    buffer.write(f'\t\t"element" {quoted(reference)}{comma}\n')
                buffer.write('\t]\n')
            return
        if value_type in _KV2_ARRAY_NAMES:
            kind = _KV2_ARRAY_NAMES[value_type]
            scalar_kind = kind[:-6]
            values = ', '.join(quoted(self._scalar_text(scalar_kind, item, element_ids))
                               for item in value)
            buffer.write(f'\t{quoted(key)} {quoted(kind)} [ {values} ]\n')
            return
        if value_type is Element or value_type is NoneElement:
            kind = 'element'
        else:
            kind = _KV2_TYPE_NAMES.get(value_type)
        if kind is None:
            raise TypeError(f'Unsupported KeyValues2 attribute {key!r}: {value_type.__name__}')
        buffer.write(f'\t{quoted(key)} {quoted(kind)} '
                     f'{quoted(self._scalar_text(kind, value, element_ids))}\n')

    def write(self, buffer: TextIO):
        document = self.document
        if not document.elements or document.root is not document.elements[0]:
            raise ValueError('KeyValues2 DMX requires a root element first')
        ids = {element.id for element in document.elements}
        if len(ids) != len(document.elements):
            raise ValueError('KeyValues2 DMX has duplicate element IDs')
        quoted = self._quote
        buffer.write(header_format.format('keyvalues2', self.version,
                                          document.format, document.format_ver) + '\n')
        if document.prefix_attributes:
            buffer.write('"$prefix_element$"\n{\n')
            for key, value in document.prefix_attributes.items():
                self._write_attribute(buffer, key, value, ids)
            buffer.write('}\n')
        for element in document.elements:
            if element.placeholder:
                raise ValueError(f'Cannot emit unresolved element {element.id}')
            buffer.write(f'{quoted(element.dm_type)}\n{{\n')
            buffer.write(f'\t"id" "elementid" {quoted(str(element.id))}\n')
            if element.name is not None:
                buffer.write(f'\t"name" "string" {quoted(element.name)}\n')
            for key, value in element.items():
                self._write_attribute(buffer, key, value, ids)
            buffer.write('}\n')


def make_array(initial, data_type):
    assert data_type in _valid_dmx_type_to_array_type, f"{data_type} does not have array type defined or already array"
    return _valid_dmx_type_to_array_type[data_type].from_list(initial)


def dump(document: DataModel, in_buffer: Optional[Union[BinaryIO, TextIO]] = None,
         encoding: Optional[str] = None, encoding_ver: Optional[int] = None):
    """Serialize a document to a stream, or return bytes (binary) / str (KeyValues2)."""
    encoding = document.encoding if encoding is None else encoding
    version = document.encoding_ver if encoding_ver is None else encoding_ver
    if encoding == 'binary':
        codec = _BinaryCodec(document, version)
        if in_buffer is None:
            buffer = BytesIO()
            codec.write(buffer)
            return buffer.getvalue()
        codec.write(in_buffer)
        return None
    if encoding == 'keyvalues2':
        codec = _KeyValues2Writer(document, version)
        if in_buffer is None:
            buffer = StringIO()
            codec.write(buffer)
            return buffer.getvalue()
        codec.write(in_buffer)
        return None
    raise ValueError(f'Unsupported DMX encoding {encoding!r}')


def _parse_header(header: str):
    match = re.fullmatch(header_format_regex, header)
    if match:
        encoding, version, format_, format_ver = match.groups()
        return encoding, int(version), format_, int(format_ver)
    match = re.fullmatch(header_proto2_regex, header)
    if match:
        return 'binary_proto', int(match.group(1)), 'undefined_format', 0
    raise ValueError(f'Invalid DMX header {header!r}')


def parse(text: str) -> DataModel:
    """Parse a complete KeyValues2 DMX string, including its header."""
    if not isinstance(text, str):
        raise TypeError('KeyValues2 input must be text')
    end = text.find('>')
    if end < 0 or end > 256:
        raise ValueError('Missing or oversized DMX header')
    encoding, version, format_, format_ver = _parse_header(text[:end + 1].lstrip('\ufeff'))
    if encoding != 'keyvalues2':
        raise ValueError(f'Expected keyvalues2 DMX, got {encoding!r}')
    document = DataModel(encoding, version, format_, format_ver)
    _KeyValues2Reader(document, version, text[end + 1:]).read()
    return document


def load(filepath: Optional[Union[str, Path]] = None,
         file: Optional[Union[BinaryIO, TextIO]] = None) -> DataModel:
    """Load binary or KeyValues2 DMX by its header; caller-owned streams stay open."""
    if (filepath is None) == (file is None):
        raise ValueError('Provide exactly one of filepath or file')
    if filepath is not None:
        with open(filepath, 'rb') as stream:
            return _load_stream(stream)
    return _load_stream(file)


def _load_stream(file: Union[BinaryIO, TextIO]) -> DataModel:
    header_parts = []
    for _ in range(256):
        char = file.read(1)
        if char in (b'', ''):
            raise ValueError('Missing DMX header terminator')
        header_parts.append(char)
        if char in (b'>', '>'):
            break
    else:
        raise ValueError('DMX header exceeds 256 characters')
    is_text = isinstance(header_parts[0], str)
    header = ''.join(header_parts) if is_text else b''.join(header_parts).decode('utf-8-sig')
    encoding, version, format_, format_ver = _parse_header(header)
    if encoding == 'keyvalues2':
        remainder = file.read()
        text = remainder if is_text else remainder.decode('utf-8')
        document = DataModel(encoding, version, format_, format_ver)
        _KeyValues2Reader(document, version, text).read()
        return document
    if is_text:
        raise TypeError('Binary DMX requires a binary stream')
    if encoding not in ('binary', 'binary_proto'):
        raise ValueError(f'Unsupported DMX encoding {encoding!r}')
    document = DataModel(encoding, version, format_, format_ver)
    _BinaryCodec(document, version).from_file(file)
    return document
