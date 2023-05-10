from collections.abc import Collection
from pathlib import Path
from typing import Dict, TypeVar, List

from ..data_types.keyvalues3.ascii_keyvalues import AsciiKeyValues
from ..data_types.keyvalues3.enums import KV3Encodings, KV3Formats
from ...utils import Buffer

K = TypeVar("K", bound=str)
V = TypeVar("V")


class Node(Dict[K, V]):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


N = TypeVar("N", bound=Node)


class ClassNode(Node):
    def __init__(self, **kwargs):
        super().__init__(_class=self.__class__.__name__, **kwargs)


class NodeList(Node, Collection[N]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._children: List[Node] = []
        self["children"] = self._children

    def insert(self, index: int, value: Node) -> None:
        self._children.insert(index, value)

    def append(self, value: Node):
        self._children.append(value)

    def __len__(self) -> int:
        return len(self._children)

    def __contains__(self, item: V):
        return item in self._children

    def __iter__(self):
        return iter(self._children)


class ClassNodeList(NodeList, Collection[N]):
    def __init__(self, **kwargs):
        super().__init__(_class=self.__class__.__name__, **kwargs)
        self._children: List[ClassNode] = []
        self["children"] = self._children


class Resource:
    encoding: KV3Encodings = KV3Encodings.text
    format: KV3Formats = KV3Formats.generic

    def __init__(self) -> None:
        self._root: Node = Node()

    @classmethod
    def from_buffer(cls, buffer: Buffer, filename: Path):
        node = AsciiKeyValues.from_buffer(buffer, filename)
        self = cls()
        self._root.update(node)
        return self

    def to_buffer(self, buffer: Buffer):
        data = AsciiKeyValues.dump_str("kv3", (self.encoding.name, self.encoding.value),
                                       (self.format.name, self.format.value), self._root)
        buffer.write(data.encode("utf8"))
