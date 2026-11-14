"""Source 1 SMD version-1 text interchange (mesh, skeletal animation, and VTA)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, TextIO

from SourceIO.library.shared.vector_types import Vector2, Vector3

__all__ = [
    'SMDDocument', 'SMDNode', 'SMDBone', 'SMDWeight', 'SMDVertex',
    'SMDTriangle', 'SMDVertexAnimationVertex', 'parse', 'dump', 'load',
]


@dataclass(slots=True)
class SMDNode:
    id: int
    name: str
    parent: int


@dataclass(slots=True)
class SMDBone:
    bone: int
    position: Vector3[float]
    rotation: Vector3[float]


@dataclass(slots=True)
class SMDWeight:
    bone: int
    weight: float


@dataclass(slots=True)
class SMDVertex:
    bone: int
    position: Vector3[float]
    normal: Vector3[float]
    uv: Vector2[float]
    # None omits the optional link count; () writes an explicit count of zero.
    weights: tuple[SMDWeight, ...] | None = None


@dataclass(slots=True)
class SMDTriangle:
    material: str
    vertices: tuple[SMDVertex, SMDVertex, SMDVertex]


@dataclass(slots=True)
class SMDVertexAnimationVertex:
    index: int
    position: Vector3[float]
    normal: Vector3[float]


@dataclass(slots=True)
class SMDDocument:
    nodes: list[SMDNode] = field(default_factory=list)
    skeleton: dict[int, list[SMDBone]] = field(default_factory=dict)
    triangles: list[SMDTriangle] = field(default_factory=list)
    vertex_animation: dict[int, list[SMDVertexAnimationVertex]] = field(default_factory=dict)
    version: int = 1


_NODE = re.compile(r'(\d+)\s+"((?:[^"\\]|\\["\\])*)"\s+(-?\d+)')
_SECTIONS = {'nodes', 'skeleton', 'triangles', 'vertexanimation'}


class _Reader:
    def __init__(self, text: str):
        self.lines = iter(enumerate(text.lstrip('\ufeff').splitlines(), 1))
        self.line = 0

    def next(self) -> str | None:
        for number, raw in self.lines:
            self.line = number
            if '//' in raw:
                quoted = escaped = False
                for index, char in enumerate(raw):
                    if escaped:
                        escaped = False
                    elif char == '\\' and quoted:
                        escaped = True
                    elif char == '"':
                        quoted = not quoted
                    elif (char == '/' and raw[index:index + 2] == '//' and not quoted
                          and (index == 0 or raw[index - 1].isspace())):
                        raw = raw[:index]
                        break
            line = raw.strip()
            if line:
                return line
        return None

    def required(self, section: str) -> str:
        line = self.next()
        if line is None:
            raise ValueError(f'Unterminated SMD {section} section')
        return line

    def integer(self, token: str) -> int:
        try:
            return int(token)
        except ValueError as exc:
            raise ValueError(f'Invalid integer {token!r} on SMD line {self.line}') from exc

    def number(self, token: str) -> float:
        try:
            result = float(token)
        except ValueError as exc:
            raise ValueError(f'Invalid number {token!r} on SMD line {self.line}') from exc
        if not math.isfinite(result):
            raise ValueError(f'Non-finite number on SMD line {self.line}')
        return result

    def numbers(self, tokens: list[str], count: int) -> tuple[float, ...]:
        if len(tokens) != count:
            raise ValueError(f'Expected {count} values on SMD line {self.line}, got {len(tokens)}')
        return tuple(self.number(token) for token in tokens)

    def nodes(self, document: SMDDocument):
        while (line := self.required('nodes')) != 'end':
            match = _NODE.fullmatch(line)
            if match is None:
                raise ValueError(f'Invalid SMD node on line {self.line}')
            uid, name, parent = match.groups()
            document.nodes.append(SMDNode(self.integer(uid),
                                          name.replace('\\"', '"').replace('\\\\', '\\'),
                                          self.integer(parent)))

    def skeleton(self, document: SMDDocument):
        frame = None
        while (line := self.required('skeleton')) != 'end':
            tokens = line.split()
            if tokens[0] == 'time':
                if len(tokens) != 2:
                    raise ValueError(f'Invalid SMD time on line {self.line}')
                time = self.integer(tokens[1])
                if time < 0 or time in document.skeleton:
                    raise ValueError(f'Invalid or duplicate SMD frame {time} on line {self.line}')
                frame = document.skeleton[time] = []
            else:
                if frame is None or len(tokens) != 7:
                    raise ValueError(f'Invalid SMD skeleton pose on line {self.line}')
                frame.append(SMDBone(self.integer(tokens[0]), self.numbers(tokens[1:4], 3),
                                     self.numbers(tokens[4:7], 3)))

    def vertex(self, line: str) -> SMDVertex:
        tokens = line.split()
        if len(tokens) < 9:
            raise ValueError(f'Invalid SMD triangle vertex on line {self.line}')
        bone = self.integer(tokens[0])
        position = self.numbers(tokens[1:4], 3)
        normal = self.numbers(tokens[4:7], 3)
        uv = self.numbers(tokens[7:9], 2)
        weights = None
        if len(tokens) > 9:
            count = self.integer(tokens[9])
            if count < 0 or len(tokens) != 10 + count * 2:
                raise ValueError(f'Invalid SMD vertex link count on line {self.line}')
            weights = tuple(SMDWeight(self.integer(tokens[10 + 2 * i]),
                                      self.number(tokens[11 + 2 * i])) for i in range(count))
            if any(weight.weight < 0 or weight.weight > 1 for weight in weights) or (
                    sum(weight.weight for weight in weights) > 1 + 1e-6):
                raise ValueError(f'Invalid SMD vertex weights on line {self.line}')
        return SMDVertex(bone, position, normal, uv, weights)

    def triangles(self, document: SMDDocument):
        while (material := self.required('triangles')) != 'end':
            vertices = tuple(self.vertex(self.required('triangles')) for _ in range(3))
            document.triangles.append(SMDTriangle(material, vertices))

    def vertexanimation(self, document: SMDDocument):
        frame = None
        while (line := self.required('vertexanimation')) != 'end':
            tokens = line.split()
            if tokens[0] == 'time':
                if len(tokens) != 2:
                    raise ValueError(f'Invalid SMD vertexanimation time on line {self.line}')
                time = self.integer(tokens[1])
                if time < 0 or time in document.vertex_animation:
                    raise ValueError(f'Invalid or duplicate SMD vertexanimation frame {time} on line {self.line}')
                frame = document.vertex_animation[time] = []
            else:
                if frame is None or len(tokens) != 7:
                    raise ValueError(f'Invalid SMD vertexanimation vertex on line {self.line}')
                index = self.integer(tokens[0])
                if index < 0:
                    raise ValueError(f'Invalid SMD vertexanimation index on line {self.line}')
                frame.append(SMDVertexAnimationVertex(index, self.numbers(tokens[1:4], 3),
                                                      self.numbers(tokens[4:7], 3)))


def _validate(document: SMDDocument):
    by_id = {node.id: node.parent for node in document.nodes}
    ids = by_id.keys()
    if (len(by_id) != len(document.nodes) or any(node.id < 0 or
            (node.parent != -1 and node.parent not in ids) for node in document.nodes)):
        raise ValueError('Duplicate SMD node or unknown parent')
    checked = set()
    for node in document.nodes:
        ancestors = set()
        parent = node.id
        while parent != -1 and parent not in checked:
            if parent in ancestors:
                raise ValueError('Cyclic SMD node hierarchy')
            ancestors.add(parent)
            parent = by_id[parent]
        checked.update(ancestors)
    for frame in document.skeleton.values():
        bones = [pose.bone for pose in frame]
        if len(bones) != len(set(bones)) or any(bone not in ids for bone in bones):
            raise ValueError('Duplicate or unknown SMD skeleton bone')
    for triangle in document.triangles:
        for vertex in triangle.vertices:
            if vertex.bone not in ids or any(link.bone not in ids for link in vertex.weights or ()):
                raise ValueError('Unknown SMD triangle bone')


def parse(text: str) -> SMDDocument:
    """Read SMD/VTA version 1 text; malformed sections raise ValueError."""
    if not isinstance(text, str):
        raise TypeError('SMD input must be text')
    reader = _Reader(text)
    header = reader.next()
    if header != 'version 1':
        raise ValueError('SMD input must start with version 1')
    document = SMDDocument()
    seen = set()
    while (section := reader.next()) is not None:
        if section not in _SECTIONS or section in seen:
            raise ValueError(f'Unknown or duplicate SMD section {section!r} on line {reader.line}')
        seen.add(section)
        getattr(reader, section)(document)
    _validate(document)
    return document


def _number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError('SMD values must be finite')
    return str(float(value))


def dump(document: SMDDocument, stream: TextIO | None = None) -> str | None:
    """Write canonical version-1 text; return it when no text stream is supplied."""
    if document.version != 1:
        raise ValueError('Only SMD version 1 is supported')
    _validate(document)
    lines = ['version 1', 'nodes']
    ids = set()
    for node in document.nodes:
        if node.id < 0 or node.id in ids or '\n' in node.name or '\r' in node.name:
            raise ValueError('Invalid or duplicate SMD node')
        ids.add(node.id)
        name = node.name.replace('\\', '\\\\').replace('"', '\\"')
        lines.append(f'{node.id} "{name}" {node.parent}')
    lines.extend(('end', 'skeleton'))
    for time, frame in sorted(document.skeleton.items()):
        if time < 0:
            raise ValueError('Invalid SMD frame time')
        lines.append(f'time {time}')
        for pose in frame:
            fields = (*pose.position, *pose.rotation)
            if len(fields) != 6:
                raise ValueError('SMD bone pose needs position and rotation vectors')
            lines.append(f'{pose.bone} ' + ' '.join(map(_number, fields)))
    lines.append('end')
    if document.triangles:
        lines.append('triangles')
        for triangle in document.triangles:
            if not triangle.material or triangle.material == 'end' or '\n' in triangle.material or (
                    '\r' in triangle.material or '//' in triangle.material) or len(triangle.vertices) != 3:
                raise ValueError('Invalid SMD triangle material or vertex count')
            lines.append(triangle.material)
            for vertex in triangle.vertices:
                fields = (*vertex.position, *vertex.normal, *vertex.uv)
                if len(fields) != 8:
                    raise ValueError('SMD vertex needs position, normal, and UV values')
                line = f'{vertex.bone} ' + ' '.join(map(_number, fields))
                if vertex.weights is not None:
                    if any(link.weight < 0 or link.weight > 1 for link in vertex.weights) or (
                            sum(link.weight for link in vertex.weights) > 1 + 1e-6):
                        raise ValueError('Invalid SMD vertex weights')
                    line += f' {len(vertex.weights)}'
                    for link in vertex.weights:
                        line += f' {link.bone} {_number(link.weight)}'
                lines.append(line)
        lines.append('end')
    if document.vertex_animation:
        lines.append('vertexanimation')
        for time, frame in sorted(document.vertex_animation.items()):
            if time < 0:
                raise ValueError('Invalid SMD vertexanimation time')
            lines.append(f'time {time}')
            for vertex in frame:
                fields = (*vertex.position, *vertex.normal)
                if vertex.index < 0 or len(fields) != 6:
                    raise ValueError('Invalid SMD vertexanimation vertex')
                lines.append(f'{vertex.index} ' + ' '.join(map(_number, fields)))
        lines.append('end')
    result = '\n'.join(lines) + '\n'
    if stream is None:
        return result
    stream.write(result)
    return None


def load(filepath: str | Path | None = None, file: TextIO | BinaryIO | None = None) -> SMDDocument:
    """Load SMD/VTA from a path or caller-owned text/binary stream."""
    if (filepath is None) == (file is None):
        raise ValueError('Provide exactly one of filepath or file')
    if filepath is not None:
        with open(filepath, 'rb') as stream:
            data = stream.read()
    else:
        data = file.read()
    return parse(data.decode('utf-8-sig') if isinstance(data, bytes) else data)
