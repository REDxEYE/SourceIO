"""Source 1 SMD version-1 text interchange (mesh, skeletal animation, and VTA)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np

from SourceIO.library.shared.vector_types import Vector3

__all__ = [
    'SMDDocument', 'SMDNode', 'SMDBone', 'SMDVertexAnimationVertex', 'parse', 'dump', 'load',
    'triangle_dtype', 'triangle_array',
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
class SMDVertexAnimationVertex:
    index: int
    position: Vector3[float]
    normal: Vector3[float]


_LINK = np.dtype([('bone', np.int32), ('weight', np.float64)])


def triangle_dtype(max_links: int = 0) -> np.dtype:
    """One row per triangle: a ``materials`` index and three corners.

    ``link_count`` is -1 when the optional count is not written and 0 for an explicit zero; only the
    first ``link_count`` entries of ``links`` are meaningful, the rest is zero padding.
    """
    corner = np.dtype([('bone', np.int32), ('position', np.float64, 3), ('normal', np.float64, 3),
                       ('uv', np.float64, 2), ('link_count', np.int32), ('links', _LINK, (max_links,))])
    return np.dtype([('material', np.int32), ('corners', corner, 3)])


def triangle_array(count: int = 0, max_links: int = 0) -> np.ndarray:
    """Zeroed triangles whose corners carry no link count."""
    triangles = np.zeros(count, triangle_dtype(max_links))
    triangles['corners']['link_count'] = -1
    return triangles


@dataclass(slots=True, eq=False)
class SMDDocument:
    nodes: list[SMDNode] = field(default_factory=list)
    skeleton: dict[int, list[SMDBone]] = field(default_factory=dict)
    materials: list[str] = field(default_factory=list)
    triangles: np.ndarray = field(default_factory=triangle_array)
    vertex_animation: dict[int, list[SMDVertexAnimationVertex]] = field(default_factory=dict)
    version: int = 1

    def __eq__(self, other):
        if not isinstance(other, SMDDocument):
            return NotImplemented
        return (self.nodes == other.nodes and self.skeleton == other.skeleton and self.materials == other.materials
                and self.vertex_animation == other.vertex_animation and self.version == other.version
                and self.triangles.dtype == other.triangles.dtype
                and np.array_equal(self.triangles, other.triangles))


_NODE = re.compile(r'(\d+)\s+"([^"]*)"\s+(-?\d+)')
_SECTIONS = {'nodes', 'skeleton', 'triangles', 'vertexanimation'}


class _Reader:
    def __init__(self, text: str):
        self.lines = iter(enumerate(text.lstrip('\ufeff').splitlines(), 1))
        self.line = 0

    def next(self) -> str | None:
        for number, raw in self.lines:
            self.line = number
            if '//' in raw:
                quoted = False
                for index, char in enumerate(raw):
                    if char == '"':
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
            document.nodes.append(SMDNode(self.integer(uid), name, self.integer(parent)))

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

    def vertex(self, line: str) -> tuple:
        tokens = line.split()
        if len(tokens) < 9:
            raise ValueError(f'Invalid SMD triangle vertex on line {self.line}')
        bone = self.integer(tokens[0])
        position = self.numbers(tokens[1:4], 3)
        normal = self.numbers(tokens[4:7], 3)
        uv = self.numbers(tokens[7:9], 2)
        links = None
        if len(tokens) > 9:
            count = self.integer(tokens[9])
            if count < 0 or len(tokens) != 10 + count * 2:
                raise ValueError(f'Invalid SMD vertex link count on line {self.line}')
            links = tuple((self.integer(tokens[10 + 2 * i]), self.number(tokens[11 + 2 * i])) for i in range(count))
            if any(weight < 0 or weight > 1 for _, weight in links) or (
                    sum(weight for _, weight in links) > 1 + 1e-6):
                raise ValueError(f'Invalid SMD vertex weights on line {self.line}')
        return bone, position, normal, uv, links

    def triangles(self, document: SMDDocument):
        lookup: dict[str, int] = {}
        rows = []
        while (material := self.required('triangles')) != 'end':
            rows.append((lookup.setdefault(material, len(lookup)),
                         [self.vertex(self.required('triangles')) for _ in range(3)]))
        width = max((len(links or ()) for _, corners in rows for *_, links in corners), default=0)
        document.materials = list(lookup)
        document.triangles = np.array(
            [(material, [(bone, position, normal, uv, -1 if links is None else len(links),
                          [*(links or ()), *[(0, 0.)] * (width - len(links or ()))])
                         for bone, position, normal, uv, links in corners])
             for material, corners in rows], dtype=triangle_dtype(width))

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


def _validate_triangles(document: SMDDocument, ids):
    triangles = document.triangles
    if triangles.ndim != 1:
        raise ValueError('SMD triangles must be a one-dimensional array')
    if not len(triangles):
        return
    corners = triangles['corners']
    used = np.unique(triangles['material'])
    if used[0] < 0 or used[-1] >= len(document.materials):
        raise ValueError('Invalid SMD triangle material or vertex count')
    for index in used.tolist():
        name = document.materials[index]
        if not name or name == 'end' or any(c in name for c in '\n\r') or '//' in name:
            raise ValueError('Invalid SMD triangle material or vertex count')
    known = np.fromiter(ids, np.int64, len(ids))
    if not np.isin(corners['bone'], known).all():
        raise ValueError('Unknown SMD triangle bone')
    if not all(np.isfinite(corners[name]).all() for name in ('position', 'normal', 'uv')):
        raise ValueError('SMD values must be finite')
    counts, links = corners['link_count'], corners['links']
    if ((counts < -1) | (counts > links.shape[-1])).any():
        raise ValueError('Invalid SMD vertex link count')
    active = np.arange(links.shape[-1]) < counts[..., None]
    weights = np.where(active, links['weight'], 0.)
    if not np.isfinite(weights).all():
        raise ValueError('SMD values must be finite')
    if (weights < 0).any() or (weights > 1).any() or (weights.sum(-1) > 1 + 1e-6).any():
        raise ValueError('Invalid SMD vertex weights')
    if not np.isin(links['bone'][active], known).all():
        raise ValueError('Unknown SMD triangle bone')


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
    _validate_triangles(document, ids)


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


def _triangle_text(document: SMDDocument, chunk_size: int = 1 << 14) -> str:
    """Triangle lines in the text of ``'%d %r ...' % values``, formatted per chunk and link count."""
    names = np.array(document.materials, dtype=object)
    parts = []
    for start in range(0, len(document.triangles), chunk_size):
        chunk = document.triangles[start:start + chunk_size]
        corners = chunk['corners'].reshape(-1)
        counts = corners['link_count']
        lines = np.empty(len(corners), dtype=object)
        for count in np.unique(counts).tolist():
            mask = counts == count
            rows = corners[mask]
            matrix = np.empty((len(rows), 9 + 2 * max(count, 0)))
            matrix[:, 0] = rows['bone']
            matrix[:, 1:4] = rows['position']
            matrix[:, 4:7] = rows['normal']
            matrix[:, 7:9] = rows['uv']
            template = '%d' + ' %r' * 8
            if count >= 0:
                matrix[:, 9::2] = rows['links']['bone'][:, :count]
                matrix[:, 10::2] = rows['links']['weight'][:, :count]
                template += f' {count}' + ' %d %r' * count
            text = '\n'.join([template] * len(rows)) % tuple(matrix.ravel().tolist())
            lines[mask] = np.array(text.split('\n'), dtype=object)
        table = np.empty((len(chunk), 4), dtype=object)
        table[:, 0] = names[chunk['material']]
        table[:, 1:] = lines.reshape(-1, 3)
        parts.append('\n'.join(table.ravel().tolist()))
    return '\n'.join(parts)


def dump(document: SMDDocument, stream: TextIO | None = None) -> str | None:
    """Write canonical version-1 text; return it when no text stream is supplied."""
    if document.version != 1:
        raise ValueError('Only SMD version 1 is supported')
    _validate(document)
    lines = ['version 1', 'nodes']
    ids = set()
    for node in document.nodes:
        if node.id < 0 or node.id in ids or any(c in node.name for c in '\n\r"'):
            raise ValueError('Invalid or duplicate SMD node')
        ids.add(node.id)
        lines.append(f'{node.id} "{node.name}" {node.parent}')
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
    if len(document.triangles):
        lines.append('triangles')
        lines.append(_triangle_text(document))
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
