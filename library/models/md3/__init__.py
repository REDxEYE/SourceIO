from dataclasses import dataclass

import numpy as np

from SourceIO.library.shared.types import Vector3
from SourceIO.library.utils import Buffer
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger("MD3")


@dataclass
class MD3Shader:
    name: str
    shader_id: int


def read_md3_shader(buffer: Buffer) -> MD3Shader:
    return MD3Shader(buffer.read_ascii_string(64), buffer.read_int32())


@dataclass
class MD3Frame:
    bbox_min: Vector3
    bbox_max: Vector3
    origin: Vector3
    radius: float
    name: str


def read_md3_frame(buffer: Buffer):
    return MD3Frame(buffer.read_fmt("3f"), buffer.read_fmt("3f"), buffer.read_fmt("3f"), buffer.read_float(),
                    buffer.read_ascii_string(16))


@dataclass
class MD3Tag:
    name: str
    origin: Vector3
    axis: tuple[Vector3, Vector3, Vector3]


def read_md3_tag(buffer: Buffer):
    return MD3Tag(buffer.read_ascii_string(64), buffer.read_fmt("3f"),
                  (buffer.read_fmt("3f"), buffer.read_fmt("3f"), buffer.read_fmt("3f")))


@dataclass
class MD3Surface:
    vertex_dtype = np.dtype([
        ("position", np.int16, (3,)),
        ("normal", np.uint16, (1,)),
    ])
    name: str
    flags: int
    frames: list[np.ndarray]
    indices: np.ndarray
    uv: np.ndarray
    shaders: list[MD3Shader]

    def positions(self, frame: int = 0):
        return self.frames[frame]["position"].astype(np.float32) / 64

    def normals(self, frame: int = 0):
        lats = (((self.frames[frame]["normal"] >> 8) & 0xFF).astype(np.float32) / 255) * np.pi * 2
        lngs = (self.frames[frame]["normal"] & 0xFF).astype(np.float32) / 255
        x = np.cos(lats) * np.sin(lngs)
        y = np.sin(lats) * np.sin(lngs)
        z = np.cos(lngs)
        return np.dstack([x, y, z]).reshape(-1, 3)


def read_md3_surface(buffer: Buffer) -> MD3Surface:
    entry = buffer.tell()
    sig = buffer.read_fourcc()
    if sig != "IDP3":
        return MD3Surface("ERROR", 0, 0, np.asarray([]), np.asarray([]), np.asarray([]), [])
    name = buffer.read_ascii_string(64)
    (flags,
     frame_count, shader_count, vertex_count, triangle_count,
     triangles_offset, shaders_offset, uv_offset, vertices_offset, end_offset) = buffer.read_fmt("10I")
    with buffer.read_from_offset(entry + vertices_offset):
        frames = np.frombuffer(buffer.read(MD3Surface.vertex_dtype.itemsize * vertex_count * frame_count),
                               dtype=MD3Surface.vertex_dtype).reshape(frame_count,vertex_count)
    with buffer.read_from_offset(entry + uv_offset):
        uvs = np.frombuffer(buffer.read(vertex_count * 8), np.float32).reshape(-1, 2)
    with buffer.read_from_offset(entry + triangles_offset):
        indices = np.frombuffer(buffer.read(triangle_count * 12), np.uint32).reshape(-1, 3)
    with buffer.read_from_offset(entry + shaders_offset):
        material = [read_md3_shader(buffer) for _ in range(shader_count)]
    buffer.seek(entry + end_offset)
    return MD3Surface(name, flags, frames, indices, uvs, material)


@dataclass
class MD3Model:
    version: int
    name: str
    flags: int
    frames: list[MD3Frame]
    tags: list[MD3Tag]
    surfaces: list[MD3Surface]
    # skins: list[Skin]


def read_md3_model(buffer: Buffer) -> MD3Model:
    ident = buffer.read_fourcc()
    if ident != "IDP3":
        logger.error("Wrong md3 identier")
        return None
    version = buffer.read_uint32()
    if version != 15:
        logger.error("Only md3 version 15 supported")
        return None
    name = buffer.read_ascii_string(64)
    flags = buffer.read_uint32()
    frame_count, tag_count, surface_count, skin_count = buffer.read_fmt("4I")
    frame_offset, tag_offset, surface_offset, end_offset = buffer.read_fmt("4I")

    frames = []
    with buffer.read_from_offset(frame_offset):
        for _ in range(frame_count):
            frames.append(read_md3_frame(buffer))
    tags = []
    with buffer.read_from_offset(tag_offset):
        for _ in range(tag_count):
            tags.append(read_md3_tag(buffer))
    surfaces = []
    with buffer.read_from_offset(surface_offset):
        for _ in range(surface_count):
            surfaces.append(read_md3_surface(buffer))
    return MD3Model(version, name, flags, frames, tags, surfaces)
