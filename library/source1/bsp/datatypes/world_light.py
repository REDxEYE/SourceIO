import math
from dataclasses import dataclass, field
from enum import IntEnum

from SourceIO.library.shared.types import Vector3
from SourceIO.library.utils.file_utils import Buffer
from SourceIO.library.source1.bsp.bsp_file import BSPFile


class EmitType(IntEnum):
    emit_surface = 0  # 90 degree spotlight
    emit_point = 1  # simple point light source
    emit_spotlight = 2  # spotlight with penumbra
    emit_skylight = 3  # directional light with no falloff (surface must trace to SKY texture)
    emit_quakelight = 4  # linear falloff, non-lambertian
    emit_skyambient = 5  # spherical light source with no falloff (surface must trace to SKY texture)


@dataclass(slots=True)
class Color32:
    r: float = field(default=0)
    g: float = field(default=0)
    b: float = field(default=0)
    a: float = field(default=1)

    @staticmethod
    def from_array(rgba):
        return Color32(*rgba)

    def magnitude(self):
        return math.sqrt(self.r ** 2 + self.g ** 2 + self.b ** 2)

    def normalize(self):
        magn = self.magnitude()
        if magn == 0:
            return self
        self.r = self.r / magn
        self.g = self.g / magn
        self.b = self.b / magn
        return self

    def normalized(self):
        magn = self.magnitude()
        if magn == 0:
            return self
        return Color32(self.r / magn, self.g / magn, self.b / magn)

    def __repr__(self):
        magn = self.magnitude()
        if magn == 0:
            return self
        r = self.r / magn
        g = self.g / magn
        b = self.b / magn
        return "<Color R:{} G:{} B:{}>".format(r, g, b)

    @property
    def rgba(self):
        return self.r, self.g, self.b, self.a

    @property
    def rgb(self):
        return self.r, self.g, self.b


@dataclass(slots=True)
class WorldLight:
    origin: Vector3[float]
    intensity: Color32
    normal: Vector3[float]
    shadow_cast_offset: Vector3[float]
    cluster: int
    type: EmitType
    style: int
    stopdot: float
    stopdot2: float
    exponent: float
    radius: float
    constant_attn: float
    linear_attn: float
    quadratic_attn: float
    flags: int
    tex_info_id: int
    owner: int

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int, bsp: BSPFile):
        origin = buffer.read_fmt('3f')
        intensity = Color32.from_array(buffer.read_fmt('3f'))
        normal = buffer.read_fmt('3f')
        if bsp.version[0] > 20:
            shadow_cast_offset = buffer.read_fmt('3f')
        else:
            shadow_cast_offset = (0, 0, 0)
        cluster = buffer.read_int32()
        emit_type = EmitType(buffer.read_int32())
        style = buffer.read_int32()
        stopdot = buffer.read_float()
        stopdot2 = buffer.read_float()
        exponent = buffer.read_float()
        radius = buffer.read_float()
        constant_attn = buffer.read_float()
        linear_attn = buffer.read_float()
        quadratic_attn = buffer.read_float()
        flags = buffer.read_int32()
        tex_info_id = buffer.read_int32()
        owner = buffer.read_int32()
        return cls(origin, intensity, normal, shadow_cast_offset, cluster, emit_type, style, stopdot, stopdot2,
                   exponent,
                   radius, constant_attn, linear_attn, quadratic_attn, flags, tex_info_id, owner)
