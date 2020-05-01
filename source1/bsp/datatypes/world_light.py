from enum import IntEnum

import math

from ....byte_io_mdl import ByteIO


class EmitType(IntEnum):
    emit_surface = 0  # 90 degree spotlight
    emit_point = 1  # simple point light source
    emit_spotlight = 2  # spotlight with penumbra
    emit_skylight = 3  # directional light with no falloff (surface must trace to SKY texture)
    emit_quakelight = 4  # linear falloff, non-lambertian
    emit_skyambient = 5  # spherical light source with no falloff (surface must trace to SKY texture)


class Color32:

    def __init__(self):
        self.r, self.g, self.b, self.a = 1, 1, 1, 1

    @staticmethod
    def from_array(rgba):
        color = Color32()
        if len(rgba) >= 4:
            color.r, color.g, color.b, color.a = rgba
        color.r, color.g, color.b = rgba
        return color

    def magnitude(self):
        magn = math.sqrt(self.r ** 2 + self.g ** 2 + self.b ** 2)
        return magn

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
        color = Color32()
        color.r = self.r / magn
        color.g = self.g / magn
        color.b = self.b / magn
        return color

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


class WorldLight:
    def __init__(self):
        self.origin = []
        self.intensity = Color32()
        self.normal = []
        self.shadow_cast_offset = []
        self.cluster = 0
        self.type = []
        self.style = 0
        self.stopdot = 0.0
        self.stopdot2 = 0.0
        self.exponent = 0.0
        self.radius = 0.0
        self.constant_attn = 0.0
        self.linear_attn = 0.0
        self.quadratic_attn = 0.0
        self.flags = 0
        self.texinfo = 0
        self.owner = 0

    def parse(self, reader: ByteIO, version):
        self.origin = reader.read_fmt('3f')
        self.intensity = Color32.from_array(reader.read_fmt('3f'))
        self.normal = reader.read_fmt('3f')
        if version > 20:
            self.shadow_cast_offset = reader.read_fmt('3f')
        self.cluster = reader.read_int32()
        self.type = EmitType(reader.read_int32())
        self.style = reader.read_int32()
        self.stopdot = reader.read_float()
        self.stopdot2 = reader.read_float()
        self.exponent = reader.read_float()
        self.radius = reader.read_float()
        self.constant_attn = reader.read_float()
        self.linear_attn = reader.read_float()
        self.quadratic_attn = reader.read_float()
        self.flags = reader.read_int32()
        self.texinfo = reader.read_int32()
        self.owner = reader.read_int32()
        return self
