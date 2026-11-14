import math

import bpy
import numpy as np

from SourceIO.blender_bindings.ui.export_nodes.nodes.base_node import (
    SourceIOTextureTreeNode)
from SourceIO.library.utils.math_utilities import (
    ensure_f32, linear_to_srgb, srgb_to_linear)


def _is_texture(a):
    """Return True if array looks like HxWx3/4."""
    return isinstance(a, np.ndarray) and a.ndim == 3 and a.shape[-1] in (3, 4)


def _is_channel(a):
    """Return True if array looks like HxW."""
    return isinstance(a, np.ndarray) and a.ndim == 2


def _clamp01(a):
    """Clamp to [0,1]."""
    return np.clip(a, 0.0, 1.0)


def _apply_bc(x, brightness, contrast):
    """Brightness/Contrast in [0,1] domain."""
    return (x - 0.5) * (contrast + 1.0) + 0.5 + brightness


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """Return smoothstep(x) with x in 0..1."""
    return x * x * (3.0 - 2.0 * x)


def _smootherstep(x: np.ndarray) -> np.ndarray:
    """Return smootherstep(x) with x in 0..1."""
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _normalized(v):
    """Scale vectors along the last axis to unit length."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def _rgb_to_unit_vec(tex):
    """Map RGB 0..1 to unit-length XYZ in [-1,1]."""
    return _normalized(ensure_f32(tex[..., :3]) * 2.0 - 1.0)


def _unit_vec_to_rgb(v, alpha=None):
    """Map unit-length XYZ [-1,1] to RGB(A) 0..1."""
    rgb = (v + 1.0) * 0.5
    if alpha is None:
        a = np.ones_like(rgb[..., :1], dtype=np.float32)
    else:
        a = alpha[..., None] if alpha.ndim == 2 else alpha[..., 3:4]
    return np.concatenate((rgb, a), axis=-1)


def _with_alpha(texture):
    """An RGBA copy of an RGB texture; RGBA textures are returned as is."""
    if texture.shape[-1] == 4:
        return texture
    alpha = np.ones(texture.shape[:2] + (1,), dtype=texture.dtype)
    return np.concatenate((texture, alpha), axis=-1)


def _vec_to_latlong(v):
    """Return (lat, lon) from XYZ where lat=asin(z), lon=atan2(y,x)."""
    lon = np.arctan2(v[..., 1], v[..., 0])
    lat = np.arcsin(np.clip(v[..., 2], -1.0, 1.0))
    return lat, lon


def _latlong_to_vec(lat, lon):
    """Return XYZ from (lat, lon)."""
    cl = np.cos(lat)
    x = np.cos(lon) * cl
    y = np.sin(lon) * cl
    z = np.sin(lat)
    return _normalized(np.stack((x, y, z), axis=-1))


def _blend_angle(a0, a1, t):
    """Weighted circular blend of angles using vector sum."""
    c = (1.0 - t) * np.cos(a0) + t * np.cos(a1)
    s = (1.0 - t) * np.sin(a0) + t * np.sin(a1)
    return np.arctan2(s, c)


class SourceIOTextureInvertChannelsNode(SourceIOTextureTreeNode):
    bl_idname = "SourceIOTextureInvertChannelsNode"
    bl_label = "Invert Channels"

    invert_r: bpy.props.BoolProperty(name="Invert R", default=False)
    invert_b: bpy.props.BoolProperty(name="Invert B", default=False)
    invert_g: bpy.props.BoolProperty(name="Invert G", default=False)
    invert_a: bpy.props.BoolProperty(name="Invert A", default=False)

    def draw_buttons(self, context, layout):
        layout.prop(self, "invert_r")
        layout.prop(self, "invert_g")
        layout.prop(self, "invert_b")
        layout.prop(self, "invert_a")

    def init(self, context):
        self.inputs.new('SourceIOTextureSocket', "texture")
        self.outputs.new('SourceIOTextureSocket', "texture")

    def process(self, inputs: dict) -> dict | None:
        img_data = inputs.get("texture")
        if not _is_texture(img_data):
            return None
        has_alpha = img_data.shape[-1] == 4
        if self.invert_a and not has_alpha:
            result = np.empty((*img_data.shape[:2], 4), dtype=img_data.dtype)
            result[..., :3] = img_data
            # Invert the implicit opaque alpha of RGB inputs.
            result[..., 3] = 0.0
            img_data = result
        else:
            img_data = img_data.copy()
        if self.invert_r:
            img_data[..., 0] = 1.0 - img_data[..., 0]
        if self.invert_g:
            img_data[..., 1] = 1.0 - img_data[..., 1]
        if self.invert_b:
            img_data[..., 2] = 1.0 - img_data[..., 2]
        if self.invert_a and has_alpha:
            img_data[..., 3] = 1.0 - img_data[..., 3]

        return {"texture": img_data}


class SourceIOTextureSplitChannelsNode(SourceIOTextureTreeNode):
    bl_idname = "SourceIOTextureSplitChannelsNode"
    bl_label = "Split Channels"

    def init(self, context):
        self.inputs.new('SourceIOTextureSocket', "texture")
        self.outputs.new('SourceIOTextureChannelSocket', "R")
        self.outputs.new('SourceIOTextureChannelSocket', "G")
        self.outputs.new('SourceIOTextureChannelSocket', "B")
        self.outputs.new('SourceIOTextureChannelSocket', "A")

    def process(self, inputs: dict) -> dict | None:
        img_data = inputs.get("texture")
        if not _is_texture(img_data):
            return None
        if img_data.shape[-1] == 4:
            alpha = img_data[..., 3].copy()
        else:
            alpha = np.ones(img_data.shape[:2], dtype=img_data.dtype)
        return {
            "R": img_data[..., 0].copy(),
            "G": img_data[..., 1].copy(),
            "B": img_data[..., 2].copy(),
            "A": alpha,
        }


class SourceIOTextureCombineChannelsNode(SourceIOTextureTreeNode):
    bl_idname = "SourceIOTextureCombineChannelsNode"
    bl_label = "Combine Channels"

    def init(self, context):
        for channel in "RGBA":
            socket = self.inputs.new('SourceIOTextureChannelSocket', channel)
            if channel == 'A':
                socket.default_value = 1.0

        self.outputs.new('SourceIOTextureSocket', "texture")

    def process(self, inputs: dict[str, np.ndarray]) -> dict | None:
        arrays = [value for value in inputs.values()
                  if isinstance(value, np.ndarray)]
        if not arrays or arrays[0].ndim != 2:
            return None
        if any(array.shape != arrays[0].shape for array in arrays):
            return None
        height, width = arrays[0].shape

        rgba = np.zeros((height, width, 4), np.float32)

        if "R" in inputs:
            rgba[..., 0] = inputs["R"]
        if "G" in inputs:
            rgba[..., 1] = inputs["G"]
        if "B" in inputs:
            rgba[..., 2] = inputs["B"]
        if "A" in inputs:
            rgba[..., 3] = inputs["A"]

        return {"texture": rgba}


class SourceIOTextureBrightnessContrastNode(SourceIOTextureTreeNode):
    bl_idname = "SourceIOTextureBrightnessContrastNode"
    bl_label = "Brightness/Contrast"

    input_mode: bpy.props.EnumProperty(
        name="Input",
        items=[('AUTO', 'Auto', ''), ('TEXTURE', 'Texture', ''),
               ('CHANNEL', 'Channel', '')],
        default='AUTO',
        update=lambda self, ctx: self._sync_io_visibility()
    )
    brightness: bpy.props.FloatProperty(name="Brightness", default=0.0,
                                        soft_min=-1.0, soft_max=1.0)
    contrast: bpy.props.FloatProperty(name="Contrast", default=0.0,
                                      soft_min=-1.0, soft_max=1.0)
    affect_alpha: bpy.props.BoolProperty(name="Affect Alpha", default=False)
    clamp_result: bpy.props.BoolProperty(name="Clamp 0..1", default=True)

    def _resolve_kind(self):
        """Return 'TEXTURE', 'CHANNEL', or None based on mode and links."""
        tex = self.inputs.get("texture")
        ch = self.inputs.get("channel")
        if self.input_mode == 'TEXTURE':
            return 'TEXTURE'
        if self.input_mode == 'CHANNEL':
            return 'CHANNEL'
        if tex and tex.is_linked:
            return 'TEXTURE'
        if ch and ch.is_linked:
            return 'CHANNEL'
        return None

    def _sync_io_visibility(self):
        """Hide/show inputs and outputs according to resolved kind."""
        tex = self.inputs.get("texture")
        ch = self.inputs.get("channel")
        otex = self.outputs.get("texture")
        och = self.outputs.get("channel")
        kind = self._resolve_kind()
        if tex and ch:
            if self.input_mode == 'TEXTURE':
                tex.hide, ch.hide = False, not ch.is_linked
            elif self.input_mode == 'CHANNEL':
                tex.hide, ch.hide = not tex.is_linked, False
            else:
                tex.hide, ch.hide = False, False
        if otex and och:
            if kind == 'TEXTURE':
                otex.hide, och.hide = False, True
            elif kind == 'CHANNEL':
                otex.hide, och.hide = True, False
            else:
                otex.hide, och.hide = False, False

    def draw_buttons(self, context, layout):
        """Draw UI."""
        layout.prop(self, "input_mode", text="")
        layout.prop(self, "brightness")
        layout.prop(self, "contrast")
        layout.prop(self, "affect_alpha")
        layout.prop(self, "clamp_result")

    def init(self, context):
        """Create sockets."""
        self.inputs.new('SourceIOTextureSocket', "texture")
        self.inputs.new('SourceIOTextureChannelSocket', "channel")
        self.outputs.new('SourceIOTextureSocket', "texture")
        self.outputs.new('SourceIOTextureChannelSocket', "channel")
        self._sync_io_visibility()

    def process(self, inputs: dict) -> dict | None:
        """Apply brightness/contrast to the selected input and output type."""
        kind = self._resolve_kind()
        if kind == 'TEXTURE':
            tex = inputs.get("texture")
            if not _is_texture(tex):
                return None
            img = ensure_f32(tex.copy())
            if self.affect_alpha:
                img = _apply_bc(img, self.brightness, self.contrast)
            else:
                img[..., :3] = _apply_bc(img[..., :3], self.brightness,
                                         self.contrast)
            if self.clamp_result:
                img = _clamp01(img)
            return {"texture": img}
        if kind == 'CHANNEL':
            ch = inputs.get("channel")
            if not _is_channel(ch):
                return None
            value = _apply_bc(ensure_f32(ch.copy()), self.brightness,
                              self.contrast)
            if self.clamp_result:
                value = _clamp01(value)
            return {"channel": value}
        return None


class SourceIOTextureMathNode(SourceIOTextureTreeNode):
    bl_idname = "SourceIOTextureMathNode"
    bl_label = "Math"

    operation: bpy.props.EnumProperty(
        name="Operation",
        items=[
            ('ADD', 'Add', ''), ('SUB', 'Subtract', ''),
            ('MUL', 'Multiply', ''), ('DIV', 'Divide', ''),
            ('POW', 'Power', ''), ('MIN', 'Min', ''), ('MAX', 'Max', ''),
            ('ABS', 'Abs', ''), ('NEG', 'Negate', ''), ('INV', 'Invert', ''),
            ('LERP', 'Lerp', ''),
        ],
        default='ADD'
    )
    factor: bpy.props.FloatProperty(name="Factor", default=0.5,
                                    min=0.0, max=1.0)
    clamp_result: bpy.props.BoolProperty(name="Clamp 0..1", default=False)

    def _sync_io_visibility(self):
        """Show the result type selected by the linked input shapes."""
        at = self.inputs.get("A texture")
        ac = self.inputs.get("A channel")
        bt = self.inputs.get("B texture")
        bc = self.inputs.get("B channel")
        otex = self.outputs.get("texture")
        och = self.outputs.get("channel")
        if any(socket is None for socket in (at, ac, bt, bc, otex, och)):
            return

        at.hide = ac.is_linked and not at.is_linked
        ac.hide = at.is_linked and not ac.is_linked
        bt.hide = bc.is_linked and not bt.is_linked
        bc.hide = bt.is_linked and not bc.is_linked
        has_texture = at.is_linked or bt.is_linked
        has_channel = ac.is_linked or bc.is_linked
        otex.hide = has_channel and not has_texture
        och.hide = has_texture

    def draw_buttons(self, context, layout):
        """Draw UI."""
        layout.prop(self, "operation")
        if self.operation == 'LERP':
            layout.prop(self, "factor")
        layout.prop(self, "clamp_result")

    def init(self, context):
        """Create sockets."""
        self.inputs.new('SourceIOTextureSocket', "A texture")
        self.inputs.new('SourceIOTextureChannelSocket', "A channel")
        self.inputs.new('SourceIOTextureSocket', "B texture")
        self.inputs.new('SourceIOTextureChannelSocket', "B channel")
        self.outputs.new('SourceIOTextureSocket', "texture")
        self.outputs.new('SourceIOTextureChannelSocket', "channel")
        self._sync_io_visibility()

    @staticmethod
    def _pick(inputs: dict,
              operand: str) -> tuple[str | None, np.ndarray | None]:
        """Return the ``A``/``B`` operand as (kind, array), texture first."""
        texture = inputs.get(f"{operand} texture")
        if _is_texture(texture):
            return "texture", texture
        channel = inputs.get(f"{operand} channel")
        if _is_channel(channel):
            return "channel", channel
        return None, None

    def process(self, inputs: dict) -> dict | None:
        """Combine A and B per pixel; a channel applies to every component.

        An unlinked B uses the B channel socket value.
        """
        kind_a, a = self._pick(inputs, "A")
        if a is None:
            return None
        kind_b, b = self._pick(inputs, "B")
        if b is None:
            b = float(self.inputs["B channel"].default_value)
        elif a.shape[:2] != b.shape[:2]:
            return None
        kind = "texture" if "texture" in (kind_a, kind_b) else "channel"
        a = ensure_f32(a)
        if isinstance(b, np.ndarray):
            b = ensure_f32(b)
        if kind == "texture":
            if kind_a == "channel":
                a = a[..., None]
                if self.operation in {'ABS', 'NEG', 'INV'}:
                    a = np.broadcast_to(a, b.shape)
            if kind_b == "channel":
                b = b[..., None]
            elif (kind_a == kind_b == "texture"
                  and a.shape[-1] != b.shape[-1]):
                # RGB with RGBA: the RGB side gets an opaque alpha.
                a, b = (_with_alpha(item) for item in (a, b))

        if self.operation == 'ADD':
            r = a + b
        elif self.operation == 'SUB':
            r = a - b
        elif self.operation == 'MUL':
            r = a * b
        elif self.operation == 'DIV':
            r = a / np.maximum(b, 1e-6)
        elif self.operation == 'POW':
            r = np.power(np.maximum(a, 0.0), b)
        elif self.operation == 'MIN':
            r = np.minimum(a, b)
        elif self.operation == 'MAX':
            r = np.maximum(a, b)
        elif self.operation == 'ABS':
            r = np.abs(a)
        elif self.operation == 'NEG':
            r = -a
        elif self.operation == 'INV':
            r = 1.0 - a
        elif self.operation == 'LERP':
            t = float(self.factor)
            r = a * (1.0 - t) + b * t
        else:
            return None
        if self.clamp_result:
            r = _clamp01(r)
        return {kind: r}


class SourceIOChannelRemapNode(SourceIOTextureTreeNode):
    bl_idname = "SourceIOChannelRemapNode"
    bl_label = "Remap (Channel)"

    in_min: bpy.props.FloatProperty(name="In Min", default=0.0,
                                    soft_min=-10.0, soft_max=10.0)
    in_max: bpy.props.FloatProperty(name="In Max", default=1.0,
                                    soft_min=-10.0, soft_max=10.0)
    out_min: bpy.props.FloatProperty(name="Out Min", default=0.0,
                                     soft_min=-10.0, soft_max=10.0)
    out_max: bpy.props.FloatProperty(name="Out Max", default=1.0,
                                     soft_min=-10.0, soft_max=10.0)
    gamma: bpy.props.FloatProperty(name="Gamma", default=1.0, min=1e-6,
                                   soft_min=0.1, soft_max=4.0)
    curve: bpy.props.EnumProperty(
        name="Curve",
        items=[('LINEAR', 'Linear', ''), ('SMOOTHSTEP', 'Smoothstep', ''),
               ('SMOOTHER', 'Smootherstep', '')],
        default='LINEAR'
    )
    clamp_input: bpy.props.BoolProperty(name="Clamp Input 0..1",
                                        default=True)
    clamp_result: bpy.props.BoolProperty(name="Clamp Result 0..1",
                                         default=False)

    def draw_buttons(self, context, layout):
        """Draw the node UI."""
        col = layout.column(align=True)
        row = col.row(align=True)
        row.prop(self, "in_min")
        row.prop(self, "in_max")
        row = col.row(align=True)
        row.prop(self, "out_min")
        row.prop(self, "out_max")
        col.prop(self, "gamma")
        col.prop(self, "curve", text="")
        row = col.row(align=True)
        row.prop(self, "clamp_input")
        row.prop(self, "clamp_result")

    def init(self, context):
        """Create sockets."""
        self.inputs.new('SourceIOTextureChannelSocket', "channel")
        self.outputs.new('SourceIOTextureChannelSocket', "channel")

    def process(self, inputs: dict) -> dict | None:
        """Remap a channel from [In Min, In Max] to [Out Min, Out Max].

        Gamma and the curve shape the normalized value in between.
        """
        ch = inputs.get("channel")
        if not _is_channel(ch):
            return None
        x = ensure_f32(ch.copy())
        # A reversed input range (In Min > In Max) inverts the channel.
        denom = self.in_max - self.in_min
        if abs(denom) < 1e-6:
            denom = math.copysign(1e-6, denom)
        x = (x - self.in_min) / denom
        if self.clamp_input:
            x = np.clip(x, 0.0, 1.0)
        if self.gamma != 1.0:
            x = np.power(np.clip(x, 0.0, 1.0), 1.0 / float(self.gamma))
        if self.curve == 'SMOOTHSTEP':
            x = _smoothstep(np.clip(x, 0.0, 1.0))
        elif self.curve == 'SMOOTHER':
            x = _smootherstep(np.clip(x, 0.0, 1.0))
        y = self.out_min + x * (self.out_max - self.out_min)
        if self.clamp_result:
            y = np.clip(y, 0.0, 1.0)
        return {"channel": y}


class SourceIONormalOpsNode(SourceIOTextureTreeNode):
    bl_idname = "SourceIONormalOpsNode"
    bl_label = "Normal Map Ops"

    flip_green: bpy.props.BoolProperty(name="Flip Green (DX/GL)",
                                       default=False)
    swap_rg: bpy.props.BoolProperty(name="Swap R/G", default=False)
    renormalize: bpy.props.BoolProperty(name="Renormalize", default=True)
    rebuild_z: bpy.props.BoolProperty(name="Rebuild Z", default=False)

    def draw_buttons(self, context, layout):
        layout.prop(self, "flip_green")
        layout.prop(self, "swap_rg")
        layout.prop(self, "renormalize")
        layout.prop(self, "rebuild_z")

    def init(self, context):
        self.inputs.new('SourceIOTextureSocket', "texture")
        self.outputs.new('SourceIOTextureSocket', "texture")

    def process(self, inputs: dict) -> dict | None:
        tex = inputs.get("texture")
        if not _is_texture(tex):
            return None
        img = ensure_f32(tex.copy())
        rgb = img[..., :3]
        if self.flip_green:
            rgb[..., 1] = 1.0 - rgb[..., 1]
        if self.swap_rg:
            r = rgb[..., 0].copy()
            rgb[..., 0] = rgb[..., 1]
            rgb[..., 1] = r
        if self.renormalize or self.rebuild_z:
            v = rgb * 2.0 - 1.0
            if self.rebuild_z:
                x = v[..., 0]
                y = v[..., 1]
                z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
                v = np.stack((x, y, z), axis=-1)
            img[..., :3] = (_normalized(v) + 1.0) * 0.5
        return {"texture": _clamp01(img)}


class SourceIOHeightToNormalNode(SourceIOTextureTreeNode):
    bl_idname = "SourceIOHeightToNormalNode"
    bl_label = "Height → Normal"

    strength: bpy.props.FloatProperty(name="Strength", default=1.0,
                                      soft_min=0.0, soft_max=10.0)
    invert_y: bpy.props.BoolProperty(name="Invert Y", default=False)
    preserve_scale: bpy.props.BoolProperty(name="Preserve Scale",
                                           default=True)

    def draw_buttons(self, context, layout):
        layout.prop(self, "strength")
        layout.prop(self, "preserve_scale")
        layout.prop(self, "invert_y")

    def init(self, context):
        self.inputs.new('SourceIOTextureChannelSocket', "height")
        self.outputs.new('SourceIOTextureSocket', "texture")

    def process(self, inputs: dict) -> dict | None:
        h = inputs.get("height")
        if not _is_channel(h):
            return None
        height = ensure_f32(h)
        dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
        dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5
        s = float(self.strength)
        normal = np.stack((-dx * s, -dy * s, np.ones_like(height)), axis=-1)
        if self.preserve_scale:
            normal = _normalized(normal)
        out = _unit_vec_to_rgb(normal)
        if self.invert_y:
            out[..., 1] = 1.0 - out[..., 1]
        return {"texture": _clamp01(out)}


class SourceIONormalBlendNode(SourceIOTextureTreeNode):
    bl_idname = "SourceIONormalBlendNode"
    bl_label = "Blend Normals"

    method: bpy.props.EnumProperty(
        name="Method",
        items=[
            ('WHITEOUT', 'Whiteout', ''),
            ('OVERLAY', 'Overlay', ''),
            ('LATLONG', 'Lat/Long', '')
        ],
        default='WHITEOUT'
    )

    factor: bpy.props.FloatProperty(name="Factor", default=1.0,
                                    min=0.0, max=1.0)
    renormalize: bpy.props.BoolProperty(name="Renormalize", default=True)

    def draw_buttons(self, context, layout):
        layout.prop(self, "method", text="")
        layout.prop(self, "factor")
        layout.prop(self, "renormalize")

    def init(self, context):
        self.inputs.new('SourceIOTextureSocket', "Base")
        self.inputs.new('SourceIOTextureSocket', "Detail")
        self.outputs.new('SourceIOTextureSocket', "texture")

    def process(self, inputs: dict) -> dict | None:
        base = inputs.get("Base")
        detail = inputs.get("Detail")
        if not (_is_texture(base) and _is_texture(detail)):
            return None
        if base.shape[:2] != detail.shape[:2]:
            return None

        a = _rgb_to_unit_vec(base)
        b = _rgb_to_unit_vec(detail)
        t = self.factor

        if self.method == 'WHITEOUT':
            out = np.empty_like(a)
            out[..., 0] = a[..., 0] + b[..., 0]
            out[..., 1] = a[..., 1] + b[..., 1]
            out[..., 2] = a[..., 2] * b[..., 2]
            out = a * (1.0 - t) + out * t
        elif self.method == 'OVERLAY':
            out = a + b - a * b
            out = a * (1.0 - t) + out * t
        else:  # LATLONG
            lat_a, lon_a = _vec_to_latlong(a)
            lat_b, lon_b = _vec_to_latlong(b)
            lon = _blend_angle(lon_a, lon_b, t)
            lat = _blend_angle(lat_a, lat_b, t)
            out = _latlong_to_vec(lat, lon)

        if self.renormalize:
            out = _normalized(out)

        alpha = ensure_f32(base[..., 3]) if base.shape[-1] == 4 else None
        res = _unit_vec_to_rgb(out, alpha=alpha)
        return {"texture": _clamp01(res)}


class SourceIOColorSpaceNode(SourceIOTextureTreeNode):
    bl_idname = "SourceIOColorSpaceNode"
    bl_label = "Color Space"

    direction: bpy.props.EnumProperty(
        name="Direction",
        items=[('SRGB2LIN', 'sRGB → Linear', ''),
               ('LIN2SRGB', 'Linear → sRGB', '')],
        default='SRGB2LIN'
    )
    clamp_result: bpy.props.BoolProperty(name="Clamp 0..1", default=True)

    def _sync_io_visibility(self):
        t = self.inputs.get("texture")
        c = self.inputs.get("channel")
        ot = self.outputs.get("texture")
        oc = self.outputs.get("channel")
        if any(socket is None for socket in (t, c, ot, oc)):
            return
        t.hide = c.is_linked and not t.is_linked
        c.hide = t.is_linked and not c.is_linked
        kind = self._resolve_kind()
        ot.hide = kind == 'CHANNEL'
        oc.hide = kind == 'TEXTURE'

    def _resolve_kind(self):
        t = self.inputs.get("texture")
        c = self.inputs.get("channel")
        if t and t.is_linked:
            return 'TEXTURE'
        if c and c.is_linked:
            return 'CHANNEL'
        return None

    def draw_buttons(self, context, layout):
        layout.prop(self, "direction", text="")
        layout.prop(self, "clamp_result")

    def init(self, context):
        self.inputs.new('SourceIOTextureSocket', "texture")
        self.inputs.new('SourceIOTextureChannelSocket', "channel")
        self.outputs.new('SourceIOTextureSocket', "texture")
        self.outputs.new('SourceIOTextureChannelSocket', "channel")
        self._sync_io_visibility()

    def _convert(self, values):
        if self.direction == 'SRGB2LIN':
            return srgb_to_linear(values, 1)
        return linear_to_srgb(values, 1)

    def process(self, inputs: dict) -> dict | None:
        kind = self._resolve_kind()
        if kind == 'TEXTURE':
            tex = inputs.get("texture")
            if not _is_texture(tex):
                return None
            img = ensure_f32(tex.copy())
            img[..., :3] = self._convert(img[..., :3])
            if self.clamp_result:
                img = _clamp01(img)
            return {"texture": img}
        if kind == 'CHANNEL':
            ch = inputs.get("channel")
            if not _is_channel(ch):
                return None
            v = self._convert(ensure_f32(ch.copy()))
            if self.clamp_result:
                v = _clamp01(v)
            return {"channel": v}
        return None


def _channel_like(value, shape, label):
    """Broadcast an unlinked channel default (float) or validate a linked
    HxW channel.
    """
    if isinstance(value, np.ndarray):
        if value.ndim != 2 or value.shape != shape:
            raise ValueError(f"PBR to Phong: {label} is {value.shape[:2]}, "
                             f"normal map is {shape}")
        return ensure_f32(value)
    return np.full(shape, float(value if value is not None else 0.0),
                   np.float32)


class SourceIOPBRToPhongNode(SourceIOTextureTreeNode):
    """Roughness/metalness + normal map -> VertexLitGeneric
    $phongexponenttexture and $bumpmap.

    Phong exponent texture: R = exponent ((n - 1) / 149, Source maps R to
    1..150), G = albedo tint mask (metalness, used with $phongalbedotint),
    B = 0, A = rim light mask.
    Normal output: the normal map RGB with the specular mask in alpha (VLG
    reads the phong mask from the bumpmap alpha; $normalmapalphaenvmapmask
    also uses it as the envmap mask).
    """
    bl_idname = "SourceIOPBRToPhongNode"
    bl_label = "PBR to Phong"

    exponent_mode: bpy.props.EnumProperty(name="Exponent", items=(
        ("PHYSICAL", "GGX matched",
         "Blinn-Phong exponent matching GGX: n = 2 / roughness^4 - 2"),
        ("LINEAR", "Linear gloss", "Exponent channel = 1 - roughness"),
    ), default="PHYSICAL")
    max_exponent: bpy.props.FloatProperty(
        name="Max exponent", default=150.0, min=1.0, max=150.0,
        description="Exponent written for perfectly smooth texels")
    dielectric_mask: bpy.props.FloatProperty(
        name="Dielectric specular", default=0.25, min=0.0, max=1.0,
        description="Specular mask of non-metals (metals use 1)")
    mask_gloss_power: bpy.props.FloatProperty(
        name="Mask gloss power", default=1.0, min=0.0,
        description="Mask is multiplied by (1 - roughness) ^ power")

    def init(self, context):
        self.width = 200
        self.inputs.new('SourceIOTextureChannelSocket',
                        "Roughness").default_value = 0.5
        self.inputs.new('SourceIOTextureChannelSocket',
                        "Metalness").default_value = 0.0
        self.inputs.new('SourceIOTextureSocket', "Normal")
        self.inputs.new('SourceIOTextureChannelSocket',
                        "Rim mask").default_value = 1.0
        self.outputs.new('SourceIOTextureSocket', "Phong exponent")
        self.outputs.new('SourceIOTextureSocket', "Normal")
        self.outputs.new('SourceIOTextureChannelSocket', "Specular mask")

    def draw_buttons(self, context, layout):
        layout.prop(self, "exponent_mode")
        if self.exponent_mode == "PHYSICAL":
            layout.prop(self, "max_exponent")
        layout.prop(self, "dielectric_mask")
        layout.prop(self, "mask_gloss_power")

    def process(self, inputs: dict) -> dict | None:
        normal = inputs.get("Normal")
        if not _is_texture(normal):
            return None
        shape = normal.shape[:2]
        roughness = _clamp01(_channel_like(inputs.get("Roughness"), shape,
                                           "Roughness"))
        metalness = _clamp01(_channel_like(inputs.get("Metalness"), shape,
                                           "Metalness"))
        rim = _clamp01(_channel_like(inputs.get("Rim mask"), shape,
                                     "Rim mask"))
        gloss = 1.0 - roughness
        if self.exponent_mode == "PHYSICAL":
            alpha = np.maximum(roughness * roughness, 1e-4)
            exponent = np.clip(2.0 / (alpha * alpha) - 2.0, 1.0,
                               self.max_exponent)
            exponent_channel = (exponent - 1.0) / 149.0
        else:
            exponent_channel = gloss
        phong = np.zeros((*shape, 4), np.float32)
        phong[..., 0] = exponent_channel
        phong[..., 1] = metalness
        phong[..., 3] = rim
        specular = (self.dielectric_mask
                    + (1.0 - self.dielectric_mask) * metalness)
        mask = specular * gloss ** self.mask_gloss_power
        mask = _clamp01(mask).astype(np.float32)
        bump = np.empty((*shape, 4), np.float32)
        bump[..., :3] = ensure_f32(normal[..., :3])
        bump[..., 3] = mask
        return {"Phong exponent": phong, "Normal": bump,
                "Specular mask": mask}
