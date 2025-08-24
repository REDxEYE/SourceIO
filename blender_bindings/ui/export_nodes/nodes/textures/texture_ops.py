import bpy
import numpy as np
from bpy.types import Node

from SourceIO.blender_bindings.ui.export_nodes.nodes.base_node import SourceIOTextureTreeNode


def _is_texture(a):
    """Return True if array looks like HxWx3/4."""
    return isinstance(a, np.ndarray) and a.ndim == 3 and a.shape[-1] in (3, 4)


def _is_channel(a):
    """Return True if array looks like HxW."""
    return isinstance(a, np.ndarray) and a.ndim == 2


def _ensure_f32(a):
    """Return a contiguous float32 view/copy."""
    return np.ascontiguousarray(a, dtype=np.float32)


def _clamp01(a):
    """Clamp to [0,1]."""
    return np.clip(a, 0.0, 1.0)


def _broadcast_to_texture(x, shape_tex):
    """Broadcast scalar or HxW channel to HxWx4 texture with alpha=1."""
    if np.isscalar(x):
        return np.full(shape_tex, float(x), dtype=np.float32)
    if _is_channel(x):
        out = np.zeros(shape_tex, dtype=np.float32)
        out[..., 0] = x
        out[..., 1] = x
        out[..., 2] = x
        out[..., 3] = 1.0
        return out
    return _ensure_f32(x)


def _broadcast_to_channel(x, shape_ch):
    """Broadcast scalar or texture to HxW using luminance."""
    if np.isscalar(x):
        return np.full(shape_ch, float(x), dtype=np.float32)
    if _is_channel(x):
        return _ensure_f32(x)
    if _is_texture(x):
        x = _ensure_f32(x)
        return (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]).astype(np.float32)
    return _ensure_f32(x)


def _apply_bc(x, brightness, contrast):
    """Brightness/Contrast in [0,1] domain."""
    return (x - 0.5) * (contrast + 1.0) + 0.5 + brightness


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """Return smoothstep(x) with x in 0..1."""
    return x * x * (3.0 - 2.0 * x)


def _smootherstep(x: np.ndarray) -> np.ndarray:
    """Return smootherstep(x) with x in 0..1."""
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _srgb_to_linear(x):
    """Per-element sRGB->Linear 0..1."""
    x = _ensure_f32(x)
    a = x <= 0.04045
    y = np.empty_like(x, dtype=np.float32)
    y[a] = x[a] / 12.92
    y[~a] = ((x[~a] + 0.055) / 1.055) ** 2.4
    return y


def _linear_to_srgb(x):
    """Per-element Linear->sRGB 0..1."""
    x = _ensure_f32(x)
    a = x <= 0.0031308
    y = np.empty_like(x, dtype=np.float32)
    y[a] = x[a] * 12.92
    y[~a] = 1.055 * (x[~a] ** (1.0 / 2.4)) - 0.055
    return y

def _rgb_to_unit_vec(tex):
    """Map RGB 0..1 to unit-length XYZ in [-1,1]."""
    v = _ensure_f32(tex[..., :3]) * 2.0 - 1.0
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-8)
    return v / n

def _unit_vec_to_rgb(v, alpha=None):
    """Map unit-length XYZ [-1,1] to RGB(A) 0..1."""
    rgb = (v + 1.0) * 0.5
    if alpha is None:
        a = np.ones_like(rgb[..., :1], dtype=np.float32)
    else:
        a = alpha[..., None] if alpha.ndim == 2 else alpha[..., 3:4]
    return np.concatenate((rgb, a), axis=-1)

def _vec_to_latlong(v):
    """Return (lat, lon) from XYZ where lat=asin(z), lon=atan2(y,x)."""
    x = v[..., 0]; y = v[..., 1]; z = v[..., 2]
    lon = np.arctan2(y, x)
    lat = np.arcsin(np.clip(z, -1.0, 1.0))
    return lat, lon

def _latlong_to_vec(lat, lon):
    """Return XYZ from (lat, lon)."""
    cl = np.cos(lat)
    x = np.cos(lon) * cl
    y = np.sin(lon) * cl
    z = np.sin(lat)
    v = np.stack((x, y, z), axis=-1)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-8)
    return v / n

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
        if "texture" not in inputs:
            return None
        img_data: np.ndarray = inputs["texture"]
        if img_data is None:
            return None

        if self.invert_r:
            img_data[..., 0] = 1.0 - img_data[..., 0]
        if self.invert_g:
            img_data[..., 1] = 1.0 - img_data[..., 1]
        if self.invert_b:
            img_data[..., 2] = 1.0 - img_data[..., 2]
        if self.invert_a:
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
        if "texture" not in inputs:
            return None
        img_data: np.ndarray = inputs["texture"]
        if img_data is None:
            return None

        return {
            "R": img_data[..., 0].copy(),
            "G": img_data[..., 1].copy(),
            "B": img_data[..., 2].copy(),
            "A": img_data[..., 3].copy()
        }


class SourceIOTextureCombineChannelsNode(SourceIOTextureTreeNode):
    bl_idname = "SourceIOTextureCombineChannelsNode"
    bl_label = "Combine Channels"

    def init(self, context):
        for c in "RGBA":
            socket = self.inputs.new('SourceIOTextureChannelSocket', c)

        self.outputs.new('SourceIOTextureSocket', "texture")

    def process(self, inputs: dict[str, np.ndarray]) -> dict | None:
        any_channel = next(filter(lambda a: a is not None and isinstance(a, np.ndarray), inputs.values()), None)

        if any_channel is None:
            return None

        for channel in inputs.values():
            if isinstance(channel, np.ndarray):
                if any_channel is not None and channel.shape != any_channel.shape:
                    return None

        if len(any_channel.shape) != 2:
            return None
        height, width = any_channel.shape

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
        items=[('AUTO', 'Auto', ''), ('TEXTURE', 'Texture', ''), ('CHANNEL', 'Channel', '')],
        default='AUTO',
        update=lambda self, ctx: self._sync_io_visibility()
    )
    brightness: bpy.props.FloatProperty(name="Brightness", default=0.0, soft_min=-1.0, soft_max=1.0)
    contrast: bpy.props.FloatProperty(name="Contrast", default=0.0, soft_min=-1.0, soft_max=1.0)
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
                tex.hide, ch.hide = False, True
            elif self.input_mode == 'CHANNEL':
                tex.hide, ch.hide = True, False
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
        """Apply brightness/contrast to chosen kind."""
        tex = inputs.get("texture")
        ch = inputs.get("channel")
        kind = self._resolve_kind()
        if kind == 'TEXTURE' and _is_texture(tex):
            img = _ensure_f32(tex.copy())
            if self.affect_alpha:
                img = _apply_bc(img, self.brightness, self.contrast)
            else:
                img[..., :3] = _apply_bc(img[..., :3], self.brightness, self.contrast)
            if self.clamp_result: img = _clamp01(img)
            return {"texture": img}
        if kind == 'CHANNEL' and _is_channel(ch):
            v = _apply_bc(_ensure_f32(ch.copy()), self.brightness, self.contrast)
            if self.clamp_result: v = _clamp01(v)
            return {"channel": v}
        if _is_texture(tex):
            img = _ensure_f32(tex.copy())
            if self.affect_alpha:
                img = _apply_bc(img, self.brightness, self.contrast)
            else:
                img[..., :3] = _apply_bc(img[..., :3], self.brightness, self.contrast)
            if self.clamp_result: img = _clamp01(img)
            return {"texture": img}
        if _is_channel(ch):
            v = _apply_bc(_ensure_f32(ch.copy()), self.brightness, self.contrast)
            if self.clamp_result: v = _clamp01(v)
            return {"channel": v}
        return None


class SourceIOTextureMathNode(SourceIOTextureTreeNode):
    bl_idname = "SourceIOTextureMathNode"
    bl_label = "Math"

    operation: bpy.props.EnumProperty(
        name="Operation",
        items=[
            ('ADD', 'Add', ''), ('SUB', 'Subtract', ''), ('MUL', 'Multiply', ''), ('DIV', 'Divide', ''),
            ('POW', 'Power', ''), ('MIN', 'Min', ''), ('MAX', 'Max', ''), ('ABS', 'Abs', ''),
            ('NEG', 'Negate', ''), ('INV', 'Invert', ''), ('LERP', 'Lerp', ''),
        ],
        default='ADD'
    )
    factor: bpy.props.FloatProperty(name="Factor", default=0.5, min=0.0, max=1.0)
    clamp_result: bpy.props.BoolProperty(name="Clamp 0..1", default=False)

    def _sync_io_visibility(self):
        """Hide/show inputs for A/B and outputs according to A's resolved kind."""
        at = self.inputs.get("A texture")
        ac = self.inputs.get("A channel")
        bt = self.inputs.get("B texture")
        bc = self.inputs.get("B channel")
        otex = self.outputs.get("texture")
        och = self.outputs.get("channel")

        print(f"AC->{ac.is_linked}, AT->{at.is_linked}, BC->{bc.is_linked}, BT->{bt.is_linked}")

        if at.is_linked:
            at.hide, ac.hide = False, True
        elif ac.is_linked:
            at.hide, ac.hide = True, False
        else:
            at.hide, ac.hide = False, False

        if bt.is_linked and ac.is_linked:
            bt.hide, bc.hide = False, True
        elif bc.is_linked or ac.is_linked:
            bt.hide, bc.hide = True, False
        else:
            bt.hide, bc.hide = False, False

        if at.is_linked or bt.is_linked:
            otex.hide, och.hide = False, True
        elif ac.is_linked:
            otex.hide, och.hide = True, False
        else:
            otex.hide, och.hide = False, False

    def draw_buttons(self, context, layout):
        """Draw UI."""
        r = layout.row(align=True)
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

    def _pick_a(self, inputs: dict) -> tuple[str | None, np.ndarray | None]:
        """Resolve A value by mode and links."""
        at = inputs.get("A texture", None)
        ac = inputs.get("A channel", None)

        if at is not None and _is_texture(at):
            return "texture", at
        if ac is not None and _is_channel(ac):
            return "channel", ac

        return None, None

    def _pick_b(self, inputs) -> tuple[str | None, np.ndarray | None]:
        """Resolve B value by mode and links."""
        bt = inputs.get("B texture", None)
        bc = inputs.get("B channel", None)

        if bt is not None and _is_texture(bt):
            return "texture", bt
        if bc is not None and _is_channel(bc):
            return "channel", bc
        return None, None

    def process(self, inputs: dict) -> dict | None:
        """Apply math with outputs following a's kind."""
        kind_a, a = self._pick_a(inputs)
        if a is None:
            return None
        kind_b, b = self._pick_b(inputs)
        if b is None:
            b = np.full((a.shape[:2]), self.inputs["B channel"].default_value, np.float32)
        else:
            if a.shape[:2] != b.shape[:2]:
                return None
            b = _ensure_f32(b.copy())

        a = _ensure_f32(a.copy())

        if kind_a == "texture":
            b = _broadcast_to_texture(b, a.shape)

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
        if kind_a == 'texture':
            return {"texture": r}
        elif kind_a == 'channel':
            if r.ndim == 3:
                raise ValueError("Internal error: expected channel but got texture")
            return {"channel": r}
        return None


class SourceIOChannelRemapNode(SourceIOTextureTreeNode):
    bl_idname = "SourceIOChannelRemapNode"
    bl_label = "Remap (Channel)"

    in_min: bpy.props.FloatProperty(name="In Min", default=0.0, soft_min=-10.0, soft_max=10.0)
    in_max: bpy.props.FloatProperty(name="In Max", default=1.0, soft_min=-10.0, soft_max=10.0)
    out_min: bpy.props.FloatProperty(name="Out Min", default=0.0, soft_min=-10.0, soft_max=10.0)
    out_max: bpy.props.FloatProperty(name="Out Max", default=1.0, soft_min=-10.0, soft_max=10.0)
    gamma: bpy.props.FloatProperty(name="Gamma", default=1.0, min=1e-6, soft_min=0.1, soft_max=4.0)
    curve: bpy.props.EnumProperty(
        name="Curve",
        items=[('LINEAR', 'Linear', ''), ('SMOOTHSTEP', 'Smoothstep', ''), ('SMOOTHER', 'Smootherstep', '')],
        default='LINEAR'
    )
    clamp_input: bpy.props.BoolProperty(name="Clamp Input 0..1", default=True)
    clamp_result: bpy.props.BoolProperty(name="Clamp Result 0..1", default=False)

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
        s_in = self.inputs.new('SourceIOTextureChannelSocket', "channel")
        s_out = self.outputs.new('SourceIOTextureChannelSocket', "channel")

    def process(self, inputs: dict) -> dict | None:
        """Remap a channel from [In Min, In Max] to [Out Min, Out Max] with optional gamma and curve."""
        ch = inputs.get("channel")
        if not isinstance(ch, np.ndarray) or ch.ndim != 2:
            return None
        x = _ensure_f32(ch.copy())
        denom = max(self.in_max - self.in_min, 1e-6)
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

    flip_green: bpy.props.BoolProperty(name="Flip Green (DX/GL)", default=False)
    swap_rg: bpy.props.BoolProperty(name="Swap R/G", default=False)
    renormalize: bpy.props.BoolProperty(name="Renormalize", default=True)
    rebuild_z: bpy.props.BoolProperty(name="Rebuild Z", default=False)

    def draw_buttons(self, context, layout):
        layout.prop(self, "flip_green")
        layout.prop(self, "swap_rg")
        layout.prop(self, "renormalize")
        layout.prop(self, "rebuild_z")

    def init(self, context):
        i_tex = self.inputs.new('SourceIOTextureSocket', "texture")
        o_tex = self.outputs.new('SourceIOTextureSocket', "texture")

    def process(self, inputs: dict) -> dict | None:
        tex = inputs.get("texture")
        if not isinstance(tex, np.ndarray) or tex.ndim != 3 or tex.shape[-1] not in (3, 4):
            return None
        img = _ensure_f32(tex.copy())
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
            n = np.linalg.norm(v, axis=-1, keepdims=True)
            n = np.maximum(n, 1e-8)
            v = v / n
            rgb = v * 2.0 - 0.0
            rgb = (v + 1.0) * 0.5
            img[..., :3] = rgb
        return {"texture": _clamp01(img)}


class SourceIOHeightToNormalNode(SourceIOTextureTreeNode):
    bl_idname = "SourceIOHeightToNormalNode"
    bl_label = "Height → Normal"

    strength: bpy.props.FloatProperty(name="Strength", default=1.0, soft_min=0.0, soft_max=10.0)
    invert_y: bpy.props.BoolProperty(name="Invert Y", default=False)
    preserve_scale: bpy.props.BoolProperty(name="Preserve Scale", default=True)

    def draw_buttons(self, context, layout):
        layout.prop(self, "strength")
        layout.prop(self, "preserve_scale")

    def init(self, context):
        self.inputs.new('SourceIOTextureChannelSocket', "height")
        self.outputs.new('SourceIOTextureSocket', "texture")

    def process(self, inputs: dict) -> dict | None:
        h = inputs.get("height")
        if not isinstance(h, np.ndarray) or h.ndim != 2:
            return None
        H = _ensure_f32(h)
        dx = (np.roll(H, -1, axis=1) - np.roll(H, 1, axis=1)) * 0.5
        dy = (np.roll(H, -1, axis=0) - np.roll(H, 1, axis=0)) * 0.5
        s = float(self.strength)
        nx = -dx * s
        ny = -dy * s
        nz = np.ones_like(H)
        if self.preserve_scale:
            l = np.sqrt(nx * nx + ny * ny + nz * nz)
            l = np.maximum(l, 1e-8)
            nx /= l
            ny /= l
            nz /= l
        rgb = np.stack(((nx + 1.0) * 0.5, (ny + 1.0) * 0.5, (nz + 1.0) * 0.5), axis=-1)
        a = np.ones_like(H)
        out = np.concatenate((rgb, a[..., None]), axis=-1)
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
            ('OVERLAY',  'Overlay',  ''),
            ('LATLONG',  'Lat/Long', '')
        ],
        default='WHITEOUT'
    )

    factor: bpy.props.FloatProperty(name="Factor", default=1.0, min=0.0, max=1.0)
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
        n1 = inputs.get("Base"); n2 = inputs.get("Detail")
        if not (isinstance(n1, np.ndarray) and isinstance(n2, np.ndarray)):
            return None
        if n1.ndim != 3 or n2.ndim != 3 or n1.shape[-1] < 3 or n2.shape[-1] < 3:
            return None
        if n1.shape[0] != n2.shape[0] or n1.shape[1] != n2.shape[1]:
            return None

        A_img = _ensure_f32(n1.copy())
        B_img = _ensure_f32(n2)
        a = _rgb_to_unit_vec(A_img)
        b = _rgb_to_unit_vec(B_img)
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
            latA, lonA = _vec_to_latlong(a)
            latB, lonB = _vec_to_latlong(b)
            lon = _blend_angle(lonA, lonB, t)
            lat = _blend_angle(latA, latB, t)
            out = _latlong_to_vec(lat, lon)

        if self.renormalize:
            l = np.linalg.norm(out, axis=-1, keepdims=True)
            l = np.maximum(l, 1e-8)
            out = out / l

        alpha = A_img[..., 3] if A_img.shape[-1] == 4 else None
        res = _unit_vec_to_rgb(out, alpha=alpha)
        return {"texture": _clamp01(res)}


class SourceIOColorSpaceNode(SourceIOTextureTreeNode):
    bl_idname = "SourceIOColorSpaceNode"
    bl_label = "Color Space"

    direction: bpy.props.EnumProperty(
        name="Direction",
        items=[('SRGB2LIN', 'sRGB → Linear', ''), ('LIN2SRGB', 'Linear → sRGB', '')],
        default='SRGB2LIN'
    )
    clamp_result: bpy.props.BoolProperty(name="Clamp 0..1", default=True)

    def _sync_io_visibility(self):
        t = self.inputs.get("texture")
        c = self.inputs.get("channel")
        ot = self.outputs.get("texture")
        oc = self.outputs.get("channel")
        kind = self._resolve_kind()
        if kind == 'TEXTURE':
            t.hide, c.hide = False, True
            ot.hide, oc.hide = False, True
        elif kind == 'CHANNEL':
            t.hide, c.hide = True, False
            ot.hide, oc.hide = True, False
        else:
            t.hide, c.hide = False, False
            ot.hide, oc.hide = False, False

    def _resolve_kind(self):
        t = self.inputs.get("texture")
        c = self.inputs.get("channel")
        if t and t.is_linked: return 'TEXTURE'
        if c and c.is_linked: return 'CHANNEL'
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

    def process(self, inputs: dict) -> dict | None:
        kind = self._resolve_kind()
        if kind == 'TEXTURE':
            tex = inputs.get("texture")
            if not isinstance(tex, np.ndarray) or tex.ndim != 3 or tex.shape[-1] not in (3, 4):
                return None
            img = _ensure_f32(tex.copy())
            if self.direction == 'SRGB2LIN':
                img[..., :3] = _srgb_to_linear(img[..., :3])
            else:
                img[..., :3] = _linear_to_srgb(img[..., :3])
            if self.clamp_result: img = _clamp01(img)
            return {"texture": img}
        if kind == 'CHANNEL':
            ch = inputs.get("channel")
            if not isinstance(ch, np.ndarray) or ch.ndim != 2:
                return None
            v = _ensure_f32(ch.copy())
            v = _srgb_to_linear(v) if self.direction == 'SRGB2LIN' else _linear_to_srgb(v)
            if self.clamp_result: v = _clamp01(v)
            return {"channel": v}
        return None
