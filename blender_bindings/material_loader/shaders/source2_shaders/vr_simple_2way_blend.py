from typing import Optional
import bpy
import numpy as np

from .blend import Blend
from ..source2_shader_base import Source2ShaderBase
from ...shader_base import Nodes


class VRSimple2WayBlend(Blend):
    SHADER: str = 'vr_simple_2way_blend.vfx'


class SteamPalSimple2WayBlend(Blend):
    SHADER: str = 'steampal_2way_blend_mask.vfx'

