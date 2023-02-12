from typing import Optional

import bpy
import numpy as np

from ...shader_base import Nodes
from ..source2_shader_base import Source2ShaderBase
from .blend import Blend


class VRSimple2WayBlend(Blend):
    SHADER: str = 'vr_simple_2way_blend.vfx'


class SteamPalSimple2WayBlend(Blend):
    SHADER: str = 'steampal_2way_blend_mask.vfx'

