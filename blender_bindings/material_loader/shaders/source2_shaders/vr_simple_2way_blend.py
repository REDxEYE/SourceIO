from .blend import Blend


class VRSimple2WayBlend(Blend):
    SHADER: str = 'vr_simple_2way_blend.vfx'


class SteamPalSimple2WayBlend(Blend):
    SHADER: str = 'steampal_2way_blend_mask.vfx'

