from ....goldsrc.mdl.structs.texture import StudioTexture
from ..shader_base import ShaderBase


class GoldSrcShaderBase(ShaderBase):
    SHADER: str = 'goldsrc_shader'

    def __init__(self, goldsrc_material: StudioTexture):
        super().__init__()
        self._vavle_material: StudioTexture = goldsrc_material
