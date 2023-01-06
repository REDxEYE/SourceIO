from typing import Optional

import bpy

from ....library.goldsrc.mdl_v10.structs.texture import StudioTexture
from ..shader_base import Nodes, ShaderBase


class GoldSrcShaderBase(ShaderBase):
    SHADER: str = 'goldsrc_shader'

    def __init__(self, goldsrc_material: StudioTexture):
        super().__init__()
        self._valve_material: StudioTexture = goldsrc_material

    def _emit_surface(self, basetexture, rad_info):
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader_emit = self.create_node(Nodes.ShaderNodeEmission)
        shader_emit.inputs['Strength'].default_value = rad_info[3]

        color_mix = self.create_node(Nodes.ShaderNodeMixRGB)
        color_mix.blend_type = 'MULTIPLY'
        color_mix.inputs['Fac'].default_value = 1.0

        self.connect_nodes(color_mix.outputs['Color'], shader_emit.inputs['Color'])
        self.connect_nodes(basetexture.outputs['Color'], color_mix.inputs['Color1'])
        color_mix.inputs['Color2'].default_value = (*rad_info[:3], 1.0)

        self.connect_nodes(shader_emit.outputs['Emission'], material_output.inputs['Surface'])

    def load_texture(self, material_name, model_name: Optional[str] = None, **kwargs):
        model_texture_info = self._valve_material
        if model_name:
            texture_name = f"{model_name}_{material_name}"
        else:
            texture_name = material_name
        model_texture = bpy.data.images.get(texture_name, None)
        if model_texture is None:
            model_texture = bpy.data.images.new(
                texture_name,
                width=model_texture_info.width,
                height=model_texture_info.height,
                alpha=True
            )

            model_texture.pixels.foreach_set(model_texture_info.data.ravel())

            model_texture.pack()
        return model_texture
