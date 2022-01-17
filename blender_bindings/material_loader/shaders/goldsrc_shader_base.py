import bpy

from ....library.goldsrc.mdl_v10.structs.texture import StudioTexture
from ..shader_base import ShaderBase, Nodes


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

    def load_texture(self, material_name, **kwargs):
        model_texture_info = self._valve_material
        model_texture = bpy.data.images.get(material_name, None)
        if model_texture is None:
            model_texture = bpy.data.images.new(
                material_name,
                width=model_texture_info.width,
                height=model_texture_info.height,
                alpha=True
            )

            if bpy.app.version > (2, 83, 0):
                model_texture.pixels.foreach_set(model_texture_info.data.flatten().tolist())
            else:
                model_texture.pixels[:] = model_texture_info.data.flatten().tolist()

            model_texture.pack()
        return model_texture
