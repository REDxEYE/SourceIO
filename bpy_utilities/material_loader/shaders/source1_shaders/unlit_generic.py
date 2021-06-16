from ...shader_base import Nodes
from ..source1_shader_base import Source1ShaderBase


class UnlitGeneric(Source1ShaderBase):
    SHADER: str = 'unlitgeneric'

    @property
    def basetexture(self):
        texture_path = self._vavle_material.get_param('$basetexture', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
        return None

    @property
    def selfillummask(self):
        texture_path = self._vavle_material.get_param('$selfillummask', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.0, 0.0, 0.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def color2(self):
        color_value, value_type = self._vavle_material.get_vector('$color2', None)
        if color_value is None:
            return None
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        return self.ensure_length(color_value, 4, 1.0)

    @property
    def color(self):
        color_value, value_type = self._vavle_material.get_vector('$color', None)
        if color_value is None:
            return None
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        return self.ensure_length(color_value, 4, 1.0)

    @property
    def translucent(self):
        return self._vavle_material.get_int('$translucent', 0) == 1

    @property
    def alphatest(self):
        return self._vavle_material.get_int('$alphatest', 0) == 1

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        basetexture = self.basetexture
        if basetexture:
            basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture')
            basetexture_node.image = basetexture
            shader.inputs['Roughness'].default_value = 1
            shader.inputs['Specular'].default_value = 0
            color = self.color or self.color2
            if color:
                shader.inputs['Emission Strength'].default_value = max(color[:3])  # excluding alpha value
                color_mix = self.create_node(Nodes.ShaderNodeMixRGB)
                color_mix.blend_type = 'MULTIPLY'
                self.connect_nodes(basetexture_node.outputs['Color'], color_mix.inputs['Color1'])
                color_mix.inputs['Color2'].default_value = color
                color_mix.inputs['Fac'].default_value = 1.0
                self.connect_nodes(color_mix.outputs['Color'], shader.inputs['Base Color'])
                self.connect_nodes(color_mix.outputs['Color'], shader.inputs['Emission'])
            else:
                self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Base Color'])
                self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Emission'])

            if self.translucent or self.alphatest:
                self.connect_nodes(basetexture_node.outputs['Alpha'], shader.inputs['Alpha'])

