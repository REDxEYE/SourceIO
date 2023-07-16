from ...shader_base import Nodes
from ..source1_shader_base import Source1ShaderBase


class UnlitGeneric(Source1ShaderBase):
    SHADER: str = 'unlitgeneric'

    @property
    def basetexture(self):
        texture_path = self._vmt.get_string('$basetexture', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
        return None

    @property
    def color2(self):
        color_value, value_type = self._vmt.get_vector('$color2', None)
        if color_value is None:
            return None
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        return self.ensure_length(color_value, 4, 1.0)

    @property
    def color(self):
        color_value, value_type = self._vmt.get_vector('$color', None)
        if color_value is None:
            return None
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        return self.ensure_length(color_value, 4, 1.0)

    @property
    def translucent(self):
        return self._vmt.get_int('$translucent', 0) == 1

    @property
    def alphatest(self):
        return self._vmt.get_int('$alphatest', 0) == 1

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeEmission, self.SHADER)
        if self.translucent or self.alphatest:
            self.bpy_material.blend_method = 'HASHED'
            self.bpy_material.shadow_method = 'HASHED'
            mix_node = self.create_node(Nodes.ShaderNodeMixShader)
            transparent_node = self.create_node(Nodes.ShaderNodeBsdfTransparent)
            self.connect_nodes(shader.outputs['Emission'], mix_node.inputs[2])
            self.connect_nodes(transparent_node.outputs['BSDF'], mix_node.inputs[1])
            self.connect_nodes(mix_node.outputs['Shader'], material_output.inputs['Surface'])
        else:
            self.connect_nodes(shader.outputs['Emission'], material_output.inputs['Surface'])

        basetexture = self.basetexture
        if basetexture:
            basetexture_node = self.create_and_connect_texture_node(basetexture, name='$basetexture')
            shader.inputs['Strength'].default_value = 1.0
            color = self.color or self.color2
            if self.translucent or self.alphatest:
                self.connect_nodes(basetexture_node.outputs['Alpha'], mix_node.inputs['Fac'])
            if color:
                color_mix = self.create_node(Nodes.ShaderNodeMixRGB)
                color_mix.blend_type = 'MULTIPLY'
                self.connect_nodes(basetexture_node.outputs['Color'], color_mix.inputs['Color1'])
                color_mix.inputs['Color2'].default_value = color
                color_mix.inputs['Fac'].default_value = 1.0
                self.connect_nodes(color_mix.outputs['Color'], shader.inputs['Color'])
            else:
                self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Color'])


class SDKUnlitGeneric(Source1ShaderBase):
    SHADER: str = 'sdk_unlitgeneric'
