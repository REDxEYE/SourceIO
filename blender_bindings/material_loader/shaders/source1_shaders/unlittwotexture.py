from ...shader_base import Nodes
from ..source1_shader_base import Source1ShaderBase


class UnlitGeneric(Source1ShaderBase):
    SHADER: str = 'unlittwotexture'

    @property
    def basetexture(self):
        texture_path = self._vmt.get_string('$basetexture', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
        return None

    @property
    def texture2(self):
        texture_path = self._vmt.get_string('$texture2', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.0, 0.0, 0.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def color2(self):
        color_value, value_type = self._vmt.get_vector('$color2', [1, 1, 1])
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        elif len(color_value) > 3:
            color_value = color_value[:3]
        return color_value

    @property
    def color(self):
        color_value, value_type = self._vmt.get_vector('$color', [1, 1, 1])
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        elif len(color_value) > 3:
            color_value = color_value[:3]
        return color_value

    @property
    def additive(self):
        return self._vmt.get_int('$additive', 0) == 1

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        basetexture = self.basetexture
        texture2 = self.texture2
        if basetexture:
            basetexture_node = self.create_and_connect_texture_node(basetexture, name='$basetexture')
            if texture2:
                texture2_node = self.create_and_connect_texture_node(texture2, name='$basetexture')

                color_mix = self.create_node(Nodes.ShaderNodeMixRGB)
                color_mix.blend_type = 'MULTIPLY'
                color_mix.inputs['Fac'].default_value = 1.0
                self.connect_nodes(basetexture_node.outputs['Color'], color_mix.inputs['Color1'])
                self.connect_nodes(texture2_node.outputs['Color'], color_mix.inputs['Color2'])
                texture_output = color_mix.outputs['Color']
            else:
                texture_output = basetexture_node.outputs['Color']

            if self.color or self.color2:
                color_mix = self.create_node(Nodes.ShaderNodeMixRGB)
                color_mix.blend_type = 'MULTIPLY'
                self.connect_nodes(texture_output, color_mix.inputs['Color1'])
                color_mix.inputs['Color2'].default_value = (*(self.color or self.color2), 1.0)
                color_mix.inputs['Fac'].default_value = 1.0
                self.connect_nodes(color_mix.outputs['Color'], shader.inputs['Base Color'])
            else:
                self.connect_nodes(texture_output, shader.inputs['Base Color'])
            if self.additive:
                if self.additive:
                    basetexture_invert_node = self.create_node(Nodes.ShaderNodeInvert)
                    basetexture_additive_mix_node = self.create_node(Nodes.ShaderNodeMixRGB)

                    self.insert_node(texture_output, basetexture_additive_mix_node.inputs['Color1'],
                                     basetexture_additive_mix_node.outputs['Color'])
                    basetexture_additive_mix_node.inputs['Color2'].default_value = (1.0, 1.0, 1.0, 1.0)

                    self.connect_nodes(texture_output, basetexture_invert_node.inputs['Color'])
                    self.connect_nodes(basetexture_invert_node.outputs['Color'], shader.inputs['Transmission'])
                    self.connect_nodes(basetexture_invert_node.outputs['Color'],
                                       basetexture_additive_mix_node.inputs['Fac'])
