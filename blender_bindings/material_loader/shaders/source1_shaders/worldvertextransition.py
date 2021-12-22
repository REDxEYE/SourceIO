from ...shader_base import Nodes
from ..source1_shader_base import Source1ShaderBase


class WorldVertexTransition(Source1ShaderBase):
    SHADER = 'worldvertextransition'

    @property
    def basetexture(self):
        texture_path = self._vmt.get_string('$basetexture', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0.0, 0.3, 1.0))
        return None

    @property
    def basetexture2(self):
        texture_path = self._vmt.get_string('$basetexture2', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0.3, 0.0, 1.0))
        return None

    @property
    def bumpmap(self):
        texture_path = self._vmt.get_string('$bumpmap', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.6, 0.0, 0.6, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def bumpmap2(self):
        texture_path = self._vmt.get_string('$bumpmap2', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.6, 0.6, 0.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def selfillum(self):
        return self._vmt.get_int('$selfillum', 0) == 1

    @property
    def ssbump(self):
        return self._vmt.get_int('ssbump', 0) == 1

    @property
    def translucent(self):
        return self._vmt.get_int('$translucent', 0) == 1

    @property
    def alpha(self):
        return self._vmt.get_float('alpha', 1.0)

    @property
    def phong(self):
        return self._vmt.get_int('$phong', 0) == 1

    @property
    def phongboost(self):
        return self._vmt.get_float('$phongboost', 1)

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        basetexture = self.basetexture
        basetexture2 = self.basetexture2

        if basetexture and basetexture2:
            vertex_color = self.create_node(Nodes.ShaderNodeVertexColor)
            
            color_mix = self.create_node(Nodes.ShaderNodeMixRGB)
            self.connect_nodes(vertex_color.outputs['Color'], color_mix.inputs['Fac'])
            color_mix.blend_type = 'MIX'

            self.create_and_connect_texture_node(basetexture,
                                                 color_mix.inputs['Color1'],
                                                 name='$basetexture')
            self.create_and_connect_texture_node(basetexture2,
                                                 color_mix.inputs['Color2'],
                                                 name='$basetexture2')

            self.connect_nodes(color_mix.outputs['Color'], shader.inputs['Base Color'])

        if not self.phong:
            shader.inputs['Specular'].default_value = 0
