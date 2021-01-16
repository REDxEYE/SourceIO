from ...shader_base import Nodes
from ..source1_shader_base import Source1ShaderBase


class LightmapGeneric(Source1ShaderBase):
    SHADER = 'lightmappedgeneric'

    @property
    def basetexture(self):
        texture_path = self._vavle_material.get_param('$basetexture', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
        return None

    @property
    def bumpmap(self):
        texture_path = self._vavle_material.get_param('$bumpmap', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.6, 0.0, 0.6, 1.0))
            if self.ssbump:
                image = self.convert_ssbump(image)
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def ssbump(self):
        return self._vavle_material.get_param('ssbump', 0) == 1

    @property
    def phong(self):
        return self._vavle_material.get_param('$phong', 0) == 1

    @property
    def alpha(self):
        return self._vavle_material.get_param('$alpha', None)

    @property
    def alphatest(self):
        return self._vavle_material.get_param('$alphatest', 0) == 1

    @property
    def translucent(self):
        return self._vavle_material.get_param('$translucent', 0) == 1

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        basetexture = self.basetexture

        if basetexture:
            basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture')
            basetexture_node.image = basetexture

            self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Base Color'])

            if self.translucent or self.alphatest:
                self.connect_nodes(basetexture_node.outputs['Alpha'], shader.inputs['Alpha'])

        if not self.phong:
            shader.inputs['Specular'].default_value = 0


class ReflectiveLightmapGeneric(LightmapGeneric):
    SHADER = 'lightmappedreflective'
