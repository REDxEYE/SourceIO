import numpy as np

from ...shader_base import Nodes
from ..source1_shader_base import Source1ShaderBase


class VertexLitGeneric(Source1ShaderBase):
    SHADER: str = 'vertexlitgeneric'

    @property
    def bumpmap(self):
        texture_path = self._vavle_material.get_param('$bumpmap', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.5, 1.0, 1.0))
            image = self.convert_normalmap(image)
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

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
    def phongexponenttexture(self):
        texture_path = self._vavle_material.get_param('$phongexponenttexture', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.0, 0.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def color2(self):
        color_value = self._vavle_material.get_param('$color2', None)
        if color_value:
            if type(color_value) is float:
                color_value = [color_value, color_value, color_value]
            elif len(color_value) == 1:
                color_value = [color_value[0], color_value[0], color_value[0]]
        return color_value

    @property
    def color(self):
        color_value = self._vavle_material.get_param('$color', None)
        if color_value:
            if type(color_value) is float:
                color_value = [color_value, color_value, color_value]
            elif len(color_value) == 1:
                color_value = [color_value[0], color_value[0], color_value[0]]
        return color_value

    @property
    def translucent(self):
        return self._vavle_material.get_param('$translucent', 0) == 1

    @property
    def alphatest(self):
        return self._vavle_material.get_param('$alphatest', 0) == 1

    @property
    def additive(self):
        return self._vavle_material.get_param('$additive', 0) == 1

    @property
    def phong(self):
        return self._vavle_material.get_param('$phong', 0) == 1

    @property
    def selfillum(self):
        return self._vavle_material.get_param('$selfillum', 0) == 1

    @property
    def phongexponent(self):
        return self._vavle_material.get_param('$phongexponent', 5.0)

    @property
    def phongboost(self):
        value = self._vavle_material.get_param('$phongboost', 1)
        if type(value) is tuple:
            value = value[0]
        return value

    @property
    def phongtint(self):
        return self._vavle_material.get_param('$phongtint', (1.0, 1.0, 1.0))

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

            if self.color or self.color2:
                color_mix = self.create_node(Nodes.ShaderNodeMixRGB)
                color_mix.blend_type = 'MULTIPLY'
                self.connect_nodes(basetexture_node.outputs['Color'], color_mix.inputs['Color1'])
                color = (self.color or self.color2)
                if sum(color) > 3:
                    color = list(np.divide(color, 255))
                color_mix.inputs['Color2'].default_value = (*color, 1.0)
                color_mix.inputs['Fac'].default_value = 1.0
                self.connect_nodes(color_mix.outputs['Color'], shader.inputs['Base Color'])
            else:
                self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Base Color'])
            if self.translucent or self.alphatest:
                self.connect_nodes(basetexture_node.outputs['Alpha'], shader.inputs['Alpha'])

            if self.additive:
                basetexture_invert_node = self.create_node(Nodes.ShaderNodeInvert)
                basetexture_additive_mix_node = self.create_node(Nodes.ShaderNodeMixRGB)
                self.insert_node(basetexture_node.outputs['Color'], basetexture_additive_mix_node.inputs['Color1'],
                                 basetexture_additive_mix_node.outputs['Color'])
                basetexture_additive_mix_node.inputs['Color2'].default_value = (1.0, 1.0, 1.0, 1.0)


                self.connect_nodes(basetexture_node.outputs['Color'], basetexture_invert_node.inputs['Color'])
                self.connect_nodes(basetexture_invert_node.outputs['Color'], shader.inputs['Transmission'])
                self.connect_nodes(basetexture_invert_node.outputs['Color'],
                                   basetexture_additive_mix_node.inputs['Fac'])

        bumpmap = self.bumpmap
        if bumpmap:
            bumpmap_node = self.create_node(Nodes.ShaderNodeTexImage, '$bumpmap')
            bumpmap_node.image = bumpmap

            normalmap_node = self.create_node(Nodes.ShaderNodeNormalMap)

            self.connect_nodes(bumpmap_node.outputs['Color'], normalmap_node.inputs['Color'])
            self.connect_nodes(normalmap_node.outputs['Normal'], shader.inputs['Normal'])

        if self.selfillum:
            selfillummask = self.selfillummask
            basetexture_node = self.get_node('$basetexture')
            if selfillummask is not None:
                selfillummask_node = self.create_node(Nodes.ShaderNodeTexImage, '$selfillummask')
                selfillummask_node.image = selfillummask
                self.connect_nodes(selfillummask_node.outputs['Color'], shader.inputs['Emission Strength'])

            else:
                self.connect_nodes(basetexture_node.outputs['Alpha'], shader.inputs['Emission Strength'])
            self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Emission'])

        if not self.phong:
            shader.inputs['Specular'].default_value = 0
        elif self.phongboost is not None:
            shader.inputs['Specular'].default_value = self.clamp_value(self.phongboost / 64)
        phongexponenttexture = self.phongexponenttexture
        if self.phongexponent is not None and phongexponenttexture is None:
            shader.inputs['Roughness'].default_value = self.clamp_value(self.phongexponent / 256)
        elif self.phongexponenttexture is not None:
            phongexponenttexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$phongexponenttexture')
            phongexponenttexture_node.image = phongexponenttexture
            phongexponenttexture_split_node = self.create_node(Nodes.ShaderNodeSeparateRGB)
            self.connect_nodes(phongexponenttexture_node.outputs['Color'],
                               phongexponenttexture_split_node.inputs['Image'])

            phongexponenttexture_r_invert_node = self.create_node(Nodes.ShaderNodeInvert)
            self.connect_nodes(phongexponenttexture_split_node.outputs['R'],
                               phongexponenttexture_r_invert_node.inputs['Color'])
            self.connect_nodes(phongexponenttexture_split_node.outputs['G'],
                               shader.inputs['Metallic'])

            self.connect_nodes(phongexponenttexture_r_invert_node.outputs['Color'], shader.inputs['Roughness'])
