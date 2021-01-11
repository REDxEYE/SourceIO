import numpy as np

from SourceIO.source2.common import SourceVector4D
from ..source2_shader_base import Source2ShaderBase
from ...shader_base import Nodes


class VrComplex(Source2ShaderBase):
    SHADER: str = 'vr_complex.vfx'

    @property
    def color_texture(self):
        texture_path = self.get_texture('g_tColor', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.3, 0.3, 0.3, 1.0))
            return image
        return None

    @property
    def ambient_occlusion(self):
        texture_path = self.get_texture('g_tAmbientOcclusion', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (1.0, 1.0, 1.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def normalmap(self):
        texture_path = self.get_texture('g_tNormal', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.5, 1.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            image, roughness = self.split_normal(image)
            return image, roughness
        return None

    @property
    def color(self):
        return self.get_vector('g_vColorTint', SourceVector4D([1.0, 1.0, 1.0, 1.0])).as_list

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        color_texture = self.color_texture
        if color_texture:
            albedo_node = self.create_node(Nodes.ShaderNodeTexImage, 'albedo')
            albedo_node.image = color_texture

            if self.color:
                color_mix = self.create_node(Nodes.ShaderNodeMixRGB)
                color_mix.blend_type = 'MULTIPLY'
                self.connect_nodes(albedo_node.outputs['Color'], color_mix.inputs['Color1'])
                color = self.color
                if sum(color) > 3:
                    color = list(np.divide(color, 255))
                color_mix.inputs['Color2'].default_value = color
                color_mix.inputs['Fac'].default_value = 1.0
                self.connect_nodes(color_mix.outputs['Color'], shader.inputs['Base Color'])
            else:
                self.connect_nodes(albedo_node.outputs['Color'], shader.inputs['Base Color'])
            # if self.translucent or self.alphatest:
            #     self.connect_nodes(albedo_node.outputs['Alpha'], shader.inputs['Alpha'])

        normalmap, roughness = self.normalmap
        if normalmap:
            bumpmap_node = self.create_node(Nodes.ShaderNodeTexImage, 'normalmap')
            bumpmap_node.image = normalmap

            normalmap_node = self.create_node(Nodes.ShaderNodeNormalMap)

            self.connect_nodes(bumpmap_node.outputs['Color'], normalmap_node.inputs['Color'])
            self.connect_nodes(normalmap_node.outputs['Normal'], shader.inputs['Normal'])
        if roughness:
            roughness_node = self.create_node(Nodes.ShaderNodeTexImage, 'roughness')
            roughness_node.image = roughness
            self.connect_nodes(roughness_node.outputs['Color'], shader.inputs['Roughness'])
        # if self.selfillum:
        #     selfillummask = self.selfillummask
        #     albedo_node = self.get_node('$basetexture')
        #     if selfillummask is not None:
        #         selfillummask_node = self.create_node(Nodes.ShaderNodeTexImage, '$selfillummask')
        #         selfillummask_node.image = selfillummask
        #         self.connect_nodes(selfillummask_node.outputs['Color'], shader.inputs['Emission Strength'])
        #
        #     else:
        #         self.connect_nodes(albedo_node.outputs['Alpha'], shader.inputs['Emission Strength'])
        #     self.connect_nodes(albedo_node.outputs['Color'], shader.inputs['Emission'])
