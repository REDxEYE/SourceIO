from typing import Optional
import bpy
import numpy as np

from ..source2_shader_base import Source2ShaderBase
from ...shader_base import Nodes


class Blend(Source2ShaderBase):
    SHADER: str = 'blend.vfx'

    @property
    def color_a_texture(self):
        texture_path = self.get_texture('g_tColorA', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.3, 0.3, 0.3, 1.0))
            return image
        return None

    @property
    def color_b_texture(self):
        texture_path = self.get_texture('g_tColorB', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.3, 0.3, 0.3, 1.0))
            return image
        return None

    @property
    def normal_a_texture(self):
        texture_path = self.get_texture('g_tNormalA', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.5, 1.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            image, roughness = self.split_normal(image)
            return image, roughness
        return None

    @property
    def normal_b_texture(self):
        texture_path = self.get_texture('g_tNormalB', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.5, 1.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            image, roughness = self.split_normal(image)
            return image, roughness
        return None

    @property
    def blend_mask(self):
        texture_path = self.get_texture('g_tMask', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (1.0, 1.0, 1.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def tint_mask(self):
        texture_path = self.get_texture('g_tTintMask', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (1.0, 1.0, 1.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def color(self):
        return self.get_vector('g_vColorTint', np.ones(4, dtype=np.float32))

    @property
    def alpha_test(self):
        return self.get_int('F_ALPHA_TEST', 0)

    @property
    def metalness(self):
        return self.get_int('F_METALNESS_TEXTURE', 0)

    @property
    def translucent(self):
        return self.get_int('F_TRANSLUCENT', 0)

    @property
    def specular(self):
        return self.get_int('F_SPECULAR', 0)

    @property
    def roughness_value(self):
        value = self.get_vector('TextureRoughness', None)
        if value is None:
            return
        return value[0]

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        normal_a,roughness_a = self.normal_a_texture
        normal_b,roughness_b = self.normal_b_texture

        self.create_texture_node(self.color_a_texture, 'COLOR_A')
        self.create_texture_node(self.color_b_texture, 'COLOR_B')
        self.create_texture_node(normal_a, 'NORMAL_A')
        self.create_texture_node(normal_b, 'NORMAL_B')
        self.create_texture_node(roughness_a, 'ROUGHNESS_A')
        self.create_texture_node(roughness_b, 'ROUGHNESS_B')
        self.create_texture_node(self.blend_mask, 'BLEND_MASK')
        self.create_texture_node(self.tint_mask, 'TINT_MASK')

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
