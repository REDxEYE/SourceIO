import numpy as np

from ...shader_base import Nodes
from ..source2_shader_base import Source2ShaderBase


class Blend(Source2ShaderBase):
    SHADER: str = 'blend.vfx'

    @property
    def metalness_a(self):
        return self._material_resource.get_float_property('g_flMetalnessA', 0)

    @property
    def metalness_b(self):
        return self._material_resource.get_float_property('g_flMetalnessB', 0)

    @property
    def color_a_texture(self):
        texture_path = self._material_resource.get_texture_property('g_tColorA', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.3, 0.3, 0.3, 1.0))
            return image
        return None

    @property
    def color_b_texture(self):
        texture_path = self._material_resource.get_texture_property('g_tColorB', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.3, 0.3, 0.3, 1.0))
            return image
        return None

    @property
    def normal_a_texture(self):
        texture_path = self._material_resource.get_texture_property('g_tNormalA', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.5, 1.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            image, roughness = self.split_normal(image)
            return image, roughness
        return None

    @property
    def normal_b_texture(self):
        texture_path = self._material_resource.get_texture_property('g_tNormalB', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.5, 1.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            image, roughness = self.split_normal(image)
            return image, roughness
        return None

    @property
    def blend_mask(self):
        texture_path = self._material_resource.get_texture_property('g_tMask', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (1.0, 1.0, 1.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def tint_mask(self):
        texture_path = self._material_resource.get_texture_property('g_tTintMask', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (1.0, 1.0, 1.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def color(self):
        return self._material_resource.get_vector_property('g_vColorTint', np.ones(4, dtype=np.float32))

    @property
    def alpha_test(self):
        return self._material_resource.get_int_property('F_ALPHA_TEST', 0)

    @property
    def metalness(self):
        return self._material_resource.get_int_property('F_METALNESS_TEXTURE', 0)

    @property
    def translucent(self):
        return self._material_resource.get_int_property('F_TRANSLUCENT', 0)

    @property
    def specular(self):
        return self._material_resource.get_int_property('F_SPECULAR', 0)

    @property
    def roughness_value(self):
        value = self._material_resource.get_vector_property('TextureRoughness', None)
        if value is None:
            return
        return value[0]

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        normal_a, roughness_a = self.normal_a_texture
        normal_b, roughness_b = self.normal_b_texture

        split_mask = self.create_node(Nodes.ShaderNodeSeparateRGB)
        mix_metalnes = self.create_node(Nodes.ShaderNodeMixRGB)
        mix_metalnes.inputs['Color1'].default_value = [self.metalness_a] * 3 + [1.0]
        mix_metalnes.inputs['Color2'].default_value = [self.metalness_a] * 3 + [1.0]

        mix_normals = self.create_node(Nodes.ShaderNodeMixRGB)
        mix_roughness = self.create_node(Nodes.ShaderNodeMixRGB)
        mix_color = self.create_node(Nodes.ShaderNodeMixRGB)
        normalmap_node = self.create_node(Nodes.ShaderNodeNormalMap)

        self.create_and_connect_texture_node(self.blend_mask, split_mask.inputs['Image'], name='BLEND_MASK')

        self.create_and_connect_texture_node(self.color_a_texture, mix_color.inputs['Color1'], name='COLOR_A')
        self.create_and_connect_texture_node(self.color_b_texture, mix_color.inputs['Color2'], name='COLOR_B')

        self.create_and_connect_texture_node(normal_a, mix_normals.inputs['Color1'], name='NORMAL_A')
        self.create_and_connect_texture_node(normal_b, mix_normals.inputs['Color2'], name='NORMAL_B')

        self.create_and_connect_texture_node(normal_a, mix_roughness.inputs['Color1'], name='ROUGHNESS_A')
        self.create_and_connect_texture_node(normal_b, mix_roughness.inputs['Color2'], name='ROUGHNESS_B')

        texture_tint_mask = self.create_texture_node(self.tint_mask, 'TINT_MASK')

        self.connect_nodes(mix_color.outputs['Color'], shader.inputs['Base Color'])
        self.connect_nodes(split_mask.outputs['R'], mix_color.inputs['Fac'])
        self.connect_nodes(split_mask.outputs['R'], mix_normals.inputs['Fac'])
        self.connect_nodes(split_mask.outputs['R'], mix_roughness.inputs['Fac'])
        self.connect_nodes(split_mask.outputs['R'], mix_metalnes.inputs['Fac'])
        self.connect_nodes(mix_normals.outputs['Color'], normalmap_node.inputs['Color'])
        self.connect_nodes(mix_roughness.outputs['Color'], shader.inputs['Roughness'])
        self.connect_nodes(mix_metalnes.outputs['Color'], shader.inputs['Metallic'])
        self.connect_nodes(normalmap_node.outputs['Normal'], shader.inputs['Normal'])
