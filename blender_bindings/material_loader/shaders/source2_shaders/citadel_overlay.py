import numpy as np

from SourceIO.blender_bindings.material_loader.shader_base import Nodes
from SourceIO.blender_bindings.material_loader.shaders.source2_shader_base import Source2ShaderBase
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4_3


class CitadelOverlay(Source2ShaderBase):
    SHADER: str = 'citadel_overlay.vfx'

    @property
    def color_texture(self):
        texture_path = self._material_resource.get_texture_property('g_tColor', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.5, 0.5, 0.0))
            return image
        return None

    # todo support AO
    @property
    def ambient_occlusion(self):
        texture_path = self._material_resource.get_texture_property('g_tAmbientOcclusion', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (1.0, 1.0, 1.0, 0.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def normal_roughness_texture(self):
        texture_path = self._material_resource.get_texture_property('g_tNormalRoughness', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.5, 0.5, 0.0))
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def self_illum_mask_texture(self):
        texture_path = self._material_resource.get_texture_property('g_tDisplacementSelfIllumTintMask', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.0, 1.0, 0.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def color(self):
        return self._material_resource.get_vector_property('g_vColorTint', np.ones(4, dtype=np.float32))

    def create_nodes(self, material):
        if super().create_nodes(material) in ['UNKNOWN', 'LOADED']:
            return

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node_group("citadel_overlay.vfx", name=self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        color_texture = self.color_texture
        normal_roughness_texture = self.normal_roughness_texture
        self_illum_mask_texture = self.self_illum_mask_texture

        albedo_node = self.create_node(Nodes.ShaderNodeTexImage, 'albedo')
        albedo_node.image = color_texture

        self.connect_nodes(albedo_node.outputs['Color'], shader.inputs['TextureColor'])

        # Blender 4.2 changed blend & shadow method, plus made hashed default.
        if not is_blender_4_3():
            self.bpy_material.blend_method = 'HASHED'
            self.bpy_material.shadow_method = 'HASHED'
        self.connect_nodes(albedo_node.outputs['Alpha'], shader.inputs['Alpha'])

        normal_roughness_node = self.create_node(Nodes.ShaderNodeTexImage, 'normal')
        normal_roughness_node.image = normal_roughness_texture

        self.connect_nodes(normal_roughness_node.outputs['Color'], shader.inputs['TextureNormal'])
        self.connect_nodes(normal_roughness_node.outputs['Alpha'], shader.inputs['Roughness'])

        self_illum_mask_node = self.create_node(Nodes.ShaderNodeTexImage, 'displacement selfillum tintmask')
        self_illum_mask_node.image = self_illum_mask_texture

        self.connect_nodes(self_illum_mask_node.outputs['Color'], shader.inputs['DisplacementSelfIllumTintMask'])

        shader.inputs['GlobalTint'].default_value = self.color

        if self.tinted:
            vcolor_node = self.create_node(Nodes.ShaderNodeVertexColor)
            vcolor_node.layer_name = "TINT"
            self.connect_nodes(vcolor_node.outputs[0], shader.inputs["ModelTint"])
