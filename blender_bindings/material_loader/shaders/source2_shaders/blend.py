from typing import Any

import bpy
import numpy as np

from SourceIO.blender_bindings.material_loader.shader_base import Nodes, ExtraMaterialParameters
from SourceIO.blender_bindings.material_loader.shaders.source2_shader_base import Source2ShaderBase


class Blend(Source2ShaderBase):
    SHADER: str = 'blend.vfx'

    @property
    def metalness_a(self):
        return self._material_resource.get_float_property('g_flMetalnessA', 0)

    @property
    def metalness_b(self):
        return self._material_resource.get_float_property('g_flMetalnessB', 0)

    # @property
    # def color_a_texture(self):
    #     texture_path = self._material_resource.get_texture_property('g_tColorA', None)
    #     if texture_path is not None:
    #         image = self.load_texture_or_default(texture_path, (0.3, 0.3, 0.3, 1.0))
    #         return image
    #     return None
    #
    # @property
    # def color_b_texture(self):
    #     texture_path = self._material_resource.get_texture_property('g_tColorB', None)
    #     if texture_path is not None:
    #         image = self.load_texture_or_default(texture_path, (0.3, 0.3, 0.3, 1.0))
    #         return image
    #     return None
    #
    # @property
    # def normal_a_texture(self):
    #     texture_path = self._material_resource.get_texture_property('g_tNormalA', None)
    #     if texture_path is not None:
    #         image = self.load_texture_or_default(texture_path, (0.5, 0.5, 1.0, 1.0))
    #         image.colorspace_settings.is_data = True
    #         image.colorspace_settings.name = 'Non-Color'
    #         image, roughness = self.split_normal(image)
    #         return image, roughness
    #     return None
    #
    # @property
    # def normal_b_texture(self):
    #     texture_path = self._material_resource.get_texture_property('g_tNormalB', None)
    #     if texture_path is not None:
    #         image = self.load_texture_or_default(texture_path, (0.5, 0.5, 1.0, 1.0))
    #         image.colorspace_settings.is_data = True
    #         image.colorspace_settings.name = 'Non-Color'
    #         image, roughness = self.split_normal(image)
    #         return image, roughness
    #     return None
    #
    # @property
    # def blend_mask(self):
    #     texture_path = self._material_resource.get_texture_property('g_tMask', None)
    #     if texture_path is not None:
    #         image = self.load_texture_or_default(texture_path, (1.0, 1.0, 1.0, 1.0))
    #         image.colorspace_settings.is_data = True
    #         image.colorspace_settings.name = 'Non-Color'
    #         return image
    #     return None
    #
    # @property
    # def tint_mask(self):
    #     texture_path = self._material_resource.get_texture_property('g_tTintMask', None)
    #     if texture_path is not None:
    #         image = self.load_texture_or_default(texture_path, (1.0, 1.0, 1.0, 1.0))
    #         image.colorspace_settings.is_data = True
    #         image.colorspace_settings.name = 'Non-Color'
    #         return image
    #     return None

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

    def create_nodes(self, material:bpy.types.Material, extra_parameters: dict[ExtraMaterialParameters, Any]):

        if self._check_flag("F_METALNESS_TEXTURE",0):
            raise NotImplementedError("Metalness texture not supported for blend shader")

        if self._check_flag("F_ENABLE_TINT_MASKS",0):
            raise NotImplementedError("Tint masks not supported for blend shader")

        if self._material_resource.get_int_property("F_ALPHA_TEST", 0):
            raise NotImplementedError("Alpha test not supported for blend shader")
        elif self._material_resource.get_int_property("F_OVERLAY", 0):
            raise NotImplementedError("Overlay not supported for blend shader")
        elif self._material_resource.get_int_property("F_BLEND_MODE", 0):
            raise NotImplementedError("Blend mode not supported for blend shader")

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        material_data = self._material_resource
        center = material_data.get_vector_property("g_vTexCoordCenter", (0.5, 0.5, 0.0))
        offset = material_data.get_vector_property("g_vTexCoordOffset", (0.0, 0.0, 0.0))
        scale = material_data.get_vector_property("g_vTexCoordScale", (1.0, 1.0, 0.0))
        transform_node = self.create_transform("TEXCOORD", scale, offset, center)

        albedo1_node = self._get_texture("g_tColorA", (0.3, 0.3, 0.3, 1.0), False)
        albedo2_node = self._get_texture("g_tColorB", (0.3, 0.3, 0.3, 1.0), False)
        normal1_node = self._get_texture("g_tNormalA", (0.5, 0.5, 1.0, 1.0), True)
        normal2_node = self._get_texture("g_tNormalB", (0.5, 0.5, 1.0, 1.0), True)
        metalness1 = self._material_resource.get_float_property('g_flMetalnessA', 0)
        metalness2 = self._material_resource.get_float_property('g_flMetalnessB', 0)
        mask_node = self._get_texture("g_tMask", (1.0, 1.0, 1.0, 1.0), True)
        # tint_mask_node = self._get_texture("g_tTintMask", (1.0, 1.0, 1.0, 1.0), True)
        color_tint = self._material_resource.get_vector_property("g_vColorTint", None)
        model_tint_amount = self._material_resource.get_float_property("g_flModelTintAmount", 1.0)

        self.connect_nodes(transform_node.outputs[0], albedo1_node.inputs[0])
        self.connect_nodes(transform_node.outputs[0], albedo2_node.inputs[0])
        self.connect_nodes(transform_node.outputs[0], normal1_node.inputs[0])
        self.connect_nodes(transform_node.outputs[0], normal2_node.inputs[0])
        self.connect_nodes(transform_node.outputs[0], mask_node.inputs[0])

        split_mask = self.create_node(Nodes.ShaderNodeSeparateRGB)
        self.connect_nodes(mask_node.outputs[0], split_mask.inputs[0])
        mask_output = split_mask.outputs[0]

        mix_color = self.create_node(Nodes.ShaderNodeMixRGB)
        self.connect_nodes(mask_output,mix_color.inputs[0])
        self.connect_nodes(albedo1_node.outputs[0],mix_color.inputs[1])
        self.connect_nodes(albedo2_node.outputs[0],mix_color.inputs[2])

        color_output = mix_color.outputs[0]

        if color_tint is not None and all(c == 1.0 for c in color_tint):
            color_output = self.insert_generic_tint(color_output, color_tint, model_tint_amount)

        if extra_parameters.get(ExtraMaterialParameters.USE_OBJECT_TINT, False):
            color_output = self.insert_object_tint(color_output)

        mix_metalness = self.create_node(Nodes.ShaderNodeMixRGB)
        self.connect_nodes(mask_output,mix_metalness.inputs[0])
        mix_metalness.inputs[1].default_value = [metalness1] * 3 + [1.0]
        mix_metalness.inputs[2].default_value = [metalness2] * 3 + [1.0]

        mix_normals = self.create_node(Nodes.ShaderNodeMixRGB)
        self.connect_nodes(mask_output,mix_normals.inputs[0])
        self.connect_nodes(normal1_node.outputs[0],mix_normals.inputs[1])
        self.connect_nodes(normal2_node.outputs[0],mix_normals.inputs[2])

        mix_roughness = self.create_node(Nodes.ShaderNodeMixRGB)
        self.connect_nodes(mask_output,mix_roughness.inputs[0])
        self.connect_nodes(normal1_node.outputs[1],mix_roughness.inputs[1])
        self.connect_nodes(normal2_node.outputs[1],mix_roughness.inputs[2])

        normalmap_node = self.create_node(Nodes.ShaderNodeNormalMap)
        self.connect_nodes(mix_normals.outputs[0], normalmap_node.inputs["Color"])

        self.connect_nodes(color_output, shader.inputs['Base Color'])
        self.connect_nodes(mix_normals.outputs[0], normalmap_node.inputs['Color'])
        self.connect_nodes(mix_roughness.outputs[0], shader.inputs['Roughness'])
        self.connect_nodes(mix_metalness.outputs[0], shader.inputs['Metallic'])
        self.connect_nodes(normalmap_node.outputs[0], shader.inputs['Normal'])
