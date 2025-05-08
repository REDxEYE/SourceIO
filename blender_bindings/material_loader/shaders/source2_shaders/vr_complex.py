from typing import Any

import bpy
import numpy as np

from SourceIO.blender_bindings.material_loader.shader_base import ExtraMaterialParameters

from SourceIO.blender_bindings.material_loader.shader_base import Nodes
from SourceIO.blender_bindings.material_loader.shaders.source2_shader_base import Source2ShaderBase
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4_3, is_blender_4


class VrComplex(Source2ShaderBase):
    SHADER: str = 'vr_complex.vfx'

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

    def create_nodes(self, material: bpy.types.Material, extra_parameters: dict[ExtraMaterialParameters, Any]):
        if "g_tAmbientOcclusion" in self.unused_textures:
            self.unused_textures.remove("g_tAmbientOcclusion")

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader_node = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader_node.outputs['BSDF'], material_output.inputs['Surface'])

        material_data = self._material_resource
        center = material_data.get_vector_property("g_vTexCoordCenter", (0.5, 0.5, 0.0))
        offset = material_data.get_vector_property("g_vTexCoordOffset", (0.0, 0.0, 0.0))
        scale = material_data.get_vector_property("g_vTexCoordScale", (1.0, 1.0, 0.0))
        transform_node = self.create_transform("TEXCOORD", scale, offset, center)

        albedo_node = self._get_texture("g_tColor", (0.3, 0.3, 0.3, 1.0), False)
        normal_node = self._get_texture("g_tNormal", (0.5, 0.5, 1.0, 1.0), True)

        self.connect_nodes(transform_node.outputs[0], albedo_node.inputs[0])
        self.connect_nodes(transform_node.outputs[0], normal_node.inputs[0])

        normal_output = normal_node.outputs[0]
        color_input_socket = shader_node.inputs['Base Color']
        color_output_socket = albedo_node.outputs['Color']

        color_tint = self._material_resource.get_vector_property("g_vColorTint", None)
        model_tint_amount = self._material_resource.get_float_property("g_flModelTintAmount", 1.0)

        tint_mask_output = None
        if self._check_flag("F_TINT_MASK"):
            tint_mask_node = self._get_texture("g_tTintMask", (1.0, 1.0, 1.0, 1.0), True)
            split_rgb_node = self.create_node(Nodes.ShaderNodeSeparateRGB)
            self.connect_nodes(tint_mask_node.outputs['Color'], split_rgb_node.inputs[0])
            tint_mask_output = split_rgb_node.outputs[0]

        if color_tint is not None and all(c == 1.0 for c in color_tint):
            color_output_socket = self.insert_generic_tint(color_output_socket, color_tint, model_tint_amount,
                                                           tint_mask_output)

        if extra_parameters.get(ExtraMaterialParameters.USE_OBJECT_TINT, False):
            color_output_socket = self.insert_object_tint(color_output_socket, 1.0, tint_mask_output)

        detail_mode = self._material_resource.get_int_property("F_DETAIL_TEXTURE", 0)
        if detail_mode:
            assert not self._check_flag("F_SECONDARY_UV", 0)
            detail_node = self._get_texture("g_tDetail", (1.0, 1.0, 1.0, 1.0), False)
            detail_mask_node = self._get_texture("g_tDetailMask", (1.0, 0.0, 0.0, 1.0), False)
            detail_scale = self._material_resource.get_vector_property("g_vDetailTexCoordScale", (1.0, 1.0, 1.0))
            detail_offset = self._material_resource.get_vector_property("g_vDetailTexCoordOffset", (0.0, 0.0, 0.0))
            detail_transform_node = self.create_transform("TEXCOORD", detail_scale, detail_offset, (0.5, 0.5, 0.0))
            self.connect_nodes(detail_transform_node.outputs[0], detail_node.inputs[0])
            self.connect_nodes(transform_node.outputs[0], detail_mask_node.inputs[0])

            detail_mask_split = self.create_node(Nodes.ShaderNodeSeparateXYZ)
            self.connect_nodes(detail_mask_node.outputs[0], detail_mask_split.inputs[0])

            mask_multiply_node = self.create_node(Nodes.ShaderNodeMath)
            mask_multiply_node.operation = 'MULTIPLY'
            mask_multiply_node.inputs[0].default_value = self._material_resource.get_float_property(
                "g_flDetailBlendFactor", 1.0)
            self.connect_nodes(detail_mask_split.outputs[0], mask_multiply_node.inputs[1])

            if detail_mode == 1:
                detail_blend_node = self.create_node(Nodes.ShaderNodeMixRGB)
                detail_blend_node.blend_type = 'MULTIPLY'
                self.connect_nodes(mask_multiply_node.outputs[0], detail_blend_node.inputs[0])

                detail_multiply_node = self.create_node(Nodes.ShaderNodeVectorMath)
                detail_multiply_node.operation = 'MULTIPLY'
                detail_multiply_node.inputs[1].default_value = [self._material_resource.get_float_property(
                    "g_flDetailModX", 2.0)] * 3
                self.connect_nodes(detail_node.outputs[0], detail_multiply_node.inputs[0])

                self.connect_nodes(color_output_socket, detail_blend_node.inputs[1])
                self.connect_nodes(detail_multiply_node.outputs[0], detail_blend_node.inputs[2])

                color_output_socket = detail_blend_node.outputs[0]

            if detail_mode == 2 or detail_mode == 4:
                detail_blend_node = self.create_node(Nodes.ShaderNodeMixRGB)
                detail_blend_node.blend_type = 'OVERLAY'
                self.connect_nodes(mask_multiply_node.outputs[0], detail_blend_node.inputs[0])
                self.connect_nodes(color_output_socket, detail_blend_node.inputs[1])
                self.connect_nodes(detail_node.outputs[0], detail_blend_node.inputs[2])

                color_output_socket = detail_blend_node.outputs[0]

            if detail_mode == 3 or detail_mode == 4:
                blend_normals_node = self.create_node_group("Blend Normals")
                detail_normal = self._get_texture("g_tNormalDetail", (0.5, 0.5, 1.0, 1.0), True)
                self.connect_nodes(detail_transform_node.outputs[0], detail_normal.inputs[0])
                self.connect_nodes(mask_multiply_node.outputs[0], blend_normals_node.inputs[0])
                self.connect_nodes(normal_node.outputs[0], blend_normals_node.inputs[1])
                self.connect_nodes(detail_normal.outputs[0], blend_normals_node.inputs[2])
                normal_output = blend_normals_node.outputs[0]


        self.connect_nodes(color_output_socket, color_input_socket)


        normalmap_node = self.create_node(Nodes.ShaderNodeNormalMap)

        self.connect_nodes(normal_output, normalmap_node.inputs['Color'])
        self.connect_nodes(normal_node.outputs['Alpha'], shader_node.inputs['Roughness'])
        self.connect_nodes(normalmap_node.outputs['Normal'], shader_node.inputs['Normal'])

        is_transparent =  (self._material_resource.get_int_property("F_ALPHA_TEST", 0) or self._material_resource.get_int_property("S_TRANSLUCENT", 0) or self._material_resource.get_int_property("F_OVERLAY", 0))

        if is_transparent:
            if material_data.get_int_property("F_ALPHA_TEST", 0):
                alpha_test_reference = material_data.get_float_property("g_flAlphaTestReference", 0.5)
                self._handle_alpha_modes("TEST", alpha_test_reference,
                                         albedo_node.outputs['Alpha'], shader_node.inputs['Alpha'])
            elif material_data.get_int_property("S_TRANSLUCENT", 0):
                self._handle_alpha_modes("TRANSLUCENT", 0.5,
                                         albedo_node.outputs['Alpha'], shader_node.inputs['Alpha'])
            elif material_data.get_int_property("F_OVERLAY", 0):
                self._handle_alpha_modes("OVERLAY", 0.5,
                                         albedo_node.outputs['Alpha'], shader_node.inputs['Alpha'])

        if self._check_flag("F_METALNESS_TEXTURE", 0):
            if is_transparent:
                metalness_node = self._get_texture("g_tMetalness", (0.0, 0.0, 0.0, 1.0), True)
                metalness_split_node = self.create_node(Nodes.ShaderNodeSeparateXYZ)
                self.connect_nodes(metalness_node.outputs[0], metalness_split_node.inputs[0])
                self.connect_nodes(metalness_split_node.outputs[1], shader_node.inputs['Metallic'])
            else:
                self.connect_nodes(albedo_node.outputs['Alpha'], shader_node.inputs['Metallic'])

        if self._have_texture("g_tSelfIllumMask") and self._check_flag("F_SELF_ILLUM"):
            self_illum_mask_node = self._get_texture("g_tSelfIllumMask", (1.0, 1.0, 1.0, 1.0), True)
            self._handle_self_illum(albedo_node.outputs[0], self_illum_mask_node.outputs[0],
                                    self._material_resource.get_vector_property('g_vSelfIllumTint'),
                                    self._material_resource.get_float_property("'g_flSelfIllumAlbedoFactor'", 1.0),
                                    self._material_resource.get_float_property("g_flSelfIllumBrightness", 1.0),
                                    shader_node.inputs['Emission Color'] if is_blender_4() else shader_node.inputs['Emission'],
                                    shader_node.inputs["Emission Strength"]
                                    )
