from pprint import pformat
from typing import Any

import bpy

from SourceIO.blender_bindings.material_loader.shader_base import Nodes, ExtraMaterialParameters
from SourceIO.blender_bindings.material_loader.shaders.source2_shader_base import Source2ShaderBase
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4_3
from SourceIO.library.source2.blocks.kv3_block import KVBlock


class CSGOComplex(Source2ShaderBase):
    SHADER: str = 'csgo_complex.vfx'

    def create_nodes(self, material: bpy.types.Material, extra_parameters: dict[ExtraMaterialParameters, Any]):
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node_group("csgo_complex.vfx", name=self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])
        material_data = self._material_resource
        data = self._material_resource.get_block(KVBlock, block_name='DATA')
        self.logger.info(pformat(dict(data)))

        center = material_data.get_vector_property("g_vTexCoordCenter", (0.5, 0.5, 0.0))
        offset = material_data.get_vector_property("g_vTexCoordOffset", (0.0, 0.0, 0.0))
        scale = material_data.get_vector_property("g_vTexCoordScale", (1.0, 1.0, 0.0))
        transform_node = self.create_transform("TEXCOORD", scale, offset, center)
        if self._have_texture("g_tColor"):
            color_texture = self._get_texture("g_tColor", (1, 1, 1, 1))
            self.connect_nodes(transform_node.outputs[0], color_texture.inputs[0])
            color_output = color_texture.outputs[0]
            if extra_parameters.get(ExtraMaterialParameters.USE_OBJECT_TINT, False):
                color_output = self.insert_object_tint(color_texture.outputs[0])
            self.connect_nodes(color_output, shader.inputs["TextureColor"])
            alpha_output = color_texture.outputs[1]
        else:
            alpha_output = None
        if self._have_texture("g_tDetail"):
            scale = material_data.get_vector_property("g_vDetailTexCoordScale", None)
            offset = material_data.get_vector_property("g_vDetailTexCoordOffset", None)

            detail_texture = self._get_texture("g_tDetail", (1, 1, 1, 1))
            detail_mask_texture = self._get_texture("g_tDetailMask", (1, 0, 0, 1))
            detail_uv_slot = "TEXCOORD1" if material_data.get_int_property("F_SECONDARY_UV", 0) else "TEXCOORD"
            detail_transform_node = self.create_transform(detail_uv_slot, scale, offset, (0.5, 0.5, 0))
            self.connect_nodes(detail_transform_node.outputs[0], detail_texture.inputs[0])

            self.connect_nodes(detail_texture.outputs[0], shader.inputs["TextureDetail"])
            self.connect_nodes(detail_mask_texture.outputs[0], shader.inputs["TextureDetailMask"])
            shader.inputs["F_DETAIL_TEXTURE"].default_value = float(
                material_data.get_int_property("F_DETAIL_TEXTURE", 0))
            shader.inputs["g_flDetailBlendFactor"].default_value = material_data.get_float_property(
                "g_flDetailBlendFactor", 0)

        if self._have_texture("g_tNormal"):
            normal_texture = self._get_texture("g_tNormal", (0.5, 0.5, 1, 1), True, True)
            self.connect_nodes(transform_node.outputs[0], normal_texture.inputs[0])
            self.connect_nodes(normal_texture.outputs[0], shader.inputs["TextureNormal"])
            self.connect_nodes(normal_texture.outputs[1], shader.inputs["TextureRoughness"])

        if self._have_texture("g_tTintMask") and self._check_flag("F_TINT_MASK", 0):
            tint_texture = self._get_texture("g_tTintMask", (1, 0, 0, 1), True)
            self.connect_nodes(transform_node.outputs[0], tint_texture.inputs[0])
            self.connect_nodes(tint_texture.outputs[0], shader.inputs["TextureTintMask"])

        if self._have_texture("g_tSelfIllumMask") and self._check_flag("F_SELF_ILLUM", 0):
            tint_texture = self._get_texture("g_tSelfIllumMask", (0, 0, 0, 1), True)
            self.connect_nodes(transform_node.outputs[0], tint_texture.inputs[0])
            self.connect_nodes(tint_texture.outputs[0], shader.inputs["TextureSelfIllumMask"])

            shader.inputs["SelfIllumTint"].default_value = material_data.get_vector_property(
                "g_vSelfIllumTint", (1, 1, 1, 1))

            shader.inputs["Emission Strength"].default_value = material_data.get_float_property(
                "g_flSelfIllumScale", 1)

        tint = material_data.get_vector_property("g_vColorTint", (1.0, 1.0, 1.0, 0.0))
        if all(c == 1.0 for c in tint[:3]):
            shader.inputs["g_vColorTint"].default_value = tint

        if self.tinted:
            vcolor_node = self.create_node(Nodes.ShaderNodeVertexColor)
            vcolor_node.layer_name = "TINT"
            self.connect_nodes(vcolor_node.outputs[0], shader.inputs["m_vColorTint"])
        else:
            object_info_node = self.create_node(Nodes.ShaderNodeObjectInfo)
            self.connect_nodes(object_info_node.outputs["Color"], shader.inputs["m_vColorTint"])

        shader.inputs["g_flModelTintAmount"].default_value = material_data.get_float_property("g_flModelTintAmount",
                                                                                              0.0)

        if self._check_flag("F_METALNESS_TEXTURE", 0) and alpha_output is not None:
            self.connect_nodes(alpha_output, shader.inputs["TextureMetalness"])
        else:
            shader.inputs["TextureMetalness"].default_value = material_data.get_float_property(
                "g_flMetalness", 0)

        if self._check_flag("F_ALPHA_TEST", 0):
            alpha_test_reference = material_data.get_float_property("g_flAlphaTestReference", 0.5)
            self._handle_alpha_modes("TEST", alpha_test_reference,
                                     alpha_output, shader.inputs['Alpha'])
        elif self._check_flag("S_TRANSLUCENT", 0):
            self._handle_alpha_modes("TRANSLUCENT", 0.5,
                                     alpha_output, shader.inputs['Alpha'])
        elif self._check_flag("F_OVERLAY", 0):
            self._handle_alpha_modes("OVERLAY", 0.5,
                                     alpha_output, shader.inputs['Alpha'])
