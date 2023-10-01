from pprint import pformat

from ..source2_shader_base import Source2ShaderBase
from ...shader_base import Nodes


class CSGOStaticOverlay(Source2ShaderBase):
    SHADER: str = 'csgo_static_overlay.vfx'

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node_group("csgo_complex.vfx", name=self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])
        material_data = self._material_resource
        data, = material_data.get_data_block(block_name='DATA')
        self.logger.info(pformat(dict(data)))

        if self._have_texture("g_tColor"):
            scale = material_data.get_vector_property("g_vTexCoordScale", None)
            offset = material_data.get_vector_property("g_vTexCoordOffset", None)
            center = material_data.get_vector_property("g_vTexCoordCenter", None)

            color_texture = self._get_texture("g_tColor", (1, 1, 1, 1))
            if scale is not None or offset is not None or center is not None:
                uv_node = self.create_node(Nodes.ShaderNodeUVMap)
                uv_transform = self.create_node_group("UVTransform")
                if scale is not None:
                    uv_transform.inputs["g_vTexCoordScale"].default_value = scale[:3]
                if offset is not None:
                    uv_transform.inputs["g_vTexCoordOffset"].default_value = offset[:3]
                if center is not None:
                    uv_transform.inputs["g_vTexCoordCenter"].default_value = center[:3]

                self.connect_nodes(uv_node.outputs[0], uv_transform.inputs[0])

                self.connect_nodes(uv_transform.outputs[0], color_texture.inputs[0])

            self.connect_nodes(color_texture.outputs[0], shader.inputs["TextureColor"])
            alpha_output = color_texture.outputs[1]
        else:
            alpha_output = None
        if self._have_texture("g_tDetail"):
            scale = material_data.get_vector_property("g_vDetailTexCoordScale", None)
            offset = material_data.get_vector_property("g_vDetailTexCoordOffset", None)

            detail_texture = self._get_texture("g_tDetail", (1, 1, 1, 1))
            detail_mask_texture = self._get_texture("g_tDetailMask", (1, 0, 0, 1))
            if scale is not None or offset is not None:
                uv_node = self.create_node(Nodes.ShaderNodeUVMap)
                uv_transform = self.create_node_group("UVTransform")
                if scale is not None:
                    uv_transform.inputs["g_vTexCoordScale"].default_value = scale[:3]
                if offset is not None:
                    uv_transform.inputs["g_vTexCoordOffset"].default_value = offset[:3]
                uv_transform.inputs["g_vTexCoordCenter"].default_value = (0.5, 0.5, 0)

                self.connect_nodes(uv_node.outputs[0], uv_transform.inputs[0])

                self.connect_nodes(uv_transform.outputs[0], detail_texture.inputs[0])

            self.connect_nodes(detail_texture.outputs[0], shader.inputs["TextureDetail"])
            self.connect_nodes(detail_mask_texture.outputs[0], shader.inputs["TextureDetailMask"])
            shader.inputs["F_DETAIL_TEXTURE"].default_value = float(
                material_data.get_int_property("F_DETAIL_TEXTURE", 0))
            shader.inputs["g_flDetailBlendFactor"].default_value = material_data.get_float_property(
                "g_flDetailBlendFactor", 0)

        if self._have_texture("g_tNormal"):
            normal_texture = self._get_texture("g_tNormal", (0.5, 0.5, 1, 1), True, True)
            self.connect_nodes(normal_texture.outputs[0], shader.inputs["TextureNormal"])
            self.connect_nodes(normal_texture.outputs[1], shader.inputs["TextureRoughness"])

        if self._have_texture("g_tTintMask") and material_data.get_int_property("F_TINT_MASK", 0) == 1:
            tint_texture = self._get_texture("g_tTintMask", (1, 0, 0, 1), True, True)
            self.connect_nodes(tint_texture.outputs[0], shader.inputs["TextureTintMask"])

        if self._have_texture("g_tSelfIllumMask") and material_data.get_int_property("F_SELF_ILLUM", 0) == 1:
            tint_texture = self._get_texture("g_tSelfIllumMask", (0, 0, 0, 1), True, False)
            self.connect_nodes(tint_texture.outputs[0], shader.inputs["TextureSelfIllumMask"])

            shader.inputs["g_vSelfIllumTint"].default_value = material_data.get_vector_property(
                "g_vSelfIllumTint", (1, 1, 1, 1))

            shader.inputs["g_flSelfIllumScale"].default_value = material_data.get_float_property(
                "g_flSelfIllumScale", 1)

        tint = material_data.get_vector_property("g_vColorTint", None)
        if tint is not None and (tint[0] != 1.0 or tint[1] != 1.0 or tint[2] != 1.0):
            shader.inputs["g_vColorTint"].default_value = tint

        if self.tinted:
            vcolor_node = self.create_node(Nodes.ShaderNodeVertexColor)
            vcolor_node.layer_name = "TINT"
            self.connect_nodes(vcolor_node.outputs[0], shader.inputs["m_vColorTint"])

        shader.inputs["g_flModelTintAmount"].default_value = material_data.get_float_property(
            "g_flModelTintAmount", 0.0)

        if material_data.get_int_property("F_METALNESS_TEXTURE", 0) == 1 and alpha_output is not None:
            self.connect_nodes(alpha_output, shader.inputs["TextureMetalness"])
        else:
            shader.inputs["TextureMetalness"].default_value = material_data.get_float_property(
                "g_flMetalness", 0)

        if self._material_resource.get_int_property("F_ALPHA_TEST", 0) and alpha_output is not None:
            self.bpy_material.blend_method = 'BLEND'
            self.bpy_material.shadow_method = 'CLIP'
            self.connect_nodes(alpha_output, shader.inputs["Alpha"])
        elif self._material_resource.get_int_property("F_OVERLAY", 0) and alpha_output is not None:
            self.bpy_material.blend_method = 'BLEND'
            self.bpy_material.shadow_method = 'CLIP'
            self.connect_nodes(alpha_output, shader.inputs["Alpha"])
        elif self._material_resource.get_int_property("F_BLEND_MODE", 0) and alpha_output is not None:
            self.bpy_material.blend_method = 'BLEND'
            self.bpy_material.shadow_method = 'CLIP'
            self.connect_nodes(alpha_output, shader.inputs["Alpha"])
