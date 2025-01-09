from pprint import pformat

from SourceIO.blender_bindings.material_loader.shader_base import Nodes
from SourceIO.blender_bindings.material_loader.shaders.source2_shader_base import Source2ShaderBase
from SourceIO.library.source2.blocks.kv3_block import KVBlock

# todo support border effects, as well as modifiers to textures like softness, sharpness, brightness, darkness, and contrast
class EnvironmentBlend(Source2ShaderBase):
    SHADER: str = 'environment_blend.vfx'

    def create_nodes(self, material):
        if super().create_nodes(material) in ['UNKNOWN', 'LOADED']:
            return
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node_group("environment_blend.vfx", name=self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])
        material_data = self._material_resource
        data = material_data.get_block(KVBlock, block_name='DATA')
        self.logger.info(pformat(dict(data)))

        if self._have_texture("g_tColor1"):
            color_texture0 = self._get_texture("g_tColor1", (1, 1, 1, 1))
            self.connect_nodes(color_texture0.outputs[0], shader.inputs["TextureColor0"])
            self.connect_nodes(color_texture0.outputs[1], shader.inputs["TextureTintMask0"])

        if self._have_texture("g_tColor2"):
            color_texture1 = self._get_texture("g_tColor2", (1, 1, 1, 1))
            self.connect_nodes(color_texture1.outputs[0], shader.inputs["TextureColor1"])
            self.connect_nodes(color_texture1.outputs[1], shader.inputs["TextureTintMask1"])

        if self._have_texture("g_tColor3"):
            color_texture2 = self._get_texture("g_tColor3", (1, 1, 1, 1))
            self.connect_nodes(color_texture2.outputs[0], shader.inputs["TextureColor2"])
            self.connect_nodes(color_texture2.outputs[1], shader.inputs["TextureTintMask2"])

        if self._have_texture("g_tNormalRoughness1"):
            normal_texture0 = self._get_texture("g_tNormalRoughness1", (0.5, 0.5, 1, 1), True)
            self.connect_nodes(normal_texture0.outputs[0], shader.inputs["TextureNormal0"])
            self.connect_nodes(normal_texture0.outputs[1], shader.inputs["TextureRoughness0"])

        if self._have_texture("g_tNormalRoughness2"):
            normal_texture1 = self._get_texture("g_tNormalRoughness2", (0.5, 0.5, 1, 1), True)
            self.connect_nodes(normal_texture1.outputs[0], shader.inputs["TextureNormal1"])
            self.connect_nodes(normal_texture1.outputs[1], shader.inputs["TextureRoughness1"])

        if self._have_texture("g_tNormalRoughness3"):
            normal_texture2 = self._get_texture("g_tNormalRoughness3", (0.5, 0.5, 1, 1), True)
            self.connect_nodes(normal_texture2.outputs[0], shader.inputs["TextureNormal2"])
            self.connect_nodes(normal_texture2.outputs[1], shader.inputs["TextureRoughness2"])

        if self._have_texture("g_tPacked1"):
            packed_texture0 = self._get_texture("g_tPacked1", (0.5, 0.5, 1, 1), True)
            self.connect_nodes(packed_texture0.outputs[0], shader.inputs["TexturePacked0"])

        if self._have_texture("g_tPacked2"):
            packed_texture1 = self._get_texture("g_tPacked2", (0.5, 0.5, 1, 1), True)
            self.connect_nodes(packed_texture1.outputs[0], shader.inputs["TexturePacked1"])

        if self._have_texture("g_tPacked3"):
            packed_texture2 = self._get_texture("g_tPacked3", (0.5, 0.5, 1, 1), True)
            self.connect_nodes(packed_texture2.outputs[0], shader.inputs["TexturePacked2"])

        if self._have_texture("g_tRevealMask2"):
            reveal_texture0 = self._get_texture("g_tRevealMask2", (1, 1, 1, 1), True)
            self.connect_nodes(reveal_texture0.outputs[0], shader.inputs["RevealMask1"])

        if self._have_texture("g_tRevealMask3"):
            reveal_texture1 = self._get_texture("g_tRevealMask3", (1, 1, 1, 1), True)
            self.connect_nodes(reveal_texture1.outputs[0], shader.inputs["RevealMask2"])

        # re-use this node for all of the overlays
        uv_node = self.create_node(Nodes.ShaderNodeUVMap)
        uv_node.uv_map = "TEXCOORD"

        if self._have_texture("g_tColorOverlay"):
            # todo support rotation and translation
            scale = material_data.get_vector_property("g_vOverlayScaleRotationTranslation", None)

            detail_texture = self._get_texture("g_tColorOverlay", (0.5, 0.5, 0.5, 0))
            if scale is not None:
                uv_transform = self.create_node_group("UVTransform")

                uv_transform.inputs["g_vTexCoordScale"].default_value = scale[:3]

                self.connect_nodes(uv_node.outputs[0], uv_transform.inputs[0])

                self.connect_nodes(uv_transform.outputs[0], detail_texture.inputs[0])

            self.connect_nodes(detail_texture.outputs[0], shader.inputs["ColorOverlay"])
            # shader.inputs["F_DETAIL_TEXTURE"].default_value = 2.0
            # shader.inputs["g_flDetailBlendFactor"].default_value = 2

        if self._have_texture("g_tColorWorldOverlay"):
            # todo support scale, rotation and translation
            # world overlay uses a different system for scale, instead setting a "tiling" value.
            world_detail_texture = self._get_texture("g_tColorWorldOverlay", (0.5, 0.5, 0.5, 0))
            self.connect_nodes(world_detail_texture.outputs[0], shader.inputs["WorldOverlay"])

        if self._have_texture("g_tNormalRoughnessOverlay"):
            scale = material_data.get_vector_property("g_vOverlayScaleRotationTranslation", None)

            normal_detail_texture = self._get_texture("g_tNormalRoughnessOverlay", (0.5, 0.5, 0.5, 0), True)

            if scale is not None:
                uv_transform = self.create_node_group("UVTransform")

                uv_transform.inputs["g_vTexCoordScale"].default_value = scale[:3]

                self.connect_nodes(uv_node.outputs[0], uv_transform.inputs[0])

                self.connect_nodes(uv_transform.outputs[0], detail_texture.inputs[0])

            self.connect_nodes(normal_detail_texture.outputs[0], shader.inputs["NormalOverlay"])
            self.connect_nodes(normal_detail_texture.outputs[1], shader.inputs["RoughnessOverlay"])

        if self._have_texture("g_tRevealOverlay"):
            scale = material_data.get_vector_property("g_vOverlayScaleRotationTranslation", None)

            reveal_detail_texture = self._get_texture("g_tRevealOverlay", (0, 0, 0, 0), True)

            if scale is not None:
                uv_transform = self.create_node_group("UVTransform")

                uv_transform.inputs["g_vTexCoordScale"].default_value = scale[:3]

                self.connect_nodes(uv_node.outputs[0], uv_transform.inputs[0])

                self.connect_nodes(uv_transform.outputs[0], detail_texture.inputs[0])

            self.connect_nodes(reveal_detail_texture.outputs[0], shader.inputs["RevealOverlay"])

        shader.inputs["WorldOverlayStrength0"].default_value = material_data.get_float_property(
            "g_flWorldOverlayStrength1", 1.0)
        shader.inputs["WorldOverlayStrength1"].default_value = material_data.get_float_property(
            "g_flWorldOverlayStrength2", 1.0)
        shader.inputs["WorldOverlayStrength2"].default_value = material_data.get_float_property(
            "g_flWorldOverlayStrength3", 1.0)

        shader.inputs["ColorOverlayStrength0"].default_value = material_data.get_float_property(
            "g_flColorOverlayStrength1", 1.0)
        shader.inputs["ColorOverlayStrength1"].default_value = material_data.get_float_property(
            "g_flColorOverlayStrength2", 1.0)
        shader.inputs["ColorOverlayStrength2"].default_value = material_data.get_float_property(
            "g_flColorOverlayStrength3", 1.0)

        shader.inputs["NormalOverlayStrength0"].default_value = material_data.get_float_property(
            "g_flNormalOverlayStrength1", 1.0)
        shader.inputs["NormalOverlayStrength1"].default_value = material_data.get_float_property(
            "g_flNormalOverlayStrength2", 1.0)
        shader.inputs["NormalOverlayStrength2"].default_value = material_data.get_float_property(
            "g_flNormalOverlayStrength3", 1.0)

        shader.inputs["RoughnessOverlayStrength0"].default_value = material_data.get_float_property(
            "g_flRoughnessOverlayStrength1", 1.0)
        shader.inputs["RoughnessOverlayStrength1"].default_value = material_data.get_float_property(
            "g_flRoughnessOverlayStrength2", 1.0)
        shader.inputs["RoughnessOverlayStrength2"].default_value = material_data.get_float_property(
            "g_flRoughnessOverlayStrength3", 1.0)

        if self.tinted:
            vcolor_node = self.create_node(Nodes.ShaderNodeVertexColor)
            vcolor_node.layer_name = "TINT"
            self.connect_nodes(vcolor_node.outputs[0], shader.inputs["WorldTint"])

        shader.inputs["ModelTint0"].default_value = material_data.get_vector_property("g_vColorTint1", (1, 1, 1, 1))
        shader.inputs["ModelTintEnable0"].default_value = material_data.get_int_property("g_bModelTint1", 1)

        shader.inputs["ModelTint1"].default_value = material_data.get_vector_property("g_vColorTint2", (1, 1, 1, 1))
        shader.inputs["ModelTintEnable1"].default_value = material_data.get_int_property("g_bModelTint2", 1)

        shader.inputs["ModelTint2"].default_value = material_data.get_vector_property("g_vColorTint3", (1, 1, 1, 1))
        shader.inputs["ModelTintEnable2"].default_value = material_data.get_int_property("g_bModelTint3", 1)
