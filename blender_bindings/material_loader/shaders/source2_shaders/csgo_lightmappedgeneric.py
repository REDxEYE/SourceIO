from pprint import pformat

from SourceIO.blender_bindings.material_loader.shader_base import Nodes
from SourceIO.blender_bindings.material_loader.shaders.source2_shader_base import Source2ShaderBase
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4_3
from SourceIO.library.source2.blocks.kv3_block import KVBlock


class CSGOLightmappedGeneric(Source2ShaderBase):
    SHADER: str = 'csgo_lightmappedgeneric.vfx'

    def create_nodes(self, material):
        if super().create_nodes(material) in ['UNKNOWN', 'LOADED']:
            return
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node_group("csgo_lightmappedgeneric.vfx", name=self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])
        material_data = self._material_resource
        data = self._material_resource.get_block(KVBlock,block_name='DATA')
        self.logger.info(pformat(dict(data)))

        if self._have_texture("g_tColor"):
            color0_texture = self._get_texture("g_tColor", (1, 1, 1, 1))
            self.connect_nodes(color0_texture.outputs[0], shader.inputs["TextureColor0"])
            if (material_data.get_int_property("F_ALPHA_TEST", 0) or
                material_data.get_int_property("S_TRANSLUCENT", 0) or
                material_data.get_int_property("F_OVERLAY", 0)):
                self.connect_nodes(color0_texture.outputs[1], shader.inputs["TextureAlpha0"])

        if self._have_texture("g_tLayer2Color"):
            color_texture = self._get_texture("g_tLayer2Color", (1, 1, 1, 1))
            if (material_data.get_int_property("F_ALPHA_TEST", 0) or
                material_data.get_int_property("S_TRANSLUCENT", 0) or
                material_data.get_int_property("F_OVERLAY", 0)):
                self.connect_nodes(color_texture.outputs[1], shader.inputs["TextureAlpha1"])

            self.connect_nodes(color_texture.outputs[0], shader.inputs["TextureColor1"])

        if self._have_texture("g_tLayer1NormalRoughness"):
            normal0_texture = self._get_texture("g_tLayer1NormalRoughness", (0.5, 0.5, 1, 1))
            self.connect_nodes(normal0_texture.outputs[0], shader.inputs["TextureNormal0"])
            self.connect_nodes(normal0_texture.outputs[1], shader.inputs["TextureRoughness0"])

        if self._have_texture("g_tLayer2NormalRoughness"):
            normal_texture = self._get_texture("g_tLayer2NormalRoughness", (0.5, 0.5, 1, 1))
            self.connect_nodes(normal_texture.outputs[0], shader.inputs["TextureNormal1"])
            self.connect_nodes(normal_texture.outputs[1], shader.inputs["TextureRoughness1"])

        if self._have_texture("g_tBlendModulation"):
            color_texture = self._get_texture("g_tBlendModulation", (1, 1, 1, 1))

            self.connect_nodes(color_texture.outputs[0], shader.inputs["BlendModulate"])

        # TODO: tinting and details

        if self.tinted:
            vcolor_node = self.create_node(Nodes.ShaderNodeVertexColor)
            vcolor_node.layer_name = "TINT"
            self.connect_nodes(vcolor_node.outputs[0], shader.inputs["ModelTint"])

        shader.inputs["Softness"].default_value = material_data.get_float_property("g_flBlendSoftness", 0.5)
        shader.inputs["Sharpness"].default_value = material_data.get_float_property("g_flBevelBlendSharpness", 4)

        if material_data.get_int_property("F_ALPHA_TEST", 0):
            if not is_blender_4_3():
                self.bpy_material.blend_method = 'CLIP'
                self.bpy_material.shadow_method = 'CLIP'
                self.bpy_material.alpha_threshold = material_data.get_float_property("g_flAlphaTestReference", 0.5)
        elif material_data.get_int_property("S_TRANSLUCENT", 0):
            if not is_blender_4_3():
                self.bpy_material.blend_method = 'HASHED'
                self.bpy_material.shadow_method = 'CLIP'
        elif material_data.get_int_property("F_OVERLAY", 0):
            if not is_blender_4_3():
                self.bpy_material.blend_method = 'HASHED'
                self.bpy_material.shadow_method = 'CLIP'
