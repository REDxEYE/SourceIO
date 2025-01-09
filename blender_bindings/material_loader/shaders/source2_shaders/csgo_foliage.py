from pprint import pformat

from SourceIO.blender_bindings.material_loader.shader_base import Nodes
from SourceIO.blender_bindings.material_loader.shaders.source2_shader_base import Source2ShaderBase
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4_3
from SourceIO.library.source2.blocks.kv3_block import KVBlock


class CSGOFoliage(Source2ShaderBase):
    SHADER: str = 'csgo_foliage.vfx'

    def create_nodes(self, material):
        if super().create_nodes(material) in ['UNKNOWN', 'LOADED']:
            return
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])
        data_block = self._material_resource.get_block(KVBlock,block_name='DATA')
        self.logger.info(pformat(dict(data_block)))

        color_texture = self._get_texture("g_tColor", (1, 1, 1, 1))

        color_output = color_texture.outputs[0]
        alpha_output = color_texture.outputs[1]

        self.connect_nodes(color_output, shader.inputs["Base Color"])
        self.connect_nodes(color_output, shader.inputs["Alpha"])

        normal_texture = self._get_texture("g_tNormal", (1, 1, 1, 1), True, True)
        normal_conv = self.create_node(Nodes.ShaderNodeNormalMap)
        self.connect_nodes(normal_texture.outputs[0], normal_conv.inputs[1])
        self.connect_nodes(normal_conv.outputs[0], shader.inputs["Normal"])
        self.connect_nodes(normal_texture.outputs[1], shader.inputs["Roughness"])

        if self._material_resource.get_int_property("F_ALPHA_TEST", 0):
            if not is_blender_4_3():
                self.bpy_material.blend_method = 'CLIP'
                self.bpy_material.shadow_method = 'CLIP'
                self.bpy_material.alpha_threshold = self._material_resource.get_float_property("g_flAlphaTestReference",
                                                                                           0.5)
            self.connect_nodes(alpha_output, shader.inputs["Alpha"])

        if self._material_resource.get_int_property("S_TRANSLUCENT", 0):
            if not is_blender_4_3():
                self.bpy_material.blend_method = 'HASHED'
                self.bpy_material.shadow_method = 'CLIP'
            self.connect_nodes(color_texture.outputs[1], shader.inputs["Alpha"])
