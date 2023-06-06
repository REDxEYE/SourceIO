from pprint import pformat
from typing import Tuple

from ..source2_shader_base import Source2ShaderBase
from ...shader_base import Nodes


class CSGOLightmappedGeneric(Source2ShaderBase):
    SHADER: str = 'csgo_lightmappedgeneric.vfx'

    def _get_texture(self, slot_name: str, default_color: Tuple[float, float, float, float],
                     is_data=False,
                     invert_y: bool = False):
        texture_path = self._material_resource.get_texture_property(slot_name, None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, default_color, invert_y)
            if is_data:
                image.colorspace_settings.is_data = True
                image.colorspace_settings.name = 'Non-Color'
        else:
            image = self.get_missing_texture(slot_name, default_color)
        texture_node = self.create_node(Nodes.ShaderNodeTexImage, slot_name)
        texture_node.image = image
        return texture_node

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])
        data, = self._material_resource.get_data_block(block_name='DATA')
        self.logger.info(pformat(dict(data)))

        color_texture = self._get_texture("g_tColor", (1, 1, 1, 1))

        color_output = color_texture.outputs[0]

        self.connect_nodes(color_output, shader.inputs["Base Color"])

        normal_texture = self._get_texture("g_tLayer1NormalRoughness", (1, 1, 1, 1), True, True)
        normal_conv = self.create_node(Nodes.ShaderNodeNormalMap)
        self.connect_nodes(normal_texture.outputs[0], normal_conv.inputs[1])
        self.connect_nodes(normal_conv.outputs[0], shader.inputs["Normal"])
        self.connect_nodes(normal_texture.outputs[1], shader.inputs["Roughness"])

        # metalness_texture = self._get_texture("g_tMetalness", (1, 1, 1, 1), True)
        # metalness_conv = self.create_node(Nodes.ShaderNodeSeparateRGB)
        # self.connect_nodes(metalness_texture.outputs[0], metalness_conv.inputs[0])
        # roughness_override = self._material_resource.get_vector_property("TextureNormal", None)
        # if roughness_override is not None:
        #     shader.inputs["Roughness"].default_value = roughness_override[0]
        # else:
        #     self.connect_nodes(normal_conv.outputs[1], shader.inputs["Roughness"])

        # metallic_override = self._material_resource.get_vector_property("TextureMetalness", None)
        # if metallic_override is not None:
        #     shader.inputs["Metallic"].default_value = metallic_override[0]
        # else:
        #     self.connect_nodes(metalness_conv.outputs[1], shader.inputs["Metallic"])

        if self._material_resource.get_int_property("F_ALPHA_TEST", 0):
            self.bpy_material.blend_method = 'HASHED'
            self.connect_nodes(color_texture.outputs[1], shader.inputs["Alpha"])
        if self._material_resource.get_int_property("F_OVERLAY", 0):
            self.bpy_material.blend_method = 'BLEND'
            self.connect_nodes(color_texture.outputs[1], shader.inputs["Alpha"])
        